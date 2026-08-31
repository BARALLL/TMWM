"""
Training loop for the Trackmania world model.

Features:
  - Multi-step rollout with BPTT through the model's own predictions
  - Teacher forcing with linear annealing (applied independently per
    sample in the batch, not per batch)
  - Per-step track context queries at the (predicted) car pose
  - Proper pose integration (position + orientation)
  - Explicit damper_rate recomputation during rollout
  - Gradient clipping
"""

from __future__ import annotations
import math
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn.functional as F

from torch.profiler import (
    profile, record_function, ProfilerActivity, schedule,
    tensorboard_trace_handler,
)


from track_context import TrackContextExtractor
from model import WorldModel
from config import RunConfig
from dynamics import step_dynamics, NormTensors

def compute_lr(
    epoch: int,
    run_config: RunConfig,
    warmup_epochs: int = 5,
    min_lr_ratio: float = 0.1,
) -> float:
    """
    Linear warmup for `warmup_epochs`, then cosine decay to
    `run_config.lr * min_lr_ratio` by the final epoch.

    Deliberately a pure function of (epoch, run_config) — mirroring how
    run_pipeline.py's rollout-length/teacher-forcing curriculum is
    computed fresh each epoch from `progress`, rather than stored as
    incrementally-mutated state. This means it needs no extra
    checkpoint field and resumes correctly automatically: recomputing
    compute_lr(epoch, run_config) after --resume reproduces exactly the
    LR that would have applied had the run never been interrupted
    (assuming --epochs is unchanged — extending --epochs shifts the
    decay schedule's endpoint, exactly like it already does for
    current_k/current_tf).

    If --lr is overridden on resume (`--resume ckpt --lr 1e-4`),
    run_config.lr reflects the new value, and it's used as the new
    warmup/decay peak from that point forward — the same fractional
    schedule shape applies, just anchored to the new peak.
    """
    warmup_epochs = min(warmup_epochs, max(run_config.epochs - 1, 0))
    if warmup_epochs > 0 and epoch < warmup_epochs:
        return run_config.lr * (epoch + 1) / warmup_epochs

    decay_span = max(run_config.epochs - 1 - warmup_epochs, 1)
    decay_progress = min(max((epoch - warmup_epochs) / decay_span, 0.0), 1.0)
    min_lr = run_config.lr * min_lr_ratio
    cosine = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
    return min_lr + (run_config.lr - min_lr) * cosine

# ──────────────────────────────────────────────────────────────────────
# Rollout
# ──────────────────────────────────────────────────────────────────────

def rollout(
    model: WorldModel,
    track_ctx: TrackContextExtractor,
    state_gt: torch.Tensor,
    action_gt: torch.Tensor,
    target_gt: torch.Tensor,
    pos_gt: torch.Tensor,
    quat_gt: torch.Tensor,
    norm_torch: dict[str, torch.Tensor],
    device: torch.device,
    rollout_steps: int | torch.Tensor,
    dt: float,
    teacher_forcing_prob: float = 0.0,
    return_pos_error: bool = False,
) -> tuple[torch.Tensor, dict | None]:
    B, W, _ = state_gt.shape
    D = target_gt.shape[-1]

    if isinstance(rollout_steps, torch.Tensor):
        K = int(rollout_steps.max().item())
        needs_mask = (int(rollout_steps.min().item()) != K)
        per_sample_steps = rollout_steps if needs_mask else None
    else:
        K = min(int(rollout_steps), W)
        needs_mask = False
        per_sample_steps = None
    K = min(K, W)

    norm = NormTensors.from_norm_torch(norm_torch)

    raw_state = state_gt[:, 0] * norm.state_std + norm.state_mean
    pos = pos_gt[:, 0].clone()
    quat = quat_gt[:, 0].clone()
    hidden = model.init_hidden(B, device)

    total_loss        = torch.tensor(0.0, device=device)
    total_weight_t     = torch.tensor(0.0, device=device)
    total_pos_err_t    = torch.tensor(0.0, device=device)
    total_pos_weight_t = torch.tensor(0.0, device=device)

    for t in range(K):
        if needs_mask:
            active   = t < per_sample_steps
            active_f = active.float()
        else:
            active_f = None

        if teacher_forcing_prob > 0:
            tf_mask = torch.rand(B, device=device) < teacher_forcing_prob
            if needs_mask:
                tf_mask = tf_mask & active
            mask_s = tf_mask.unsqueeze(-1)
            gt_state_t = state_gt[:, t] * norm.state_std + norm.state_mean
            raw_state = torch.where(mask_s, gt_state_t, raw_state)
            pos = torch.where(mask_s, pos_gt[:, t], pos)
            quat = torch.where(mask_s, quat_gt[:, t], quat)

        with record_function("dynamics_step"):
            result = step_dynamics(
                model, track_ctx, raw_state, pos, quat, hidden,
                action_gt[:, t], norm, dt,
            )

        with record_function("loss_compute"):
            pred = result.pred_norm
            if needs_mask:
                sq_err = (pred - target_gt[:, t]) ** 2
                total_loss = total_loss + (sq_err * active_f.unsqueeze(-1)).sum()
                total_weight_t = total_weight_t + active_f.sum()
            else:
                total_loss = total_loss + F.mse_loss(pred, target_gt[:, t]) * (B * D)
                total_weight_t = total_weight_t + B

        raw_state, pos, quat, hidden = result.raw_state, result.pos, result.quat, result.hidden

        if return_pos_error:
            with record_function("pos_error_metric"), torch.no_grad():
                err = torch.norm(pos - pos_gt[:, t + 1], dim=-1)
                if needs_mask:
                    total_pos_err_t += (err * active_f).sum()
                    total_pos_weight_t += active_f.sum()
                else:
                    total_pos_err_t += err.sum()
                    total_pos_weight_t += B

    total_weight = total_weight_t.item()
    if total_weight == 0:
        metrics = {'pos_error_m': 0.0} if return_pos_error else None
        return torch.tensor(0.0, device=device), metrics

    avg_loss = total_loss / (total_weight * D)
    metrics = None
    if return_pos_error:
        metrics = {'pos_error_m': (total_pos_err_t / total_pos_weight_t).item()}
    return avg_loss, metrics


# ──────────────────────────────────────────────────────────────────────
# Training epoch
# ──────────────────────────────────────────────────────────────────────

def train_epoch(
    model: WorldModel,
    dataloader,
    optimizer: torch.optim.Optimizer,
    track_ctx: TrackContextExtractor,
    norm_torch: dict[str, torch.Tensor],
    device: torch.device,
    min_rollout_steps: int,
    max_rollout_steps: int,
    teacher_forcing_prob: float,
    dt: float,
    grad_clip: float = 1.0,
    max_nonfinite_fraction: float = 0.5,
    min_batches_for_nonfinite_check: int = 10,
) -> float:
    """
    One training epoch. Returns average loss over FINITE-loss batches
    only (see below) — not a plain mean over all batches attempted.

    NaN/Inf handling: an autoregressive rollout can occasionally diverge
    (quaternion blow-up, exploding hidden state deep in a long BPTT
    unroll) and produce a non-finite loss or gradient.
    clip_grad_norm_ does NOT fix this — it just returns a NaN/Inf norm,
    and an optimizer.step() on NaN gradients silently corrupts every
    parameter permanently (e.g. Adam's moment estimates absorb the NaN
    and every subsequent step stays poisoned, with no crash or error
    anywhere). So, per batch:
      - a non-finite LOSS never reaches .backward() at all
      - a finite loss but non-finite GRADIENT NORM has its gradients
        discarded (optimizer.zero_grad()) instead of stepped
      - either way, the batch is skipped: excluded from the returned
        average and the optimizer/model are left untouched by it
    If too large a fraction of an epoch's batches are non-finite
    (default: > 50%, and only once there are enough batches to judge —
    see min_batches_for_nonfinite_check), this is treated as genuine
    divergence rather than a one-off fluke: continuing to "successfully"
    skip most of the epoch would silently produce a stalled/garbage
    model while still printing a plausible-looking train_loss. We raise
    instead of continuing silently.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0
    n_skipped = 0
    n_total = 0

    batch_pbar = tqdm(dataloader, leave=False)

    for batch in batch_pbar:
        n_total += 1

        current_batch_size = batch["state"].shape[0]
        current_rollout_steps = torch.randint(
            min_rollout_steps, max_rollout_steps + 1, (current_batch_size,)
        ).to(device)

        state_gt  = batch['state'].to(device)
        action_gt = batch['action'].to(device)
        target_gt = batch['target'].to(device)
        pos_gt    = batch['pos'].to(device)
        quat_gt   = batch['quat'].to(device)

        loss, _ = rollout(
            model, track_ctx,
            state_gt, action_gt, target_gt,
            pos_gt, quat_gt,
            norm_torch, device,
            rollout_steps=current_rollout_steps,
            dt=dt,
            teacher_forcing_prob=teacher_forcing_prob,
        )

        if not torch.isfinite(loss):
            print(f'  WARNING: non-finite loss ({loss.item()!r}) at batch '
                  f'{n_total} this epoch — skipping (no backward/step).')
            n_skipped += 1
            continue

        optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        if not torch.isfinite(grad_norm):
            print(f'  WARNING: non-finite grad norm ({grad_norm.item()!r}) at '
                  f'batch {n_total} this epoch — discarding gradients, no '
                  f'optimizer step.')
            optimizer.zero_grad()
            n_skipped += 1
            continue

        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if (n_total >= min_batches_for_nonfinite_check
            and n_skipped / n_total > max_nonfinite_fraction):
        raise FloatingPointError(
            f'{n_skipped}/{n_total} batches this epoch had non-finite loss '
            f'or gradients (> {max_nonfinite_fraction:.0%} threshold) — '
            f'training has likely diverged, not just hit a one-off bad '
            f'batch. Common causes: learning rate too high for the current '
            f'rollout length / teacher-forcing schedule. Aborting rather '
            f'than continuing to silently skip most of the epoch.'
        )
    elif n_skipped > 0:
        print(f'  ({n_skipped}/{n_total} batches skipped this epoch due to '
              f'non-finite loss/gradients)')

    return total_loss / max(n_batches, 1)


def profile_training(
    model: WorldModel,
    dataloader,
    optimizer: torch.optim.Optimizer,
    track_ctx: TrackContextExtractor,
    norm_torch: dict[str, torch.Tensor],
    device: torch.device,
    min_rollout_steps: int,
    max_rollout_steps: int,
    teacher_forcing_prob: float,
    dt: float,
    output_dir,
    n_wait: int = 2,
    n_warmup: int = 3,
    n_active: int = 5,
    grad_clip: float = 1.0,
) -> None:
    """
    Run a short profiled training session and report where wall-clock
    time actually goes, per training step.

    This runs REAL training steps (forward rollout + backward +
    optimizer.step) on a handful of batches from `dataloader` — not a
    synthetic benchmark — so the breakdown reflects your actual model,
    data, and rollout-length settings.

    wait/warmup/active (see torch.profiler.schedule):
      - `n_wait` batches run untouched by the profiler at all.
      - `n_warmup` batches run WITH the profiler attached but are
        discarded from the report — the profiler's own instrumentation
        has startup cost, and (if torch.compile is active) this is
        also where graph compilation happens; including that in the
        measured average would be wildly misleading.
      - `n_active` batches are the ones actually measured and reported.

    NOTE ON OVERHEAD: the profiler adds real per-op tracing cost. The
    it/s you see while --profile is running will be noticeably SLOWER
    than an unprofiled `train_epoch` — that's expected and not a
    regression. What's meaningful here is the *relative* time split
    between named blocks (track_ctx_query vs model_forward vs backward
    etc.), not the absolute it/s.

    NOTE ON torch.compile: a compiled model's forward pass may show up
    as a small number of opaque fused kernels rather than individual
    nn.Linear/ReLU ops — that's expected and is itself useful signal
    (fewer, larger kernels = less launch overhead). If you want a
    finer-grained per-op breakdown of what's inside the model, run
    this once against an uncompiled model too.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.train()

    activities = [ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(ProfilerActivity.CUDA)

    total_steps = n_wait + n_warmup + n_active
    data_iter = iter(dataloader)

    prof_schedule = schedule(wait=n_wait, warmup=n_warmup, active=n_active, repeat=1)

    print(f'Profiling {n_active} active batches ({n_wait} wait + {n_warmup} '
          f'warmup discarded first)  —  device={device}, '
          f'rollout=[{min_rollout_steps},{max_rollout_steps}], '
          f'tf={teacher_forcing_prob}')

    with profile(
        activities=activities,
        schedule=prof_schedule,
        on_trace_ready=tensorboard_trace_handler(str(output_dir)),
        record_shapes=True,
        with_stack=False,   # with_stack=True is much heavier; enable manually if needed
    ) as prof:
        for _ in range(total_steps):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
                batch = next(data_iter)

            with record_function("h2d_transfer"):
                current_batch_size = batch['state'].shape[0]
                current_rollout_steps = torch.randint(
                    min_rollout_steps, max_rollout_steps + 1, (current_batch_size,)
                ).to(device)
                state_gt  = batch['state'].to(device)
                action_gt = batch['action'].to(device)
                target_gt = batch['target'].to(device)
                pos_gt    = batch['pos'].to(device)
                quat_gt   = batch['quat'].to(device)

            with record_function("forward_rollout"):
                loss, _ = rollout(
                    model, track_ctx,
                    state_gt, action_gt, target_gt,
                    pos_gt, quat_gt,
                    norm_torch, device,
                    rollout_steps=current_rollout_steps,
                    dt=dt,
                    teacher_forcing_prob=teacher_forcing_prob,
                )

            if torch.isfinite(loss):
                with record_function("backward"):
                    optimizer.zero_grad()
                    loss.backward()
                with record_function("optimizer_step"):
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if torch.isfinite(grad_norm):
                        optimizer.step()
                    else:
                        optimizer.zero_grad()

            if device.type == 'cuda':
                torch.cuda.synchronize()  # so profiler step boundaries are accurate

            prof.step()

    sort_key = "self_cuda_time_total" if device.type == 'cuda' else "self_cpu_time_total"
    print("\n" + "=" * 100)
    print(f"PROFILING SUMMARY — top ops by {sort_key} "
          f"(over {n_active} measured batches)")
    print("=" * 100)
    print(prof.key_averages().table(sort_by=sort_key, row_limit=30))

    with open(output_dir / "profile_table.txt", "w") as f:
        f.write(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=50))


    trace_path = output_dir / "trace.json"
    prof.export_chrome_trace(str(trace_path))
    print(f"\nChrome trace: {trace_path}")
    print(f"TensorBoard:  tensorboard --logdir {output_dir}  "
          f"(then open the 'PyTorch Profiler' tab)")


# ──────────────────────────────────────────────────────────────────────
# Validation
# ──────────────────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(
    model: WorldModel,
    dataloader,
    track_ctx: TrackContextExtractor,
    norm_torch: dict[str, torch.Tensor],
    device: torch.device,
    rollout_steps: int,
    dt: float,
) -> dict[str, float]:
    """
    Evaluate on validation set.

    Returns dict with:
      - 'mse': average per-step prediction loss
      - 'pos_error_m': average position error in metres
    """
    model.eval()
    total_loss = 0.0
    total_pos_err = 0.0
    total_samples = 0

    for batch in dataloader:
        state_gt  = batch['state'].to(device)
        action_gt = batch['action'].to(device)
        target_gt = batch['target'].to(device)
        pos_gt    = batch['pos'].to(device)
        quat_gt   = batch['quat'].to(device)
        B = state_gt.shape[0]

        loss, metrics = rollout(
            model, track_ctx,
            state_gt, action_gt, target_gt,
            pos_gt, quat_gt,
            norm_torch, device,
            rollout_steps=rollout_steps,
            dt=dt,
            teacher_forcing_prob=0.0,  # no TF during eval
            return_pos_error=True,
        )

        total_loss     += loss.item() * B
        total_pos_err  += metrics['pos_error_m'] * B
        total_samples  += B

    return {
        'mse':         total_loss    / max(total_samples, 1),
        'pos_error_m': total_pos_err / max(total_samples, 1),
    }
