"""
End-to-end pipeline: ingest → geometry → norms → dataset → train.

Usage:
    python run_pipeline.py \
        --csv-dir data/raw_csvs/ \
        --mesh data/track_collision.obj \
        --fastest-replay data/raw_csvs/fastest.csv \
        --output-dir data/processed \
        --epochs 100 \
        --batch-size 32 \
        --window-len 32

    # Resume an interrupted or extended run:
    python run_pipeline.py ... --epochs 200 --resume data/processed/checkpoints/latest.pt

Stage caching:
  - Step 1 (ingestion) is skipped only if BOTH replays.h5 and
    segment_info.json already exist. Delete either to force re-ingestion
    (there is no cheap way to detect "did the CSVs change" short of
    hashing every file, so this stays existence-based).
  - Steps 2 (geometry) and 4 (norm stats) are parameter-aware: each
    writes a small `<output>.meta.json` sidecar recording the
    parameters used to build it. On the next run, if the *current*
    CLI parameters don't match the sidecar, the stage is automatically
    recomputed instead of silently reusing a stale artifact.

Checkpoints:
  - Every checkpoint is self-contained: weights, optimizer state, RNG
    state, and the full RunConfig used to produce it. `latest.pt` is
    overwritten every epoch and is what `--resume` typically targets.

Resume semantics (--resume):
  - --csv-dir/--mesh/--fastest-replay are ALWAYS required, resumed or
    not — the full command must always be restated. This is more
    verbose than inferring paths from the checkpoint, but avoids ever
    silently resolving a stale/relative path differently than intended.
  - DATA_FIELDS (paths, seed, val_fraction, ...) differing from the
    checkpoint is refused unless --resume-force is passed (it changes
    what the model's inputs *mean*, without being a crash).
  - ARCH_FIELDS (window_len, subsample_factor, hidden, track_out,
    num_materials) differing is ALWAYS fatal — this breaks weight
    loading and --resume-force cannot help.
  - MUTABLE_FIELDS (epochs, batch_size, lr, rollout_*, tf_start,
    checkpoint_every, num_workers, device) are inherited from the
    checkpoint for any flag you don't explicitly re-pass, and
    overridden for any flag you do — so a forgotten `--rollout-end`
    can't silently reset your curriculum, but `--resume ckpt --lr 1e-4`
    to manually drop the learning rate mid-run works as expected.

Experiment tracking:
  - Each invocation writes `output_dir/runs/run_<id>.json` (full config
    + git commit + resume provenance, including exactly which mutable
    fields were overridden vs inherited) and appends per-epoch metrics
    to `output_dir/metrics.jsonl`, tagged by run_id.
"""

from __future__ import annotations
import argparse
import math
from pathlib import Path

from tqdm import tqdm

import torch
import json

from config import (
    SamplingConfig, RunConfig, stage_is_stale, save_stage_meta,
    validate_args, get_explicit_cli_args, merge_mutable_fields_into_args,
    DATA_FIELDS, ARCH_FIELDS, EARLY_ARCH_FIELDS, LATE_ARCH_FIELDS, set_seed,
)
from features import FEATURE_SCHEMA_HASH
from checkpointing import save_checkpoint, load_checkpoint, restore_rng_state
from experiment_logging import JsonlExperimentLogger, make_run_id
from ingest_raw import ingest_csv_dir
from preprocess_geometry import preprocess_geometry
from compute_norm_stats import (
    split_replays,
    compute_and_save_norm_stats,
    norm_stats_cache_key,
)
from dataset import TMWorldModelDataset, create_dataloader, NormStats
from track_context import load_extractor
from model import WorldModel
from train import train_epoch, evaluate, compute_lr, profile_training


def _format_mismatches(mismatches: dict) -> str:
    return '\n'.join(
        f'  {f}: checkpoint={a!r}  current={b!r}' for f, (a, b) in mismatches.items()
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description='Trackmania World Model — End-to-end pipeline'
    )

    parser.add_argument('--csv-dir', type=Path, required=True)
    parser.add_argument('--mesh', type=Path, required=True)
    parser.add_argument('--fastest-replay', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('data/processed'))

    parser.add_argument('--min-segment-length', type=int, default=20)

    parser.add_argument('--sample-interval', type=float, default=1.0)
    parser.add_argument('--radius', type=float, default=15.0)
    parser.add_argument('--n-points', type=int, default=128)

    parser.add_argument('--val-fraction', type=float, default=0.1)
    parser.add_argument('--seed', type=int, default=42)

    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--weight-decay', type=int, default=1e-3) #? good default?
    parser.add_argument('--window-len', type=int, default=32)
    parser.add_argument('--subsample-factor', type=int, default=1)
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--track-out', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr-warmup-epochs', type=int, default=5,
                        help='Linear LR warmup duration in epochs, before cosine decay')
    parser.add_argument('--lr-min-ratio', type=float, default=0.1,
                        help='Final LR as a fraction of --lr (cosine decay floor)')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--rollout-start', type=int, default=4)
    parser.add_argument('--rollout-end', type=int, default=None)
    parser.add_argument('--tf-start', type=float, default=0.5)
    parser.add_argument('--checkpoint-every', type=int, default=10)
    parser.add_argument('--device', type=str, default='auto')

    parser.add_argument('--profile', action='store_true',
                        help='Run a short torch.profiler session on real '
                             'training batches (forward rollout + backward '
                             '+ optimizer step), print a time breakdown, '
                             'save a trace, then exit WITHOUT training or '
                             'checkpointing. Adds profiler overhead — it/s '
                             'measured under --profile is not representative '
                             'of real training speed, only the relative '
                             'breakdown between named blocks is.')
    parser.add_argument('--profile-wait', type=int, default=2,
                        help='Batches run untouched before profiling starts '
                             '(lets CUDA/cuDNN/torch.compile warm up).')
    parser.add_argument('--profile-warmup', type=int, default=3,
                        help='Batches run under the profiler but discarded '
                             'from the report (profiler instrumentation '
                             'itself has startup cost).')
    parser.add_argument('--profile-active', type=int, default=100,
                        help='Batches actually measured and reported.')
    parser.add_argument('--profile-dir', type=Path, default=None,
                        help='Where to write trace files. Defaults to '
                             '<output-dir>/checkpoints/profiler.')

    parser.add_argument('--resume', type=Path, default=None,
                        help='Path to a checkpoint (e.g. checkpoints/latest.pt) to '
                             'resume training from')
    parser.add_argument('--resume-force', action='store_true',
                        help='Proceed even if data-provenance fields differ from the '
                             'checkpoint. Has no effect on architecture mismatches, '
                             'which are always fatal.')

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.rollout_end is None:
        args.rollout_end = args.window_len

    issues = validate_args(args)
    if issues:
        parser.error('Invalid configuration:\n' + '\n'.join(f'  - {i}' for i in issues))

    # Captured now, from raw argv, independent of any later mutation of
    # `args` (e.g. the rollout_end default-fill above, or the resume
    # merge below) — this is what lets us tell "explicitly passed" from
    # "happens to equal the default".
    explicit = get_explicit_cli_args(parser)

    # ── Resume: load checkpoint + merge mutable fields ────────────────
    # This must happen BEFORE device resolution, dataloader construction,
    # and RunConfig.from_args — all of which read from `args`.
    resumed = None
    old_config = None
    mutable_overrides_applied: dict = {}
    mutable_inherited: dict = {}

    if args.resume is not None:
        print(f'\n═══ Loading checkpoint for resume: {args.resume} ═══')
        # map_location='cpu' here — device isn't resolved yet, and this
        # avoids requiring CUDA to even inspect a checkpoint. Tensors
        # move to the real device once the model itself is placed there.
        resumed = load_checkpoint(args.resume, map_location='cpu')
        old_config = RunConfig.from_dict(resumed['run_config'])
        print(f'  Checkpoint is from end of epoch {resumed["epoch"] + 1}, '
              f'val_mse={resumed["val_mse"]:.6f}')

        mutable_overrides_applied, mutable_inherited, runtime_diffs = merge_mutable_fields_into_args(
            args, old_config, explicit,
        )
        if mutable_overrides_applied:
            print(f'  Resume: CLI overrides applied     : {mutable_overrides_applied}')
        if mutable_inherited:
            print(f'  Resume: inherited from checkpoint : {mutable_inherited}')
        if runtime_diffs:
            print(f'  Resume: runtime/environment fields differ from checkpoint '
                  f'(NOT inherited — using current invocation\'s values): {runtime_diffs}')

        # Re-validate: merging fields from two independently-valid
        # sources (checkpoint + explicit CLI overrides) can still
        # produce an invalid combination (e.g. checkpoint's rollout_end
        # + a freshly-overridden, larger rollout_start).
        issues = validate_args(args)
        if issues:
            raise SystemExit(
                'Invalid configuration after merging with checkpoint values:\n'
                + '\n'.join(f'  - {i}' for i in issues)
            )

        # Early immutable-field check — fail fast, before touching any
        # data. num_materials isn't knowable yet (needs the geometry/
        # dataset stages below), so it's checked again later.
        provisional = RunConfig.from_args(
            args,
            num_materials=old_config.num_materials,
            material_registry_hash=old_config.material_registry_hash,
            feature_schema_hash=FEATURE_SCHEMA_HASH,   # real value — known immediately
        )
        early_mismatches = old_config.diff(provisional, DATA_FIELDS + EARLY_ARCH_FIELDS)
        arch_mismatches = {f: v for f, v in early_mismatches.items() if f in ARCH_FIELDS}
        data_mismatches = {f: v for f, v in early_mismatches.items() if f in DATA_FIELDS}

        if arch_mismatches:
            raise SystemExit(
                'Refusing to resume — architecture differs from the checkpoint '
                '(this would break weight loading; --resume-force cannot help):\n'
                + _format_mismatches(arch_mismatches)
            )
        if data_mismatches and not args.resume_force:
            raise SystemExit(
                'Refusing to resume — data provenance differs from the '
                'checkpoint:\n' + _format_mismatches(data_mismatches) +
                '\n(pass --resume-force to proceed anyway)'
            )
        if data_mismatches:
            print('⚠ --resume-force: proceeding despite data-provenance mismatch:')
            print(_format_mismatches(data_mismatches))

    # ── Device (resolved after any inherited/overridden --device) ────
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f'Device: {device}')

    # ── Reproducibility ────────────────────────────────────────────
    # Seeded AFTER the resume merge above (so a resumed run's `seed` —
    # a DATA_FIELDS entry, mismatch-checked against the checkpoint
    # above — is what actually gets used), and BEFORE anything that
    # draws from a global RNG: model weight init (Step 6) and
    # DataLoader shuffling order (Step 5). For a resumed run,
    # restore_rng_state() later in Step 6 overwrites this with the
    # checkpoint's exact saved RNG state anyway, continuing the
    # original random stream rather than restarting it — this call
    # only matters for what happens between here and that point, and
    # for fresh (non-resumed) runs.
    set_seed(args.seed)

    # Single source of truth for sampling parameters — passed to every
    # stage that needs window_len + subsample_factor.
    config = SamplingConfig(window_len=args.window_len, subsample_factor=args.subsample_factor)

    # Effective timestep between consecutive (possibly subsampled) frames.
    effective_dt = config.effective_dt
    print(f'Subsample factor: {config.subsample_factor}  '
          f'(effective dt={effective_dt * 1000:.1f} ms)')

    # Derived paths
    h5_path       = args.output_dir / 'replays.h5'
    seg_info_path = args.output_dir / 'segment_info.json'
    geo_path      = args.output_dir / 'track_geo.h5'
    norm_path     = args.output_dir / 'norm_stats.npz'
    ckpt_dir      = args.output_dir / 'checkpoints'

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1: Ingest CSVs → HDF5 ───────────────────────────────────
    if not h5_path.exists() or not seg_info_path.exists():
        print('\n═══ Step 1: Ingesting CSVs → HDF5 ═══')
        ingest_csv_dir(args.csv_dir, h5_path, args.min_segment_length)
    else:
        print(f'\n═══ Step 1: Using existing {h5_path} ═══')

    with open(seg_info_path) as f:
        seg_info = json.load(f)

    # ── Step 2: Track geometry ────────────────────────────────────────
    geo_meta_path = geo_path.with_suffix('.meta.json')
    geo_key = {
        'mesh': str(args.mesh.resolve()),
        'mesh_mtime': args.mesh.stat().st_mtime,
        'fastest_replay': str(args.fastest_replay.resolve()),
        'fastest_replay_mtime': args.fastest_replay.stat().st_mtime,
        'sample_interval': args.sample_interval,
        'radius': args.radius,
        'n_points': args.n_points,
    }
    geo_is_stale = stage_is_stale(geo_meta_path, geo_key)

    if not geo_path.exists() or geo_is_stale:
        reason = 'parameters changed' if geo_path.exists() else 'no cached geometry'
        print(f'\n═══ Step 2: Building track geometry ({reason}) ═══')
        preprocess_geometry(
            args.mesh, args.fastest_replay, geo_path,
            sample_interval=args.sample_interval, radius=args.radius, n_points=args.n_points,
        )
        save_stage_meta(geo_meta_path, geo_key)
    else:
        print(f'\n═══ Step 2: Using existing {geo_path} (parameters unchanged) ═══')

    # ── Step 3: Train/val split ───────────────────────────────────────
    print('\n═══ Step 3: Splitting replays ═══')
    all_ids = sorted(set(s['replay_id'] for s in seg_info))
    train_ids, val_ids = split_replays(all_ids, args.val_fraction, args.seed)
    print(f'  {len(train_ids)} train replays, {len(val_ids)} val replays')
    print(f'  Val IDs: {val_ids}')

    # ── Step 4: Normalization stats (train only) ─────────────────────
    norm_meta_path = norm_path.with_suffix('.meta.json')
    norm_key = norm_stats_cache_key(config, train_ids, h5_path, seg_info_path)

    if not norm_path.exists() or stage_is_stale(norm_meta_path, norm_key):
        reason = "parameters changed" if norm_path.exists() else "no cached stats"
        print(f'\n═══ Step 4: Computing normalization stats ({reason}) ═══')
        compute_and_save_norm_stats(
            h5_path, norm_path, seg_info, config, seg_info_path, train_ids,
        )
    else:
        print(f'\n═══ Step 4: Using existing {norm_path} (parameters unchanged) ═══')

    # ── Step 5: Build datasets ────────────────────────────────────────
    print('\n═══ Step 5: Building datasets ═══')
    track_ctx = load_extractor(str(geo_path)).to(device)
    num_materials = track_ctx.num_materials

    train_ds = TMWorldModelDataset(
        h5_path=str(h5_path),
        seg_info_path=str(seg_info_path),
        norm_stats_path=str(norm_path),
        config=config,
        replay_ids=train_ids,
    )
    val_ds = TMWorldModelDataset(
        h5_path=str(h5_path),
        seg_info_path=str(seg_info_path),
        norm_stats_path=str(norm_path),
        config=config,
        replay_ids=val_ids,
    )

    assert len(train_ds) > 0, "Training dataset is empty — check val_fraction and replay IDs"
    assert len(val_ds) > 0, (
        "Validation dataset is empty — every val replay's segments are too "
        "short to produce a single window at this window_len/subsample_factor "
        "(or val_fraction produced zero val replays). evaluate() would "
        "otherwise silently report val_mse=0.0/pos_error=0.0 from zero real "
        "samples, which looks like a suspiciously perfect model rather than "
        "a misconfiguration. Increase --val-fraction, add more replay data, "
        "or reduce --window-len."
    )

    print(f'  Train: {len(train_ds)} windows')
    print(f'  Val:   {len(val_ds)} windows')

    train_loader = create_dataloader(
        train_ds, args.batch_size, args.num_workers, shuffle=True, seed=args.seed,
    )
    val_loader = create_dataloader(
        val_ds, args.batch_size, args.num_workers, shuffle=False, seed=args.seed,
    )

    # ── Step 5.5: RunConfig + resume compatibility (final check) ────────
    run_config = RunConfig.from_args(
        args,
        num_materials=num_materials,
        material_registry_hash=track_ctx.material_registry_hash,
        feature_schema_hash=FEATURE_SCHEMA_HASH,
    )

    if resumed is not None:
        late_mismatch = old_config.diff(run_config, LATE_ARCH_FIELDS)
        if late_mismatch:
            raise SystemExit(
                'Refusing to resume — geometry/material provenance differs from '
                'the checkpoint (this would silently reinterpret compact material '
                'IDs or embedding table size; --resume-force cannot help):\n'
                + _format_mismatches(late_mismatch)
            )

    # ── Step 6: Model + optimizer ─────────────────────────────────────
    print('\n═══ Step 6: Building model ═══')

    model = WorldModel.from_config(run_config).to(device)
    print(f'  Parameters: {model.num_parameters:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=run_config.lr, weight_decay=run_config.weight_decay) 
    norm_torch = NormStats.load(str(norm_path)).to_torch(device)

    resumed_from_record = None
    if resumed is not None:
        model.load_state_dict(resumed['model_state_dict'])
        try:
            optimizer.load_state_dict(resumed['optimizer_state_dict'])
            # No need to manually reapply run_config.lr here anymore —
            # the per-epoch compute_lr() call at the top of the loop
            # sets optimizer.param_groups[*]['lr'] unconditionally every
            # epoch, including the first resumed one, which supersedes
            # whatever LR was baked into the saved optimizer state.
        except (ValueError, RuntimeError) as e:
            print(f'  WARNING: could not restore optimizer state ({e}); '
                  f'reinitialized fresh. Momentum/Adam moment estimates '
                  f'from the original run are lost — safe, but may cause '
                  f'a transient blip in loss for a few steps.')
        restore_rng_state(resumed['rng_state'])
        start_epoch = resumed['epoch'] + 1
        best_val_mse = resumed['best_val_mse']
        resumed_from_record = {
            'path': str(args.resume),
            'epoch': resumed['epoch'],
            'mutable_overrides_applied': mutable_overrides_applied,
            'mutable_inherited_from_checkpoint': mutable_inherited,
            'runtime_fields_not_inherited': runtime_diffs,
        }
        print(f'  Resumed at epoch {start_epoch}, best_val_mse so far={best_val_mse:.6f}')
    else:
        start_epoch = 0
        best_val_mse = float('inf')

    # ── Compile once — shared by --profile and the real training loop ──
    model = torch.compile(model, mode="reduce-overhead")
    print(f'  Compiled model device: {next(model.parameters()).device}')

    # ── Optional: profiling mode ──────────────────────────────────────
    # Runs a handful of REAL training steps (forward rollout, backward,
    # optimizer.step) under torch.profiler, reports a breakdown, then
    # exits. No checkpointing, no logger run, no epoch loop.
    if args.profile:
        print('\n═══ Profiling mode (no training/checkpointing will occur) ═══')
        profile_dir = args.profile_dir or (ckpt_dir / 'profiler')

        # Mirror the curriculum a real train_epoch() call would use at
        # `start_epoch` (0 for a fresh run, or wherever a resumed run
        # left off) so the profiled rollout length / teacher-forcing
        # rate is representative of that point in training, rather than
        # always profiling the easiest (shortest-rollout) regime.
        progress = start_epoch / max(run_config.epochs - 1, 1)
        profile_k = int(
            run_config.rollout_start
            + (progress ** 0.5) * (run_config.rollout_end - run_config.rollout_start)
        )
        profile_k = min(profile_k, config.window_len)
        profile_tf = run_config.tf_start * (1.0 - progress)

        profile_training(
            model, train_loader, optimizer,
            track_ctx, norm_torch, device,
            min_rollout_steps=run_config.rollout_start,
            max_rollout_steps=profile_k,
            teacher_forcing_prob=profile_tf,
            dt=effective_dt,
            output_dir=profile_dir,
            n_wait=args.profile_wait,
            n_warmup=args.profile_warmup,
            n_active=args.profile_active,
        )
        train_ds.close()
        val_ds.close()
        return

    # ── Experiment tracking ────────────────────────────────────────────
    logger = JsonlExperimentLogger(args.output_dir, make_run_id())
    logger.log_run_start(run_config, resumed_from=resumed_from_record)

    if start_epoch >= run_config.epochs:
        print(f'\nCheckpoint is already at epoch {start_epoch} >= --epochs {run_config.epochs}; '
              f'nothing to do. Pass a larger --epochs to continue training.')
        train_ds.close()
        val_ds.close()
        return

    # Loader sanity
    assert len(train_loader) > 0, (
        f'Training dataloader yields zero batches — batch_size '
        f'({run_config.batch_size}) is likely larger than the training set '
        f'({len(train_ds)} windows, drop_last=True). Reduce --batch-size '
        f'or --val-fraction.'
    )

    # ── Step 7: Training ──────────────────────────────────────────────
    print('\n═══ Step 7: Training ═══')
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    #? test
    # one of these 2 lines below may have caused:
    # RuntimeError: Error: accessing tensor output of CUDAGraphs that has been overwritten by a subsequent run. 
    # Stack trace: File "model.py", line 165, in forward
    # new_hidden = self.gru_cell(combined, hidden)     # (B, H). To prevent overwriting, clone the tensor outside of torch.compile() or call torch.compiler.cudagraph_mark_step_begin() before each model invocation.
    # torch.set_float32_matmul_precision('high')

    epoch_pbar = tqdm(range(start_epoch, run_config.epochs), desc="Epochs")
    for epoch in epoch_pbar:
        # Curriculum computed over the FULL [0, run_config.epochs) range
        # — resuming with the same --epochs continues the same
        # curriculum rather than restarting its schedule from epoch 0.
        linear_progress = epoch / max(run_config.epochs - 1, 1)
        sqrt_progress = (epoch / max(run_config.epochs - 1, 1)) ** 0.5
        current_k = int(
            run_config.rollout_start
            + sqrt_progress * (run_config.rollout_end - run_config.rollout_start)
        )
        current_k = min(current_k, config.window_len)
        current_tf = run_config.tf_start * (1.0 - linear_progress)
        current_lr = compute_lr(
            epoch,
            run_config,
            warmup_epochs=run_config.lr_warmup_epochs,
            min_lr_ratio=run_config.lr_min_ratio,
        )
        for pg in optimizer.param_groups:
            pg["lr"] = current_lr

        train_loss = train_epoch(
            model, train_loader, optimizer,
            track_ctx, norm_torch, device,
            min_rollout_steps=run_config.rollout_start,
            max_rollout_steps=current_k,
            teacher_forcing_prob=current_tf,
            dt=effective_dt,
        )

        # Validation ALWAYS uses the final target rollout length
        # (args.rollout_end), independent of the training curriculum.
        # If this varied with `current_k` (as it did previously), early
        # epochs would trivially get a lower val_mse just because they're
        # evaluated on a much shorter, easier rollout — making
        # "best_val_mse" incomparable across epochs and biasing model
        # selection toward the earliest (least-trained) checkpoints.
        val_metrics = evaluate(
            model, val_loader,
            track_ctx, norm_torch, device,
            rollout_steps=run_config.rollout_end,
            dt=effective_dt,
        )

        if not math.isfinite(val_metrics['mse']):
            raise FloatingPointError(
                f'val_mse is non-finite ({val_metrics["mse"]!r}) at epoch '
                f'{epoch+1} — the model diverged during this epoch\'s '
                f'training (train_epoch\'s own per-batch guard only skips '
                f'individual bad batches, it can\'t catch a model that has '
                f'become globally unstable). Aborting rather than '
                f'continuing to train/checkpoint a broken model. Consider '
                f'lowering --lr / adding warmup, or inspect the last good '
                f'checkpoint in {ckpt_dir}.'
            )

        is_best = val_metrics['mse'] < best_val_mse

        # Logging
        print(
            f'Epoch {epoch+1:3d}/{run_config.epochs}  '
            f'lr={current_lr:.2e}  '
            f'K_train={current_k:2d}  K_eval={run_config.rollout_end:2d}  TF={current_tf:.2f}  '
            f'train_loss={train_loss:.6f}  '
            f'val_mse={val_metrics["mse"]:.6f}  '
            f'val_pos_err={val_metrics["pos_error_m"]:.3f}m',
            end='',
        )

        # Best model tracking
        if is_best:
            best_val_mse = val_metrics['mse']
            print('  ★', end='')
        print()

        logger.log_epoch(epoch, {
            'k_train': current_k,
            'k_eval': run_config.rollout_end,
            'lr': current_lr,
            'tf': current_tf,
            'train_loss': train_loss,
            'val_mse': val_metrics['mse'],
            'val_pos_error_m': val_metrics['pos_error_m'],
            'is_best': is_best,
        })

        save_checkpoint(
            ckpt_dir / 'latest.pt',
            model=model, optimizer=optimizer, epoch=epoch,
            run_config=run_config, best_val_mse=best_val_mse,
            train_loss=train_loss, val_metrics=val_metrics,
        )

        if is_best:
            save_checkpoint(
                ckpt_dir / 'best_model.pt',
                model=model, optimizer=optimizer, epoch=epoch,
                run_config=run_config, best_val_mse=best_val_mse,
                train_loss=train_loss, val_metrics=val_metrics,
            )

        if (epoch + 1) % run_config.checkpoint_every == 0:
            save_checkpoint(
                ckpt_dir / f'epoch_{epoch+1:04d}.pt',
                model=model, optimizer=optimizer, epoch=epoch,
                run_config=run_config, best_val_mse=best_val_mse,
                train_loss=train_loss, val_metrics=val_metrics,
            )

    train_ds.close()
    val_ds.close()
    print(f'\nTraining complete. Best val MSE: {best_val_mse:.6f}')
    print(f'Checkpoints saved to {ckpt_dir}')


if __name__ == '__main__':
    main()
