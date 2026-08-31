"""
Shared single-step dynamics primitive.

This is the ONE place that calls model.forward() + integrate_state() +
integrate_pose() together — used identically by:
  - train.py's rollout() (BPTT training)
  - inference/belief.py's BeliefTracker (real-telemetry re-anchoring)
  - inference/planner.py's CEMPlanner (imagined candidate rollouts)

Do not reimplement this loop body anywhere else. Any divergence between
training-time and inference-time dynamics stepping is exactly the class
of bug this file exists to prevent.

integrate_state/integrate_pose (moved here from train.py, unchanged)
also live here since they're physics-integration primitives shared by
the same set of callers, not training-loop concerns.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from model import WorldModel
from track_context import TrackContextExtractor
from features import (
    TARGET_FEATURES, TARGET_FEATURE_MAP, STATE_FEATURE_MAP,
    DeltaAdd, DirectAssign, PosePosUpdate, PoseOrientUpdate, SuperviseOnly,
    RecomputeCausalGradient,
)
from torch_quaternion_utils import (
    torch_quat_normalize, torch_quat_multiply, torch_quat_rotate, torch_quat_from_rvec,
)
from inference.reward import LiveCenterlineState # not very clean


# ──────────────────────────────────────────────────────────────────────
# integrate_state — moved verbatim from train.py
# ──────────────────────────────────────────────────────────────────────

def _feature_slice_to_indices(s: slice | int) -> list[int]:
    if isinstance(s, slice):
        return list(range(s.start, s.stop, s.step if s.step is not None else 1))
    return [s]


class _IntegrateStateIndices:
    __slots__ = (
        'delta_add_state_idx', 'delta_add_target_idx',
        'direct_state_idx', 'direct_target_idx',
        'direct_clamp_min', 'direct_clamp_max', 'direct_has_clamp',
        'recompute_state_idx', 'recompute_source_idx',
    )


def _build_integrate_state_indices(device: torch.device) -> _IntegrateStateIndices:
    delta_add_state, delta_add_target = [], []
    direct_state, direct_target = [], []
    direct_clamp_min, direct_clamp_max = [], []
    seen_state_slots: set[int] = set()

    for tf in TARGET_FEATURES:
        match tf.integrate:
            case DeltaAdd(state_target=st):
                s_state = STATE_FEATURE_MAP[st].s
                state_idxs = _feature_slice_to_indices(s_state)
                target_idxs = _feature_slice_to_indices(tf.s)
                assert len(state_idxs) == len(target_idxs), (
                    f"DeltaAdd width mismatch: state_target={st!r} has "
                    f"{len(state_idxs)} columns but target feature {tf!r} "
                    f"has {len(target_idxs)}"
                )
                for si in state_idxs:
                    assert si not in seen_state_slots, (
                        f"state slot {si} written by more than one target feature"
                    )
                    seen_state_slots.add(si)
                delta_add_state.extend(state_idxs)
                delta_add_target.extend(target_idxs)

            case DirectAssign(state_target=st, clamp=clamp):
                s_state = STATE_FEATURE_MAP[st].s
                state_idxs = _feature_slice_to_indices(s_state)
                target_idxs = _feature_slice_to_indices(tf.s)
                assert len(state_idxs) == len(target_idxs), (
                    f"DirectAssign width mismatch: state_target={st!r} has "
                    f"{len(state_idxs)} columns but target feature {tf!r} "
                    f"has {len(target_idxs)}"
                )
                for si in state_idxs:
                    assert si not in seen_state_slots, (
                        f"state slot {si} written by more than one target feature"
                    )
                    seen_state_slots.add(si)
                direct_state.extend(state_idxs)
                direct_target.extend(target_idxs)
                lo, hi = clamp if clamp is not None else (-float('inf'), float('inf'))
                direct_clamp_min.extend([lo] * len(state_idxs))
                direct_clamp_max.extend([hi] * len(state_idxs))

            case PosePosUpdate() | PoseOrientUpdate() | SuperviseOnly():
                pass

    recompute_state, recompute_source = [], []
    for sf in STATE_FEATURE_MAP.values():
        if sf.recompute is not None:
            match sf.recompute:
                case RecomputeCausalGradient(source=src):
                    state_idxs = _feature_slice_to_indices(sf.s)
                    source_idxs = _feature_slice_to_indices(STATE_FEATURE_MAP[src].s)
                    assert len(state_idxs) == len(source_idxs)
                    recompute_state.extend(state_idxs)
                    recompute_source.extend(source_idxs)

    idx = _IntegrateStateIndices()
    idx.delta_add_state_idx  = torch.as_tensor(delta_add_state, dtype=torch.long, device=device)
    idx.delta_add_target_idx = torch.as_tensor(delta_add_target, dtype=torch.long, device=device)
    idx.direct_state_idx     = torch.as_tensor(direct_state, dtype=torch.long, device=device)
    idx.direct_target_idx    = torch.as_tensor(direct_target, dtype=torch.long, device=device)
    idx.direct_clamp_min     = torch.as_tensor(direct_clamp_min, dtype=torch.float32, device=device)
    idx.direct_clamp_max     = torch.as_tensor(direct_clamp_max, dtype=torch.float32, device=device)
    idx.direct_has_clamp     = bool(len(direct_state))
    idx.recompute_state_idx  = torch.as_tensor(recompute_state, dtype=torch.long, device=device)
    idx.recompute_source_idx = torch.as_tensor(recompute_source, dtype=torch.long, device=device)
    return idx


_INTEGRATE_STATE_INDEX_CACHE: dict[torch.device, _IntegrateStateIndices] = {}


def _get_integrate_state_indices(device: torch.device) -> _IntegrateStateIndices:
    idx = _INTEGRATE_STATE_INDEX_CACHE.get(device)
    if idx is None:
        idx = _build_integrate_state_indices(device)
        _INTEGRATE_STATE_INDEX_CACHE[device] = idx
    return idx


def integrate_state(raw_state: torch.Tensor, raw_delta: torch.Tensor, dt: float) -> torch.Tensor:
    """Apply predicted deltas to update the state vector. See original
    docstring in train.py's history — unchanged logic, moved verbatim."""
    idx = _get_integrate_state_indices(raw_state.device)
    new_state = raw_state.clone()

    if idx.delta_add_state_idx.numel() > 0:
        new_state[:, idx.delta_add_state_idx] = (
            raw_state[:, idx.delta_add_state_idx] + raw_delta[:, idx.delta_add_target_idx]
        )

    if idx.direct_state_idx.numel() > 0:
        vals = raw_delta[:, idx.direct_target_idx]
        if idx.direct_has_clamp:
            vals = torch.clamp(vals, min=idx.direct_clamp_min, max=idx.direct_clamp_max)
        new_state[:, idx.direct_state_idx] = vals

    if idx.recompute_state_idx.numel() > 0:
        new_state[:, idx.recompute_state_idx] = (
            new_state[:, idx.recompute_source_idx] - raw_state[:, idx.recompute_source_idx]
        ) / dt

    return new_state


_DELTA_POS_BODY_SLICE    = TARGET_FEATURE_MAP['delta_pos_body'].s
_DELTA_ORIENT_RVEC_SLICE = TARGET_FEATURE_MAP['delta_orient_rvec'].s


def integrate_pose(
    pos: torch.Tensor, quat: torch.Tensor, raw_delta: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Update world-frame pose from predicted deltas. Unchanged, moved verbatim."""
    delta_pos_body = raw_delta[:, _DELTA_POS_BODY_SLICE]
    pos = pos + torch_quat_rotate(delta_pos_body, quat)

    delta_rvec = raw_delta[:, _DELTA_ORIENT_RVEC_SLICE]
    delta_quat = torch_quat_from_rvec(delta_rvec)
    quat = torch_quat_normalize(torch_quat_multiply(delta_quat, quat))

    return pos, quat


# ──────────────────────────────────────────────────────────────────────
# NEW: the shared step primitive
# ──────────────────────────────────────────────────────────────────────

@dataclass
class NormTensors:
    """Denorm/norm stats as clamped torch tensors, built once per
    session and reused across every step_dynamics call, rather than
    re-clamped every timestep."""
    state_mean: torch.Tensor
    state_std: torch.Tensor
    target_mean: torch.Tensor
    target_std: torch.Tensor

    @classmethod
    def from_norm_torch(cls, norm_torch: dict[str, torch.Tensor]) -> 'NormTensors':
        return cls(
            state_mean=norm_torch['state_mean'],
            state_std=torch.clamp(norm_torch['state_std'], min=1e-8),
            target_mean=norm_torch['target_mean'],
            target_std=torch.clamp(norm_torch['target_std'], min=1e-8),
        )


@dataclass
class DynamicsState:
    """Everything step_dynamics needs to carry forward one step.
    Deliberately opaque beyond this to callers like Planner — if the
    model's recurrent core ever changes (GRU hidden vector -> a
    transformer KV-cache), only this dataclass's `hidden` field's type
    changes, not every call site that carries a DynamicsState around."""
    raw_state: torch.Tensor   # (B, STATE_DIM) physical units
    pos: torch.Tensor         # (B, 3) world
    quat: torch.Tensor        # (B, 4) world, [W, X, Y, Z]
    hidden: torch.Tensor      # (B, H) GRU hidden state
    _live: LiveCenterlineState | None = None


@dataclass
class DynamicsStepResult:
    raw_state: torch.Tensor
    pos: torch.Tensor
    quat: torch.Tensor
    hidden: torch.Tensor
    pred_norm: torch.Tensor    # (B, TARGET_DIM) normalized model output — for loss
    raw_delta: torch.Tensor    # (B, TARGET_DIM) physical units — diagnostics


def step_dynamics(
    model: WorldModel,
    track_ctx: TrackContextExtractor,
    raw_state: torch.Tensor,
    pos: torch.Tensor,
    quat: torch.Tensor,
    hidden: torch.Tensor,
    action_norm: torch.Tensor,
    norm: NormTensors,
    dt: float,
) -> DynamicsStepResult:
    """
    One physics step: track-context query -> model forward -> integrate.

    pos/quat are detached before the track-context query (see
    track_context.py's query_torch docstring) — geometry lookup and the
    PointNet encoder's INPUT never receive gradient from the integrated
    pose, only the encoder's own weights receive gradient from the
    downstream loss. Matches train.py's original rollout() exactly.
    """
    ctx = track_ctx.query_torch(pos.detach(), quat.detach())

    s_norm = (raw_state - norm.state_mean) / norm.state_std
    new_hidden, pred = model(s_norm, action_norm, ctx, hidden)
    new_hidden = new_hidden.clone()  # cudagraphs aliasing safety; harmless in eager mode

    raw_delta = pred * norm.target_std + norm.target_mean
    new_raw_state = integrate_state(raw_state, raw_delta, dt)
    new_pos, new_quat = integrate_pose(pos, quat, raw_delta)

    return DynamicsStepResult(
        raw_state=new_raw_state, pos=new_pos, quat=new_quat, hidden=new_hidden,
        pred_norm=pred, raw_delta=raw_delta,
    )