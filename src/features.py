"""
Feature definitions and computation for the Trackmania world model.

THIS IS THE SINGLE SOURCE OF TRUTH for:
  - which state/action/target features exist
  - how they are derived from raw HDF5 data
  - how predictions feed back during rollout
  - their normalization specs

The four stages of every feature's lifecycle:
  1. COMPUTE   → derive a value from raw data / cache (training & inference)
  2. SUPERVISE → is it a prediction target? what norm?
  3. INTEGRATE → how does the predicted value update rollout state?
  4. RECOMPUTE → post-integration fixup (damper_rate)

PICKLE SAFETY: All callable fields in feature specs (Derived.fn,
PoseDerived.fn) MUST be module-level named functions — not lambdas.
PyTorch DataLoader with num_workers > 0 pickles the Dataset, and
while module-level feature lists aren't directly in the pickle stream
(they're accessed by module attribute lookup at call time, not stored
as instance attributes), named functions are still the safe default:
they're pickleable by (module, qualname) reference, debuggable via
tracebacks, and testable in isolation.

Used by:
  - compute_norm_stats.py  (statistical pass over HDF5)
  - dataset.py             (on-the-fly feature computation in __getitem__)
  - train.py               (rollout integration)
  - inference.py           (live telemetry → state)

Change a feature definition → re-run compute_norm_stats.py → re-train.
Never touch ingest_raw.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import json
from typing import Callable, Union

import numpy as np

from config import DT
from norm_specs import ZScoreNorm, MinMaxNorm, NormSpec
from quaternion_utils import quat_rotate_inv, quat_to_rvec, quat_relative


# ═══════════════════════════════════════════════════════════════
# Axis 1: COMPUTE — where does the data come from?
#
# StateComputeSpec  → applied to state features (training + inference)
# TargetComputeSpec → applied to target features (training signal)
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Raw:
    """Copy directly from an HDF5 field."""
    field: str

@dataclass(frozen=True)
class BodyFrame:
    """Rotate a world-frame HDF5 field into the car's body frame."""
    field: str

@dataclass(frozen=True)
class CausalGradient:
    """Backward difference of a cached state feature: (x[t]-x[t-1])/dt."""
    source: str

@dataclass(frozen=True)
class StateDelta:
    """Difference of a cached state feature: x[t+1] - x[t]. Exact by construction."""
    source: str

@dataclass(frozen=True)
class BodyPosDelta:
    """Body-frame position change: R_t^T · (pos[t+1] - pos[t]). Exact by construction."""

@dataclass(frozen=True)
class BodyVelDelta:
    """Body-frame velocity change: R_t^T · (vel_world[t+1] - vel_world[t]).
    /!\\  Approximate: the rotation used is R_t, not R_{t+1}.
    Error is O(ω·dt) per step. Prefer StateDelta('vel_body') for exactness."""
    field: str          # HDF5 field, typically 'vel_world'

@dataclass(frozen=True)
class OrientDelta:
    """Relative orientation as rotation vector: rvec(q_t⁻¹ · q_{t+1})."""

@dataclass(frozen=True)
class SecondOrderDelta:
    """Difference of a causal gradient: rate[t+1] - rate[t]
    where rate = causal_gradient(raw_field)."""
    field: str          # the raw HDF5 field (gradient recomputed internally)

@dataclass(frozen=True)
class NextValue:
    """Direct value at t+1 from an HDF5 field."""
    field: str

StateComputeSpec = Union[Raw, BodyFrame, CausalGradient]
TargetComputeSpec = Union[
    StateDelta, BodyPosDelta, BodyVelDelta, OrientDelta,
    SecondOrderDelta, NextValue,
]


# ═══════════════════════════════════════════════════════════════
# Axis 3: INTEGRATE — how does the prediction update rollout state?
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class DeltaAdd:
    """Add predicted delta to a named state feature."""
    state_target: str

@dataclass(frozen=True)
class DirectAssign:
    """Replace a named state feature with predicted value."""
    state_target: str
    clamp: tuple[float, float] | None = None

@dataclass(frozen=True)
class PosePosUpdate:
    """pos += quat_rotate(pred_delta_body, quat)"""

@dataclass(frozen=True)
class PoseOrientUpdate:
    """quat *= quat_from_rvec(pred_delta_rvec)"""

@dataclass(frozen=True)
class SuperviseOnly:
    """Supervised during training. Not fed back during rollout."""


# ═══════════════════════════════════════════════════════════════
# Axis 4: RECOMPUTE — post-integration derivation
# ═══════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RecomputeCausalGradient:
    """After integration: (updated_source - old_source) / dt."""
    source: str


# ═══════════════════════════════════════════════════════════════
# Feature descriptors
# ═══════════════════════════════════════════════════════════════

@dataclass
class StateFeature:
    name: str
    dim: int
    derive: StateComputeSpec        # Axis 1
    norm: NormSpec                  # Axis 2
    recompute: RecomputeCausalGradient | None = None  # Axis 4
    s: slice = field(init=False, repr=False, compare=False)

@dataclass
class TargetFeature:
    name: str
    dim: int
    compute: TargetComputeSpec      # Axis 1
    integrate: (                     # Axis 3
        DeltaAdd | DirectAssign |
        PosePosUpdate | PoseOrientUpdate |
        SuperviseOnly
    )
    norm: NormSpec                  # Axis 2
    s: slice = field(init=False, repr=False, compare=False)

@dataclass
class ActionFeature:
    name: str
    dim: int
    raw_field: str
    norm: NormSpec
    s: slice = field(init=False, repr=False, compare=False)


# ═══════════════════════════════════════════════════════════════
# Shared norm instances
# ═══════════════════════════════════════════════════════════════

GEAR_NORM  = MinMaxNorm(lo=0.0, hi=5.0, out_lo=0.0, out_hi=1.0)
STEER_NORM = MinMaxNorm(lo=-1.0, hi=1.0, out_lo=-1.0, out_hi=1.0)
GAS_NORM   = MinMaxNorm(lo=0.0, hi=1.0, out_lo=0.0, out_hi=1.0)
BRAKE_NORM = MinMaxNorm(lo=0.0, hi=1.0, out_lo=0.0, out_hi=1.0)


# ═══════════════════════════════════════════════════════════════
# Feature declarations — THE single source of truth
# ═══════════════════════════════════════════════════════════════

STATE_FEATURES: list[StateFeature] = [
    #  name               dim  derive                          norm          recompute
    StateFeature('vel_body',        3, BodyFrame('vel_world'),           ZScoreNorm()),
    StateFeature('angvel_body',     3, BodyFrame('angvel_world'),        ZScoreNorm()),
    # StateFeature('damper_len',      4, Raw('damper_len'),               ZScoreNorm()),
    # StateFeature('damper_rate',     4, CausalGradient('damper_len'),    ZScoreNorm(),
    #              recompute=RecomputeCausalGradient('damper_len')),
    # StateFeature('wheel_rot_speed', 4, Raw('wheel_rot_spd'),           ZScoreNorm()),
    # StateFeature('cur_gear',        1, Raw('cur_gear'),                GEAR_NORM),
]

TARGET_FEATURES: list[TargetFeature] = [
    #  name                     dim  compute                        integrate                              norm
    # ── Pose (exact) ──────────────────────────────────────────────────────────────────────────────────────────
    TargetFeature('delta_pos_body',    3, BodyPosDelta(),           PosePosUpdate(),                        ZScoreNorm()),
    TargetFeature('delta_orient_rvec', 3, OrientDelta(),            PoseOrientUpdate(),                     ZScoreNorm()),
    # ── Dynamics (exact) ──────────────────────────────────────────────────────────────────────────────────────
    TargetFeature('delta_vel_body',    3, StateDelta('vel_body'),   DeltaAdd('vel_body'),                   ZScoreNorm()),
    TargetFeature('delta_angvel_body', 3, StateDelta('angvel_body'),DeltaAdd('angvel_body'),                ZScoreNorm()),
    # TargetFeature('delta_damper_len',  4, StateDelta('damper_len'), DeltaAdd('damper_len'),                 ZScoreNorm()),
    # TargetFeature('delta_wheel_rot_speed', 4, StateDelta('wheel_rot_speed'), DeltaAdd('wheel_rot_speed'),  ZScoreNorm()),
    # # ── Assign ────────────────────────────────────────────────────────────────────────────────────────────────
    # TargetFeature('next_gear',         1, NextValue('cur_gear'),    DirectAssign('cur_gear', (0.0, 5.0)),   GEAR_NORM),
    # # ── Auxiliary (supervised, not fed back) ──────────────────────────────────────────────────────────────────
    # TargetFeature('delta_damper_rate', 4, SecondOrderDelta('damper_len'), SuperviseOnly(),                   ZScoreNorm()),
]

ACTION_FEATURES: list[ActionFeature] = [
                  #  name   dim   raw_field       norm
    ActionFeature('eff_steer', 1, 'input_steer', STEER_NORM),
    ActionFeature('gas',       1, 'input_gas',   GAS_NORM),
    ActionFeature('brake',     1, 'input_brake', BRAKE_NORM),
]


# ═══════════════════════════════════════════════════════════════
# Auto-derive slices, dims, maps
# ═══════════════════════════════════════════════════════════════

def _register(features: list) -> int:
    offset = 0
    for f in features:
        f.s = slice(offset, offset + f.dim)
        offset += f.dim
    return offset

STATE_DIM  = _register(STATE_FEATURES)
TARGET_DIM = _register(TARGET_FEATURES)
ACTION_DIM = _register(ACTION_FEATURES)

STATE_FEATURE_MAP:  dict[str, StateFeature]  = {f.name: f for f in STATE_FEATURES}
TARGET_FEATURE_MAP: dict[str, TargetFeature] = {f.name: f for f in TARGET_FEATURES}


# ═══════════════════════════════════════════════════════════════
# Validation — runs at import, catches wiring errors immediately
# ═══════════════════════════════════════════════════════════════

def _validate() -> None:
    for tf in TARGET_FEATURES:
        if isinstance(tf.integrate, DeltaAdd):
            assert tf.integrate.state_target in STATE_FEATURE_MAP, \
                f"TargetFeature '{tf.name}': DeltaAdd → unknown state '{tf.integrate.state_target}'"
        if isinstance(tf.integrate, DirectAssign):
            assert tf.integrate.state_target in STATE_FEATURE_MAP, \
                f"TargetFeature '{tf.name}': DirectAssign → unknown state '{tf.integrate.state_target}'"
            assert tf.integrate.clamp is not None, \
                f"TargetFeature '{tf.name}': DirectAssign requires clamp=(lo, hi)"
    for sf in STATE_FEATURES:
        if sf.recompute is not None:
            assert sf.recompute.source in STATE_FEATURE_MAP, \
                f"StateFeature '{sf.name}': recompute → unknown state '{sf.recompute.source}'"

_validate()


# ═══════════════════════════════════════════════════════════════
# Utility
# ═══════════════════════════════════════════════════════════════

def _causal_gradient(x: np.ndarray, dt: float) -> np.ndarray:
    """
    Causal (backward) first difference: rate[t] = (x[t] - x[t-1]) / dt.

    Central differences cannot be reproduced during autoregressive rollout:
    at inference time integrate_state only ever has access to x[t-1] and
    x[t], never x[t+1].  Using backward differences everywhere closes the
    train/inference gap exactly.

    The first row has no x[t-1]. Rather than fabricate a zero (which
    would bias norm stats and look like a hard brake/impact), it's filled
    with the first valid difference (rate[1]) — a "hold" convention.
    Because dataset.py pads every window with 1 frame of look-back
    whenever one is available within the segment, this fallback is only
    ever exercised at the true first frame of a segment — identically in
    both dataset.py (per-window) and compute_norm_stats.py (whole-segment),
    so statistics and training see the same values.
    """
    if len(x) < 2:
        return np.zeros_like(x)
    diffs = np.diff(x, axis=0) / dt                      # (T-1, D)
    return np.concatenate([diffs[:1], diffs], axis=0)     # (T, D)


def _ensure_2d(arr: np.ndarray) -> np.ndarray:
    """Guarantee shape (T, D)."""
    return arr[:, np.newaxis] if arr.ndim == 1 else arr


# ═══════════════════════════════════════════════════════════════
# Computation — pure functions on numpy arrays
#
# Called from:
#   1. compute_norm_stats.py — iterates over HDF5 segments
#   2. dataset.py           — __getitem__ on small windows
#   3. InferenceState.build — live telemetry (below)
#
# All callers pass a dict whose keys are HDF5_RAW_KEYS and whose
# values are numpy arrays of shape (T, D).
# ═══════════════════════════════════════════════════════════════

def compute_state_features(
    window: dict[str, np.ndarray],
    quat: np.ndarray,
    dt: float = DT,
    cache: dict[str, np.ndarray] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Compute all state features from a window of raw data.

    Args:
        window: dict mapping HDF5_RAW_KEYS → np.ndarray of shape (T, D).
        quat:   (T, 4) quaternions for this window.
        dt:     seconds between consecutive rows. Scaled if the caller
                has subsampled (DT * subsample_factor).
        cache:  optional pre-populated cache (e.g. from a preceding
                look-back frame). Entries from cache are used as-is
                for CausalGradient sources; newly computed entries
                are added or overwritten.

    Returns:
        states: (T, STATE_DIM)
        cache:  name → array, consumed by target computation
    """
    if cache is None:
        cache = {}
    else:
        cache = dict(cache)           # shallow copy — don't mutate caller's

    T = len(window['pos'])
    states = np.empty((T, STATE_DIM), dtype=np.float64)

    for sf in STATE_FEATURES:
        match sf.derive:
            case Raw(f):
                arr = _ensure_2d(window[f].astype(np.float64))

            case BodyFrame(f):
                arr = quat_rotate_inv(
                    _ensure_2d(window[f].astype(np.float64)),
                    quat,
                )

            case CausalGradient(src):
                arr = _causal_gradient(cache[src], dt)

        cache[sf.name] = arr
        states[:, sf.s] = arr

    return states, cache


def compute_target_features(
    window: dict[str, np.ndarray],
    cache: dict[str, np.ndarray],
    quat: np.ndarray,
    dt: float = DT,
) -> np.ndarray:
    """
    Compute all target features (deltas from frame t to t+1).

    Args:
        window: same dict passed to compute_state_features.
        cache:  the cache dict returned by compute_state_features.
        quat:   (T, 4) quaternions for this window.
        dt:     same meaning as in compute_state_features.

    Returns:
        targets: (T-1, TARGET_DIM). targets[i] is the delta from i → i+1.
    """
    T = len(window['pos'])
    targets = np.empty((T - 1, TARGET_DIM), dtype=np.float64)

    for tf in TARGET_FEATURES:
        match tf.compute:
            case StateDelta(src):
                arr = cache[src][1:] - cache[src][:-1]

            case SecondOrderDelta(f):
                raw = _ensure_2d(window[f].astype(np.float64))
                rate = _causal_gradient(raw, dt)
                arr = rate[1:] - rate[:-1]

            case NextValue(f):
                arr = _ensure_2d(window[f].astype(np.float64))[1:]

            case BodyPosDelta():
                delta = window['pos'][1:] - window['pos'][:-1]
                arr = quat_rotate_inv(delta, quat[:-1])

            case OrientDelta():
                arr = quat_to_rvec(quat_relative(quat[:-1], quat[1:]))

            case BodyVelDelta(f):
                delta = window[f][1:] - window[f][:-1]
                arr = quat_rotate_inv(delta, quat[:-1])

        targets[:, tf.s] = arr

    return targets


def compute_action_features(window: dict[str, np.ndarray]) -> np.ndarray:
    """
    Compute all action features.

    Returns:
        actions: (T, ACTION_DIM)
    """
    T = len(window['pos'])
    actions = np.empty((T, ACTION_DIM), dtype=np.float64)

    for af in ACTION_FEATURES:
        actions[:, af.s] = _ensure_2d(window[af.raw_field].astype(np.float64))

    return actions


def _compute_feature_schema_hash() -> str:
    """
    Fingerprint of exactly which features exist, in what order, with
    what dimensionality and derivation — NOT just STATE_DIM/TARGET_DIM/
    ACTION_DIM (which only catch *width* changes, not silent column-
    meaning changes that happen to preserve width).
    """
    def sig(f):
        extra = ''
        if hasattr(f, 'derive'):
            extra = type(f.derive).__name__
        elif hasattr(f, 'compute'):
            extra = type(f.compute).__name__
        elif hasattr(f, 'raw_field'):
            extra = f.raw_field
        return [f.name, f.dim, extra]

    payload = {
        'state':  [sig(f) for f in STATE_FEATURES],
        'target': [sig(f) for f in TARGET_FEATURES],
        'action': [sig(f) for f in ACTION_FEATURES],
    }
    canonical = json.dumps(payload, sort_keys=True)
    return hashlib.sha1(canonical.encode('utf-8')).hexdigest()

FEATURE_SCHEMA_HASH = _compute_feature_schema_hash()

# ═══════════════════════════════════════════════════════════════
# Live-inference support: how much raw history does the CURRENT
# STATE_FEATURES declaration need? Pure function of the registry —
# consumed by inference/state_builder.py to size its ring buffer
# WITHOUT hardcoding which feature needs lookback. Add, remove, or
# chain CausalGradient features and this adjusts automatically.
# ═══════════════════════════════════════════════════════════════

def required_state_lookback() -> int:
    """
    Max number of prior raw frames any STATE_FEATURE's derivation
    needs. A CausalGradient needs 1 frame of history over its source;
    a CausalGradient whose source is itself a CausalGradient needs one
    more than that, recursively. Raw/BodyFrame features need none.
    """
    depth: dict[str, int] = {}

    def compute_depth(name: str, _seen: frozenset[str] = frozenset()) -> int:
        if name in depth:
            return depth[name]
        assert name not in _seen, f"cyclic CausalGradient dependency involving '{name}'"
        sf = STATE_FEATURE_MAP[name]
        if isinstance(sf.derive, CausalGradient):
            d = 1 + compute_depth(sf.derive.source, _seen | {name})
        else:
            d = 0
        depth[name] = d
        return d

    return max((compute_depth(sf.name) for sf in STATE_FEATURES), default=0)