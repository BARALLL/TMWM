"""
Segment-windowing logic shared between compute_norm_stats.py (whole-
segment passes) and dataset.py (fixed-length training windows).

Both callers need to:
  1. Decide whether a segment is long enough to ever matter, given
     (window_len, subsample_factor) — see WindowIndex.windows_per_seg's
     original comment about this needing to match compute_norm_stats.py
     exactly. Previously this was two independently-written expressions
     that happened to be equivalent; now it's one function.
  2. Read a span of raw HDF5 frames at a given subsample_factor, with
     dataset.py additionally wanting 1 extra frame of look-back context
     (see compute_window_features's docstring for why).
  3. Run features.py's three compute_*_features functions over that
     span and trim to exactly what the model consumes: state/action
     drop the window's final frame (it exists only to supply the last
     target/pose), target keeps all frames.

This module is the single implementation of that pipeline. Before this
existed, dataset.py and compute_norm_stats.py each reimplemented steps
2-3 by hand, held in sync only by comments promising they matched.
"""

from dataclasses import dataclass

import numpy as np

from config import HDF5_RAW_KEYS
from features import (
    compute_state_features, compute_action_features, compute_target_features,
)


def effective_length(raw_length: int, subsample_factor: int) -> int:
    """Number of frames available after subsampling by subsample_factor."""
    return raw_length // subsample_factor


def segment_is_usable(raw_length: int, window_len: int, subsample_factor: int) -> bool:
    """
    True iff a segment of this raw length will EVER produce at least one
    training window at the given (window_len, subsample_factor).

    This is the single definition of "long enough" — dataset.py's
    WindowIndex and compute_norm_stats.py both call this instead of
    each re-deriving `eff_len <= window_len` independently.
    """
    return effective_length(raw_length, subsample_factor) > window_len


def windows_in_segment(raw_length: int, window_len: int, subsample_factor: int) -> int:
    """Number of distinct training windows a segment of this raw length yields."""
    eff_len = effective_length(raw_length, subsample_factor)
    return max(0, eff_len - window_len)


@dataclass
class WindowFeatures:
    """Model-ready feature tensors for one window (raw, un-normalized)."""
    state:  np.ndarray   # (L-1, STATE_DIM)  float64
    action: np.ndarray   # (L-1, ACTION_DIM) float64
    target: np.ndarray   # (L-1, TARGET_DIM) float64
    pos:    np.ndarray   # (L, 3)
    quat:   np.ndarray   # (L, 4)


def read_raw_window(
    seg_grp,
    raw_start: int,
    raw_stop: int,
    subsample_factor: int,
    lo_pad: int = 0,
) -> tuple[dict[str, np.ndarray], int]:
    """
    Read HDF5_RAW_KEYS from an HDF5 segment group over [raw_start, raw_stop),
    subsampled by subsample_factor, optionally prepending `lo_pad` extra
    EFFECTIVE (already-subsampled) frames of look-back context.

    lo_pad is silently clamped to 0 if raw_start - lo_pad*subsample_factor
    would go negative (i.e. the window starts at the true beginning of
    the segment) — there's no earlier context to read.

    Returns:
        (window_dict, actual_lo_pad_used)
    """
    sf = subsample_factor
    actual_lo_pad = lo_pad if (raw_start - lo_pad * sf) >= 0 else 0
    if actual_lo_pad != lo_pad:
        # Optional: increment a counter or log once
        pass
    pad_start = raw_start - actual_lo_pad * sf
    step = slice(pad_start, raw_stop, sf)
    window = {k: seg_grp[k][step] for k in HDF5_RAW_KEYS}
    return window, actual_lo_pad


def compute_window_features(
    raw_window: dict[str, np.ndarray],
    lo_pad: int,
    out_len: int,
    dt: float,
) -> WindowFeatures:
    """
    Compute state/action/target features from a (possibly look-back
    padded) raw window, and trim to exactly what the model consumes.

    Args:
        raw_window: dict of HDF5_RAW_KEYS arrays, length >= lo_pad + out_len.
            May include `lo_pad` extra frames of look-back at the front
            (see read_raw_window) purely so gradient-based features
            (damper_rate) get a real previous frame instead of hitting
            the repeat-first-difference fallback at a window boundary
            that isn't also a true segment boundary — see features.py's
            _causal_gradient docstring. compute_norm_stats.py always
            passes lo_pad=0 since it processes whole segments starting
            at frame 0, so it hits that same fallback only once, at the
            true first frame of the segment — identically to dataset.py.
        lo_pad: number of look-back frames to discard from the front
            after feature computation (0 for whole-segment passes).
        out_len: number of "full" effective frames desired (i.e. W+1
            for a training window of length W, or eff_len for a whole
            segment pass).
        dt: effective seconds between consecutive frames (see
            config.get_effective_dt) — must match how raw_window was
            subsampled.

    Returns:
        WindowFeatures with state/action/target/pos/quat already
        trimmed to model-ready shapes: state/action drop the window's
        final frame (it exists only to supply the last target/pose);
        target keeps all out_len - 1 frames.
    """
    avail = len(raw_window['pos'])
    assert avail >= lo_pad + out_len, (
        f"raw_window has {avail} frames, but lo_pad ({lo_pad}) + "
        f"out_len ({out_len}) = {lo_pad + out_len} were requested"
    )
    
    states_p, cache_p = compute_state_features(raw_window, quat=raw_window['quat'], dt=dt)
    actions_p = compute_action_features(raw_window)
    targets_p = compute_target_features(raw_window, cache_p, quat=raw_window['quat'], dt=dt)

    states  = states_p[lo_pad: lo_pad + out_len]
    actions = actions_p[lo_pad: lo_pad + out_len]
    targets = targets_p[lo_pad: lo_pad + out_len - 1]
    pos     = raw_window['pos'][lo_pad: lo_pad + out_len]
    quat    = raw_window['quat'][lo_pad: lo_pad + out_len]

    return WindowFeatures(
        state=states[:-1],
        action=actions[:-1],
        target=targets,
        pos=pos,
        quat=quat,
    )