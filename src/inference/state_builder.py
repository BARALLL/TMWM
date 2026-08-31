"""
LiveStateBuilder: raw per-tick telemetry -> physical STATE_DIM vector.

Replaces features.InferenceState. Differences that matter:
  - The ring buffer holds only REAL observed telemetry (including real
    pos) — never fabricated data. Any current or future StateFeature
    that reads position directly is safe by construction.
  - Required lookback is read from features.required_state_lookback(),
    not hardcoded to a specific feature (e.g. damper_len).
  - Calls features.compute_state_features directly — the same pure
    function windowing.py/dataset.py call — so there is no parallel
    reimplementation of feature math.

Lives here, not in features.py: this is a stateful, single-consumer
(BeliefTracker) adapter, not part of the declarative registry that
dataset.py/compute_norm_stats.py also depend on.
"""
from __future__ import annotations
from collections import deque

import numpy as np

from config import HDF5_RAW_KEYS, DT
from features import compute_state_features, required_state_lookback


class LiveStateBuilder:
    def __init__(self, dt: float = DT):
        self.dt = dt
        self._lookback = required_state_lookback()
        self._buffer: deque[dict[str, np.ndarray]] = deque(maxlen=self._lookback + 1)

    def reset(self) -> None:
        """Call on lap/segment start and on every respawn — no history
        should carry across a discontinuity, mirroring exactly which
        points ingest_raw.py's detect_discontinuities cuts training
        sequences at."""
        self._buffer.clear()

    def push_and_build(self, telemetry: dict) -> np.ndarray:
        """
        Args:
            telemetry: dict containing every key in HDF5_RAW_KEYS, one
                real observed tick (pos, quat, vel_world, angvel_world,
                damper_len, wheel_rot_spd, cur_gear, ...) — exactly what
                a bridge tick naturally produces, per bridge_protocol.py.

        Returns:
            state_phys: (STATE_DIM,) — current-frame physical state.

        Note: on the very first tick after reset() (buffer length 1),
        any CausalGradient-derived feature falls back to zero rather
        than training's "hold first valid diff" convention (see
        features._causal_gradient's two different fallback paths for
        T<2 vs. a true window boundary). This is a one-tick-per-reset
        discrepancy, currently unreachable since no CausalGradient
        feature is active — noted rather than engineered around.
        """
        self._buffer.append({k: np.asarray(telemetry[k], dtype=np.float64) for k in HDF5_RAW_KEYS})

        window = {k: np.stack([tick[k] for tick in self._buffer], axis=0) for k in HDF5_RAW_KEYS}
        states, _ = compute_state_features(window, window['quat'], self.dt)
        return states[-1]