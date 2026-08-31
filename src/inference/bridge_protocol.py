"""
Contract the (separately-built) Python<->game bridge must satisfy.
Not implemented here.
"""
from __future__ import annotations
from typing import Protocol

from action_codec import GameAction


class GameBridge(Protocol):
    def step(self, action: GameAction, hold_ticks: int) -> dict:
        """
        Apply `action` for `hold_ticks` consecutive raw 100Hz physics
        ticks (action-hold), returning telemetry sampled at the LAST of
        those ticks — mirroring exactly how windowing.read_raw_window's
        stride samples training data. `hold_ticks` should always be
        `run_config.sampling.subsample_factor`, never hardcoded.

        Returned dict must contain every key in config.HDF5_RAW_KEYS
        (same physical units as ingest_raw.py's extract_segment,
        'pos'/'quat' included — quat hemisphere-fixed & normalized)
        plus:
          'launched_respawn': bool
          'static_respawn': bool
        """
        ...

    def reset_to_start(self) -> dict:
        """Respawn at track start; return initial telemetry (same shape as step())."""
        ...