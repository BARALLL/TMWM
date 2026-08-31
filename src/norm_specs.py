"""
Normalization strategy declarations.

These are pure parameter types. Actual normalization computation
lives in compute_norm_stats.py and dataset.py.
"""

from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class ZScoreNorm:
    """Standard z-score: mean and std computed from training data."""
    pass


@dataclass(frozen=True)
class MinMaxNorm:
    """
    Linear map from [lo, hi] → [out_lo, out_hi].

    Equivalent to z-score with:
        mean = lo - out_lo * (hi - lo) / (out_hi - out_lo)
        std  = (hi - lo) / (out_hi - out_lo)
    """
    lo: float
    hi: float
    out_lo: float = 0.0
    out_hi: float = 1.0

    def to_mean_std(self, dim: int) -> tuple[np.ndarray, np.ndarray]:
        s = (self.hi - self.lo) / (self.out_hi - self.out_lo)
        m = self.lo - self.out_lo * s
        return (
            np.full(dim, m, dtype=np.float32),
            np.full(dim, s, dtype=np.float32),
        )


@dataclass(frozen=True)
class IdentityNorm:
    """No-op: mean=0, std=1."""
    pass


NormSpec = ZScoreNorm | MinMaxNorm | IdentityNorm