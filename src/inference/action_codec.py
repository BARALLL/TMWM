"""
Action <-> tensor codec.

Given ACTION_FEATURES' current norms (steer/gas/brake are all
MinMaxNorm with out-range == physical range), normalization here is
the identity — but nothing downstream should hardcode that fact.
If ACTION_FEATURES changes (e.g. the action-key steering limiters get
added back), only this file needs to change.
"""
from __future__ import annotations
from dataclasses import dataclass

import torch

from features import ACTION_FEATURES, ACTION_DIM


@dataclass(frozen=True)
class GameAction:
    """One tick's worth of game input, in the game's own representation."""
    steer: float   # [-1, 1]
    gas: bool
    brake: bool


def encode(action: GameAction, device: torch.device) -> torch.Tensor:
    """GameAction -> (ACTION_DIM,) tensor, ordered per ACTION_FEATURES."""
    values = {
        'eff_steer': float(max(-1.0, min(1.0, action.steer))),
        'gas':   1.0 if action.gas else 0.0,
        'brake': 1.0 if action.brake else 0.0,
    }
    out = torch.empty(ACTION_DIM, dtype=torch.float32, device=device)
    for af in ACTION_FEATURES:
        assert af.dim == 1, f"encode() assumes scalar action features, got {af}"
        out[af.s] = values[af.name]
    return out


def encode_batch(steer: torch.Tensor, gas: torch.Tensor, brake: torch.Tensor) -> torch.Tensor:
    """
    Vectorized encode for CEM candidate populations.
    steer, gas, brake: (...,) tensors, same shape. Returns (..., ACTION_DIM).
    """
    steer = torch.clamp(steer, -1.0, 1.0)
    parts = {'eff_steer': steer, 'gas': gas.float(), 'brake': brake.float()}
    cols = []
    for af in ACTION_FEATURES:
        assert af.dim == 1, f"encode_batch() assumes scalar action features, got {af}"
        cols.append(parts[af.name].unsqueeze(-1))
    return torch.cat(cols, dim=-1)


def decode(action_tensor: torch.Tensor) -> GameAction:
    """(ACTION_DIM,) tensor -> GameAction, for sending to the bridge."""
    d = {af.name: action_tensor[af.s] for af in ACTION_FEATURES}
    return GameAction(
        steer=float(d['eff_steer'].reshape(-1)[0].item()),
        gas=bool(d['gas'].reshape(-1)[0].item() > 0.5),
        brake=bool(d['brake'].reshape(-1)[0].item() > 0.5),
    )