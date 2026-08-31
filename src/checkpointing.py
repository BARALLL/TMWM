"""
Self-contained checkpoint save/load for the Trackmania world model.

A checkpoint written by `save_checkpoint` carries everything needed to
either resume training or load the model standalone for inference:
the weights, optimizer state, the full RunConfig it was produced
under, training progress, and RNG state for exact resume continuity.
"""
from __future__ import annotations
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import RunConfig

FORMAT_VERSION = 1


def capture_rng_state() -> dict[str, Any]:
    state = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state['torch_cuda'] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    if 'torch_cuda' in state and torch.cuda.is_available():
        try:
            torch.cuda.set_rng_state_all(state['torch_cuda'])
        except RuntimeError:
            # e.g. resuming on a machine with a different GPU count
            # than the run that saved this checkpoint — not fatal,
            # just fall back to fresh CUDA RNG state.
            pass


def save_checkpoint(
    path: Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    run_config: RunConfig,
    best_val_mse: float,
    train_loss: float,
    val_metrics: dict[str, float],
) -> None:
    """
    Write a self-contained checkpoint. `epoch` is the last *completed*
    epoch (0-indexed) — resuming continues from `epoch + 1`.

    Written atomically (temp file + rename) so a crash mid-write never
    leaves a corrupt file at `path` — important since `path` may also
    be the target of a later --resume.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'format_version': FORMAT_VERSION,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'run_config': run_config.to_dict(),
        'best_val_mse': best_val_mse,
        'train_loss': train_loss,
        'val_mse': val_metrics['mse'],
        'val_pos_error_m': val_metrics['pos_error_m'],
        'rng_state': capture_rng_state(),
    }
    tmp_path = path.with_suffix(path.suffix + '.tmp')
    torch.save(payload, tmp_path)
    tmp_path.replace(path)


def load_checkpoint(path: Path, map_location=None) -> dict[str, Any]:
    path = Path(path)
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if 'format_version' not in ckpt:
        raise ValueError(
            f'{path} looks like a legacy checkpoint (bare state_dict, or '
            f'missing format_version) — not loadable via load_checkpoint(). '
            f'It was likely saved by an older version of this pipeline; '
            f'if it\'s just weights, load it directly with '
            f'model.load_state_dict(torch.load(path)) instead of --resume.'
        )
    return ckpt

def strip_compile_prefix(state_dict: dict) -> dict:
    """
    Strip the '_orig_mod.' prefix torch.compile's OptimizedModule adds
    to every state_dict key, so weights saved from a compiled model
    load cleanly into a plain (uncompiled) WorldModel at inference
    time. No-op if the prefix isn't present.
    """
    if not any(k.startswith('_orig_mod.') for k in state_dict):
        return state_dict
    return {k[len('_orig_mod.'):]: v for k, v in state_dict.items()}