"""
PyTorch Dataset for the Trackmania world model.

Architecture-agnostic: returns temporal windows of (state, action,
track_context, target) that work with MLPs, frame-stacking, RNNs,
and Transformers.

Design:
  - Index map is built from segment_info.json at init (fast, memory-light).
  - HDF5 is read on-the-fly in __getitem__ (one segment read per sample).
  - Feature computation is done per-window using features.py functions,
    with 1 extra frame of context read on each side (when available) so
    gradient-based features (damper_rate) don't see artificial boundary
    artifacts at window edges that wouldn't exist if the whole segment
    were processed at once (as compute_norm_stats.py does). The padding
    is transparent to callers — returned tensors always have the
    documented shapes.
  - Track context is queried from the TrackContextExtractor.
  - Normalization stats are loaded from norm_stats.npz.

For DataLoader with num_workers > 0, HDF5 handles are managed via
__getstate__/__setstate__ so each worker opens the file independently.
"""

import json
from typing import Optional
import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from config import SamplingConfig
from windowing import read_raw_window, compute_window_features, windows_in_segment

# ──────────────────────────────────────────────────────────────────────
# Index map — maps flat dataset index → (replay, segment, start_frame)
# ──────────────────────────────────────────────────────────────────────

class SegmentMeta:
    """One entry in the segment_info.json list."""
    __slots__ = ('replay_id', 'segment_id', 'length')

    def __init__(self, replay_id: int, segment_id: int, length: int):
        self.replay_id = replay_id
        self.segment_id = segment_id
        self.length = length


class WindowIndex:
    """
    Maps a flat index to a specific window in a specific HDF5 segment.

    Stores only segment metadata and cumulative window counts — O(S)
    memory where S = number of segments.  Lookup is O(log S) via
    numpy.searchsorted.
    """

    def __init__(self, seg_info: list[dict], config: SamplingConfig):
        self.config = config
        self.segments = [
            SegmentMeta(s['replay_id'], s['segment_id'], s['length'])
            for s in seg_info
        ]

        # Windows per segment — single definition shared with
        # compute_norm_stats.py, see windowing.windows_in_segment.
        self.windows_per_seg = [
            windows_in_segment(s.length, config.window_len, config.subsample_factor)
            for s in self.segments
        ]

        # Cumulative window counts for binary search
        self.cum_windows = np.cumsum([0] + self.windows_per_seg).astype(np.int64)
        self.total_windows = int(self.cum_windows[-1])

    def __len__(self) -> int:
        return self.total_windows

    def get_location(self, idx: int) -> tuple[int, int, int]:
        """
        Args:
            idx: flat index in [0, total_windows)

        Returns:
            (replay_id, segment_id, start_frame)
            where start_frame is in the SUBSAMPLED frame space
        """
        seg_idx = int(np.searchsorted(self.cum_windows, idx, side='right') - 1)
        seg_idx = min(seg_idx, len(self.segments) - 1)

        start_in_seg = idx - int(self.cum_windows[seg_idx])
        seg = self.segments[seg_idx]

        return seg.replay_id, seg.segment_id, start_in_seg


# ──────────────────────────────────────────────────────────────────────
# Normalization stats
# ──────────────────────────────────────────────────────────────────────

class NormStats:
    """
    Holds mean/std arrays for state, action, and target normalization.

    All normalization is: x_norm = (x - mean) / std
    Denormalization is:    x = x_norm * std + mean

    Loaded from norm_stats.npz produced by compute_norm_stats.py.

    NOTE on dtype: feature computation in features.py deliberately uses
    float64 internally (quaternion math and finite differences benefit
    from the extra precision, especially over long segments). The
    normalize_*/denormalize_* methods below are the single point where
    values cross from "numpy feature space" into "model tensor space",
    so they always return float32 — matching the model's parameter
    dtype and avoiding silent float64 tensors reaching torch layers.
    """

    def __init__(self, state_mean, state_std, action_mean, action_std,
                 target_mean, target_std):
        self.state_mean  = state_mean.astype(np.float32)
        self.state_std   = state_std.astype(np.float32)
        self.action_mean = action_mean.astype(np.float32)
        self.action_std  = action_std.astype(np.float32)
        self.target_mean = target_mean.astype(np.float32)
        self.target_std  = target_std.astype(np.float32)

    @classmethod
    def load(cls, path: str) -> 'NormStats':
        with np.load(path) as d:
            return cls(
                d['state_mean'], d['state_std'],
                d['action_mean'], d['action_std'],
                d['target_mean'], d['target_std'],
            )

    def normalize_state(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.state_mean) / np.maximum(self.state_std, 1e-8)).astype(np.float32)

    def normalize_action(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.action_mean) / np.maximum(self.action_std, 1e-8)).astype(np.float32)

    def normalize_target(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.target_mean) / np.maximum(self.target_std, 1e-8)).astype(np.float32)

    def denormalize_target(self, x: np.ndarray) -> np.ndarray:
        return (x * self.target_std + self.target_mean).astype(np.float32)

    def denormalize_state(self, x: np.ndarray) -> np.ndarray:
        return (x * self.state_std + self.state_mean).astype(np.float32)

    def to_torch(self, device: torch.device) -> dict[str, torch.Tensor]:
        """Convert all stats to torch tensors on the given device."""
        return {
            'state_mean':  torch.tensor(self.state_mean,  dtype=torch.float32, device=device),
            'state_std':   torch.tensor(self.state_std,   dtype=torch.float32, device=device),
            'action_mean': torch.tensor(self.action_mean, dtype=torch.float32, device=device),
            'action_std':  torch.tensor(self.action_std,  dtype=torch.float32, device=device),
            'target_mean': torch.tensor(self.target_mean, dtype=torch.float32, device=device),
            'target_std':  torch.tensor(self.target_std,  dtype=torch.float32, device=device),
        }


# ──────────────────────────────────────────────────────────────────────
# PyTorch Dataset
# ──────────────────────────────────────────────────────────────────────

class TMWorldModelDataset(Dataset):
    """
    Yields temporal windows for world model training.

    Each sample is a dict:
        state:   (W, STATE_DIM)   — normalized car state
        action:  (W, ACTION_DIM)  — normalized action
        target:  (W, TARGET_DIM)  — normalized deltas from t → t+1
        pos:     (W+1, 3)         — raw world positions
        quat:    (W+1, 4)         — raw world quaternions

    Track context is intentionally NOT computed here. The training loop
    (train.py::rollout) queries geometry at each *predicted* pose during
    autoregressive rollout, which is not knowable ahead of time — so a
    per-sample ground-truth-pose track query here would be both wasted
    compute (never read by the training loop) and semantically wrong
    (the model needs geometry at where it *thinks* it is, not where the
    recorded replay was).
    """

    def __init__(
        self,
        h5_path: str,
        seg_info_path: str,
        norm_stats_path: str,
        config: SamplingConfig,
        replay_ids: list[int] | None = None,
    ):
        self.h5_path = str(h5_path)
        self.config = config
        self.norm = NormStats.load(norm_stats_path)

        # Build index map from segment metadata
        with open(seg_info_path) as f:
            seg_info = json.load(f)

        # Filter to requested replay IDs (prevents train/val data leakage)
        if replay_ids is not None:
            allowed = set(replay_ids)
            seg_info = [s for s in seg_info if s['replay_id'] in allowed]

        self.index = WindowIndex(seg_info, config)

        # HDF5 handle — managed for multi-worker safety
        self._h5: Optional[h5py.File] = None

    def _get_h5(self) -> h5py.File:
        """Get HDF5 file handle, opening lazily if needed.

        Plain read-only mode is used (no SWMR): there is never a
        concurrent writer to replays.h5 during training, so SWMR adds
        compatibility risk (it requires the writer to have opened the
        file with libver='latest' and swmr_mode=True, which
        ingest_raw.py does not do) without providing any benefit here.
        """
        if self._h5 is None or not self._h5.id.valid:
            self._h5 = h5py.File(self.h5_path, 'r')
        return self._h5

    # ── Pickle safety for DataLoader multi-worker ─────────────────────
    # HDF5 file handles can't be pickled.  By setting _h5 to None
    # during pickling, each worker re-opens the file independently.

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_h5'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Don't open immediately — let _get_h5() open lazily

    # ── Dataset interface ─────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        replay_id, seg_id, start_t = self.index.get_location(idx)
        W = self.config.window_len
        sf = self.config.subsample_factor

        raw_start = start_t * sf
        raw_end = raw_start + (W + 1) * sf   # exclusive

        h5 = self._get_h5()
        seg_grp = h5[f'replay_{replay_id}/segment_{seg_id}']

        # 1 frame of look-back padding (when available) so gradient
        # features don't see boundary artifacts — see
        # windowing.compute_window_features's docstring.
        window, lo_pad = read_raw_window(seg_grp, raw_start, raw_end, sf, lo_pad=1)
        wf = compute_window_features(window, lo_pad=lo_pad, out_len=W + 1, dt=self.config.effective_dt)

        norm_state  = self.norm.normalize_state(wf.state)
        norm_action = self.norm.normalize_action(wf.action)
        norm_target = self.norm.normalize_target(wf.target)

        return {
            'state':  torch.from_numpy(norm_state),
            'action': torch.from_numpy(norm_action),
            'target': torch.from_numpy(norm_target),
            'pos':    torch.from_numpy(wf.pos.astype(np.float32)),
            'quat':   torch.from_numpy(wf.quat.astype(np.float32)),
        }

    def close(self) -> None:
        if self._h5 is not None and self._h5.id.valid:
            self._h5.close()
            self._h5 = None


# ──────────────────────────────────────────────────────────────────────
# DataLoader factory
# ──────────────────────────────────────────────────────────────────────

def _seed_worker(worker_id: int, base_seed: int | None) -> None:
    """
    Per-DataLoader-worker init: seed each worker's RNGs deterministically
    from (base_seed, worker_id) — NOT the same seed in every worker,
    which would make every worker's "randomness" identical and defeat
    the point of parallelism. Currently __getitem__ has no randomness of
    its own (fully determined by idx), so this doesn't change today's
    behavior — it's precautionary, so that if randomized
    augmentation/sampling is ever added inside __getitem__, it's
    reproducible for free rather than silently not being so.

    Also resets `dataset._h5` to force each worker to open its own HDF5
    handle (see class docstring / __getstate__).
    """
    if base_seed is not None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    worker_info = torch.utils.data.get_worker_info()
    worker_info.dataset._h5 = None


def _worker_init_fn(worker_id: int) -> None:
    """
    Called once per DataLoader worker.

    Each worker opens its own HDF5 handle so there's no contention
    on a shared file descriptor.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # Force a fresh HDF5 open in this worker
    dataset._h5 = None


def create_dataloader(
    dataset: TMWorldModelDataset,
    batch_size: int = 32,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    seed: int | None = None,
) -> DataLoader:
    """
    Create a DataLoader with proper multi-worker HDF5 handling.

    Uses __getstate__/__setstate__ on the Dataset for pickle safety.
    drop_last is only applied during training (shuffle=True) to avoid
    silently discarding validation samples.

    `seed`, if given, is used to derive a per-worker RNG seed (see
    _seed_worker) when num_workers > 0. The shuffling ORDER itself is
    controlled separately, by torch's global RNG in the main process
    (see config.set_seed) — this only affects randomness that might
    ever run inside a worker process's __getitem__ call.
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=_worker_init_fn if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
        drop_last=shuffle,
    )