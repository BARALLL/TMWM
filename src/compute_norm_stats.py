"""
Normalization statistics computation.

Iterates through all HDF5 segments, applies feature computation from
features.py, and accumulates per-dimension mean/std using a vectorized
(Chan et al.) parallel variance merge — mathematically equivalent to
Welford's algorithm but computed as one numpy operation per segment
instead of a Python loop per frame.

Output: norm_stats.npz (+ norm_stats.meta.json) consumed by dataset.py
and by run_pipeline.py's cache-staleness check.

Re-run when:
  - Feature definitions change (features.py)
  - New replays are ingested (replays.h5 changes)
  - You want different normalization (e.g., switch ZScore → MinMax)
  - SamplingConfig changes — gradient-based features (damper_rate)
    have a scale that depends on the effective timestep between
    frames, so stats computed at the wrong subsample_factor will be
    systematically wrong for those dimensions.

Does NOT need to re-run for:
  - Model architecture changes
  - Hyperparameter changes
  - Track geometry changes
"""

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from config import (
    HDF5_RAW_KEYS, SamplingConfig,
    stage_is_stale, save_stage_meta,
)
from features import (
    STATE_DIM, TARGET_DIM, ACTION_DIM,
    STATE_FEATURES, TARGET_FEATURES, ACTION_FEATURES,
)
from norm_specs import ZScoreNorm, MinMaxNorm, IdentityNorm
from windowing import segment_is_usable, effective_length, read_raw_window, compute_window_features


class RunningStats:
    """
    Online mean/variance accumulator using Chan et al.'s parallel-merge
    formula for combining (count, mean, M2) summaries.

    Mathematically equivalent to running Welford's algorithm one row at
    a time, but each call to `update` merges an entire batch (segment)
    in a single vectorized numpy pass — critical for performance, since
    segments can have thousands of frames and there are many segments.

    Handles NaN/Inf per-dimension independently: if dimension j has
    fewer finite observations than dimension i across the corpus (e.g.
    different NaN patterns across sensors), each dimension's mean/std
    is computed only over its own valid observations.
    """
    def __init__(self, dim: int):
        self.dim = dim
        self.count = np.zeros(dim, dtype=np.int64)
        self.mean = np.zeros(dim, dtype=np.float64)
        self.m2 = np.zeros(dim, dtype=np.float64)

    def update(self, batch: np.ndarray) -> None:
        """Merge a batch of shape (T, dim) into the running accumulator."""
        if len(batch) == 0:
            return

        valid = np.isfinite(batch)                        # (T, dim)
        batch_count = valid.sum(axis=0).astype(np.int64)   # (dim,)
        has_data = batch_count > 0
        safe_batch_count = np.maximum(batch_count, 1)

        # Mean/M2 of this batch, over its finite values only.
        batch_zeroed = np.where(valid, batch, 0.0)
        batch_mean = batch_zeroed.sum(axis=0) / safe_batch_count
        sq_dev = np.where(valid, (batch - batch_mean) ** 2, 0.0)
        batch_m2 = sq_dev.sum(axis=0)

        # Chan et al. parallel merge of two (count, mean, M2) summaries:
        # existing accumulator (A) and this batch (B).
        new_count = self.count + batch_count
        safe_new_count = np.maximum(new_count, 1)
        delta = batch_mean - self.mean
        merged_mean = self.mean + delta * (batch_count / safe_new_count)
        merged_m2 = (
            self.m2 + batch_m2
            + delta ** 2 * (self.count * batch_count) / safe_new_count
        )

        # Only touch dimensions that actually had finite data in this batch.
        self.mean = np.where(has_data, merged_mean, self.mean)
        self.m2 = np.where(has_data, merged_m2, self.m2)
        self.count = new_count

    @property
    def std(self) -> np.ndarray:
        safe = np.maximum(self.count, 2)
        variance = self.m2 / (safe - 1)
        return np.sqrt(np.maximum(variance, 0.0))


def _build_norm_arrays(
    features: list,
    total_dim: int,
    data_mean: np.ndarray,
    data_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assemble flat (mean, std) arrays from feature declarations and data stats.

    Per-feature rules:
        ZScoreNorm   → mean/std from data; std values below 1e-6 (i.e.
                       effectively-constant dimensions where division
                       would blow up) are replaced with 1.0 instead of
                       being used directly.
        MinMaxNorm   → mean/std from declared bounds (ignores data)
        IdentityNorm → mean=0, std=1
    """
    mean = np.zeros(total_dim, dtype=np.float32)
    std  = np.ones(total_dim, dtype=np.float32)

    for feat in features:
        spec = feat.norm
        if isinstance(spec, ZScoreNorm):
            m = data_mean[feat.s].astype(np.float32)
            s = data_std[feat.s].astype(np.float32)
            s = np.where(s < 1e-6, 1.0, s)
            mean[feat.s] = m
            std[feat.s] = s
        elif isinstance(spec, MinMaxNorm):
            m, s = spec.to_mean_std(feat.dim)
            mean[feat.s] = m
            std[feat.s] = s
        elif isinstance(spec, IdentityNorm):
            pass  # already 0, 1
        else:
            raise TypeError(f'Unknown NormSpec: {type(spec)}')

    return mean, std


def list_replay_ids(h5_path: Path) -> list[int]:
    """List all replay IDs present in the HDF5 file."""
    ids = []
    with h5py.File(h5_path, 'r') as h5:
        for key in h5.keys():
            if key.startswith('replay_'):
                try:
                    ids.append(int(key.split('_')[1]))
                except ValueError:
                    pass
    return sorted(ids)


def split_replays(
    ids: list[int],
    val_fraction: float = 0.1,
    seed: int = 42,
) -> tuple[list[int], list[int]]:
    """
    Split replay IDs into train and validation sets.

    Args:
        ids: all replay IDs
        val_fraction: fraction of replays to use for validation
        seed: random seed for reproducibility

    Returns:
        (train_ids, val_ids)

    val_fraction <= 0 returns an empty validation set (previously this
    silently forced at least 1 validation replay via `max(1, ...)` even
    when the caller explicitly asked for none).
    """
    ids_arr = np.array(ids)
    if val_fraction <= 0:
        return ids_arr.tolist(), []

    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(ids_arr) * val_fraction))
    n_val = min(n_val, len(ids_arr))  # can't hold out more replays than exist
    val_ids = rng.choice(ids_arr, size=n_val, replace=False)
    train_ids = np.setdiff1d(ids_arr, val_ids)
    return train_ids.tolist(), val_ids.tolist()


def norm_stats_cache_key(
    config: SamplingConfig,
    train_replay_ids: list[int] | None,
    h5_path: Path,
    seg_info_path: Path,
) -> dict:
    """
    Every input that determines norm_stats.npz's numeric content.
    This is the ONLY definition of "what makes norm stats stale" —
    used both to build the meta.json this module writes and to check
    staleness in run_pipeline.py. Diagnostic-only fields (n_segments,
    n_frames) are NOT part of this key — they're written to meta.json
    for visibility but never compared, since they're a deterministic
    consequence of the key fields, not an independent input.

    seg_info_path's mtime/size are included as defense-in-depth: in the
    normal workflow segment_info.json is always rewritten in lockstep
    with replays.h5 by ingest_raw.py, so h5_path's mtime/size alone
    would suffice — but nothing enforces that coupling at the type
    level. A hand-edited or independently-regenerated segment_info.json
    (e.g. while iterating on segmentation logic without re-running
    ingestion) would otherwise silently escape staleness detection even
    when it changes *which* segments/replays are visited without
    changing any individual segment's recorded length (the one thing
    the in-loop actual-vs-recorded length check below would catch).
    """
    seg_info_path = Path(seg_info_path)
    return {
        'subsample_factor': config.subsample_factor,
        'window_len': config.window_len,
        'train_replay_ids': sorted(train_replay_ids) if train_replay_ids is not None else None,
        'h5_mtime': h5_path.stat().st_mtime,
        'h5_size_bytes': h5_path.stat().st_size,
        'seg_info_mtime': seg_info_path.stat().st_mtime,
        'seg_info_size_bytes': seg_info_path.stat().st_size,
    }


def compute_and_save_norm_stats(
    h5_path: Path,
    output_path: Path,
    seg_info: list[dict],
    config: SamplingConfig,
    seg_info_path: Path,
    train_replay_ids: list[int] | None = None,
    verbose: bool = True,
) -> None:
    """
    Compute normalization statistics from the raw HDF5 file.

    Only processes segments belonging to train_replay_ids.  This prevents
    data leakage from validation replays into normalization parameters.

    Args:
        h5_path: path to replays.h5
        output_path: path to output norm_stats.npz
        seg_info: segment metadata list from segment_info.json — the
            SAME list dataset.py's WindowIndex is built from. Used to
            decide which segments to visit and whether each is long
            enough to matter (see windowing.segment_is_usable), so
            this script and the Dataset always agree on the segment
            population. `length` values in seg_info are NOT trusted
            for the actual slicing math — see the shape check below.
        config: SamplingConfig — MUST match the config passed to
            dataset.py. Determines effective_dt (affects damper_rate
            feature scale) and which segments are long enough to ever
            produce a training window.
        train_replay_ids: if None, uses all replays.
        verbose: print progress

    Raises:
        ValueError: if train_replay_ids is an empty list (as opposed to
            None, which means "use all replays"), or if zero segments
            end up usable — either would silently produce a garbage
            norm_stats.npz (ZScoreNorm dimensions falling back to
            mean=0/std=1 via the "std < 1e-6" guard in
            _build_norm_arrays) if not caught here. This is the one
            function guaranteed to run regardless of entry point
            (run_pipeline.py or standalone CLI), so it's the right
            place for this check — downstream `assert len(train_ds) > 0`
            in run_pipeline.py is not a substitute, since it doesn't
            run for standalone invocations.
    """
    if train_replay_ids is not None and len(train_replay_ids) == 0:
        raise ValueError(
            'train_replay_ids is an empty list — norm stats would be '
            'computed over zero replays. Pass train_replay_ids=None to '
            'use all replays, or check your train/val split (val_fraction '
            'may be too close to 1.0 for the number of replays available).'
        )

    state_stats  = RunningStats(STATE_DIM)
    action_stats = RunningStats(ACTION_DIM)
    target_stats = RunningStats(TARGET_DIM)

    effective_dt = config.effective_dt

    if train_replay_ids is not None:
        allowed = set(train_replay_ids)
        seg_info = [s for s in seg_info if s['replay_id'] in allowed]

    eligible = [
        s for s in seg_info
        if segment_is_usable(s['length'], config.window_len, config.subsample_factor)
    ]
    n_skipped_too_short = len(seg_info) - len(eligible)

    n_segments = 0
    n_frames = 0

    with h5py.File(h5_path, 'r') as h5:
        for seg in eligible:
            seg_grp = h5[f'replay_{seg["replay_id"]}/segment_{seg["segment_id"]}']

            # seg_info tells us WHICH segments to look at — cheap, no
            # HDF5 access needed for ineligible ones. But we never trust
            # its recorded `length` for the actual slice math: .shape on
            # an HDF5 dataset is a metadata-only read (no data I/O), so
            # this costs nothing and converts "segment_info.json is
            # stale relative to replays.h5" from a silent wrong-answer
            # bug into an immediate, loud failure.
            actual_len = seg_grp[HDF5_RAW_KEYS[0]].shape[0]
            if actual_len != seg['length']:
                raise ValueError(
                    f"segment_info.json length ({seg['length']}) != actual "
                    f"HDF5 data ({actual_len}) for replay {seg['replay_id']} "
                    f"segment {seg['segment_id']} — segment_info.json is "
                    f"stale; re-run ingestion."
                )

            eff_len = effective_length(actual_len, config.subsample_factor)
            stop = eff_len * config.subsample_factor
            window, lo_pad = read_raw_window(seg_grp, 0, stop, config.subsample_factor, lo_pad=0)
            T = len(window['pos'])
            assert T == eff_len

            if T < 2:
                continue

            wf = compute_window_features(window, lo_pad=lo_pad, out_len=eff_len, dt=effective_dt)

            state_stats.update(wf.state)
            action_stats.update(wf.action)
            target_stats.update(wf.target)

            n_segments += 1
            n_frames += T

    if n_segments == 0:
        n_replays_str = 'all replays' if train_replay_ids is None else f'{len(train_replay_ids)} train replays'
        raise ValueError(
            f'No usable segments found to compute normalization stats '
            f'({n_replays_str}, {len(eligible)} eligible of {len(seg_info)} '
            f'total segments considered, {n_skipped_too_short} skipped as '
            f'too short). Check window_len/subsample_factor against actual '
            f'segment lengths, and verify the train/val split actually '
            f'produced a non-empty training set.'
        )

    if verbose:
        id_str = (f'{len(train_replay_ids)} train replays'
                  if train_replay_ids is not None else 'all replays')
        print(f'Processed {n_segments} segments, {n_frames} frames ({id_str})')
        print(f'Skipped {n_skipped_too_short} segments too short to ever '
              f'produce a window at window_len={config.window_len}, '
              f'subsample_factor={config.subsample_factor}')
        print(f'Subsample factor: {config.subsample_factor}  '
              f'(effective dt={effective_dt * 1000:.1f} ms)')

    # Build final mean/std arrays
    state_mean, state_std = _build_norm_arrays(
        STATE_FEATURES, STATE_DIM, state_stats.mean, state_stats.std,
    )
    action_mean, action_std = _build_norm_arrays(
        ACTION_FEATURES, ACTION_DIM, action_stats.mean, action_stats.std,
    )
    target_mean, target_std = _build_norm_arrays(
        TARGET_FEATURES, TARGET_DIM, target_stats.mean, target_stats.std,
    )

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        state_mean=state_mean,   state_std=state_std,
        action_mean=action_mean, action_std=action_std,
        target_mean=target_mean, target_std=target_std,
    )

    # Sidecar metadata — lets run_pipeline.py (or anyone) detect that
    # cached norm_stats.npz was built with different train/val split or
    # SamplingConfig than the current run wants, without needing to
    # inspect the .npz contents.
    save_stage_meta(
        output_path.with_suffix('.meta.json'),
        norm_stats_cache_key(config, train_replay_ids, h5_path, seg_info_path),
        n_segments=n_segments,
        n_frames=n_frames,
    )

    if verbose:
        meta_path = output_path.with_suffix('.meta.json')
        print(f'Saved to {output_path}')
        print(f'Saved metadata to {meta_path}')
        _print_summary(state_mean, state_std, STATE_FEATURES, 'State')
        _print_summary(action_mean, action_std, ACTION_FEATURES, 'Action')
        _print_summary(target_mean, target_std, TARGET_FEATURES, 'Target')


def _print_summary(mean, std, features, label):
    print(f'\n{label} normalization:')
    for feat in features:
        m = mean[feat.s]
        s = std[feat.s]
        if feat.dim == 1:
            print(f'  {feat.name:24s}  mean={m[0]:>10.4f}  std={s[0]:>10.4f}')
        else:
            print(f'  {feat.name:24s}  mean=[{m.min():.4f}..{m.max():.4f}]'
                  f'  std=[{s.min():.4f}..{s.max():.4f}]')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute normalization statistics')
    parser.add_argument('--h5', type=Path, required=True, help='Input replays.h5')
    parser.add_argument('--seg-info', type=Path, required=True,
                        help='segment_info.json (from ingest_raw.py)')
    parser.add_argument('--output', type=Path, required=True, help='Output norm_stats.npz')
    parser.add_argument('--val-fraction', type=float, default=0.1,
                        help='Fraction of replays to hold out for validation')
    parser.add_argument('--seed', type=int, default=42, help='Split random seed')
    parser.add_argument('--subsample-factor', type=int, default=1,
                        help='MUST match the subsample_factor passed to dataset.py '
                             'at training time — affects gradient-based feature scale')
    parser.add_argument('--window-len', type=int, default=32,
                        help='MUST match the window_len passed to dataset.py / '
                            'run_pipeline.py at training time — determines which '
                            'segments are long enough to ever produce a training '
                            'window (see WindowIndex.windows_per_seg). Using the '
                            'wrong value here silently lets stats be computed '
                            'over a different frame population than training '
                            'ever sees.')
    parser.add_argument('--use-all', action='store_true',
                        help='Use all replays (no train/val split for norms)')
    args = parser.parse_args()

    with open(args.seg_info) as f:
        seg_info = json.load(f)

    all_ids = sorted(set(s['replay_id'] for s in seg_info))

    if args.use_all:
        train_ids = None
    else:
        train_ids, val_ids = split_replays(all_ids, args.val_fraction, args.seed)
        print(f'Replay split: {len(train_ids)} train, {len(val_ids)} val')
        print(f'Val IDs: {val_ids}')

    config = SamplingConfig(window_len=args.window_len, subsample_factor=args.subsample_factor)

    compute_and_save_norm_stats(
        args.h5, args.output, seg_info, config, args.seg_info, train_ids,
    )