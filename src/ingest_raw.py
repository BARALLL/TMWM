"""
Layer A: Offline Ingestion (CSV → HDF5).

Parses raw replay CSVs, splits at physics discontinuities, and stores
continuous segments in an HDF5 file.  No feature engineering.

Output file structure:
    /replay_0/segment_0/pos        (N, 3) float32
    /replay_0/segment_0/quat       (N, 4) float32  (hemisphere-fixed)
    ...
    /replay_0/segment_1/...
    /replay_1/segment_0/...

Also writes segment_info.json with metadata for the Dataset index map.
"""

import json
import argparse
from pathlib import Path
import hashlib

import h5py
import numpy as np
import pandas as pd

from quaternion_utils import quat_normalize, fix_quat_hemisphere

# ──────────────────────────────────────────────────────────────────────
# CSV column names (the ONLY place they appear)
# ──────────────────────────────────────────────────────────────────────

_QUAT_COLS = (
    'orientationQuat_W', 'orientationQuat_X',
    'orientationQuat_Y', 'orientationQuat_Z',
)
_POS_COLS  = ('worldPosition_X', 'worldPosition_Y', 'worldPosition_Z')
_VEL_COLS  = ('linearVelocity_X', 'linearVelocity_Y', 'linearVelocity_Z')
_AVel_COLS = ('angularVelocity_X', 'angularVelocity_Y', 'angularVelocity_Z')
_DAMP_COLS = ('DamperLen_FL', 'DamperLen_FR', 'DamperLen_RL', 'DamperLen_RR')
_WHEL_COLS = ('WheelRotSpeed_FL', 'WheelRotSpeed_FR',
              'WheelRotSpeed_RL', 'WheelRotSpeed_RR')

# All CSV columns we need (used for validation)
REQUIRED_CSV_COLS: frozenset[str] = frozenset({
    'tick',
    'LaunchedRespawn', 'StaticRespawn',
    *_QUAT_COLS, *_POS_COLS, *_VEL_COLS, *_AVel_COLS,
    *_DAMP_COLS, *_WHEL_COLS,
    'CurGear', 'InputSteer', 'InputForward', 'InputBackward/brake',
})

# Discontinuity detection
_POS_JUMP_THRESHOLD = 3.0   # metres

# HDF5 storage dtype
_H5_DTYPE = np.float32


# ──────────────────────────────────────────────────────────────────────
# Discontinuity detection
# ──────────────────────────────────────────────────────────────────────

def detect_discontinuities(df: pd.DataFrame) -> np.ndarray:
    """
    Boolean mask where True marks the first frame of a new continuous segment.

    Detects:
      - Explicit respawn flags
      - Non-consecutive tick values
      - World-position jumps > 3 m
    """
    n = len(df)
    is_cut = np.zeros(n, dtype=bool)
    is_cut[0] = True

    is_cut |= df['LaunchedRespawn'].values.astype(bool)
    is_cut |= df['StaticRespawn'].values.astype(bool)

    not_first = np.arange(n) > 0

    ticks = df['tick'].values
    tick_diff = np.diff(ticks, prepend=ticks[0])
    is_cut |= (tick_diff != 10) & not_first

    print(is_cut[1::].nonzero())

    pos = np.stack([df[c].values for c in _POS_COLS], axis=1).astype(np.float64)
    pos_diff = np.linalg.norm(np.diff(pos, axis=0, prepend=pos[:1]), axis=1)
    is_cut |= (pos_diff > _POS_JUMP_THRESHOLD) & not_first

    print(is_cut[1::].any())

    return is_cut


# ──────────────────────────────────────────────────────────────────────
# Segment extraction
# ──────────────────────────────────────────────────────────────────────

def _extract_cols(df: pd.DataFrame, cols: tuple[str, ...]) -> np.ndarray:
    """Stack named columns into (N, D) array."""
    return np.stack([df[c].values for c in cols], axis=1)


def extract_segment(df: pd.DataFrame, start: int, end: int) -> dict[str, np.ndarray]:
    """
    Extract one contiguous segment [start, end) as a dict of arrays.

    Applies quaternion hemisphere fix and normalisation.
    """
    seg = df.iloc[start:end]

    # Quaternion hemispherication
    quat = _extract_cols(seg, _QUAT_COLS).astype(np.float64)
    quat = fix_quat_hemisphere(quat)
    quat = quat_normalize(quat)

    return {
        'pos':          _extract_cols(seg, _POS_COLS).astype(_H5_DTYPE),
        'quat':         quat.astype(_H5_DTYPE),
        'vel_world':    _extract_cols(seg, _VEL_COLS).astype(_H5_DTYPE),
        'angvel_world': _extract_cols(seg, _AVel_COLS).astype(_H5_DTYPE),
        'damper_len':   _extract_cols(seg, _DAMP_COLS).astype(_H5_DTYPE),
        'wheel_rot_spd':_extract_cols(seg, _WHEL_COLS).astype(_H5_DTYPE),
        'cur_gear':     seg['CurGear'].values.astype(_H5_DTYPE),
        'input_steer':  seg['InputSteer'].values.astype(_H5_DTYPE),
        'input_gas':    seg['InputForward'].values.astype(_H5_DTYPE),
        'input_brake':  seg['InputBackward/brake'].values.astype(_H5_DTYPE),
    }


# ──────────────────────────────────────────────────────────────────────
# HDF5 writing
# ──────────────────────────────────────────────────────────────────────

def _write_segment(h5_group: h5py.Group, data: dict[str, np.ndarray]) -> None:
    """Write a segment's arrays into an HDF5 group."""
    for key, arr in data.items():
        h5_group.create_dataset(key, data=arr, chunks=True)


# ──────────────────────────────────────────────────────────────────────
# Main ingestion
# ──────────────────────────────────────────────────────────────────────

def _stable_replay_id(csv_path: Path) -> int:
    """
    Derive a stable integer replay ID from the CSV filename — NOT from
    enumeration order over the directory listing.

    Why this matters: run_pipeline.py persists train/val splits and
    detects normalization-stats staleness by comparing sets of
    replay_ids across runs. If replay_id were merely the sorted-order
    index (the previous scheme), adding or removing a single CSV would
    silently renumber every replay sorting after it. Two ingestion runs
    over slightly different directory contents could then assign the
    SAME id to two different physical replays, or different ids to the
    same replay, silently defeating every staleness/leakage check that
    depends on replay_id being a stable identity — with no way to
    detect it after the fact.

    Hashing the filename instead makes the id: (a) stable regardless of
    what else is in the directory or ingestion order, and (b) collision
    risk is negligible for realistic dataset sizes (60 bits of a
    cryptographic hash) — and any actual filename collision is caught
    explicitly below rather than silently accepted.
    """
    digest = hashlib.sha1(csv_path.name.encode('utf-8')).hexdigest()
    return int(digest[:15], 16)  # 60 bits — fits safely in an int64 group name

def ingest_csv_dir(
    csv_dir: Path,
    output_h5: Path,
    min_segment_length: int = 20,
    verbose: bool = True,
) -> None:
    """
    Ingest all CSVs in csv_dir into a single HDF5 file.

    Also writes segment_info.json alongside the HDF5 for the Dataset index map.
    """
    csv_files = sorted(csv_dir.glob('*.csv'))
    if not csv_files:
        raise FileNotFoundError(f'No .csv files found in {csv_dir}')
    if verbose:
        print(f'Found {len(csv_files)} CSV files')

    seg_info: list[dict] = []
    id_to_filename: dict[int, str] = {}
    total_segments = 0
    total_frames = 0

    with h5py.File(output_h5, 'w') as h5:
        for i, csv_path in enumerate(csv_files):
            if verbose and (i + 1) % 20 == 0:
                print(f'  replay {i + 1}/{len(csv_files)}')

            # Validate CSV columns
            df = pd.read_csv(csv_path)
            missing = REQUIRED_CSV_COLS - set(df.columns)
            if missing:
                if verbose:
                    print(f'  SKIP {csv_path.name}: missing columns {sorted(missing)}')
                continue

            replay_id = _stable_replay_id(csv_path)
            if replay_id in id_to_filename:
                raise ValueError(
                    f"Hash collision deriving replay_id for '{csv_path.name}' "
                    f"— collides with already-ingested '{id_to_filename[replay_id]}'. "
                    f"This should be astronomically unlikely; check for "
                    f"duplicate or renamed files in {csv_dir}."
                )
            id_to_filename[replay_id] = csv_path.name

            cuts = detect_discontinuities(df)
            cut_indices = np.where(cuts)[0]

            replay_grp = h5.create_group(f'replay_{replay_id}')
            seg_count = 0

            for j, start in enumerate(cut_indices):
                end = cut_indices[j + 1] if j + 1 < len(cut_indices) else len(df)
                length = end - start

                if length < min_segment_length:
                    continue

                data = extract_segment(df, start, end)
                seg_grp = replay_grp.create_group(f'segment_{seg_count}')
                _write_segment(seg_grp, data)

                seg_info.append({
                    'replay_id': replay_id,
                    'segment_id': seg_count,
                    'length': length,
                })
                seg_count += 1
                total_frames += length

            total_segments += seg_count

        # Write segment info JSON (for Dataset index map)
        seg_info_path = output_h5.parent / "segment_info.json"
        with open(seg_info_path, "w") as f:
            json.dump([{k: int(v) for k, v in x.items()} for x in seg_info], f, indent=2)

        # Human-readable id → filename mapping, purely for debugging/
        # traceability (replay_ids themselves are now opaque hashes).
        id_map_path = output_h5.parent / 'replay_id_map.json'
        with open(id_map_path, "w") as f:
            json.dump(
                {str(rid): name for rid, name in id_to_filename.items()}, f, indent=2
            )

    if verbose:
        print(f'\nDone: {total_segments} segments, {total_frames} frames')
        print(f'  HDF5 → {output_h5}')
        print(f'  info → {output_h5.parent / "segment_info.json"}')
        print(f'  id map → {output_h5.parent / "replay_id_map.json"}')

# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Layer A: CSV → HDF5')
    parser.add_argument('--csv-dir', type=Path, required=True)
    parser.add_argument('--output-h5', type=Path, required=True)
    parser.add_argument('--min-segment-length', type=int, default=20)
    args = parser.parse_args()

    args.output_h5.parent.mkdir(parents=True, exist_ok=True)
    ingest_csv_dir(args.csv_dir, args.output_h5, args.min_segment_length)
