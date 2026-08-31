"""
Layer B: Track Geometry Preprocessing (offline, once per track).

Reads a collision .obj and the fastest replay CSV to produce track_geo.h5
containing pre-sampled point-cloud patches along the centerline.
Each patch point is (x, y, z, material_id)

Material ID handling:
    Game-native material IDs are sparse (e.g. 0, 4, 9, 65540, 458761).
    These are remapped to dense sequential compact IDs (0, 1, 2, …) via
    a material registry:

        Compact ID 0   → globally reserved "no geometry" sentinel.
                          Fallback patches (no mesh near centerline)
                          use this ID.  The model learns a dedicated
                          embedding for it.
        Compact ID 1..N → real materials, assigned in sorted order of
                          game-native ID.  Deterministic: same set of
                          native IDs always produces the same mapping.

    The registry is persisted as a JSON sidecar so that:
      - Multi-map preprocessing uses the same mapping (pass --registry)
      - Adding a new OBJ with new materials extends the registry
        without disturbing existing compact IDs (new IDs are appended)
      - The model's embedding table size is consistent across all maps

The Dataset's TrackContextExtractor queries this file at train time to
provide the model with local track geometry relative to the car's pose.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
from scipy.spatial import KDTree

from ingest_raw import detect_discontinuities

def hash_registry(native_to_compact: dict[int, int]) -> str:
    """
    Stable fingerprint of a material registry's content (not its file
    path or mtime) — travels with track_geo.h5 as an attribute so
    ModelBundle can verify, at inference load time, that the geometry
    file's compact material IDs mean the same surfaces the checkpoint
    was trained against.
    """
    canonical = json.dumps(
        {str(k): v for k, v in sorted(native_to_compact.items())}, sort_keys=True,
    )
    return hashlib.sha1(canonical.encode('utf-8')).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# Material registry — sparse game IDs → dense compact IDs
# ──────────────────────────────────────────────────────────────────────

NO_GEOMETRY_COMPACT: int = 0  # Always compact ID 0


def scan_obj_materials(obj_path: str) -> set[int]:
    """
    Scan an OBJ file for all game-native material IDs.

    Returns the set of unique IDs found in ``usemtl mat_<N>`` directives.
    """
    ids: set[int] = set()
    with open(obj_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('usemtl '):
                name = line.split(maxsplit=1)[1].strip()
                try:
                    ids.add(int(name.split('mat_')[1]))
                except (IndexError, ValueError):
                    ids.add(0)
    return ids


def build_material_registry(all_native_ids: set[int]) -> dict[int, int]:
    """
    Build a native→compact mapping from a set of game-native material IDs.

    Compact ID 0 is implicitly reserved for the "no geometry" sentinel.
    Real materials get compact IDs 1..N in sorted order of native ID.
    Deterministic: the same set of native IDs always produces the same
    mapping, regardless of discovery order.

    Returns:
        dict mapping native_id → compact_id
    """
    return {native: i + 1 for i, native in enumerate(sorted(all_native_ids))}


def load_or_extend_registry(
    native_ids: set[int],
    registry_path: Path | None = None,
) -> dict[int, int]:
    """
    Load an existing registry and extend it with any new material IDs,
    or build a fresh one.

    Existing mappings are never changed — new native IDs are appended
    with the next available compact ID.  This means:
      - Reprocessing the same OBJ is idempotent
      - Adding a new OBJ with new materials only appends
      - Previously-preprocessed tracks remain valid (their compact IDs
        haven't changed)

    Args:
        native_ids: set of game-native IDs found in the current OBJ
        registry_path: path to existing registry JSON (or None to build fresh)

    Returns:
        native_to_compact: dict mapping native_id → compact_id
    """
    if registry_path is not None and registry_path.exists():
        with open(registry_path) as f:
            saved = json.load(f)
        native_to_compact = {int(k): v for k, v in saved.items()}

        new_ids = native_ids - set(native_to_compact.keys())
        if new_ids:
            next_id = max(native_to_compact.values()) + 1
            for native in sorted(new_ids):
                native_to_compact[native] = next_id
                next_id += 1
            print(f'  Registry extended with {len(new_ids)} new materials: '
                  f'{sorted(new_ids)}')
    else:
        native_to_compact = build_material_registry(native_ids)

    return native_to_compact


def save_registry(native_to_compact: dict[int, int], path: Path) -> None:
    """Save material registry as JSON (keyed by native ID string for JSON compat)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump({str(k): v for k, v in sorted(native_to_compact.items())}, f, indent=2)


# ──────────────────────────────────────────────────────────────────────
# OBJ parsing — reads game-native material IDs, applies registry
# ──────────────────────────────────────────────────────────────────────

def load_obj(
    obj_path: str,
    native_to_compact: dict[int, int],
) -> tuple[np.ndarray, np.ndarray, dict[int, str]]:
    """
    Parse a Wavefront .obj file, extracting triangles and per-face material IDs.

    Game-native material IDs from ``usemtl mat_<N>`` directives are
    remapped to compact IDs via the provided registry.  Any native ID
    not in the registry is mapped to compact ID 0 (no-geometry sentinel)
    — this should never happen if the registry was built from the same
    OBJ, but the fallback prevents crashes from partially-mismatched
    registries.

    Handles fan triangulation for polygons with >3 vertices and OBJ's
    1-based face indexing.

    Args:
        obj_path: path to .obj file
        native_to_compact: mapping from game-native → compact material IDs

    Returns:
        triangles:     (F, 3, 3) float32 — vertex positions per face
        materials:     (F,) int32 — compact material ID per face
        material_names: dict mapping compact_id → usemtl name string
                        (for debugging / human-readable logging)
    """
    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    face_native_ids: list[int] = []
    current_native_id = 0

    with open(obj_path, 'r') as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            if line.startswith('v '):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])

            elif line.startswith('usemtl '):
                name = line.split(maxsplit=1)[1].strip()
                try:
                    current_native_id = int(name.split('mat_')[1])
                except (IndexError, ValueError):
                    current_native_id = 0

            elif line.startswith('f '):
                parts = line.split()[1:]
                idx = [int(p.split('/')[0]) - 1 for p in parts]
                if len(idx) == 3:
                    faces.append(idx)
                    face_native_ids.append(current_native_id)
                elif len(idx) > 3:
                    # Fan triangulation for quads/ngons
                    for j in range(1, len(idx) - 1):
                        faces.append([idx[0], idx[j], idx[j + 1]])
                        face_native_ids.append(current_native_id)

    if not vertices or not faces:
        raise ValueError(f"No geometry found in {obj_path}")

    verts_arr = np.array(vertices, dtype=np.float32)
    triangles = np.array([[verts_arr[i] for i in tri] for tri in faces], dtype=np.float32)

    # Remap native → compact
    compact_ids = np.array(
        [native_to_compact.get(nid, NO_GEOMETRY_COMPACT) for nid in face_native_ids],
        dtype=np.int32,
    )

    # Build human-readable name map (compact → native usemtl string)
    compact_to_native = {v: k for k, v in native_to_compact.items()}
    material_names = {
        cid: f'mat_{compact_to_native[cid]}'
        for cid in sorted(set(compact_ids))
        if cid != NO_GEOMETRY_COMPACT
    }

    print(f"  Loaded {len(triangles)} triangles, {len(material_names)} materials "
          f"(compact IDs {sorted(material_names.keys())})")

    return triangles, compact_ids, material_names


# ──────────────────────────────────────────────────────────────────────
# Point-cloud generation from mesh triangles
# ──────────────────────────────────────────────────────────────────────

def _sample_triangle_surfaces(
    vertices: np.ndarray,
    n_samples: int,
    rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """
    Area-weighted uniform sampling on triangle surfaces.

    Args:
        vertices: (n_tri, 3, 3) triangle vertex coordinates
        n_samples: number of points to sample

    Returns:
        points:    (n_samples, 3) points on triangle surfaces
        tri_idx:   (n_samples,) indices into the vertices array (0..n_tri-1)
    """
    n_tri = len(vertices)
    if n_tri == 0:
        return np.zeros((n_samples, 3), dtype=np.float32), np.zeros(n_samples, dtype=np.intp)

    # Triangle edge vectors and area
    e1 = vertices[:, 1] - vertices[:, 0]
    e2 = vertices[:, 2] - vertices[:, 0]
    areas = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
    areas = np.maximum(areas, 1e-10)

    # Area-weighted triangle selection
    weights = areas / areas.sum()
    chosen = rng.choice(n_tri, size=n_samples, p=weights)
    chosen_verts = vertices[chosen]

    # Random barycentric coordinates (proper uniform sampling)
    u = rng.uniform(0, 1, (n_samples, 1))
    v = rng.uniform(0, 1, (n_samples, 1))
    sqrt_u = np.sqrt(u)
    a = 1.0 - sqrt_u
    b = sqrt_u * (1.0 - v)
    c = sqrt_u * v

    points = (a * chosen_verts[:, 0]
              + b * chosen_verts[:, 1]
              + c * chosen_verts[:, 2])

    return points.astype(np.float32), chosen


def generate_patches_for_centerline(
    obj_path: str,
    centerline_pos: np.ndarray,
    native_to_compact: dict[int, int],
    radius: float = 15.0,
    n_points: int = 128,
    seed: int = 42,
) -> tuple[np.ndarray, dict[int, str]]:
    """
    For each centerline point, sample a point cloud from nearby mesh surfaces.

    Each point includes a material ID as the 4th channel, allowing the
    model to learn surface-dependent physics.

    Material IDs in the output patches are compact (dense, 0-indexed).
    When no mesh geometry falls within range, the patch is stored as
    all-zeros with material_id = NO_GEOMETRY_COMPACT (0).

    Proximity is tested against each triangle's own bounding sphere, not
    just its centroid: a triangle is included whenever its bounding
    sphere intersects the query ball of radius `radius` around the
    centerline point (query at radius + per-triangle bounding_radius,
    then filter exactly per-triangle). This avoids silently missing
    large triangles whose centroid happens to sit just outside `radius`
    but whose surface still comes within `radius` of the query point —
    which a naive centroid-only ball query would miss, causing patchy
    under-sampling near sharp mesh density transitions.

    Args:
        obj_path: path to .obj collision mesh
        centerline_pos: (M, 3) centerline positions in world coordinates
        native_to_compact: game-native → compact material ID mapping
        radius: search radius (metres) for nearby triangles
        n_points: number of points per patch
        seed: random seed for reproducibility

    Returns:
        patches: (M, n_points, 4) — [x, y, z, compact_material_id]
        material_names: dict mapping compact_id → usemtl name string
    """
    rng = np.random.default_rng(seed)

    triangles, materials, material_names = load_obj(obj_path, native_to_compact)

    # Triangle centroids for proximity query
    centroids = triangles.mean(axis=1)  # (F, 3)
    # Per-triangle bounding-sphere radius: max distance from centroid to
    # any of its 3 vertices. Any point on the triangle's surface is
    # within this radius of the centroid.
    tri_bounding_radius = np.max(
        np.linalg.norm(triangles - centroids[:, np.newaxis, :], axis=-1),
        axis=-1,
    )  # (F,)
    max_bounding_radius = float(tri_bounding_radius.max()) if len(tri_bounding_radius) else 0.0

    tree = KDTree(centroids)

    M = len(centerline_pos)
    # Initialize with sentinel.  Real patches overwrite; fallback keeps this.
    patches = np.zeros((M, n_points, 4), dtype=np.float32)
    patches[:, :, 3] = NO_GEOMETRY_COMPACT

    n_fallback = 0

    for i in range(M):
        c = centerline_pos[i]

        # Conservative candidate set: any triangle whose bounding sphere
        # COULD intersect the true query ball.
        candidate_idx = tree.query_ball_point(c, radius + max_bounding_radius)

        nearby_idx: np.ndarray
        if candidate_idx:
            candidate_idx = np.asarray(candidate_idx)
            dists = np.linalg.norm(centroids[candidate_idx] - c, axis=1)
            # Exact per-triangle test using its own bounding radius.
            keep = dists <= (radius + tri_bounding_radius[candidate_idx])
            nearby_idx = candidate_idx[keep]
        else:
            nearby_idx = np.array([], dtype=int)

        if len(nearby_idx) > 0:
            nearby_tris = triangles[nearby_idx]
            nearby_mats = materials[nearby_idx]

            # Sample points from nearby triangles (area-weighted).
            pts, tri_idx = _sample_triangle_surfaces(nearby_tris, n_points, rng)
            mat_ids = nearby_mats[tri_idx].astype(np.float32)

            patches[i, :, :3] = pts
            patches[i, :, 3] = mat_ids
        else:
            n_fallback += 1

    if n_fallback > 0:
        print(f'  Warning: {n_fallback}/{M} centerline points have no '
              f'mesh geometry within {radius}m (compact material {NO_GEOMETRY_COMPACT})')

    return patches, material_names


# ──────────────────────────────────────────────────────────────────────
# Centerline extraction
# ──────────────────────────────────────────────────────────────────────

def extract_centerline(
    replay_csv: Path,
    sample_interval: float = 1.0,
) -> np.ndarray:
    """
    Sample evenly-spaced points along the fastest replay trajectory.

    Discontinuities (respawns, position jumps) are detected and the
    trajectory is split so the arc-length computation doesn't bridge
    across teleports.

    Args:
        replay_csv: path to the fastest replay CSV
        sample_interval: distance between consecutive centerline points (metres)

    Returns:
        (M, 3) centerline positions
    """
    df = pd.read_csv(replay_csv)
    pos = np.stack([
        df['worldPosition_X'].values,
        df['worldPosition_Y'].values,
        df['worldPosition_Z'].values,
    ], axis=1).astype(np.float64)

    # Detect discontinuities so we don't bridge across respawns
    cuts = detect_discontinuities(df)
    cut_indices = np.where(cuts)[0].tolist()
    cut_indices.append(len(df))  # sentinel

    print(f"rows in replay csv: {len(df)}")
    print(f"num discontinuity cuts: {cuts.sum()} / {len(df)}")

    all_sampled = []

    for seg_i in range(len(cut_indices) - 1):
        start = cut_indices[seg_i]
        end = cut_indices[seg_i + 1]
        seg_len = end - start

        if seg_len < 2:
            continue

        seg_pos = pos[start:end]

        # Cumulative arc-length within this segment
        diffs = np.linalg.norm(np.diff(seg_pos, axis=0), axis=1)
        cumlen = np.concatenate([[0.0], np.cumsum(diffs)])
        total_len = cumlen[-1]

        if total_len < sample_interval:
            # Segment too short for even one sample — use its midpoint
            all_sampled.append(seg_pos[len(seg_pos) // 2])
            continue

        # Sample at regular arc-length intervals
        targets = np.arange(0, total_len, sample_interval)
        j = 0
        for t in targets:
            while j < len(cumlen) - 2 and cumlen[j + 1] < t:
                j += 1
            alpha = (t - cumlen[j]) / max(cumlen[j + 1] - cumlen[j], 1e-10)
            all_sampled.append(seg_pos[j] + alpha * (seg_pos[j + 1] - seg_pos[j]))

    return np.array(all_sampled, dtype=np.float32)


# ──────────────────────────────────────────────────────────────────────
# Main entry point
# ──────────────────────────────────────────────────────────────────────

def preprocess_geometry(
    obj_path: Path,
    fastest_replay_csv: Path,
    output_h5: Path,
    registry_path: Path | None = None,
    sample_interval: float = 1.0,
    radius: float = 15.0,
    n_points: int = 128,
    seed: int = 42,
    verbose: bool = True,
) -> None:
    """
    Full Layer B pipeline: centerline + material-aware patches.

    Args:
        obj_path: collision .obj mesh
        fastest_replay_csv: fastest replay CSV (for centerline)
        output_h5: output track_geo.h5
        registry_path: material registry JSON.  If provided and it exists,
            loaded and extended with any new materials from this OBJ.  If
            None, a fresh registry is built from this OBJ and saved
            alongside the H5.  For multi-map training, pre-compute a
            shared registry from all OBJs and pass it to every run.
        sample_interval: centerline sample spacing (metres)
        radius: patch search radius (metres)
        n_points: points per patch
        seed: random seed
        verbose: print progress
    """
    # ── Material registry ─────────────────────────────────────────────
    native_ids = scan_obj_materials(str(obj_path))
    native_to_compact = load_or_extend_registry(native_ids, registry_path)

    # Always save the (possibly extended) registry so it persists
    # regardless of whether the caller passed --registry.
    if registry_path is None:
        registry_path = output_h5.with_suffix('.registry.json')
    save_registry(native_to_compact, registry_path)
    if verbose:
        print(f'Material registry: {len(native_to_compact)} game materials '
              f'→ compact IDs 1..{len(native_to_compact)} '
              f'(0 = "no geometry" sentinel)')
        print(f'  Saved to {registry_path}')

    # ── Centerline ────────────────────────────────────────────────────
    if verbose:
        print('Extracting centerline...')
    centerline = extract_centerline(fastest_replay_csv, sample_interval)
    if verbose:
        print(f'  {len(centerline)} centerline points, '
              f'total length ≈ {len(centerline) * sample_interval:.0f} m')

    # ── Patches ───────────────────────────────────────────────────────
    if verbose:
        print('Generating material-aware patches from mesh...')
    patches, material_names = generate_patches_for_centerline(
        str(obj_path), centerline, native_to_compact,
        radius, n_points, seed,
    )

    # ── Save ──────────────────────────────────────────────────────────
    if verbose:
        print(f'Saving to {output_h5}...')
    output_h5.parent.mkdir(parents=True, exist_ok=True)
    num_materials = len(native_to_compact) + 1  # +1 for sentinel at 0
    registry_hash = hash_registry(native_to_compact)
    with h5py.File(output_h5, 'w') as h5:
        h5.create_dataset('centerline_pos', data=centerline)
        h5.create_dataset('patches', data=patches,
                          chunks=(min(1000, len(centerline)), n_points, 4))
        h5.attrs['patch_radius'] = radius
        h5.attrs['patch_points'] = n_points
        h5.attrs['num_materials'] = num_materials
        h5.attrs['material_registry_hash'] = registry_hash
        # Compact→native reverse map for debugging / introspection
        compact_to_native = {v: k for k, v in native_to_compact.items()}
        compact_to_native[NO_GEOMETRY_COMPACT] = -1  # sentinel convention
        h5.attrs['compact_to_native'] = json.dumps(
            {str(k): v for k, v in sorted(compact_to_native.items())}
        )

    if verbose:
        print(f'Done. {len(native_to_compact)} real materials → '
              f'compact IDs 1..{len(native_to_compact)}, '
              f'embedding table size = {num_materials}')


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Layer B: Track geometry → track_geo.h5')
    parser.add_argument('--mesh', type=Path, required=True, help='Collision .obj')
    parser.add_argument('--fastest-replay', type=Path, required=True, help='Fastest replay CSV')
    parser.add_argument('--output-h5', type=Path, required=True)
    parser.add_argument('--registry', type=Path, default=None,
                        help='Material registry JSON. Load existing (and extend '
                             'with any new materials from this OBJ) or create new. '
                             'For multi-map: pre-compute from all OBJs and pass to '
                             'every run so compact IDs are consistent.')
    parser.add_argument('--sample-interval', type=float, default=1.0)
    parser.add_argument('--radius', type=float, default=15.0)
    parser.add_argument('--n-points', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    preprocess_geometry(
        args.mesh, args.fastest_replay, args.output_h5,
        args.registry, args.sample_interval, args.radius,
        args.n_points, args.seed,
    )