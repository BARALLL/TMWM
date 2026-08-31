"""
Track geometry query interface for the Trackmania world model.

Provides:
  - TrackContextExtractor: abstract base class defining the query contract
  - PointCloudExtractor: concrete implementation using pre-baked patches

The Dataset and training loop both use the same extractor so the model
sees identical geometry encoding at train time and inference time.

Patches are (x, y, z, material_id).  Only the xyz channels are rotated
into the car frame; the material channel passes through unchanged.
"""

from abc import ABC, abstractmethod

import h5py
import numpy as np
import torch
from scipy.spatial import KDTree

from quaternion_utils import quat_rotate_inv
from torch_quaternion_utils import torch_quat_rotate_inv


class TrackContextExtractor(ABC):
    """
    Abstract interface for "what does the track look like here?"

    Concrete implementations load precomputed data and query it at
    runtime given the car's pose.
    """

    @abstractmethod
    def query(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """
        Get local track geometry at each position.

        Args:
            pos:  (T, 3) world positions
            quat: (T, 4) unit quaternions [W, X, Y, Z]

        Returns:
            (T, N_POINTS, 4) track geometry in the car's local frame
                              [x, y, z, material_id]
        """
        ...

    @abstractmethod
    def query_torch(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        """
        Torch-native equivalent of query(), operating entirely on
        `pos`'s device — no CPU round-trip.

        Used by the training loop's per-step rollout (train.py), where
        pos/quat already live on the GPU and a scipy KDTree query would
        force a device sync every single rollout step (K times per
        batch). Nearest-centerline lookup uses brute-force squared
        distance + argmin (torch.cdist) instead of a spatial tree —
        centerlines have at most a few thousand points, so a dense
        (B, M) distance matrix is cheap on GPU, and this is an *exact*
        nearest-neighbor search (identical results to the KDTree), not
        an approximation.
        """
        ...

    @abstractmethod
    def progress_torch(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        For each query position, find the nearest centerline point and
        return (index, lateral_distance, arclength) — reusing the same
        nearest-centerline computation query_torch already does
        internally, exposed for the reward function / planner rather
        than only for geometry patches.

        Returns:
            idx:        (M,) long — nearest centerline index
            distance:   (M,) float — distance to that point (off-track proxy)
            arclength:  (M,) float — cumulative arc-length at that index
                        (progress signal)
        """
        ...

    @abstractmethod
    def to(self, device: torch.device) -> 'TrackContextExtractor':
        """
        Eagerly move any device-resident state (e.g. the centerline/
        patches lookup tables used by query_torch) to `device`, and
        return self.

        Intended to be called once, right after construction, by the
        caller (run_pipeline.py) — NOT left to lazy-on-first-
        query_torch-call placement. This matters for two reasons:
          1. Predictable performance: without this, the table upload
             cost is paid silently on the first rollout step of the
             first training batch, indistinguishable from a mysterious
             first-batch slowdown when profiling, instead of an
             accounted-for, visible startup cost.
          2. Fail-fast: if the geometry table doesn't fit on the target
             device, you find out immediately at startup, not partway
             through the first epoch.
        """
        ...

    @property
    @abstractmethod
    def is_empty(self) -> bool:
        """True if no track geometry data is available."""
        ...

    @property
    @abstractmethod
    def n_points(self) -> int:
        """Number of points per patch."""
        ...

    @property
    @abstractmethod
    def num_materials(self) -> int:
        """Number of distinct material types."""
        ...

    @property
    @abstractmethod
    def material_registry_hash(self) -> str:
        """Fingerprint of the material registry this geometry was built
        against — empty string if unknown (e.g. no geometry loaded, or
        a geo file predating this field)."""
        ...

class PointCloudExtractor(TrackContextExtractor):
    """
    Query pre-baked point-cloud patches from track_geo.h5.

    Workflow:
      1. Find the nearest centerline point to the car's position (KDTree).
      2. Fetch the corresponding pre-sampled point cloud.
      3. Translate xyz to car-relative coordinates.
      4. Rotate xyz into the car's body frame.
      5. Scale xyz by 1/patch_radius so the model sees roughly unit-scale
         geometry regardless of the `radius` used at preprocessing time —
         consistent with `state`/`action` inputs, which are all z-scored
         or min-max'd to O(1) scale (see features.py). Without this, the
         track branch would be the only model input on a raw physical
         scale (±radius metres), and that scale would silently change
         any time the `--radius` pipeline flag changes, shifting the
         conditioning of TrackPointEncoder's first layer between runs
         with no corresponding change to any model-visible config.
         Real sampled points can slightly exceed magnitude 1 after
         scaling (candidate triangles are accepted up to
         radius + per-triangle bounding radius — see
         generate_patches_for_centerline), so this is "roughly unit
         scale," not a hard clamp.
      6. Concatenate material_id back onto the rotated/scaled xyz.

    The material channel is NOT rotated or scaled — it's a categorical ID.
    """

    def __init__(self, geo_h5_path: str):
        """
        Args:
            geo_h5_path: path to track_geo.h5 produced by preprocess_geometry.py
        """
        with h5py.File(geo_h5_path, 'r') as h5:
            self._centerline = h5['centerline_pos'][:]    # (M, 3)
            self._patches    = h5['patches'][:]            # (M, N_PTS, 4)
            self._num_materials = int(h5.attrs.get('num_materials', 1))
            self._patch_radius = float(h5.attrs.get('patch_radius', 15.0))
            self._material_registry_hash = str(h5.attrs.get('material_registry_hash', ''))

            if self._patch_radius <= 0:
                raise ValueError(
                    f"track_geo.h5 has non-positive patch_radius="
                    f"{self._patch_radius}; cannot normalize track context."
                )

        if len(self._centerline) > 0:
            self._tree = KDTree(self._centerline)
            self._empty = False
            diffs = np.linalg.norm(np.diff(self._centerline, axis=0), axis=1)
            self._arclength = np.concatenate([[0.0], np.cumsum(diffs)]).astype(np.float32)
        else:
            self._tree = None
            self._empty = True
            self._arclength = np.zeros(0, dtype=np.float32)

        # Lazily-populated torch mirror of _centerline/_patches, cached
        # per device so repeated query_torch calls on the same device
        # (the common case: one device for an entire training run)
        # don't re-upload the tables every rollout step.
        self._centerline_t: torch.Tensor | None = None
        self._patches_t: torch.Tensor | None = None
        self._centerline_sq_t: torch.Tensor | None = None
        self._torch_device: torch.device | None = None

    def query(self, pos: np.ndarray, quat: np.ndarray) -> np.ndarray:
        """
        Get point-cloud patches in the car's local frame.

        Args:
            pos:  (T, 3) world positions
            quat: (T, 4) unit quaternions [W, X, Y, Z]

        Returns:
            (T, N_PTS, 4) — [x, y, z, material_id] in car body frame
        """
        T = len(pos)
        n_pts = self._patches.shape[1]

        if self._empty:
            return np.zeros((T, n_pts, 4), dtype=np.float32)

        # Find nearest centerline point for each car position
        _, indices = self._tree.query(pos)  # (T,)

        # Fetch world-frame patches (M, N_PTS, 4)
        patches_world = self._patches[indices]

        # Separate xyz and material channels
        xyz_world = patches_world[..., :3]           # (T, N_PTS, 3)
        mat_ids   = patches_world[..., 3:]           # (T, N_PTS, 1)

        # Transform xyz to car-local frame:
        #   1. Translate: subtract car position
        #   2. Rotate:    apply inverse car orientation
        xyz_rel = xyz_world - pos[:, np.newaxis, :]   # (T, N_PTS, 3)
        xyz_local = quat_rotate_inv(xyz_rel, quat)    # (T, N_PTS, 3)
        xyz_local = xyz_local / self._patch_radius   # <- normalize to ~unit scale

        # Recombine: rotated xyz + unchanged material ID
        return np.concatenate([xyz_local, mat_ids], axis=-1).astype(np.float32) # (T, N_PTS, 4)

    def _torch_tables(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._torch_device != device:
            self._centerline_t = torch.from_numpy(self._centerline).to(
                device=device, dtype=torch.float32)
            self._patches_t = torch.from_numpy(self._patches).to(
                device=device, dtype=torch.float32)
            self._centerline_sq_t = (self._centerline_t ** 2).sum(dim=-1)
            self._arclength_t = torch.from_numpy(self._arclength).to(device=device, dtype=torch.float32)
            self._torch_device = device
        return self._centerline_t, self._patches_t, self._centerline_sq_t

    def query_torch(self, pos: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
        n_pts = self._patches.shape[1]
        if self._empty:
            return torch.zeros((pos.shape[0], n_pts, 4), dtype=torch.float32, device=pos.device)

        with torch.no_grad():
            centerline_t, patches_t, centerline_sq_t = self._torch_tables(pos.device)

            # Exact nearest-centerline lookup via brute-force distance;
            # index selection is inherently non-differentiable regardless
            # of no_grad, but wrapping the whole block avoids building
            # unnecessary autograd graph nodes for the gather/rotate ops.
            dists = torch.cdist(pos.unsqueeze(0), centerline_t.unsqueeze(0)).squeeze(0)  # (B, M)
            indices = torch.argmin(dists, dim=-1)                                        # (B,)

            patches_world = patches_t[indices]      # (B, N_PTS, 4)
            xyz_world = patches_world[..., :3]
            mat_ids = patches_world[..., 3:]

            xyz_rel = xyz_world - pos.unsqueeze(1)                 # (B, N_PTS, 3)
            xyz_local = torch_quat_rotate_inv(xyz_rel, quat.unsqueeze(1))  # (B, 1, 4) broadcasts against (B, N_PTS, 3)
            xyz_local = xyz_local / self._patch_radius

            return torch.cat([xyz_local, mat_ids], dim=-1)

    def progress_torch(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._empty:
            z = torch.zeros(pos.shape[0], device=pos.device)
            return z.long(), z, z
        with torch.no_grad():
            centerline_t, _, _ = self._torch_tables(pos.device)
            dists = torch.cdist(pos.unsqueeze(0), centerline_t.unsqueeze(0)).squeeze(0)
            min_dist, idx = torch.min(dists, dim=-1)
            arclength = self._arclength_t[idx]
        return idx, min_dist, arclength

    def to(self, device: torch.device) -> 'PointCloudExtractor':
        self._torch_tables(device)   # populates & caches _centerline_t/_patches_t
        return self

    @property
    def is_empty(self) -> bool:
        return self._empty

    @property
    def n_points(self) -> int:
        return self._patches.shape[1] if not self._empty else 0

    @property
    def num_materials(self) -> int:
        return self._num_materials


    @property
    def material_registry_hash(self) -> str:
        return self._material_registry_hash


def load_extractor(geo_h5_path: str | None) -> TrackContextExtractor:
    """
    Factory: load a PointCloudExtractor, or return a no-op extractor
    if the path is None or the file doesn't exist.
    """
    if geo_h5_path is not None:
        from pathlib import Path
        if Path(geo_h5_path).exists():
            return PointCloudExtractor(geo_h5_path)
    return _NullExtractor()


class _NullExtractor(TrackContextExtractor):
    """No geometry available. Returns zeros."""

    def query(self, pos, quat):
        return np.zeros((len(pos), 128, 4), dtype=np.float32)

    def query_torch(self, pos, quat):
        return torch.zeros((pos.shape[0], 128, 4), dtype=torch.float32, device=pos.device)

    def progress_torch(self, pos):
        z = torch.zeros(pos.shape[0], device=pos.device)
        return z.long(), z, z

    def to(self, device: torch.device) -> '_NullExtractor':
        return self  # no device-resident state to move

    @property
    def is_empty(self):
        return True

    @property
    def n_points(self):
        return 128

    @property
    def num_materials(self):
        return 1

    @property
    def material_registry_hash(self) -> str:
        return ''