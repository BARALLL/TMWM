import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.interpolate import splprep, splev

def _hz_axes(up_axis: int):
    if up_axis == 1:      # Y-up (TM2020)
        return 0, 2
    if up_axis == 2:      # Z-up
        return 0, 1
    return 1, 2

def _horizontal_speed(vel: np.ndarray, up_axis: int) -> np.ndarray:
    ah, bh = _hz_axes(up_axis)
    return np.sqrt(vel[:, ah] ** 2 + vel[:, bh] ** 2)


def _box_smooth_1d(v: np.ndarray, passes: int = 2, periodic: bool = False) -> np.ndarray:
    out = np.asarray(v, dtype=np.float64).copy()
    for _ in range(max(int(passes), 0)):
        if periodic and out.size >= 2:
            pad = np.concatenate([out[-1:], out, out[:1]])
        else:
            pad = np.pad(out, 1, mode="edge")
        out = 0.25 * pad[:-2] + 0.5 * pad[1:-1] + 0.25 * pad[2:]
    return out


class TrajectoryCenterlineTorch(nn.Module):
    """
    GPU-ready centerline: batched *horizontal* projection + arc-length queries.

    Speed profile
    -------------
    If the CSV has linearVelocity_{X,Y,Z}, `v_target` is the recorded
    horizontal speed, lightly box-smoothed. No backward brake pass —
    the ghost already encoded "lift now". The μ/κ bicycle profile is
    only a fallback when velocity columns are missing.

    Projection
    ----------
    Distances and the along-segment parameter `t` are horizontal.
    `progress_trajectory` is a metre-windowed walk. Pass `s0` (live
    arc-length) every call; `t = 0` is then a local search, not a
    global nearest-node snap. Each later step may only pick a foot
    whose Δs lies in [-back, +max_fwd]. A hairpin therefore cannot
    reassign `s` to the exit.
    """

    _KEYS = (
        "centerline", "seg_A", "seg_B", "seg_AB", "seg_len_sq", "seg_len",
        "s_A", "s_B", "s_nodes", "tangent_nodes", "kappa_nodes",
        "speed_target_nodes", "speed_recorded_nodes",
    )

    _POS_COLS = ("worldPosition_X", "worldPosition_Y", "worldPosition_Z")
    _VEL_COLS = ("linearVelocity_X", "linearVelocity_Y", "linearVelocity_Z")

    def __init__(
        self,
        csv_paths,
        num_interp_points=1000,
        is_loop=False,
        spline_s=0.5,
        dtype=torch.float32,
        device="cpu",
        up_axis=1,
        friction_mu: float = 1.2,
        gravity: float = 9.81,
        max_speed_mps: float | None = None,
        max_brake_decel: float = 12.0,
        speed_profile_smoothing_passes: int = 3,
        recorded_speed_smooth_passes: int = 2,
        kappa_smooth_passes: int = 3,
    ):
        super().__init__()
        if isinstance(csv_paths, (str, bytes)):
            raise TypeError(
                "csv_paths must be a list/tuple of paths, not a string "
                f"(got {csv_paths!r} — that would iterate characters)"
            )
        csv_paths = list(csv_paths)
        if not csv_paths:
            raise ValueError("csv_paths is required")

        self.is_loop = bool(is_loop)
        self.num_points = int(num_interp_points)
        self.default_dtype = dtype
        self.default_device = torch.device(device)
        self.up_axis = int(up_axis)
        self.used_recorded_speed = False

        resampled_pts = []
        resampled_spd = []
        had_speed = True
        for path in csv_paths:
            pts, spd, has = self._load_csv(path)
            pts_r, spd_r = self._resample_by_arclength(
                pts, self.num_points, extras=spd.reshape(-1, 1),
            )
            resampled_pts.append(pts_r)
            resampled_spd.append(spd_r[:, 0])
            had_speed = had_speed and has

        mean_pts = np.mean(np.stack(resampled_pts, axis=0), axis=0)
        mean_spd = np.mean(np.stack(resampled_spd, axis=0), axis=0)

        tck, _ = splprep(
            [mean_pts[:, 0], mean_pts[:, 1], mean_pts[:, 2]],
            s=spline_s,
            per=self.is_loop,
        )
        # Periodic spline: splev(0) == splev(1). Drop the duplicate so
        # the closing segment is a real chord, not a point.
        u_fine = np.linspace(0.0, 1.0, self.num_points, endpoint=not self.is_loop)
        self.num_points = int(u_fine.shape[0])
        centerline_np = np.array(splev(u_fine, tck)).T
        deriv1_np = np.array(splev(u_fine, tck, der=1)).T
        deriv2_np = np.array(splev(u_fine, tck, der=2)).T

        tangent_nodes_np = deriv1_np / (
            np.linalg.norm(deriv1_np, axis=1, keepdims=True) + 1e-8
        )

        ah, bh = _hz_axes(self.up_axis)
        d1x, d1y = deriv1_np[:, ah], deriv1_np[:, bh]
        d2x, d2y = deriv2_np[:, ah], deriv2_np[:, bh]
        cross_h = d1x * d2y - d1y * d2x
        speed_h = np.sqrt(d1x * d1x + d1y * d1y)
        kappa_raw_np = cross_h / (np.power(speed_h, 3) + 1e-8)
        kappa_nodes_np = _box_smooth_1d(
            kappa_raw_np, kappa_smooth_passes, periodic=self.is_loop,
        )

        dists = np.linalg.norm(np.diff(centerline_np, axis=0), axis=1)
        s_np = np.insert(np.cumsum(dists), 0, 0.0)
        if self.is_loop:
            loop_dist = np.linalg.norm(centerline_np[0] - centerline_np[-1])
            s_np = np.append(s_np, s_np[-1] + loop_dist)
        self.total_length = float(s_np[-1])

        if self.is_loop:
            seg_A_np = centerline_np
            seg_B_np = np.roll(centerline_np, -1, axis=0)
        else:
            seg_A_np = centerline_np[:-1]
            seg_B_np = centerline_np[1:]

        s_A_np = s_np[:-1]
        s_B_np = s_np[1:]
        seg_AB_np = seg_B_np - seg_A_np
        seg_len_sq_np = (seg_AB_np ** 2).sum(axis=-1)
        seg_len_np = np.sqrt(np.maximum(seg_len_sq_np, 1e-12))
        self.median_seg_len = float(np.median(seg_len_np))
        s_nodes_np = s_np[: self.num_points]

        src_s = np.linspace(0.0, 1.0, mean_spd.shape[0])
        dst_s = s_nodes_np / max(self.total_length, 1e-8)
        recorded_np = np.interp(dst_s, src_s, mean_spd).astype(np.float64)
        recorded_np = _box_smooth_1d(
            recorded_np, recorded_speed_smooth_passes, periodic=self.is_loop,
        )
        recorded_np = np.clip(recorded_np, 0.0, None)

        if had_speed and np.nanmax(recorded_np) > 0.5:
            # Ghost speed already has late, hard TM braking in it.
            # A 12 m/s² backward pass would invent lift ~1 s early.
            v_profile_np = recorded_np.copy()
            self.used_recorded_speed = True
        else:
            v_profile_np = np.sqrt(
                friction_mu * gravity
                / np.clip(np.abs(kappa_raw_np), 1e-6, None)
            )
            self.used_recorded_speed = False
            n_pass = speed_profile_smoothing_passes if self.is_loop else 1
            for _ in range(max(int(n_pass), 1)):
                for i in range(len(v_profile_np) - 2, -1, -1):
                    ds = s_nodes_np[i + 1] - s_nodes_np[i]
                    v_profile_np[i] = min(
                        v_profile_np[i],
                        np.sqrt(max(v_profile_np[i + 1] ** 2 + 2.0 * max_brake_decel * ds, 0.0)),
                    )
                if self.is_loop:
                    ds_wrap = self.total_length - float(s_nodes_np[-1])
                    v_profile_np[-1] = min(
                        v_profile_np[-1],
                        np.sqrt(max(v_profile_np[0] ** 2 + 2.0 * max_brake_decel * ds_wrap, 0.0)),
                    )

        if max_speed_mps is not None:
            v_profile_np = np.clip(v_profile_np, 0.0, float(max_speed_mps))
            recorded_np = np.clip(recorded_np, 0.0, float(max_speed_mps))

        self._np_source = dict(
            centerline=centerline_np,
            seg_A=seg_A_np,
            seg_B=seg_B_np,
            seg_AB=seg_AB_np,
            seg_len_sq=seg_len_sq_np,
            seg_len=seg_len_np,
            s_A=s_A_np,
            s_B=s_B_np,
            s_nodes=s_nodes_np,
            tangent_nodes=tangent_nodes_np,
            kappa_nodes=kappa_nodes_np,
            speed_target_nodes=v_profile_np,
            speed_recorded_nodes=recorded_np,
        )
        self._aux_cache = {}

        default_tensors = self._build_tensors(self.default_device, self.default_dtype)
        for name, t in default_tensors.items():
            self.register_buffer(name, t)

    def _load_csv(self, path):
        df = pd.read_csv(path)
        missing = [c for c in self._POS_COLS if c not in df.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}")
        pts = df.loc[:, list(self._POS_COLS)].to_numpy(dtype=np.float64)
        if all(c in df.columns for c in self._VEL_COLS):
            vel = df.loc[:, list(self._VEL_COLS)].to_numpy(dtype=np.float64)
            spd = _horizontal_speed(vel, self.up_axis)
            return pts, spd, True
        return pts, np.zeros(len(pts), dtype=np.float64), False

    def _build_tensors(self, device, dtype):
        src = self._np_source
        return {
            name: torch.tensor(src[name], dtype=dtype, device=device)
            for name in self._KEYS
        }

    def _resolve_tensors(self, device, dtype):
        if device == self.centerline.device and dtype == self.centerline.dtype:
            return {name: getattr(self, name) for name in self._KEYS}
        key = (device, dtype)
        cached = self._aux_cache.get(key)
        if cached is None:
            cached = self._build_tensors(device, dtype)
            self._aux_cache[key] = cached
        return cached

    def warm_cache(self, device=None, dtype=None):
        device = torch.device(device) if device is not None else self.centerline.device
        dtype = dtype if dtype is not None else self.centerline.dtype
        self._resolve_tensors(device, dtype)
        return self

    def _resample_by_arclength(self, pts, num_samples, extras=None):
        mask = np.ones(len(pts), dtype=bool)
        mask[1:] = np.linalg.norm(np.diff(pts, axis=0), axis=1) > 1e-5
        pts = pts[mask]
        if extras is not None:
            extras = np.asarray(extras)[mask]
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        s = np.insert(np.cumsum(dists), 0, 0.0)
        total = float(s[-1]) if len(s) else 0.0
        if total <= 1e-8:
            raise ValueError("trajectory has zero length after de-dup")
        s_uniform = np.linspace(0.0, total, num_samples)
        pts_r = np.column_stack([
            np.interp(s_uniform, s, pts[:, i]) for i in range(3)
        ])
        if extras is None:
            return pts_r
        if extras.ndim == 1:
            extras = extras.reshape(-1, 1)
        extras_r = np.column_stack([
            np.interp(s_uniform, s, extras[:, i]) for i in range(extras.shape[1])
        ])
        return pts_r, extras_r

    def _wrap_s(self, s: torch.Tensor) -> torch.Tensor:
        L = self.total_length
        if self.is_loop:
            return torch.remainder(s, L)
        return s.clamp(0.0, max(L - 1e-4, 0.0))

    def signed_ds(self, s_to: torch.Tensor, s_from: torch.Tensor) -> torch.Tensor:
        """Forward-positive Δs, unwrapped into (-L/2, L/2] on a loop."""
        ds = s_to - s_from
        if self.is_loop:
            L = self.total_length
            half = 0.5 * L
            ds = torch.remainder(ds + half, L) - half
        return ds

    def _node_interp(self, s_flat: torch.Tensor, s_nodes: torch.Tensor,
                     values: torch.Tensor) -> torch.Tensor:
        if self.is_loop:
            s_ext = torch.cat([s_nodes, s_nodes.new_tensor([self.total_length])])
            values_ext = torch.cat([values, values[:1]], dim=0)
        else:
            s_ext = s_nodes
            values_ext = values

        idx1 = torch.searchsorted(
            s_ext, s_flat.contiguous(), right=True
        ).clamp(1, s_ext.numel() - 1)
        idx0 = idx1 - 1
        denom = (s_ext[idx1] - s_ext[idx0]).clamp_min(1e-8)
        w = (s_flat - s_ext[idx0]) / denom
        v0, v1 = values_ext[idx0], values_ext[idx1]
        if values.ndim == 1:
            return v0 * (1.0 - w) + v1 * w
        return v0 * (1.0 - w.unsqueeze(-1)) + v1 * w.unsqueeze(-1)

    def _seg_idx_at_s(self, s: torch.Tensor, tensors) -> torch.Tensor:
        s_A = tensors["s_A"]
        s_wrapped = self._wrap_s(s)
        idx = torch.searchsorted(s_A, s_wrapped.contiguous(), right=True) - 1
        return idx.clamp(0, s_A.shape[0] - 1)

    def _candidate_indices(self, idx, back_m, fwd_m, tensors, device):
        NumSeg = tensors["seg_A"].shape[0]
        med = max(self.median_seg_len, 1e-3)
        back_n = max(2, int(np.ceil(back_m / med)))
        fwd_n = max(4, int(np.ceil(fwd_m / med)))
        offsets = torch.arange(-back_n, fwd_n + 1, device=device)
        cand = idx.unsqueeze(1) + offsets.unsqueeze(0)
        if self.is_loop:
            return cand % NumSeg
        return cand.clamp(0, NumSeg - 1)

    def _project_candidates(
        self,
        chunk,
        cand_indices,
        tensors,
        s_ref=None,
        max_back_ds=None,
        max_fwd_ds=None,
    ):
        """Horizontal projection of chunk (m, 3) onto candidate segments.

        If `s_ref` and the Δs caps are set, a candidate whose signed
        advance is outside [-max_back_ds, +max_fwd_ds] is illegal.
        All-illegal rows fall back to the smallest Δs violation so the
        walk crawls the polyline instead of snapping.
        """
        ah, bh = _hz_axes(self.up_axis)
        cand_A = tensors["seg_A"][cand_indices]
        cand_AB = tensors["seg_AB"][cand_indices]
        cand_s_A = tensors["s_A"][cand_indices]
        cand_s_B = tensors["s_B"][cand_indices]

        ab_x = cand_AB[..., ah]
        ab_y = cand_AB[..., bh]
        len_sq_h = ab_x * ab_x + ab_y * ab_y
        tiny = len_sq_h < 1e-10
        len_sq_h = len_sq_h.clamp_min(1e-10)

        vec_x = chunk[:, ah].unsqueeze(1) - cand_A[..., ah]
        vec_y = chunk[:, bh].unsqueeze(1) - cand_A[..., bh]
        t = (vec_x * ab_x + vec_y * ab_y) / len_sq_h
        t = torch.where(tiny, t.new_full(t.shape, 0.5), t)
        t_clamped = t.clamp(0.0, 1.0)

        proj_pts = cand_A + t_clamped.unsqueeze(-1) * cand_AB
        delta = chunk.unsqueeze(1) - proj_pts
        dists_sq = delta[..., ah] ** 2 + delta[..., bh] ** 2

        s = cand_s_A + t_clamped * (cand_s_B - cand_s_A)

        if s_ref is not None and (max_back_ds is not None or max_fwd_ds is not None):
            ds = self.signed_ds(s, s_ref.unsqueeze(-1))
            illegal = torch.zeros_like(dists_sq, dtype=torch.bool)
            viol = torch.zeros_like(ds)
            if max_back_ds is not None:
                back = ds.new_tensor(float(max_back_ds))
                illegal = illegal | (ds < -back)
                viol = torch.maximum(viol, -back - ds)
            if max_fwd_ds is not None:
                fwd = ds.new_tensor(float(max_fwd_ds))
                illegal = illegal | (ds > fwd)
                viol = torch.maximum(viol, ds - fwd)
            all_bad = illegal.all(dim=-1)
            # Legal rows: hide illegal candidates. All-illegal rows:
            # pick the smallest Δs violation (crawl, don't snap).
            cost = dists_sq.masked_fill(illegal, 1.0e12)
            cost = torch.where(all_bad.unsqueeze(-1), viol.square() + dists_sq * 1e-3, cost)
        else:
            cost = dists_sq

        best_idx = torch.argmin(cost, dim=-1)
        row = torch.arange(chunk.shape[0], device=chunk.device)
        best_t = t_clamped[row, best_idx]
        s_best = cand_s_A[row, best_idx] + best_t * (
            cand_s_B[row, best_idx] - cand_s_A[row, best_idx]
        )
        tangent = F.normalize(cand_AB[row, best_idx], dim=-1, eps=1e-8)
        return (
            proj_pts[row, best_idx],
            torch.sqrt(dists_sq[row, best_idx].clamp_min(0.0)),
            s_best,
            cand_indices[row, best_idx],
            tangent,
        )

    def progress_torch(
        self,
        pos: torch.Tensor,
        chunk_size: int = 4_096,
        return_tangent: bool = False,
        return_seg_idx: bool = False,
    ):
        orig_shape = pos.shape[:-1]
        flat_pos = pos.reshape(-1, 3)
        M = flat_pos.shape[0]
        device, dtype = flat_pos.device, flat_pos.dtype
        tensors = self._resolve_tensors(device, dtype)
        centerline = tensors["centerline"]
        NumSeg = tensors["seg_A"].shape[0]
        offsets = torch.arange(-2, 3, device=device)
        ah, bh = _hz_axes(self.up_axis)

        max_prod = 8_000_000
        chunk_size = int(min(chunk_size, max(256, max_prod // max(centerline.shape[0], 1))))

        out_proj = flat_pos.new_empty(M, 3)
        out_dist = flat_pos.new_empty(M)
        out_s = flat_pos.new_empty(M)
        out_idx = torch.empty(M, dtype=torch.long, device=device)
        out_tan = flat_pos.new_empty(M, 3)

        cl_h = centerline[:, [ah, bh]]
        for start in range(0, M, chunk_size):
            end = min(start + chunk_size, M)
            chunk = flat_pos[start:end]
            dists_to_nodes = torch.cdist(chunk[:, [ah, bh]], cl_h)
            nearest_node = torch.argmin(dists_to_nodes, dim=1)
            del dists_to_nodes
            cand_indices = nearest_node.unsqueeze(1) + offsets.unsqueeze(0)
            if self.is_loop:
                cand_indices = cand_indices % NumSeg
            else:
                cand_indices = cand_indices.clamp(0, NumSeg - 1)
            proj, dist, s, idx, tan = self._project_candidates(
                chunk, cand_indices, tensors,
            )
            out_proj[start:end] = proj
            out_dist[start:end] = dist
            out_s[start:end] = s
            out_idx[start:end] = idx
            out_tan[start:end] = tan

        out = [
            out_proj.view(*orig_shape, 3),
            out_dist.view(*orig_shape),
            out_s.view(*orig_shape),
        ]
        if return_tangent:
            out.append(out_tan.view(*orig_shape, 3))
        if return_seg_idx:
            out.append(out_idx.view(*orig_shape))
        return tuple(out)

    def progress_trajectory(
        self,
        pos_traj: torch.Tensor,
        back_window_m: float = 4.0,
        fwd_window_m: float = 10.0,
        s0: torch.Tensor | None = None,
        seg_idx0: torch.Tensor | None = None,
        max_back_ds: float | None = None,
        max_fwd_ds: float | None = None,
    ):
        """
        Project an (N, T, 3) plan with temporal consistency.

        Parameters
        ----------
        s0, seg_idx0
            Live arc-length / segment at t = 0. Strongly recommended.
            Without them t = 0 is a global nearest-node search and will
            snap to the wrong ribbon on a folding TM map.
        max_back_ds, max_fwd_ds
            Hard signed-Δs gate per step, in metres. Defaults to the
            window sizes. A hairpin exit 8 m away but 25 m of line
            ahead is then illegal, regardless of Euclidean distance.
        """
        if pos_traj.ndim != 3 or pos_traj.shape[-1] != 3:
            raise ValueError(f"expected (N, T, 3), got {tuple(pos_traj.shape)}")
        N, T, _ = pos_traj.shape
        device, dtype = pos_traj.device, pos_traj.dtype
        tensors = self._resolve_tensors(device, dtype)

        if max_back_ds is None:
            max_back_ds = float(back_window_m)
        if max_fwd_ds is None:
            max_fwd_ds = float(fwd_window_m)

        if s0 is not None:
            s0 = torch.as_tensor(s0, device=device, dtype=dtype).reshape(N)
            if seg_idx0 is None:
                idx = self._seg_idx_at_s(s0, tensors)
            else:
                idx = torch.as_tensor(seg_idx0, device=device, dtype=torch.long).reshape(N)
            cand = self._candidate_indices(idx, back_window_m, fwd_window_m, tensors, device)
            # t = 0 is a measurement, not a time step: localise near s0
            # with a symmetric window, do not apply the forward step cap.
            loc = max(float(back_window_m), float(fwd_window_m))
            proj, dist, s, idx, tan = self._project_candidates(
                pos_traj[:, 0], cand, tensors,
                s_ref=s0, max_back_ds=loc, max_fwd_ds=loc,
            )
        else:
            proj, dist, s, tan, idx = self.progress_torch(
                pos_traj[:, 0], return_tangent=True, return_seg_idx=True,
            )

        projs = [proj]
        dists = [dist]
        ss = [s]
        tans = [tan]
        idxs = [idx]

        for t in range(1, T):
            cand = self._candidate_indices(idx, back_window_m, fwd_window_m, tensors, device)
            proj, dist, s, idx, tan = self._project_candidates(
                pos_traj[:, t], cand, tensors,
                s_ref=ss[-1], max_back_ds=max_back_ds, max_fwd_ds=max_fwd_ds,
            )
            projs.append(proj)
            dists.append(dist)
            ss.append(s)
            tans.append(tan)
            idxs.append(idx)

        return (
            torch.stack(projs, dim=1),
            torch.stack(dists, dim=1),
            torch.stack(ss, dim=1),
            torch.stack(tans, dim=1),
            torch.stack(idxs, dim=1),
        )

    def query_at_s(self, s: torch.Tensor):
        """
        Path pose at arc-length `s` (any shape).

        Returns
        -------
        pos       (..., 3)   point on the polyline
        tangent   (..., 3)   unit spline tangent
        heading   (...)      horizontal yaw [rad]
        kappa     (...)      signed horizontal curvature [1/m] (smoothed)
        v_target  (...)      target speed [m/s]
        """
        orig = s.shape
        s_flat = self._wrap_s(s.reshape(-1))
        device, dtype = s_flat.device, s_flat.dtype
        tensors = self._resolve_tensors(device, dtype)
        s_A, s_B = tensors["s_A"], tensors["s_B"]
        NumSeg = s_A.shape[0]

        idx = torch.searchsorted(s_A, s_flat.contiguous(), right=True) - 1
        idx = idx.clamp(0, NumSeg - 1)
        denom = (s_B[idx] - s_A[idx]).clamp_min(1e-8)
        t = ((s_flat - s_A[idx]) / denom).clamp(0.0, 1.0)
        pos = tensors["seg_A"][idx] + t.unsqueeze(-1) * tensors["seg_AB"][idx]

        tangent = self._node_interp(s_flat, tensors["s_nodes"], tensors["tangent_nodes"])
        tangent = F.normalize(tangent, dim=-1, eps=1e-8)
        kappa = self._node_interp(s_flat, tensors["s_nodes"], tensors["kappa_nodes"])
        v_target = self._node_interp(
            s_flat, tensors["s_nodes"], tensors["speed_target_nodes"]
        )

        ah, bh = _hz_axes(self.up_axis)
        heading = torch.atan2(tangent[..., bh], tangent[..., ah])

        return (
            pos.view(*orig, 3),
            tangent.view(*orig, 3),
            heading.view(*orig),
            kappa.view(*orig),
            v_target.view(*orig),
        )

    def signed_lateral(
        self, pos: torch.Tensor, proj: torch.Tensor, tangent: torch.Tensor
    ) -> torch.Tensor:
        """Positive = left of path tangent, horizontal plane."""
        ah, bh = _hz_axes(self.up_axis)
        dx = pos[..., ah] - proj[..., ah]
        dy = pos[..., bh] - proj[..., bh]
        tx, ty = tangent[..., ah], tangent[..., bh]
        return -dx * ty + dy * tx

    def project(self, point):
        pt = torch.as_tensor(
            point, dtype=self.centerline.dtype, device=self.centerline.device
        ).reshape(1, 3)
        proj, dist, s = self.progress_torch(pt)
        return s.item(), dist.item(), proj.squeeze(0).detach().cpu().numpy()