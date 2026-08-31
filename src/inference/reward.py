"""
RewardFunction: scores an imagined rollout for the planner.

Kept separate from Planner deliberately — reward shaping ("what does
good driving mean") is a different design axis than search strategy,
and should be swappable independently of CEM vs MPPI vs a learned
planner.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import csv
from pathlib import Path

import numpy as np
import torch

from track_context import TrackContextExtractor

from trajectory_centerline import TrajectoryCenterlineTorch, _hz_axes

# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------

def wrap_angle(a: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(a), torch.cos(a))


def huber(x: torch.Tensor, delta: float) -> torch.Tensor:
    """Standard Huber on a signed or non-negative input. Piecewise C1."""
    absx = x.abs()
    quad = 0.5 * absx.square()
    lin = delta * (absx - 0.5 * delta)
    return torch.where(absx <= delta, quad, lin)


def quat_apply(quat: torch.Tensor, vec: torch.Tensor, layout: str = "xyzw") -> torch.Tensor:
    """Rotate `vec` by `quat`. `vec` is (3,) or broadcastable to quat[..., 3]."""
    if layout == "xyzw":
        x, y, z, w = quat.unbind(-1)
    elif layout == "wxyz":
        w, x, y, z = quat.unbind(-1)
    else:
        raise ValueError(f"unknown quat layout {layout!r}")
    if vec.ndim == 1:
        vec = vec.expand(quat.shape[:-1] + (3,))
    vx, vy, vz = vec.unbind(-1)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    rx = vx + w * tx + (y * tz - z * ty)
    ry = vy + w * ty + (z * tx - x * tz)
    rz = vz + w * tz + (x * ty - y * tx)
    return torch.stack((rx, ry, rz), dim=-1)


def yaw_from_forward(fwd: torch.Tensor, up_axis: int) -> torch.Tensor:
    ah, bh = _hz_axes(up_axis)
    return torch.atan2(fwd[..., bh], fwd[..., ah])


def _hz_axes(up_axis: int):
    if up_axis == 1:      # Y-up (TM2020)
        return 0, 2
    if up_axis == 2:      # Z-up
        return 0, 1
    return 1, 2


class LiveCenterlineState:
    """
    Stateful (s, seg, vel) of the real car.

    First observation: global nearest-node (start line is unambiguous).
    Every later tick: the same Δs-capped walk used on plans, so a
    hairpin or a parallel ribbon cannot reassign s.
    """

    def __init__(
        self,
        centerline: TrajectoryCenterlineTorch,
        dt: float,
        back_window_m: float = 4.0,
        s_step_vmax: float = 70.0,
        s_step_slack_m: float = 2.0,
        up_axis: int = 1,
    ):
        self.cl = centerline
        self.dt = float(dt)
        self.back_window_m = float(back_window_m)
        self.s_step_vmax = float(s_step_vmax)
        self.s_step_slack_m = float(s_step_slack_m)
        self.up_axis = int(up_axis)
        self.reset()

    def reset(self):
        self.s = None          # scalar float
        self.seg_idx = None    # int
        self.vel = None        # (3,) tensor, world frame
        self.pos_prev = None   # (3,) last real position

    def _max_step(self) -> float:
        return self.s_step_vmax * self.dt + self.s_step_slack_m

    @torch.no_grad()
    def update(
        self,
        pos,
        vel=None,
        *,
        quat=None,
    ) -> dict:
        """
        pos : (3,) current world position (game or WM decode of *current* state)
        vel : (3,) world linear velocity, same frame as pos. Prefer this.
              If omitted, finite-diff vs the previous real pose.
        """
        pos_t = torch.as_tensor(pos, dtype=torch.float32).reshape(1, 1, 3)
        device = self.cl.centerline.device
        pos_t = pos_t.to(device)

        if self.s is None:
            proj, dist, s, tan, idx = self.cl.progress_torch(
                pos_t[:, 0], return_tangent=True, return_seg_idx=True,
            )
        else:
            s0 = torch.tensor([self.s], device=device, dtype=pos_t.dtype)
            idx0 = torch.tensor([int(self.seg_idx)], device=device, dtype=torch.long)
            proj, dist, s, tan, idx = self.cl.progress_trajectory(
                pos_t,
                back_window_m=self.back_window_m,
                fwd_window_m=self._max_step(),
                s0=s0,
                seg_idx0=idx0,
                max_back_ds=self.back_window_m,
                max_fwd_ds=self._max_step(),
            )
            proj, dist, s, tan, idx = proj[:, 0], dist[:, 0], s[:, 0], tan[:, 0], idx[:, 0]

        self.s = float(s.reshape(-1)[0].item())
        self.seg_idx = int(idx.reshape(-1)[0].item())

        if vel is not None:
            self.vel = torch.as_tensor(vel, dtype=torch.float32, device=device).reshape(3)
        elif self.pos_prev is not None:
            self.vel = (pos_t.reshape(3) - self.pos_prev) / self.dt
        else:
            self.vel = pos_t.new_zeros(3)
        self.pos_prev = pos_t.reshape(3).clone()

        lat = self.cl.signed_lateral(pos_t.reshape(3), proj.reshape(3), tan.reshape(3))
        return {
            "s": self.s,
            "seg_idx": self.seg_idx,
            "vel": self.vel,
            "dist": float(dist.reshape(-1)[0].item()),
            "lat": float(lat.item()),
        }

    def score_kwargs(self, batch: int) -> dict:
        """Broadcast the live state to the CEM batch."""
        if self.s is None:
            raise RuntimeError("LiveCenterlineState.update() was never called")
        device = self.cl.centerline.device
        return dict(
            s0=torch.full((batch,), self.s, device=device, dtype=torch.float32),
            seg_idx0=torch.full((batch,), self.seg_idx, device=device, dtype=torch.long),
            vel0=self.vel.to(device).reshape(1, 3).expand(batch, 3).clone(),
        )

# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------

class RewardFunction(ABC):
    @abstractmethod
    def score(
        self,
        pos_traj: torch.Tensor,    # (N, H+1, 3)
        quat_traj: torch.Tensor,   # (N, H+1, 4)
        track_ctx: TrackContextExtractor,
    ) -> torch.Tensor:              # (N,) higher = better
        ...

_TERM_ORDER = (
    "progress",
    "cte",
    "heading",
    "lookahead",
    "pursuit",
    "cut",
    "speed",
    "boundary",
    "lost",
    "total",
)

# last_diag keys that are (N, T) [or (N, T, K) — those get a mean over K]
_DIAG_SCALAR_KEYS = (
    "s",
    "lat",
    "dist",
    "speed",
    "v_target",
    "overshoot",
    "yaw",
    "path_yaw",
    "e_psi",
    "kappa",
    "on_road",
    "hinge",
    "inside_m",
)



class RewardCsvLogger:
    """
    Two CSVs, both appended every control tick:

      terms.csv   one row per tick  — CEM-weighted mean of last_terms + sent action
      diag.csv    T rows per tick   — elite sample (argmax weight) per-step traces

    Plot `diag.csv` for speed vs v_target / lat / hinge / e_psi.
    Plot `terms.csv` for which cost is winning over a run.
    """

    def __init__(self, out_dir, run_name="reward"):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.terms_path = self.out_dir / f"{run_name}_terms.csv"
        self.diag_path = self.out_dir / f"{run_name}_diag.csv"
        self.tick = 0
        self._terms_header_written = self.terms_path.exists() and self.terms_path.stat().st_size > 0
        self._diag_header_written = self.diag_path.exists() and self.diag_path.stat().st_size > 0

    def _append(self, path, header, rows, header_written_attr):
        write_header = not getattr(self, header_written_attr)
        with path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=header)
            if write_header:
                w.writeheader()
                setattr(self, header_written_attr, True)
            w.writerows(rows)

    @staticmethod
    def weighted_terms(reward_fn, weights):
        terms = reward_fn.last_terms
        keys = [k for k in _TERM_ORDER if k in terms] or list(terms.keys())
        stacked = torch.stack([terms[k].reshape(-1) for k in keys], dim=1).float()
        w = weights.reshape(-1).to(device=stacked.device, dtype=stacked.dtype)
        w = w / w.sum().clamp_min(1e-8)
        means = (w.unsqueeze(1) * stacked).sum(dim=0)
        return keys, means, w, stacked

    def log(self, reward_fn, weights, ga, print_line=True):
        if not isinstance(reward_fn, RacingLineReward):
            return

        keys, means, w, stacked = self.weighted_terms(reward_fn, weights)
        if print_line:
            print(
                " | ".join(f"{k}: {v:.2f}" for k, v in zip(keys, means.tolist())),
                f"|| Sent: Steer: {ga.steer} | Gas: {ga.gas} | Brake: {ga.brake}",
            )

        term_row = {
            "tick": self.tick,
            "steer": float(ga.steer),
            "gas": float(ga.gas),
            "brake": float(ga.brake),
        }
        for k, v in zip(keys, means.tolist()):
            term_row[k] = float(v)
        self._append(
            self.terms_path,
            ["tick", "steer", "gas", "brake"] + list(keys),
            [term_row],
            "_terms_header_written",
        )

        diag = getattr(reward_fn, "last_diag", None) or {}
        if diag:
            elite = int(torch.argmax(w).item())
            # T from any (N, T) field
            T = next(
                v.shape[1]
                for v in diag.values()
                if isinstance(v, torch.Tensor) and v.ndim >= 2
            )
            header = ["tick", "t", "elite"] + list(_DIAG_SCALAR_KEYS) + ["alpha_abs"]
            rows = []
            for t in range(T):
                row = {"tick": self.tick, "t": t, "elite": elite}
                for k in _DIAG_SCALAR_KEYS:
                    v = diag.get(k)
                    if v is None:
                        row[k] = ""
                        continue
                    row[k] = float(v[elite, t].item())
                alpha = diag.get("alpha")
                if alpha is None:
                    row["alpha_abs"] = ""
                else:
                    # (N, T, K) → mean |α| over lookahead times
                    row["alpha_abs"] = float(alpha[elite, t].abs().mean().item())
                rows.append(row)
            self._append(self.diag_path, header, rows, "_diag_header_written")

        self.tick += 1


class RacingLineReward(RewardFunction):
    """
    Receding-horizon racing-line cost for CEM / MPPI.

    Score (higher is better), all running costs in metre-seconds so the
    personality does not flip when you change T or action_repeat:

        Σ_t  gated_and_capped Δs
        - Σ_t dt w_t (cte + pursuit + yaw_rate + cut + speed)
        - Σ_t dt · boundary_hinge
        - lost_flag · lost_penalty

    One preview law
    ---------------
    Short pure-pursuit (`T = 0.4 s`, L ≤ 20 m) is the entry term.
    `heading_penalty` defaults to 0: current-s heading fights pursuit
    on every curve. Weave is `|ψ̇ - κ v|`, which is free on a well
    tracked corner and expensive on a straight wiggle.
    `lookahead_penalty` stays 0 (early-apex).

    Projection
    ----------
    Pass `s0` (and `seg_idx0` if you have it) from the *live* car
    every call. Per-step Δs is capped at `s_step_vmax · dt + slack`,
    and progress credit uses the same cap — a hairpin cut cannot
    buy 20 m of arc-length.

    Road
    ----
    Constant half-widths are a tube around the ghost, not the track.
    Prefer `track_ctx.half_widths_at_s(s)` or per-node arrays. If
    `track_ctx.edge_distances(pos)` exists it is treated as leftover
    width from the car (positive = still on asphalt).
    """

    def __init__(
        self,
        csv_paths,
        dt: float,
        # --- road ribbon --------------------------------------------------
        road_half_width_left: float | np.ndarray = 5.0,
        road_half_width_right: float | np.ndarray = 5.0,
        boundary_margin_m: float = 0.6,
        boundary_penalty_per_m_s: float = 8.0,
        lost_distance_m: float = 12.0,
        lost_penalty: float = 50.0,
        # gate Δs to the ribbon only when half-widths are the real road
        gate_progress_to_road: bool = False,
        # --- CTE (Huber on signed lat) -----------------------------------
        cte_penalty: float = 2.0,
        cte_huber_delta: float = 2.0,
        # --- current-s heading: OFF. fights pursuit. ---------------------
        heading_penalty: float = 0.0,
        # --- one short pursuit (the entry term) --------------------------
        lookahead_times=(0.4,),
        lookahead_weights=(1.0,),
        lookahead_penalty: float = 0.0,
        pursuit_penalty: float = 4.0,
        lookahead_min_m: float = 4.0,
        lookahead_max_m: float = 20.0,
        # --- weave: yaw-rate tracking error, not heading-at-s ------------
        yaw_rate_penalty: float = 0.4,
        # --- inside-of-turn cut ------------------------------------------
        cut_penalty: float = 3.0,
        cut_slack_m: float = 1.0,
        cut_kappa_min: float = 0.02,
        # --- speed ceiling vs recorded v_target(s) -----------------------
        speed_overshoot_penalty: float = 0.04,
        # --- extra weight on the terminal state --------------------------
        terminal_scale: float = 3.0,
        # --- sequential projection ---------------------------------------
        back_window_m: float = 4.0,
        fwd_window_m: float = 10.0,
        sequential_projection: bool = True,
        s_step_vmax: float = 70.0,
        s_step_slack_m: float = 2.0,
        # --- frames ------------------------------------------------------
        up_axis: int = 1,
        quat_layout: str = "xyzw",
        forward_local=(0.0, 0.0, 1.0),
        is_loop: bool = False,
        # --- speed-profile params (forwarded; unused if CSV has speed) ---
        friction_mu: float = 1.2,
        max_speed_mps: float | None = None,
        max_brake_decel: float = 12.0,
        store_diag: bool = True,
    ):
        if dt <= 0.0:
            raise ValueError(f"dt must be the pos_traj sample spacing in seconds, got {dt}")
        if isinstance(csv_paths, (str, bytes)):
            raise TypeError("csv_paths must be a list of paths, not a string")
        if not csv_paths:
            raise ValueError("csv_paths is required")
        if len(lookahead_times) != len(lookahead_weights):
            raise ValueError("lookahead_times and lookahead_weights must have the same length")

        self.dt = float(dt)
        self.boundary_margin_m = float(boundary_margin_m)
        self.boundary_penalty_per_m_s = float(boundary_penalty_per_m_s)
        self.lost_distance_m = float(lost_distance_m)
        self.lost_penalty = float(lost_penalty)
        self.gate_progress_to_road = bool(gate_progress_to_road)
        self.cte_penalty = float(cte_penalty)
        self.cte_huber_delta = float(cte_huber_delta)
        self.heading_penalty = float(heading_penalty)
        self.lookahead_times = torch.tensor(list(lookahead_times), dtype=torch.float32)
        w = torch.tensor(list(lookahead_weights), dtype=torch.float32)
        self.lookahead_weights = w / w.sum().clamp_min(1e-8)
        self.lookahead_penalty = float(lookahead_penalty)
        self.pursuit_penalty = float(pursuit_penalty)
        self.lookahead_min_m = float(lookahead_min_m)
        self.lookahead_max_m = float(lookahead_max_m)
        self.yaw_rate_penalty = float(yaw_rate_penalty)
        self.cut_penalty = float(cut_penalty)
        self.cut_slack_m = float(cut_slack_m)
        self.cut_kappa_min = float(cut_kappa_min)
        self.speed_overshoot_penalty = float(speed_overshoot_penalty)
        self.terminal_scale = float(terminal_scale)
        self.back_window_m = float(back_window_m)
        self.fwd_window_m = float(fwd_window_m)
        self.sequential_projection = bool(sequential_projection)
        self.s_step_vmax = float(s_step_vmax)
        self.s_step_slack_m = float(s_step_slack_m)
        self.up_axis = int(up_axis)
        self.quat_layout = quat_layout
        self.forward_local = tuple(forward_local)
        self.store_diag = bool(store_diag)
        self.last_terms = {}
        self.last_diag = {}

        if self.heading_penalty > 0.0 and self.pursuit_penalty > 0.0:
            raise ValueError(
                "heading_penalty and pursuit_penalty both > 0: they fight on "
                "every curve. Leave heading_penalty=0 and use yaw_rate_penalty "
                "for weave, or drop pursuit and use heading alone."
            )

        self.trajectory_centerline = TrajectoryCenterlineTorch(
            csv_paths=csv_paths,
            num_interp_points=10_000,
            is_loop=is_loop,
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            up_axis=up_axis,
            friction_mu=friction_mu,
            max_speed_mps=max_speed_mps,
            max_brake_decel=max_brake_decel,
        )
        if not self.trajectory_centerline.used_recorded_speed:
            raise RuntimeError(
                "CSV has no usable linearVelocity_{X,Y,Z}; refuse to fall "
                "back to μ/κ for a TM racing line. Pass a CSV with speed."
            )

        self._hw_left_nodes = None
        self._hw_right_nodes = None
        self._hw_left_scalar = None
        self._hw_right_scalar = None
        self._set_half_widths(road_half_width_left, road_half_width_right)

    def _set_half_widths(self, left, right):
        left_arr = np.atleast_1d(np.asarray(left, dtype=np.float64))
        right_arr = np.atleast_1d(np.asarray(right, dtype=np.float64))
        if np.min(left_arr) <= 0.0 or np.min(right_arr) <= 0.0:
            raise ValueError("road half-widths must be positive")
        P = self.trajectory_centerline.num_points
        if left_arr.size == 1 and right_arr.size == 1:
            self._hw_left_scalar = float(left_arr[0])
            self._hw_right_scalar = float(right_arr[0])
            return
        if left_arr.size != P or right_arr.size != P:
            raise ValueError(
                f"per-node half-widths must have length num_interp_points={P}, "
                f"got left={left_arr.size}, right={right_arr.size}"
            )
        dev = self.trajectory_centerline.centerline.device
        dt = self.trajectory_centerline.centerline.dtype
        self._hw_left_nodes = torch.tensor(left_arr, device=dev, dtype=dt)
        self._hw_right_nodes = torch.tensor(right_arr, device=dev, dtype=dt)

    def _max_step_m(self) -> float:
        return self.s_step_vmax * self.dt + self.s_step_slack_m

    def _yaw_from_quat(self, quat_traj: torch.Tensor):
        fwd_local = quat_traj.new_tensor(self.forward_local)
        fwd = quat_apply(quat_traj, fwd_local, layout=self.quat_layout)
        return yaw_from_forward(fwd, self.up_axis), fwd

    def _speed_and_yaw(self, pos_traj: torch.Tensor, vel0: torch.Tensor | None):
        """Horizontal speed [m/s] and finite-diff heading. Endpoints one-sided."""
        N, T, _ = pos_traj.shape
        ah, bh = _hz_axes(self.up_axis)
        vel = pos_traj.new_zeros(N, T, 3)
        if T >= 2:
            dpos = pos_traj[:, 1:] - pos_traj[:, :-1]
            step_vel = dpos / self.dt
            vel[:, 0] = step_vel[:, 0]
            vel[:, -1] = step_vel[:, -1]
            if T > 2:
                vel[:, 1:-1] = 0.5 * (step_vel[:, :-1] + step_vel[:, 1:])
        if vel0 is not None:
            vel[:, 0] = vel0.to(device=pos_traj.device, dtype=pos_traj.dtype).reshape(N, 3)
        speed = torch.sqrt(vel[..., ah] ** 2 + vel[..., bh] ** 2 + 1e-8)
        yaw = torch.atan2(vel[..., bh], vel[..., ah])
        return speed, yaw, vel

    def _time_weights(self, T, device, dtype):
        w = torch.ones(T, device=device, dtype=dtype)
        w[-1] = self.terminal_scale
        return w / w.mean().clamp_min(1e-8)  # mean(w)=1, terminal heavier

    def _integrate(self, per_t: torch.Tensor, tw: torch.Tensor) -> torch.Tensor:
        """Σ_t dt · w_t · per_t  →  (N,)."""
        return (tw * per_t * self.dt).sum(dim=1)

    def _half_widths_at_s(self, s: torch.Tensor, track_ctx):
        if track_ctx is not None:
            for name in ("half_widths_at_s", "road_half_widths", "query_half_widths"):
                fn = getattr(track_ctx, name, None)
                if callable(fn):
                    left, right = fn(s)
                    return left, right

        if self._hw_left_nodes is not None:
            cl = self.trajectory_centerline
            tensors = cl._resolve_tensors(s.device, s.dtype)
            left_n = self._hw_left_nodes.to(device=s.device, dtype=s.dtype)
            right_n = self._hw_right_nodes.to(device=s.device, dtype=s.dtype)
            flat = cl._wrap_s(s.reshape(-1))
            left = cl._node_interp(flat, tensors["s_nodes"], left_n).view_as(s)
            right = cl._node_interp(flat, tensors["s_nodes"], right_n).view_as(s)
            return left, right

        return (
            s.new_full(s.shape, self._hw_left_scalar),
            s.new_full(s.shape, self._hw_right_scalar),
        )

    def _ribbon(self, lat, dist, s, pos, track_ctx):
        """
        Returns over_edge, on_road, hinge.

        If track_ctx exposes leftover asphalt from the *car*
        (`edge_distances` / `leftover_width` / `distance_to_edges`),
        that is the real road and the constant tube is ignored.
        """
        if track_ctx is not None:
            for name in ("edge_distances", "leftover_width", "distance_to_edges"):
                fn = getattr(track_ctx, name, None)
                if callable(fn):
                    d_left, d_right = fn(pos)
                    # positive = still inside
                    over_edge = torch.maximum((-d_left).clamp(min=0.0), (-d_right).clamp(min=0.0))
                    on_road = (d_left > 0.0) & (d_right > 0.0)
                    hinge = torch.maximum(
                        (self.boundary_margin_m - d_left).clamp(min=0.0),
                        (self.boundary_margin_m - d_right).clamp(min=0.0),
                    )
                    return over_edge, on_road, hinge

        left_hw, right_hw = self._half_widths_at_s(s, track_ctx)
        over_left = (lat - left_hw).clamp(min=0.0)
        over_right = (-lat - right_hw).clamp(min=0.0)
        over_edge = torch.maximum(over_left, over_right)
        on_road = over_edge <= 0.0
        margin = self.boundary_margin_m
        hinge = torch.maximum(
            (lat - (left_hw - margin)).clamp(min=0.0),
            (-lat - (right_hw - margin)).clamp(min=0.0),
        )
        return over_edge, on_road, hinge

    def score(
        self,
        pos_traj: torch.Tensor,
        quat_traj: torch.Tensor | None = None,
        track_ctx=None,
        *,
        s0: torch.Tensor | None = None,
        seg_idx0: torch.Tensor | None = None,
        vel0: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, T, _ = pos_traj.shape
        cl = self.trajectory_centerline
        tw = self._time_weights(T, pos_traj.device, pos_traj.dtype)
        max_step = self._max_step_m()
        fwd_m = min(self.fwd_window_m, max_step)

        if self.sequential_projection:
            proj, dist, s, tan, seg_idx = cl.progress_trajectory(
                pos_traj,
                back_window_m=self.back_window_m,
                fwd_window_m=fwd_m,
                s0=s0,
                seg_idx0=seg_idx0,
                max_back_ds=self.back_window_m,
                max_fwd_ds=max_step,
            )
        else:
            proj, dist, s, tan, seg_idx = cl.progress_torch(
                pos_traj, return_tangent=True, return_seg_idx=True,
            )

        # Frenet frame from the *projection* tangent so CTE and heading
        # share a source. kappa / v_target / lookahead pos come from s.
        lat = cl.signed_lateral(pos_traj, proj, tan)
        path_yaw = yaw_from_forward(tan, self.up_axis)
        _, _, _, kappa, v_target = cl.query_at_s(s)

        over_edge, on_road, hinge = self._ribbon(lat, dist, s, pos_traj, track_ctx)

        if T >= 2:
            ds = cl.signed_ds(s[:, 1:], s[:, :-1])
            ds = ds.clamp(-self.back_window_m, max_step)
            if self.gate_progress_to_road:
                step_ok = on_road[:, :-1] & on_road[:, 1:]
            else:
                step_ok = torch.ones_like(ds, dtype=torch.bool)
            progress = (ds * step_ok.to(ds.dtype)).sum(dim=1)
        else:
            ds = pos_traj.new_zeros(N, 0)
            progress = pos_traj.new_zeros(N)

        boundary_term = self.boundary_penalty_per_m_s * (hinge * self.dt).sum(dim=1)
        lost_term = (
            (dist > self.lost_distance_m).any(dim=1).to(pos_traj.dtype)
            * self.lost_penalty
        )

        speed, yaw_from_vel, _ = self._speed_and_yaw(pos_traj, vel0)
        if quat_traj is not None:
            yaw, fwd = self._yaw_from_quat(quat_traj)
        else:
            yaw = yaw_from_vel
            ah, bh = _hz_axes(self.up_axis)
            fwd = pos_traj.new_zeros(*pos_traj.shape)
            fwd[..., ah] = torch.cos(yaw)
            fwd[..., bh] = torch.sin(yaw)

        e_psi = wrap_angle(yaw - path_yaw)

        cte_term = self.cte_penalty * self._integrate(huber(lat, self.cte_huber_delta), tw)
        heading_term = pos_traj.new_zeros(N)
        if self.heading_penalty > 0.0:
            heading_term = self.heading_penalty * self._integrate(e_psi.abs(), tw)

        yaw_rate_term = pos_traj.new_zeros(N)
        yaw_rate = pos_traj.new_zeros(N, T)
        if self.yaw_rate_penalty > 0.0 and T >= 2:
            dyaw = wrap_angle(yaw[:, 1:] - yaw[:, :-1]) / self.dt
            yaw_rate[:, 0] = dyaw[:, 0]
            yaw_rate[:, -1] = dyaw[:, -1]
            if T > 2:
                yaw_rate[:, 1:-1] = 0.5 * (dyaw[:, :-1] + dyaw[:, 1:])
            e_r = yaw_rate - kappa * speed
            yaw_rate_term = self.yaw_rate_penalty * self._integrate(e_r.abs(), tw)

        turning = (kappa.abs() >= self.cut_kappa_min).to(lat.dtype)
        inside_m = lat * torch.sign(kappa) * turning
        cut_over = (inside_m - self.cut_slack_m).clamp(min=0.0)
        cut_term = self.cut_penalty * self._integrate(cut_over, tw)

        la_term = pos_traj.new_zeros(N)
        pu_term = pos_traj.new_zeros(N)
        alpha = None
        if self.lookahead_penalty > 0.0 or self.pursuit_penalty > 0.0:
            times = self.lookahead_times.to(device=pos_traj.device, dtype=pos_traj.dtype)
            la_w = self.lookahead_weights.to(device=pos_traj.device, dtype=pos_traj.dtype)
            L = (speed.unsqueeze(-1) * times).clamp(
                self.lookahead_min_m, self.lookahead_max_m
            )
            s_la = s.unsqueeze(-1) + L
            pos_la, _, yaw_la, _, _ = cl.query_at_s(s_la)

            if self.lookahead_penalty > 0.0:
                e_la = wrap_angle(yaw.unsqueeze(-1) - yaw_la)
                la_per_t = (e_la.abs() * la_w).sum(dim=-1)
                la_term = self.lookahead_penalty * self._integrate(la_per_t, tw)

            if self.pursuit_penalty > 0.0:
                ah, bh = _hz_axes(self.up_axis)
                vec = pos_la - pos_traj.unsqueeze(2)
                vx, vy = vec[..., ah], vec[..., bh]
                fx, fy = fwd[..., ah].unsqueeze(-1), fwd[..., bh].unsqueeze(-1)
                alpha = torch.atan2(fx * vy - fy * vx, fx * vx + fy * vy)
                pu_per_t = (alpha.abs() * la_w).sum(dim=-1)
                pu_term = self.pursuit_penalty * self._integrate(pu_per_t, tw)

        overshoot = (speed - v_target).clamp(min=0.0)
        speed_term = pos_traj.new_zeros(N)
        if self.speed_overshoot_penalty > 0.0:
            speed_term = self.speed_overshoot_penalty * self._integrate(overshoot.square(), tw)

        total = (
            progress
            - cte_term
            - heading_term
            - la_term
            - pu_term
            - yaw_rate_term
            - cut_term
            - speed_term
            - boundary_term
            - lost_term
        )

        self.last_terms = dict(
            progress=progress.detach(),
            cte=cte_term.detach(),
            heading=heading_term.detach(),
            lookahead=la_term.detach(),
            pursuit=pu_term.detach(),
            yaw_rate=yaw_rate_term.detach(),
            cut=cut_term.detach(),
            speed=speed_term.detach(),
            boundary=boundary_term.detach(),
            lost=lost_term.detach(),
            total=total.detach(),
        )
        if self.store_diag:
            self.last_diag = dict(
                s=s.detach(),
                ds=ds.detach() if T >= 2 else None,
                lat=lat.detach(),
                dist=dist.detach(),
                speed=speed.detach(),
                v_target=v_target.detach(),
                overshoot=overshoot.detach(),
                yaw=yaw.detach(),
                path_yaw=path_yaw.detach(),
                e_psi=e_psi.detach(),
                kappa=kappa.detach(),
                on_road=on_road.detach(),
                hinge=hinge.detach(),
                inside_m=inside_m.detach(),
                yaw_rate=yaw_rate.detach(),
                seg_idx=seg_idx.detach(),
                alpha=None if alpha is None else alpha.detach(),
            )
        return total