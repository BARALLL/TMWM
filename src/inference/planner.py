"""
Planner ABC + CEM/MPPI implementations.

Planner doesn't know anything about GRUs/PointNets/CEM internals — it
only needs dynamics.step_dynamics and a RewardFunction. Swapping CEM
for MPPI or a learned policy means writing a new Planner subclass, not
touching BeliefTracker or main_loop.py.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from dynamics import step_dynamics, DynamicsState, NormTensors
from model import WorldModel
from track_context import TrackContextExtractor
from action_codec import GameAction, encode_batch
from reward import RacingLineReward, RewardCsvLogger, RewardFunction


def _rollout_and_score(
    model: WorldModel,
    norm: NormTensors,
    reward_fn: RewardFunction,
    dt: float,
    device: torch.device,
    belief: DynamicsState,
    track_ctx: TrackContextExtractor,
    steer_full: torch.Tensor,
    gas_full: torch.Tensor,
    brake_full: torch.Tensor,
) -> torch.Tensor:
    """
    Shared imagined-rollout + scoring step, used by every Planner
    implementation below. Only the sampling distribution and the
    update rule (CEM's elite-refit vs. MPPI's softmax-weighted average
    vs. SimpleMPPI's categorical version of the same) differ between
    planners — "simulate N candidate action sequences through the
    world model and score them" is identical, so it lives here once.
    """
    N, H = steer_full.shape
    raw_state = belief.raw_state.expand(N, -1).clone()
    pos       = belief.pos.expand(N, -1).clone()
    quat      = belief.quat.expand(N, -1).clone()
    hidden    = belief.hidden.expand(N, -1).clone()

    pos_traj  = torch.empty(N, H + 1, 3, device=device)
    quat_traj = torch.empty(N, H + 1, 4, device=device)
    pos_traj[:, 0] = pos
    quat_traj[:, 0] = quat

    for t in range(H):
        action_norm = encode_batch(steer_full[:, t], gas_full[:, t], brake_full[:, t])
        result = step_dynamics(model, track_ctx, raw_state, pos, quat, hidden,
                                action_norm, norm, dt)
        raw_state, pos, quat, hidden = result.raw_state, result.pos, result.quat, result.hidden
        pos_traj[:, t + 1] = pos
        quat_traj[:, t + 1] = quat

    # disp = (pos_traj[:, -1] - pos_traj[:, 0]).norm(dim=-1)
    # print("pos0", pos_traj[0, 0].tolist(), "posH", pos_traj[0, -1].tolist())
    # print("disp mean/std/min/max", disp.mean().item(), disp.std().item(), disp.min().item(), disp.max().item())

    return reward_fn.score(pos_traj, quat_traj, track_ctx, **belief._live.score_kwargs(batch=pos_traj.shape[0]))


class Planner(ABC):
    @abstractmethod
    def plan(self, belief: DynamicsState, track_ctx: TrackContextExtractor) -> GameAction:
        ...

    @abstractmethod
    def warm_start_from_last(self) -> None:
        """Called once per real tick after plan() returns, to shift the
        planner's internal prior by one step for next tick's receding-
        horizon replan. No-op for planners with no persistent prior."""
        ...


# ─────────────────────────────────────────────────────────────────────────
# CEM
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class PlannerConfig:
    horizon: int = 30               # model steps per imagined rollout
    action_repeat: int = 1          # hold each sampled decision for this many model steps
    n_samples: int = 512
    n_elite: int = 64
    n_iterations: int = 4
    steer_init_std: float = 0.5
    steer_min_std: float = 0.05
    gas_init_p: float = 0.7
    brake_init_p: float = 0.1
    disable_gas_brake_search: bool = False  # if True: gas forced 1, brake forced 0, neither is searched


class CEMPlanner(Planner):
    def __init__(
        self,
        model: WorldModel,
        norm: NormTensors,
        reward_fn: RewardFunction,
        dt: float,
        device: torch.device,
        config: PlannerConfig = PlannerConfig(),
    ):
        self.model = model
        self.norm = norm
        self.reward_fn = reward_fn
        self.dt = dt
        self.device = device
        self.cfg = config
        self.n_decisions = max(1, config.horizon // config.action_repeat)

        self._steer_mean = torch.zeros(self.n_decisions, device=device)
        self._steer_std  = torch.full((self.n_decisions,), config.steer_init_std, device=device)
        if config.disable_gas_brake_search:
            self._gas_p   = torch.ones(self.n_decisions, device=device)
            self._brake_p = torch.zeros(self.n_decisions, device=device)
        else:
            self._gas_p      = torch.full((self.n_decisions,), config.gas_init_p, device=device)
            self._brake_p    = torch.full((self.n_decisions,), config.brake_init_p, device=device)

    @torch.no_grad()
    def plan(self, belief: DynamicsState, track_ctx: TrackContextExtractor) -> GameAction:
        cfg = self.cfg
        N = cfg.n_samples

        steer_mean, steer_std = self._steer_mean.clone(), self._steer_std.clone()
        gas_p, brake_p = self._gas_p.clone(), self._brake_p.clone()

        best_steer_seq = steer_mean
        best_gas_seq = (gas_p > 0.5).float()
        best_brake_seq = (brake_p > 0.5).float()

        rewards = None

        for _ in range(cfg.n_iterations):
            steer_samples = torch.clamp(
                steer_mean.unsqueeze(0)
                + steer_std.unsqueeze(0) * torch.randn(N, self.n_decisions, device=self.device),
                -1.0, 1.0,
            )
            if cfg.disable_gas_brake_search:
                gas_samples   = torch.ones(N, self.n_decisions, device=self.device)
                brake_samples = torch.zeros(N, self.n_decisions, device=self.device)
            else:
                gas_samples   = (torch.rand(N, self.n_decisions, device=self.device) < gas_p.unsqueeze(0)).float()
                brake_samples = (torch.rand(N, self.n_decisions, device=self.device) < brake_p.unsqueeze(0)).float()

            steer_full = steer_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]
            gas_full   = gas_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]
            brake_full = brake_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]

            rewards = _rollout_and_score(self.model, self.norm, self.reward_fn, self.dt, self.device,
                                          belief, track_ctx, steer_full, gas_full, brake_full)

            elite_idx = torch.topk(rewards, min(cfg.n_elite, N)).indices
            steer_mean = steer_samples[elite_idx].mean(dim=0)
            steer_std  = torch.clamp(steer_samples[elite_idx].std(dim=0), min=cfg.steer_min_std)
            if not cfg.disable_gas_brake_search:
                gas_p      = gas_samples[elite_idx].mean(dim=0).clamp(0.02, 0.98)
                brake_p    = brake_samples[elite_idx].mean(dim=0).clamp(0.02, 0.98)

            best_i = elite_idx[0]
            best_steer_seq = steer_samples[best_i]
            best_gas_seq = gas_samples[best_i]
            best_brake_seq = brake_samples[best_i]

        # print(rewards)

        self._steer_mean, self._steer_std = steer_mean, steer_std
        self._gas_p, self._brake_p = gas_p, brake_p

        return GameAction(
            steer=float(best_steer_seq[0].item()),
            gas=bool(best_gas_seq[0].item() > 0.5),
            brake=bool(best_brake_seq[0].item() > 0.5),
        )

    def warm_start_from_last(self) -> None:
        def shift(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x[1:], x[-1:].clone()])
        self._steer_mean = shift(self._steer_mean)
        self._steer_std  = torch.clamp(shift(self._steer_std), min=self.cfg.steer_min_std)
        self._steer_std[-1] = self.cfg.steer_init_std   # the new tail entry is an unoptimized guess
        self._gas_p   = shift(self._gas_p)
        self._brake_p = shift(self._brake_p)


# ─────────────────────────────────────────────────────────────────────────
# MPPI (continuous steer, Bernoulli gas/brake)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class MPPIConfig:
    horizon: int = 30
    action_repeat: int = 1
    n_samples: int = 512
    n_iterations: int = 1          # canonical MPPI does one importance-sampling
                                    # update per real tick; refinement then comes
                                    # from the receding-horizon warm start across
                                    # ticks, not from looping here. >1 is supported
                                    # as an offline-refinement knob if you want it.
    temperature: float = 1.0       # a.k.a. lambda; higher = softer/more uniform
                                    # weighting, lower = closer to greedy argmax
    steer_std: float = 0.4         # FIXED noise std (no CEM-style shrinking —
                                    # that's what separates MPPI from CEM here)
    gas_init_p: float = 0.7
    brake_init_p: float = 0.1
    disable_gas_brake_search: bool = False  # if True: gas forced 1, brake forced 0


class MPPIPlanner(Planner):
    """
    Model-Predictive Path Integral control.

    Same imagined-rollout machinery as CEMPlanner (dynamics.step_dynamics
    + a RewardFunction) but a different update rule: instead of CEM's
    truncate-to-elite-then-refit-Gaussian, every sampled trajectory
    contributes to the next mean in proportion to
    softmax(reward / temperature) — no sample is ever fully discarded.
    That's the whole point of MPPI vs. CEM: smoother, less noisy control
    (nothing gets hard-cut at a percentile threshold) at the cost of
    needing a reasonably-tuned temperature to avoid weight collapse
    (too cold -> behaves like argmax over one lucky sample) or a
    near-uniform average (too hot -> ignores the reward signal).

    Continuous steer is sampled as mean + fixed-std Gaussian noise.
    gas/brake are kept as [0, 1] "propensity" scalars and sampled
    Bernoulli each iteration; since the importance-weighted MPPI update
    is just an expectation estimate, it applies unchanged to the
    Bernoulli case (weighted average of the sampled 0/1 outcomes).
    """
    def __init__(
        self,
        model: WorldModel,
        norm: NormTensors,
        reward_fn: RewardFunction,
        dt: float,
        device: torch.device,
        config: MPPIConfig = MPPIConfig(),
    ):
        self.model = model
        self.norm = norm
        self.reward_fn = reward_fn
        self.dt = dt
        self.device = device
        self.cfg = config
        self.n_decisions = max(1, config.horizon // config.action_repeat)

        self._steer_mean = torch.zeros(self.n_decisions, device=device)
        if config.disable_gas_brake_search:
            self._gas_p   = torch.ones(self.n_decisions, device=device)
            self._brake_p = torch.zeros(self.n_decisions, device=device)
        else:
            self._gas_p   = torch.full((self.n_decisions,), config.gas_init_p, device=device)
            self._brake_p = torch.full((self.n_decisions,), config.brake_init_p, device=device)


    @torch.no_grad()
    def plan(self, belief: DynamicsState, track_ctx: TrackContextExtractor) -> GameAction:
        cfg = self.cfg
        N = cfg.n_samples

        steer_mean = self._steer_mean.clone()
        gas_p, brake_p = self._gas_p.clone(), self._brake_p.clone()

        for _ in range(cfg.n_iterations):
            steer_samples = torch.clamp(
                steer_mean.unsqueeze(0)
                + cfg.steer_std * torch.randn(N, self.n_decisions, device=self.device),
                -1.0, 1.0,
            )
            if cfg.disable_gas_brake_search:
                gas_samples   = torch.ones(N, self.n_decisions, device=self.device)
                brake_samples = torch.zeros(N, self.n_decisions, device=self.device)
            else:
                gas_samples   = (torch.rand(N, self.n_decisions, device=self.device) < gas_p.unsqueeze(0)).float()
                brake_samples = (torch.rand(N, self.n_decisions, device=self.device) < brake_p.unsqueeze(0)).float()

            steer_full = steer_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]
            gas_full   = gas_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]
            brake_full = brake_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]

            rewards = _rollout_and_score(self.model, self.norm, self.reward_fn, self.dt, self.device,
                                          belief, track_ctx, steer_full, gas_full, brake_full)

            # torch.softmax subtracts max(rewards) internally, so this is
            # numerically safe regardless of the reward function's scale.
            weights = torch.softmax(rewards / max(cfg.temperature, 1e-6), dim=0)  # (N,)

            steer_mean = (weights.unsqueeze(1) * steer_samples).sum(dim=0)
            if not cfg.disable_gas_brake_search:
                gas_p   = (weights.unsqueeze(1) * gas_samples).sum(dim=0).clamp(0.02, 0.98)
                brake_p = (weights.unsqueeze(1) * brake_samples).sum(dim=0).clamp(0.02, 0.98)

        self._steer_mean = steer_mean
        self._gas_p, self._brake_p = gas_p, brake_p

        return GameAction(
            steer=float(steer_mean[0].item()),
            gas=bool(gas_p[0].item() > 0.5),
            brake=bool(brake_p[0].item() > 0.5),
        )

    def warm_start_from_last(self) -> None:
        def shift(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x[1:], x[-1:].clone()])
        self._steer_mean = shift(self._steer_mean)
        self._gas_p   = shift(self._gas_p)
        self._brake_p = shift(self._brake_p)


# ─────────────────────────────────────────────────────────────────────────
# SimpleCEM (steer in {-1, 0, 1}, gas/brake in {0, 1})
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SimpleCEMConfig:
    horizon: int = 30
    action_repeat: int = 10
    n_samples: int = 512
    n_elite: int = 64
    n_iterations: int = 4
    steer_init_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)  # P(-1), P(0), P(+1)
    min_prob: float = 0.02          # floor so a category is never fully starved out
    gas_init_p: float = 0.7
    brake_init_p: float = 0.1
    disable_gas_brake_search: bool = False  # if True: gas forced 1, brake forced 0


class SimpleCEMPlanner(Planner):
    """
    CEM restricted to a tiny discrete action set: steer in
    {-1, 0, +1} (full-left / straight / full-right), gas/brake in
    {0, 1}. This is the discrete-action analogue of CEMPlanner in
    exactly the way SimpleMPPIPlanner is the discrete-action analogue
    of MPPIPlanner: same categorical steer distribution + Bernoulli
    gas/brake as SimpleMPPIPlanner, but refit via CEM's hard elite
    cutoff (keep the top n_elite samples, refit category frequencies
    to them) rather than a softmax-weighted average over every sample.
    """
    def __init__(
        self,
        model: WorldModel,
        norm: NormTensors,
        reward_fn: RewardFunction,
        dt: float,
        device: torch.device,
        config: SimpleCEMConfig = SimpleCEMConfig(),
    ):
        self.model = model
        self.norm = norm
        self.reward_fn = reward_fn
        self.dt = dt
        self.device = device
        self.cfg = config
        self.n_decisions = max(1, config.horizon // config.action_repeat)
        self._steer_values = torch.tensor([-1.0, 0.0, 1.0], device=device)

        init_probs = torch.tensor(config.steer_init_probs, dtype=torch.float32, device=device)
        init_probs = init_probs / init_probs.sum()
        self._steer_probs = init_probs.unsqueeze(0).repeat(self.n_decisions, 1)  # (n_decisions, 3)

        if config.disable_gas_brake_search:
            self._gas_p   = torch.ones(self.n_decisions, device=device)
            self._brake_p = torch.zeros(self.n_decisions, device=device)
        else:
            self._gas_p   = torch.full((self.n_decisions,), config.gas_init_p, device=device)
            self._brake_p = torch.full((self.n_decisions,), config.brake_init_p, device=device)

    @torch.no_grad()
    def plan(self, belief: DynamicsState, track_ctx: TrackContextExtractor) -> GameAction:
        cfg = self.cfg
        N = cfg.n_samples

        steer_probs = self._steer_probs.clone()   # (n_decisions, 3)
        gas_p, brake_p = self._gas_p.clone(), self._brake_p.clone()

        best_steer_seq = self._steer_values[torch.argmax(steer_probs, dim=1)]
        best_gas_seq = (gas_p > 0.5).float()
        best_brake_seq = (brake_p > 0.5).float()

        rewards = None

        for _ in range(cfg.n_iterations):
            idx = torch.multinomial(steer_probs, N, replacement=True).T   # (N, n_decisions), values in {0,1,2}
            steer_samples = self._steer_values[idx]                       # (N, n_decisions), values in {-1,0,1}

            if cfg.disable_gas_brake_search:
                gas_samples   = torch.ones(N, self.n_decisions, device=self.device)
                brake_samples = torch.zeros(N, self.n_decisions, device=self.device)
            else:
                gas_samples   = (torch.rand(N, self.n_decisions, device=self.device) < gas_p.unsqueeze(0)).float()
                brake_samples = (torch.rand(N, self.n_decisions, device=self.device) < brake_p.unsqueeze(0)).float()

            steer_full = steer_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]
            gas_full   = gas_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]
            brake_full = brake_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]

            rewards = _rollout_and_score(self.model, self.norm, self.reward_fn, self.dt, self.device,
                                          belief, track_ctx, steer_full, gas_full, brake_full)

            elite_idx = torch.topk(rewards, min(cfg.n_elite, N)).indices

            onehot = F.one_hot(idx[elite_idx], num_classes=3).float()     # (n_elite, n_decisions, 3)
            steer_probs = onehot.mean(dim=0)                              # elite category frequencies
            steer_probs = steer_probs.clamp(min=cfg.min_prob)
            steer_probs = steer_probs / steer_probs.sum(dim=1, keepdim=True)

            if not cfg.disable_gas_brake_search:
                gas_p   = gas_samples[elite_idx].mean(dim=0).clamp(0.02, 0.98)
                brake_p = brake_samples[elite_idx].mean(dim=0).clamp(0.02, 0.98)

            best_i = elite_idx[0]
            best_steer_seq = steer_samples[best_i]
            best_gas_seq = gas_samples[best_i]
            best_brake_seq = brake_samples[best_i]

        # print(rewards)

        self._steer_probs = steer_probs
        self._gas_p, self._brake_p = gas_p, brake_p

        return GameAction(
            steer=float(best_steer_seq[0].item()),
            gas=bool(best_gas_seq[0].item() > 0.5),
            brake=bool(best_brake_seq[0].item() > 0.5),
        )

    def warm_start_from_last(self) -> None:
        def shift(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x[1:], x[-1:].clone()])
        self._steer_probs = shift(self._steer_probs)
        self._steer_probs[-1] = 1.0 / 3.0   # new tail decision is an unoptimized guess -> uniform prior
        self._gas_p   = shift(self._gas_p)
        self._brake_p = shift(self._brake_p)


# ─────────────────────────────────────────────────────────────────────────
# SimpleMPPI (steer in {-1, 0, 1}, gas/brake in {0, 1})
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SimpleMPPIConfig:
    horizon: int = 30
    action_repeat: int = 5
    n_samples: int = 512
    n_iterations: int = 1
    temperature: float = 1.0
    steer_init_probs: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3)  # P(-1), P(0), P(+1)
    min_prob: float = 0.02          # floor so a category is never fully starved out
    gas_init_p: float = 0.7
    brake_init_p: float = 0.1
    disable_gas_brake_search: bool = True  # if True: gas forced 1, brake forced 0


class SimpleMPPIPlanner(Planner):
    """
    MPPI restricted to a tiny discrete action set: steer in
    {-1, 0, +1} (full-left / straight / full-right), gas/brake in
    {0, 1}. A bang-bang controller — much lower-variance search space
    than MPPIPlanner's continuous steer, useful as a cheap baseline or
    when fine steering control isn't expected to matter.

    Each decision step maintains a categorical distribution over the 3
    steer values (instead of a Gaussian mean). The softmax-weighted
    MPPI update is applied to those category probabilities exactly the
    way it's applied to Bernoulli gas/brake probabilities in
    MPPIPlanner: both are importance-weighted expectations of one-hot
    (or 0/1) samples under softmax(reward / temperature) weights — no
    hard elite cutoff, unlike CEM.
    """
    def __init__(
        self,
        model: WorldModel,
        norm: NormTensors,
        reward_fn: RewardFunction,
        dt: float,
        device: torch.device,
        config: SimpleMPPIConfig = SimpleMPPIConfig(),
    ):
        self.model = model
        self.norm = norm
        self.reward_fn = reward_fn
        self.dt = dt
        self.device = device
        self.cfg = config
        self.n_decisions = max(1, config.horizon // config.action_repeat)
        self._steer_values = torch.tensor([-1.0, 0.0, 1.0], device=device)

        init_probs = torch.tensor(config.steer_init_probs, dtype=torch.float32, device=device)
        init_probs = init_probs / init_probs.sum()
        self._steer_probs = init_probs.unsqueeze(0).repeat(self.n_decisions, 1)  # (n_decisions, 3)

        if config.disable_gas_brake_search:
            # self._gas_p   = torch.ones(self.n_decisions, device=device)
            self._gas_p   = torch.full((self.n_decisions,), config.gas_init_p, device=device)
            self._brake_p = torch.zeros(self.n_decisions, device=device)
        else:
            self._gas_p   = torch.full((self.n_decisions,), config.gas_init_p, device=device)
            self._brake_p = torch.full((self.n_decisions,), config.brake_init_p, device=device)

        self.reward_log = RewardCsvLogger(out_dir="logs/reward", run_name="map01")

    @torch.no_grad()
    def plan(self, belief: DynamicsState, track_ctx: TrackContextExtractor) -> GameAction:
        cfg = self.cfg
        N = cfg.n_samples

        steer_probs = self._steer_probs.clone()   # (n_decisions, 3)
        gas_p, brake_p = self._gas_p.clone(), self._brake_p.clone()

        for _ in range(cfg.n_iterations):
            # torch.multinomial on a 2D input samples independently per
            # row -> (n_decisions, N); transpose to match the (N, ...)
            # convention every other tensor here uses.
            idx = torch.multinomial(steer_probs, N, replacement=True).T   # (N, n_decisions), values in {0,1,2}
            steer_samples = self._steer_values[idx]                       # (N, n_decisions), values in {-1,0,1}

            if cfg.disable_gas_brake_search:
                # gas_samples   = torch.ones(N, self.n_decisions, device=self.device)
                gas_samples   = (torch.rand(N, self.n_decisions, device=self.device) < gas_p.unsqueeze(0)).float()
                brake_samples = torch.zeros(N, self.n_decisions, device=self.device)
            else:
                gas_samples   = (torch.rand(N, self.n_decisions, device=self.device) < gas_p.unsqueeze(0)).float()
                brake_samples = (torch.rand(N, self.n_decisions, device=self.device) < brake_p.unsqueeze(0)).float()

            steer_full = steer_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]
            gas_full   = gas_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]
            brake_full = brake_samples.repeat_interleave(cfg.action_repeat, dim=1)[:, :cfg.horizon]

            rewards = _rollout_and_score(self.model, self.norm, self.reward_fn, self.dt, self.device,
                                          belief, track_ctx, steer_full, gas_full, brake_full)
            weights = torch.softmax(rewards / max(cfg.temperature, 1e-6), dim=0)  # (N,)

            onehot = F.one_hot(idx, num_classes=3).float()               # (N, n_decisions, 3)
            steer_probs = torch.einsum('n,ndc->dc', weights, onehot)     # weighted category frequencies
            steer_probs = steer_probs.clamp(min=cfg.min_prob)
            steer_probs = steer_probs / steer_probs.sum(dim=1, keepdim=True)

            gas_p   = (weights.unsqueeze(1) * gas_samples).sum(dim=0).clamp(0.02, 0.98)
            if not cfg.disable_gas_brake_search:
                brake_p = (weights.unsqueeze(1) * brake_samples).sum(dim=0).clamp(0.02, 0.98)

        self._steer_probs = steer_probs
        self._gas_p, self._brake_p = gas_p, brake_p

        best_steer_idx = torch.argmax(steer_probs[0])

        ga = GameAction(
            steer=float(self._steer_values[best_steer_idx].item()),
            gas=bool(gas_p[0].item() > 0.5),
            brake=bool(brake_p[0].item() > 0.5),
        )

        if isinstance(self.reward_fn, RacingLineReward):
            keys, means, _, _ = RewardCsvLogger.weighted_terms(self.reward_fn, weights)
            self.reward_log.log(self.reward_fn, weights, ga)
            print(
                " | ".join(f"{k}: {v:.2f}" for k, v in zip(keys, means.tolist())),
                f"|| Sent: Steer: {ga.steer} | Gas: {ga.gas} | Brake: {ga.brake}",
            )

        if isinstance(self.reward_fn):
            terms_dict = self.reward_fn.last_terms
            keys_to_print = list(terms_dict.keys())
            terms_tensor = torch.stack([terms_dict[key] for key in keys_to_print], dim=1)
            weighted_means = (weights.unsqueeze(1) * terms_tensor).sum(dim=0)
            print(
                " | ".join(
                    [f"{k}: {v:.2f}" for k, v in zip(keys_to_print, weighted_means)]
                ),
                f"|| Sent: Steer: {ga.steer} | Gas: {ga.gas} | Brake: {ga.brake}",
            )

        return ga

    def warm_start_from_last(self) -> None:
        def shift(x: torch.Tensor) -> torch.Tensor:
            return torch.cat([x[1:], x[-1:].clone()])
        self._steer_probs = shift(self._steer_probs)
        self._steer_probs[-1] = 1.0 / 3.0   # new tail decision is an unoptimized guess -> uniform prior
        self._gas_p   = shift(self._gas_p)
        self._brake_p = shift(self._brake_p)