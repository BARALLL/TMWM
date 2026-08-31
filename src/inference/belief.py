"""
BeliefTracker: the model's persistent memory during live driving.

Design: raw_state/pos/quat are ALWAYS overwritten with real telemetry
every tick — never integrated from the model's own predictions, since
the game gives us exact ground truth for free. The GRU hidden state is
the only genuine carried belief, advanced once per tick via
step_dynamics using the REAL previous state and the REAL action that
was actually sent (i.e. exactly a teacher-forced, tf=1.0 training
step). Compounding rollout error therefore only ever happens inside a
planner's throwaway imagined branches — never in the state you're
reasoning from tick to tick.
"""
from __future__ import annotations

import torch

from dynamics import step_dynamics, DynamicsState, NormTensors
from model import WorldModel
from track_context import TrackContextExtractor
from state_builder import LiveStateBuilder
from action_codec import GameAction, encode


class BeliefTracker:
    def __init__(
        self,
        model: WorldModel,
        track_ctx: TrackContextExtractor,
        norm: NormTensors,
        dt: float,
        device: torch.device,
    ):
        self.model = model
        self.track_ctx = track_ctx
        self.norm = norm
        self.dt = dt
        self.device = device
        self._state_builder = LiveStateBuilder(dt=dt)

        self._hidden: torch.Tensor | None = None
        self._raw_state: torch.Tensor | None = None
        self._pos: torch.Tensor | None = None
        self._quat: torch.Tensor | None = None
        self._last_action_sent: torch.Tensor | None = None

    def reset(self, telemetry: dict) -> None:
        """Call at run/lap start and on every reported respawn."""
        self._state_builder.reset()
        state_phys = self._state_builder.push_and_build(telemetry)

        self._raw_state = torch.as_tensor(state_phys, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._pos  = torch.as_tensor(telemetry['pos'],  dtype=torch.float32, device=self.device).unsqueeze(0)
        self._quat = torch.as_tensor(telemetry['quat'], dtype=torch.float32, device=self.device).unsqueeze(0)
        self._hidden = self.model.init_hidden(1, self.device)
        self._last_action_sent = None

    def update(self, telemetry: dict, is_respawn: bool) -> None:
        """Call once per real tick with telemetry observed AFTER the
        last action was applied."""
        if self._hidden is None or is_respawn:
            self.reset(telemetry)
            return

        if self._last_action_sent is not None:
            with torch.no_grad():
                result = step_dynamics(
                    self.model, self.track_ctx,
                    self._raw_state, self._pos, self._quat, self._hidden,
                    self._last_action_sent, self.norm, self.dt,
                )
            self._hidden = result.hidden
            # raw_state/pos/quat from this call are discarded — real
            # telemetry below always wins (see module docstring).

        state_phys = self._state_builder.push_and_build(telemetry)
        self._raw_state = torch.as_tensor(state_phys, dtype=torch.float32, device=self.device).unsqueeze(0)
        self._pos  = torch.as_tensor(telemetry['pos'],  dtype=torch.float32, device=self.device).unsqueeze(0)
        self._quat = torch.as_tensor(telemetry['quat'], dtype=torch.float32, device=self.device).unsqueeze(0)

    def record_action_sent(self, action: GameAction) -> None:
        """Must be called with whatever action was actually sent to the
        game this tick — the NEXT update() call's hidden-state advance
        uses this, not the planner's full imagined sequence."""
        self._last_action_sent = encode(action, self.device).unsqueeze(0)

    def current(self) -> DynamicsState:
        assert self._hidden is not None, "BeliefTracker.reset() must be called first"
        return DynamicsState(self._raw_state, self._pos, self._quat, self._hidden)