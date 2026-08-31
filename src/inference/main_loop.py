"""
The live driving loop: telemetry -> belief -> plan -> action -> bridge.
"""
from __future__ import annotations

from model_bundle import ModelBundle
from belief import BeliefTracker
from planner import Planner, SimpleMPPIPlanner
from reward import RacingLineReward, LiveCenterlineState
from bridge_protocol import GameBridge
from live_bridge import TMWorldModelBridge


def run(
    bundle: ModelBundle,
    bridge: GameBridge,
    planner: Planner | None = None,
    max_ticks: int = 100_000,
) -> None:
    dt = bundle.run_config.sampling.effective_dt
    hold_ticks = bundle.run_config.sampling.subsample_factor

    belief = BeliefTracker(bundle.model, bundle.track_ctx, bundle.norm, dt, bundle.device)

    if planner is None:
        reward_fn = RacingLineReward(
            csv_paths=[

            ],
            quat_layout="wxyz",
            dt=dt,
            is_loop=False,
            yaw_rate_penalty=0.6,
            lookahead_penalty=0.5,
            # cte_penalty=2,

        )
        
        live = LiveCenterlineState(reward_fn.trajectory_centerline, dt=dt)
        live.reset()
        planner = SimpleMPPIPlanner(
            bundle.model, bundle.norm, reward_fn, dt, bundle.device
        )  # , PlannerConfig())

    telemetry = bridge.reset_to_start()
    belief.reset(telemetry)
    live.update(belief._pos, telemetry['vel_world'])


    for _ in range(max_ticks):
        cur = belief.current()
        cur._live = live # not clean but temporary to get it working
        action = planner.plan(cur, bundle.track_ctx)
        # action = GameAction(
        #     steer=float(1),
        #     gas=bool(1),
        #     brake=bool(0),
        # )
        # print(f"sending brake: {action.brake} | steer: {action.steer} | gas: {action.gas}")
        belief.record_action_sent(action)

        telemetry = bridge.step(action, hold_ticks=hold_ticks)
        # print(f"got telemetry: {telemetry}")
        is_respawn = bool(telemetry.get('launched_respawn') or telemetry.get('static_respawn'))
        belief.update(telemetry, is_respawn=is_respawn)
        live.update(belief._pos, telemetry['vel_world'])

        planner.warm_start_from_last()

        # TODO: lap-complete / finish detection to break the loop


if __name__ == "__main__":
    bundle = ModelBundle.load(
        checkpoint_path="",
        geo_h5_path="",
    )
    bridge = TMWorldModelBridge()
    run(bundle, bridge)
