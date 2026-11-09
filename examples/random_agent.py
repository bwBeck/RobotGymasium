"""Drive the robot with random actions and print what it observes.

    python examples/random_agent.py            # the real robot (PC on its "ELEGOO-xxxx" WiFi)
    python examples/random_agent.py --fake     # no hardware: simulated robot

Start with the robot on a stand (tracks off the ground) the first time.
"""

import argparse
import logging

import numpy as np

from robotgym import RoboGym
from robotgym.fake_robot import FakeRobotServer


def reward_fn(obs, action, info):
    """Example task: drive forward, keep away from obstacles, don't waste energy."""
    forward = float(obs["delta_pose"][0])
    too_close = float(obs["distance"][0]) < 0.20
    return forward - 0.05 * too_close - 0.001 * float(np.abs(action).sum())


def terminated_fn(obs, info):
    return float(obs["distance"][0]) < 0.10  # about to hit something


def run(env: RoboGym, steps: int) -> None:
    obs, info = env.reset()
    print(f"battery {info['battery_v']:.2f} V, distance {obs['distance'][0]:.2f} m, "
          f"image {obs['image'].shape if 'image' in obs else 'off'}")
    print(" step  action(L,R)    dx[m]   dy[m]  dth[rad] turns(L,R)     dist[m]  dt[s] robot_dt  reward")
    for t in range(steps):
        action = env.action_space.sample() * 0.7
        action[1] = action[0] * 0.7 + action[1] * 0.3  # bias towards driving, less spinning
        obs, reward, terminated, truncated, info = env.step(action)
        dx, dy, dth = obs["delta_pose"]
        tl, tr = obs["motor_turns"]
        print(f"{t:5d}  {action[0]:+.2f},{action[1]:+.2f}   {dx:+.4f} {dy:+.4f} {dth:+.4f}  "
              f"{tl:+.3f},{tr:+.3f}  {obs['distance'][0]:6.3f}  {obs['dt'][0]:.3f}  "
              f"{obs['robot_dt'][0]:.3f}  {reward:+.4f}")
        if terminated or truncated:
            print("episode over - resetting")
            obs, info = env.reset()
    print(f"odometry since last reset: x={info['pose'][0]:.3f} m  y={info['pose'][1]:.3f} m  "
          f"heading={info['pose'][2]:.3f} rad")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fake", action="store_true", help="use the simulated robot")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--no-camera", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    kwargs = {"reward_fn": reward_fn, "terminated_fn": terminated_fn,
              "camera": "none" if args.no_camera else "stream"}
    if args.fake:
        with FakeRobotServer() as fake:
            env = RoboGym(**fake.env_kwargs(), **kwargs)
            try:
                run(env, args.steps)
            finally:
                env.close()
    else:
        env = RoboGym(host=args.host, **kwargs)
        try:
            run(env, args.steps)
        finally:
            env.close()


if __name__ == "__main__":
    main()
