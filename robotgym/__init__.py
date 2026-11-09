"""Gymnasium environment for the ELEGOO Conqueror robot tank.

    from robotgym import RoboGym
    env = RoboGym()                      # PC on the robot's "ELEGOO-xxxx" WiFi
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step([0.5, 0.5])

Importing the package also registers ``gymnasium.make("RoboGym-v0")``.
"""

import gymnasium as gym

from .env import RobotCalibration, RobotConnectionError, RoboGym
from .protocol import PROTOCOL_VERSION, RobotState

__all__ = ["PROTOCOL_VERSION", "RobotCalibration", "RobotConnectionError", "RobotState", "RoboGym"]
__version__ = "0.1.0"

if "RoboGym-v0" not in gym.registry:
    gym.register(id="RoboGym-v0", entry_point="robotgym.env:RoboGym", nondeterministic=True)
