import pytest

from robotgym import RoboGym
from robotgym.fake_robot import FakeRobotServer

# Keep resets quick in tests: short settle, no gyro calibration pause.
FAST = {"reset_settle_time": 0.05, "calibrate_gyro_on_reset": False}


@pytest.fixture
def fake():
    with FakeRobotServer() as server:
        yield server


@pytest.fixture
def make_env(fake):
    """Factory for envs wired to the `fake` robot; closes them after the test."""
    envs = []

    def _make(**kwargs):
        env = RoboGym(**{**fake.env_kwargs(), **FAST, **kwargs})
        envs.append(env)
        return env

    yield _make
    for env in envs:
        env.close()
