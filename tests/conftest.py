import pytest

from robotgym import RoboGym
from robotgym.fake_robot import FakeRobotServer

# Actions map straight to PWM (no deadzone, no smoothing), so tests can predict the PWM
# and the motion; tests of those two features turn them on themselves.
DIRECT = {"pwm_deadzone": 0, "action_smoothing": 0.0}
# Test defaults: DIRECT, and quick resets (short settle, no gyro calibration pause).
FAST = {"reset_settle_time": 0.05, "calibrate_gyro_on_reset": False, **DIRECT}


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
