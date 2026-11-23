"""The safety layer: its checks on their own, then inside the env against the fake robot."""

import math
import re

import numpy as np
import pytest

from robotgym import RoboGym, SafetyConfig, SafetyLayer
from robotgym.fake_robot import FakeRobotServer, SimRobot

from .conftest import FAST


def _step(accel_x=0.0, yaw_deg_s=0.0, distance=2.0, robot_dt=0.2, **info):
    """An env step's (obs, info) with the values the checks read."""
    obs = {"accel": np.array([accel_x, 0.0, 0.0], np.float32),
           "delta_pose": np.array([0.0, 0.0, math.radians(yaw_deg_s) * robot_dt], np.float32),
           "distance": np.array([distance], np.float32),
           "robot_dt": np.array([robot_dt], np.float32)}
    info = {"imu_ok": True, "accel_raw": np.zeros(3), "robot_rebooted": False,
            "watchdog_stopped": False, **info}
    return obs, info


def _check(prev, pwm, **step):
    layer = SafetyLayer()
    return layer.check(prev, pwm, *_step(**step)), layer


# ------------------------------------------------------------------ stall

def test_stall_trips_when_a_big_pwm_change_brings_no_acceleration():
    trip, layer = _check((0, 0), (150, 150), accel_x=0.03)
    assert trip.trigger == "stall" and layer.tripped and "150" in trip.message


@pytest.mark.parametrize("prev, pwm, accel_x", [
    ((0, 0), (150, 150), 0.5),  # it did accelerate
    ((150, 150), (0, 0), -0.5),  # braking counts too
    ((0, 0), (100, 100), 0.06),  # a small change needs only a little: 0.001 * (100 - 50)
    ((0, 0), (50, 50), 0.0),  # changed by 50: not above b, not checked
    ((110, 110), (160, 160), 0.0),  # same at speed
    ((150, 150), (150, 150), 0.0),  # steady cruise, nothing changed
])
def test_stall_stays_quiet(prev, pwm, accel_x):
    trip, _ = _check(prev, pwm, accel_x=accel_x)
    assert trip is None


def test_stall_judges_the_change_in_forward_speed():
    # Forward speed follows the tracks' mean PWM, so one track counts half.
    assert _check((0, 0), (0, -120), accel_x=0.0)[0].trigger == "stall"  # reversing, right track only: 60
    assert _check((120, 120), (120, 0), accel_x=0.0)[0].trigger == "stall"  # one track stopped: 60
    # Curving (one track slowed by 51 = 25.5) and starting a spin (0) bring little or no
    # forward acceleration on a healthy robot: not a stall. The yaw shows they turned.
    assert _check((153, 153), (153, 102), accel_x=-0.2, yaw_deg_s=-8.0)[0] is None
    assert _check((0, 0), (150, -150), accel_x=0.0, yaw_deg_s=-40.0)[0] is None


@pytest.mark.parametrize("prev, pwm, accel_x, trips", [
    ((0, 0), (150, 150), 0.09, True), ((0, 0), (150, 150), 0.11, False),  # needs 0.1
    ((0, 0), (250, 250), 0.19, True), ((0, 0), (250, 250), 0.21, False),  # bigger change: 0.2
    ((250, 250), (0, 0), -0.19, True), ((250, 250), (0, 0), -0.21, False),  # braking: mirrored
    ((0, 0), (-250, -250), -0.19, True), ((0, 0), (-250, -250), 0.21, False),  # either sign
])
def test_stall_needs_more_acceleration_the_bigger_the_change(prev, pwm, accel_x, trips):
    trip, _ = _check(prev, pwm, accel_x=accel_x)
    assert (trip is not None and trip.trigger == "stall") == trips, trip


def test_stall_line():
    config = SafetyConfig()  # n = 0.001, b = 50
    assert config.stall_min_accel(50) is None and config.stall_min_accel(-50) is None
    assert config.stall_min_accel(150) == pytest.approx(0.1)
    assert config.stall_min_accel(-250) == pytest.approx(0.2)
    assert SafetyConfig(stall_slope=-0.002).stall_min_accel(150) == pytest.approx(0.2)  # |n (...)|
    assert SafetyConfig(stall_offset=100).stall_min_accel(150) == pytest.approx(0.05)


def test_stall_is_checked_at_any_speed_once_the_change_is_above_b():
    # Both tracks under pwm_threshold (which only gates the no-turn check): still checked.
    assert _check((0, 0), (80, 80), accel_x=0.0)[0].trigger == "stall"  # needs 0.03


def test_stall_counts_from_rest_after_the_robots_hold_timeout_stopped_it():
    trip, _ = _check((150, 150), (150, 150), accel_x=0.0, watchdog_stopped=True)
    assert trip.trigger == "stall"


# --------------------------------------------------------------- no turn

def test_no_turn_trips_when_unequal_tracks_bring_no_yaw():
    trip, _ = _check((150, 110), (150, 110), yaw_deg_s=0.3)
    assert trip.trigger == "no_turn" and "+40" in trip.message


@pytest.mark.parametrize("pwm, yaw", [
    ((150, 110), 15.0),  # it turned
    ((150, 110), -15.0),  # either direction
    ((150, 120), 0.0),  # differ by 30: not above 30
    ((90, 30), 0.0),  # neither above 100
])
def test_no_turn_stays_quiet(pwm, yaw):
    assert _check(pwm, pwm, yaw_deg_s=yaw)[0] is None


def test_motion_checks_skip_steps_they_cant_judge():
    for info in ({"imu_ok": False}, {"accel_raw": None}, {"robot_rebooted": True}):
        assert _check((0, 0), (200, 50), **info)[0] is None, info
    assert _check((0, 0), (200, 50), robot_dt=0.0)[0] is None


# --------------------------------------------------------------- obstacle

def test_obstacle_trips_only_while_driving_forward():
    assert _check((60, 60), (60, 60), distance=0.12)[0].trigger == "obstacle"
    assert _check((60, 60), (60, 60), distance=0.16)[0] is None
    assert _check((-60, -60), (-60, -60), distance=0.05)[0] is None  # backing away
    assert _check((0, 0), (0, 0), distance=0.05)[0] is None  # parked
    assert _check((-80, 80), (-80, 80), distance=0.05, yaw_deg_s=30)[0] is None  # spinning away


def test_forward_command_is_refused_near_an_obstacle():
    layer = SafetyLayer()
    assert layer.command((-80, -80), 0.10) == (-80, -80)  # reversing is fine
    assert layer.command((80, 80), None) == (80, 80)  # no echo: nothing in range
    assert layer.command((80, 80), 0.10) == (0, 0)
    assert layer.trip.trigger == "obstacle" and "refused" in layer.trip.message


# ------------------------------------------------------------------ latch

def test_a_trip_latches_until_cleared():
    trip, layer = _check((0, 0), (150, 150), accel_x=0.0)
    assert layer.command((-200, 200), 3.0) == (0, 0)  # everything held at zero
    assert layer.check((0, 0), (0, 0), *_step(distance=0.01)) is None  # no second trip
    assert layer.trip is trip  # the first reason stays
    layer.clear()
    assert not layer.tripped and layer.command((-200, 200), 3.0) == (-200, 200)


def test_thresholds_come_from_the_config():
    layer = SafetyLayer(SafetyConfig(min_distance=0.5, stall_slope=0.01))
    assert layer.command((50, 50), 0.4) == (0, 0)
    layer.clear()
    trip = layer.check((0, 0), (150, 150), *_step(accel_x=0.5))  # the default line needs 0.1
    assert trip.trigger == "stall" and "needs ±1.00" in trip.message


# --------------------------------------------------------- in the env

@pytest.fixture
def robot_env(request):
    """An env with the safety layer on, against a fake robot built from ``SimRobot(**param)``."""
    robot_class, kwargs = getattr(request, "param", (SimRobot, {}))
    with FakeRobotServer(robot=robot_class(**kwargs)) as fake:
        env = RoboGym(**fake.env_kwargs(), **FAST, camera="none", safety=SafetyConfig())
        try:
            yield env, fake.robot
        finally:
            env.close()


def _last_command(robot) -> str:
    return [s for s in robot.received if s.startswith("A,")][-1]


def test_off_by_default(make_env):
    env = make_env(camera="none")
    _, info = env.reset()
    assert env.safety is None and "safety_trip" not in info


@pytest.mark.parametrize("robot_env", [(SimRobot, {"start": (2.55, 1.5, 0.0)})], indirect=True)
def test_obstacle_stops_the_robot_at_once_and_latches(robot_env):
    env, robot = robot_env  # 0.36 m from the wall, facing it
    _, info = env.reset()
    assert info["safety_trip"] is None
    for _ in range(20):
        obs, _, _, _, info = env.step([0.4, 0.4])
        if info["safety_trip"] is not None:
            break
    assert info["safety_trip"].trigger == "obstacle" and obs["distance"][0] < 0.15
    assert info["pwm"] == (102, 102)  # this step still drove...
    assert re.fullmatch(r"A,\d+,0,0,0", _last_command(robot))  # ...and the env stopped it right after
    assert robot.cmd == [0, 0]

    obs, _, _, _, info = env.step([1.0, -1.0])  # latched: nothing moves
    assert info["pwm"] == (0, 0) and info["safety_trip"].trigger == "obstacle"
    env.reset()
    assert env.safety.tripped  # a reset doesn't clear it

    env.safety.clear()
    _, _, _, _, info = env.step([0.4, 0.4])  # forward into the wall: refused before moving
    assert info["pwm"] == (0, 0) and "refused" in info["safety_trip"].message
    env.safety.clear()
    obs, _, _, _, info = env.step([-0.4, -0.4])  # backing away is allowed
    assert info["pwm"] == (-102, -102) and info["safety_trip"] is None


@pytest.mark.parametrize("robot_env", [(SimRobot, {"start": (0.12, 1.5, 0.0)})], indirect=True)
def test_stall_when_reversing_into_a_wall(robot_env):
    env, robot = robot_env  # back against the wall: reversing goes nowhere
    env.reset()
    obs, _, _, _, info = env.step([-0.6, -0.6])
    assert info["safety_trip"].trigger == "stall" and abs(obs["accel"][0]) < 0.1
    assert robot.cmd == [0, 0]


class _SlippingTracks(SimRobot):
    SKID = 0.0  # the tracks spin but the robot never rotates


@pytest.mark.parametrize("robot_env", [(_SlippingTracks, {})], indirect=True)
def test_no_turn_when_the_robot_does_not_rotate(robot_env):
    env, robot = robot_env
    env.reset()
    _, _, _, _, info = env.step([0.6, 0.6])  # straight: accelerates fine
    assert info["safety_trip"] is None
    _, _, _, _, info = env.step([0.6, 0.45])  # 153 vs 115: should turn, doesn't
    assert info["safety_trip"].trigger == "no_turn"
    assert robot.cmd == [0, 0]


def test_normal_driving_does_not_trip(robot_env):
    env, _ = robot_env  # the default arena: 1.4 m of room ahead
    env.reset()
    # Speed up, curve right, straighten out, stop.
    plan = [[0.6, 0.6]] * 3 + [[0.6, 0.4]] * 3 + [[0.6, 0.6]] * 2 + [[0.0, 0.0]] * 2
    for action in plan:
        _, _, _, _, info = env.step(action)
        assert info["safety_trip"] is None, (action, info["safety_trip"])


def test_safety_judges_the_pwm_the_robot_gets_not_the_action():
    # 0.05 m from the wall, facing it. Deadzone 120 and smoothing 0.5, as the env defaults.
    with FakeRobotServer(robot=SimRobot(start=(2.86, 1.5, 0.0))) as fake:
        env = RoboGym(**{**fake.env_kwargs(), **FAST, "camera": "none", "safety": SafetyConfig(),
                         "pwm_deadzone": 120, "action_smoothing": 0.5})
        try:
            env.reset()
            seen = []  # every PWM pair the safety layer was asked about
            command, check = env.safety.command, env.safety.check
            env.safety.command = lambda pwm, distance: seen.append(("command", pwm)) or command(pwm, distance)
            env.safety.check = lambda prev, pwm, obs, info: seen.append(("check", prev, pwm)) or check(
                prev, pwm, obs, info)

            # Forward into the wall, but the smoothed PWM (76, then 115) is still in the
            # deadzone: the robot gets nothing, so there's nothing to refuse yet.
            for _ in range(2):
                _, _, _, _, info = env.step([0.6, 0.6])
                assert info["pwm"] == (0, 0) and info["pwm_target"] == (153, 153)
                assert info["safety_trip"] is None
            assert seen == [("command", (0, 0)), ("check", (0, 0), (0, 0))] * 2
            # Smoothed 134: now the robot would drive forward into the wall. Refused.
            _, _, _, _, info = env.step([0.6, 0.6])
            assert seen[-2:] == [("command", (134, 134)), ("check", (0, 0), (0, 0))]  # 0 was sent
            assert info["pwm"] == (0, 0) and "refused" in info["safety_trip"].message
            assert fake.robot.cmd == [0, 0]

            env.safety.clear()  # the smoothing restarted at the trip: backing off ramps from rest
            _, _, _, _, info = env.step([-1.0, -1.0])
            assert info["pwm"] == (-128, -128) and info["safety_trip"] is None
            assert seen[-2:] == [("command", (-128, -128)), ("check", (0, 0), (-128, -128))]
        finally:
            env.close()
