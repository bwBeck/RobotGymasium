"""End-to-end tests of RoboGym against the fake robot (ESP32 bridge + sketch + camera)."""

import math
import threading
import time

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

import robotgym  # noqa: F401 - registers RoboGym-v0
from robotgym import RobotCalibration, RobotConnectionError, RoboGym
from robotgym.fake_robot import FakeEsp32Bridge, FakeRobotServer, SimRobot

from .conftest import FAST


def test_reset_returns_zero_deltas_and_configures_the_camera(fake, make_env):
    env = make_env()
    obs, info = env.reset()
    assert env.observation_space.contains(obs)
    assert obs["image"].shape == (120, 160, 3) and obs["image"].std() > 10
    for key in ("delta_pose", "motor_turns", "dt", "robot_dt"):
        assert not obs[key].any(), key
    # Facing a wall 1.5 m away (sensor sits 9 cm ahead of the centre).
    assert obs["distance"][0] == pytest.approx(1.41, abs=0.02)
    assert info["imu_ok"] and info["battery_v"] == pytest.approx(7.9)
    # The stream forces face detection on; the env must have switched it back off.
    assert fake.camera.face_detect == 0
    assert fake.camera.framesize == 4


def test_forward_and_spin_show_up_in_the_observation(make_env):
    env = make_env(action_duration=0.15)
    env.reset()
    obs, reward, terminated, truncated, info = env.step([0.8, 0.8])
    assert (reward, terminated, truncated) == (0.0, False, False)
    dx, dy, dth = obs["delta_pose"]
    assert dx > 0.01 and abs(dy) < 1e-3 and abs(dth) < 1e-3
    assert obs["motor_turns"][0] > 0.1 and obs["motor_turns"][0] == pytest.approx(obs["motor_turns"][1])
    assert info["pwm"] == (204, 204)

    for _ in range(2):
        obs, *_ = env.step([-0.8, 0.8])  # left track back, right forward = turn left
    assert obs["delta_pose"][2] > 0.1  # counter-clockwise is positive
    assert obs["motor_turns"][0] < 0 < obs["motor_turns"][1]


def test_deltas_add_up_to_the_cumulative_odometry(make_env):
    env = make_env(action_duration=0.1)
    env.reset()
    turns = np.zeros(2)
    heading = 0.0
    position = np.zeros(2)
    robot_time = 0.0
    for action in ([0.7, 0.7], [0.3, 0.9], [0.9, 0.9], [-0.5, 0.5], [0.6, 0.6]):
        obs, _, _, _, info = env.step(action)
        dx, dy, dth = obs["delta_pose"]
        c, s = math.cos(heading), math.sin(heading)
        position += (c * dx - s * dy, s * dx + c * dy)  # back into the odometry frame
        heading += dth
        turns += obs["motor_turns"]
        robot_time += obs["robot_dt"][0]
        assert obs["dt"][0] == pytest.approx(obs["robot_dt"][0], abs=0.05)
    np.testing.assert_allclose(position, info["pose"][:2], atol=2e-3)
    assert heading == pytest.approx(info["pose"][2], abs=2e-3)
    np.testing.assert_allclose(turns, info["motor_turns_total"], atol=2e-3)
    assert robot_time > 0.5


def test_camera_view_changes_when_the_robot_turns(make_env):
    env = make_env(action_duration=0.3)
    obs0, _ = env.reset()
    obs, *_ = env.step([-1.0, 1.0])
    diff = np.abs(obs["image"].astype(int) - obs0["image"].astype(int)).mean()
    assert diff > 5


def test_calibration_and_hold_timeout_are_sent_to_the_robot(fake, make_env):
    cal = RobotCalibration(max_rpm=120, deadband_pwm=30, time_constant=0.05,
                           sprocket_diameter=0.05, track_width=0.2, gyro_sign=-1)
    env = make_env(calibration=cal, hold_timeout=0.8)
    env.reset()
    robot = fake.robot
    assert robot.max_rps == pytest.approx(2.0)
    assert (robot.deadband, robot.tau, robot.track_width) == (30, 0.05, 0.2)
    assert robot.circumference == pytest.approx(math.pi * 0.05)
    assert robot.hold_timeout == pytest.approx(0.8)
    obs, *_ = env.step([-0.8, 0.8])
    assert obs["delta_pose"][2] < 0  # gyro_sign=-1 flips the heading


def test_robot_stops_by_itself_when_the_agent_goes_quiet(fake, make_env):
    env = make_env(hold_timeout=0.2, action_duration=0.05)
    env.reset()
    env.step([0.5, 0.5])
    time.sleep(0.5)
    assert fake.robot.cmd == [0, 0]
    _, _, _, _, info = env.step([0.0, 0.0])
    assert info["watchdog_stopped"]


def test_heartbeats_keep_the_bridge_connection_alive_while_idle(fake, make_env):
    env = make_env()
    env.reset()
    time.sleep(5.5)  # the ESP32 drops silent clients after ~4 s
    env.step([0.0, 0.0])
    assert fake.bridge.heartbeat_drops == 0
    assert fake.bridge.clients_served == 1


def test_reconnects_after_the_connection_drops(fake, make_env):
    env = make_env()
    env.reset()
    env.step([0.5, 0.5])
    fake.bridge.drop_client()
    obs, *_ = env.step([0.5, 0.5])
    assert fake.bridge.clients_served == 2
    assert obs["delta_pose"][0] > 0  # odometry carried across the reconnect


def test_retries_a_command_the_robot_never_answered():
    class FlakyRobot(SimRobot):
        drops = 1

        def _handle(self, s, now):
            if s.startswith("A,") and self.drops:
                self.drops -= 1
                return
            super()._handle(s, now)

    with FakeRobotServer(robot=FlakyRobot()) as fake:
        env = RoboGym(**fake.env_kwargs(), **FAST, camera="none", timeout=0.3)
        try:
            env.reset()  # its first A frame (the stop) is swallowed
            obs, *_ = env.step([0.5, 0.5])
            assert obs["delta_pose"][0] > 0
        finally:
            env.close()


def test_detects_a_robot_reboot_and_restores_the_calibration(fake, make_env):
    env = make_env(calibration=RobotCalibration(max_rpm=150))
    env.reset()
    env.step([0.5, 0.5])
    fake.robot.reboot()
    assert fake.robot.max_rps == pytest.approx(3.0)  # back to the sketch default
    obs, _, _, _, info = env.step([0.5, 0.5])
    assert info["robot_rebooted"]
    assert not obs["delta_pose"].any() and obs["robot_dt"][0] == 0.0
    assert fake.robot.max_rps == pytest.approx(2.5)
    _, _, _, _, info = env.step([0.5, 0.5])
    assert not info["robot_rebooted"]


def test_snapshot_camera_mode(fake, make_env):
    env = make_env(camera="snapshot")
    obs, _ = env.reset()
    assert obs["image"].std() > 10
    assert any(r.startswith("/capture") for r in fake.camera.requests)
    assert "/control?var=face_detect&val=0" in fake.camera.requests


def test_camera_none_drops_the_image_key(make_env):
    env = make_env(camera="none")
    obs, _ = env.reset()
    assert "image" not in obs and "image" not in env.observation_space.spaces


def test_render_rgb_array_returns_the_full_camera_frame(make_env):
    env = make_env(render_mode="rgb_array")
    env.reset()
    frame = env.render()
    assert frame.shape == (240, 320, 3) and frame.dtype == np.uint8


def test_action_mapping_with_min_pwm_and_clipping(make_env):
    env = make_env(camera="none", min_pwm=60, max_pwm=200, action_duration=0.0)
    env.reset()
    assert env.step([0.5, -1.0])[4]["pwm"] == (130, -200)
    assert env.step([0.0, 0.0])[4]["pwm"] == (0, 0)
    assert env.step(np.array([2.0, -3.0]))[4]["pwm"] == (200, -200)


def test_invalid_use_is_rejected(make_env):
    env = make_env(camera="none")
    with pytest.raises(gym.error.ResetNeeded):
        env.step([0.0, 0.0])
    env.reset()
    with pytest.raises(ValueError):
        env.step([0.0, 0.0, 0.0])
    with pytest.raises(ValueError):
        env.step([float("nan"), 0.0])
    with pytest.raises(ValueError):
        RoboGym(camera="webcam")


def test_reward_and_termination_hooks(make_env):
    env = make_env(
        camera="none",
        reward_fn=lambda obs, action, info: float(obs["delta_pose"][0]) - 0.01 * float(np.abs(action).sum()),
        terminated_fn=lambda obs, info: obs["distance"][0] < 5.0,
    )
    env.reset()
    _, reward, terminated, _, _ = env.step([1.0, 1.0])
    assert reward > 0 and terminated


def test_passes_the_gymnasium_env_checker(fake):
    # A physical robot can't give identical observations twice; the registered spec says
    # so (nondeterministic=True), which tells the checker to skip its determinism test.
    env = gym.make("RoboGym-v0", action_duration=0.05, **fake.env_kwargs(), **FAST)
    try:
        assert env.spec.nondeterministic
        check_env(env.unwrapped, skip_render_check=True)
    finally:
        env.close()


def test_gym_make_with_time_limit(fake):
    env = gym.make("RoboGym-v0", max_episode_steps=3, camera="none", **fake.env_kwargs(), **FAST)
    try:
        env.reset()
        truncated = [env.step(env.action_space.sample() * 0.3)[3] for _ in range(3)]
        assert truncated == [False, False, True]
    finally:
        env.close()


class _StockFirmware:
    """ELEGOO's factory sketch: ignores RoboGym frames, answers {"N":100} with {ok}."""

    def __init__(self):
        self._out = bytearray()
        self._lock = threading.Lock()

    def write(self, data):
        if data == b'{"N":100}':
            with self._lock:
                self._out += b"{ok}"

    def read(self, timeout=0.0):
        with self._lock:
            out, self._out = bytes(self._out), bytearray()
        return out


class _Silent:
    def write(self, data):
        pass

    def read(self, timeout=0.0):
        return b""


@pytest.mark.parametrize("device, message", [(_StockFirmware(), "factory firmware"),
                                             (_Silent(), "did not answer")])
def test_helpful_error_when_the_robot_does_not_answer(device, message):
    bridge = FakeEsp32Bridge(device)
    bridge.start()
    try:
        env = RoboGym(host="127.0.0.1", control_port=bridge.port, camera="none", connect_timeout=1.0)
        with pytest.raises(RobotConnectionError, match=message):
            env.reset()
        env.close()
    finally:
        bridge.stop()


class _SwitchedRobot(SimRobot):
    """A robot whose UART is cut off from the ESP32 until `connected` is set - the
    shield's Upload-Cam switch left on "Upload"."""

    connected = False

    def write(self, data):
        if self.connected:
            super().write(data)


def test_recovers_and_reconfigures_once_the_upload_cam_switch_is_flipped():
    robot = _SwitchedRobot()
    with FakeRobotServer(robot=robot) as fake:
        env = RoboGym(**fake.env_kwargs(), **FAST, connect_timeout=1.0, render_mode="rgb_array",
                      calibration=RobotCalibration(max_rpm=150))
        try:
            with pytest.raises(RobotConnectionError, match="Upload-Cam switch"):
                env.reset()
            # The camera is on the ESP32: it must work even while the Arduino is cut off.
            assert env.render() is not None
            robot.connected = True  # switch flipped to "Cam" while the program keeps retrying
            env.reset()
            assert robot.max_rps == pytest.approx(2.5)  # the handshake ran again: calibration sent
        finally:
            env.close()


def test_helpful_error_when_nothing_listens():
    env = RoboGym(host="127.0.0.1", control_port=1, camera="none", connect_timeout=1.0)
    with pytest.raises(RobotConnectionError, match="cannot reach the robot"):
        env.reset()
