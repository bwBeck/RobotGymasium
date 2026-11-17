"""RoboGym: a gymnasium environment for the ELEGOO Conqueror robot tank."""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .camera import Frame, SnapshotCamera, StreamCamera, decode_jpeg, esp32_set
from .link import ProtocolError, RobotLink, SerialTransport, TcpTransport
from .protocol import RobotState, relative_pose, u32_diff
from .safety import SafetyConfig, SafetyLayer

logger = logging.getLogger("robotgym")

LOW_BATTERY_V = 7.0  # ELEGOO's own low-voltage threshold for the 2S pack

_TROUBLESHOOTING = """Checklist:
  * robot switched on, ESP32 camera module plugged into the shield
  * this computer is on the robot's WiFi "ELEGOO-xxxxxxxx" (the robot is 192.168.4.1)
  * arduino/RoboGym/RoboGym.ino is flashed to the UNO
  * the shield's "Upload-Cam" switch is on "Cam" ("Upload" only while flashing)
  * the ELEGOO phone app is not connected (the bridge serves one client at a time)"""


class RobotConnectionError(ConnectionError):
    """The robot or its camera could not be reached, or stopped answering."""


class _Unreachable(RobotConnectionError):
    """Nothing answers at the robot's address at all (wrong WiFi, robot off)."""


@dataclass
class RobotCalibration:
    """Constants of the motor model the robot uses to estimate turns and odometry.

    The stock Conqueror motors have no encoders, so sprocket turns and distance are
    integrated from the PWM command on the robot. These defaults are rough guesses:
    run ``examples/calibrate.py`` and pass the results here. They are sent to the robot
    every time the env connects, so changing them never needs a re-flash.
    """

    max_rpm: float = 180.0  # sprocket RPM at PWM 255 with a 7.4 V battery, on the ground
    deadband_pwm: float = 40.0  # PWM below which the tracks don't move
    time_constant: float = 0.08  # motor speed lag [s]
    sprocket_diameter: float = 0.043  # effective drive sprocket diameter [m]
    track_width: float = 0.15  # between track centres [m]; only used if the IMU fails
    gyro_sign: int = 1  # -1 if left turns read as negative (see calibrate.py spin)
    imu_forward: str = "+x"  # MPU6050 axis pointing to the robot's front (calibrate.py imu)


STANDARD_GRAVITY = 9.80665
_IMU_AXES = {"+x": (1.0, 0.0, 0.0), "-x": (-1.0, 0.0, 0.0), "+y": (0.0, 1.0, 0.0), "-y": (0.0, -1.0, 0.0)}


def imu_rotation(calibration: RobotCalibration) -> np.ndarray:
    """3x3 matrix taking MPU6050 axes to robot axes (x forward, y left, z up).

    The chip lies flat: ``imu_forward`` says which of its axes points forward, and the
    gyro sign says whether its z axis points up (1) or down (-1, chip upside down).
    """
    try:
        forward = np.array(_IMU_AXES[calibration.imu_forward])
    except KeyError:
        raise ValueError(f"imu_forward must be one of {sorted(_IMU_AXES)}, "
                         f"not {calibration.imu_forward!r}") from None
    up = np.array([0.0, 0.0, 1.0 if calibration.gyro_sign >= 0 else -1.0])
    return np.array([forward, np.cross(up, forward), up])


class RoboGym(gym.Env):
    """Gymnasium environment for the ELEGOO Conqueror robot tank.

    ``step(action)`` sends the track speeds to the robot, which applies them for
    ``action_duration`` seconds and then replies with its sensor packet. The camera
    frame comes straight from the ESP32 module. Observation (``spaces.Dict``):

    ===========  =========  ====================================================
    image        (H, W, 3)  camera frame, uint8 RGB (absent if camera="none")
    delta_pose   (3,)       [dx, dy, dtheta] since the last observation, in the
                            robot frame at that observation [m, m, rad]
    motor_turns  (2,)       [left, right] sprocket turns since the last
                            observation (estimated: the motors have no encoders)
    distance     (1,)       ultrasonic range [m]; max_distance if nothing in range
    dt           (1,)       host time since the last observation [s]
    robot_dt     (1,)       robot-clock time since the robot's last packet [s]
    accel        (3,)       mean acceleration since the last observation, robot
                            frame [m/s^2], minus the at-rest reading taken at
                            reset (so ~0 when still; raw value in info)
    ===========  =========  ====================================================

    Action: ``Box(-1, 1, (2,))`` = [left, right] track speed, sign = direction. Each step
    turns it into the PWM the robot gets, per track:

    1. scale to PWM (``min_pwm``..``max_pwm``);
    2. smooth: ``s = action_smoothing * s + (1 - action_smoothing) * pwm``, so the robot
       gets a smoothed version of the actions (``s`` restarts from 0 after a reset, a
       safety stop, or a pause long enough for the robot's hold timeout to stop it);
    3. round ``s``; below ``pwm_deadzone`` (120) -> 0. Only the smoothed PWM goes through
       the deadzone, so the robot never gets a PWM between 0 and it, also while ramping;
    4. the safety layer, if on, judges this PWM and may replace it with 0.

    ``info["pwm"]`` is the PWM the robot received, ``info["pwm_target"]`` the result of 1.

    Reward and termination depend on the task: pass ``reward_fn(obs, action, info)``
    and ``terminated_fn(obs, info)`` (defaults: 0.0 and never). Bound episodes with
    ``gymnasium.wrappers.TimeLimit`` or ``gym.make("RoboGym-v0", max_episode_steps=N)``.

    The robot keeps executing the last action after ``step`` returns, until the next
    command arrives or ``hold_timeout`` passes (then it stops by itself), so the deltas
    in an observation cover everything since the previous one, including the time the
    agent spent thinking.

    ``safety=SafetyConfig()`` adds the PC-side safety layer (``env.safety``, see
    :mod:`robotgym.safety`): it stops the robot when it stalls, fails to turn or nears an
    obstacle, and holds it stopped until ``env.safety.clear()``. ``info["safety_trip"]``
    is then the latched :class:`~robotgym.safety.SafetyTrip`, or None while armed.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 5}

    IMAGE_WAIT_S = 0.3  # max wait for a camera frame newer than the action

    def __init__(
        self,
        host: str = "192.168.4.1",
        control_port: int = 100,
        serial_port: str | None = None,
        baudrate: int = 9600,
        camera: str = "stream",
        http_port: int = 80,
        camera_port: int = 81,
        camera_framesize: int | None = 4,
        image_shape: tuple[int, int] = (120, 160),
        action_duration: float = 0.1,
        max_pwm: int = 255,
        min_pwm: int = 0,
        pwm_deadzone: int = 120,
        action_smoothing: float = 0.5,
        hold_timeout: float = 0.5,
        max_distance: float = 4.0,
        reset_settle_time: float = 0.5,
        calibrate_gyro_on_reset: bool = True,
        calibration: RobotCalibration | None = None,
        safety: SafetyConfig | None = None,
        reward_fn: Callable[[dict, np.ndarray, dict], float] | None = None,
        terminated_fn: Callable[[dict, dict], bool] | None = None,
        render_mode: str | None = None,
        timeout: float = 1.0,
        retries: int = 2,
        connect_timeout: float = 5.0,
    ):
        """
        Args:
            host: IP of the ESP32 module (its WiFi access point is always 192.168.4.1).
            control_port: TCP port of the ESP32's serial bridge.
            serial_port: e.g. ``"/dev/ttyUSB0"`` to drive over USB instead of WiFi
                (bench testing with the Upload-Cam switch on Upload; needs pyserial).
            baudrate: must match ``SERIAL_BAUD`` in the sketch.
            camera: ``"stream"`` (background MJPEG reader, freshest frames),
                ``"snapshot"`` (fetch /capture each step) or ``"none"`` (no image key).
            http_port / camera_port: ESP32 ports for /capture + /control and /stream.
            camera_framesize: ESP32 frame size index to request, or None to leave it.
                On ELEGOO's firmware 4 = 320x240 (QVGA), 7 = 800x600 (its default).
                Frames are resized to ``image_shape`` either way.
            image_shape: (height, width) of the observation image.
            action_duration: seconds each action runs before the robot reports back.
            max_pwm: PWM for an action of +-1.
            min_pwm: PWM for the smallest non-zero action - set it near the motor
                deadband so small actions still move the robot (0 = linear mapping).
            pwm_deadzone: a smoothed track PWM whose magnitude is below this is sent as 0
                (0 = off). It applies after the smoothing, not to the action itself. Unlike
                ``min_pwm`` it doesn't remap: an action that settles below it stops that track.
            action_smoothing: weight of the previous smoothed PWM in each step's
                exponential average, 0 <= x < 1 (0 = off). At 0.5 a step change in the
                action reaches the robot as 50 %, 75 %, 88 %, ... of it, one step each.
            hold_timeout: the robot stops if no command arrives within this many
                seconds after it replied (safety net if the PC hangs or disconnects).
            max_distance: clip / no-echo value for the distance observation [m].
            reset_settle_time: pause between stopping and zeroing on reset() [s].
            calibrate_gyro_on_reset: re-measure the gyro bias on every reset (~0.5 s,
                the robot must be still).
            calibration: motor model constants, see :class:`RobotCalibration`.
            safety: thresholds of the safety layer, see :class:`~robotgym.safety.SafetyConfig`;
                None (default) = no safety layer. A trip latches until ``env.safety.clear()``;
                end episodes on it with
                ``terminated_fn=lambda obs, info: info["safety_trip"] is not None``.
            reward_fn: ``f(obs, action, info) -> float``; default 0.0.
            terminated_fn: ``f(obs, info) -> bool``; default never.
            render_mode: ``"rgb_array"`` (full-size camera frame) or ``"human"``
                (OpenCV window; needs opencv-python).
            timeout: seconds to wait for a reply on top of ``action_duration``.
            retries: re-sends (with reconnect) before a step gives up.
            connect_timeout: seconds allowed for connecting and the first reply.
        """
        if camera not in ("stream", "snapshot", "none"):
            raise ValueError(f"camera must be 'stream', 'snapshot' or 'none', not {camera!r}")
        if not 0.0 <= action_duration <= 10.0:
            raise ValueError("action_duration must be between 0 and 10 s")
        if not 0 <= min_pwm <= max_pwm <= 255:
            raise ValueError("need 0 <= min_pwm <= max_pwm <= 255")
        if not 0 <= pwm_deadzone <= 255:
            raise ValueError("pwm_deadzone must be between 0 and 255")
        if not 0.0 <= action_smoothing < 1.0:
            raise ValueError("action_smoothing must be >= 0 and < 1 (1 would never move)")
        if not 0.05 <= hold_timeout <= 60.0:
            raise ValueError("hold_timeout must be between 0.05 and 60 s")
        if render_mode is not None and render_mode not in self.metadata["render_modes"]:
            raise ValueError(f"render_mode must be one of {self.metadata['render_modes']}")

        self.host = host
        self.control_port = control_port
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.camera_mode = camera
        self.http_port = http_port
        self.camera_port = camera_port
        self.camera_framesize = camera_framesize
        self.image_shape = (int(image_shape[0]), int(image_shape[1]))
        self.action_duration = float(action_duration)
        self.max_pwm = int(max_pwm)
        self.min_pwm = int(min_pwm)
        self.pwm_deadzone = int(pwm_deadzone)
        self.action_smoothing = float(action_smoothing)
        self.hold_timeout = float(hold_timeout)
        self.max_distance = float(max_distance)
        self.reset_settle_time = float(reset_settle_time)
        self.calibrate_gyro_on_reset = calibrate_gyro_on_reset
        self.calibration = calibration if calibration is not None else RobotCalibration()
        self.safety: SafetyLayer | None = SafetyLayer(safety) if safety is not None else None
        self.reward_fn = reward_fn
        self.terminated_fn = terminated_fn
        self.render_mode = render_mode
        self.timeout = float(timeout)
        self.retries = int(retries)
        self.connect_timeout = float(connect_timeout)

        obs_spaces: dict[str, spaces.Space] = {}
        if camera != "none":
            obs_spaces["image"] = spaces.Box(0, 255, (*self.image_shape, 3), np.uint8)
        obs_spaces["delta_pose"] = spaces.Box(-np.inf, np.inf, (3,), np.float32)
        obs_spaces["motor_turns"] = spaces.Box(-np.inf, np.inf, (2,), np.float32)
        obs_spaces["distance"] = spaces.Box(0.0, self.max_distance, (1,), np.float32)
        obs_spaces["dt"] = spaces.Box(0.0, np.inf, (1,), np.float32)
        obs_spaces["robot_dt"] = spaces.Box(0.0, np.inf, (1,), np.float32)
        obs_spaces["accel"] = spaces.Box(-np.inf, np.inf, (3,), np.float32)
        imu_rotation(self.calibration)  # reject a bad imu_forward now, not mid-episode
        self.observation_space = spaces.Dict(obs_spaces)
        self.action_space = spaces.Box(-1.0, 1.0, (2,), np.float32)

        self._link: RobotLink | None = None
        self._camera: StreamCamera | SnapshotCamera | None = None
        self._needs_reset = True
        self._packets = 0  # O frames received; a boot flag after the first means a reboot
        self._robot_rebooted = False
        self._prev_state: RobotState | None = None
        self._prev_t = 0.0
        self._last_state: RobotState | None = None
        self._low_battery_warned = False
        self._image_index = -1
        self._image: np.ndarray | None = None
        self._render_index = -1
        self._render_image: np.ndarray | None = None
        self._window_open = False
        self._accel_rest: np.ndarray | None = None  # robot-frame reading at the last reset
        self._accel_v1_warned = False
        self._last_pwm = (0, 0)  # what the robot is running now: the safety checks' baseline
        self._smoothed = np.zeros(2)  # the smoothed [left, right] PWM, before the output deadzone

    # ------------------------------------------------------------------ gym API

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Stops the robot, waits for it to settle, zeroes its odometry.

        Options: ``settle_time`` (s) and ``calibrate_gyro`` (bool) override the
        constructor values for this reset. Repositioning the robot between episodes is
        up to you - e.g. subclass and drive it back, or pause for a human.
        """
        super().reset(seed=seed)
        options = options or {}
        self.connect()
        self._request("A", [0, 0, 0], self.timeout)
        self._last_pwm = (0, 0)
        self._smoothed[:] = 0.0
        settle = float(options.get("settle_time", self.reset_settle_time))
        if settle > 0:
            time.sleep(settle)
        calibrate = bool(options.get("calibrate_gyro", self.calibrate_gyro_on_reset))
        t_sent = time.monotonic()
        state = self._request("R", [int(calibrate)], self.timeout + (1.5 if calibrate else 0.0))
        if calibrate and state.calibration_failed:
            logger.warning("gyro calibration failed (robot moving during reset?); kept the old bias")
        # The R reply averages the accelerometer over the still period: the at-rest reading.
        self._accel_rest = self._accel_robot(state)
        self._prev_state = None
        self._robot_rebooted = False
        self._low_battery_warned = False
        obs, info = self._observe(state, time.monotonic(), image_newer_than=t_sent)
        if self.safety is not None:
            info["safety_trip"] = self.safety.trip  # a reset doesn't clear the latch
        self._needs_reset = False
        if self.render_mode == "human":
            self.render()
        return obs, info

    def step(self, action):
        if self._needs_reset:
            raise gym.error.ResetNeeded("call reset() before step()")
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (2,) or not np.all(np.isfinite(a)):
            raise ValueError(f"action must be two finite numbers, got {action!r}")
        a = np.clip(a, -1.0, 1.0)
        target = self._action_to_pwm(a)  # the deadzone comes after the smoothing, in _smooth
        left, right = self._smooth(target)
        if self.safety is not None:  # judges what the robot would get: zero while latched,
            last = self._last_state  # forward refused near an obstacle
            left, right = self.safety.command((left, right), last.distance if last is not None else None)
            if self.safety.tripped:
                self._smoothed[:] = 0.0  # nothing moves: once cleared, ramp up from rest

        t_sent = time.monotonic()
        duration_ms = round(self.action_duration * 1000)
        state = self._request("A", [left, right, duration_ms], self.action_duration + self.timeout)
        prev_pwm, self._last_pwm = self._last_pwm, (left, right)
        t_reply = time.monotonic()
        obs, info = self._observe(state, t_reply, image_newer_than=t_sent + self.action_duration)
        info["pwm"] = (left, right)
        info["pwm_target"] = target
        info["latency"] = max(0.0, t_reply - t_sent - self.action_duration)
        if self.safety is not None:
            if self.safety.check(prev_pwm, (left, right), obs, info) is not None:
                # Stop now: the robot would keep running this step's action until the next command.
                self._request("A", [0, 0, 0], self.timeout)
                self._last_pwm = (0, 0)
                self._smoothed[:] = 0.0
            info["safety_trip"] = self.safety.trip

        reward = float(self.reward_fn(obs, a, info)) if self.reward_fn else 0.0
        terminated = bool(self.terminated_fn(obs, info)) if self.terminated_fn else False
        if self.render_mode == "human":
            self.render()
        return obs, reward, terminated, False, info

    def render(self):
        """The newest full-size camera frame ("rgb_array"), or shows it in a window ("human").

        In "rgb_array" mode this only reads the camera's newest frame, so a UI thread may
        call it while another thread runs ``step()`` (see examples/manual_control.py).
        """
        if self.render_mode is None:
            gym.logger.warn("render() needs render_mode='rgb_array' or 'human'")
            return None
        frame = self._camera.get() if self._camera is not None else None
        if frame is None:
            return None
        if frame.index != self._render_index:
            self._render_image = decode_jpeg(frame.jpeg)
            self._render_index = frame.index
        if self.render_mode == "rgb_array":
            return self._render_image.copy()
        try:
            import cv2
        except ImportError as exc:
            raise ImportError("render_mode='human' needs opencv-python") from exc
        cv2.imshow("RoboGym", self._render_image[:, :, ::-1])
        cv2.waitKey(1)
        self._window_open = True
        return None

    def close(self):
        if self._link is not None:
            if self._link.alive:
                with suppress(Exception):
                    self._link.request("A", [0, 0, 0], 0.5)
            self._link.close()
            self._link = None
        if self._camera is not None:
            self._camera.stop()
            self._camera = None
        if self._window_open:
            with suppress(Exception):
                import cv2

                cv2.destroyWindow("RoboGym")
            self._window_open = False
        self._needs_reset = True

    # ----------------------------------------------------------- extra helpers

    def connect(self) -> None:
        """Connects to the robot and camera. ``reset()`` calls this; use it to fail fast.

        The camera is on the ESP32 and works even when the Arduino doesn't answer, so it
        is opened whenever the ESP32 is reachable - the live view helps troubleshooting.
        """
        link_error: RobotConnectionError | None = None
        if self._link is None or not self._link.alive:
            try:
                self._open_link()
            except RobotConnectionError as exc:
                link_error = exc
        # Over WiFi, "unreachable" means the ESP32 itself is: no point waiting for its camera.
        esp32_down = isinstance(link_error, _Unreachable) and not self.serial_port
        if self.camera_mode != "none" and self._camera is None and not esp32_down:
            try:
                self._open_camera()
            except RobotConnectionError:
                if link_error is None:
                    raise
        if link_error is not None:
            raise link_error

    @property
    def last_state(self) -> RobotState | None:
        """The latest raw packet (cumulative odometry since reset, battery, flags)."""
        return self._last_state

    # ---------------------------------------------------------------- internals

    def _action_to_pwm(self, a: np.ndarray) -> tuple[int, int]:
        mag = np.abs(a)
        pwm = np.where(mag > 0, self.min_pwm + mag * (self.max_pwm - self.min_pwm), 0.0)
        pwm = np.sign(a) * pwm
        return int(round(float(pwm[0]))), int(round(float(pwm[1])))

    def _deadzone(self, pwm) -> tuple[int, int]:
        """Each track's PWM, or 0 where its magnitude is below ``pwm_deadzone``."""
        return tuple(0 if abs(p) < self.pwm_deadzone else int(p) for p in pwm)

    def _smooth(self, target: tuple[int, int]) -> tuple[int, int]:
        """Advances the exponential average towards ``target``; returns the PWM to send."""
        if self._prev_state is not None and time.monotonic() - self._prev_t > self.hold_timeout:
            self._smoothed[:] = 0.0  # the robot's hold timeout stopped it meanwhile
        k = self.action_smoothing
        self._smoothed = k * self._smoothed + (1.0 - k) * np.asarray(target, dtype=float)
        return self._deadzone(int(v) for v in np.round(self._smoothed))

    def _config_fields(self) -> list[int]:
        c = self.calibration
        return [
            round(self.hold_timeout * 1000),
            round(c.max_rpm * 10),
            round(c.deadband_pwm),
            round(c.time_constant * 1000),
            round(c.sprocket_diameter * 10000),
            round(c.track_width * 1000),
            -1 if c.gyro_sign < 0 else 1,
        ]

    def _open_link(self) -> None:
        if self._link is not None:
            self._link.close()
            self._link = None
        if self.serial_port:
            transport = SerialTransport(self.serial_port, self.baudrate)
        else:
            transport = TcpTransport(self.host, self.control_port, self.connect_timeout)
        link = RobotLink(transport)
        try:
            link.open()
        except OSError as exc:
            raise _Unreachable(
                f"cannot reach the robot at {transport.describe()}: {exc}\n{_TROUBLESHOOTING}"
            ) from exc
        try:
            state = self._handshake(link)
        except BaseException:
            link.close()  # never reuse an unconfigured link: the next attempt starts over
            raise
        self._link = link
        self._after_reply(state, configured=True)
        logger.info("connected to the robot at %s (battery %.2f V)", transport.describe(), state.battery_v)

    def _handshake(self, link: RobotLink) -> RobotState:
        """Sends the calibration (C frame). Retries while the UNO may still be booting
        (opening a USB serial port resets it)."""
        deadline = time.monotonic() + self.connect_timeout
        while True:
            try:
                return link.request("C", self._config_fields(), 0.75)
            except TimeoutError:
                if time.monotonic() >= deadline:
                    raise self._no_reply_error(link) from None
            except ProtocolError as exc:
                raise RobotConnectionError(
                    f"{exc} - is the sketch on the robot from this version of robotgym?"
                ) from exc
            except ConnectionError as exc:
                raise RobotConnectionError(
                    f"lost the robot during the handshake: {exc}\n{_TROUBLESHOOTING}"
                ) from exc

    def _no_reply_error(self, link: RobotLink) -> RobotConnectionError:
        if not link.stock_firmware_seen:
            # ELEGOO's factory sketch answers {"N":100} with {ok}; RoboGym.ino stays silent.
            with suppress(OSError, ConnectionError):
                link.send(b'{"N":100}')
                time.sleep(0.6)
        if link.stock_firmware_seen:
            return RobotConnectionError(
                "the robot runs ELEGOO's factory firmware, not RoboGym: flash "
                "arduino/RoboGym/RoboGym.ino (Upload-Cam switch on 'Upload' while "
                "uploading, then back to 'Cam')"
            )
        if isinstance(link.transport, TcpTransport):
            hint = ("The ESP32 accepted the connection, so the WiFi is fine; the usual cause is "
                    "the shield's Upload-Cam switch still on 'Upload' after flashing - set it to 'Cam'.")
        else:
            hint = "Over USB the shield's Upload-Cam switch must be on 'Upload'."
        return RobotConnectionError(
            f"connected to {link.transport.describe()} but the Arduino did not answer. {hint}\n{_TROUBLESHOOTING}"
        )

    def _open_camera(self) -> None:
        host, port = self.host, self.http_port
        if self.camera_framesize is not None:
            esp32_set(host, port, "framesize", self.camera_framesize)
        if self.camera_mode == "stream":
            url = f"http://{host}:{self.camera_port}/stream"
            # Opening the stream switches the firmware's face detection on; undo it.
            camera = StreamCamera(url, on_connect=lambda: esp32_set(host, port, "face_detect", 0))
        else:
            url = f"http://{host}:{port}/capture"
            esp32_set(host, port, "face_detect", 0)  # in case a browser/app opened the stream
            camera = SnapshotCamera(url)
        camera.start()
        if camera.get(newer_than=float("-inf"), timeout=self.connect_timeout) is None:
            camera.stop()
            raise RobotConnectionError(
                f"no image from the camera at {url} ({camera.last_error}). Is the ESP32 "
                "module plugged in and this computer on its WiFi? camera='none' runs without images."
            )
        self._camera = camera

    def _request(self, kind: str, fields: list[int], timeout: float) -> RobotState:
        """Sends a command, reconnecting and retrying on timeouts or a dropped link."""
        last_exc: BaseException | None = None
        for attempt in range(self.retries + 1):
            try:
                if self._link is None or not self._link.alive:
                    self._open_link()
                state = self._link.request(kind, fields, timeout)
            except ProtocolError as exc:
                raise RobotConnectionError(f"{exc} - sketch and robotgym versions differ?") from exc
            except (TimeoutError, ConnectionError, OSError) as exc:
                last_exc = exc
                logger.warning("robot request %s failed (%s), attempt %d/%d", kind, exc, attempt + 1, self.retries + 1)
                continue
            self._after_reply(state, configured=kind == "C")
            return state
        raise RobotConnectionError(f"the robot stopped responding: {last_exc}\n{_TROUBLESHOOTING}") from last_exc

    def _after_reply(self, state: RobotState, configured: bool) -> None:
        if state.booted and self._packets > 0:
            logger.warning("the robot rebooted (battery brown-out?); its odometry restarted from zero")
            self._robot_rebooted = True
            if not configured:  # it is running the sketch's default calibration now
                try:
                    self._link.request("C", self._config_fields(), self.timeout)
                except (TimeoutError, ConnectionError, ProtocolError) as exc:
                    logger.warning("could not restore the calibration after the reboot: %s", exc)
        self._packets += 1
        self._last_state = state
        if not self._low_battery_warned and 3.0 < state.battery_v < LOW_BATTERY_V:
            logger.warning("robot battery low: %.2f V - charge it soon", state.battery_v)
            self._low_battery_warned = True

    def _observe(self, state: RobotState, t_reply: float, image_newer_than: float) -> tuple[dict, dict]:
        obs: dict[str, np.ndarray] = {}
        info: dict[str, Any] = {}
        if self._camera is not None:
            frame = self._camera.get(newer_than=image_newer_than, timeout=self.IMAGE_WAIT_S)
            obs["image"] = self._obs_image(frame)
            info["image_age"] = time.monotonic() - frame.stamp if frame is not None else None
            info["image_stale"] = frame is None or frame.stamp <= image_newer_than

        prev = self._prev_state
        rebooted = self._robot_rebooted
        if prev is None or rebooted:
            dpose, dturns, dt, robot_dt = (0.0, 0.0, 0.0), (0.0, 0.0), 0.0, 0.0
        else:
            dpose = relative_pose(prev, state)
            dturns = (state.turns_left - prev.turns_left, state.turns_right - prev.turns_right)
            dt = t_reply - self._prev_t
            robot_dt = u32_diff(state.t_us, prev.t_us) / 1e6
        distance = self.max_distance if state.distance is None else min(state.distance, self.max_distance)

        obs["delta_pose"] = np.array(dpose, dtype=np.float32)
        obs["motor_turns"] = np.array(dturns, dtype=np.float32)
        obs["distance"] = np.array([distance], dtype=np.float32)
        obs["dt"] = np.array([max(dt, 0.0)], dtype=np.float32)
        obs["robot_dt"] = np.array([robot_dt], dtype=np.float32)
        accel_raw = self._accel_robot(state)
        if accel_raw is None:
            obs["accel"] = np.zeros(3, dtype=np.float32)
        else:
            rest = self._accel_rest if self._accel_rest is not None else (0.0, 0.0, STANDARD_GRAVITY)
            obs["accel"] = (accel_raw - rest).astype(np.float32)

        info.update(
            accel_raw=None if accel_raw is None else accel_raw.astype(np.float32),
            pose=np.array([state.x, state.y, state.theta], dtype=np.float32),
            motor_turns_total=np.array([state.turns_left, state.turns_right], dtype=np.float32),
            battery_v=state.battery_v,
            distance_valid=state.distance is not None,
            no_echo=state.no_echo,
            imu_ok=state.imu_ok,
            watchdog_stopped=state.watchdog_stopped,
            robot_rebooted=rebooted,
            robot_time=state.t_us / 1e6,
            seq=state.seq,
        )
        self._prev_state = state
        self._prev_t = t_reply
        self._robot_rebooted = False
        return obs, info

    def _accel_robot(self, state: RobotState) -> np.ndarray | None:
        """The packet's mean accelerometer reading in robot axes (gravity included)."""
        if state.accel is None:
            if not self._accel_v1_warned:
                logger.warning("the robot runs an older RoboGym.ino without accelerometer output - "
                               "reflash arduino/RoboGym/RoboGym.ino; 'accel' reads 0 until then")
                self._accel_v1_warned = True
            return None
        if not state.imu_ok:
            return None
        return imu_rotation(self.calibration) @ np.asarray(state.accel, dtype=np.float64)

    def _obs_image(self, frame: Frame | None) -> np.ndarray:
        if frame is not None and frame.index != self._image_index:
            try:
                self._image = decode_jpeg(frame.jpeg, self.image_shape)
                self._image_index = frame.index
            except (OSError, ValueError) as exc:  # a corrupted JPEG: keep the previous one
                logger.warning("could not decode camera frame: %s", exc)
        if self._image is None:
            return np.zeros((*self.image_shape, 3), dtype=np.uint8)
        return self._image.copy()
