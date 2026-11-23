"""Safety layer: stops the robot when its sensors disagree with what it was told to do.

``RoboGym(safety=SafetyConfig())`` checks every step for three things:

1. **stall**: ``x`` is how much the forward command (the mean of the two tracks' PWM)
   changed since the previous step. When ``|x| > stall_offset`` (b, 50), the forward
   acceleration (``accel`` x) must reach ``|stall_slope * (|x| - b)|`` (n, 0.001 m/s² per
   PWM): a bigger change must bring a bigger acceleration, or braking. If it doesn't, the
   speed didn't follow the command: blocked tracks, robot wedged against something. It's
   the mean because forward speed is: one track changing alone moves it half as much, and
   starting a spin (tracks changing in opposite directions) doesn't move it at all, which
   check 2 covers. The manual-control plot *accel x vs ΔPWM* draws this line.
2. **no turn**: either track's |PWM| is above ``pwm_threshold`` (100) and the tracks
   differ by more than ``turn_pwm_diff`` (30), yet the gyro's yaw rate stayed within
   ±``min_yaw_rate``. The robot should be turning and isn't.
3. **obstacle**: the ultrasonic range is below ``min_distance`` (0.15 m) while driving
   forward (left + right PWM > 0). A forward command is also refused outright while the
   last range reading is that close. Reversing and spinning in place stay allowed, so
   the robot can back away from what it nearly hit.

A trip latches: the env stops the robot at once (it would otherwise keep running the
step's action until the next command), and every later step sends zero PWM until
``env.safety.clear()``. ``reset()`` does not clear it.

The checks run on the PC, so they react one step late (~0.2 s) and only as fast as the
link allows. They complement the robot's own hold-timeout stop; they don't replace it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping

logger = logging.getLogger("robotgym")


@dataclass
class SafetyConfig:
    """Thresholds of the safety layer. Tune the sensor ones (``stall_slope`` and
    ``stall_offset``, ``min_yaw_rate``) with the plots in ``examples/manual_control.py``:
    accel x vs ΔPWM (which draws the stall line) and yaw rate vs PWM L-R show what the
    robot really does."""

    stall_offset: float = 50.0  # b: stall is checked once the forward PWM changes by more than this...
    stall_slope: float = 0.001  # n: ...and needs |accel x| >= |n (|change| - b)| [m/s^2 per PWM]
    pwm_threshold: int = 100  # no-turn check applies while either |PWM| is above this
    turn_pwm_diff: int = 30  # no turn: |left - right PWM| above this...
    min_yaw_rate: float = 1.0  # ...but |yaw rate| stayed below this [deg/s]
    min_distance: float = 0.15  # obstacle: closest ultrasonic range allowed while driving forward [m]


    def stall_min_accel(self, change: float) -> float | None:
        """The |accel x| [m/s^2] a forward PWM change of ``change`` must bring, or None
        where the stall check doesn't apply (``|change| <= stall_offset``)."""
        if abs(change) <= self.stall_offset:
            return None
        return abs(self.stall_slope * (abs(change) - self.stall_offset))


@dataclass(frozen=True)
class SafetyTrip:
    """Why the safety layer stopped the robot."""

    trigger: str  # "stall", "no_turn" or "obstacle"
    message: str

    def __str__(self) -> str:
        return self.message


def _forward(pwm: tuple[int, int]) -> bool:
    return pwm[0] + pwm[1] > 0


class SafetyLayer:
    """The checks plus the latch. The env owns one as ``env.safety``. Not thread-safe:
    call it (``clear()`` included) from the thread that steps the env."""

    def __init__(self, config: SafetyConfig | None = None):
        self.config = config if config is not None else SafetyConfig()
        self.trip: SafetyTrip | None = None  # the latched trip; None = armed

    @property
    def tripped(self) -> bool:
        return self.trip is not None

    def clear(self) -> None:
        """Releases the latch; the next step sends the requested PWM again."""
        self.trip = None

    def command(self, pwm: tuple[int, int], distance: float | None) -> tuple[int, int]:
        """The PWM to send instead of the requested ``pwm``: zero while latched.

        ``distance`` is the latest ultrasonic range [m] (None = no echo). Driving
        forward with it below ``min_distance`` trips the obstacle check before anything moves.
        """
        limit = self.config.min_distance
        if self.trip is None and _forward(pwm) and distance is not None and distance < limit:
            self._latch("obstacle", f"OBSTACLE: {distance:.3f} m ahead, forward refused (limit {limit:.2f} m)")
        return (0, 0) if self.trip is not None else (int(pwm[0]), int(pwm[1]))

    def check(self, prev_pwm: tuple[int, int], pwm: tuple[int, int],
              obs: Mapping[str, Any], info: Mapping[str, Any]) -> SafetyTrip | None:
        """Checks a finished step: ``pwm`` was sent after ``prev_pwm`` and produced the env's
        ``obs`` and ``info``. Latches and returns the trip if one fired; otherwise None."""
        if self.trip is not None:
            return None
        c = self.config
        left, right = pwm

        distance = float(obs["distance"][0])
        if _forward(pwm) and distance < c.min_distance:
            return self._latch("obstacle", f"OBSTACLE: {distance:.3f} m ahead while driving forward "
                                           f"(limit {c.min_distance:.2f} m)")

        # The motion checks need the IMU, and a real interval to measure over.
        robot_dt = float(obs["robot_dt"][0])
        if (not info.get("imu_ok", False) or info.get("accel_raw") is None
                or info.get("robot_rebooted", False) or robot_dt <= 0.0):
            return None
        if info.get("watchdog_stopped", False):
            prev_pwm = (0, 0)  # the robot's hold timeout stopped it: it starts from rest
        change = ((left - prev_pwm[0]) + (right - prev_pwm[1])) / 2  # of the forward command
        need = c.stall_min_accel(change)
        accel_x = float(obs["accel"][0])
        if need is not None and abs(accel_x) < need:
            return self._latch("stall", f"STALL: forward PWM changed by {change:+g} but accel x was "
                                        f"{accel_x:+.3f} m/s² (needs ±{need:.2f})")

        if max(abs(left), abs(right)) <= c.pwm_threshold:
            return None
        diff = left - right
        yaw_rate = math.degrees(float(obs["delta_pose"][2]) / robot_dt)
        if abs(diff) > c.turn_pwm_diff and abs(yaw_rate) < c.min_yaw_rate:
            return self._latch("no_turn", f"NO TURN: PWM left - right {diff:+d} but yaw rate was "
                                          f"{yaw_rate:+.2f} deg/s (needs ±{c.min_yaw_rate:.1f})")
        return None

    def _latch(self, trigger: str, message: str) -> SafetyTrip:
        logger.warning("safety stop - %s; the robot stays stopped until safety.clear()", message)
        self.trip = SafetyTrip(trigger, message)
        return self.trip
