"""Drive the robot by hand through the RoboGym env, with a live camera feed.

    python examples/manual_control.py           # the real robot (PC on its "ELEGOO-xxxx" WiFi)
    python examples/manual_control.py --fake    # simulated robot, no hardware

Three levers set the same two track speeds:
  * the joystick: up/down = forward/reverse, left/right = turn (both tracks at once);
  * the "left" and "right" levers: one track each. They also show what each track is
    getting from the joystick, and the joystick follows them.
Let go of a lever and it springs back, which stops the robot - the joystick stops both
tracks, a side lever its own track. Tick "Hold levers" to make them stay where you leave
them, e.g. to drive tank-style with the two side levers. The STOP button or Space always
stops; Esc, closing the window or Ctrl+C in the terminal stops the robot and quits.

The env's safety layer is on (``--no-safety`` turns it off). When it stops the robot (a
stall, a turn that doesn't happen, an obstacle ahead) the panel under the levers latches
red with the reason, the levers centre, and the robot stays stopped until you press its
"Clear" button.

The window shows the live camera feed, the image the agent receives in its observation,
and the values of the latest observation. A second window (also opened by the "Plots"
button) plots the PWM sent to each track and the accelerometer over the last 20 s, and
scatters each accelerometer axis against the PWM and against the change in PWM since the
previous step (ΔPWM), and the gyro's yaw rate against the left - right PWM difference
(and their changes), one dot per env step. The env runs in a background thread that steps
as fast as the robot answers (~5 Hz) with whatever the levers say, so the window never
waits for the robot.

Needs tkinter, which comes with Python (Debian/Ubuntu: sudo apt install python3-tk).
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import logging
import math
import signal
import threading
import time
import tkinter as tk
from collections import deque
from dataclasses import dataclass, field
from tkinter import ttk

import numpy as np
from PIL import Image

from robotgym import RoboGym, RobotConnectionError, SafetyConfig, SafetyTrip
from robotgym.fake_robot import FakeRobotServer

logger = logging.getLogger("manual_control")

CAMERA_BOX = (640, 480)  # the live feed is scaled to fit this (width, height)
OBS_ZOOM = 2  # the agent's image is shown enlarged, nearest-neighbour
UI_PERIOD_MS = 40  # window refresh, ~25 fps
UI_STALE_S = 1.0  # the env thread stops the robot if the window stops refreshing
RETRY_S = 2.0  # wait between connection attempts
_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


def arcade_mix(x: float, y: float) -> tuple[float, float]:
    """Lever position (x = right, y = forward, each -1..1) -> (left, right) track speeds.

    Straight up drives forward, straight right spins clockwise on the spot, diagonals
    pivot around one track. Every [left, right] action is reachable.
    """
    left, right = y + x, y - x
    scale = max(1.0, abs(left), abs(right))
    return left / scale, right / scale


def to_photo(master: tk.Misc, rgb: np.ndarray) -> tk.PhotoImage:
    """uint8 RGB array -> Tk image, through an in-memory PPM (no PIL.ImageTk needed)."""
    height, width = rgb.shape[:2]
    data = b"P6 %d %d 255 " % (width, height) + np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
    return tk.PhotoImage(master=master, width=width, height=height, data=data, format="PPM")


@dataclass
class StepResult:
    obs: dict
    info: dict
    action: np.ndarray
    count: int  # increments with every step and reset
    rate: float  # steps per second, smoothed
    stamp: float = field(default_factory=time.monotonic)  # when the step finished


def format_readout(r: StepResult) -> str:
    o, i = r.obs, r.info
    dx, dy, dth = (float(v) for v in o["delta_pose"])
    turns_l, turns_r = (float(v) for v in o["motor_turns"])
    ax, ay, az = (float(v) for v in o["accel"])
    x, y, th = (float(v) for v in i["pose"])
    pwm_l, pwm_r = i.get("pwm", (0, 0))
    age = i.get("image_age")
    flags = [name for name, on in (
        ("SAFETY STOP", i.get("safety_trip") is not None),
        ("IMU NOT OK", not i["imu_ok"]),
        ("watchdog stop", i["watchdog_stopped"]),
        ("ROBOT REBOOTED", i["robot_rebooted"]),
        ("no distance reading", not i["distance_valid"]),
        ("image stale", i.get("image_stale", False)),
        ("LOW BATTERY", 3.0 < i["battery_v"] < 7.0),
    ) if on]
    return "\n".join([
        f"distance   {float(o['distance'][0]):5.2f} m     battery {i['battery_v']:5.2f} V"
        + (f"     image age {age:.2f} s" if age is not None else ""),
        f"delta_pose dx {dx:+.4f} m   dy {dy:+.4f} m   dtheta {dth:+.4f} rad",
        f"turns      left {turns_l:+.3f}   right {turns_r:+.3f}   (estimated)",
        f"accel      x {ax:+.2f}   y {ay:+.2f}   z {az:+.2f} m/s^2   (mean over the step)",
        f"dt         {float(o['dt'][0]):.3f} s   robot_dt {float(o['robot_dt'][0]):.3f} s"
        f"   latency {i.get('latency', 0.0):.3f} s",
        f"pose       x {x:+.3f} m   y {y:+.3f} m   theta {th:+.3f} rad   (since reset)",
        f"action     left {r.action[0]:+.2f}   right {r.action[1]:+.2f}"
        f"   PWM {pwm_l:+d} / {pwm_r:+d}   {r.rate:.1f} steps/s",
        "flags      " + (", ".join(flags) if flags else "-"),
    ])


class Shared:
    """What the window and the env thread exchange. Every access holds the lock."""

    def __init__(self, max_speed: float):
        self._lock = threading.Lock()
        self._command = (0.0, 0.0)  # [left, right] from the levers, -1..1, before max speed
        self._max_speed = max_speed
        self._ui_alive = time.monotonic()
        self._reset_requested = False
        self._result: StepResult | None = None
        self._status = ("starting...", False)
        self._safety_trip: SafetyTrip | None = None  # the env's latched trip, as last seen
        self._safety_clear_requested = False
        # Every step, for the plots: (time, pwm_left, pwm_right, accel_x, accel_y, accel_z,
        # yaw_rate) - yaw rate in deg/s, counter-clockwise positive
        self._history: deque[tuple[float, ...]] = deque(maxlen=5000)

    def set_command(self, left: float, right: float) -> None:
        with self._lock:
            self._command = (left, right)

    def set_max_speed(self, speed: float) -> None:
        with self._lock:
            self._max_speed = speed

    def touch(self) -> None:
        """Called on every window refresh."""
        with self._lock:
            self._ui_alive = time.monotonic()

    def action(self) -> np.ndarray:
        """The levers as an env action; zero if the window has stopped refreshing."""
        with self._lock:
            command, speed, alive = self._command, self._max_speed, self._ui_alive
        if time.monotonic() - alive > UI_STALE_S:
            return np.zeros(2, dtype=np.float32)
        return np.array(command, dtype=np.float32) * speed

    def request_reset(self) -> None:
        with self._lock:
            self._reset_requested = True

    def take_reset_request(self) -> bool:
        with self._lock:
            requested, self._reset_requested = self._reset_requested, False
        return requested

    def request_safety_clear(self) -> None:
        with self._lock:
            self._safety_clear_requested = True

    def safety_clear_requested(self) -> bool:
        with self._lock:
            return self._safety_clear_requested

    def safety_cleared(self) -> None:
        """The env thread cleared the latch. One update, so the window never sees the
        request gone while the old trip is still there."""
        with self._lock:
            self._safety_clear_requested = False
            self._safety_trip = None

    def safety(self) -> tuple[SafetyTrip | None, bool]:
        """(latched trip or None, clear requested but not done yet)."""
        with self._lock:
            return self._safety_trip, self._safety_clear_requested

    def publish(self, result: StepResult) -> None:
        pwm_left, pwm_right = result.info.get("pwm", (0, 0))  # resets send no PWM
        accel = result.obs.get("accel", (0.0, 0.0, 0.0))
        # The gyro's mean output over the step: measured heading change / step duration.
        dtheta = float(result.obs["delta_pose"][2])
        robot_dt = float(result.obs["robot_dt"][0])
        yaw_rate = math.degrees(dtheta / robot_dt) if robot_dt > 0 else 0.0
        row = (result.stamp, float(pwm_left), float(pwm_right), *(float(a) for a in accel), yaw_rate)
        with self._lock:
            self._result = result
            self._history.append(row)
            if "safety_trip" in result.info:  # absent without a safety layer
                self._safety_trip = result.info["safety_trip"]

    def history(self) -> list[tuple[float, ...]]:
        with self._lock:
            return list(self._history)

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def latest(self) -> StepResult | None:
        with self._lock:
            return self._result

    def set_status(self, text: str, error: bool = False) -> None:
        with self._lock:
            self._status = (text, error)

    def status(self) -> tuple[str, bool]:
        with self._lock:
            return self._status


class EnvRunner(threading.Thread):
    """Owns the env: resets it, then steps it with the lever position until stopped."""

    def __init__(self, env: RoboGym, shared: Shared):
        super().__init__(name="env-runner", daemon=True)
        self.env = env
        self.shared = shared
        self.stop_event = threading.Event()
        self._count = 0

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                self._clear_safety_if_requested()  # also while the robot is unreachable
                try:
                    self._session()
                except RobotConnectionError as exc:
                    logger.warning("%s", exc)
                    first_line = str(exc).splitlines()[0]
                    self.shared.set_status(f"{first_line} - retrying in {RETRY_S:.0f} s", error=True)
                    self.stop_event.wait(RETRY_S)
        except Exception as exc:  # a bug: show it in the window instead of dying silently
            logger.exception("env thread crashed")
            self.shared.set_status(f"env thread crashed: {exc!r}", error=True)
        finally:
            self.env.close()  # stops the robot

    def _session(self) -> None:
        self.shared.set_status("connecting to the robot...")
        self._reset()
        self.shared.set_status("connected")
        last, rate = time.monotonic(), 0.0
        while not self.stop_event.is_set():
            if self.shared.take_reset_request():
                self._reset()
                last = time.monotonic()
                continue
            self._clear_safety_if_requested()
            action = self.shared.action()
            obs, _, terminated, truncated, info = self.env.step(action)
            now = time.monotonic()
            step_rate = 1.0 / max(now - last, 1e-6)
            rate = step_rate if rate == 0.0 else 0.8 * rate + 0.2 * step_rate
            last = now
            self._publish(obs, info, action, rate)
            if terminated or truncated:
                self._reset()

    def _clear_safety_if_requested(self) -> None:
        """The latch is cleared here, on the thread that steps the env, as SafetyLayer requires."""
        if self.shared.safety_clear_requested():
            if self.env.safety is not None:
                self.env.safety.clear()
            self.shared.safety_cleared()

    def _reset(self) -> None:
        obs, info = self.env.reset()
        self._publish(obs, info, np.zeros(2, dtype=np.float32), 0.0)

    def _publish(self, obs: dict, info: dict, action: np.ndarray, rate: float) -> None:
        self._count += 1
        self.shared.publish(StepResult(obs, info, action, self._count, rate))


LEVER_IDLE, LEVER_ACTIVE = "#2f6fd6", "#5b93f2"


class Joystick(tk.Canvas):
    """Joystick-style lever: ``on_move(x, y)`` (x = right, y = forward, each -1..1) while
    dragged, ``on_release()`` when let go. The app decides what a release does."""

    def __init__(self, master: tk.Misc, size: int, on_move, on_release, bg: str | None = None):
        super().__init__(master, width=size, height=size, highlightthickness=0, bg=bg)
        self.on_move = on_move
        self.on_release = on_release
        self.position = (0.0, 0.0)  # where the knob is drawn
        self.k = 20  # knob radius
        self.c = size / 2
        self.r = size / 2 - self.k - 19  # travel of the knob's centre; keeps clear of the labels
        c, r = self.c, self.r
        self.create_oval(c - r, c - r, c + r, c + r, fill="#eef1f5", outline="#9aa5b1", width=2)
        self.create_line(c - r, c, c + r, c, fill="#c8d0d8")
        self.create_line(c, c - r, c, c + r, fill="#c8d0d8")
        for text, x, y in (("FWD", c, 9), ("REV", c, size - 9), ("L", 9, c), ("R", size - 9, c)):
            self.create_text(x, y, text=text, fill="#5f6b7a", font=("TkDefaultFont", 9, "bold"))
        self.stem = self.create_line(c, c, c, c, fill="#7b8794", width=7, capstyle="round")
        self.knob = self.create_oval(0, 0, 0, 0, fill=LEVER_IDLE, outline="#1d4f9e", width=2)
        self.show(0.0, 0.0)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", lambda _e: self.release())

    def set(self, x: float, y: float) -> None:
        """As if dragged to (x, y)."""
        self.show(x, y)
        self.on_move(x, y)

    def release(self) -> None:
        """As if let go."""
        self.itemconfigure(self.knob, fill=LEVER_IDLE)
        self.on_release()

    def show(self, x: float, y: float) -> None:
        """Draws the knob at (x, y) without reporting a move."""
        c, r, k = self.c, self.r, self.k
        px, py = c + x * r, c - y * r
        self.coords(self.stem, c, c, px, py)
        self.coords(self.knob, px - k, py - k, px + k, py + k)
        self.position = (x, y)

    def _press(self, event) -> None:
        self.itemconfigure(self.knob, fill=LEVER_ACTIVE)
        self._drag(event)

    def _drag(self, event) -> None:
        dx, dy = event.x - self.c, event.y - self.c
        dist = math.hypot(dx, dy)
        if dist > self.r:
            dx, dy = dx * self.r / dist, dy * self.r / dist
        self.set(dx / self.r, -dy / self.r)


class TrackLever(tk.Canvas):
    """Vertical lever for one track, up = forward: ``on_move(value)`` (-1..1) while
    dragged, ``on_release()`` when let go. It also displays the track's command, whichever
    lever set it, labelled with the speed actually sent."""

    SNAP = 0.05  # this close to the middle counts as exactly 0: that track stopped

    def __init__(self, master: tk.Misc, name: str, height: int, on_move, on_release,
                 bg: str | None = None):
        super().__init__(master, width=58, height=height, highlightthickness=0, bg=bg)
        self.on_move = on_move
        self.on_release = on_release
        self.cx = 29
        self.top, self.bottom = 28, height - 28
        self.mid = (self.top + self.bottom) / 2
        self.half = (self.bottom - self.top) / 2
        cx = self.cx
        self.create_text(cx, 11, text=name, fill="#3c4650", font=("TkDefaultFont", 9, "bold"))
        self.create_rectangle(cx - 7, self.top, cx + 7, self.bottom, fill="#eef1f5", outline="#9aa5b1")
        self.bar = self.create_rectangle(cx - 6, self.mid, cx + 6, self.mid, fill="#3aa757", outline="")
        self.create_line(cx - 13, self.mid, cx + 13, self.mid, fill="#5f6b7a")
        self.handle = self.create_rectangle(0, 0, 0, 0, fill=LEVER_IDLE, outline="#1d4f9e", width=2)
        self.value_text = self.create_text(cx, height - 11, text="+0.00", fill="#3c4650",
                                           font=("TkFixedFont", 9))
        self.show(0.0, 0.0)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", lambda _e: self.release())

    def set(self, value: float) -> None:
        """As if dragged to ``value``."""
        self.on_move(value)

    def release(self) -> None:
        """As if let go."""
        self.itemconfigure(self.handle, fill=LEVER_IDLE)
        self.on_release()

    def show(self, value: float, sent: float) -> None:
        """Draws the lever at ``value``, labelled with the speed ``sent`` to the robot."""
        cx, y = self.cx, self.mid - value * self.half
        self.coords(self.bar, cx - 6, min(y, self.mid), cx + 6, max(y, self.mid))
        self.itemconfigure(self.bar, fill="#3aa757" if value >= 0 else "#e8833a")
        self.coords(self.handle, cx - 17, y - 7, cx + 17, y + 7)
        self.itemconfigure(self.value_text, text=f"{sent:+.2f}")

    def _press(self, event) -> None:
        self.itemconfigure(self.handle, fill=LEVER_ACTIVE)
        self._drag(event)

    def _drag(self, event) -> None:
        value = max(-1.0, min(1.0, (self.mid - event.y) / self.half))
        self.set(0.0 if abs(value) < self.SNAP else value)


class SafetyLatch(tk.Frame):
    """The env's safety layer: quiet while armed; once it trips, a red latch with the reason
    and a "Clear" button. Every state has the same size, so the window never resizes."""

    RED, RED_DARK, GREEN, GREY = "#d93025", "#b3261e", "#1e8e3e", "#5f6b7a"

    def __init__(self, master: tk.Misc, on_clear, bg: str, wrap: int = 320):
        """``wrap``: the reason's line width [px] until the panel knows its own width."""
        super().__init__(master, padx=8, pady=5, highlightthickness=1)
        self.bg = bg
        self.title = tk.Label(self, anchor="w", font=("TkDefaultFont", 10, "bold"))
        self.button = tk.Button(self, text="Clear", width=9, command=on_clear, takefocus=False,
                                font=("TkDefaultFont", 9, "bold"), pady=1)
        self.detail = tk.Label(self, anchor="nw", justify="left", height=2, wraplength=wrap,
                               font=("TkDefaultFont", 9))
        self.title.grid(row=0, column=0, sticky="w")
        self.button.grid(row=0, column=1, sticky="e", padx=(8, 0))
        self.detail.grid(row=1, column=0, columnspan=2, sticky="w", pady=(3, 0))
        self.columnconfigure(0, weight=1)
        # Wrap the reason at the panel's width, which the widgets around it decide.
        self.bind("<Configure>", lambda e: self.detail.configure(wraplength=max(wrap, e.width - 20)))

    def show_off(self) -> None:
        self._paint(False, "Safety layer off", "The env has no safety layer (--no-safety): only you and "
                    "the robot's hold timeout stop it.", self.GREY, "Clear", enabled=False)

    def show_armed(self, config: SafetyConfig) -> None:
        self._paint(False, "\u25cf Safety armed",
                    f"Stops the robot on a stall, a turn that doesn't happen, or an obstacle "
                    f"closer than {config.min_distance:.2f} m ahead.", self.GREEN, "Clear", enabled=False)

    def show_tripped(self, trip: SafetyTrip, clearing: bool) -> None:
        self._paint(True, "SAFETY STOP - robot held stopped", trip.message, "white",
                    "Clearing\u2026" if clearing else "Clear", enabled=not clearing)

    @property
    def latched(self) -> bool:
        return self.cget("bg") == self.RED

    def _paint(self, red: bool, title: str, detail: str, title_fg: str, button: str, enabled: bool) -> None:
        bg = self.RED if red else self.bg
        self.configure(bg=bg, highlightbackground=self.RED_DARK if red else "#c8d0d8")
        self.title.configure(text=title, bg=bg, fg=title_fg)
        self.detail.configure(text=detail, bg=bg, fg="white" if red else self.GREY)
        self.button.configure(text=button, state="normal" if enabled else "disabled",
                              bg="white" if red else self.bg, fg=self.RED_DARK,
                              activebackground="#fde8e7", activeforeground=self.RED_DARK,
                              disabledforeground=_blend(self.RED, 0.45) if red else "#a0a8b0",
                              highlightbackground=bg)


PLOT_COLORS = {"left": "#2f6fd6", "right": "#e8833a", "x": "#d93025", "y": "#1e8e3e", "z": "#8e44ad",
               "yaw": "#00897b"}
# Distinct shapes keep coinciding points visible (e.g. left = right PWM when driving straight).
PLOT_MARKERS = {"left": "dot", "right": "ring", "x": "dot", "y": "square", "z": "ring", "yaw": "dot"}
_MARKER_LAYER = {"square": 0, "dot": 1, "ring": 2}  # big filled first, rings on top


def nice_ticks(lo: float, hi: float, target: int = 6) -> list[float]:
    """Round tick values (1, 2 or 5 x 10^k apart) covering [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    magnitude = 10.0 ** math.floor(math.log10(span / target))
    step = next(m * magnitude for m in (1, 2, 5, 10) if span / (m * magnitude) <= target)
    k = math.ceil(lo / step - 1e-9)
    ticks = []
    while k * step <= hi + 1e-9 * step:
        ticks.append(k * step + 0.0)  # integer multiples: no accumulated rounding; +0.0 kills -0
        k += 1
    return ticks


def _tick_labels(ticks: list[float]) -> list[str]:
    step = ticks[1] - ticks[0] if len(ticks) > 1 else 1.0
    decimals = max(0, -math.floor(math.log10(step) + 1e-9))
    return [f"{t:.{decimals}f}" for t in ticks]


def _blend(color: str, amount: float) -> str:
    """Mixes a #rrggbb colour with white."""
    rgb = (int(color[i:i + 2], 16) for i in (1, 3, 5))
    return "#%02x%02x%02x" % tuple(round(c + (255 - c) * amount) for c in rgb)


class Plot(tk.Canvas):
    """A small live plot: auto-scaled axes with round ticks and one marker per point,
    optionally joined by a faint line. Cheap enough to redraw at the step rate."""

    LEFT, RIGHT, TOP, BOTTOM = 52, 12, 26, 34

    def __init__(self, master: tk.Misc, width: int, height: int, title: str, xlabel: str,
                 ylabel: str, series: list[tuple[str, str, str]], *, xlim=None, ylim=None,
                 min_span: float = 1.0, join: bool = False):
        super().__init__(master, width=width, height=height, bg="white", highlightthickness=1,
                         highlightbackground="#c8d0d8")
        self.series = series  # [(name, colour, marker)], marker: "dot", "square" or "ring"
        self.xlim, self.ylim, self.min_span, self.join = xlim, ylim, min_span, join
        self.box = (self.LEFT, self.TOP, width - self.RIGHT, height - self.BOTTOM)
        x0, y0, x1, y1 = self.box
        self.create_text(x0, 13, text=title, anchor="w", fill="#1f2933", font=("TkDefaultFont", 9, "bold"))
        x = x1
        for name, color, marker in reversed(series):  # legend, right-aligned
            label = self.create_text(x, 13, text=name, anchor="e", fill="#3c4650", font=("TkDefaultFont", 8))
            x = self.bbox(label)[0] - 8
            self._marker(x, 13, color, marker, tags=())
            x -= 10
        self.create_text((x0 + x1) / 2, height - 7, text=xlabel, fill="#5f6b7a", font=("TkDefaultFont", 8))
        self.create_text(11, (y0 + y1) / 2, text=ylabel, angle=90, fill="#5f6b7a", font=("TkDefaultFont", 8))
        self.redraw({})

    def redraw(self, data: dict, xlim=None) -> None:
        """Redraws axes and points. ``data`` maps a series name to (xs, ys)."""
        self.delete("dyn")
        xlim = xlim or self.xlim or self._limits([v[0] for v in data.values()])
        ylim = self.ylim or self._limits([v[1] for v in data.values()])
        x0, y0, x1, y1 = self.box
        sx = (x1 - x0) / (xlim[1] - xlim[0])
        sy = (y1 - y0) / (ylim[1] - ylim[0])
        small = ("TkDefaultFont", 8)
        ticks = nice_ticks(*xlim)
        for v, text in zip(ticks, _tick_labels(ticks)):
            px = x0 + (v - xlim[0]) * sx
            self.create_line(px, y0, px, y1, fill="#eef1f5", tags="dyn")
            self.create_text(px, y1 + 9, text=text, fill="#5f6b7a", font=small, tags="dyn")
        ticks = nice_ticks(*ylim)
        for v, text in zip(ticks, _tick_labels(ticks)):
            py = y1 - (v - ylim[0]) * sy
            self.create_line(x0, py, x1, py, fill="#eef1f5", tags="dyn")
            self.create_text(x0 - 4, py, text=text, anchor="e", fill="#5f6b7a", font=small, tags="dyn")
        if ylim[0] < 0 < ylim[1]:
            py = y1 + ylim[0] * sy
            self.create_line(x0, py, x1, py, fill="#9aa5b1", tags="dyn")
        self.create_rectangle(x0, y0, x1, y1, outline="#9aa5b1", tags="dyn")
        for name, color, marker in sorted(self.series, key=lambda s: _MARKER_LAYER[s[2]]):
            if name not in data:
                continue
            xs = np.clip(x0 + (np.asarray(data[name][0], float) - xlim[0]) * sx, x0, x1)
            ys = np.clip(y1 - (np.asarray(data[name][1], float) - ylim[0]) * sy, y0, y1)
            if self.join and len(xs) > 1:
                self.create_line(*np.column_stack((xs, ys)).ravel().tolist(),
                                 fill=_blend(color, 0.6), tags="dyn")
            for px, py in zip(xs.tolist(), ys.tolist()):
                self._marker(px, py, color, marker, tags=("dyn", "point"))

    def _marker(self, x: float, y: float, color: str, marker: str, tags) -> None:
        if marker == "ring":
            self.create_oval(x - 4, y - 4, x + 4, y + 4, outline=color, width=1.5, tags=tags)
        elif marker == "square":
            self.create_rectangle(x - 3, y - 3, x + 3, y + 3, fill=color, outline="", tags=tags)
        else:
            self.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5, fill=color, outline="", tags=tags)

    def _limits(self, arrays) -> tuple[float, float]:
        values = np.concatenate([np.asarray(a, float) for a in arrays]) if arrays else np.empty(0)
        values = values[np.isfinite(values)]
        lo = min(float(values.min()), 0.0) if values.size else 0.0  # keep 0 in view
        hi = max(float(values.max()), 0.0) if values.size else 0.0
        if hi - lo < self.min_span:
            mid = (hi + lo) / 2
            lo, hi = mid - self.min_span / 2, mid + self.min_span / 2
        pad = 0.08 * (hi - lo)
        return lo - pad, hi + pad


class PlotWindow:
    """Second window: the PWM sent and the accelerometer over a rolling time window; each
    accelerometer axis against the PWM of both tracks and against the change in PWM since
    the previous step; and, for turning, the gyro's yaw rate against the left - right PWM
    difference and its change against the change in that difference."""

    MAX_STEP_GAP_S = 1.0  # a longer gap (pause, reconnect, reset) has no meaningful PWM change

    def __init__(self, root: tk.Tk, seconds: float, on_close, on_clear):
        self.seconds = seconds
        self.window = tk.Toplevel(root)
        self.window.title("RoboGym plots - PWM and accelerometer")
        self.window.resizable(False, False)
        self.window.protocol("WM_DELETE_WINDOW", on_close)
        frame = ttk.Frame(self.window, padding=8)
        frame.grid(row=0, column=0)
        tracks = [(name, PLOT_COLORS[name], PLOT_MARKERS[name]) for name in ("left", "right")]
        axes = [(name, PLOT_COLORS[name], PLOT_MARKERS[name]) for name in "xyz"]
        yaw = [("yaw", PLOT_COLORS["yaw"], PLOT_MARKERS["yaw"])]
        # Four rows sized to fit a 768-pixel-high screen; the 4th column is for turning.
        self.pwm_plot = Plot(frame, 1040, 140, "Motor PWM sent", "time (s)", "PWM", tracks,
                             ylim=(-275, 275), join=True)
        self.accel_plot = Plot(frame, 1040, 140, "Accelerometer (mean over each step, robot frame)",
                               "time (s)", "m/s²", axes, join=True)
        self.scatters = [Plot(frame, 254, 180, f"accel {a} vs PWM", "PWM", f"accel {a} (m/s²)",
                              tracks, xlim=(-275, 275)) for a in "xyz"]
        self.change_scatters = [Plot(frame, 254, 180, f"accel {a} vs ΔPWM",
                                     "ΔPWM since the previous step", f"accel {a} (m/s²)",
                                     tracks, xlim=(-540, 540)) for a in "xyz"]
        # Counter-clockwise yaw is positive, so it comes with right > left: a falling slope.
        self.turn_scatter = Plot(frame, 254, 180, "yaw rate vs PWM L-R", "PWM left - right",
                                 "yaw rate (deg/s)", yaw, xlim=(-540, 540), min_span=10.0)
        self.turn_change_scatter = Plot(frame, 254, 180, "Δyaw rate vs Δ(L-R)", "Δ(PWM left - right)",
                                        "Δyaw rate (deg/s)", yaw, xlim=(-1080, 1080), min_span=10.0)
        self.pwm_plot.grid(row=0, column=0, columnspan=4)
        self.accel_plot.grid(row=1, column=0, columnspan=4, pady=(5, 0))
        for row, plots in ((2, [*self.scatters, self.turn_scatter]),
                           (3, [*self.change_scatters, self.turn_change_scatter])):
            for i, plot in enumerate(plots):
                plot.grid(row=row, column=i, padx=(0 if i == 0 else 6, 0), pady=(5, 0))
        controls = ttk.Frame(frame)
        controls.grid(row=4, column=0, columnspan=4, sticky="w", pady=(5, 0))
        ttk.Button(controls, text="Clear", command=on_clear, takefocus=False).pack(side="left")
        self.info = tk.StringVar()
        ttk.Label(controls, textvariable=self.info, foreground="#5f6b7a").pack(side="left", padx=10)
        self.window.after(300, lambda: self._place_beside(root))

    def redraw(self, history, now: float) -> None:
        """``history`` rows: (time, pwm_left, pwm_right, accel_x, accel_y, accel_z, yaw_rate)."""
        everything = np.array(history, dtype=float).reshape(-1, 7)
        # Changes since the previous step (PWM left, PWM right, yaw rate). A PWM change
        # takes effect at the start of a step, so it pairs with that step's response.
        # Undefined for the first step and after a gap (pause, reconnect, reset).
        change = np.full((len(everything), 3), np.nan)
        change[1:] = np.diff(everything[:, [1, 2, 6]], axis=0)
        change[1:][np.diff(everything[:, 0]) > self.MAX_STEP_GAP_S] = np.nan

        recent = now - everything[:, 0] <= self.seconds
        rows, change = everything[recent], change[recent]
        t = rows[:, 0] - now
        pwm_l, pwm_r, ax, ay, az, yaw = rows[:, 1:].T
        window = (-self.seconds, 0.0)
        self.pwm_plot.redraw({"left": (t, pwm_l), "right": (t, pwm_r)}, xlim=window)
        self.accel_plot.redraw({"x": (t, ax), "y": (t, ay), "z": (t, az)}, xlim=window)
        ok = ~np.isnan(change[:, 0])
        d_left, d_right, d_yaw = change[ok].T
        for accel, plot, change_plot in zip((ax, ay, az), self.scatters, self.change_scatters):
            plot.redraw({"left": (pwm_l, accel), "right": (pwm_r, accel)})
            change_plot.redraw({"left": (d_left, accel[ok]), "right": (d_right, accel[ok])})
        self.turn_scatter.redraw({"yaw": (pwm_l - pwm_r, yaw)})
        self.turn_change_scatter.redraw({"yaw": (d_left - d_right, d_yaw)})
        self.info.set(f"{len(rows)} steps in the last {self.seconds:.0f} s - one dot per env step")

    def _place_beside(self, root: tk.Tk) -> None:
        """Next to the main window if the screen has room; otherwise the window manager decides."""
        x = root.winfo_x() + root.winfo_width() + 10
        if x + self.window.winfo_width() <= root.winfo_screenwidth():
            self.window.geometry(f"+{x}+{root.winfo_y()}")


class ManualControlApp:
    """The window. Only this (main) thread touches Tk; the env lives in an EnvRunner."""

    def __init__(self, root: tk.Tk, env: RoboGym, max_speed: float = 0.6, show_plots: bool = True,
                 plot_seconds: float = 20.0):
        self.root = root
        self.env = env
        self.shared = Shared(max_speed)
        self.runner = EnvRunner(env, self.shared)
        self.closing = False
        self.steps_shown = 0
        self.camera_frames = 0
        self._camera_probe: np.ndarray | None = None
        self._camera_fps = 0.0
        self._camera_t: float | None = None
        self._camera_photo: tk.PhotoImage | None = None
        self._obs_photo: tk.PhotoImage | None = None
        self._command = (0.0, 0.0)  # [left, right], -1..1: what all three levers show
        self._max_speed = max_speed
        self._tick_error: str | None = None
        self._safety_trip: SafetyTrip | None = None  # the latched trip the window has seen
        self._latch_shown: tuple | None = None  # what the latch panel shows now
        self.plot_seconds = plot_seconds
        self.plots: PlotWindow | None = None
        self._plots_dirty = False
        self._plots_drawn_at = 0.0
        self._build(max_speed)
        if show_plots:
            self.open_plots()

    def open_plots(self) -> PlotWindow:
        """Opens the plot window, or brings it to the front if it is already open."""
        if self.plots is None:
            self.plots = PlotWindow(self.root, self.plot_seconds, self._close_plots, self.clear_history)
            self._plots_dirty = True
        else:
            self.plots.window.deiconify()
            self.plots.window.lift()
        return self.plots

    def clear_history(self) -> None:
        self.shared.clear_history()
        self._plots_dirty = True

    def _close_plots(self) -> None:
        if self.plots is not None:
            self.plots.window.destroy()
            self.plots = None

    def start(self) -> None:
        self.runner.start()
        self._tick()

    def run(self) -> None:
        """start() and run the window until it closes. Ctrl+C quits cleanly too."""
        # Without this, Ctrl+C either escapes mainloop mid-step or gets swallowed by
        # Tk's callback wrapper. Instead: stop the robot and close like the window does.
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        for sig in previous:
            signal.signal(sig, lambda *_: self.root.after(0, self.quit))
        self.start()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit()
        finally:  # the handlers reference this app; don't keep it (and Tk) alive
            for sig, handler in previous.items():
                signal.signal(sig, handler)

    def stop_robot(self) -> None:
        self.joystick.show(0.0, 0.0)
        self._set_command(0.0, 0.0)

    def clear_safety(self) -> None:
        """The latch's "Clear" button: the env thread releases the latch before its next step."""
        self.shared.request_safety_clear()
        self._refresh_safety()  # shows "Clearing..." now, not on the next refresh

    def reset_odometry(self) -> None:
        self.stop_robot()
        self.shared.request_reset()

    def quit(self) -> None:
        if self.closing:
            return
        self.closing = True  # stops _tick, so nothing renders while the env closes
        self.shared.set_command(0.0, 0.0)
        self.status.set("stopping the robot...")
        self.root.update_idletasks()
        self.runner.stop_event.set()
        if self.runner.is_alive():
            self.runner.join(timeout=8.0)  # finishes its step, stops the robot, closes the env
        self.root.destroy()

    # ------------------------------------------------------------------ layout

    def _build(self, max_speed: float) -> None:
        root = self.root
        root.title("RoboGym manual control")
        root.resizable(False, False)
        style = ttk.Style(root)
        if root.tk.call("tk", "windowingsystem") == "x11":
            style.theme_use("clam")
        bg = style.lookup("TFrame", "background") or None
        bold = ("TkDefaultFont", 10, "bold")

        main = ttk.Frame(root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")

        # Left: live camera feed and the observation values.
        left = ttk.Frame(main)
        left.grid(row=0, column=0, sticky="n", padx=(0, 14))
        self.camera_title = tk.StringVar(value="Camera (live)")
        ttk.Label(left, textvariable=self.camera_title, font=bold).grid(row=0, column=0, sticky="w")
        w, h = CAMERA_BOX
        self.camera_canvas = tk.Canvas(left, width=w, height=h, bg="black", highlightthickness=0)
        self.camera_canvas.grid(row=1, column=0, pady=(4, 8))
        self._camera_item = self.camera_canvas.create_image(w // 2, h // 2)
        self._camera_wait = self.camera_canvas.create_text(w // 2, h // 2, fill="#bbbbbb",
                                                           text="waiting for the camera...")
        self.readout = tk.StringVar(value="waiting for the first observation...")
        ttk.Label(left, textvariable=self.readout, font="TkFixedFont", justify="left").grid(
            row=2, column=0, sticky="w")

        # Right: the agent's image and the controls.
        right = ttk.Frame(main)
        right.grid(row=0, column=1, sticky="n")
        oh, ow = self.env.image_shape
        ttk.Label(right, text=f"Agent observation ({ow}x{oh}, shown {OBS_ZOOM}x)", font=bold).grid(
            row=0, column=0, columnspan=2, sticky="w")
        self.obs_canvas = tk.Canvas(right, width=ow * OBS_ZOOM, height=oh * OBS_ZOOM, bg="black",
                                    highlightthickness=0)
        self.obs_canvas.grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 10))
        self._obs_item = self.obs_canvas.create_image(ow * OBS_ZOOM // 2, oh * OBS_ZOOM // 2)

        self.hold = tk.BooleanVar(value=False)
        self.joystick = Joystick(right, size=240, on_move=self._on_joystick,
                                 on_release=self._on_joystick_release, bg=bg)
        self.joystick.grid(row=2, column=0)
        tracks = ttk.Frame(right)
        tracks.grid(row=2, column=1, padx=(4, 0))
        self.left_lever = TrackLever(tracks, "left", 240, lambda v: self._on_track(0, v),
                                     lambda: self._on_track_release(0), bg=bg)
        self.right_lever = TrackLever(tracks, "right", 240, lambda v: self._on_track(1, v),
                                      lambda: self._on_track_release(1), bg=bg)
        self.left_lever.pack(side="left")
        self.right_lever.pack(side="left")

        self.latch = SafetyLatch(right, self.clear_safety, bg=bg or "#dcdad5")
        self.latch.grid(row=3, column=0, columnspan=2, sticky="we", pady=(8, 0))
        if self.env.safety is None:
            self.latch.show_off()
        else:
            self.latch.show_armed(self.env.safety.config)

        speed = ttk.Frame(right)
        speed.grid(row=4, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(speed, text="Max speed").pack(side="left")
        self.speed_text = tk.StringVar()
        scale = ttk.Scale(speed, from_=0.1, to=1.0, orient="horizontal", length=140,
                          command=self._on_speed, takefocus=False)
        scale.set(max_speed)
        scale.pack(side="left", padx=8)
        ttk.Label(speed, textvariable=self.speed_text, width=5).pack(side="left")
        ttk.Button(speed, text="Plots", command=self.open_plots, takefocus=False).pack(
            side="left", padx=(4, 0))
        self._on_speed(max_speed)

        buttons = ttk.Frame(right)
        buttons.grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))
        tk.Button(buttons, text="STOP", command=self.stop_robot, bg="#d93025", fg="white",
                  activebackground="#b3261e", activeforeground="white", font=bold, width=7,
                  takefocus=False).pack(side="left")
        ttk.Checkbutton(buttons, text="Hold levers", variable=self.hold, takefocus=False).pack(
            side="left", padx=12)
        ttk.Button(buttons, text="Reset odometry", command=self.reset_odometry, takefocus=False).pack(
            side="left")
        ttk.Label(right, foreground="#5f6b7a", justify="left",
                  text="Joystick: drive and turn.  Side levers: one track each.\n"
                       "Let go to stop.  Space = stop,  Esc = quit.").grid(
            row=6, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.status = tk.StringVar(value="starting...")
        self.status_label = ttk.Label(root, textvariable=self.status, anchor="w", padding=(10, 3),
                                      relief="sunken")
        self.status_label.grid(row=1, column=0, sticky="we")

        root.bind("<space>", lambda _e: self.stop_robot())
        root.bind("<Escape>", lambda _e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

    # ----------------------------------------------------------------- refresh

    # ------------------------------------------------------------------ levers

    def _on_joystick(self, x: float, y: float) -> None:
        self._set_command(*arcade_mix(x, y))

    def _on_joystick_release(self) -> None:
        if not self.hold.get():
            self.stop_robot()

    def _on_track(self, index: int, value: float) -> None:
        command = list(self._command)
        command[index] = value
        left, right = command
        self.joystick.show((left - right) / 2, (left + right) / 2)  # same command, joystick view
        self._set_command(left, right)

    def _on_track_release(self, index: int) -> None:
        if not self.hold.get():
            self._on_track(index, 0.0)  # only this lever's track stops

    def _set_command(self, left: float, right: float) -> None:
        self._command = (left, right)
        self.shared.set_command(left, right)
        self._draw_track_levers()

    def _draw_track_levers(self) -> None:
        held = self._safety_trip is not None  # latched: the env sends zero whatever the levers say
        for lever, value in zip((self.left_lever, self.right_lever), self._command):
            lever.show(value, 0.0 if held else value * self._max_speed)

    def _on_speed(self, value) -> None:
        speed = float(value)
        self._max_speed = speed
        self.shared.set_max_speed(speed)
        self.speed_text.set(f"{speed:.0%}")
        self._draw_track_levers()

    def _tick(self) -> None:
        if self.closing:
            return
        try:
            self.shared.touch()
            self._refresh_camera()
            result = self.shared.latest()
            if result is not None and result.count != self.steps_shown:
                self.steps_shown = result.count
                self._show_observation(result)
                self._plots_dirty = True
            self._refresh_safety()
            now = time.monotonic()
            # Redraw on new data, and once a second anyway so the time axis keeps rolling.
            if self.plots is not None and (self._plots_dirty or now - self._plots_drawn_at > 1.0):
                self.plots.redraw(self.shared.history(), now)
                self._plots_dirty = False
                self._plots_drawn_at = now
            text, error = self.shared.status()
            self._tick_error = None
        except Exception as exc:  # never let one bad refresh freeze the window
            text, error = f"window refresh failed: {exc!r} (details in the terminal)", True
            if repr(exc) != self._tick_error:  # log each new error once, not 25 times a second
                logger.exception("window refresh failed")
                self._tick_error = repr(exc)
        self.status.set(text)
        self.status_label.configure(foreground="#b3261e" if error else "")
        self.root.after(UI_PERIOD_MS, self._tick)

    def _refresh_safety(self) -> None:
        if self.env.safety is None:
            return  # the panel says "off" for good
        trip, clearing = self.shared.safety()
        new_trip = trip is not None and trip is not self._safety_trip
        self._safety_trip = trip
        if new_trip:  # centre the levers, so clearing the latch never drives off by itself
            self.stop_robot()
        shown = (trip, clearing and trip is not None)
        if shown != self._latch_shown:
            self._latch_shown = shown
            if trip is None:
                self.latch.show_armed(self.env.safety.config)
            else:
                self.latch.show_tripped(trip, clearing)
            self._draw_track_levers()

    def _refresh_camera(self) -> None:
        frame = self.env.render()  # newest full-size frame; safe while the runner steps
        if frame is None:
            return
        probe = frame[::32, ::32]
        if (self._camera_probe is not None and probe.shape == self._camera_probe.shape
                and np.array_equal(probe, self._camera_probe)):
            return  # no new frame since the last refresh
        self._camera_probe = probe.copy()
        h, w = frame.shape[:2]
        scale = min(CAMERA_BOX[0] / w, CAMERA_BOX[1] / h)
        size = (max(1, round(w * scale)), max(1, round(h * scale)))
        if size != (w, h):
            frame = np.asarray(Image.fromarray(frame).resize(size, _BILINEAR))
        self._camera_photo = to_photo(self.root, frame)
        self.camera_canvas.itemconfigure(self._camera_item, image=self._camera_photo)
        self.camera_canvas.itemconfigure(self._camera_wait, state="hidden")
        self.camera_frames += 1
        now = time.monotonic()
        if self._camera_t is not None:
            fps = 1.0 / max(now - self._camera_t, 1e-6)
            self._camera_fps = fps if self._camera_fps == 0.0 else 0.9 * self._camera_fps + 0.1 * fps
            self.camera_title.set(f"Camera (live, {self._camera_fps:.0f} fps)")
        self._camera_t = now

    def _show_observation(self, result: StepResult) -> None:
        image = result.obs.get("image")
        if image is not None:
            zoomed = np.repeat(np.repeat(image, OBS_ZOOM, axis=0), OBS_ZOOM, axis=1)
            self._obs_photo = to_photo(self.root, zoomed)
            self.obs_canvas.itemconfigure(self._obs_item, image=self._obs_photo)
        self.readout.set(format_readout(result))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fake", action="store_true", help="drive the simulated robot instead")
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--max-speed", type=float, default=0.6, help="initial speed limit, 0.1-1 (default 0.6)")
    parser.add_argument("--min-pwm", type=int, default=0,
                        help="PWM for the smallest lever movement; ~40 (the motor deadband) makes "
                             "small movements drive instead of doing nothing")
    parser.add_argument("--action-duration", type=float, default=0.1,
                        help="seconds per env step before the robot reports back (default 0.1)")
    parser.add_argument("--no-safety", action="store_true",
                        help="drive without the env's safety layer (stall / no-turn / obstacle stop)")
    parser.add_argument("--no-plots", action="store_true",
                        help="don't open the plot window at start (the Plots button still does)")
    parser.add_argument("--plot-seconds", type=float, default=20.0,
                        help="how many seconds the plots show (default 20)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(1, f"cannot open a window: {exc}\n")
    max_speed = min(max(args.max_speed, 0.1), 1.0)
    env_kwargs = {"render_mode": "rgb_array", "action_duration": args.action_duration,
                  "min_pwm": args.min_pwm, "safety": None if args.no_safety else SafetyConfig()}
    app_kwargs = {"max_speed": max_speed, "show_plots": not args.no_plots,
                  "plot_seconds": max(args.plot_seconds, 1.0)}
    with contextlib.ExitStack() as stack:
        if args.fake:
            fake = stack.enter_context(FakeRobotServer())
            env = RoboGym(**fake.env_kwargs(), **env_kwargs)
        else:
            env = RoboGym(host=args.host, **env_kwargs)
        app = ManualControlApp(root, env, **app_kwargs)
        app.run()
        # Free Tk now, on the main thread. Tcl aborts the whole process if the garbage
        # collector happens to free its interpreter on another thread (e.g. the fake
        # robot's, still running here); pausing automatic collection rules that out.
        gc.disable()
        del app, env, root
        gc.collect()
        gc.enable()


if __name__ == "__main__":
    main()
