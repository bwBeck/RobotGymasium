"""Drive the robot by hand through the RoboGym env, with a live camera feed.

    python examples/manual_control.py           # the real robot (PC on its "ELEGOO-xxxx" WiFi)
    python examples/manual_control.py --fake    # simulated robot, no hardware

Drag the lever: up/down = forward/reverse, left/right = turn. Let go and it springs back
to the centre, which stops the robot; tick "Hold lever" to make it stay where you leave
it. The STOP button or Space always stops; Esc, closing the window or Ctrl+C in the
terminal stops the robot and quits.

The window shows the live camera feed, the image the agent receives in its observation,
and the values of the latest observation. The env runs in a background thread that steps
as fast as the robot answers (~5 Hz) with whatever the lever says, so the window never
waits for the robot.

Needs tkinter, which comes with Python (Debian/Ubuntu: sudo apt install python3-tk).
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from tkinter import ttk

import numpy as np
from PIL import Image

from robotgym import RoboGym, RobotConnectionError
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


def format_readout(r: StepResult) -> str:
    o, i = r.obs, r.info
    dx, dy, dth = (float(v) for v in o["delta_pose"])
    turns_l, turns_r = (float(v) for v in o["motor_turns"])
    x, y, th = (float(v) for v in i["pose"])
    pwm_l, pwm_r = i.get("pwm", (0, 0))
    age = i.get("image_age")
    flags = [name for name, on in (
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
        self._stick = (0.0, 0.0)
        self._max_speed = max_speed
        self._ui_alive = time.monotonic()
        self._reset_requested = False
        self._result: StepResult | None = None
        self._status = ("starting...", False)

    def set_stick(self, x: float, y: float) -> None:
        with self._lock:
            self._stick = (x, y)

    def set_max_speed(self, speed: float) -> None:
        with self._lock:
            self._max_speed = speed

    def touch(self) -> None:
        """Called on every window refresh."""
        with self._lock:
            self._ui_alive = time.monotonic()

    def action(self) -> np.ndarray:
        """The lever as an env action; zero if the window has stopped refreshing."""
        with self._lock:
            (x, y), speed, alive = self._stick, self._max_speed, self._ui_alive
        if time.monotonic() - alive > UI_STALE_S:
            return np.zeros(2, dtype=np.float32)
        return np.array(arcade_mix(x, y), dtype=np.float32) * speed

    def request_reset(self) -> None:
        with self._lock:
            self._reset_requested = True

    def take_reset_request(self) -> bool:
        with self._lock:
            requested, self._reset_requested = self._reset_requested, False
        return requested

    def publish(self, result: StepResult) -> None:
        with self._lock:
            self._result = result

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
            action = self.shared.action()
            obs, _, terminated, truncated, info = self.env.step(action)
            now = time.monotonic()
            step_rate = 1.0 / max(now - last, 1e-6)
            rate = step_rate if rate == 0.0 else 0.8 * rate + 0.2 * step_rate
            last = now
            self._publish(obs, info, action, rate)
            if terminated or truncated:
                self._reset()

    def _reset(self) -> None:
        obs, info = self.env.reset()
        self._publish(obs, info, np.zeros(2, dtype=np.float32), 0.0)

    def _publish(self, obs: dict, info: dict, action: np.ndarray, rate: float) -> None:
        self._count += 1
        self.shared.publish(StepResult(obs, info, action, self._count, rate))


class Lever(tk.Canvas):
    """Joystick-style lever. Calls ``on_move(x, y)``: x = right, y = forward, each -1..1.

    Springs back to the centre when released, unless ``hold`` is set.
    """

    IDLE, ACTIVE = "#2f6fd6", "#5b93f2"

    def __init__(self, master: tk.Misc, size: int, on_move, hold: tk.BooleanVar, bg: str | None = None):
        super().__init__(master, width=size, height=size, highlightthickness=0, bg=bg)
        self.on_move = on_move
        self.hold = hold
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
        self.knob = self.create_oval(0, 0, 0, 0, fill=self.IDLE, outline="#1d4f9e", width=2)
        self._draw(0.0, 0.0)
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<B1-Motion>", self._drag)
        self.bind("<ButtonRelease-1>", self._release)

    def set(self, x: float, y: float) -> None:
        self._draw(x, y)
        self.on_move(x, y)

    def _press(self, event) -> None:
        self.itemconfigure(self.knob, fill=self.ACTIVE)
        self._drag(event)

    def _drag(self, event) -> None:
        dx, dy = event.x - self.c, event.y - self.c
        dist = math.hypot(dx, dy)
        if dist > self.r:
            dx, dy = dx * self.r / dist, dy * self.r / dist
        self.set(dx / self.r, -dy / self.r)

    def _release(self, _event=None) -> None:
        self.itemconfigure(self.knob, fill=self.IDLE)
        if not self.hold.get():
            self.set(0.0, 0.0)

    def _draw(self, x: float, y: float) -> None:
        c, r, k = self.c, self.r, self.k
        px, py = c + x * r, c - y * r
        self.coords(self.stem, c, c, px, py)
        self.coords(self.knob, px - k, py - k, px + k, py + k)


class TrackGauges(tk.Canvas):
    """Two vertical bars: the left / right track command being sent (-1..1)."""

    def __init__(self, master: tk.Misc, height: int, bg: str | None = None):
        super().__init__(master, width=100, height=height, highlightthickness=0, bg=bg)
        self.top, self.bottom = 24, height - 24
        self.mid = (self.top + self.bottom) / 2
        self.half = (self.bottom - self.top) / 2
        self.columns = []
        for i, name in enumerate(("left", "right")):
            x0 = 12 + i * 46
            x1 = x0 + 30
            self.create_rectangle(x0, self.top, x1, self.bottom, fill="#eef1f5", outline="#9aa5b1")
            bar = self.create_rectangle(x0 + 1, self.mid, x1 - 1, self.mid, fill="#3aa757", outline="")
            self.create_line(x0 - 4, self.mid, x1 + 4, self.mid, fill="#5f6b7a")
            self.create_text((x0 + x1) / 2, self.top - 12, text=name, fill="#3c4650",
                             font=("TkDefaultFont", 9, "bold"))
            value = self.create_text((x0 + x1) / 2, self.bottom + 12, text="+0.00", fill="#3c4650",
                                     font=("TkFixedFont", 9))
            self.columns.append((x0, x1, bar, value))

    def set(self, left: float, right: float) -> None:
        for (x0, x1, bar, value), v in zip(self.columns, (left, right)):
            y = self.mid - v * self.half
            self.coords(bar, x0 + 1, min(y, self.mid), x1 - 1, max(y, self.mid))
            self.itemconfigure(bar, fill="#3aa757" if v >= 0 else "#e8833a")
            self.itemconfigure(value, text=f"{v:+.2f}")


class ManualControlApp:
    """The window. Only this (main) thread touches Tk; the env lives in an EnvRunner."""

    def __init__(self, root: tk.Tk, env: RoboGym, max_speed: float = 0.6):
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
        self._build(max_speed)

    def start(self) -> None:
        self.runner.start()
        self._tick()

    def run(self) -> None:
        """start() and run the window until it closes. Ctrl+C quits cleanly too."""
        # Without this, Ctrl+C either escapes mainloop mid-step or gets swallowed by
        # Tk's callback wrapper. Instead: stop the robot and close like the window does.
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: self.root.after(0, self.quit))
        self.start()
        try:
            self.root.mainloop()
        except KeyboardInterrupt:
            self.quit()

    def stop_robot(self) -> None:
        self.lever.set(0.0, 0.0)

    def reset_odometry(self) -> None:
        self.lever.set(0.0, 0.0)
        self.shared.request_reset()

    def quit(self) -> None:
        if self.closing:
            return
        self.closing = True  # stops _tick, so nothing renders while the env closes
        self.shared.set_stick(0.0, 0.0)
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
        self.lever = Lever(right, size=240, on_move=self.shared.set_stick, hold=self.hold, bg=bg)
        self.lever.grid(row=2, column=0)
        self.gauges = TrackGauges(right, height=240, bg=bg)
        self.gauges.grid(row=2, column=1, padx=(6, 0))

        speed = ttk.Frame(right)
        speed.grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Label(speed, text="Max speed").pack(side="left")
        self.speed_text = tk.StringVar()
        scale = ttk.Scale(speed, from_=0.1, to=1.0, orient="horizontal", length=200,
                          command=self._on_speed, takefocus=False)
        scale.set(max_speed)
        scale.pack(side="left", padx=8)
        ttk.Label(speed, textvariable=self.speed_text, width=5).pack(side="left")
        self._on_speed(max_speed)

        buttons = ttk.Frame(right)
        buttons.grid(row=4, column=0, columnspan=2, sticky="w", pady=(10, 0))
        tk.Button(buttons, text="STOP", command=self.stop_robot, bg="#d93025", fg="white",
                  activebackground="#b3261e", activeforeground="white", font=bold, width=7,
                  takefocus=False).pack(side="left")
        ttk.Checkbutton(buttons, text="Hold lever", variable=self.hold, takefocus=False).pack(
            side="left", padx=12)
        ttk.Button(buttons, text="Reset odometry", command=self.reset_odometry, takefocus=False).pack(
            side="left")
        ttk.Label(right, text="Drag the lever to drive.  Space = stop,  Esc = quit.",
                  foreground="#5f6b7a").grid(row=5, column=0, columnspan=2, sticky="w", pady=(10, 0))

        self.status = tk.StringVar(value="starting...")
        self.status_label = ttk.Label(root, textvariable=self.status, anchor="w", padding=(10, 3),
                                      relief="sunken")
        self.status_label.grid(row=1, column=0, sticky="we")

        root.bind("<space>", lambda _e: self.stop_robot())
        root.bind("<Escape>", lambda _e: self.quit())
        root.protocol("WM_DELETE_WINDOW", self.quit)

    # ----------------------------------------------------------------- refresh

    def _on_speed(self, value) -> None:
        speed = float(value)
        self.shared.set_max_speed(speed)
        self.speed_text.set(f"{speed:.0%}")

    def _tick(self) -> None:
        if self.closing:
            return
        self.shared.touch()
        self._refresh_camera()
        result = self.shared.latest()
        if result is not None and result.count != self.steps_shown:
            self.steps_shown = result.count
            self._show_observation(result)
        self.gauges.set(*self.shared.action())
        text, error = self.shared.status()
        self.status.set(text)
        self.status_label.configure(foreground="#b3261e" if error else "")
        self.root.after(UI_PERIOD_MS, self._tick)

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
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    try:
        root = tk.Tk()
    except tk.TclError as exc:
        parser.exit(1, f"cannot open a window: {exc}\n")
    max_speed = min(max(args.max_speed, 0.1), 1.0)
    env_kwargs = {"render_mode": "rgb_array", "action_duration": args.action_duration,
                  "min_pwm": args.min_pwm}
    if args.fake:
        with FakeRobotServer() as fake:
            ManualControlApp(root, RoboGym(**fake.env_kwargs(), **env_kwargs), max_speed).run()
    else:
        ManualControlApp(root, RoboGym(host=args.host, **env_kwargs), max_speed).run()


if __name__ == "__main__":
    main()
