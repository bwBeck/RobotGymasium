"""examples/manual_control.py, driven programmatically against the fake robot.

The Tk window is withdrawn, so nothing appears on screen; the test is skipped where Tk
can't open a display at all (e.g. headless CI).
"""

import contextlib
import gc
import importlib.util
import re
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from robotgym import RoboGym, SafetyConfig
from robotgym.fake_robot import FakeRobotServer, SimRobot

tk = pytest.importorskip("tkinter")

SCRIPT = Path(__file__).resolve().parents[1] / "examples" / "manual_control.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("manual_control", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up here
    spec.loader.exec_module(module)
    return module


mc = _load_script()


@pytest.mark.parametrize("x, y, expected", [
    (0.0, 1.0, (1.0, 1.0)),    # forward
    (0.0, -1.0, (-1.0, -1.0)),  # reverse
    (1.0, 0.0, (1.0, -1.0)),   # spin clockwise
    (-1.0, 0.0, (-1.0, 1.0)),  # spin counter-clockwise
    (1.0, 1.0, (1.0, 0.0)),    # pivot on the right track
    (0.3, 0.5, (0.8, 0.2)),
    (0.0, 0.0, (0.0, 0.0)),
])
def test_arcade_mix(x, y, expected):
    assert mc.arcade_mix(x, y) == pytest.approx(expected)


def test_nice_ticks():
    assert mc.nice_ticks(-20, 0) == [-20, -15, -10, -5, 0]
    assert mc.nice_ticks(-275, 275) == [-200, -100, 0, 100, 200]
    assert mc.nice_ticks(-0.6, 0.6) == pytest.approx([-0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6])
    assert mc.nice_ticks(-1.37, 2.9) == [-1, 0, 1, 2]
    assert mc._tick_labels([-0.5, 0.0, 0.5]) == ["-0.5", "0.0", "0.5"]


def test_every_track_command_is_reachable_with_the_lever():
    for left in np.linspace(-1, 1, 9):
        for right in np.linspace(-1, 1, 9):
            x, y = (left - right) / 2, (left + right) / 2
            assert mc.arcade_mix(x, y) == pytest.approx((left, right))


def _pump(root, seconds=None, until=None, timeout=10.0):
    deadline = time.monotonic() + (seconds if seconds is not None else timeout)
    while time.monotonic() < deadline:
        root.update()
        if until is not None and until():
            return True
        time.sleep(0.01)
    return until is None


# One Tk interpreter for all tests, kept alive until Python exits. Creating and dropping
# one per test let the garbage collector free an interpreter on a background thread (the
# env's, the camera's, the fake robot's), and Tcl aborts the process when that happens.
_TK_MASTER: list = []


@pytest.fixture
def root():
    """A fresh hidden top-level window per test; the app treats it as its root."""
    if not _TK_MASTER:
        try:
            master = tk.Tk()
        except tk.TclError:
            pytest.skip("tkinter has no display")
        master.withdraw()
        _TK_MASTER.append(master)
    gc.collect()  # free the previous test's windows here, on the main thread
    window = tk.Toplevel(_TK_MASTER[0])
    window.withdraw()  # never shown: nothing pops up while testing
    yield window
    try:
        window.destroy()
    except tk.TclError:
        pass  # the app already destroyed it


@contextlib.contextmanager
def _running_app(root, robot=None, **env_kwargs):
    """The app driving a fake robot (default ``SimRobot()``) until the block ends; then
    checks that closing the window stopped the robot and disconnected."""
    with FakeRobotServer(robot=robot) as fake:
        env = RoboGym(**fake.env_kwargs(), render_mode="rgb_array",
                      reset_settle_time=0.05, calibrate_gyro_on_reset=False, **env_kwargs)
        app = mc.ManualControlApp(root, env, max_speed=0.8, show_plots=False)  # see test_plot_window
        app.start()
        try:
            # A parked fake robot renders identical frames, which the window doesn't
            # repaint - so expect one frame now and fresh ones once it moves.
            assert _pump(root, until=lambda: app.steps_shown >= 2 and app.camera_frames >= 1), app.status.get()
            yield app, fake.robot
        finally:
            app.quit()
        assert not app.runner.is_alive()
        # The bridge's disconnect notice crosses the simulated 9600-baud link: wait for it.
        received = fake.robot.received
        deadline = time.monotonic() + 3.0
        while received[-1] != '"N":100' and time.monotonic() < deadline:
            time.sleep(0.02)
        assert received[-1] == '"N":100'  # the env disconnected...
        assert re.fullmatch(r"A,\d+,0,0,0", received[-2])  # ...right after a final stop


@pytest.fixture
def app_and_robot(root):
    with _running_app(root) as running:
        yield running


@pytest.fixture
def safe_app_and_robot(root):
    """Safety layer on; the robot 0.46 m from a wall, facing it."""
    with _running_app(root, robot=SimRobot(start=(2.45, 1.5, 0.0)), safety=SafetyConfig()) as running:
        yield running


def _label(lever) -> str:
    return lever.itemcget(lever.value_text, "text")


def test_joystick_drives_the_robot_and_the_window_shows_the_images(root, app_and_robot):
    app, robot = app_and_robot
    assert app._camera_photo.width() == 640  # 320x240 stream, shown 2x
    assert app._obs_photo.width() == 2 * app.env.image_shape[1]
    assert app.status.get() == "connected"
    assert app.latch.title.cget("text") == "Safety layer off" and not app.latch.latched

    start, frames = robot.x, app.camera_frames
    app.joystick.set(0.0, 1.0)  # joystick fully forward
    assert (_label(app.left_lever), _label(app.right_lever)) == ("+0.80", "+0.80")  # they follow
    # Wait for each effect rather than a fixed time, so a busy machine can't fail the test.
    assert _pump(root, until=lambda: robot.cmd == [204, 204])  # 0.8 * 255
    assert _pump(root, until=lambda: robot.x > start + 0.05)
    assert _pump(root, until=lambda: app.camera_frames >= frames + 5)  # the feed follows the motion
    assert _pump(root, until=lambda: "steps/s" in app.readout.get())

    app.joystick.release()  # let go: springs back, the robot stops
    assert _pump(root, until=lambda: robot.cmd == [0, 0])

    app.hold.set(True)
    app.joystick.set(-1.0, 0.0)  # full left...
    app.joystick.release()  # ...and let go: held, keeps turning left
    assert _pump(root, until=lambda: robot.cmd[0] < 0 < robot.cmd[1])
    _pump(root, seconds=0.6)
    assert robot.cmd[0] < 0 < robot.cmd[1]  # still turning: the release didn't stop it
    app.stop_robot()
    assert _pump(root, until=lambda: robot.cmd == [0, 0])


def test_plot_window(root, app_and_robot):
    app, robot = app_and_robot
    plots = app.open_plots()
    plots.window.withdraw()  # never shown during tests

    def logged(condition):  # some step in the history (t, pwm_l, pwm_r, ax, ay, az, yaw) matches
        return lambda: any(condition(row) for row in app.shared.history())

    start = robot.x
    app.joystick.set(0.0, 1.0)  # drive forward, then stop
    assert _pump(root, until=logged(lambda row: row[1] == row[2] == 204))
    assert _pump(root, until=lambda: robot.x > start + 0.05)
    app.stop_robot()
    assert _pump(root, until=lambda: robot.cmd == [0, 0])
    assert _pump(root, until=logged(lambda row: row[3] < -0.5))  # the braking step got logged
    app.joystick.set(-1.0, 0.0)  # spin left: left track back, right forward
    assert _pump(root, until=logged(lambda row: row[1] - row[2] == -408 and row[6] > 20))
    app.stop_robot()
    assert _pump(root, until=lambda: robot.cmd == [0, 0])

    history = app.shared.history()
    assert any(row[1] == row[2] == 204 for row in history)  # the drive...
    assert any(row[3] > 0.5 for row in history)  # ...its forward acceleration...
    assert any(row[3] < -0.5 for row in history)  # ...and the braking
    # A PWM change shows up as acceleration in the same step: that's what the ΔPWM row plots.
    # (Only straight-line speed changes, where both tracks change alike: starting a spin
    # changes one track by -204 too, but spinning in place has no forward acceleration.)
    rows = np.array(history)
    d_left, d_right, accel_x = np.diff(rows[:, 1]), np.diff(rows[:, 2]), rows[1:, 3]
    speed_up = (d_left == 204) & (d_right == 204)
    slow_down = (d_left == -204) & (d_right == -204)
    assert speed_up.any() and (accel_x[speed_up] > 0.5).all()
    assert slow_down.any() and (accel_x[slow_down] < -0.5).all()
    # Turning: right > left turns counter-clockwise, which the gyro reports as positive yaw...
    left_minus_right, yaw = rows[:, 1] - rows[:, 2], rows[:, 6]
    assert (yaw[left_minus_right == -408] > 0).all()
    # ...and starting the spin speeds the rotation up within that same step.
    d_lr, d_yaw = np.diff(left_minus_right), np.diff(yaw)
    assert (d_lr == -408).any() and (d_yaw[d_lr == -408] > 20).all()

    def points(plot):
        return len(plot.find_withtag("point"))

    def plots_match_history():  # the env keeps stepping while parked, so compare atomically
        rows = len(app.shared.history())
        return (points(plots.pwm_plot) == 2 * rows  # left + right per step
                and points(plots.accel_plot) == 3 * rows  # x, y, z per step
                and all(points(p) == 2 * rows for p in plots.scatters)  # left + right vs each axis
                # ΔPWM needs a previous step: every step but the first
                and all(points(p) == 2 * (rows - 1) for p in plots.change_scatters)
                and points(plots.turn_scatter) == rows  # yaw vs left - right
                and points(plots.turn_change_scatter) == rows - 1  # their changes
                and f"{rows} steps" in plots.info.get())

    assert _pump(root, until=plots_match_history)

    app.clear_history()  # the Clear button
    _pump(root, seconds=0.2)
    assert len(app.shared.history()) <= 2  # only steps that finished since

    app._close_plots()  # closing the plot window leaves manual control running
    assert app.plots is None
    app.joystick.set(0.0, 1.0)
    assert _pump(root, until=lambda: robot.cmd == [204, 204])
    app.stop_robot()
    assert _pump(root, until=lambda: robot.cmd == [0, 0])
    reopened = app.open_plots()  # the Plots button
    reopened.window.withdraw()
    assert reopened is app.plots and reopened is not plots


def test_a_failing_refresh_is_reported_and_the_window_keeps_running(root, app_and_robot, monkeypatch):
    app, robot = app_and_robot

    def broken_readout(result):  # e.g. an outdated robotgym install without 'accel'
        raise KeyError("accel")

    monkeypatch.setattr(mc, "format_readout", broken_readout)
    assert _pump(root, until=lambda: "window refresh failed" in app.status.get())
    frames = app.camera_frames
    app.joystick.set(0.0, 1.0)  # still drivable: the refresh loop (and its heartbeat) kept going
    assert _pump(root, until=lambda: robot.cmd == [204, 204] and app.camera_frames > frames + 3)
    app.stop_robot()
    assert _pump(root, until=lambda: robot.cmd == [0, 0])


def test_side_levers_drive_one_track_each(root, app_and_robot):
    app, robot = app_and_robot
    app.left_lever.set(0.5)  # left track only
    assert _pump(root, until=lambda: robot.cmd == [102, 0])  # 0.5 * 0.8 * 255
    assert _label(app.left_lever) == "+0.40" and _label(app.right_lever) == "+0.00"
    app.left_lever.release()  # let go: only that track stops
    assert _pump(root, until=lambda: robot.cmd == [0, 0])

    app.right_lever.set(0.5)
    assert _pump(root, until=lambda: robot.cmd == [0, 102])
    app.right_lever.release()
    assert _pump(root, until=lambda: robot.cmd == [0, 0])

    app.hold.set(True)  # tank-style: set one lever, then the other; both stay
    app.left_lever.set(0.5)
    app.left_lever.release()
    app.right_lever.set(-0.5)
    app.right_lever.release()
    assert _pump(root, until=lambda: robot.cmd == [102, -102])
    assert app.joystick.position == pytest.approx((0.5, 0.0))  # joystick shows it: spin right

    app.joystick.set(0.0, 1.0)  # the joystick takes over both tracks
    assert _pump(root, until=lambda: robot.cmd == [204, 204])
    app.stop_robot()
    assert _pump(root, until=lambda: robot.cmd == [0, 0])
    assert app.joystick.position == (0.0, 0.0)
    assert (_label(app.left_lever), _label(app.right_lever)) == ("+0.00", "+0.00")


def test_safety_latch_goes_red_holds_the_robot_and_clears(root, safe_app_and_robot):
    app, robot = safe_app_and_robot
    latch = app.latch
    assert "armed" in latch.title.cget("text") and not latch.latched
    assert str(latch.button.cget("state")) == "disabled"  # nothing to clear

    app.hold.set(True)  # held forward: only the safety layer can stop it
    app.joystick.set(0.0, 1.0)
    app.joystick.release()
    assert _pump(root, until=lambda: latch.latched)  # drove at the wall: tripped
    assert latch.title.cget("text").startswith("SAFETY STOP")
    assert latch.detail.cget("text").startswith("OBSTACLE")
    assert str(latch.button.cget("state")) == "normal"
    assert _pump(root, until=lambda: "SAFETY STOP" in app.readout.get())  # in the flags too
    assert robot.cmd == [0, 0]  # the env stopped the robot itself...
    assert app.joystick.position == (0.0, 0.0)  # ...and the window centred the levers
    assert (_label(app.left_lever), _label(app.right_lever)) == ("+0.00", "+0.00")

    app.joystick.set(-1.0, 0.0)  # latched: the levers move, but nothing is sent
    assert (_label(app.left_lever), _label(app.right_lever)) == ("+0.00", "+0.00")
    count = app.steps_shown
    assert _pump(root, until=lambda: app.steps_shown >= count + 3)
    assert robot.cmd == [0, 0] and latch.latched
    app.stop_robot()

    latch.button.invoke()  # Clear
    assert latch.button.cget("text") == "Clearing\u2026"
    assert _pump(root, until=lambda: not latch.latched)
    assert "armed" in latch.title.cget("text")
    count = app.steps_shown
    assert _pump(root, until=lambda: app.steps_shown >= count + 2)
    assert not latch.latched and robot.cmd == [0, 0]  # clearing doesn't drive off by itself

    app.joystick.set(0.0, -1.0)  # backing away from the wall is allowed
    assert _pump(root, until=lambda: robot.cmd == [-204, -204])
    assert (_label(app.left_lever), _label(app.right_lever)) == ("-0.80", "-0.80")
    app.stop_robot()
    assert _pump(root, until=lambda: robot.cmd == [0, 0])
    assert not latch.latched
