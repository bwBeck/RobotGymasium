"""examples/manual_control.py, driven programmatically against the fake robot.

The Tk window is withdrawn, so nothing appears on screen; the test is skipped where Tk
can't open a display at all (e.g. headless CI).
"""

import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from robotgym import RoboGym
from robotgym.fake_robot import FakeRobotServer

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


@pytest.fixture
def root():
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("tkinter has no display")
    root.withdraw()
    yield root
    try:
        root.destroy()
    except tk.TclError:
        pass  # the app already destroyed it


def test_lever_drives_the_robot_and_the_window_shows_the_images(root):
    with FakeRobotServer() as fake:
        env = RoboGym(**fake.env_kwargs(), render_mode="rgb_array",
                      reset_settle_time=0.05, calibrate_gyro_on_reset=False)
        app = mc.ManualControlApp(root, env, max_speed=0.8)
        app.start()
        try:
            # A parked fake robot renders identical frames, which the window doesn't
            # repaint - so expect one frame now and fresh ones once it moves.
            assert _pump(root, until=lambda: app.steps_shown >= 2 and app.camera_frames >= 1), app.status.get()
            assert app._camera_photo.width() == 640  # 320x240 stream, shown 2x
            assert app._obs_photo.width() == 2 * env.image_shape[1]
            assert app.status.get() == "connected"

            start, frames = fake.robot.x, app.camera_frames
            app.lever.set(0.0, 1.0)  # lever fully forward
            _pump(root, seconds=1.0)
            assert fake.robot.cmd == [204, 204]  # 0.8 * 255
            assert fake.robot.x > start + 0.05
            assert app.camera_frames >= frames + 5  # the live feed follows the motion
            assert "steps/s" in app.readout.get()

            app.lever._release()  # let go: springs back, the robot stops
            assert _pump(root, until=lambda: fake.robot.cmd == [0, 0])

            app.hold.set(True)
            app.lever.set(-1.0, 0.0)  # full left...
            app.lever._release()  # ...and let go: held, keeps turning left
            _pump(root, seconds=0.6)
            assert fake.robot.cmd[0] < 0 < fake.robot.cmd[1]
            app.stop_robot()
            assert _pump(root, until=lambda: fake.robot.cmd == [0, 0])
        finally:
            app.quit()
        assert not app.runner.is_alive()
        assert fake.robot.received[-1] == '"N":100'  # env closed: the bridge saw the disconnect
