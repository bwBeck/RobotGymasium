"""Runs the real arduino/RoboGym/RoboGym.ino - compiled for this PC against the Arduino
stand-ins in tests/native - behind the fake ESP32 bridge, and drives it with RoboGym.

This checks the sketch's protocol, timing and sign conventions against the Python side
without hardware. The simulated robot (tests/native/sim_main.cpp) has an obstacle 1 m
ahead, a 7.90 V battery, a gyro driven by the track commands, and a micros() counter
that wraps a few seconds into every run.
"""

import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from robotgym import RoboGym
from robotgym.link import ProtocolError, RobotLink, TcpTransport
from robotgym.fake_robot import FakeEsp32Bridge

ROOT = Path(__file__).resolve().parents[1]
SKETCH = ROOT / "arduino" / "RoboGym" / "RoboGym.ino"
NATIVE = ROOT / "tests" / "native"


@pytest.fixture(scope="module")
def native_sketch(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("native") / "robogym_native"
    subprocess.run(
        [compiler, "-std=gnu++17", "-O1", "-Wall", "-Wextra", "-I", str(NATIVE / "shim"),
         "-x", "c++", str(SKETCH), "-x", "none", str(NATIVE / "sim_main.cpp"), "-o", str(exe)],
        check=True,
    )
    return exe


class ProcessDevice:
    """The compiled sketch as a serial device: its stdin/stdout are the UNO's UART."""

    def __init__(self, exe):
        self.proc = subprocess.Popen([str(exe)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        self._buf = bytearray()
        self._lock = threading.Lock()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        while True:
            data = self.proc.stdout.read(4096)
            if not data:
                return
            with self._lock:
                self._buf += data

    def write(self, data):
        try:
            self.proc.stdin.write(data)
        except (BrokenPipeError, ValueError):
            pass

    def read(self, timeout=0.0):
        with self._lock:
            out, self._buf = bytes(self._buf), bytearray()
        return out

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=2)
        except Exception:
            self.proc.kill()


@pytest.fixture
def robot(native_sketch):
    device = ProcessDevice(native_sketch)
    bridge = FakeEsp32Bridge(device)
    bridge.start()
    yield bridge
    bridge.stop()
    device.close()


@pytest.fixture
def env(robot):
    env = RoboGym(host="127.0.0.1", control_port=robot.port, camera="none",
                  reset_settle_time=0.05, action_duration=0.3)
    yield env
    env.close()


def test_sketch_reports_sensors_and_motion(env):
    obs, info = env.reset()
    assert info["imu_ok"]
    assert info["battery_v"] == pytest.approx(7.90)
    assert obs["distance"][0] == pytest.approx(1.0, abs=0.03)

    obs, _, _, _, info = env.step([0.8, 0.8])
    dx, dy, dth = obs["delta_pose"]
    # Motor model from rest: PWM 204 at 7.9 V -> ~2.48 rev/s with an 80 ms lag -> ~0.55 turns.
    assert obs["motor_turns"][0] == pytest.approx(0.55, rel=0.15)
    assert obs["motor_turns"][0] == pytest.approx(obs["motor_turns"][1])
    assert dx == pytest.approx(obs["motor_turns"][0] * 3.14159 * 0.043, rel=0.02)
    assert abs(dy) < 1e-3 and abs(dth) < 1e-3
    assert obs["robot_dt"][0] == pytest.approx(obs["dt"][0], abs=0.05)

    obs, *_ = env.step([-0.8, 0.8])  # turn left
    assert obs["delta_pose"][2] > 0.3
    assert obs["motor_turns"][0] < 0 < obs["motor_turns"][1]


def test_sketch_timestamps_survive_the_micros_wraparound(env):
    env.reset()
    times = []
    for _ in range(12):  # ~4 s: micros() wraps during this loop
        obs, _, _, _, info = env.step([0.0, 0.0])
        times.append(info["robot_time"])
        assert 0.2 < obs["robot_dt"][0] < 1.0
    assert any(b < a for a, b in zip(times, times[1:])), "micros() should have wrapped"


def test_sketch_watchdog_stops_the_motors(robot):
    env = RoboGym(host="127.0.0.1", control_port=robot.port, camera="none",
                  reset_settle_time=0.05, action_duration=0.05, hold_timeout=0.2)
    try:
        env.reset()
        env.step([0.6, 0.6])
        time.sleep(0.6)
        obs, _, _, _, info = env.step([0.0, 0.0])
        assert info["watchdog_stopped"]
        _, _, _, _, info = env.step([0.0, 0.0])
        assert not info["watchdog_stopped"]  # latched for one packet only
    finally:
        env.close()


def test_sketch_rejects_malformed_commands(robot):
    link = RobotLink(TcpTransport("127.0.0.1", robot.port))
    link.open()
    try:
        link.request("P", [], 5.0)  # waits out the sketch's boot
        with pytest.raises(ProtocolError, match="malformed"):
            link.request("A", [100], 2.0)
        assert link.request("P", [], 2.0).imu_ok
    finally:
        link.close()
