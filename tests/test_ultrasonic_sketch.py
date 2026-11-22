"""Runs arduino/UltrasonicTest/UltrasonicTest.ino - compiled for this PC against the Arduino
stand-ins in tests/native - against a fake HC-SR04 in each fault mode, and checks that it
names the right failure and verdict (tests/native/ultrasonic_main.cpp). The simulated
clock makes each run instant and repeatable.
"""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SKETCH = ROOT / "arduino" / "UltrasonicTest" / "UltrasonicTest.ino"
NATIVE = ROOT / "tests" / "native"


@pytest.fixture(scope="module")
def sketch(tmp_path_factory):
    compiler = shutil.which("g++") or shutil.which("clang++")
    if compiler is None:
        pytest.skip("no C++ compiler")
    exe = tmp_path_factory.mktemp("native") / "ultrasonic_native"
    subprocess.run(
        [compiler, "-std=gnu++17", "-O1", "-Wall", "-Wextra", "-Werror", "-I", str(NATIVE / "shim"),
         "-x", "c++", str(SKETCH), "-x", "none", str(NATIVE / "ultrasonic_main.cpp"), "-o", str(exe)],
        check=True,
    )
    return exe


def run(sketch, mode, serial_input="", seconds=2.3):
    out = subprocess.run([str(sketch), mode, serial_input, str(seconds)], capture_output=True,
                         text=True, timeout=30, check=True).stdout
    return out.replace("\r\n", "\n")


def pings(out):
    return [line for line in out.splitlines() if line.startswith("#")]


def summary(out):
    lines = out.splitlines()
    i = next(i for i, line in enumerate(lines) if line.startswith("--- last 20 pings:"))
    return lines[i], lines[i + 1]


# The wall is 500 mm away (a 2915 us pulse). Polling the echo pin takes a few us per pass,
# so the sketch measures it a few us short: 499 or 500 mm.
MM = r"(499|500)"


@pytest.mark.parametrize("mode, first_ping, counts, verdict", [
    ("ok", rf"OK +{MM} mm +29\d\d us +\|#{{9,10}}", rf"20 OK \| OK range {MM}-{MM} mm, mean {MM} mm",
     "all OK"),
    ("trigger_open", "NO ANSWER", "20 NO ANSWER", "ignores the trigger: the trigger wire is open"),
    ("echo_open", "ECHO HIGH", "20 ECHO HIGH", "nothing holds echo low"),
    ("flapping", rf"OK +{MM} mm.*", r"15 OK, 5 NO ANSWER \| OK range", "INTERMITTENT: a contact "
     "keeps making and breaking (trigger line)"),
    ("bouncing", rf"OK +{MM} mm.*", r"10 OK, 10 GLITCH \| OK range", "INTERMITTENT: a contact "
     "keeps making and breaking (echo line bouncing)"),
    ("stuck", r"STUCK HIGH +2500\d\d us", "19 ECHO HIGH, 1 STUCK HIGH", "echo stuck high: the "
     "sensor locked up"),
    ("out_of_range", r"NO ECHO +3(799\d|800\d) us", "20 NO ECHO", "Not a wiring fault"),
])
def test_names_each_fault(sketch, mode, first_ping, counts, verdict):
    out = run(sketch, mode)
    first = pings(out)[0]
    assert re.fullmatch(r"# +1  " + first_ping, first), first
    totals, conclusion = summary(out)
    assert re.match(r"--- last 20 pings: " + counts, totals), totals
    assert conclusion.startswith("    -> ") and verdict in conclusion, conclusion


def test_failed_pings_say_nothing_more(sketch):
    assert pings(run(sketch, "trigger_open"))[1] == "#     2  NO ANSWER"


@pytest.mark.parametrize("mode, rest", [
    ("ok", "echo line at rest: LOW - the sensor is holding it low"),
    ("trigger_open", "echo line at rest: LOW"),  # powered, echo wire fine: points at the trigger
    ("echo_open", "echo line at rest: HIGH - nothing holds it low"),
])
def test_startup_checks_the_echo_line(sketch, mode, rest):
    assert rest in run(sketch, mode)


def test_pan_sweep_and_counter_reset(sketch):
    out = run(sketch, "ok", "sr", seconds=6.0)  # 60 pings: 90 -> 120 -> 60 -> 99
    assert "pan sweep ON" in out and "counters reset" in out
    angles = [int(m) for m in re.findall(r"^# +\d+  pan +(\d+)  OK", out, re.M)]
    assert len(angles) > 20
    assert min(angles) == 60 and max(angles) == 120  # turns all the way, both ways
    assert all(abs(b - a) == 3 for a, b in zip(angles, angles[1:]))


def test_pull_up_toggle_exposes_a_weak_echo_high_level(sketch):
    # A worn contact: the echo pulse only reads high with the pull-up's help.
    totals, conclusion = summary(run(sketch, "weak_high"))
    assert re.match(rf"--- last 20 pings: 20 OK \| OK range {MM}-{MM} mm.* \[pull-up on, every 100 ms\]$",
                    totals), totals
    out = run(sketch, "weak_high", "p")  # pull-up off, as RoboGym.ino before the fix
    assert "echo pull-up OFF" in out
    totals, conclusion = summary(out)
    assert totals == "--- last 20 pings: 20 NO ANSWER [pull-up off, every 100 ms]"
    assert "(pull-up off: if pings succeed with it on" in out


def test_fast_pings_like_robogym(sketch):
    out = run(sketch, "ok", "f")
    assert "pinging every 60 ms" in out
    assert len(pings(out)) >= 36  # 2.3 s at 60 ms, not 23 at 100 ms
    assert summary(out)[0].endswith("[pull-up on, every 60 ms]")
