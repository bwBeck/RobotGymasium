# RoboGym

A [Gymnasium](https://gymnasium.farama.org/) environment for the **ELEGOO Conqueror
robot tank**, so a reinforcement-learning agent on your PC can drive the real robot.

```python
from robotgym import RoboGym

env = RoboGym()                       # PC joined to the robot's "ELEGOO-xxxxxxxx" WiFi
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step([0.5, 0.5])   # [left, right] track speed
env.close()
```

It has two halves:

* **`arduino/RoboGym/RoboGym.ino`**, the robot side, flashed to the tank's Arduino UNO.
  It drives the tracks and reports the robot's sensors.
* **`robotgym`**, a Python package containing the `RoboGym` class (a `gymnasium.Env`).
  `step(action)` sends the motor speeds and returns once the robot has reported back.

## How the pieces talk

```
                 WiFi (robot is the access point, 192.168.4.1)
  PC ─────────────────────────────────────────────▶ ESP32 camera module
  robotgym.RoboGym                                    (ELEGOO's stock firmware, unchanged)
    │                                                    │
    ├─ TCP :100  "{A,7,120,120,100}" ──────────▶ bridge ─┼─ UART 9600 baud ─▶ Arduino UNO
    │            "{O,7,...}"         ◀──────────        ─┼─                ◀─ RoboGym.ino
    │                                                    │                    motors, gyro,
    └─ HTTP :81/stream (MJPEG)       ◀────────── camera ─┘                    ultrasonic,
                                                                              battery
```

The **camera is wired to the ESP32, not to the UNO**. One camera frame is tens of
kilobytes and the UNO has 2 KB of RAM, so the image can't pass through the sketch. The
env reads the frames straight from the ESP32's video stream and combines them with the
sensor packet from the UNO into one observation. The ESP32 keeps ELEGOO's factory
firmware: nothing to flash there.

## Observation and action

`step()` returns a `spaces.Dict` observation:

| key | shape | meaning |
|---|---|---|
| `image` | (120, 160, 3) uint8 | camera frame, RGB (size set by `image_shape`; absent with `camera="none"`) |
| `delta_pose` | (3,) float32 | `[dx, dy, dθ]` since the last observation, in the robot's frame at that observation: metres forward, metres left, radians counter-clockwise |
| `motor_turns` | (2,) float32 | `[left, right]` drive-sprocket turns since the last observation, **estimated** (see below) |
| `distance` | (1,) float32 | ultrasonic range in metres; `max_distance` (4 m) when nothing is in range |
| `dt` | (1,) float32 | time since the last observation, measured on the PC |
| `robot_dt` | (1,) float32 | time since the robot's last packet, measured by the robot's own clock |
| `accel` | (3,) float32 | MPU6050 accelerometer, **averaged over the step**, robot frame (x forward, y left, z up) in m/s², minus the at-rest reading taken at `reset()`: ~0 when still, + while speeding up, − while braking |

About `accel`:
- Averaging over the step smooths out track vibration, and the average equals the change in
  velocity divided by the step's duration.
- The at-rest reading taken at each reset is subtracted, which cancels the chip's own
  offsets and gravity on a level floor.
- `info["accel_raw"]` has the same average without that subtraction, so gravity
  (~9.8 m/s² on z) is included.
- Which way the chip faces is set by `RobotCalibration(imu_forward=...)`. Find it with
  `calibrate.py imu` (see Calibration below).

**Action:** `Box(-1, 1, shape=(2,))` = `[left, right]` track speed. The sign is the
direction, and ±1 is full PWM (`max_pwm`). Each step turns the action into the PWM the
robot gets, per track:

1. **Scale** to PWM (`min_pwm`…`max_pwm`).
2. **Smoothing:** the robot gets an exponential average of the actions,
   `s = action_smoothing·s + (1 − action_smoothing)·pwm` (0.5 by default). A jump from 0
   to full speed reaches the robot as 128, 191, 223, … over successive steps. `s` restarts
   from 0 after `reset()`, after a safety stop, and after a pause long enough for the
   robot's hold timeout to stop it.
3. **Deadzone**, on the smoothed PWM only: below `pwm_deadzone` (120) in magnitude it becomes
   0, so the robot never gets a PWM between 0 and 120, not even while ramping. It doesn't
   remap: an action of 0.4 (PWM 102) never gets the track going, 0.5 (PWM 128) does. Slowing
   from 200 to 100 passes through 150 and 125 before the track stops. Stopping is quick
   too: from below about PWM 240, the first step after a stop already sends 0.
4. **Safety layer**, if on: it judges this PWM, the one the robot would really get, and may
   replace it with 0.

`pwm_deadzone=0, action_smoothing=0` gives the plain linear mapping.

**Reward / termination** depend on the task you're training, so you supply them:
`RoboGym(reward_fn=lambda obs, action, info: ..., terminated_fn=lambda obs, info: ...)`.
The defaults are a reward of 0.0 and never terminating. Limit episode length with
`gym.make("RoboGym-v0", max_episode_steps=200)` or `gymnasium.wrappers.TimeLimit`.

`info` also carries the cumulative odometry since `reset()` (`pose`,
`motor_turns_total`), `accel_raw`, `battery_v`, `imu_ok`, `watchdog_stopped`,
`robot_rebooted`, `latency`, `image_age`, the PWM the robot actually received (`pwm`),
the action's PWM before smoothing and the deadzone (`pwm_target`), and with the safety
layer on, `safety_trip`.

### What one step does

1. `step(a)` turns `a` into PWM (smoothing, deadzone, safety: see above) and sends
   `{A,seq,left,right,duration}` to the robot.
2. The robot applies the speeds, waits `action_duration` (0.1 s by default), then sends
   its sensor packet.
3. The env takes the newest camera frame that arrived after the action finished,
   computes the deltas and returns.
4. The robot **keeps executing `a`** until the next command arrives. If no command
   arrives within `hold_timeout` (0.5 s) it stops by itself, so a crashed or paused
   training script never leaves the robot driving into a wall.

The robot sends cumulative values (odometry, turns, its clock) and the env differences
them. So each observation covers everything since the previous one, including time the
agent spent thinking, and a lost packet can't silently drop motion.

### Safety layer

`RoboGym(safety=SafetyConfig())` adds checks on the PC that stop the robot when its
sensors disagree with what it was told to do:

| trip | fires when |
|---|---|
| `stall` | a track is above PWM 100, the forward command (the tracks' mean PWM) changed by more than 50, but `accel` x stayed within ±0.3 m/s² |
| `no_turn` | a track is above PWM 100, the tracks differ by more than 30, but the gyro's yaw rate stayed within ±2 deg/s |
| `obstacle` | the ultrasonic range is under 0.15 m while driving forward. A forward command is also refused while the last reading is that close, but reversing and spinning stay allowed, so the robot can back away |

A trip **latches**. The env stops the robot at once, then every step sends zero PWM until
`env.safety.clear()`. `reset()` does not clear it. `info["safety_trip"]` holds the latched
`SafetyTrip` (`.trigger`, `.message`), or `None` while armed. To end episodes on a trip, pass
`terminated_fn=lambda obs, info: info["safety_trip"] is not None`.

The checks judge the PWM the robot actually receives (after the smoothing and deadzone), not
the action passed to `step()`. Smoothing spreads a speed change over several steps, so the
`stall` check, which needs a change of more than 50 in one step, sees fewer of them. The
checks run once per step on the PC, so they react a step late (~0.2 s). They add to the
robot's own hold-timeout stop; they don't replace it. The thresholds are fields of
`SafetyConfig`. Tune `min_accel` and `min_yaw_rate` for your robot with the manual-control
plots: *accel x vs ΔPWM* and *yaw rate vs PWM L−R* show what it really does.

## Things to know about this hardware

* **The motors have no encoders.** Each motor connector on the shield has only two
  wires, and every UNO pin is already in use. `motor_turns` and the distance part of
  `delta_pose` are therefore integrated on the robot from a motor model: PWM, battery
  voltage, friction deadband and a speed lag. If the tracks are blocked, the estimate
  keeps counting. **Heading (`dθ`) is measured** by the MPU6050 gyro. Calibrate the model
  (below) if you need real units. If you fit encoder motors, replace the motor-model
  lines at the top of `tick()` in the sketch with encoder counts. The sketch leaves D2,
  D4, D9 and A0–A2 unused, but the shield wires them to its button, RGB LED, IR receiver
  and line sensors.
* **About 5 steps per second.** ELEGOO's ESP32 firmware links to the UNO at 9600 baud.
  An action plus its reply is ~95 characters, about 100 ms on the wire, on top of
  `action_duration`. With the defaults, a step takes ~0.19–0.22 s.
* **The robot is the WiFi access point.** While your PC is on `ELEGOO-xxxxxxxx` it
  usually has no internet, unless it has a second network adapter.
* **One client at a time.** Close the ELEGOO phone app before connecting.

## Setup

### 1. Try it without the robot

```bash
pip install -e ".[dev]"
python examples/manual_control.py --fake    # drive a simulated tank in a 3 m arena yourself
python examples/random_agent.py --fake      # ...or let random actions drive it
pytest                                      # ~45 s, includes running the real sketch logic
```

`robotgym.fake_robot` emulates the ESP32 bridge (including its heartbeat rules), the
sketch's protocol on a simulated tank, and the camera server. You can develop agents
against it: `python -m robotgym.fake_robot` and connect with the printed arguments.

### 2. Flash the robot

1. Open `arduino/RoboGym/RoboGym.ino` in the Arduino IDE. It needs only the built-in
   `Wire` and `Servo` libraries.
2. Check `MOTOR_DRIVER` at the top. Kits sold since 2022 have the **TB6612** (the
   default); older shields have the DRV8835. `ReadMeFirst.pdf` in ELEGOO's download
   shows how to tell them apart.
3. Set the shield's **Upload-Cam switch to "Upload"**, choose board **Arduino Uno**,
   upload.
4. Set the switch **back to "Cam"**. Otherwise the ESP32 can't talk to the UNO.

Re-flash whenever you update this repo and the sketch changed. If the robot runs an
older sketch, the env still works but logs a warning, and anything that sketch doesn't
send (such as `accel`, added in protocol v2) reads 0.

At boot the robot centres the camera/ultrasonic servo and measures the gyro bias, so
keep it still for ~2 seconds after switching it on.

### 3. Run it

1. Switch the robot on and join your PC to its WiFi `ELEGOO-xxxxxxxx` (no password).
2. **The first time, put the robot on a stand so the tracks are off the ground.**
3. `python examples/manual_control.py` to drive it by hand, or
   `python examples/random_agent.py` for random actions.

## Manual control

`examples/manual_control.py` drives the robot through the gym env from a window:

* **Live camera feed** at the camera's full rate (~20 fps), plus the **image the agent
  gets** in its observation, and every value of the latest observation and `info`.
* **Three levers** set the same `[left, right]` action:
  * **the joystick**: up/down for forward/reverse, left/right to turn, so one drag
    controls both tracks;
  * **the left and right levers**: one track each. They also show what each track is
    getting from the joystick, and the joystick moves to match them.

  **Let go of a lever and it springs back, which stops the robot.** The joystick stops
  both tracks; a side lever stops only its own track. Tick *Hold levers* to make them stay
  where you leave them. For tank-style driving, set the left lever, then the right one.
  *Max speed* scales all three. The levers go through the env's smoothing and deadzone like
  any agent's actions, and the side levers' labels show the PWM the robot actually got in the
  latest step. The readout shows that next to the target PWM, so you can watch the ramp.
* **STOP** or **Space** always stops. *Reset odometry* calls `env.reset()`, and **Esc**
  or closing the window stops the robot and disconnects.
* **The safety layer is on.** The panel under the levers says *Safety armed* while all is
  well. When the layer stops the robot, the panel **latches red** with the reason and the
  levers centre. The robot stays stopped, whatever the levers say, until you press
  **Clear**. Clearing never restarts the robot by itself. `--no-safety` drives without the
  layer.
* **A plot window** opens beside it (the *Plots* button reopens it). It shows, one point per
  env step:
  * the PWM sent to the left and right tracks over the last 20 s;
  * the accelerometer (`accel`, x/y/z) over the same window;
  * scatter plots of accel x, y and z against the PWM of both tracks;
  * scatter plots of accel x, y and z against **ΔPWM**, the change in PWM since the previous
    step. A command change takes effect at the start of a step, so it lines up with that
    step's acceleration, which makes this the clearer view of how the tracks respond. It's
    skipped after gaps over 1 s (pauses, reconnects, resets);
  * a turning column: the gyro's **yaw rate** (heading change ÷ step time, deg/s,
    counter-clockwise positive) against the **left − right PWM difference**, and the change
    in yaw rate against the change in that difference. Right > left turns the robot
    counter-clockwise, so expect a falling slope.

  Overlapping series use different marker shapes, e.g. left = dot and right = ring, so
  equal values stay visible. *Clear* starts the plots fresh. `--no-plots` skips the plot
  window at start, and `--plot-seconds 30` changes the time window.

The env runs in a background thread and steps it as fast as the robot answers (~5 Hz),
so the window never waits for the robot. If the window freezes, that thread sends
zero speed. Options: `--fake`, `--max-speed 0.6`, `--min-pwm 40` (small lever movements
drive instead of sitting in the motor deadband), `--action-duration`, `--pwm-deadzone 120`,
`--smoothing 0.5` (0 turns either off), `--no-safety`. It needs
tkinter, which ships with Python. On Debian/Ubuntu: `sudo apt install python3-tk`.

## Calibration

The motor model's defaults are rough guesses. `examples/calibrate.py` measures better
values using the gyro and accelerometer. Run it on the floor, in this order:

```bash
python examples/calibrate.py spin                  # is the gyro sign right?
python examples/calibrate.py deadband              # PWM where the tracks start moving
python examples/calibrate.py straight --pwm 200 --deadband 42   # then type the distance you measured
python examples/calibrate.py imu                   # which accelerometer axis points forward
```

Then pass the results in:

```python
from robotgym import RoboGym, RobotCalibration
env = RoboGym(calibration=RobotCalibration(max_rpm=165, deadband_pwm=42, imu_forward="+x"))
```

The values are sent to the robot each time the env connects, so recalibrating never
needs a re-flash. Measuring `sprocket_diameter` with calipers (default 43 mm) also
improves `motor_turns`.

## Useful options

| argument | default | |
|---|---|---|
| `action_duration` | 0.1 | seconds each action runs before the robot reports |
| `hold_timeout` | 0.5 | robot stops if no command arrives this long after its reply |
| `camera` | `"stream"` | `"snapshot"` fetches `/capture` per step (slower, no background traffic); `"none"` drops the image |
| `image_shape` | (120, 160) | observation image size (height, width) |
| `camera_framesize` | 4 | ESP32 resolution index: 4 = 320×240 on ELEGOO's firmware; `None` leaves it alone |
| `min_pwm` | 0 | PWM for the smallest non-zero action. Set it near the deadband so small actions still move the robot |
| `pwm_deadzone` | 120 | smoothed track PWM below this is sent as 0 (0 = off) |
| `action_smoothing` | 0.5 | weight of the previous PWM in the per-step exponential average, 0 ≤ x < 1 (0 = off) |
| `reset_settle_time` | 0.5 | pause after stopping in `reset()` before zeroing |
| `calibrate_gyro_on_reset` | True | re-measure the gyro bias each reset (~0.5 s, robot must be still) |
| `safety` | None | `SafetyConfig()` turns on the safety layer (see above) |
| `serial_port` | None | e.g. `"/dev/ttyUSB0"`: drive over USB instead of WiFi (bench testing, switch on "Upload", `pip install pyserial`) |
| `render_mode` | None | `"rgb_array"` returns the full camera frame; `"human"` opens a window (`pip install opencv-python`) |

`reset()` stops the robot and zeroes its odometry. It does **not** move the robot back
to a start position; subclass `RoboGym` and override `reset()` if your task needs that.

With Stable-Baselines3, the Dict observation works with `MultiInputPolicy`:

```python
from stable_baselines3 import PPO
model = PPO("MultiInputPolicy", RoboGym(reward_fn=my_reward), n_steps=256).learn(5_000)
```

## Wire protocol

ASCII frames `{...}`, comma-separated integers, no spaces. The ESP32 bridge strips
spaces and flushes on `}`, which is why the format looks the way it does. The sketch
header has the full description.

| direction | frame | meaning |
|---|---|---|
| PC → robot | `{A,seq,left,right,dur_ms}` | track PWM −255…255; reply `O` after `dur_ms` |
| | `{R,seq,calib}` | stop, optionally recalibrate the gyro, zero odometry; reply `O` |
| | `{C,seq,hold_ms,rpm×10,deadband,tau_ms,sprocket_0.1mm,track_mm,gyro_sign}` | configure; reply `O` |
| | `{P,seq}` | reply `O` now |
| robot → PC | `{O,seq,t_us,x,y,θ,turnsL,turnsR,dist_mm,batt_cV,flags,ax,ay,az}` | x, y in 0.1 mm; θ in 1e-4 rad; turns in 1e-3; flags: 1 IMU ok, 2 watchdog stop, 4 no echo, 8 first packet since boot, 16 gyro calibration failed; a: accelerometer mean since the last packet, mm/s², chip axes, gravity included (the reply to R covers only the still period, so it is the at-rest reading) |
| | `{H,RoboGym,2}` | boot banner with the protocol version |
| | `{E,seq,code}` | rejected command |

The PC also sends `{Heartbeat}` every 0.5 s, which the ESP32 requires. Without it the
ESP32 drops the connection after ~4 s. When the PC disconnects, the ESP32 sends
`{"N":100}` and the sketch stops the motors.

## Tests

`pytest` runs six layers:

* `tests/test_protocol.py`: frame parsing and pose math.
* `tests/test_env.py`: the full env against the fake robot. Covers spaces, motion signs,
  odometry consistency, the hold-timeout stop, heartbeats over idle periods, reconnects,
  retries, reboot detection, the camera modes and the gymnasium env checker.
* `tests/test_safety.py`: the safety layer's checks on their own, then the env stopping
  the fake robot at a wall, against a blocked reverse, and on tracks that slip instead of turning.
* `tests/test_sketch_native.py`: compiles **the real `RoboGym.ino`** for your PC against
  small Arduino stand-ins (`tests/native/`) and drives it with the env. Includes a
  simulated `micros()` wraparound. Needs `g++`; skipped otherwise.
* `tests/test_ultrasonic_sketch.py`: compiles `UltrasonicTest.ino` the same way and runs it
  against a fake HC-SR04 with each fault (trigger or echo wire open, flapping contact, stuck
  sensor, nothing in range), checking it names the right one. Needs `g++`.
* `tests/test_manual_control.py`: the manual-control window driving the fake robot,
  with the window hidden. Skipped where Tk has no display.

## Troubleshooting

`python examples/diagnose_link.py` (on the robot's WiFi, ideally with the USB cable also
plugged in, `pip install pyserial`) tests each part of the link: the ESP32 bridge
firmware, the Arduino's answers over WiFi, and over USB whether the Arduino booted and
received the commands. It then names the broken part.

| symptom | fix |
|---|---|
| `cannot reach the robot at tcp://192.168.4.1:100` | PC not on the robot's WiFi, or robot off |
| `the robot runs ELEGOO's factory firmware` | flash `RoboGym.ino` (switch on Upload, then back to Cam) |
| `connected ... but the robot did not answer` | Upload-Cam switch left on "Upload"; or the phone app is connected |
| `no image from the camera` | ESP32 module unplugged or still booting; or use `camera="none"` |
| tracks run the wrong way | set `INVERT_LEFT_MOTOR` / `INVERT_RIGHT_MOTOR` in the sketch, or check `MOTOR_DRIVER` |
| `dθ` has the wrong sign | `calibrate.py spin`, then `RobotCalibration(gyro_sign=-1)` |
| `robot_rebooted` in `info` | battery brown-out when the motors start: charge the battery |
| `no distance reading` flag | the ultrasonic sensor isn't answering. Re-flash `RoboGym.ino` if it's older than 2026-11-22 (echo pull-up), else run the test sketch below |
| new features missing, or errors about keys like `accel` | `robotgym` was installed as a copy (`pip install .`): reinstall with `pip install -e .` so it follows the repo |

### Ultrasonic test sketch

`arduino/UltrasonicTest/UltrasonicTest.ino` tests the HC-SR04 and its wiring on their own.
Flash it in place of `RoboGym.ino` (switch on "Upload"), keep the USB cable in, and open the
Serial Monitor at **115200 baud**. It pings 10 times a second, prints each result, and every
2 s adds a summary with a verdict. The motors stay off.

Point the sensor at a wall 0.3–2 m away, then wiggle the cable and press on each plug and
header one spot at a time. Failures that follow your hand mark the bad contact. Send:

* `s` to sweep the head 60–120° while pinging, which bends the cable the way driving does;
* `p` to switch the echo pin's pull-up off and on. If pings only succeed with it on, the
  echo pulse's high level is weak, usually a worn contact in the echo or 5 V line.
  `RoboGym.ino` uses the pull-up too (since 2026-11-22), so such a sensor still reads;
* `f` to ping every 60 ms, as `RoboGym.ino` does, instead of 100 ms;
* `r` to reset the counters.

| result | meaning |
|---|---|
| `OK` | a range of 2 cm – 4 m |
| `NO ANSWER` | echo idles low, so the sensor has power and the echo wire is fine, but it ignores the trigger: **trigger wire** open, or the sensor is dead |
| `ECHO HIGH` | nothing holds echo low: **echo wire or the sensor's 5 V / GND wire** open, or the sensor locked up |
| `NO ECHO` | the sensor works but heard nothing within 4 m: not a wiring fault |
| `STUCK HIGH` | echo went high and stayed high: the sensor locked up (power-cycle), or echo is shorted to 5 V |
| `GLITCH` | a pulse under 2 cm: noise, usually a bouncing contact |

`NO ANSWER` and `ECHO HIGH` are what the env reports as `no distance reading`. A mix of `OK`
with either of them, or with `GLITCH`, is a contact that keeps making and breaking. The
summary says which line it is. Re-flash `RoboGym.ino` and set the switch back to "Cam"
when done.
