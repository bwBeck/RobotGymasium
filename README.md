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

**Action:** `Box(-1, 1, shape=(2,))` = `[left, right]` track speed. The sign is the
direction, and ±1 is full PWM (`max_pwm`).

**Reward / termination** depend on the task you're training, so you supply them:
`RoboGym(reward_fn=lambda obs, action, info: ..., terminated_fn=lambda obs, info: ...)`.
The defaults are a reward of 0.0 and never terminating. Limit episode length with
`gym.make("RoboGym-v0", max_episode_steps=200)` or `gymnasium.wrappers.TimeLimit`.

`info` also carries the cumulative odometry since `reset()` (`pose`,
`motor_turns_total`), `battery_v`, `imu_ok`, `watchdog_stopped`, `robot_rebooted`,
`latency`, `image_age` and the PWM actually sent (`pwm`).

### What one step does

1. `step(a)` sends `{A,seq,left,right,duration}` to the robot.
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
  An action plus its reply is ~80 characters, about 80 ms on the wire, on top of
  `action_duration`. With the defaults, a step takes ~0.17–0.2 s.
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
* **The lever**: drag it up/down for forward/reverse and left/right to turn. One drag
  controls both tracks, and the bars beside it show the `[left, right]` action being sent.
  **Let go and it springs back to the centre, which stops the robot.** Tick *Hold lever*
  to make it stay where you leave it. *Max speed* scales the lever's range.
* **STOP** or **Space** always stops. *Reset odometry* calls `env.reset()`, and **Esc**
  or closing the window stops the robot and disconnects.

The env runs in a background thread and steps it as fast as the robot answers (~5 Hz),
so the window never waits for the robot. If the window freezes, that thread sends
zero speed. Options: `--fake`, `--max-speed 0.6`, `--min-pwm 40` (small lever movements
drive instead of sitting in the motor deadband), `--action-duration`. It needs
tkinter, which ships with Python. On Debian/Ubuntu: `sudo apt install python3-tk`.

## Calibration

The motor model's defaults are rough guesses. `examples/calibrate.py` measures better
values using the gyro. Run it on the floor, in this order:

```bash
python examples/calibrate.py spin                  # is the gyro sign right?
python examples/calibrate.py deadband              # PWM where the tracks start moving
python examples/calibrate.py straight --pwm 200 --deadband 42   # then type the distance you measured
```

Then pass the results in:

```python
from robotgym import RoboGym, RobotCalibration
env = RoboGym(calibration=RobotCalibration(max_rpm=165, deadband_pwm=42))
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
| `reset_settle_time` | 0.5 | pause after stopping in `reset()` before zeroing |
| `calibrate_gyro_on_reset` | True | re-measure the gyro bias each reset (~0.5 s, robot must be still) |
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
| robot → PC | `{O,seq,t_us,x,y,θ,turnsL,turnsR,dist_mm,batt_cV,flags}` | x, y in 0.1 mm; θ in 1e-4 rad; turns in 1e-3; flags: 1 IMU ok, 2 watchdog stop, 4 no echo, 8 first packet since boot, 16 gyro calibration failed |
| | `{H,RoboGym,1}` | boot banner |
| | `{E,seq,code}` | rejected command |

The PC also sends `{Heartbeat}` every 0.5 s, which the ESP32 requires. Without it the
ESP32 drops the connection after ~4 s. When the PC disconnects, the ESP32 sends
`{"N":100}` and the sketch stops the motors.

## Tests

`pytest` runs four layers:

* `tests/test_protocol.py`: frame parsing and pose math.
* `tests/test_env.py`: the full env against the fake robot. Covers spaces, motion signs,
  odometry consistency, the hold-timeout stop, heartbeats over idle periods, reconnects,
  retries, reboot detection, the camera modes and the gymnasium env checker.
* `tests/test_sketch_native.py`: compiles **the real `RoboGym.ino`** for your PC against
  small Arduino stand-ins (`tests/native/`) and drives it with the env. Includes a
  simulated `micros()` wraparound. Needs `g++`; skipped otherwise.
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
