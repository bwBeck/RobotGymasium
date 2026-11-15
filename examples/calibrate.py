"""Measure the motor-model constants for RobotCalibration.

The Conqueror's motors have no encoders, so RoboGym.ino estimates sprocket turns (and
the distance part of the odometry) from the PWM it applies. The gyro does measure
rotation, which is enough to calibrate that estimate. Run these on the floor with some
room, in this order, and pass the printed values to RoboGym(calibration=...):

    python examples/calibrate.py spin                  # gyro sign
    python examples/calibrate.py deadband              # PWM where the tracks start moving
    python examples/calibrate.py straight --pwm 200    # drive, then type the measured distance
    python examples/calibrate.py imu                   # which accelerometer axis points forward

Add --fake to try any of them on the simulated robot.
"""

import argparse
import math

import numpy as np

from robotgym import RobotCalibration, RoboGym
from robotgym.fake_robot import FakeRobotServer


def pwm(left: int, right: int) -> np.ndarray:
    return np.array([left / 255.0, right / 255.0], dtype=np.float32)  # max_pwm=255, min_pwm=0


def spin(env: RoboGym, args, fake) -> None:
    env.action_duration = args.seconds
    env.reset()
    env.step(pwm(-args.pwm, args.pwm))  # left track back, right forward = turn left
    env.action_duration = 0.5
    _, _, _, _, info = env.step(pwm(0, 0))  # stop and let it settle
    turned = float(info["pose"][2])
    print(f"left spin at PWM {args.pwm} for {args.seconds} s: gyro says {math.degrees(turned):+.0f} deg")
    if turned > 0.2:
        print("OK - left turns read positive: keep gyro_sign=1")
    elif turned < -0.2:
        print("Left turns read negative: use RobotCalibration(gyro_sign=-1)")
    else:
        print("The robot barely turned - raise --pwm, check the battery, or check imu_ok")


def deadband(env: RoboGym, args, fake) -> None:
    env.action_duration = 0.4
    _, info = env.reset()
    battery = info["battery_v"]
    moving = []  # (pwm, deg/s) once the robot is clearly turning
    for p in range(10, 256, 5):
        obs, *_ = env.step(pwm(-p, p))  # spinning in place shows up on the gyro
        rate = abs(math.degrees(obs["delta_pose"][2] / max(float(obs["robot_dt"][0]), 1e-3)))
        print(f"PWM {p:3d}: {rate:6.1f} deg/s")
        if rate > 3.0:
            moving.append((p, rate))
        if rate > 40.0 and len(moving) >= 3:
            break
    env.step(pwm(0, 0))
    if len(moving) < 2:
        print("never really started turning - is the battery switched on?")
        return
    # Speed is ~linear in PWM above the deadband: extrapolate the moving points to zero.
    slope, intercept = np.polyfit([m[0] for m in moving], [m[1] for m in moving], 1)
    zero = -intercept / slope
    # The sketch compares the deadband to PWM scaled by battery voltage / 7.4 V.
    print(f"turning starts near PWM {moving[0][0]}; the linear fit reaches zero at PWM {zero:.0f} "
          f"({battery:.2f} V) -> RobotCalibration(deadband_pwm={round(zero * battery / 7.4)})")


def straight(env: RoboGym, args, fake) -> None:
    env.action_duration = args.seconds
    env.reset()
    start = (fake.robot.x, fake.robot.y) if fake else None
    if not fake:
        input("Mark where the robot is, then press Enter to drive forward... ")
    env.step(pwm(args.pwm, args.pwm))
    env.action_duration = 0.5
    _, _, _, _, info = env.step(pwm(0, 0))
    estimated = math.hypot(info["pose"][0], info["pose"][1])
    if fake:
        measured = math.hypot(fake.robot.x - start[0], fake.robot.y - start[1])
        print(f"(simulator) true distance: {measured:.3f} m")
    else:
        measured = float(input("Distance driven, in metres: "))
    current = env.calibration.max_rpm
    print(f"the model estimated {estimated:.3f} m with max_rpm={current}")
    if estimated < 1e-3:
        print("the model thinks the robot did not move - lower deadband_pwm or raise --pwm")
        return
    # Estimated distance and turns are proportional to max_rpm, so scale it.
    print(f"-> RobotCalibration(max_rpm={current * measured / estimated:.1f})  "
          f"(with sprocket_diameter={env.calibration.sprocket_diameter})")


def imu(env: RoboGym, args, fake) -> None:
    """Drive off sharply a few times; the chip axis that jumps is the forward one."""
    votes = []
    for trial in range(3):
        env.action_duration = 0.15
        env.reset()  # stops the robot; the reset's reading is the at-rest one
        rest = env.last_state.accel
        if rest is None:
            print("the robot's sketch has no accelerometer output yet - reflash arduino/RoboGym/RoboGym.ino")
            return
        env.step(pwm(args.pwm, args.pwm))  # speeding up from standstill
        change = np.array(env.last_state.accel) - rest  # chip axes, gravity cancelled
        env.action_duration = 0.6
        env.step(pwm(0, 0))  # stop
        axis = int(np.argmax(np.abs(change[:2])))
        votes.append(("+" if change[axis] > 0 else "-") + "xy"[axis])
        print(f"run {trial + 1}: change in chip x/y/z = {np.round(change, 2)} m/s^2 -> forward is {votes[-1]}")
    best = max(set(votes), key=votes.count)
    if votes.count(best) < 2:
        print("inconsistent runs - try a higher --pwm on a smooth floor")
        return
    print(f"-> RobotCalibration(imu_forward={best!r})")
    up = "up" if rest[2] > 0 else "down"
    print(f"(at rest the chip's z axis reads {rest[2]:+.2f} m/s^2: it points {up}; "
          f"gyro_sign {'1' if up == 'up' else '-1'} should match that)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=["spin", "deadband", "straight", "imu"])
    parser.add_argument("--pwm", type=int, default=180, help="PWM for spin / straight / imu (0-255)")
    parser.add_argument("--seconds", type=float, default=1.5, help="drive time for spin / straight")
    parser.add_argument("--deadband", type=float, default=RobotCalibration.deadband_pwm,
                        help="your measured deadband (for straight)")
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--fake", action="store_true", help="use the simulated robot")
    args = parser.parse_args()

    run = {"spin": spin, "deadband": deadband, "straight": straight, "imu": imu}[args.mode]
    calibration = RobotCalibration(deadband_pwm=args.deadband)
    kwargs = {"camera": "none", "calibration": calibration, "hold_timeout": 1.0}
    if args.fake:
        with FakeRobotServer() as fake:
            env = RoboGym(**fake.env_kwargs(), **kwargs)
            try:
                run(env, args, fake)
            finally:
                env.close()
    else:
        env = RoboGym(host=args.host, **kwargs)
        try:
            run(env, args, None)
        finally:
            env.close()


if __name__ == "__main__":
    main()
