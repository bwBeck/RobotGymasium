import math

import pytest

from robotgym.protocol import (
    FLAG_BOOT,
    FLAG_IMU_OK,
    FLAG_NO_ECHO,
    FrameParser,
    Hello,
    RobotError,
    RobotState,
    encode_command,
    is_stock_firmware_reply,
    parse_frame,
    relative_pose,
    u32_diff,
)


def state(x=0.0, y=0.0, theta=0.0, t_us=0):
    return RobotState(0, t_us, x, y, theta, 0.0, 0.0, None, 7.4, 0)


def test_encode_command():
    assert encode_command("A", 7, 120, -120, 100) == b"{A,7,120,-120,100}"
    assert encode_command("P", 1) == b"{P,1}"
    assert encode_command("R", 3, True) == b"{R,3,1}"


def test_parser_splits_frames_across_chunks_and_drops_noise():
    p = FrameParser()
    assert p.feed(b"MPU6050_chip_id: 104\r\n{O,1,2") == []
    assert p.feed(b",3}{Heart") == ["O,1,2,3"]
    assert p.feed(b"beat}junk") == ["Heartbeat"]


def test_parser_resyncs_on_open_brace_and_drops_oversized_frames():
    p = FrameParser(max_len=8)
    assert p.feed(b"{trunc{P,1}") == ["P,1"]
    assert p.feed(b"{0123456789}{ok}") == ["ok"]


def test_parse_observation():
    msg = parse_frame("O,12,4294967295,12345,-500,31416,2500,-1250,876,784,%d" % (FLAG_IMU_OK | FLAG_NO_ECHO))
    assert isinstance(msg, RobotState)
    assert msg.seq == 12 and msg.t_us == 0xFFFFFFFF
    assert msg.x == pytest.approx(1.2345) and msg.y == pytest.approx(-0.05)
    assert msg.theta == pytest.approx(3.1416)
    assert (msg.turns_left, msg.turns_right) == (pytest.approx(2.5), pytest.approx(-1.25))
    assert msg.distance == pytest.approx(0.876)
    assert msg.battery_v == pytest.approx(7.84)
    assert msg.imu_ok and msg.no_echo and not msg.booted and not msg.watchdog_stopped


def test_parse_observation_invalid_distance_and_boot_flag():
    msg = parse_frame("O,1,0,0,0,0,0,0,-1,790,%d" % FLAG_BOOT)
    assert msg.distance is None and msg.booted


def test_parse_other_frames():
    assert parse_frame("H,RoboGym,1") == Hello("RoboGym", 1)
    assert parse_frame("E,9,1") == RobotError(9, 1)
    assert parse_frame("Heartbeat") == "Heartbeat"
    assert parse_frame("O,1,2,3") is None  # wrong field count
    assert parse_frame("O,a,b,c,d,e,f,g,h,i,j") is None


def test_stock_firmware_replies_are_recognised():
    assert is_stock_firmware_reply("ok")
    assert is_stock_firmware_reply("12_ok")
    assert is_stock_firmware_reply("3_false")
    assert not is_stock_firmware_reply("Heartbeat")
    assert not is_stock_firmware_reply("WA_OK")


def test_u32_diff_handles_wraparound():
    assert u32_diff(100, 50) == 50
    assert u32_diff(5, 0xFFFFFFFF - 4) == 10


def test_relative_pose_is_expressed_in_the_previous_robot_frame():
    # Facing +y (90 deg) and moving +y in the odometry frame = driving straight ahead.
    dx, dy, dth = relative_pose(state(1.0, 1.0, math.pi / 2), state(1.0, 1.5, math.pi / 2 + 0.1))
    assert dx == pytest.approx(0.5) and dy == pytest.approx(0.0, abs=1e-12)
    assert dth == pytest.approx(0.1)
    # Facing +x and moving +y = sliding to the left.
    dx, dy, _ = relative_pose(state(0.0, 0.0, 0.0), state(0.0, 0.2, 0.0))
    assert dx == pytest.approx(0.0) and dy == pytest.approx(0.2)


def test_relative_pose_does_not_wrap_heading():
    _, _, dth = relative_pose(state(theta=3.1), state(theta=3.3))
    assert dth == pytest.approx(0.2)
