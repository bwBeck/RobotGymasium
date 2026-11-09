"""Wire protocol shared with ``arduino/RoboGym/RoboGym.ino``.

Frames are ASCII ``{...}`` with comma-separated integers and no spaces. The stock
ELEGOO ESP32 bridge strips spaces and flushes on ``}``, and its link to the UNO runs at
9600 baud, so the format is deliberately compact and text-only. The full table lives in
the sketch header and in README.md.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

PROTOCOL_VERSION = 1
HEARTBEAT = "Heartbeat"
MAX_FRAME_LEN = 256

# Bits of the ``flags`` field of an O frame.
FLAG_IMU_OK = 1
FLAG_WATCHDOG = 2
FLAG_NO_ECHO = 4
FLAG_BOOT = 8
FLAG_CALIB_FAILED = 16

ERROR_CODES = {1: "malformed command"}


@dataclass(frozen=True)
class RobotState:
    """One O frame: the robot's cumulative state since its last reset (R frame)."""

    seq: int
    t_us: int  # robot micros() when the packet was built; wraps at 2**32
    x: float  # dead-reckoned position [m], x forward / y left at reset
    y: float
    theta: float  # heading [rad], counter-clockwise positive, unwrapped
    turns_left: float  # cumulative sprocket turns (estimated, the motors have no encoders)
    turns_right: float
    distance: float | None  # ultrasonic range [m]; None = no valid reading
    battery_v: float
    flags: int

    @property
    def imu_ok(self) -> bool:
        return bool(self.flags & FLAG_IMU_OK)

    @property
    def watchdog_stopped(self) -> bool:
        """The robot's hold timeout stopped the motors since the previous packet."""
        return bool(self.flags & FLAG_WATCHDOG)

    @property
    def no_echo(self) -> bool:
        """The last ping found nothing within range (``distance`` is the max range)."""
        return bool(self.flags & FLAG_NO_ECHO)

    @property
    def booted(self) -> bool:
        """First packet since the robot powered up (or browned out)."""
        return bool(self.flags & FLAG_BOOT)

    @property
    def calibration_failed(self) -> bool:
        return bool(self.flags & FLAG_CALIB_FAILED)


@dataclass(frozen=True)
class Hello:
    """Boot banner ``{H,RoboGym,<version>}``."""

    name: str
    version: int


@dataclass(frozen=True)
class RobotError:
    """``{E,seq,code}``: the robot rejected a command."""

    seq: int
    code: int

    @property
    def message(self) -> str:
        return ERROR_CODES.get(self.code, f"error {self.code}")


Message = Union[RobotState, Hello, RobotError, str]


def encode_command(kind: str, seq: int, *fields: int) -> bytes:
    """``encode_command("A", 7, 120, -120, 100) == b"{A,7,120,-120,100}"``."""
    parts = [kind, str(int(seq))]
    parts.extend(str(int(f)) for f in fields)
    return ("{" + ",".join(parts) + "}").encode("ascii")


class FrameParser:
    """Splits a byte stream into the contents of ``{...}`` frames.

    Bytes outside braces (boot noise, debug prints) are dropped. A ``{`` always starts
    a new frame, which resynchronises after a corrupted or truncated frame.
    """

    def __init__(self, max_len: int = MAX_FRAME_LEN):
        self.max_len = max_len
        self._buf = bytearray()
        self._in_frame = False

    def feed(self, data: bytes) -> list[str]:
        frames = []
        for byte in data:
            if byte == 0x7B:  # "{"
                self._buf.clear()
                self._in_frame = True
            elif not self._in_frame:
                continue
            elif byte == 0x7D:  # "}"
                frames.append(self._buf.decode("ascii", "replace"))
                self._buf.clear()
                self._in_frame = False
            else:
                self._buf.append(byte)
                if len(self._buf) > self.max_len:
                    self._buf.clear()
                    self._in_frame = False
        return frames


def parse_frame(content: str) -> Message | None:
    """Decodes a frame's content (without braces).

    Returns a :class:`RobotState`, :class:`Hello` or :class:`RobotError`; ``None`` for a
    malformed robot frame; or the raw string for anything else (``"Heartbeat"``, replies
    from the stock ELEGOO firmware such as ``"ok"``, ESP32 chatter).
    """
    head, _, rest = content.partition(",")
    try:
        if head == "O":
            values = [int(v) for v in rest.split(",")]
            if len(values) != 10:
                return None
            seq, t_us, x, y, th, tl, tr, dist, batt, flags = values
            return RobotState(
                seq=seq,
                t_us=t_us & 0xFFFFFFFF,
                x=x / 1e4,
                y=y / 1e4,
                theta=th / 1e4,
                turns_left=tl / 1e3,
                turns_right=tr / 1e3,
                distance=None if dist < 0 else dist / 1000.0,
                battery_v=batt / 100.0,
                flags=flags,
            )
        if head == "H":
            name, _, version = rest.partition(",")
            return Hello(name, int(version))
        if head == "E":
            seq, code = (int(v) for v in rest.split(","))
            return RobotError(seq, code)
    except ValueError:
        return None
    return content


def is_stock_firmware_reply(content: str) -> bool:
    """True for the replies of ELEGOO's factory sketch (``{ok}``, ``{<id>_ok}``, ...)."""
    return content == "ok" or content.endswith(("_ok", "_true", "_false"))


def u32_diff(newer: int, older: int) -> int:
    """Difference of two wrapping 32-bit timestamps (micros() wraps every ~71 minutes)."""
    return (newer - older) & 0xFFFFFFFF


def relative_pose(prev: RobotState, cur: RobotState) -> tuple[float, float, float]:
    """Motion from ``prev`` to ``cur`` expressed in the robot frame at ``prev``.

    Returns ``(dx, dy, dtheta)``: metres forward, metres to the left, radians
    counter-clockwise. The robot's heading is unwrapped, so dtheta needs no wrapping.
    """
    dxw = cur.x - prev.x
    dyw = cur.y - prev.y
    c = math.cos(prev.theta)
    s = math.sin(prev.theta)
    return c * dxw + s * dyw, -s * dxw + c * dyw, cur.theta - prev.theta
