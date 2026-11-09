"""Byte transports to the robot and the request/response link on top of them."""

from __future__ import annotations

import logging
import socket
import threading
import time
from contextlib import suppress

from .protocol import (
    HEARTBEAT,
    FrameParser,
    Hello,
    RobotError,
    RobotState,
    encode_command,
    is_stock_firmware_reply,
    parse_frame,
)

logger = logging.getLogger("robotgym")

_POLL_S = 0.1


class ProtocolError(RuntimeError):
    """The robot answered, but rejected the command (sketch / package version mismatch?)."""


class TcpTransport:
    """TCP connection to the ESP32 camera module's bridge (stock firmware, port 100)."""

    needs_heartbeat = True  # the ESP32 drops clients that stay silent for ~4 s

    def __init__(self, host: str, port: int = 100, connect_timeout: float = 5.0):
        self.host = host
        self.port = port
        self.connect_timeout = connect_timeout
        self._sock: socket.socket | None = None

    def describe(self) -> str:
        return f"tcp://{self.host}:{self.port}"

    def open(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.connect_timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(_POLL_S)
        self._sock = sock

    def write(self, data: bytes) -> None:
        if self._sock is None:
            raise ConnectionError("transport is closed")
        try:
            self._sock.sendall(data)
        except socket.timeout as exc:  # a partial send would corrupt the stream
            raise ConnectionError("send to robot timed out") from exc

    def read(self) -> bytes:
        """Returns available bytes, or b"" if nothing arrived within the poll interval."""
        if self._sock is None:
            raise ConnectionError("transport is closed")
        try:
            data = self._sock.recv(4096)
        except socket.timeout:
            return b""
        if not data:
            raise ConnectionError("the ESP32 bridge closed the connection")
        return data

    def close(self) -> None:
        sock, self._sock = self._sock, None
        if sock is not None:
            with suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()


class SerialTransport:
    """USB cable straight to the UNO (bench testing; "Upload-Cam" switch on Upload).

    Opening the port resets the UNO, so the env's handshake retries until the sketch
    has booted.
    """

    needs_heartbeat = False

    def __init__(self, port: str, baudrate: int = 9600):
        self.port = port
        self.baudrate = baudrate
        self._ser = None

    def describe(self) -> str:
        return f"serial://{self.port}@{self.baudrate}"

    def open(self) -> None:
        try:
            import serial  # pyserial
        except ImportError as exc:
            raise ImportError("serial_port= needs pyserial: pip install pyserial") from exc
        self._ser = serial.Serial(self.port, self.baudrate, timeout=_POLL_S)

    def write(self, data: bytes) -> None:
        if self._ser is None:
            raise ConnectionError("transport is closed")
        self._ser.write(data)

    def read(self) -> bytes:
        if self._ser is None:
            raise ConnectionError("transport is closed")
        return self._ser.read(self._ser.in_waiting or 1)

    def close(self) -> None:
        ser, self._ser = self._ser, None
        if ser is not None:
            with suppress(Exception):
                ser.close()


class RobotLink:
    """Request/response over a transport.

    A background thread reads frames, keeps the ESP32 bridge alive with heartbeats even
    while the agent is busy (e.g. training between episodes), and hands each reply to
    the request waiting for its sequence number. Late replies to abandoned requests are
    dropped.
    """

    HEARTBEAT_INTERVAL_S = 0.5

    def __init__(self, transport):
        self.transport = transport
        self.hello: Hello | None = None  # set when the robot (re)boots while connected
        self.stock_firmware_seen = False
        self._cond = threading.Condition()
        self._send_lock = threading.Lock()
        self._pending_seq: int | None = None
        self._reply: RobotState | RobotError | None = None
        self._error: BaseException | None = None
        self._closed = False
        self._seq = 0
        self._thread: threading.Thread | None = None

    def open(self) -> None:
        self.transport.open()
        self._thread = threading.Thread(target=self._reader, name="robotgym-link", daemon=True)
        self._thread.start()

    @property
    def alive(self) -> bool:
        return (
            not self._closed
            and self._error is None
            and self._thread is not None
            and self._thread.is_alive()
        )

    def close(self) -> None:
        self._closed = True
        self.transport.close()
        if self._thread is not None and self._thread is not threading.current_thread():
            self._thread.join(timeout=1.0)
        with self._cond:
            self._cond.notify_all()

    def send(self, data: bytes) -> None:
        with self._send_lock:
            self.transport.write(data)

    def request(self, kind: str, fields: list[int], timeout: float) -> RobotState:
        """Sends a command and waits for the O frame carrying its sequence number."""
        self._seq = self._seq % 65535 + 1
        seq = self._seq
        with self._cond:
            if self._error is not None:
                raise ConnectionError(f"link to robot is down: {self._error}") from self._error
            self._pending_seq = seq
            self._reply = None
        self.send(encode_command(kind, seq, *fields))
        deadline = time.monotonic() + timeout
        with self._cond:
            while self._reply is None and self._error is None and not self._closed:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._cond.wait(remaining)
            reply, error = self._reply, self._error
            self._reply = None
            self._pending_seq = None
        if isinstance(reply, RobotState):
            return reply
        if isinstance(reply, RobotError):
            raise ProtocolError(f"robot rejected {kind} frame: {reply.message}")
        if error is not None:
            raise ConnectionError(f"link to robot is down: {error}") from error
        if self._closed:
            raise ConnectionError("link closed")
        raise TimeoutError(f"no reply to {kind} (seq {seq}) within {timeout:.2f} s")

    def _reader(self) -> None:
        parser = FrameParser()
        last_heartbeat = 0.0
        try:
            while not self._closed:
                if self.transport.needs_heartbeat:
                    now = time.monotonic()
                    if now - last_heartbeat >= self.HEARTBEAT_INTERVAL_S:
                        self.send(b"{" + HEARTBEAT.encode() + b"}")
                        last_heartbeat = now
                for content in parser.feed(self.transport.read()):
                    self._dispatch(content)
        except Exception as exc:  # noqa: BLE001 - any failure means the link is gone
            if not self._closed:
                logger.warning("robot link %s failed: %s", self.transport.describe(), exc)
                with self._cond:
                    self._error = exc
        finally:
            with self._cond:
                self._cond.notify_all()

    def _dispatch(self, content: str) -> None:
        msg = parse_frame(content)
        if isinstance(msg, (RobotState, RobotError)):
            with self._cond:
                if msg.seq == self._pending_seq:
                    self._reply = msg
                    self._cond.notify_all()
                else:
                    logger.debug("dropping stale reply %r", content)
        elif isinstance(msg, Hello):
            self.hello = msg
            logger.info("robot booted: %s protocol v%d", msg.name, msg.version)
        elif msg is None:
            logger.debug("malformed frame %r", content)
        elif msg != HEARTBEAT:
            if is_stock_firmware_reply(msg):
                self.stock_firmware_seen = True
            logger.debug("ignoring frame %r", content)
