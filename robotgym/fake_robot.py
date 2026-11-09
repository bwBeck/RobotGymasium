"""Software stand-in for the robot, so RoboGym runs without hardware.

    python -m robotgym.fake_robot

    env = RoboGym(host="127.0.0.1", control_port=8100, http_port=8080, camera_port=8081)

or, in-process::

    with FakeRobotServer() as fake:
        env = RoboGym(**fake.env_kwargs())

It emulates everything the env talks to on the real robot:

* :class:`FakeEsp32Bridge` - the stock ESP32 firmware's TCP<->UART bridge: strips
  spaces, flushes on "}", sends {Heartbeat} every second and drops a client that has
  been silent for ~4 s, and sends {"N":100} to the UNO when the client goes away.
* :class:`SimRobot` - RoboGym.ino's protocol on top of a kinematic tank in a walled
  square arena, including the 9600-baud transmission delays.
* :class:`FakeCameraServer` - /stream (chunked MJPEG, like the ESP32), /capture,
  /control and /status, rendering a first-person view of the arena.

The simulated tank follows the motor model the sketch uses for its estimates, but walls
stop it while the estimated turns keep counting - as on the real, encoder-less robot.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import socket
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Callable
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

from .protocol import FLAG_BOOT, FLAG_IMU_OK, FLAG_NO_ECHO, FLAG_WATCHDOG, PROTOCOL_VERSION, FrameParser

# ESP32 frame size index -> (width, height), as in ELEGOO's firmware build.
FRAMESIZES = {
    0: (160, 120), 1: (128, 160), 2: (176, 144), 3: (240, 176), 4: (320, 240), 5: (400, 296),
    6: (640, 480), 7: (800, 600), 8: (1024, 768), 9: (1280, 1024), 10: (1600, 1200),
}
BOUNDARY = b"123456789000000000000987654321"


def _clamp(v: int, lo: int, hi: int) -> int:
    return lo if v < lo else hi if v > hi else v


def _parse_fields(rest: str) -> list[int] | None:
    """Mirror of the sketch's parseFields(): ",<int>,<int>..." -> list, None if malformed."""
    if rest == "":
        return []
    if not rest.startswith(","):
        return None
    try:
        return [int(p) for p in rest[1:].split(",")]
    except ValueError:
        return None


class SimRobot:
    """RoboGym.ino's behaviour plus a simple tank, exposed as a serial device.

    ``write()`` receives bytes as the UNO would from the ESP32; ``read()`` returns the
    bytes the UNO has finished transmitting.
    """

    WALL_COLORS = np.array([[205, 70, 60], [70, 165, 75], [65, 95, 200], [215, 190, 70]], np.float32)
    RADIUS = 0.12  # m, keeps the robot's centre this far from the walls
    SKID = 0.8  # tracks slip when turning: true rotation vs ideal differential drive
    SENSOR_OFFSET = 0.09  # m, ultrasonic sensor ahead of the centre
    CALIBRATION_TIME = 0.45  # s the sketch spends measuring the gyro bias on R

    def __init__(self, arena_size: float = 3.0, start=(1.5, 1.5, 0.0), battery_v: float = 7.9,
                 baud: int = 9600, seed: int = 0):
        self.arena = arena_size
        self.battery_v = battery_v
        self.byte_time = 10.0 / baud if baud else 0.0
        self.rng = np.random.default_rng(seed)
        self.received: list[str] = []  # every frame the "UNO" got, for tests

        self.x, self.y, self.th = start  # true pose in the arena
        self.rps = [0.0, 0.0]
        self.cmd = [0, 0]
        self._default_config()
        self._zero_odometry()
        self.flags = FLAG_BOOT
        self.pending: tuple[int, float] | None = None

        self._cond = threading.Condition()
        self._inbox: deque[tuple[float, bytes]] = deque()
        self._outbox: deque[tuple[float, bytes]] = deque()
        self._uplink_free = 0.0
        self._downlink_free = 0.0
        self._parser = FrameParser(max_len=63)  # the sketch's 64-byte receive buffer
        self._boot = time.monotonic()
        self.last_reply = self._boot
        self._last_tick = self._boot
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="sim-robot", daemon=True)
        self._emit("{H,RoboGym,%d}" % PROTOCOL_VERSION)

    # ----------------------------------------------------------- serial device

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

    def write(self, data: bytes) -> None:
        with self._cond:
            start = max(time.monotonic(), self._uplink_free)
            self._uplink_free = start + len(data) * self.byte_time
            self._inbox.append((self._uplink_free, data))

    def read(self, timeout: float = 0.0) -> bytes:
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                now = time.monotonic()
                out = bytearray()
                while self._outbox and self._outbox[0][0] <= now:
                    out += self._outbox.popleft()[1]
                if out or now >= deadline:
                    return bytes(out)
                wake = min(deadline, self._outbox[0][0]) if self._outbox else deadline
                self._cond.wait(max(0.0, wake - now))

    # ------------------------------------------------------------------ robot

    def _emit(self, text: str) -> None:
        data = text.encode("ascii")
        with self._cond:
            start = max(time.monotonic(), self._downlink_free)
            self._downlink_free = start + len(data) * self.byte_time
            self._outbox.append((self._downlink_free, data))
            self._cond.notify_all()

    def reboot(self) -> None:
        """Simulate a brown-out: config back to defaults, odometry zeroed, boot flag set."""
        with self._cond:
            self._default_config()
            self._zero_odometry()
            self.cmd = [0, 0]
            self.rps = [0.0, 0.0]
            self.pending = None
            self.flags = FLAG_BOOT
            self._boot = time.monotonic()
        self._emit("{H,RoboGym,%d}" % PROTOCOL_VERSION)

    def _default_config(self) -> None:
        """Runtime configuration (C frame), with the sketch's defaults."""
        self.hold_timeout = 0.5
        self.max_rps = 3.0
        self.deadband = 40.0
        self.tau = 0.08
        self.circumference = math.pi * 0.043
        self.track_width = 0.15
        self.gyro_sign = 1

    def _zero_odometry(self) -> None:
        self.ox = self.oy = self.oth = self.heading = 0.0
        self.turns = [0.0, 0.0]

    def _run(self) -> None:
        while not self._stop.is_set():
            now = time.monotonic()
            with self._cond:
                while self._inbox and self._inbox[0][0] <= now:
                    for content in self._parser.feed(self._inbox.popleft()[1]):
                        self._handle(content, now)
                self._physics(now - self._last_tick)
                self._last_tick = now
                if self.pending is not None:
                    if now >= self.pending[1]:
                        seq, self.pending = self.pending[0], None
                        self._send_observation(seq, now)
                elif any(self.cmd) and now - self.last_reply >= self.hold_timeout:
                    self.cmd = [0, 0]
                    self.flags |= FLAG_WATCHDOG
            self._stop.wait(0.004)

    def _handle(self, s: str, now: float) -> None:
        self.received.append(s)
        if s == '"N":100':
            self.cmd = [0, 0]
            self.pending = None
            return
        kind = s[:1]
        if kind not in ("A", "R", "C", "P"):
            return
        v = _parse_fields(s[1:])
        n = -1 if v is None else len(v)
        seq = v[0] & 0xFFFF if n > 0 else 0
        self.pending = None
        if kind == "A" and n == 4:
            self.cmd = [_clamp(v[1], -255, 255), _clamp(v[2], -255, 255)]
            self.pending = (seq, now + _clamp(v[3], 0, 10000) / 1000.0)
        elif kind == "R" and n == 2:
            self.cmd = [0, 0]
            self.rps = [0.0, 0.0]
            self._zero_odometry()
            self.pending = (seq, now + (self.CALIBRATION_TIME if v[1] else 0.0))
        elif kind == "C" and n == 8:
            self.hold_timeout = _clamp(v[1], 50, 60000) / 1000.0
            self.max_rps = _clamp(v[2], 1, 100000) / 600.0
            self.deadband = float(_clamp(v[3], 0, 200))
            self.tau = _clamp(v[4], 0, 2000) / 1000.0
            self.circumference = math.pi * _clamp(v[5], 10, 2000) / 10000.0
            self.track_width = _clamp(v[6], 10, 1000) / 1000.0
            self.gyro_sign = -1 if v[7] < 0 else 1
            self._send_observation(seq, now)
        elif kind == "P" and n == 1:
            self._send_observation(seq, now)
        else:
            self._emit("{E,%d,1}" % seq)

    def _target_rps(self, pwm: int) -> float:
        if pwm == 0:
            return 0.0
        effective = abs(pwm) * self.battery_v / 7.4
        if effective <= self.deadband:
            return 0.0
        rps = self.max_rps * (effective - self.deadband) / (255.0 - self.deadband)
        return math.copysign(rps, pwm)

    def _physics(self, dt: float) -> None:
        a = dt / (self.tau + dt) if self.tau > 0 else 1.0
        for i in (0, 1):
            self.rps[i] += a * (self._target_rps(self.cmd[i]) - self.rps[i])
        v_left, v_right = (r * self.circumference for r in self.rps)
        v = 0.5 * (v_left + v_right)
        omega = (v_right - v_left) / self.track_width * self.SKID

        # True motion; the walls block it.
        mid = self.th + 0.5 * omega * dt
        nx, ny = self.x + v * math.cos(mid) * dt, self.y + v * math.sin(mid) * dt
        lo, hi = self.RADIUS, self.arena - self.RADIUS
        if lo <= nx <= hi and lo <= ny <= hi:
            self.x, self.y = nx, ny
        self.th += omega * dt

        # What the sketch computes: gyro heading + motor-model distance (walls unknown).
        w = self.gyro_sign * omega
        d_th = 0.0 if abs(w) < 0.005 else w * dt
        ds = v * dt
        mid = self.heading + 0.5 * d_th
        self.ox += ds * math.cos(mid)
        self.oy += ds * math.sin(mid)
        self.oth += d_th
        self.heading = math.remainder(self.heading + d_th, math.tau)
        self.turns[0] += self.rps[0] * dt
        self.turns[1] += self.rps[1] * dt

    def _ray(self, x: float, y: float, th: float) -> float:
        c, s = math.cos(th), math.sin(th)
        tx = (self.arena - x) / c if c > 1e-9 else (-x / c if c < -1e-9 else math.inf)
        ty = (self.arena - y) / s if s > 1e-9 else (-y / s if s < -1e-9 else math.inf)
        return min(tx, ty)

    def _range_mm(self) -> int:
        d = self._ray(self.x, self.y, self.th) - self.SENSOR_OFFSET
        mm = int(d * 1000 + self.rng.normal(0, 3))
        return _clamp(mm, 20, 4000)

    def _send_observation(self, seq: int, now: float) -> None:
        dist = self._range_mm()
        flags = self.flags | FLAG_IMU_OK | (FLAG_NO_ECHO if dist >= 4000 else 0)
        t_us = int((now - self._boot) * 1e6) & 0xFFFFFFFF
        self._emit("{O,%d,%d,%d,%d,%d,%d,%d,%d,%d,%d}" % (
            seq, t_us, int(self.ox * 1e4), int(self.oy * 1e4), int(self.oth * 1e4),
            int(self.turns[0] * 1e3), int(self.turns[1] * 1e3), dist,
            round(self.battery_v * 100), flags))
        self.flags = 0
        self.last_reply = now

    # ----------------------------------------------------------------- camera

    def render(self, width: int = 320, height: int = 240) -> np.ndarray:
        """First-person view: four coloured, striped walls, sky and floor."""
        with self._cond:
            x, y, th = self.x, self.y, self.th
        L = self.arena
        fov = math.radians(62)
        ang = th + np.linspace(fov / 2, -fov / 2, width)
        c, s = np.cos(ang), np.sin(ang)
        with np.errstate(divide="ignore", invalid="ignore"):
            tx = np.where(c > 1e-9, (L - x) / c, np.where(c < -1e-9, -x / c, np.inf))
            ty = np.where(s > 1e-9, (L - y) / s, np.where(s < -1e-9, -y / s, np.inf))
        hit_x = tx < ty
        dist = np.minimum(tx, ty)
        wall = np.where(hit_x, np.where(c > 0, 0, 1), np.where(s > 0, 2, 3))
        along = np.where(hit_x, y + s * dist, x + c * dist)  # position along the wall
        focal = (width / 2) / math.tan(fov / 2)
        half = np.minimum(focal * 0.25 / np.maximum(dist * np.cos(ang - th), 1e-3), height / 2)
        shade = np.clip(1.15 - dist / (1.5 * L), 0.35, 1.0)
        shade = np.where((along * 4.0) % 1.0 < 0.07, shade * 0.55, shade)  # stripes every 25 cm
        colors = self.WALL_COLORS[wall] * shade[:, None]

        rows = np.arange(height)[:, None]
        img = np.empty((height, width, 3), np.float32)
        img[: height // 2] = (150, 190, 230)
        img[height // 2:] = (115, 108, 100)
        on_wall = (rows >= height / 2 - half[None, :]) & (rows < height / 2 + half[None, :])
        img = np.where(on_wall[:, :, None], colors[None, :, :], img)
        return img.astype(np.uint8)


class FakeEsp32Bridge:
    """The stock ESP32 firmware's TCP <-> UART bridge, in front of a serial device."""

    def __init__(self, device, host: str = "127.0.0.1", port: int = 0,
                 heartbeat_period: float = 1.0, max_missed: int = 3):
        self.device = device
        self.heartbeat_period = heartbeat_period
        self.max_missed = max_missed
        self.clients_served = 0
        self.heartbeat_drops = 0
        self._srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._srv.bind((host, port))
        self._srv.listen(1)
        self._srv.settimeout(0.2)
        self.port = self._srv.getsockname()[1]
        self._conn: socket.socket | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, name="fake-esp32", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.drop_client()
        self._thread.join(timeout=2.0)
        self._srv.close()

    def drop_client(self) -> None:
        """Cut the current connection, as a WiFi dropout would."""
        conn = self._conn
        if conn is not None:
            try:
                conn.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def _serve(self) -> None:
        while not self._stop.is_set():
            try:
                conn, _ = self._srv.accept()
            except socket.timeout:
                self.device.read(0.0)  # with no client the ESP32 swallows UART output
                continue
            except OSError:
                return
            self.clients_served += 1
            self._handle(conn)

    def _handle(self, conn: socket.socket) -> None:
        conn.settimeout(0.003)
        self._conn = conn
        read_buf, send_buf = bytearray(), bytearray()
        in_frame = False
        hb_received, hb_missed, hb_time = False, 0, time.monotonic()
        try:
            while not self._stop.is_set():
                try:
                    data = conn.recv(1024)
                    if not data:
                        break
                except socket.timeout:
                    data = b""
                for b in data:  # same rules as SocketServer_Test() in the ESP32 sketch
                    if not in_frame and b == 0x7B:
                        in_frame = True
                    if in_frame and b != 0x20:
                        read_buf.append(b)
                    if in_frame and b == 0x7D:
                        in_frame = False
                        if read_buf == b"{Heartbeat}":
                            hb_received = True
                        else:
                            self.device.write(bytes(read_buf))
                        read_buf.clear()
                for b in self.device.read(0.0):
                    send_buf.append(b)
                    if b == 0x7D:
                        conn.sendall(bytes(send_buf))
                        send_buf.clear()
                now = time.monotonic()
                if now - hb_time > self.heartbeat_period:
                    conn.sendall(b"{Heartbeat}")
                    if hb_received:
                        hb_received, hb_missed = False, 0
                    else:
                        hb_missed += 1
                    if hb_missed > self.max_missed:
                        self.heartbeat_drops += 1
                        break
                    hb_time = now
        except OSError:
            pass
        finally:
            self.device.write(b'{"N":100}')
            self._conn = None
            conn.close()


class FakeCameraServer:
    """/capture, /control, /status on one port and /stream on another, like the ESP32."""

    def __init__(self, render: Callable[[int, int], np.ndarray], host: str = "127.0.0.1",
                 http_port: int = 0, stream_port: int = 0, fps: float = 20.0, framesize: int = 7):
        self.render = render
        self.fps = fps
        self.framesize = framesize
        self.face_detect = 0
        self.frames_streamed = 0
        self.requests: list[str] = []
        self._stop = threading.Event()
        handler = self._make_handler()
        self._http = ThreadingHTTPServer((host, http_port), handler)
        self._stream = ThreadingHTTPServer((host, stream_port), handler)
        for srv, is_stream in ((self._http, False), (self._stream, True)):
            srv.daemon_threads = True
            srv.is_stream = is_stream
        self._threads = [
            threading.Thread(target=srv.serve_forever, kwargs={"poll_interval": 0.1}, daemon=True)
            for srv in (self._http, self._stream)
        ]

    @property
    def http_port(self) -> int:
        return self._http.server_address[1]

    @property
    def stream_port(self) -> int:
        return self._stream.server_address[1]

    def start(self) -> None:
        for t in self._threads:
            t.start()

    def stop(self) -> None:
        self._stop.set()
        for srv in (self._http, self._stream):
            srv.shutdown()
            srv.server_close()

    def jpeg(self) -> bytes:
        width, height = FRAMESIZES.get(self.framesize, (800, 600))
        buf = io.BytesIO()
        Image.fromarray(self.render(width, height)).save(buf, format="JPEG", quality=80)
        return buf.getvalue()

    def _control(self, var: str | None, val: str | None) -> bool:
        try:
            value = int(val)
        except (TypeError, ValueError):
            return False
        if var == "framesize":
            self.framesize = value
        elif var == "face_detect":
            self.face_detect = value
        elif var not in ("quality", "brightness", "contrast", "vflip", "hmirror"):
            return False
        return True

    def _stream_to(self, handler: BaseHTTPRequestHandler) -> None:
        self.face_detect = 1  # the stock firmware's stream handler forces this on
        handler.send_response(200)
        handler.send_header("Content-Type", "multipart/x-mixed-replace;boundary=" + BOUNDARY.decode())
        handler.send_header("Transfer-Encoding", "chunked")
        handler.send_header("Access-Control-Allow-Origin", "*")
        handler.end_headers()
        try:
            while not self._stop.is_set():
                t0 = time.monotonic()
                jpeg = self.jpeg()
                if self.face_detect and FRAMESIZES.get(self.framesize, (800, 600))[0] <= 400:
                    time.sleep(0.25)  # on-board face detection is slow
                parts = (b"Content-Type: image/jpeg\r\nContent-Length: %d\r\n\r\n" % len(jpeg),
                         jpeg, b"\r\n--" + BOUNDARY + b"\r\n")
                for part in parts:
                    handler.wfile.write(b"%X\r\n%s\r\n" % (len(part), part))
                handler.wfile.flush()
                self.frames_streamed += 1
                time.sleep(max(0.0, 1.0 / self.fps - (time.monotonic() - t0)))
        except OSError:
            pass
        handler.close_connection = True

    def _make_handler(cam):  # noqa: N805 - the closure's "self" is the server
        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                url = urlparse(self.path)
                query = {k: v[0] for k, v in parse_qs(url.query).items()}
                cam.requests.append(self.path)
                if self.server.is_stream:
                    if url.path == "/stream":
                        cam._stream_to(self)
                    else:
                        self._send(404, "text/plain", b"not found")
                elif url.path == "/capture":
                    self._send(200, "image/jpeg", cam.jpeg())
                elif url.path == "/control":
                    ok = cam._control(query.get("var"), query.get("val"))
                    self._send(200 if ok else 500, "text/plain", b"")
                elif url.path == "/status":
                    body = json.dumps({"framesize": cam.framesize, "face_detect": cam.face_detect})
                    self._send(200, "application/json", body.encode())
                else:
                    self._send(404, "text/plain", b"not found")

            def _send(self, code: int, ctype: str, body: bytes) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)

        return Handler


class FakeRobotServer:
    """SimRobot + bridge + camera on localhost. ``env_kwargs()`` gives RoboGym's arguments."""

    def __init__(self, host: str = "127.0.0.1", control_port: int = 0, http_port: int = 0,
                 camera_port: int = 0, robot: SimRobot | None = None, **sim_kwargs):
        self.robot = robot if robot is not None else SimRobot(**sim_kwargs)
        self.bridge = FakeEsp32Bridge(self.robot, host, control_port)
        self.camera = FakeCameraServer(self.robot.render, host, http_port, camera_port)
        self.host = host

    def start(self) -> FakeRobotServer:
        self.robot.start()
        self.bridge.start()
        self.camera.start()
        return self

    def stop(self) -> None:
        self.camera.stop()
        self.bridge.stop()
        self.robot.stop()

    def __enter__(self) -> FakeRobotServer:
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def env_kwargs(self) -> dict:
        return {"host": self.host, "control_port": self.bridge.port,
                "http_port": self.camera.http_port, "camera_port": self.camera.stream_port}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Fake ELEGOO Conqueror robot for RoboGym (no hardware needed).")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--control-port", type=int, default=8100)
    parser.add_argument("--http-port", type=int, default=8080)
    parser.add_argument("--camera-port", type=int, default=8081)
    parser.add_argument("--arena", type=float, default=3.0, help="side of the square arena [m]")
    args = parser.parse_args(argv)
    fake = FakeRobotServer(args.host, args.control_port, args.http_port, args.camera_port, arena_size=args.arena)
    with fake:
        kw = fake.env_kwargs()
        print("Fake robot running. Connect with:\n")
        print(f"    RoboGym(host={kw['host']!r}, control_port={kw['control_port']}, "
              f"http_port={kw['http_port']}, camera_port={kw['camera_port']})\n")
        print(f"Camera in a browser: http://{kw['host']}:{kw['camera_port']}/stream   (Ctrl+C to quit)")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
