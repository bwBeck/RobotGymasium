"""Camera access for the ESP32-WROVER module running ELEGOO's stock firmware.

The firmware serves ``/capture`` (one JPEG) and ``/control`` on port 80, and an MJPEG
``/stream`` on port 81. Two quirks matter for RL:

* ``/stream`` force-enables on-board face detection, which runs on every frame up to
  400 px wide (slow, and it draws boxes into the image). :func:`esp32_set` with
  ``face_detect=0`` after the stream starts turns it back off.
* With two frame buffers the driver hands out frames that may be old, so
  :class:`SnapshotCamera` throws the first one away. :class:`StreamCamera` (the
  default) reads continuously and always holds the newest frame.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import urllib.error
import urllib.request
from typing import Callable, Iterator, NamedTuple

import numpy as np
from PIL import Image

logger = logging.getLogger("robotgym")

_BILINEAR = getattr(Image, "Resampling", Image).BILINEAR


class Frame(NamedTuple):
    jpeg: bytes
    stamp: float  # time.monotonic() when the frame finished arriving
    index: int  # increments with every new frame


def esp32_set(host: str, port: int, var: str, val: int, timeout: float = 2.0) -> bool:
    """``GET /control?var=<var>&val=<val>`` on the ESP32. Returns True on HTTP 2xx."""
    url = f"http://{host}:{port}/control?var={var}&val={int(val)}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return 200 <= resp.status < 300
    except (urllib.error.URLError, OSError) as exc:
        logger.warning("camera setting %s=%s failed: %s", var, val, exc)
        return False


def iter_mjpeg(stream) -> Iterator[bytes]:
    """Yields the JPEGs of a ``multipart/x-mixed-replace`` body.

    Relies on each part's Content-Length header, which the ESP32 firmware sends.
    ``stream`` needs ``readline()`` and ``read(n)`` (e.g. an HTTPResponse, which also
    undoes the chunked transfer encoding the ESP32 uses).
    """
    while True:
        length = None
        header_lines = 0
        while True:
            line = stream.readline(1024)
            if not line:
                return
            line = line.strip()
            if not line:
                if length is not None:
                    break
                continue
            if line.startswith(b"--"):
                continue  # part boundary
            key, _, value = line.partition(b":")
            if key.strip().lower() == b"content-length":
                length = int(value.strip())
            header_lines += 1
            if header_lines > 32:
                raise ValueError("MJPEG part without a Content-Length header")
        chunks = []
        remaining = length
        while remaining > 0:
            chunk = stream.read(remaining)
            if not chunk:
                return
            chunks.append(chunk)
            remaining -= len(chunk)
        yield b"".join(chunks)


def decode_jpeg(data: bytes, size: tuple[int, int] | None = None) -> np.ndarray:
    """JPEG bytes -> uint8 RGB array, optionally resized to ``size = (height, width)``."""
    img = Image.open(io.BytesIO(data))
    if size is not None:
        height, width = size
        img.draft("RGB", (width, height))  # let libjpeg downscale while decoding
        img = img.convert("RGB")
        if img.size != (width, height):
            img = img.resize((width, height), _BILINEAR)
    else:
        img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)


class StreamCamera:
    """Reads the MJPEG stream in a background thread and keeps the newest frame.

    ``on_connect`` runs every time the stream (re)opens - used to switch the firmware's
    face detection back off, since opening the stream switches it on.
    """

    def __init__(self, url: str, on_connect: Callable[[], None] | None = None, read_timeout: float = 3.0):
        self.url = url
        self.on_connect = on_connect
        self.read_timeout = read_timeout
        self.last_error: BaseException | None = None
        self._cond = threading.Condition()
        self._frame: Frame | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is None or not self._thread.is_alive():
            self._stop.clear()
            self._thread = threading.Thread(target=self._run, name="robotgym-camera", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        with self._cond:
            self._cond.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=self.read_timeout + 1.0)

    def get(self, newer_than: float | None = None, timeout: float = 0.0) -> Frame | None:
        """Newest frame; waits up to ``timeout`` s for one that arrived after ``newer_than``."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while newer_than is not None and (self._frame is None or self._frame.stamp <= newer_than):
                remaining = deadline - time.monotonic()
                if remaining <= 0 or self._stop.is_set():
                    break
                self._cond.wait(remaining)
            return self._frame

    def _run(self) -> None:
        index = 0
        while not self._stop.is_set():
            try:
                with urllib.request.urlopen(self.url, timeout=self.read_timeout) as resp:
                    if self.on_connect is not None:
                        self.on_connect()
                    for jpeg in iter_mjpeg(resp):
                        index += 1
                        with self._cond:
                            self._frame = Frame(jpeg, time.monotonic(), index)
                            self._cond.notify_all()
                        if self._stop.is_set():
                            return
                    raise ConnectionError("camera stream ended")
            except Exception as exc:  # noqa: BLE001 - keep retrying whatever went wrong
                if self._stop.is_set():
                    return
                if type(exc) is not type(self.last_error):
                    logger.warning("camera stream %s: %s (retrying)", self.url, exc)
                self.last_error = exc
                self._stop.wait(0.5)


class SnapshotCamera:
    """Fetches ``/capture`` on demand. Slower than the stream, but no background traffic."""

    def __init__(self, url: str, flush: bool = True, timeout: float = 3.0):
        self.url = url
        self.flush = flush
        self.timeout = timeout
        self.last_error: BaseException | None = None
        self._frame: Frame | None = None

    def start(self) -> None:
        pass

    def stop(self) -> None:
        pass

    def get(self, newer_than: float | None = None, timeout: float = 0.0) -> Frame | None:
        try:
            if self.flush:
                self._fetch()  # the driver's first buffer can hold a stale frame
            jpeg = self._fetch()
        except (urllib.error.URLError, OSError) as exc:
            self.last_error = exc
            logger.warning("camera capture %s failed: %s", self.url, exc)
            return self._frame
        index = self._frame.index + 1 if self._frame else 1
        self._frame = Frame(jpeg, time.monotonic(), index)
        return self._frame

    def _fetch(self) -> bytes:
        with urllib.request.urlopen(self.url, timeout=self.timeout) as resp:
            return resp.read()
