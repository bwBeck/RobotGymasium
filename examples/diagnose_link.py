"""Find where the PC <-> robot link breaks.

Run it with the PC on the robot's "ELEGOO-xxxx" WiFi, ideally with the Arduino's USB
cable plugged in too - then it also listens to what the Arduino itself sends:

    python examples/diagnose_link.py

It checks, in order: the ESP32 bridge (TCP port 100 and its once-a-second heartbeat,
which only ELEGOO's camera firmware sends), whether the Arduino answers through it,
and - over USB - whether the Arduino received the commands at all. The verdict is also
written to ~/robogym_diagnose.txt. Standard library only; the USB part needs pyserial.
"""

from __future__ import annotations

import argparse
import re
import socket
import threading
import time
from pathlib import Path

USB_VENDORS = {0x1A86: "CH340", 0x04D9: "HT42B534", 0x2341: "Arduino"}


def frames(data: bytes) -> list[str]:
    return [m.decode("ascii", "replace") for m in re.findall(rb"\{([^{}]*)\}", data)]


class UsbListener(threading.Thread):
    """Collects everything the Arduino transmits. Its TX feeds both USB and the ESP32."""

    def __init__(self, port: str):
        import serial

        super().__init__(daemon=True)
        self.serial = serial.Serial(port, 9600, timeout=0.1)  # opening restarts the Arduino
        self._data = bytearray()
        self._lock = threading.Lock()

    def run(self) -> None:
        while True:
            try:
                chunk = self.serial.read(256)
            except Exception:  # noqa: BLE001 - unplugged: just stop listening
                return
            with self._lock:
                self._data += chunk

    def data(self) -> bytes:
        with self._lock:
            return bytes(self._data)


def wait_for(condition, seconds: float) -> bool:
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if condition():
            return True
        time.sleep(0.05)
    return condition()


def find_usb_port() -> str | None:
    from serial.tools import list_ports

    for p in list_ports.comports():
        if p.vid in USB_VENDORS:
            return p.device
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="192.168.4.1")
    parser.add_argument("--port", type=int, default=100)
    parser.add_argument("--usb", help="Arduino serial port (default: auto-detect)")
    parser.add_argument("--no-usb", action="store_true", help="skip the Arduino-side check")
    args = parser.parse_args()
    report: list[str] = []

    def say(line: str) -> None:
        print(line, flush=True)
        report.append(line)

    # The Arduino's own view, over USB.
    usb = None
    booted = None
    if not args.no_usb:
        try:
            port = args.usb or find_usb_port()
            if port is None:
                say("(no Arduino USB port found - skipping the Arduino-side check)")
            else:
                usb = UsbListener(port)
                usb.start()
                say(f"listening to the Arduino on {port} (opening the port restarts it)...")
                booted = wait_for(lambda: b"{H,RoboGym" in usb.data(), 4.5)
                if not booted:
                    print("  no boot message yet - press the RESET button on the robot's shield "
                          "now (waiting 15 s)...", flush=True)
                    booted = wait_for(lambda: b"{H,RoboGym" in usb.data(), 15.0)
                say(f"[{' OK ' if booted else 'FAIL'}] Arduino boot message over USB: "
                    + ("RoboGym sketch running" if booted else f"none (got {usb.data()[:60]!r})"))
        except ImportError:
            say("(pyserial not installed - skipping the Arduino-side check: pip install pyserial)")
        except OSError as exc:
            say(f"(cannot open the Arduino's USB port: {exc} - is the Arduino IDE using it?)")
            usb = None

    # The ESP32 bridge.
    try:
        sock = socket.create_connection((args.host, args.port), timeout=3)
    except OSError as exc:
        say(f"[FAIL] cannot connect to {args.host}:{args.port} ({exc})")
        say("VERDICT: this PC can't reach the robot - join its ELEGOO-xxxx WiFi and run again.")
        return finish(report)
    say(f"[ OK ] TCP connection to the ESP32 bridge at {args.host}:{args.port}")
    sock.settimeout(0.1)
    received = bytearray()

    def pump(seconds: float) -> None:
        end, last_hb = time.monotonic() + seconds, 0.0
        while time.monotonic() < end:
            if time.monotonic() - last_hb > 0.5:
                sock.sendall(b"{Heartbeat}")
                last_hb = time.monotonic()
            try:
                received.extend(sock.recv(4096))
            except socket.timeout:
                pass

    pump(2.5)
    heartbeat = "Heartbeat" in frames(received)
    say("[ OK ] ESP32 sends heartbeats: it runs ELEGOO's camera/bridge firmware" if heartbeat else
        "[FAIL] no heartbeat from the ESP32 in 2.5 s: port 100 is not ELEGOO's bridge firmware")

    # The Arduino, through the bridge.
    usb_mark = len(usb.data()) if usb else 0
    for seq in (1, 2, 3):
        sock.sendall(b"{P,%d}" % seq)
        pump(1.5)
    wifi_replies = [f for f in frames(received) if f.startswith("O,")]
    usb_replies = [f for f in frames(usb.data()[usb_mark:]) if f.startswith("O,")] if usb else None
    say(f"[{' OK ' if wifi_replies else 'FAIL'}] Arduino replies over WiFi: {len(wifi_replies)} of 3")
    if usb_replies is not None:
        say(f"[{' OK ' if usb_replies else 'FAIL'}] Arduino replies seen on USB "
            f"(= it received the commands): {len(usb_replies)} of 3")
    sock.sendall(b'{"N":100}')  # ELEGOO's factory Arduino sketch answers this with {ok}
    pump(1.0)
    stock_sketch = any(f == "ok" or f.endswith("_ok") for f in frames(received))
    sock.close()

    if wifi_replies:
        verdict = "the whole link works - the Arduino answers over WiFi."
    elif not heartbeat:
        verdict = ("the camera module is not running ELEGOO's bridge firmware. Flash ELEGOO's "
                   "ESP32_CameraServer_AP code (02 Manual & Main Code & APP/04 Code of Carmer).")
    elif stock_sketch:
        verdict = "the Arduino runs ELEGOO's factory sketch - flash arduino/RoboGym/RoboGym.ino."
    elif usb_replies is None:
        verdict = ("the ESP32 bridge works but the Arduino doesn't answer through it. Plug in the "
                   "Arduino's USB cable (and pip install pyserial) and run again to see which "
                   "direction is broken. Most likely: the Upload-Cam switch is not on 'Cam'.")
    elif usb_replies:
        verdict = ("the Arduino received the commands and answered, but the answers never reached "
                   "the PC: the Arduino TX -> camera module wire is broken (shield header P8 pin 1, "
                   "or the camera module's 4-pin cable).")
    elif not booted:
        verdict = ("the Arduino never started up - no boot message even after a reset. Check the "
                   "robot's power switch is ON: on USB power alone, the servo it centres at boot "
                   "can brown it out and restart it endlessly. Then run this again.")
    else:
        verdict = ("the commands never reach the Arduino. That wire runs camera module -> shield "
                   "header P8 pin 2 -> Upload-Cam switch -> Arduino RX. Slide the switch fully to "
                   "'Cam' (it only connects this one wire), and check the camera's 4-pin cable is "
                   "seated in the shield's UART header.")
    say("VERDICT: " + verdict)
    finish(report)


def finish(report: list[str]) -> None:
    path = Path.home() / "robogym_diagnose.txt"
    path.write_text(time.strftime("%Y-%m-%d %H:%M:%S\n") + "\n".join(report) + "\n")
    print(f"(saved to {path})")


if __name__ == "__main__":
    main()
