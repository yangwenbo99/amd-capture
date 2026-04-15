#!/usr/bin/env python3
"""
mpv_hdr_controller.py

A small LAN-controllable HDR-oriented still-image display server for mpv.

What it does
------------
- Starts mpv with a Wayland/Vulkan/gpu-next-oriented configuration
- Displays still images indefinitely
- Exposes a simple HTTP API
- Stores "TV baseline" settings (real TV settings recorded externally)
- Applies small software simulation:
    - brightness multiplier
    - color temperature shift (Kelvin)
- Returns JSON status for capture orchestration

Intended use
------------
- TV is already in HDR mode
- Large brightness / color-temperature changes are done on the TV itself
- Small variations are simulated in software to avoid extra quantization issues

Requirements
------------
- Python 3.9+
- mpv in PATH
- Linux/Wayland recommended
- FFmpeg filters available in mpv build

Example
-------
Start:
    python3 mpv_hdr_controller.py --bind 0.0.0.0 --port 8080 --media-root /data/photos

Load image:
    curl -X POST http://localhost:8080/load \
      -H 'Content-Type: application/json' \
      -d '{"path":"sample.exr"}'

Set baseline TV settings metadata:
    curl -X POST http://localhost:8080/tv-baseline \
      -H 'Content-Type: application/json' \
      -d '{"brightness":"50","color_temperature":"Cold2","notes":"HDR picture mode"}'

Apply a small software simulation:
    curl -X POST http://localhost:8080/simulate \
      -H 'Content-Type: application/json' \
      -d '{"brightness_scale":1.08,"target_kelvin":6100}'

Read current state:
    curl http://HOST:8080/status

Notes
-----
- This script keeps the filter chain simple and rewrites the whole `vf` property
  whenever simulation parameters change.
- It assumes Kelvin simulation should be interpreted relative to a neutral
  baseline of 6500 K unless you choose a different value.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

# -----------------------------
# State model
# -----------------------------

@dataclass
class TVBaseline:
    brightness: str = ""
    color_temperature: str = ""
    notes: str = ""

@dataclass
class SimulationState:
    brightness_scale: float = 1.0
    target_kelvin: int = 6500
    gamma: float = 1.0
    saturation: float = 1.0
    enabled: bool = True

@dataclass
class AppState:
    media_root: str
    current_path: str = ""
    tv_baseline: TVBaseline = field(default_factory=TVBaseline)
    sim: SimulationState = field(default_factory=SimulationState)
    mpv_pid: Optional[int] = None
    mpv_socket: str = "/tmp/mpv-hdr-controller.sock"
    mpv_running: bool = False
    last_error: str = ""
    start_time: float = field(default_factory=time.time)


STATE_LOCK = threading.Lock()


# -----------------------------
# mpv IPC client
# -----------------------------

class MPVIPCError(RuntimeError):
    pass


class MPVClient:
    def __init__(self, socket_path: str):
        self.socket_path = socket_path

    def _send(self, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
        if not os.path.exists(self.socket_path):
            raise MPVIPCError(f"mpv IPC socket not found: {self.socket_path}")

        data = (json.dumps(payload) + "\n").encode("utf-8")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            sock.connect(self.socket_path)
            sock.sendall(data)

            # mpv replies with a JSON line
            buf = b""
            while b"\n" not in buf:
                chunk = sock.recv(65536)
                if not chunk:
                    break
                buf += chunk

        if not buf:
            raise MPVIPCError("No reply from mpv IPC")

        line = buf.splitlines()[0]
        reply = json.loads(line.decode("utf-8"))
        if reply.get("error") not in (None, "success"):
            raise MPVIPCError(f"mpv error: {reply}")
        return reply

    def command(self, *args: Any) -> Dict[str, Any]:
        return self._send({"command": list(args)})

    def set_property(self, name: str, value: Any) -> Dict[str, Any]:
        return self.command("set_property", name, value)

    def get_property(self, name: str) -> Any:
        reply = self.command("get_property", name)
        return reply.get("data")

    def loadfile(self, path: str, mode: str = "replace") -> None:
        self.command("loadfile", path, mode)


# -----------------------------
# Filter construction
# -----------------------------

def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def brightness_scale_to_eq_brightness(scale: float) -> float:
    """
    Map a multiplicative brightness scale into FFmpeg eq brightness.
    FFmpeg eq brightness is additive-like and ranges from -1 to 1.

    This is only an approximation suitable for *small* software variations.
    Recommended operating range for scale: about 0.75 to 1.25.

    Example:
      1.00 -> 0.00
      1.08 -> 0.08
      0.92 -> -0.08
    """
    return clamp(scale - 1.0, -0.25, 0.25)


def build_vf_string(sim: SimulationState) -> str:
    """
    Build the mpv vf option string.

    We use:
      lavfi=[eq=...,colortemperature=temperature=...]
    wrapped in mpv's lavfi graph syntax.

    References:
    - mpv supports lavfi graph syntax for vf
    - FFmpeg eq supports brightness/gamma/saturation
    - FFmpeg colortemperature supports Kelvin
    """
    if not sim.enabled:
        return ""

    eq_brightness = brightness_scale_to_eq_brightness(sim.brightness_scale)
    gamma = clamp(sim.gamma, 0.5, 3.0)
    saturation = clamp(sim.saturation, 0.0, 3.0)
    kelvin = int(clamp(sim.target_kelvin, 1000, 40000))

    graph = (
        f"eq=brightness={eq_brightness:.6f}:contrast=1.0:"
        f"saturation={saturation:.6f}:gamma={gamma:.6f},"
        f"colortemperature=temperature={kelvin}:mix=1.0:pl=0.0"
    )
    return f'lavfi=[{graph}]'


# -----------------------------
# mpv lifecycle
# -----------------------------

def launch_mpv(state: AppState) -> subprocess.Popen:
    try:
        os.unlink(state.mpv_socket)
    except FileNotFoundError:
        pass

    # gpu-next is mpv's recommended renderer. image-display-duration=inf keeps
    # still images open. input-ipc-server enables JSON IPC control.
    #
    # The HDR-related options below are reasonable starting points, but whether
    # the actual output is HDR still depends on your compositor/driver/display.
    cmd = [
        "mpv",
        "--idle=yes",
        "--force-window=yes",
        "--fullscreen=yes",
        "--keep-open=yes",
        "--keep-open-pause=yes",
        "--image-display-duration=inf",
        "--input-ipc-server=" + state.mpv_socket,
        "--vo=gpu-next",
        "--gpu-api=vulkan",
        "--target-colorspace-hint=yes",
        "--hdr-compute-peak=auto",
        "--tone-mapping=clip",
        "--osd-level=1",
        "--msg-level=all=info",
        "--no-audio",
    ]

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )

    deadline = time.time() + 8.0
    while time.time() < deadline:
        if os.path.exists(state.mpv_socket):
            break
        if proc.poll() is not None:
            raise RuntimeError(f"mpv exited early with code {proc.returncode}")
        time.sleep(0.05)

    if not os.path.exists(state.mpv_socket):
        raise RuntimeError("Timed out waiting for mpv IPC socket")

    return proc


def ensure_mpv_running(state: AppState) -> MPVClient:
    with STATE_LOCK:
        socket_path = state.mpv_socket

    client = MPVClient(socket_path)
    try:
        client.get_property("pause")
        return client
    except Exception:
        pass

    proc = launch_mpv(state)
    with STATE_LOCK:
        state.mpv_pid = proc.pid
        state.mpv_running = True
        state.last_error = ""

    # Apply current simulation chain immediately.
    apply_filters(state)
    return MPVClient(socket_path)


def apply_filters(state: AppState) -> None:
    client = MPVClient(state.mpv_socket)
    with STATE_LOCK:
        vf = build_vf_string(state.sim)

    client.set_property("vf", vf)


# -----------------------------
# Utility
# -----------------------------

def resolve_media_path(media_root: str, user_path: str) -> str:
    """
    Allow either:
    - absolute path
    - path relative to media_root
    """
    p = Path(user_path)
    if not p.is_absolute():
        p = Path(media_root) / p

    resolved = p.resolve()
    root = Path(media_root).resolve()

    # Restrict to media_root for LAN safety.
    try:
        resolved.relative_to(root)
    except ValueError:
        raise ValueError("Requested path is outside media_root")

    if not resolved.exists():
        raise FileNotFoundError(f"File not found: {resolved}")

    if not resolved.is_file():
        raise ValueError(f"Not a file: {resolved}")

    return str(resolved)


def json_response(handler: BaseHTTPRequestHandler, code: int, payload: Dict[str, Any]) -> None:
    body = json.dumps(payload, indent=2).encode("utf-8")
    handler.send_response(code)
    handler.send_header("Content-Type", "application/json; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    handler.wfile.write(body)


# -----------------------------
# HTTP API
# -----------------------------

class RequestHandler(BaseHTTPRequestHandler):
    server_version = "mpv-hdr-controller/1.0"

    def do_GET(self) -> None:
        parsed = urlparse(self.path)

        if parsed.path == "/status":
            with STATE_LOCK:
                state = self.server.app_state  # type: ignore[attr-defined]
                payload = {
                    "ok": True,
                    "state": {
                        "media_root": state.media_root,
                        "current_path": state.current_path,
                        "tv_baseline": asdict(state.tv_baseline),
                        "simulation": asdict(state.sim),
                        "mpv_pid": state.mpv_pid,
                        "mpv_running": state.mpv_running,
                        "last_error": state.last_error,
                        "uptime_seconds": round(time.time() - state.start_time, 3),
                    }
                }
            return json_response(self, 200, payload)

        if parsed.path == "/ping":
            return json_response(self, 200, {"ok": True, "message": "pong"})

        return json_response(self, 404, {"ok": False, "error": "Not found"})

    def do_POST(self) -> None:
        parsed = urlparse(self.path)

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            data = json.loads(raw.decode("utf-8"))
        except Exception as e:
            return json_response(self, 400, {"ok": False, "error": f"Invalid JSON: {e}"})

        try:
            if parsed.path == "/load":
                return self.handle_load(data)
            if parsed.path == "/tv-baseline":
                return self.handle_tv_baseline(data)
            if parsed.path == "/simulate":
                return self.handle_simulate(data)
            if parsed.path == "/reset-simulation":
                return self.handle_reset_simulation()
            if parsed.path == "/pause":
                return self.handle_pause(data)
        except Exception as e:
            with STATE_LOCK:
                state = self.server.app_state  # type: ignore[attr-defined]
                state.last_error = str(e)
            return json_response(self, 500, {"ok": False, "error": str(e)})

        return json_response(self, 404, {"ok": False, "error": "Not found"})

    def log_message(self, fmt: str, *args: Any) -> None:
        # Silence default HTTP logging
        return

    def handle_load(self, data: Dict[str, Any]) -> None:
        state: AppState = self.server.app_state  # type: ignore[attr-defined]
        rel_or_abs_path = str(data.get("path", "")).strip()
        if not rel_or_abs_path:
            return json_response(self, 400, {"ok": False, "error": "Missing 'path'"})

        full_path = resolve_media_path(state.media_root, rel_or_abs_path)
        client = ensure_mpv_running(state)
        client.loadfile(full_path, "replace")

        with STATE_LOCK:
            state.current_path = full_path

        return json_response(self, 200, {
            "ok": True,
            "message": "Image loaded",
            "path": full_path
        })

    def handle_tv_baseline(self, data: Dict[str, Any]) -> None:
        state: AppState = self.server.app_state  # type: ignore[attr-defined]
        with STATE_LOCK:
            state.tv_baseline.brightness = str(data.get("brightness", state.tv_baseline.brightness))
            state.tv_baseline.color_temperature = str(
                data.get("color_temperature", state.tv_baseline.color_temperature)
            )
            state.tv_baseline.notes = str(data.get("notes", state.tv_baseline.notes))

        return json_response(self, 200, {
            "ok": True,
            "message": "TV baseline updated",
            "tv_baseline": asdict(state.tv_baseline),
        })

    def handle_simulate(self, data: Dict[str, Any]) -> None:
        state: AppState = self.server.app_state  # type: ignore[attr-defined]
        ensure_mpv_running(state)

        with STATE_LOCK:
            sim = state.sim
            if "brightness_scale" in data:
                sim.brightness_scale = float(data["brightness_scale"])
            if "target_kelvin" in data:
                sim.target_kelvin = int(data["target_kelvin"])
            if "gamma" in data:
                sim.gamma = float(data["gamma"])
            if "saturation" in data:
                sim.saturation = float(data["saturation"])
            if "enabled" in data:
                sim.enabled = bool(data["enabled"])

        apply_filters(state)

        with STATE_LOCK:
            payload = asdict(state.sim)

        return json_response(self, 200, {
            "ok": True,
            "message": "Simulation updated",
            "simulation": payload,
            "vf": build_vf_string(state.sim),
        })

    def handle_reset_simulation(self) -> None:
        state: AppState = self.server.app_state  # type: ignore[attr-defined]
        ensure_mpv_running(state)

        with STATE_LOCK:
            state.sim = SimulationState()

        apply_filters(state)

        return json_response(self, 200, {
            "ok": True,
            "message": "Simulation reset",
            "simulation": asdict(state.sim),
        })

    def handle_pause(self, data: Dict[str, Any]) -> None:
        state: AppState = self.server.app_state  # type: ignore[attr-defined]
        client = ensure_mpv_running(state)
        paused = bool(data.get("paused", True))
        client.set_property("pause", paused)
        return json_response(self, 200, {"ok": True, "paused": paused})


# -----------------------------
# Main
# -----------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HDR-oriented mpv still-image controller")
    p.add_argument("--bind", default="127.0.0.1", help="HTTP bind address")
    p.add_argument("--port", type=int, default=8080, help="HTTP port")
    p.add_argument(
        "--media-root",
        required=True,
        help="Root directory from which images may be loaded",
    )
    p.add_argument(
        "--mpv-socket",
        default="/tmp/mpv-hdr-controller.sock",
        help="UNIX socket path for mpv JSON IPC",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    media_root = str(Path(args.media_root).resolve())

    if not os.path.isdir(media_root):
        print(f"media root does not exist or is not a directory: {media_root}", file=sys.stderr)
        return 2

    state = AppState(
        media_root=media_root,
        mpv_socket=args.mpv_socket,
    )

    # Start mpv up front so failures are obvious.
    try:
        ensure_mpv_running(state)
    except Exception as e:
        print(f"failed to start mpv: {e}", file=sys.stderr)
        return 1

    server = ThreadingHTTPServer((args.bind, args.port), RequestHandler)
    server.app_state = state  # type: ignore[attr-defined]

    print(f"Serving on http://{args.bind}:{args.port}")
    print(f"media_root={media_root}")
    print(f"mpv_socket={state.mpv_socket}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
