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
- OpenImageIO Python bindings (for scripted HDR augmentation mode)
- numpy
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
  whenever simulation parameters change (for `mpv_filters` mode).
- In `scripted_hdr` mode, this script generates a temporary EXR file via
  OpenImageIO + numpy
  and loads it in mpv with default settings (`vf=""`).
- It assumes Kelvin simulation should be interpreted relative to a neutral
  baseline of 6500 K unless you choose a different value.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, asdict, field
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Optional, Protocol
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
    augmentation_mode: str = "mpv_filters"
    crop_enabled: bool = False
    crop_ratio: str = "16:9"
    crop_mode: str = "crop"

@dataclass
class AppState:
    media_root: str
    source_path: str = ""
    current_path: str = ""
    current_generated_path: str = ""
    black_image_name: str = ".mpv_controller_black.ppm"
    tv_baseline: TVBaseline = field(default_factory=TVBaseline)
    sim: SimulationState = field(default_factory=SimulationState)
    mpv_pid: Optional[int] = None
    mpv_socket: str = "/tmp/mpv-hdr-controller.sock"
    mpv_running: bool = False
    mpv_vo: str = "gpu-next"
    mpv_gpu_api: str = "vulkan"
    mpv_msg_level: str = "all=info"
    mpv_log_file: str = "/tmp/mpv-hdr-controller-mpv.log"
    augmentation_temp_dir: str = "/tmp/mpv-hdr-controller-aug"
    fullscreen: bool = True
    mpv_last_cmd: str = ""
    mpv_recent_output: list[str] = field(default_factory=list)
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


def normalize_augmentation_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    aliases = {
        "mpv": "mpv_filters",
        "mpv-filter": "mpv_filters",
        "mpv-filters": "mpv_filters",
        "scripted": "scripted_hdr",
        "script": "scripted_hdr",
        "scripted-hdr": "scripted_hdr",
    }
    normalized = aliases.get(value, value)
    if normalized not in {"mpv_filters", "scripted_hdr"}:
        raise ValueError("augmentation_mode must be one of: mpv_filters, scripted_hdr")
    return normalized


def parse_ratio(value: str) -> float:
    text = str(value).strip()
    if ":" in text:
        left, right = text.split(":", 1)
        num = float(left.strip())
        den = float(right.strip())
        if den <= 0:
            raise ValueError("crop_ratio denominator must be > 0")
        ratio = num / den
    else:
        ratio = float(text)
    if ratio <= 0:
        raise ValueError("crop_ratio must be > 0")
    return ratio


def normalize_crop_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    aliases = {
        "crop": "crop",
        "pad": "reflect_pad",
        "padding": "reflect_pad",
        "reflect": "reflect_pad",
        "reflect_pad": "reflect_pad",
        "reflective_pad": "reflect_pad",
        "reflective_padding": "reflect_pad",
        "reflexive_pad": "reflect_pad",
        "reflexive_padding": "reflect_pad",
    }
    normalized = aliases.get(value, value)
    if normalized not in {"crop", "reflect_pad"}:
        raise ValueError("crop_mode must be one of: crop, reflect_pad")
    return normalized


class ScriptedAugmentationStep(Protocol):
    def build_filter(self, sim: SimulationState) -> Optional[str]:
        ...


class CropToRatioStep:
    def build_filter(self, sim: SimulationState) -> Optional[str]:
        if not sim.crop_enabled:
            return None
        if normalize_crop_mode(sim.crop_mode) != "crop":
            return None
        ratio = parse_ratio(sim.crop_ratio)
        ratio_expr = f"{ratio:.12f}"
        return (
            "crop="
            f"w='if(gt(iw/ih,{ratio_expr}),ih*{ratio_expr},iw)':"
            f"h='if(gt(iw/ih,{ratio_expr}),ih,iw/{ratio_expr})':"
            "x='(iw-w)/2':y='(ih-h)/2'"
        )


class EqStep:
    def build_filter(self, sim: SimulationState) -> Optional[str]:
        if not sim.enabled:
            return None
        eq_brightness = brightness_scale_to_eq_brightness(sim.brightness_scale)
        gamma = clamp(sim.gamma, 0.5, 3.0)
        saturation = clamp(sim.saturation, 0.0, 3.0)
        return (
            f"eq=brightness={eq_brightness:.6f}:contrast=1.0:"
            f"saturation={saturation:.6f}:gamma={gamma:.6f}"
        )


class ColorTemperatureStep:
    def build_filter(self, sim: SimulationState) -> Optional[str]:
        if not sim.enabled:
            return None
        kelvin = int(clamp(sim.target_kelvin, 1000, 40000))
        return f"colortemperature=temperature={kelvin}:mix=1.0:pl=0.0"


AUGMENTATION_STEPS: tuple[ScriptedAugmentationStep, ...] = (
    CropToRatioStep(),
    EqStep(),
    ColorTemperatureStep(),
)


def build_mpv_filter_components(sim: SimulationState) -> list[str]:
    filters: list[str] = []
    eq = EqStep().build_filter(sim)
    if eq:
        filters.append(eq)
    color_temp = ColorTemperatureStep().build_filter(sim)
    if color_temp:
        filters.append(color_temp)
    return filters


def build_scripted_filters(sim: SimulationState) -> list[str]:
    filters: list[str] = []
    for step in AUGMENTATION_STEPS:
        built = step.build_filter(sim)
        if built:
            filters.append(built)
    return filters


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
    graph = ",".join(build_mpv_filter_components(sim))
    if not graph:
        return ""
    return f'lavfi=[{graph}]'


def safe_unlink(path: str) -> None:
    if not path:
        return
    try:
        Path(path).unlink(missing_ok=True)
    except Exception:
        # Best-effort cleanup only.
        return


def cct_to_xy_blackbody(temperature_kelvin: float) -> tuple[float, float]:
    """
    Approximate blackbody chromaticity (x, y) from CCT in Kelvin.
    """
    t = clamp(float(temperature_kelvin), 1667.0, 25000.0)
    if t <= 4000.0:
        x = (
            -0.2661239e9 / (t ** 3)
            - 0.2343580e6 / (t ** 2)
            + 0.8776956e3 / t
            + 0.179910
        )
    else:
        x = (
            -3.0258469e9 / (t ** 3)
            + 2.1070379e6 / (t ** 2)
            + 0.2226347e3 / t
            + 0.240390
        )

    if t <= 2222.0:
        y = (
            -1.1063814 * (x ** 3)
            - 1.34811020 * (x ** 2)
            + 2.18555832 * x
            - 0.20219683
        )
    elif t <= 4000.0:
        y = (
            -0.9549476 * (x ** 3)
            - 1.37418593 * (x ** 2)
            + 2.09137015 * x
            - 0.16748867
        )
    else:
        y = (
            3.0817580 * (x ** 3)
            - 5.87338670 * (x ** 2)
            + 3.75112997 * x
            - 0.37001483
        )
    return x, y


def xy_to_xyz(x: float, y: float, luminance: float = 1.0) -> tuple[float, float, float]:
    if y == 0.0:
        raise ValueError("y must be nonzero")
    x_xyz = x * luminance / y
    z_xyz = (1.0 - x - y) * luminance / y
    return (x_xyz, luminance, z_xyz)


def xyz_to_linear_srgb(xyz: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = xyz
    # Standard XYZ -> linear sRGB matrix.
    r = 3.2404542 * x + -1.5371385 * y + -0.4985314 * z
    g = -0.9692660 * x + 1.8760108 * y + 0.0415560 * z
    b = 0.0556434 * x + -0.2040259 * y + 1.0572252 * z
    return (r, g, b)


def kelvin_channel_gains(
    target_kelvin: float,
    baseline_kelvin: float = 6500.0,
) -> tuple[float, float, float]:
    src_xy = cct_to_xy_blackbody(baseline_kelvin)
    dst_xy = cct_to_xy_blackbody(target_kelvin)
    src_rgb = xyz_to_linear_srgb(xy_to_xyz(src_xy[0], src_xy[1], 1.0))
    dst_rgb = xyz_to_linear_srgb(xy_to_xyz(dst_xy[0], dst_xy[1], 1.0))

    r_gain = dst_rgb[0] / src_rgb[0] if src_rgb[0] != 0 else 1.0
    g_gain = dst_rgb[1] / src_rgb[1] if src_rgb[1] != 0 else 1.0
    b_gain = dst_rgb[2] / src_rgb[2] if src_rgb[2] != 0 else 1.0

    # Normalize by green to keep relative luminance shifts similar
    # to the previous simulation intent.
    if g_gain != 0:
        r_gain /= g_gain
        b_gain /= g_gain
        g_gain = 1.0
    return (r_gain, g_gain, b_gain)


def create_augmented_hdr_image(state: AppState, source_path: str, sim: SimulationState) -> str:
    try:
        import OpenImageIO as oiio
    except Exception as exc:
        raise RuntimeError(
            "scripted_hdr mode requires OpenImageIO Python bindings "
            "(import OpenImageIO failed)"
        ) from exc

    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "scripted_hdr mode requires numpy (import numpy failed)"
        ) from exc

    if not sim.enabled and not sim.crop_enabled and source_path.lower().endswith(".exr"):
        return source_path

    temp_dir = Path(state.augmentation_temp_dir)
    temp_dir.mkdir(parents=True, exist_ok=True)

    sim_key = json.dumps(asdict(sim), sort_keys=True)
    cache_key = f"{source_path}\n{sim_key}\noiio-v1"
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    stem = Path(source_path).stem
    out_path = temp_dir / f"{stem}.aug-{digest}.exr"
    if out_path.exists():
        return str(out_path)

    src_buf = oiio.ImageBuf(source_path)
    src_err = src_buf.geterror()
    if src_err:
        raise RuntimeError(f"failed to read source image: {src_err}")

    pixels = src_buf.get_pixels(oiio.FLOAT)
    if pixels is None:
        raise RuntimeError(f"failed to decode source pixels: {source_path}")
    pixels = pixels.astype(np.float32, copy=False)

    if pixels.ndim != 3:
        raise RuntimeError(f"unexpected pixel layout for {source_path}: shape={pixels.shape}")

    if sim.crop_enabled:
        target_ratio = parse_ratio(sim.crop_ratio)
        crop_mode = normalize_crop_mode(sim.crop_mode)
        h, w, _ = pixels.shape
        if h <= 0 or w <= 0:
            raise RuntimeError(f"invalid source image dimensions: {w}x{h}")
        src_ratio = float(w) / float(h)
        if crop_mode == "crop":
            if src_ratio > target_ratio:
                new_w = int(round(h * target_ratio))
                new_h = h
            else:
                new_w = w
                new_h = int(round(w / target_ratio))
            new_w = max(1, min(w, new_w))
            new_h = max(1, min(h, new_h))
            x0 = (w - new_w) // 2
            y0 = (h - new_h) // 2
            pixels = pixels[y0:y0 + new_h, x0:x0 + new_w, :]
        else:
            pad_top = 0
            pad_bottom = 0
            pad_left = 0
            pad_right = 0
            if src_ratio > target_ratio:
                new_h = max(h, int(round(w / target_ratio)))
                total_pad = max(0, new_h - h)
                pad_top = total_pad // 2
                pad_bottom = total_pad - pad_top
            else:
                new_w = max(w, int(round(h * target_ratio)))
                total_pad = max(0, new_w - w)
                pad_left = total_pad // 2
                pad_right = total_pad - pad_left
            if any((pad_top, pad_bottom, pad_left, pad_right)):
                reflect_mode = "reflect" if h > 1 and w > 1 else "edge"
                pixels = np.pad(
                    pixels,
                    ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                    mode=reflect_mode,
                )

    if sim.enabled and pixels.shape[2] >= 1:
        rgb_channels = min(3, int(pixels.shape[2]))
        rgb = pixels[..., :rgb_channels]

        brightness_scale = max(0.0, float(sim.brightness_scale))
        if brightness_scale != 1.0:
            rgb *= brightness_scale

        if rgb_channels == 3:
            saturation = clamp(sim.saturation, 0.0, 3.0)
            if saturation != 1.0:
                luma = (
                    0.2126 * rgb[..., 0]
                    + 0.7152 * rgb[..., 1]
                    + 0.0722 * rgb[..., 2]
                )[..., np.newaxis]
                rgb = luma + (rgb - luma) * saturation

            target_kelvin = float(int(clamp(sim.target_kelvin, 1000, 40000)))
            gain_r, gain_g, gain_b = kelvin_channel_gains(target_kelvin, 6500.0)
            rgb[..., 0] *= gain_r
            rgb[..., 1] *= gain_g
            rgb[..., 2] *= gain_b

        gamma = clamp(sim.gamma, 0.5, 3.0)
        if gamma != 1.0:
            rgb = np.power(np.clip(rgb, 0.0, None), 1.0 / gamma)

        rgb = np.clip(rgb, 0.0, None)
        pixels[..., :rgb_channels] = rgb

    height, width, channels = pixels.shape
    out_spec = oiio.ImageSpec(width, height, channels, oiio.FLOAT)
    out_buf = oiio.ImageBuf(out_spec)
    roi = oiio.ROI(0, width, 0, height, 0, 1, 0, channels)
    out_buf.set_pixels(roi, pixels)
    if not out_buf.write(str(out_path)):
        raise RuntimeError(f"failed to write augmented image: {out_buf.geterror()}")
    return str(out_path)


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
        "--keep-open=yes",
        "--keep-open-pause=yes",
        "--image-display-duration=inf",
        "--input-ipc-server=" + state.mpv_socket,
        "--vo=" + state.mpv_vo,
        "--gpu-api=" + state.mpv_gpu_api,
        "--target-colorspace-hint=yes",
        "--hdr-compute-peak=auto",
        "--tone-mapping=clip",
        "--osd-level=1",
        "--msg-level=" + state.mpv_msg_level,
        "--log-file=" + state.mpv_log_file,
        "--no-audio",
    ]
    if state.fullscreen:
        cmd.append("--fullscreen=yes")

    # Start with a clean log file for each launch attempt.
    Path(state.mpv_log_file).parent.mkdir(parents=True, exist_ok=True)
    Path(state.mpv_log_file).write_text("", encoding="utf-8")

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    with STATE_LOCK:
        state.mpv_last_cmd = " ".join(cmd)
        state.mpv_recent_output = []

    recent_lines: deque[str] = deque(maxlen=40)
    if proc.stdout is not None:
        def _collect_output() -> None:
            for line in proc.stdout:
                recent_lines.append(line.rstrip("\n"))

        threading.Thread(target=_collect_output, daemon=True).start()

    deadline = time.time() + 8.0
    while time.time() < deadline:
        if os.path.exists(state.mpv_socket):
            break
        if proc.poll() is not None:
            with STATE_LOCK:
                state.mpv_recent_output = list(recent_lines)
            details = " | ".join(recent_lines) if recent_lines else "no process output captured"
            raise RuntimeError(
                f"mpv exited early with code {proc.returncode}. "
                f"log={state.mpv_log_file}. output={details}"
            )
        time.sleep(0.05)

    if not os.path.exists(state.mpv_socket):
        with STATE_LOCK:
            state.mpv_recent_output = list(recent_lines)
        raise RuntimeError("Timed out waiting for mpv IPC socket")

    with STATE_LOCK:
        state.mpv_recent_output = list(recent_lines)
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

    # Apply current simulation mode immediately.
    apply_simulation(state, reload_media=False)
    return MPVClient(socket_path)


def apply_simulation(state: AppState, reload_media: bool = True) -> None:
    client = MPVClient(state.mpv_socket)
    with STATE_LOCK:
        sim = SimulationState(**asdict(state.sim))
        mode = normalize_augmentation_mode(sim.augmentation_mode)
        source_path = state.source_path
        old_generated_path = state.current_generated_path

    if mode == "mpv_filters":
        client.set_property("vf", build_vf_string(sim))
        if reload_media and source_path:
            client.loadfile(source_path, "replace")
        with STATE_LOCK:
            state.current_path = source_path if source_path else state.current_path
            state.current_generated_path = ""
        safe_unlink(old_generated_path)
        return

    client.set_property("vf", "")
    if reload_media and source_path:
        generated_path = create_augmented_hdr_image(state, source_path, sim)
        client.loadfile(generated_path, "replace")
        with STATE_LOCK:
            state.current_path = generated_path
            state.current_generated_path = generated_path
        if old_generated_path and old_generated_path != generated_path:
            safe_unlink(old_generated_path)


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


def ensure_black_image(media_root: str, image_name: str) -> str:
    path = Path(media_root) / image_name
    if path.exists():
        if not path.is_file():
            raise ValueError(f"Black image path exists but is not a file: {path}")
        return str(path.resolve())

    # Tiny binary PPM to avoid extra dependencies (mpv supports .ppm images).
    width = 64
    height = 64
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    pixels = b"\x00" * (width * height * 3)
    path.write_bytes(header + pixels)
    return str(path.resolve())


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
                        "source_path": state.source_path,
                        "current_path": state.current_path,
                        "current_generated_path": state.current_generated_path,
                        "tv_baseline": asdict(state.tv_baseline),
                        "simulation": asdict(state.sim),
                        "mpv_pid": state.mpv_pid,
                        "mpv_running": state.mpv_running,
                        "mpv_vo": state.mpv_vo,
                        "mpv_gpu_api": state.mpv_gpu_api,
                        "mpv_msg_level": state.mpv_msg_level,
                        "mpv_log_file": state.mpv_log_file,
                        "augmentation_temp_dir": state.augmentation_temp_dir,
                        "mpv_last_cmd": state.mpv_last_cmd,
                        "mpv_recent_output": state.mpv_recent_output,
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
            if parsed.path == "/load-black":
                return self.handle_load_black(data)
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
        with STATE_LOCK:
            state.source_path = full_path

        ensure_mpv_running(state)
        apply_simulation(state, reload_media=True)

        return json_response(self, 200, {
            "ok": True,
            "message": "Image loaded",
            "path": full_path,
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
            if "augmentation_mode" in data:
                sim.augmentation_mode = normalize_augmentation_mode(str(data["augmentation_mode"]))
            if "crop_enabled" in data:
                sim.crop_enabled = bool(data["crop_enabled"])
            if "crop_ratio" in data:
                # Validate up front so we fail fast on malformed values.
                parse_ratio(str(data["crop_ratio"]))
                sim.crop_ratio = str(data["crop_ratio"])
            if "crop_mode" in data:
                sim.crop_mode = normalize_crop_mode(str(data["crop_mode"]))

        apply_simulation(state, reload_media=True)

        with STATE_LOCK:
            payload = asdict(state.sim)

        return json_response(self, 200, {
            "ok": True,
            "message": "Simulation updated",
            "simulation": payload,
            "vf": build_vf_string(state.sim),
            "scripted_filters": build_scripted_filters(state.sim),
        })

    def handle_load_black(self, data: Dict[str, Any]) -> None:
        state: AppState = self.server.app_state  # type: ignore[attr-defined]
        ensure_mpv_running(state)
        duration_sec = float(data.get("duration_sec", 0.0))
        if duration_sec < 0:
            return json_response(
                self,
                400,
                {"ok": False, "error": "duration_sec must be >= 0"},
            )

        black_path = ensure_black_image(state.media_root, state.black_image_name)
        client = MPVClient(state.mpv_socket)
        client.loadfile(black_path, "replace")

        with STATE_LOCK:
            old_generated = state.current_generated_path
            state.source_path = black_path
            state.current_path = black_path
            state.current_generated_path = ""
        safe_unlink(old_generated)

        return json_response(
            self,
            200,
            {
                "ok": True,
                "message": "Black image loaded",
                "path": black_path,
                "duration_sec": duration_sec,
            },
        )

    def handle_reset_simulation(self) -> None:
        state: AppState = self.server.app_state  # type: ignore[attr-defined]
        ensure_mpv_running(state)

        with STATE_LOCK:
            old_generated = state.current_generated_path
            state.sim = SimulationState()
            state.current_generated_path = ""

        safe_unlink(old_generated)
        apply_simulation(state, reload_media=True)

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
    p.add_argument(
        "--mpv-vo",
        default="gpu-next",
        help="mpv --vo backend (try gpu on older systems)",
    )
    p.add_argument(
        "--mpv-gpu-api",
        default="vulkan",
        help="mpv --gpu-api backend (try opengl if vulkan fails)",
    )
    p.add_argument(
        "--mpv-msg-level",
        default="all=info",
        help="mpv --msg-level string (e.g. all=debug)",
    )
    p.add_argument(
        "--mpv-log-file",
        default="/tmp/mpv-hdr-controller-mpv.log",
        help="path for mpv internal log output",
    )
    p.add_argument(
        "--augmentation-temp-dir",
        default="/tmp/mpv-hdr-controller-aug",
        help="directory for generated temporary augmented HDR image files",
    )
    p.add_argument(
        "--windowed",
        action="store_true",
        help="start mpv without fullscreen for debugging",
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
        mpv_vo=args.mpv_vo,
        mpv_gpu_api=args.mpv_gpu_api,
        mpv_msg_level=args.mpv_msg_level,
        mpv_log_file=args.mpv_log_file,
        augmentation_temp_dir=str(Path(args.augmentation_temp_dir).resolve()),
        fullscreen=not args.windowed,
    )

    try:
        ensure_black_image(state.media_root, state.black_image_name)
    except Exception as e:
        print(f"failed to create black image: {e}", file=sys.stderr)
        return 2

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
