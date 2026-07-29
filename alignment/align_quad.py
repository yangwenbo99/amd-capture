#!/usr/bin/env python3
"""
align_quad.py  —  Coarse reference-to-captured image alignment.

Starts a local HTTP server and opens an HTML/JS UI in the browser.
The user drags four vertex handles to wrap the recaptured image in a
quadrilateral overlaid on the reference, then saves the result as JSON.

Usage
-----
    python3 align_quad.py \
        --reference <ref_image> \
        --captured  <cap_image> \
        [--input-json <prev.json>]   \
        [--output    <out.json>]     \
        [--port 0]                   \
        [--no-browser]

Output JSON
-----------
    {
      "reference": {"path": "...", "width": W, "height": H},
      "captured":  {"path": "...", "width": W, "height": H},
      "vertices": {
        "tl": {"x_abs": ..., "y_abs": ..., "x_rel": ..., "y_rel": ...},
        "tr": { ... }, "br": { ... }, "bl": { ... }
      }
    }

    Absolute coords are in reference-image pixels.
    Relative coords are normalised to [0, 1] by reference dimensions.

Requirements
------------
    Python 3.9+  plus  Pillow or OpenImageIO  (for non-web image formats).
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

_WEB_EXTS = frozenset({".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"})
_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".png": "image/png",  ".webp": "image/webp",
    ".gif": "image/gif",  ".bmp":  "image/bmp",
}


def get_image_dims(path: Path) -> tuple[int, int]:
    """Return (width, height).  Tries Pillow then OpenImageIO."""
    try:
        from PIL import Image  # type: ignore[import]
        with Image.open(path) as im:
            return im.size  # (width, height)
    except Exception:
        pass
    try:
        import OpenImageIO as oiio  # type: ignore[import]
        buf = oiio.ImageBuf(str(path))
        s = buf.spec()
        if s.width > 0:
            return s.width, s.height
    except Exception:
        pass
    raise RuntimeError(
        f"Cannot read dimensions of '{path}'. "
        "Install Pillow (pip install pillow) or OpenImageIO."
    )


def to_web_png(path: Path) -> bytes:
    """Convert any image to 8-bit PNG bytes, pixel values clipped to [0, 1]."""
    # OIIO handles EXR, HDR, TIFF, etc.
    try:
        import OpenImageIO as oiio  # type: ignore[import]
        import numpy as np
        buf = oiio.ImageBuf(str(path))
        spec = buf.spec()
        nch = min(spec.nchannels, 4)
        pixels = buf.get_pixels(oiio.FLOAT)
        if pixels is None:
            raise RuntimeError("get_pixels returned None")
        pixels = np.clip(pixels[..., :nch], 0.0, 1.0)
        px8 = (pixels * 255.0 + 0.5).astype("uint8")
        from PIL import Image  # type: ignore[import]
        mode = {1: "L", 2: "LA", 3: "RGB", 4: "RGBA"}[nch]
        out = io.BytesIO()
        Image.fromarray(px8, mode).save(out, "PNG")
        return out.getvalue()
    except Exception:
        pass
    # Pillow-only fallback
    from PIL import Image  # type: ignore[import]
    with Image.open(path) as im:
        out = io.BytesIO()
        im.convert("RGB").save(out, "PNG")
        return out.getvalue()


_IMG_CACHE: dict[str, tuple[bytes, str]] = {}


def serve_image(path: Path) -> tuple[bytes, str]:
    """Return (bytes, content_type), converting non-web formats to PNG."""
    key = str(path)
    if key in _IMG_CACHE:
        return _IMG_CACHE[key]
    ext = path.suffix.lower()
    if ext in _WEB_EXTS:
        result: tuple[bytes, str] = (path.read_bytes(), _MIME.get(ext, "image/png"))
    else:
        result = (to_web_png(path), "image/png")
    _IMG_CACHE[key] = result
    return result


# ---------------------------------------------------------------------------
# Shared server state  (populated by main() before server starts)
# ---------------------------------------------------------------------------

_SRV: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# HTTP request handler
# ---------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        pass  # silence access log

    def _send(self, code: int, ct: str, body: bytes | str) -> None:
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ct)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj: Any) -> None:
        self._send(code, "application/json", json.dumps(obj, indent=2))

    def do_GET(self) -> None:
        route = self.path.split("?")[0]
        if route == "/":
            self._send(200, "text/html; charset=utf-8", HTML_PAGE)
        elif route == "/image/reference":
            data, ct = serve_image(_SRV["ref_path"])
            self._send(200, ct, data)
        elif route == "/image/captured":
            data, ct = serve_image(_SRV["cap_path"])
            self._send(200, ct, data)
        elif route == "/api/config":
            rw, rh = _SRV["ref_dims"]
            cw, ch = _SRV["cap_dims"]
            self._json(200, {
                "ref_filename":   _SRV["ref_path"].name,
                "cap_filename":   _SRV["cap_path"].name,
                "ref_path":       str(_SRV["ref_path"]),
                "cap_path":       str(_SRV["cap_path"]),
                "ref_width":      rw,
                "ref_height":     rh,
                "cap_width":      cw,
                "cap_height":     ch,
                "input_vertices": _SRV.get("input_vertices"),
                "output_path":    str(_SRV["output_path"]),
            })
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        route = self.path.split("?")[0]
        if route != "/api/save":
            self._json(404, {"error": "not found"})
            return
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length)
        try:
            payload = json.loads(raw)
        except Exception as exc:
            self._json(400, {"error": f"invalid JSON: {exc}"})
            return
        out: Path = _SRV["output_path"]
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"[saved] {out}")
        self._json(200, {"ok": True, "saved_to": str(out)})


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Interactive reference-to-captured quad alignment tool."
    )
    p.add_argument("--reference", required=True,
                   help="reference (high-quality input) image path")
    p.add_argument("--captured",  required=True,
                   help="recaptured image path")
    p.add_argument("--input-json", default=None,
                   help="optional existing alignment JSON to pre-load vertices")
    p.add_argument("--output", default=None,
                   help="output JSON path (default: <captured_stem>.align.json)")
    p.add_argument("--port", type=int, default=0,
                   help="HTTP port (0 = pick automatically)")
    p.add_argument("--no-browser", action="store_true",
                   help="do not open the browser automatically")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    ref_path = Path(args.reference).resolve()
    cap_path = Path(args.captured).resolve()
    for p in (ref_path, cap_path):
        if not p.is_file():
            print(f"error: file not found: {p}", file=sys.stderr)
            return 2

    try:
        ref_dims = get_image_dims(ref_path)
        cap_dims = get_image_dims(cap_path)
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    output_path = (
        Path(args.output).resolve()
        if args.output
        else Path.cwd() / f"{cap_path.stem}.align.json"
    )

    input_vertices: list[list[float]] | None = None
    if args.input_json:
        inp = Path(args.input_json).resolve()
        if not inp.is_file():
            print(f"error: input-json not found: {inp}", file=sys.stderr)
            return 2
        data = json.loads(inp.read_text(encoding="utf-8"))
        vd = data.get("vertices", {})
        rw, rh = ref_dims
        if all(k in vd for k in ("tl", "tr", "br", "bl")):
            try:
                # Prefer relative coordinates so a previous alignment maps
                # correctly onto the current reference even if its resolution
                # differs from when the JSON was created.
                input_vertices = [
                    [vd[k]["x_rel"] * rw, vd[k]["y_rel"] * rh]
                    for k in ("tl", "tr", "br", "bl")
                ]
            except KeyError:
                # Fall back to absolute coordinates for older JSON files
                # that lack relative coordinates.
                input_vertices = [
                    [vd[k]["x_abs"], vd[k]["y_abs"]]
                    for k in ("tl", "tr", "br", "bl")
                ]

    _SRV.update({
        "ref_path":       ref_path,
        "cap_path":       cap_path,
        "ref_dims":       ref_dims,
        "cap_dims":       cap_dims,
        "input_vertices": input_vertices,
        "output_path":    output_path,
    })

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    port = server.server_address[1]

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    url = f"http://127.0.0.1:{port}/"
    print(f"QuadAlign running at  {url}")
    print(f"Reference : {ref_path}  ({ref_dims[0]}x{ref_dims[1]})")
    print(f"Captured  : {cap_path}  ({cap_dims[0]}x{cap_dims[1]})")
    print(f"Output    : {output_path}")
    print("Press Ctrl-C to quit.")

    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()

    try:
        while True:
            threading.Event().wait(1)
    except KeyboardInterrupt:
        pass

    server.shutdown()
    return 0


# ---------------------------------------------------------------------------
# Embedded HTML page  (loaded from align_quad.html next to this script)
# ---------------------------------------------------------------------------

def _load_html() -> str:
    html_path = Path(__file__).parent / "align_quad.html"
    if not html_path.is_file():
        raise FileNotFoundError(
            f"UI file not found: {html_path}\n"
            "Make sure align_quad.html is in the same directory."
        )
    return html_path.read_text(encoding="utf-8")


HTML_PAGE = _load_html()


if __name__ == "__main__":
    raise SystemExit(main())
