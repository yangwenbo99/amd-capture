#!/usr/bin/env python3
"""
apply_geometry_transform.py

Batch-reproduce ONLY the geometry transform performed by the
``scripted_hdr`` augmentation mode in ``display/mpv_driver.py``
(``create_augmented_hdr_image``).

The display driver, when ``crop_enabled`` is set, performs a fixed-ratio
center *crop* or a reflective *pad* on the source pixels before applying
colour transforms (brightness / saturation / colour temperature / gamma).
This script reproduces the spatial geometry step byte-for-byte, but skips
every colour transform, so the output differs from the source only in
cropping / padding.

The geometry instruction mirrors the relevant arguments of
``control/control_capture_session.py``:

    --image-dir        (here: --input-dir)      input directory of images
    --image-suffixes                             which files to include
    --crop-enabled / --crop-disabled             enable/disable geometry
    --crop-ratio                                 target aspect ratio
    --crop-mode        crop | reflect_pad        crop vs reflective padding

Output format
-------------
By default the output extension is preserved from each input file.
Use ``--output-ext`` to override (e.g. ``.exr``, ``.png``, ``.jpg``).

- **EXR / TIFF / other float-capable formats**: written as 32-bit float,
  no value clipping.
- **JPEG** (``.jpg`` / ``.jpeg``): written as 8-bit sRGB.  Alpha channels
  are dropped.  Float pixels are clamped to [0, 1] then scaled to [0, 255].
  Use ``--jpeg-quality`` to control compression (default 95).
- **PNG** (``.png``): written as 8-bit or 16-bit integer (see
  ``--png-bit-depth``).  Float pixels are clamped to [0, 1] then scaled.

  For HDR source images (EXR) the float values often exceed 1.0; pixels are
  simply clamped — a warning is printed when clipping occurs.

Usage
-----
    # Lossless EXR geometry crop
    python3 apply_geometry_transform.py \
        --input-dir /data/photos \
        --output-dir /data/photos_geo \
        --crop-enabled \
        --crop-ratio 16:9 \
        --crop-mode crop

    # Lossy JPEG output
    python3 apply_geometry_transform.py \
        --input-dir /data/photos \
        --output-dir /data/photos_geo_jpg \
        --crop-enabled \
        --output-ext .jpg \
        --jpeg-quality 92

    # 16-bit PNG output
    python3 apply_geometry_transform.py \
        --input-dir /data/photos \
        --output-dir /data/photos_geo_png \
        --crop-enabled \
        --output-ext .png \
        --png-bit-depth 16

Requirements
------------
- OpenImageIO Python bindings (same reader/writer used by the display driver)
- numpy
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence


# ---------------------------------------------------------------------------
# Geometry helpers (kept identical to display/mpv_driver.py)
# ---------------------------------------------------------------------------

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


def apply_geometry(pixels, crop_ratio: str, crop_mode: str, np):
    """
    Apply the same crop / reflective-pad geometry as
    create_augmented_hdr_image in display/mpv_driver.py.

    ``pixels`` is an (H, W, C) float array. Returns the transformed array.
    """
    if pixels.ndim != 3:
        raise RuntimeError(f"unexpected pixel layout: shape={pixels.shape}")

    target_ratio = parse_ratio(crop_ratio)
    mode = normalize_crop_mode(crop_mode)
    h, w, _ = pixels.shape
    if h <= 0 or w <= 0:
        raise RuntimeError(f"invalid source image dimensions: {w}x{h}")
    src_ratio = float(w) / float(h)

    if mode == "crop":
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

    return pixels


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

# Extensions that require integer pixel data (no HDR float output).
_JPEG_EXTS: frozenset[str] = frozenset({".jpg", ".jpeg"})
_PNG_EXTS: frozenset[str] = frozenset({".png"})
_LDR_EXTS: frozenset[str] = _JPEG_EXTS | _PNG_EXTS


def iter_images(input_dir: Path, suffixes: Sequence[str]):
    suffix_set = {s.lower() for s in suffixes}
    for p in sorted(input_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in suffix_set:
            yield p


def process_image(
    src_path: Path,
    dst_path: Path,
    crop_enabled: bool,
    crop_ratio: str,
    crop_mode: str,
    oiio,
    np,
    jpeg_quality: int = 95,
    png_bit_depth: int = 8,
) -> None:
    src_buf = oiio.ImageBuf(str(src_path))
    src_err = src_buf.geterror()
    if src_err:
        raise RuntimeError(f"failed to read source image: {src_err}")

    pixels = src_buf.get_pixels(oiio.FLOAT)
    if pixels is None:
        raise RuntimeError(f"failed to decode source pixels: {src_path}")
    pixels = pixels.astype(np.float32, copy=False)

    if pixels.ndim != 3:
        raise RuntimeError(
            f"unexpected pixel layout for {src_path}: shape={pixels.shape}"
        )

    if crop_enabled:
        pixels = apply_geometry(pixels, crop_ratio, crop_mode, np)

    out_ext = dst_path.suffix.lower()

    if out_ext in _LDR_EXTS:
        pixels, spec_type = _prepare_ldr_pixels(
            pixels,
            out_ext=out_ext,
            png_bit_depth=png_bit_depth,
            src_path=src_path,
            oiio=oiio,
            np=np,
        )
    else:
        spec_type = oiio.FLOAT

    # Ensure a contiguous buffer for OIIO set_pixels after slicing/padding.
    pixels = np.ascontiguousarray(pixels)

    height, width, channels = pixels.shape
    out_spec = oiio.ImageSpec(width, height, channels, spec_type)
    if out_ext in _JPEG_EXTS:
        out_spec.attribute("CompressionQuality", int(jpeg_quality))

    out_buf = oiio.ImageBuf(out_spec)
    roi = oiio.ROI(0, width, 0, height, 0, 1, 0, channels)
    out_buf.set_pixels(roi, pixels)
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    if not out_buf.write(str(dst_path)):
        raise RuntimeError(f"failed to write image: {out_buf.geterror()}")


def _prepare_ldr_pixels(
    pixels,
    out_ext: str,
    png_bit_depth: int,
    src_path: Path,
    oiio,
    np,
):
    """
    Convert float pixels to an integer LDR representation for JPEG/PNG.

    Returns (integer_pixels, oiio_spec_type). Float values are clamped to
    [0, 1] and scaled to the target bit depth. JPEG output drops alpha.
    """
    # JPEG has no alpha; keep at most 3 (RGB) channels.
    if out_ext in _JPEG_EXTS and pixels.shape[2] > 3:
        pixels = pixels[..., :3]

    # Warn once per image if HDR / out-of-range values will be clipped.
    finite = pixels[np.isfinite(pixels)]
    if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
        print(
            f"[warn] {src_path.name}: pixel values outside [0, 1] "
            f"(min={float(finite.min()):.4f}, max={float(finite.max()):.4f}) "
            f"clipped for {out_ext} output"
        )

    clipped = np.clip(np.nan_to_num(pixels, nan=0.0), 0.0, 1.0)

    if out_ext in _PNG_EXTS and int(png_bit_depth) == 16:
        scaled = np.rint(clipped * 65535.0).astype(np.uint16)
        return scaled, oiio.UINT16

    scaled = np.rint(clipped * 255.0).astype(np.uint8)
    return scaled, oiio.UINT8


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Reproduce only the geometry transform (crop / reflect_pad) of "
            "the scripted_hdr augmentation mode in display/mpv_driver.py, "
            "without any colour transforms."
        )
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="directory containing input images (non-recursive)",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="directory to write geometry-transformed images",
    )
    p.add_argument(
        "--image-suffixes",
        default=".png,.bmp,.jpg,.jpeg,.webp,.tif,.tiff,.exr",
        help=(
            "comma-separated file suffixes to include, "
            "example: .bmp,.png"
        ),
    )
    # Geometry instruction, mirroring control/control_capture_session.py.
    p.add_argument(
        "--crop-enabled",
        dest="crop_enabled",
        action="store_true",
        help="enable fixed-ratio center crop / pad geometry",
    )
    p.add_argument(
        "--crop-disabled",
        dest="crop_enabled",
        action="store_false",
        help="disable geometry (copy through unchanged)",
    )
    p.set_defaults(crop_enabled=True)
    p.add_argument(
        "--crop-ratio",
        default="16:9",
        help=(
            "crop ratio used when crop is enabled (default: 16:9). "
            "Accepts values like 16:9 or 1.77778."
        ),
    )
    p.add_argument(
        "--crop-mode",
        choices=("crop", "reflect_pad"),
        default="crop",
        help=(
            "geometry handling when crop is enabled: 'crop' center-crops, "
            "'reflect_pad' keeps full image via reflective padding "
            "(default: crop)"
        ),
    )
    p.add_argument(
        "--output-ext",
        default=None,
        help=(
            "output file extension/format override (e.g. .exr, .png, .jpg). "
            "If omitted, keeps the input file extension. EXR is recommended "
            "for lossless float output."
        ),
    )
    p.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help=(
            "JPEG compression quality 1-100 when writing .jpg / .jpeg output "
            "(default: 95)"
        ),
    )
    p.add_argument(
        "--png-bit-depth",
        type=int,
        choices=(8, 16),
        default=8,
        help="bit depth when writing .png output: 8 or 16 (default: 8)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing files in --output-dir",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()

    try:
        import OpenImageIO as oiio
    except Exception as exc:
        raise SystemExit(
            "this script requires OpenImageIO Python bindings "
            f"(import OpenImageIO failed): {exc}"
        )
    try:
        import numpy as np
    except Exception as exc:
        raise SystemExit(f"this script requires numpy (import numpy failed): {exc}")

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise SystemExit(f"input dir is not a directory: {input_dir}")

    # Validate geometry args up front so we fail fast on malformed values.
    crop_mode = normalize_crop_mode(args.crop_mode)
    if args.crop_enabled:
        parse_ratio(args.crop_ratio)

    suffixes = [s.strip() for s in args.image_suffixes.split(",") if s.strip()]
    if not suffixes:
        raise SystemExit("no valid --image-suffixes provided")

    out_ext = args.output_ext
    if out_ext is not None and not out_ext.startswith("."):
        out_ext = "." + out_ext

    if not (1 <= int(args.jpeg_quality) <= 100):
        raise SystemExit("--jpeg-quality must be between 1 and 100")

    output_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped = 0
    failed = 0
    for src in iter_images(input_dir, suffixes):
        ext = out_ext if out_ext is not None else src.suffix
        dst = output_dir / (src.stem + ext)

        if dst.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            process_image(
                src_path=src,
                dst_path=dst,
                crop_enabled=bool(args.crop_enabled),
                crop_ratio=args.crop_ratio,
                crop_mode=crop_mode,
                oiio=oiio,
                np=np,
                jpeg_quality=int(args.jpeg_quality),
                png_bit_depth=int(args.png_bit_depth),
            )
            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[fail] {src}: {exc}")

    print(
        f"[ok] processed={processed} skipped={skipped} failed={failed} "
        f"crop_enabled={bool(args.crop_enabled)} "
        f"crop_ratio={args.crop_ratio} crop_mode={crop_mode} "
        f"output_dir={output_dir}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
