#!/usr/bin/env python3
r"""
sample_patches.py  —  Substep 2 of the patch-pair dataset pipeline.

Samples ``k`` random ``(P, P)`` patches from the captured image and crops the
corresponding region from the resized reference image produced by
``crop_resize_reference.py`` (substep 1).

Geometry
--------
The cropped reference has size ``(S*(H_c+2*p_r), S*(W_c+2*p_r))`` and its pixel
``(out_row, out_col)`` corresponds to the captured continuous position
``(out_col/S - p_r, out_row/S - p_r)``.  Therefore a captured patch whose
top-left corner sits at ``(x0, y0)`` maps to a reference crop whose top-left is

    ref_x0 = round(S * x0)
    ref_y0 = round(S * y0)

To leave room for the small shifts of an imperfect alignment, the reference crop
is larger than the captured patch: its size is

    ( S*(P + 2*p_r),  S*(P + 2*p_r) )

i.e. it keeps the ``p_r``-pixel border on every side.

Bayer constraint
----------------
Every sampled captured coordinate ``(x0, y0)`` is forced to be divisible by 2 so
that the Bayer mosaic phase of the patch matches the full frame.

Sampling strategies
--------------------
* ``--uniform``: sample the top-left uniformly over the valid range
  ``[0, W_c - P] x [0, H_c - P]``.  Simple, but under-represents pixels near the
  image border (they can only be covered by a few patch positions).

* edge-aware (default): sample the top-left over the extended region
  ``[-P/2, W_c + P/2) x [-P/2, H_c + P/2)`` and clamp it back into the valid
  range.  A top-left that lands outside the image "bumps" into the nearest valid
  position, so border pixels are covered as often as central ones.

Output
------
Patch images are written under ``--output-dir`` (``cap/`` and ``ref/``
subfolders) and a single ``patches.json`` metadata file records, for every pair,
the captured/reference coordinates and file names — the input to substep 3
(``sift_align_patches.py``).

Usage
-----
    python3 sample_patches.py \
        --crop-json cap.ref_crop.json \  # metadata from substep 1
        [--captured cap.bmp]          \  # overrides path in crop-json
        [--cropped-reference cap.ref_crop.png] \
        --patch-size 256              \  # P
        --num-patches 32              \  # k
        [--uniform]                   \  # disable edge-aware sampling
        [--seed 0]                    \
        [--output-dir patches]        \
        [--format png|npy|exr]

``--crop-json`` may instead be a *directory* of crop metadata JSONs (as
produced by ``crop_resize_reference.py`` in its directory mode).  Each JSON is
processed independently; patches are written to ``<output-dir>/<captured_stem>/``
so the per-image output is self-contained.  ``--captured`` and
``--cropped-reference`` are ignored in directory mode.

Requirements
------------
    Python 3.9+, numpy, and Pillow or OpenImageIO.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_any(path: Path) -> np.ndarray:
    """Load an image (.npy / OpenImageIO / Pillow) as (H, W, C) float32 in [0,1]."""
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path)).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return arr
    # Prefer OpenImageIO (handles EXR/HDR/TIFF), fall back to Pillow.
    try:
        import OpenImageIO as oiio  # type: ignore[import]
        buf = oiio.ImageBuf(str(path))
        spec = buf.spec()
        if spec.width <= 0:
            raise RuntimeError(buf.geterror() or "empty image")
        px = np.asarray(buf.get_pixels(oiio.FLOAT), dtype=np.float32)
        if px.ndim == 2:
            px = px[:, :, None]
        return px
    except Exception:
        pass
    from PIL import Image  # type: ignore[import]
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def save_patch(arr: np.ndarray, path: Path, fmt: str) -> None:
    """Save a (H, W, C) float32 patch in *fmt*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(str(path), arr.astype(np.float32))
    elif fmt == "exr":
        import OpenImageIO as oiio  # type: ignore[import]
        h, w, c = arr.shape
        spec = oiio.ImageSpec(w, h, c, oiio.FLOAT)
        buf = oiio.ImageBuf(spec)
        buf.set_pixels(oiio.ROI(0, w, 0, h, 0, 1, 0, c),
                       np.ascontiguousarray(arr, dtype=np.float32))
        if not buf.write(str(path)):
            raise RuntimeError(buf.geterror() or "EXR write failed")
    else:  # png
        from PIL import Image as PILImage  # type: ignore[import]
        arr8 = (np.clip(arr, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        c = arr8.shape[2]
        if c == 1:
            PILImage.fromarray(arr8[:, :, 0], "L").save(str(path))
        else:
            mode = {3: "RGB", 4: "RGBA"}.get(c, "RGB")
            PILImage.fromarray(arr8[:, :, : len(mode)], mode).save(str(path))


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

def _even(v: int) -> int:
    """Round *v* down to the nearest even integer."""
    return (int(v) // 2) * 2


def sample_topleft(
    rng: np.random.Generator,
    cap_w: int,
    cap_h: int,
    P: int,
    edge_aware: bool,
) -> tuple[int, int]:
    """
    Sample one patch top-left ``(x0, y0)`` (column, row), divisible by 2.

    * edge-aware: draw from ``[-P/2, dim + P/2)`` then clamp into ``[0, dim-P]``.
    * uniform:    draw directly from ``[0, dim-P]``.
    """
    max_x = cap_w - P
    max_y = cap_h - P
    if max_x < 0 or max_y < 0:
        raise ValueError(
            f"patch size P={P} exceeds captured image {cap_w}x{cap_h}"
        )

    if edge_aware:
        half = P / 2.0
        xr = rng.uniform(-half, cap_w + half)
        yr = rng.uniform(-half, cap_h + half)
        x0 = int(np.clip(round(xr), 0, max_x))
        y0 = int(np.clip(round(yr), 0, max_y))
    else:
        x0 = int(rng.integers(0, max_x + 1))
        y0 = int(rng.integers(0, max_y + 1))

    return _even(x0), _even(y0)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sample captured/reference patch pairs (patch pipeline substep 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--crop-json", required=True,
                   help="metadata JSON from crop_resize_reference.py, or a directory "
                        "containing such JSONs (every *.json with a 'cropped_reference' "
                        "key is processed)")
    p.add_argument("--captured", default=None,
                   help="captured image path (single-file mode only; "
                        "overrides the path in --crop-json)")
    p.add_argument("--cropped-reference", default=None,
                   help="resized reference image path (single-file mode only; "
                        "overrides the path in --crop-json)")
    p.add_argument("--patch-size", type=int, required=True, metavar="P",
                   help="captured patch size P (pixels); forced even for the Bayer grid")
    p.add_argument("--num-patches", type=int, required=True, metavar="K",
                   help="number of patches k to sample per crop JSON")
    p.add_argument("--uniform", action="store_true",
                   help="use plain uniform sampling instead of edge-aware sampling")
    p.add_argument("--seed", type=int, default=0,
                   help="random seed for reproducible sampling")
    p.add_argument("--output-dir", default=None,
                   help="output root directory.  If omitted, output is written next to "
                        "the input: a '<crop_stem>_patches' directory beside a single "
                        "crop JSON, or inside the input directory (one subdir per image).")
    p.add_argument("--format", choices=("png", "npy", "exr"), default="png",
                   help="patch image format")
    return p.parse_args()


def _is_crop_json(path: Path) -> bool:
    """Return True if *path* looks like a crop metadata JSON."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return "cropped_reference" in d and "captured" in d
    except Exception:
        return False


def process_one_crop(
    crop_json_path: Path,
    cap_override: "Path | None",
    cref_override: "Path | None",
    P: int,
    k: int,
    edge_aware: bool,
    seed: int,
    out_dir: Path,
    fmt: str,
) -> int:
    """Sample patches from one (captured, cropped-reference) pair.  Returns 0 on success."""
    print(f"\n{'='*60}")
    print(f"crop-json: {crop_json_path}")

    crop = json.loads(crop_json_path.read_text(encoding="utf-8"))
    S = float(crop["scale"])
    pr = float(crop["padding_radius"])
    cap_w = int(crop["captured"]["width"])
    cap_h = int(crop["captured"]["height"])

    cap_path = cap_override or Path(crop["captured"]["path"])
    cref_path = cref_override or Path(crop["cropped_reference"]["path"])
    for tag, p in (("captured", cap_path), ("cropped-reference", cref_path)):
        if not p.is_file():
            print(f"error: {tag} image not found: {p}", file=sys.stderr)
            return 1

    ref_patch_size = int(round(S * (P + 2 * pr)))

    print(f"loading captured : {cap_path}")
    cap_arr = load_any(cap_path)
    if (cap_arr.shape[1], cap_arr.shape[0]) != (cap_w, cap_h):
        print(f"[warn] captured image is {cap_arr.shape[1]}x{cap_arr.shape[0]} "
              f"but crop-json says {cap_w}x{cap_h}; using actual size", file=sys.stderr)
        cap_h, cap_w = cap_arr.shape[:2]

    print(f"loading cropped reference : {cref_path}")
    cref_arr = load_any(cref_path)
    cref_h, cref_w = cref_arr.shape[:2]
    print(f"  captured {cap_w}x{cap_h}, cropped-ref {cref_w}x{cref_h}, "
          f"S={S}, p_r={pr}, P={P}, ref_patch={ref_patch_size}")

    cap_dir = out_dir / "cap"
    ref_dir = out_dir / "ref"
    cap_dir.mkdir(parents=True, exist_ok=True)
    ref_dir.mkdir(parents=True, exist_ok=True)
    ext = {"png": ".png", "npy": ".npy", "exr": ".exr"}[fmt]

    rng = np.random.default_rng(seed)
    print(f"sampling {k} patches ({'edge-aware' if edge_aware else 'uniform'}, seed={seed})")

    records = []
    for i in range(k):
        x0, y0 = sample_topleft(rng, cap_w, cap_h, P, edge_aware)
        cap_patch = cap_arr[y0:y0 + P, x0:x0 + P, :]

        rx0 = min(int(round(S * x0)), cref_w - ref_patch_size)
        ry0 = min(int(round(S * y0)), cref_h - ref_patch_size)
        ref_patch_arr = cref_arr[ry0:ry0 + ref_patch_size, rx0:rx0 + ref_patch_size, :]

        cap_name = f"{i:06d}_cap{ext}"
        ref_name = f"{i:06d}_ref{ext}"
        save_patch(cap_patch, cap_dir / cap_name, fmt)
        save_patch(ref_patch_arr, ref_dir / ref_name, fmt)

        records.append({
            "index": i,
            "captured_patch": {
                "file": str(cap_dir / cap_name),
                "x0": x0, "y0": y0, "width": P, "height": P,
            },
            "reference_patch": {
                "file": str(ref_dir / ref_name),
                "x0": rx0, "y0": ry0, "width": ref_patch_size, "height": ref_patch_size,
                "inset": int(round(S * pr)),
            },
        })

    meta = {
        "reference": crop["reference"],
        "captured":  {"path": str(cap_path), "width": cap_w, "height": cap_h},
        "cropped_reference": {"path": str(cref_path), "width": cref_w, "height": cref_h},
        "crop_json": str(crop_json_path),
        "scale": S,
        "padding_radius": pr,
        "patch_size": P,
        "reference_patch_size": ref_patch_size,
        "num_patches": k,
        "sampling": "uniform" if not edge_aware else "edge-aware",
        "seed": seed,
        "format": fmt,
        "patches": records,
    }
    meta_path = out_dir / "patches.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {k} patch pairs → {out_dir}")
    return 0


def main() -> int:
    args = parse_args()

    P = args.patch_size
    if P % 2 != 0:
        print(f"[warn] patch size {P} is odd; rounding down to {P - 1} for the Bayer grid",
              file=sys.stderr)
        P -= 1
    k = args.num_patches
    if k <= 0:
        print("error: --num-patches must be positive", file=sys.stderr)
        return 2

    edge_aware = not args.uniform
    crop_input = Path(args.crop_json).resolve()

    # ---- single-file mode ----
    if crop_input.is_file():
        cap_override  = Path(args.captured).resolve()          if args.captured          else None
        cref_override = Path(args.cropped_reference).resolve() if args.cropped_reference else None
        # Default: a "<crop_stem>_patches" directory next to the crop JSON, not cwd.
        stem = crop_input.stem
        if stem.endswith(".ref_crop"):
            stem = stem[: -len(".ref_crop")]
        out_dir = (
            Path(args.output_dir).resolve() if args.output_dir
            else crop_input.parent / f"{stem}_patches"
        )
        return process_one_crop(
            crop_input, cap_override, cref_override,
            P, k, edge_aware, args.seed, out_dir, args.format,
        )

    # ---- directory mode ----
    if not crop_input.is_dir():
        print(f"error: not a file or directory: {crop_input}", file=sys.stderr)
        return 2

    if args.captured or args.cropped_reference:
        print("[warn] --captured / --cropped-reference are ignored in directory mode",
              file=sys.stderr)

    crop_files = sorted(
        f for f in crop_input.iterdir()
        if f.is_file() and f.suffix.lower() == ".json" and _is_crop_json(f)
    )
    if not crop_files:
        print(f"error: no crop metadata JSONs found in {crop_input}", file=sys.stderr)
        return 2

    out_base = Path(args.output_dir).resolve() if args.output_dir else crop_input
    print(f"found {len(crop_files)} crop JSON(s) to process")

    errors = 0
    for cf in crop_files:
        # Each crop JSON gets its own subdirectory named after its captured stem.
        try:
            cap_stem = Path(json.loads(cf.read_text())["captured"]["path"]).stem
        except Exception:
            cap_stem = cf.stem
        out_dir = out_base / cap_stem
        rc = process_one_crop(cf, None, None, P, k, edge_aware, args.seed, out_dir, args.format)
        if rc != 0:
            errors += 1

    print(f"\n{'='*60}")
    print(f"done: {len(crop_files) - errors}/{len(crop_files)} succeeded")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
