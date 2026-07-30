#!/usr/bin/env python3
r"""
crop_resize_reference.py  —  Substep 1 of the patch-pair dataset pipeline.

Takes the alignment JSON produced by ``align_quad.py`` / ``optimize_alignment.py``
(the input structure is identical to that script's input *and* output) and warps
the corresponding region of the reference image into an *extended* captured-image
frame.

The output image has size

    ( S * (H_c + 2*p_r),  S * (W_c + 2*p_r) )

where
  * ``H_c, W_c`` are the captured-image height and width,
  * ``S``        is the scaling factor (``S = 1`` gives a pure ISP pipeline,
                 ``S > 1`` leaves room for a super-resolution pipeline),
  * ``p_r``      is the padding radius in *captured* pixels — extra pixels
                 cropped from each side of the reference to absorb the small
                 shifts of an imperfect coarse alignment (used by the next
                 substep, ``sample_patches.py``).

Mapping
-------
For every output pixel ``(out_row, out_col)`` the corresponding position in the
captured frame is

    cap_x = out_col / S - p_r          (captured column, may be negative)
    cap_y = out_row / S - p_r          (captured row,    may be negative)

Its normalised captured coordinate ``(cap_x / W_c, cap_y / H_c)`` is mapped
through the alignment homography ``H`` (built from the four quad vertices, exactly
as in ``optimize_alignment.py``) into normalised reference space, and the
full-resolution reference is sampled there with bilinear interpolation.

When the mapped coordinate falls outside the reference image, the nearest edge
pixel is replicated ("repeated padding"), so the border is always filled even if
the reference does not extend far enough.

Usage
-----
    python3 crop_resize_reference.py \
        --alignment path/to/cap.align.opt.json \
        [--reference ref.exr]      \  # overrides the path stored in the JSON
        [--captured  cap.bmp]      \  # overrides the path stored in the JSON
        [--scale 1]                \  # S
        [--padding-radius 0]       \  # p_r
        [--output cap.ref_crop.png]\
        [--format png|npy|exr]     \
        [--device cpu|cuda]

``--alignment`` may instead be a *directory* of alignment JSONs (as produced by
``optimize_alignment.py`` in its directory mode); every ``*.json`` with a
``vertices`` key is processed and ``--output`` is then treated as an output
directory.  Output file names follow ``<captured_stem>.ref_crop.<ext>``.

A companion metadata JSON is written next to each output image (same stem,
``.json`` extension) recording every parameter the next substep needs.

Requirements
------------
    Python 3.9+, numpy, torch, and Pillow or OpenImageIO.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Re-use the image-loading and homography helpers from optimize_alignment.py
# (same directory).  Its top-level imports are only numpy/torch, so this is
# cheap and side-effect free.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from optimize_alignment import (  # noqa: E402
    load_image,
    to_tensor,
    compute_homography,
    SRC_CORNERS,
)

_V_KEYS = ("tl", "tr", "br", "bl")


# ---------------------------------------------------------------------------
# Core warp into the extended captured frame
# ---------------------------------------------------------------------------

def warp_reference_extended(
    ref: torch.Tensor,
    H: torch.Tensor,
    out_h: int,
    out_w: int,
    cap_h: int,
    cap_w: int,
    pr: float,
    S: float,
) -> torch.Tensor:
    """
    Warp the reference image into the extended captured frame.

    Parameters
    ----------
    ref   : (1, C, ref_H, ref_W) tensor — full-resolution reference.
    H     : (3, 3) homography, normalised captured -> normalised reference.
    out_h : S * (cap_h + 2*pr)   (output rows).
    out_w : S * (cap_w + 2*pr)   (output cols).
    cap_h : captured image height.
    cap_w : captured image width.
    pr    : padding radius in captured pixels.
    S     : scale factor.

    Returns
    -------
    (1, C, out_h, out_w) tensor.
    """
    device = ref.device

    rows = torch.arange(out_h, dtype=torch.float32, device=device)
    cols = torch.arange(out_w, dtype=torch.float32, device=device)

    # Output pixel -> captured continuous position -> normalised captured coord.
    u_cap = (cols / S - pr) / cap_w          # (out_w,)
    v_cap = (rows / S - pr) / cap_h          # (out_h,)
    vv, uu = torch.meshgrid(v_cap, u_cap, indexing="ij")   # (out_h, out_w)
    ones = torch.ones_like(uu)

    pts = torch.stack([uu.reshape(-1), vv.reshape(-1), ones.reshape(-1)], dim=0)  # (3, N)
    mapped = H @ pts                          # (3, N)
    denom = mapped[2].clamp(min=1e-8)
    xr = mapped[0] / denom                    # normalised ref x
    yr = mapped[1] / denom                    # normalised ref y

    grid = torch.stack(
        [(2.0 * xr - 1.0).reshape(1, out_h, out_w),
         (2.0 * yr - 1.0).reshape(1, out_h, out_w)],
        dim=-1,
    )                                          # (1, out_h, out_w, 2)

    return F.grid_sample(
        ref, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


# ---------------------------------------------------------------------------
# Output saving
# ---------------------------------------------------------------------------

def save_image(arr: np.ndarray, path: Path, fmt: str) -> None:
    """Save a (H, W, C) float32 array (values in [0, 1] for png) in *fmt*."""
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
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Crop and resize the reference into the captured frame (patch pipeline substep 1).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--alignment", required=True,
                   help="alignment JSON produced by align_quad.py / optimize_alignment.py, "
                        "or a directory of such JSONs (processed one level deep)")
    p.add_argument("--reference", default=None,
                   help="reference image path (single-file mode only; "
                        "overrides the path recorded in the JSON)")
    p.add_argument("--captured", default=None,
                   help="captured image path (single-file mode only; "
                        "overrides the path recorded in the JSON)")
    p.add_argument("--scale", type=float, default=1.0, metavar="S",
                   help="scaling factor S (1 = pure ISP, >1 = super-resolution)")
    p.add_argument("--padding-radius", type=float, default=0.0, metavar="P_R",
                   help="extra border pixels p_r (in captured-frame pixels)")
    p.add_argument("--output", default=None,
                   help="output image path (single-file mode) or output directory "
                        "(directory mode).  If omitted, output is written next to the "
                        "input: <captured_stem>.ref_crop.<ext> beside the JSON, or a "
                        "'<dirname>_ref_crop' directory beside an input directory.")
    p.add_argument("--format", choices=("png", "npy", "exr"), default="png",
                   help="output image format")
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available, else cpu)")
    return p.parse_args()


def load_vertices(align_data: dict, ref_w: int, ref_h: int) -> list[list[float]]:
    """Return the four quad vertices as normalised [x, y] pairs (TL, TR, BR, BL)."""
    vd = align_data.get("vertices", {})
    if not all(k in vd for k in _V_KEYS):
        raise ValueError(f"alignment JSON missing vertex keys {_V_KEYS}")
    try:
        return [[vd[k]["x_rel"], vd[k]["y_rel"]] for k in _V_KEYS]
    except KeyError:
        # Fall back to absolute coords normalised by JSON-reported reference dims.
        jrw = align_data.get("reference", {}).get("width", ref_w)
        jrh = align_data.get("reference", {}).get("height", ref_h)
        return [[vd[k]["x_abs"] / jrw, vd[k]["y_abs"] / jrh] for k in _V_KEYS]


def process_one(
    align_path: Path,
    ref_override: "Path | None",
    cap_override: "Path | None",
    S: float,
    pr: float,
    out_img: Path,
    fmt: str,
    device: "torch.device",
) -> int:
    """Warp and save one (alignment JSON -> cropped reference) pair.  Returns 0 on success."""
    print(f"\n{'='*60}")
    print(f"alignment: {align_path}")

    align_data = json.loads(align_path.read_text(encoding="utf-8"))

    ref_path = ref_override or Path(align_data.get("reference", {}).get("path", ""))
    cap_path = cap_override or Path(align_data.get("captured",  {}).get("path", ""))
    for tag, p in (("reference", ref_path), ("captured", cap_path)):
        if not p.is_file():
            print(f"error: {tag} image not found: {p}\n"
                  f"       (pass --{tag} to override the path stored in the JSON)",
                  file=sys.stderr)
            return 1

    print(f"loading reference : {ref_path}")
    ref_arr = load_image(ref_path)
    ref_h, ref_w = ref_arr.shape[:2]
    print(f"  {ref_w}x{ref_h} x{ref_arr.shape[2]}ch")

    print(f"reading captured dims : {cap_path}")
    cap_arr = load_image(cap_path)
    cap_h, cap_w = cap_arr.shape[:2]
    del cap_arr
    print(f"  {cap_w}x{cap_h}")

    try:
        dst_coords = load_vertices(align_data, ref_w, ref_h)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    src = SRC_CORNERS.to(device)
    dst = torch.tensor(dst_coords, dtype=torch.float32, device=device)
    H = compute_homography(src, dst)

    out_h = int(round(S * (cap_h + 2 * pr)))
    out_w = int(round(S * (cap_w + 2 * pr)))
    print(f"output size: {out_w}x{out_h}   (S={S}, p_r={pr})")

    ref_t = to_tensor(ref_arr, device)
    with torch.no_grad():
        warped = warp_reference_extended(ref_t, H, out_h, out_w, cap_h, cap_w, pr, S)
    result = warped[0].permute(1, 2, 0).contiguous().cpu().numpy()

    ext = {"png": ".png", "npy": ".npy", "exr": ".exr"}[fmt]
    if out_img.suffix.lower() != ext:
        out_img = out_img.with_suffix(ext)

    print(f"saving: {out_img}")
    save_image(result, out_img, fmt)

    meta = {
        "reference": {"path": str(ref_path), "width": ref_w, "height": ref_h},
        "captured":  {"path": str(cap_path), "width": cap_w, "height": cap_h},
        "cropped_reference": {
            "path": str(out_img),
            "width": out_w,
            "height": out_h,
            "format": fmt,
        },
        "alignment_json": str(align_path),
        "scale": S,
        "padding_radius": pr,
    }
    meta_json = out_img.with_suffix(".json")
    meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved metadata: {meta_json}")
    return 0


def _is_alignment_json(path: Path) -> bool:
    """Return True if *path* looks like an alignment JSON (has a 'vertices' key)."""
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
        return "vertices" in d
    except Exception:
        return False


def main() -> int:
    args = parse_args()

    align_input = Path(args.alignment).resolve()

    S = float(args.scale)
    pr = float(args.padding_radius)
    if S <= 0:
        print("error: --scale must be positive", file=sys.stderr)
        return 2
    if pr < 0:
        print("error: --padding-radius must be non-negative", file=sys.stderr)
        return 2

    device = (
        torch.device(args.device) if args.device
        else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device: {device}")

    ext = {"png": ".png", "npy": ".npy", "exr": ".exr"}[args.format]

    # ---- single-file mode ----
    if align_input.is_file():
        ref_override = Path(args.reference).resolve() if args.reference else None
        cap_override = Path(args.captured).resolve()  if args.captured  else None
        cap_stem = Path(json.loads(align_input.read_text())["captured"]["path"]).stem
        default_name = f"{cap_stem}.ref_crop{ext}"
        if args.output:
            out_arg = Path(args.output)
            # An existing directory, or a path clearly meant as one, receives the
            # default-named file; otherwise the path is used as the file itself.
            if out_arg.is_dir() or args.output.endswith(("/", os.sep)):
                out_img = out_arg.resolve() / default_name
            else:
                out_img = out_arg.resolve()
        else:
            # Default: write next to the input alignment JSON, not cwd.
            out_img = align_input.parent / default_name
        return process_one(align_input, ref_override, cap_override, S, pr, out_img, args.format, device)

    # ---- directory mode ----
    if not align_input.is_dir():
        print(f"error: not a file or directory: {align_input}", file=sys.stderr)
        return 2

    if args.reference or args.captured:
        print("[warn] --reference / --captured are ignored in directory mode", file=sys.stderr)

    align_files = sorted(
        f for f in align_input.iterdir()
        if f.is_file() and f.suffix.lower() == ".json" and _is_alignment_json(f)
    )
    if not align_files:
        print(f"error: no alignment JSONs found in {align_input}", file=sys.stderr)
        return 2

    # Default: a sibling directory named after the input, not cwd.
    out_dir = (
        Path(args.output).resolve() if args.output
        else align_input.parent / f"{align_input.name}_ref_crop"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"found {len(align_files)} alignment JSON(s) to process")
    print(f"output directory: {out_dir}")

    errors = 0
    for af in align_files:
        # Derive captured stem from the JSON so the output file name is predictable.
        try:
            cap_stem = Path(json.loads(af.read_text())["captured"]["path"]).stem
        except Exception:
            cap_stem = af.stem
        out_img = out_dir / f"{cap_stem}.ref_crop{ext}"
        rc = process_one(af, None, None, S, pr, out_img, args.format, device)
        if rc != 0:
            errors += 1

    print(f"\n{'='*60}")
    print(f"done: {len(align_files) - errors}/{len(align_files)} succeeded")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
