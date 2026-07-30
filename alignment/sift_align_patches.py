#!/usr/bin/env python3
r"""
sift_align_patches.py  —  Substep 3 of the patch-pair dataset pipeline.

Each captured/reference patch pair produced by ``sample_patches.py`` (substep 2)
is aligned with a SIFT feature-matching method, and the reference patch is warped
onto the captured patch using the estimated transform.

Why a per-patch refinement?
---------------------------
The coarse (and optimised) global alignment leaves a residual error of a few
pixels.  Substep 2 keeps a ``p_r``-pixel border around every reference patch to
absorb that error; this substep removes it, pixel-accurately, by matching SIFT
keypoints between the two patches and warping the reference into the captured
frame.

Resolution handling
--------------------
The captured patch is ``P x P`` and the reference patch is
``S*(P+2*p_r) x S*(P+2*p_r)``.  For matching, the captured patch is upscaled by
``S`` so both patches live at reference resolution.  SIFT keypoints are matched
(mutual nearest neighbour + Lowe ratio test), and a geometric model is fit with
RANSAC:

    --model translation   pure x/y shift (mean of inlier matches)
    --model affine         partial-affine: rotation + uniform scale + shift
    --model homography     full projective transform (default)

The estimated transform maps reference-patch coordinates to (upscaled) captured
coordinates, so warping the reference patch with it yields an ``S*P x S*P`` image
aligned to the captured patch.  If matching fails (fewer than ``--min-inliers``
RANSAC inliers), the patch is **rejected**: no aligned image is written and the
pair is recorded in the output JSON with ``"rejected": true``.

Comparison visualisation
-------------------------
``--compare-before DIR`` and ``--compare-after DIR`` save side-by-side blend
images for every accepted patch — before alignment (nominal centre crop) and
after (SIFT-warped reference) respectively.  Both images are blended with the
upscaled captured patch using the same blend modes as ``optimize_alignment.py``:

    normal   overlay the reference at --compare-opacity over the captured patch.
    gcap     replace the green channel of the captured patch with the reference
             green — colour fringing reveals any residual misalignment.

Directory input
---------------
``--metadata`` may be a directory; every ``patches.json`` found directly inside
it (or one level deeper) is processed in turn.  ``--output-dir``, if given,
mirrors the relative directory structure.

Usage
-----
    python3 sift_align_patches.py \
        --metadata patches/patches.json \
        [--model homography|affine|translation] \
        [--ratio 0.75] [--min-inliers 8] \
        [--output-dir patches/warped] \
        [--format png|npy|exr] \
        [--compare-before DIR] [--compare-after DIR] \
        [--compare-blend normal gcap] \
        [--compare-opacity 0.5] \
        [--compare-size 0]

Requirements
------------
    Python 3.9+, numpy, opencv-python (cv2) with SIFT, and Pillow or OpenImageIO.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------

def load_any(path: Path) -> np.ndarray:
    """Load an image (.npy / OpenImageIO / Pillow) as (H, W, C) float32."""
    if path.suffix.lower() == ".npy":
        arr = np.load(str(path)).astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, None]
        return arr
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
# Grayscale uint8 conversion for SIFT
# ---------------------------------------------------------------------------

def to_gray_u8(arr: np.ndarray) -> np.ndarray:
    """
    (H, W, C) float32 -> (H, W) uint8 grayscale for SIFT.

    A percentile stretch is applied so HDR reference patches still yield
    usable contrast for keypoint detection.
    """
    if arr.ndim == 3 and arr.shape[2] >= 3:
        g = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
    elif arr.ndim == 3:
        g = arr[:, :, 0]
    else:
        g = arr.astype(np.float32)

    lo, hi = np.percentile(g, (1.0, 99.0))
    if hi <= lo:
        lo, hi = float(g.min()), float(g.max())
    if hi <= lo:
        return np.zeros(g.shape, dtype=np.uint8)
    g = np.clip((g - lo) / (hi - lo), 0.0, 1.0)
    return (g * 255.0 + 0.5).astype(np.uint8)


# ---------------------------------------------------------------------------
# SIFT matching + transform estimation
# ---------------------------------------------------------------------------

_MIN_MATCHES = {"translation": 2, "affine": 3, "homography": 4}


def match_keypoints(
    ref_gray: np.ndarray,
    cap_gray: np.ndarray,
    ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return matched keypoint arrays (ref_pts, cap_pts), each (N, 2) float32.

    Keypoints are detected with SIFT and matched with a brute-force KNN matcher
    followed by Lowe's ratio test.
    """
    sift = cv2.SIFT_create()
    kp_r, des_r = sift.detectAndCompute(ref_gray, None)
    kp_c, des_c = sift.detectAndCompute(cap_gray, None)
    if des_r is None or des_c is None or len(kp_r) < 2 or len(kp_c) < 2:
        return np.empty((0, 2), np.float32), np.empty((0, 2), np.float32)

    bf = cv2.BFMatcher(cv2.NORM_L2)
    knn = bf.knnMatch(des_r, des_c, k=2)

    ref_pts, cap_pts = [], []
    for pair in knn:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < ratio * n.distance:
            ref_pts.append(kp_r[m.queryIdx].pt)
            cap_pts.append(kp_c[m.trainIdx].pt)

    return (np.asarray(ref_pts, np.float32).reshape(-1, 2),
            np.asarray(cap_pts, np.float32).reshape(-1, 2))


def estimate_transform(
    ref_pts: np.ndarray,
    cap_pts: np.ndarray,
    model: str,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Estimate a 3x3 transform (ref -> cap) plus an inlier mask.

    Returns (H_3x3, inlier_mask) or (None, None) on failure.
    """
    n = len(ref_pts)
    if n < _MIN_MATCHES[model]:
        return None, None

    if model == "translation":
        shifts = cap_pts - ref_pts
        t = np.median(shifts, axis=0)
        resid = np.linalg.norm(shifts - t, axis=1)
        mask = (resid < 3.0).astype(np.uint8).reshape(-1, 1)
        if int(mask.sum()) >= _MIN_MATCHES[model]:
            t = np.median(shifts[mask.ravel() > 0], axis=0)
        H = np.array([[1.0, 0.0, t[0]],
                      [0.0, 1.0, t[1]],
                      [0.0, 0.0, 1.0]], dtype=np.float64)
        return H, mask

    if model == "affine":
        M, mask = cv2.estimateAffinePartial2D(
            ref_pts, cap_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
        )
        if M is None:
            return None, None
        H = np.vstack([M, [0.0, 0.0, 1.0]])
        return H, mask

    # homography
    H, mask = cv2.findHomography(ref_pts, cap_pts, cv2.RANSAC, 3.0)
    if H is None:
        return None, None
    return H, mask


def warp_reference(ref_arr: np.ndarray, H: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Warp reference patch (ref -> cap frame) to (out_h, out_w) via H."""
    warped = cv2.warpPerspective(
        ref_arr.astype(np.float32), H.astype(np.float32), (out_w, out_h),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )
    if warped.ndim == 2:
        warped = warped[:, :, None]
    return warped


# ---------------------------------------------------------------------------
# Comparison visualisation
# ---------------------------------------------------------------------------

def _to_rgb3(arr: np.ndarray) -> np.ndarray:
    """Ensure (H, W, 3) float32 clipped to [0, 1]."""
    if arr.ndim == 2:
        arr = arr[:, :, None]
    c = arr.shape[2]
    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr[:, :, :3], 0.0, 1.0)


def _resize_sq(arr: np.ndarray, size: int) -> np.ndarray:
    """Resize (H, W, 3) float32 to (size, size, 3) with bilinear interpolation."""
    from PIL import Image as PILImage  # type: ignore[import]
    if arr.shape[0] == size and arr.shape[1] == size:
        return arr
    arr8 = (arr * 255.0 + 0.5).astype(np.uint8)
    img = PILImage.fromarray(arr8, "RGB").resize((size, size), PILImage.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.float32) / 255.0


def save_patch_comparison(
    cap_arr: np.ndarray,
    ref_arr: np.ndarray,
    out_path: Path,
    blend_modes: list[str],
    opacity: float,
    max_size: int,
) -> None:
    """
    Blend *ref_arr* over *cap_arr* and save one PNG per blend mode.

    Both patches are brought to the same square display size: the larger of
    the two patch heights, capped at *max_size* (0 = no cap).  Blend modes:

    normal  overlay ref at *opacity* over cap.
    gcap    replace cap's green channel with ref's green — fringing reveals
            residual misalignment.

    When multiple blend modes are requested the mode name is appended to the
    output file stem, e.g. ``000000_before_normal.png``.
    """
    from PIL import Image as PILImage  # type: ignore[import]

    cap3 = _to_rgb3(cap_arr)
    ref3 = _to_rgb3(ref_arr)

    disp = max(cap3.shape[0], ref3.shape[0])
    if max_size > 0:
        disp = min(disp, max_size)

    cap3 = _resize_sq(cap3, disp)
    ref3 = _resize_sq(ref3, disp)

    for mode in blend_modes:
        if mode == "normal":
            blended = cap3 * (1.0 - opacity) + ref3 * opacity
        elif mode == "gcap":
            blended = cap3.copy()
            blended[:, :, 1] = ref3[:, :, 1]
        else:
            print(f"[warn] unknown compare blend mode '{mode}'", file=sys.stderr)
            continue

        img8 = (np.clip(blended, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        p = (
            out_path if len(blend_modes) == 1
            else out_path.with_stem(out_path.stem + f"_{mode}")
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(img8, "RGB").save(str(p))


# ---------------------------------------------------------------------------
# Core processing for a single patches.json
# ---------------------------------------------------------------------------

def process_metadata(
    meta_path: Path,
    args: argparse.Namespace,
    out_dir: Path,
    cmp_before_dir: Path | None,
    cmp_after_dir: Path | None,
) -> int:
    """Process one patches.json.  Returns 0 on success."""
    if not meta_path.is_file():
        print(f"error: metadata not found: {meta_path}", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    S = float(meta["scale"])
    pr = float(meta["padding_radius"])
    P = int(meta["patch_size"])
    inset = int(round(S * pr))          # nominal border in reference-patch pixels
    out_w = out_h = int(round(S * P))   # aligned reference output size

    out_dir.mkdir(parents=True, exist_ok=True)
    if cmp_before_dir:
        cmp_before_dir.mkdir(parents=True, exist_ok=True)
    if cmp_after_dir:
        cmp_after_dir.mkdir(parents=True, exist_ok=True)
    ext = {"png": ".png", "npy": ".npy", "exr": ".exr"}[args.format]

    print(f"\n{'='*60}")
    print(f"metadata: {meta_path}")
    print(f"model={args.model}  ratio={args.ratio}  min_inliers={args.min_inliers}")
    print(f"output size per patch: {out_w}x{out_h}  (S={S}, P={P}, inset={inset})")

    records = []
    n_ok = n_rejected = 0

    for rec in meta["patches"]:
        idx = rec["index"]
        cap_file = Path(rec["captured_patch"]["file"])
        ref_file = Path(rec["reference_patch"]["file"])

        cap_arr = load_any(cap_file)
        ref_arr = load_any(ref_file)

        # Upscale captured to reference resolution (S*P x S*P) for matching.
        if S != 1.0:
            cap_up = cv2.resize(cap_arr.astype(np.float32), (out_w, out_h),
                                interpolation=cv2.INTER_LINEAR)
            if cap_up.ndim == 2:
                cap_up = cap_up[:, :, None]
        else:
            cap_up = cap_arr

        # Nominal reference crop (centre S*P x S*P, no SIFT): used for
        # "before" comparison and as the fallback display.
        ref_nominal = ref_arr[inset:inset + out_h, inset:inset + out_w]

        ref_gray = to_gray_u8(ref_arr)
        cap_gray = to_gray_u8(cap_up)

        ref_pts, cap_pts = match_keypoints(ref_gray, cap_gray, args.ratio)
        H, mask = estimate_transform(ref_pts, cap_pts, args.model)
        inliers = int(mask.sum()) if mask is not None else 0

        if H is not None and inliers >= args.min_inliers:
            warped = warp_reference(ref_arr, H, out_w, out_h)
            out_name = f"{idx:06d}_ref_aligned{ext}"
            save_patch(warped, out_dir / out_name, args.format)

            # "before" comparison: nominal crop vs captured.
            if cmp_before_dir is not None:
                save_patch_comparison(
                    cap_up, ref_nominal,
                    cmp_before_dir / f"{idx:06d}_before.png",
                    args.compare_blend, args.compare_opacity, args.compare_size,
                )
            # "after" comparison: SIFT-warped reference vs captured.
            if cmp_after_dir is not None:
                save_patch_comparison(
                    cap_up, warped,
                    cmp_after_dir / f"{idx:06d}_after.png",
                    args.compare_blend, args.compare_opacity, args.compare_size,
                )

            records.append({
                "index": idx,
                "captured_patch": rec["captured_patch"],
                "reference_patch": rec["reference_patch"],
                "aligned_reference": {
                    "file": str(out_dir / out_name),
                    "width": out_w, "height": out_h,
                },
                "alignment": {
                    "model": args.model,
                    "num_matches": int(len(ref_pts)),
                    "num_inliers": inliers,
                    "rejected": False,
                    "transform": H.tolist(),
                },
            })
            n_ok += 1
            print(f"  patch {idx:06d}: matches={len(ref_pts):4d} inliers={inliers:4d}  [ok]")
        else:
            records.append({
                "index": idx,
                "captured_patch": rec["captured_patch"],
                "reference_patch": rec["reference_patch"],
                "aligned_reference": None,
                "alignment": {
                    "model": args.model,
                    "num_matches": int(len(ref_pts)),
                    "num_inliers": inliers,
                    "rejected": True,
                    "transform": None,
                },
            })
            n_rejected += 1
            print(f"  patch {idx:06d}: matches={len(ref_pts):4d} inliers={inliers:4d}  [rejected]")

    out_meta = {
        "reference": meta.get("reference"),
        "captured": meta.get("captured"),
        "scale": S,
        "padding_radius": pr,
        "patch_size": P,
        "aligned_patch_size": out_w,
        "model": args.model,
        "ratio": args.ratio,
        "min_inliers": args.min_inliers,
        "source_metadata": str(meta_path),
        "num_aligned": n_ok,
        "num_rejected": n_rejected,
        "format": args.format,
        "patches": records,
    }
    out_json = out_dir / "aligned.json"
    out_json.write_text(json.dumps(out_meta, indent=2), encoding="utf-8")
    print(f"done: {n_ok} aligned, {n_rejected} rejected (of {len(records)})")
    print(f"saved: {out_dir}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="SIFT-based per-patch alignment (patch pipeline substep 3).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--metadata", required=True,
                   help="patches.json from substep 2, or a directory containing "
                        "patches.json files (processed recursively one level deep)")
    p.add_argument("--model", choices=("translation", "affine", "homography"),
                   default="homography",
                   help="geometric model estimated from SIFT matches")
    p.add_argument("--ratio", type=float, default=0.75,
                   help="Lowe ratio-test threshold for keypoint matching")
    p.add_argument("--min-inliers", type=int, default=8,
                   help="minimum RANSAC inliers to accept the estimated transform; "
                        "pairs below this threshold are rejected (no output image)")
    p.add_argument("--output-dir", default=None,
                   help="root directory for warped patches "
                        "(default: <metadata_dir>/warped for file mode, "
                        "<metadata_dir> for directory mode)")
    p.add_argument("--format", choices=("png", "npy", "exr"), default="png",
                   help="warped patch image format")

    g = p.add_argument_group("comparison output")
    g.add_argument("--compare-before", default=None, metavar="DIR",
                   help="save pre-alignment (nominal crop) comparison images here")
    g.add_argument("--compare-after", default=None, metavar="DIR",
                   help="save post-alignment (SIFT-warped) comparison images here")
    g.add_argument("--compare-blend", nargs="+", default=["normal"],
                   choices=["normal", "gcap"], metavar="MODE",
                   help="blend mode(s) for comparison: normal gcap (default: normal)")
    g.add_argument("--compare-opacity", type=float, default=0.5,
                   help="overlay opacity for the normal blend mode")
    g.add_argument("--compare-size", type=int, default=0,
                   help="max side length of comparison images in pixels "
                        "(0 = use patch size)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    meta_input = Path(args.metadata).resolve()

    # ---- single-file mode ----
    if meta_input.is_file():
        out_dir = (
            Path(args.output_dir).resolve() if args.output_dir
            else meta_input.parent / "warped"
        )
        cmp_before = Path(args.compare_before).resolve() if args.compare_before else None
        cmp_after  = Path(args.compare_after).resolve()  if args.compare_after  else None
        return process_metadata(meta_input, args, out_dir, cmp_before, cmp_after)

    # ---- directory mode ----
    if not meta_input.is_dir():
        print(f"error: not a file or directory: {meta_input}", file=sys.stderr)
        return 2

    # Find all patches.json files directly inside or one level deep.
    candidates = sorted(meta_input.glob("patches.json"))
    candidates += sorted(
        f for f in meta_input.glob("*/patches.json")
        if f.parent != meta_input
    )
    if not candidates:
        print(f"error: no patches.json found under {meta_input}", file=sys.stderr)
        return 2

    print(f"found {len(candidates)} patches.json file(s) to process")

    out_base = Path(args.output_dir).resolve() if args.output_dir else meta_input
    cmp_before_base = Path(args.compare_before).resolve() if args.compare_before else None
    cmp_after_base  = Path(args.compare_after).resolve()  if args.compare_after  else None

    errors = 0
    for mf in candidates:
        # Mirror the relative path under the output base.
        rel = mf.parent.relative_to(meta_input)
        out_dir   = out_base / rel / "warped"
        cmp_before = (cmp_before_base / rel) if cmp_before_base else None
        cmp_after  = (cmp_after_base  / rel) if cmp_after_base  else None
        rc = process_metadata(mf, args, out_dir, cmp_before, cmp_after)
        if rc != 0:
            errors += 1

    total = len(candidates)
    print(f"\n{'='*60}")
    print(f"directory done: {total - errors}/{total} metadata files succeeded")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
