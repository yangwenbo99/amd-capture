#!/usr/bin/env python3
"""
optimize_alignment.py  —  Gradient-based fine-tuning of a quad alignment.

Takes the same inputs as align_quad.py (reference image, captured image, and a
*mandatory* alignment JSON), then refines the four quadrilateral vertices with
gradient descent so the captured image best matches the reference.

Method
------
- The four vertices (in normalised reference coordinates) are the optimisation
  parameters, initialised from the input JSON's relative coordinates.
- A homography mapping the captured-image corners onto those vertices is built
  differentiably (torch.linalg.solve on the 8x8 DLT system).
- The reference image is warped into the captured-image frame with
  F.grid_sample, so gradients flow from the loss back to the vertices.
- Loss is piq.DISTS or piq.MultiScaleSSIMLoss (user's choice), optionally on a
  greyscale conversion of both images.

Usage
-----
    python3 optimize_alignment.py \
        --reference ref.exr \
        --captured  cap.bmp \
        --input-json cap.align.json \
        [--loss ms-ssim|dists] \
        [--greyscale] \
        [--iters 300] [--lr 0.005] [--optimizer adam|lbfgs] \
        [--max-size 512] [--output cap.align.opt.json]

Requirements
------------
    Python 3.9+, torch, piq, and Pillow or OpenImageIO (for image loading).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Image loading  ->  (H, W, C) float32 numpy array in [0, 1]
# ---------------------------------------------------------------------------

def load_image(path: Path) -> np.ndarray:
    # Prefer OpenImageIO (handles EXR / HDR / TIFF), fall back to Pillow.
    try:
        import OpenImageIO as oiio  # type: ignore[import]
        buf = oiio.ImageBuf(str(path))
        spec = buf.spec()
        if spec.width <= 0:
            raise RuntimeError(buf.geterror() or "empty image")
        px = buf.get_pixels(oiio.FLOAT)
        if px is None:
            raise RuntimeError("get_pixels returned None")
        px = np.asarray(px, dtype=np.float32)
        if px.ndim == 2:
            px = px[:, :, None]
        return np.clip(px[..., : min(px.shape[2], 3)], 0.0, 1.0)
    except Exception:
        pass
    from PIL import Image  # type: ignore[import]
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"), dtype=np.float32) / 255.0
    return arr


def to_tensor(arr: np.ndarray, device: torch.device) -> torch.Tensor:
    """(H, W, C) numpy  ->  (1, C, H, W) tensor."""
    t = torch.from_numpy(np.ascontiguousarray(arr)).permute(2, 0, 1).unsqueeze(0)
    return t.to(device=device, dtype=torch.float32)


def to_greyscale(t: torch.Tensor) -> torch.Tensor:
    """(1, C, H, W) -> (1, 1, H, W) using Rec.709 luma weights."""
    if t.shape[1] == 1:
        return t
    w = torch.tensor([0.2126, 0.7152, 0.0722], device=t.device).view(1, 3, 1, 1)
    return (t[:, :3] * w).sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Differentiable homography and warp
# ---------------------------------------------------------------------------

# Captured-image corners in normalised [0, 1] space, order TL, TR, BR, BL.
SRC_CORNERS = torch.tensor(
    [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]], dtype=torch.float32
)


def compute_homography(src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """
    Solve for the 3x3 homography H such that dst_i ~ H @ src_i (projectively).
    src, dst: (4, 2) tensors.  Returns a (3, 3) tensor (differentiable in dst).
    """
    rows = []
    rhs = []
    zero = src.new_zeros(())
    one = src.new_ones(())
    for i in range(4):
        sx, sy = src[i, 0], src[i, 1]
        dx, dy = dst[i, 0], dst[i, 1]
        rows.append(torch.stack([sx, sy, one, zero, zero, zero, -sx * dx, -sy * dx]))
        rows.append(torch.stack([zero, zero, zero, sx, sy, one, -sx * dy, -sy * dy]))
        rhs.append(dx)
        rhs.append(dy)
    A = torch.stack(rows)              # (8, 8)
    b = torch.stack(rhs)               # (8,)
    h = torch.linalg.solve(A, b)       # [h11..h32]
    H = torch.cat([h, one.view(1)]).reshape(3, 3)
    return H


def warp_reference(ref: torch.Tensor, H: torch.Tensor, out_h: int, out_w: int) -> torch.Tensor:
    """
    Warp the reference into the captured-image frame.

    For each output pixel (a normalised captured coordinate) we map through H to
    a normalised reference coordinate and sample the reference there.
    """
    device = ref.device
    ys = torch.linspace(0.0, 1.0, out_h, device=device)
    xs = torch.linspace(0.0, 1.0, out_w, device=device)
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    ones = torch.ones_like(gx)
    pts = torch.stack([gx.reshape(-1), gy.reshape(-1), ones.reshape(-1)], dim=0)  # (3, N)
    mapped = H @ pts                                          # (3, N)
    denom = mapped[2].clamp(min=1e-8)
    mx = mapped[0] / denom
    my = mapped[1] / denom
    grid = torch.stack(
        [(2.0 * mx - 1.0).reshape(1, out_h, out_w),
         (2.0 * my - 1.0).reshape(1, out_h, out_w)],
        dim=-1,
    )                                                         # (1, H, W, 2)
    return F.grid_sample(
        ref, grid, mode="bilinear", padding_mode="border", align_corners=True
    )


# ---------------------------------------------------------------------------
# Loss construction
# ---------------------------------------------------------------------------

def build_loss(name: str):
    import piq
    key = name.strip().lower().replace("_", "-")
    if key in ("ms-ssim", "msssim", "ms-ssim-loss", "mssim"):
        return "ms-ssim", piq.MultiScaleSSIMLoss(data_range=1.0)
    if key in ("dists",):
        return "dists", piq.DISTS()
    raise ValueError("loss must be one of: ms-ssim, dists")


def prep_for_loss(warped: torch.Tensor, target: torch.Tensor, loss_name: str):
    """DISTS requires 3 channels; replicate greyscale if needed."""
    w = warped.clamp(0.0, 1.0)
    t = target.clamp(0.0, 1.0)
    if loss_name == "dists" and w.shape[1] == 1:
        w = w.repeat(1, 3, 1, 1)
        t = t.repeat(1, 3, 1, 1)
    return w, t


# ---------------------------------------------------------------------------
# Comparison image rendering
# ---------------------------------------------------------------------------

def _to_rgb3(arr: np.ndarray) -> np.ndarray:
    """Ensure (H, W, 3) float32, clipped to [0, 1]."""
    if arr.ndim == 2:
        arr = arr[:, :, None]
    c = arr.shape[2]
    if c == 1:
        arr = np.repeat(arr, 3, axis=2)
    return np.clip(arr[:, :, :3], 0.0, 1.0)


def save_comparison(
    ref_arr: np.ndarray,
    cap_arr: np.ndarray,
    params: torch.Tensor,
    ref_w: int,
    ref_h: int,
    out_path: Path,
    blend_modes: list[str],
    opacity: float,
    device: torch.device,
    comp_max_size: int,
) -> None:
    """
    Warp the captured image into the reference frame and blend it with the
    reference, saving one PNG per blend mode.

    Warp direction: H^{-1} maps each output pixel (normalised reference space)
    to the corresponding position in normalised captured space, where the
    captured image is sampled.  Pixels that fall outside the captured frame
    (mask == 0) show the reference unchanged.

    Blend modes
    -----------
    normal  Overlay the warped capture at *opacity* over the reference.
    gcap    Replace the green channel of the reference with the captured
            green (magenta/cyan colour fringing reveals misalignment).

    When multiple blend modes are requested the mode name is appended to the
    output file stem: e.g. ``before_normal.png`` and ``before_gcap.png``.
    """
    from PIL import Image as PILImage  # type: ignore[import]

    comp_h, comp_w = compute_working_size(ref_h, ref_w, comp_max_size)

    with torch.no_grad():
        src = SRC_CORNERS.to(device)
        H = compute_homography(src, params.to(device))
        H_inv = torch.linalg.inv(H)          # normalised ref → normalised cap

        ys = torch.linspace(0.0, 1.0, comp_h, device=device)
        xs = torch.linspace(0.0, 1.0, comp_w, device=device)
        gy, gx = torch.meshgrid(ys, xs, indexing="ij")
        ones = torch.ones_like(gx)
        pts = torch.stack(
            [gx.reshape(-1), gy.reshape(-1), ones.reshape(-1)], dim=0
        )
        mapped = H_inv @ pts
        denom  = mapped[2].clamp(min=1e-8)
        cx = mapped[0] / denom
        cy = mapped[1] / denom

        # Mask: 1 where the ref pixel maps inside the captured frame.
        mask_np = (
            ((cx >= 0) & (cx <= 1) & (cy >= 0) & (cy <= 1))
            .float().reshape(comp_h, comp_w).cpu().numpy()
        )

        grid = torch.stack(
            [(2.0 * cx - 1.0).reshape(1, comp_h, comp_w),
             (2.0 * cy - 1.0).reshape(1, comp_h, comp_w)],
            dim=-1,
        )
        cap_t = to_tensor(cap_arr, device)
        warped_t = F.grid_sample(
            cap_t, grid, mode="bilinear",
            padding_mode="zeros", align_corners=True,
        )
        ref_t = resize_tensor(to_tensor(ref_arr, device), comp_h, comp_w)

    ref3 = _to_rgb3(ref_t[0].permute(1, 2, 0).cpu().numpy())
    cap3 = _to_rgb3(warped_t[0].permute(1, 2, 0).cpu().numpy())

    for mode in blend_modes:
        if mode == "normal":
            alpha   = mask_np[:, :, None] * opacity
            blended = ref3 * (1.0 - alpha) + cap3 * alpha
        elif mode == "gcap":
            blended = ref3.copy()
            where   = mask_np > 0.5
            blended[:, :, 1] = np.where(where, cap3[:, :, 1], ref3[:, :, 1])
        else:
            print(f"[warn] unknown blend mode '{mode}'", file=sys.stderr)
            continue

        img8 = (np.clip(blended, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
        p = (
            out_path if len(blend_modes) == 1
            else out_path.with_stem(out_path.stem + f"_{mode}")
        )
        p.parent.mkdir(parents=True, exist_ok=True)
        PILImage.fromarray(img8, "RGB").save(str(p))
        print(f"  comparison ({mode}): {p}  [{comp_w}x{comp_h}]")


# ---------------------------------------------------------------------------
# Sizing helpers
# ---------------------------------------------------------------------------

def compute_working_size(h: int, w: int, max_size: int) -> tuple[int, int]:
    """Scale down so the longer edge equals max_size (no up-scaling)."""
    if max(h, w) <= max_size:
        return h, w
    scale = max_size / max(h, w)
    return max(1, int(round(h * scale))), max(1, int(round(w * scale)))


def resize_tensor(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if t.shape[2] == h and t.shape[3] == w:
        return t
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)


# ---------------------------------------------------------------------------
# Output JSON
# ---------------------------------------------------------------------------

_V_KEYS = ("tl", "tr", "br", "bl")


def build_output(
    ref_path: Path, cap_path: Path,
    ref_w: int, ref_h: int, cap_w: int, cap_h: int,
    params: torch.Tensor,
    loss_name: str, greyscale: bool,
    iters: int, optimizer_name: str, lr: float,
    initial_loss: float, final_loss: float,
) -> dict:
    verts = {}
    for i, key in enumerate(_V_KEYS):
        xr = float(params[i, 0])
        yr = float(params[i, 1])
        verts[key] = {
            "x_abs": round(xr * ref_w, 4),
            "y_abs": round(yr * ref_h, 4),
            "x_rel": round(xr, 8),
            "y_rel": round(yr, 8),
        }
    return {
        "reference": {"path": str(ref_path), "width": ref_w, "height": ref_h},
        "captured":  {"path": str(cap_path), "width": cap_w, "height": cap_h},
        "vertices": verts,
        "optimization": {
            "method": "gradient-descent",
            "loss_function": loss_name,
            "greyscale": greyscale,
            "iterations": iters,
            "initial_loss": round(initial_loss, 8),
            "final_loss": round(final_loss, 8),
            "optimizer": optimizer_name,
            "lr": lr,
        },
    }


# ---------------------------------------------------------------------------
# Directory-mode helpers
# ---------------------------------------------------------------------------

_IMAGE_EXTENSIONS = {
    ".bmp", ".png", ".jpg", ".jpeg", ".tif", ".tiff",
    ".exr", ".hdr", ".ppm", ".pgm",
}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in _IMAGE_EXTENSIONS


def _extract_ref_stem(cap_path: Path) -> "str | None":
    """
    Return the reference stem encoded in a captured filename.

    Convention: cap-XXXX-YYY_<ref-stem>.<ext>
    The ref-stem is everything after the *last* underscore, without extension.
    Returns None if no underscore is present.
    """
    stem = cap_path.stem          # e.g. "cap-0001-002_0000000-032"
    idx = stem.rfind("_")
    if idx == -1:
        return None
    return stem[idx + 1:]         # e.g. "0000000-032"


def _find_reference_files(ref_dir: Path, ref_stem: str) -> list:
    """Return all image files in ref_dir whose stem matches ref_stem, sorted."""
    matches = [
        f for f in ref_dir.iterdir()
        if f.is_file() and _is_image(f) and f.stem == ref_stem
    ]
    return sorted(matches)


def _build_pairs(ref_dir: Path, cap_dir: Path) -> list:
    """
    Build the list of (reference, captured) path pairs.

    Each captured image encodes a reference stem in its filename
    (everything after the last '_', without extension).  All reference
    files with that stem (any extension) inside ref_dir are matched.
    Captured files are iterated in sorted order.
    """
    cap_files = sorted(f for f in cap_dir.iterdir() if f.is_file() and _is_image(f))
    pairs = []
    for cap in cap_files:
        ref_stem = _extract_ref_stem(cap)
        if ref_stem is None:
            print(
                f"[warn] cannot extract reference stem from '{cap.name}'; skipping",
                file=sys.stderr,
            )
            continue
        refs = _find_reference_files(ref_dir, ref_stem)
        if not refs:
            print(
                f"[warn] no reference file with stem '{ref_stem}' found for "
                f"'{cap.name}'; skipping",
                file=sys.stderr,
            )
            continue
        for ref in refs:
            pairs.append((ref, cap))
    return pairs


# ---------------------------------------------------------------------------
# Single-pair processing
# ---------------------------------------------------------------------------

def process_pair(
    ref_path: Path,
    cap_path: Path,
    json_data: dict,
    args: argparse.Namespace,
    device: "torch.device",
    loss_name: str,
    loss_fn,
    out_json: Path,
    compare_before: "Path | None",
    compare_after: "Path | None",
) -> int:
    """
    Optimise alignment for one (reference, captured) pair.
    Returns 0 on success, non-zero on error.
    """
    print(f"\n{'='*60}")
    print(f"pair: {cap_path.name}  <->  {ref_path.name}")

    # ---- load images ----
    print(f"loading reference : {ref_path}")
    ref_arr = load_image(ref_path)
    ref_h_full, ref_w_full = ref_arr.shape[:2]
    print(f"  {ref_w_full}x{ref_h_full} x{ref_arr.shape[2]}ch")

    print(f"loading captured  : {cap_path}")
    cap_arr = load_image(cap_path)
    cap_h_full, cap_w_full = cap_arr.shape[:2]
    print(f"  {cap_w_full}x{cap_h_full} x{cap_arr.shape[2]}ch")

    # ---- load vertices from JSON ----
    vd = json_data.get("vertices", {})
    if not all(k in vd for k in _V_KEYS):
        print(f"error: JSON missing vertex keys {_V_KEYS}", file=sys.stderr)
        return 2
    try:
        init_coords = [[vd[k]["x_rel"], vd[k]["y_rel"]] for k in _V_KEYS]
    except KeyError:
        # Fallback: absolute coords normalised by JSON-reported reference dims.
        jrw = json_data.get("reference", {}).get("width", ref_w_full)
        jrh = json_data.get("reference", {}).get("height", ref_h_full)
        init_coords = [
            [vd[k]["x_abs"] / jrw, vd[k]["y_abs"] / jrh] for k in _V_KEYS
        ]

    # ---- working resolution (output = captured frame) ----
    out_h, out_w = compute_working_size(cap_h_full, cap_w_full, args.max_size)
    print(f"working resolution: {out_w}x{out_h}  (max-size={args.max_size})")

    if loss_name == "ms-ssim":
        # MultiScaleSSIMLoss (5 scales, kernel 11) needs min dim >= 11*16 = 176.
        if min(out_h, out_w) < 176:
            print(
                f"error: working resolution {out_w}x{out_h} is too small for "
                "ms-ssim (minimum 176px per side). "
                "Increase --max-size or use --loss dists.",
                file=sys.stderr,
            )
            return 2

    # ---- build tensors ----
    ref_t = to_tensor(ref_arr, device)   # full res, used for sampling
    cap_t = to_tensor(cap_arr, device)
    cap_t = resize_tensor(cap_t, out_h, out_w)

    if args.greyscale:
        ref_t = to_greyscale(ref_t)
        cap_t = to_greyscale(cap_t)
        print("greyscale: yes")
    else:
        print("greyscale: no")

    # ---- optimisation parameters ----
    params = torch.tensor(
        init_coords, dtype=torch.float32, device=device, requires_grad=True
    )
    src_corners = SRC_CORNERS.to(device)

    lr = args.lr or (0.005 if args.optimizer == "adam" else 0.5)
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam([params], lr=lr)
    else:
        optimizer = torch.optim.LBFGS([params], lr=lr, max_iter=10,
                                       line_search_fn="strong_wolfe")
    print(f"optimizer: {args.optimizer}  lr={lr}  iters={args.iters}")

    # ---- compute initial loss ----
    with torch.no_grad():
        H0 = compute_homography(src_corners, params.detach())
        w0 = warp_reference(ref_t, H0, out_h, out_w)
        wl, tl = prep_for_loss(w0, cap_t, loss_name)
        initial_loss = loss_fn(wl, tl).item()
    print(f"initial loss: {initial_loss:.6f}")

    # Snapshot the starting vertices for the adjustment table and comparison.
    init_params_data = params.data.clone().cpu()

    comp_size = args.compare_size if args.compare_size > 0 else max(ref_h_full, ref_w_full)

    # ---- pre-optimisation comparison ----
    if compare_before is not None:
        print("rendering pre-optimisation comparison ...")
        save_comparison(
            ref_arr, cap_arr, init_params_data,
            ref_w_full, ref_h_full, compare_before,
            args.compare_blend, args.compare_opacity, device, comp_size,
        )

    best_loss = initial_loss
    best_params = params.data.clone()
    loss_val = initial_loss       # fallback if --iters 0

    # ---- main optimisation loop ----
    print("optimising ...")
    for i in range(args.iters):
        if args.optimizer == "lbfgs":
            def closure():
                optimizer.zero_grad()
                H = compute_homography(src_corners, params)
                warped = warp_reference(ref_t, H, out_h, out_w)
                wl, tl = prep_for_loss(warped, cap_t, loss_name)
                loss = loss_fn(wl, tl)
                loss.backward()
                return loss
            loss_val_t = optimizer.step(closure)  # type: ignore[arg-type]
            loss_val = float(loss_val_t) if loss_val_t is not None else float("nan")
        else:
            optimizer.zero_grad()
            H = compute_homography(src_corners, params)
            warped = warp_reference(ref_t, H, out_h, out_w)
            wl, tl = prep_for_loss(warped, cap_t, loss_name)
            loss = loss_fn(wl, tl)
            loss.backward()
            optimizer.step()
            loss_val = loss.item()

        # Prevent extreme vertex divergence.
        with torch.no_grad():
            params.clamp_(-0.5, 1.5)

        if loss_val < best_loss:
            best_loss = loss_val
            best_params = params.data.clone()

        if (i + 1) % args.report_every == 0 or i == 0:
            print(f"  iter {i+1:4d}/{args.iters}  loss={loss_val:.6f}  best={best_loss:.6f}")

    print(f"final loss: {loss_val:.6f}  (best: {best_loss:.6f})")

    # ---- save output JSON ----
    result = build_output(
        ref_path=ref_path, cap_path=cap_path,
        ref_w=ref_w_full, ref_h=ref_h_full,
        cap_w=cap_w_full, cap_h=cap_h_full,
        params=best_params,
        loss_name=loss_name, greyscale=args.greyscale,
        iters=args.iters, optimizer_name=args.optimizer, lr=lr,
        initial_loss=initial_loss, final_loss=best_loss,
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"saved: {out_json}")

    # ---- vertex adjustment table (terminal) ----
    best_cpu = best_params.cpu()
    print(f"\nvertex adjustments (reference pixels, W={ref_w_full} H={ref_h_full}):")
    print(f"  {'':4s}{'before (x, y)':>24s}{'after (x, y)':>24s}{'delta (px)':>20s}")
    for i, key in enumerate(_V_KEYS):
        bx = init_params_data[i, 0].item() * ref_w_full
        by = init_params_data[i, 1].item() * ref_h_full
        ax = best_cpu[i, 0].item() * ref_w_full
        ay = best_cpu[i, 1].item() * ref_h_full
        dx, dy = ax - bx, ay - by
        print(f"  {key.upper():4s}"
              f"({bx:9.2f},{by:9.2f}){ax:11.2f},{ay:9.2f} "
              f"   ({dx:+7.2f},{dy:+7.2f})")

    # ---- post-optimisation comparison ----
    if compare_after is not None:
        print("rendering post-optimisation comparison ...")
        save_comparison(
            ref_arr, cap_arr, best_cpu,
            ref_w_full, ref_h_full, compare_after,
            args.compare_blend, args.compare_opacity, device, comp_size,
        )

    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Gradient-based fine-tuning of a quad alignment.\n\n"
            "When --reference and --captured are both directories the script\n"
            "processes every matching (captured, reference) pair.  Each captured\n"
            "image must follow the naming convention:\n"
            "  cap-XXXX-YYY_<ref-stem>.<ext>\n"
            "The part after the last '_' (without extension) is used to locate\n"
            "the matching reference file(s) inside --reference.  --output must\n"
            "then also be a directory.  --input-json may still be a single file."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--reference",  required=True,
                   help="reference image path or directory of reference images")
    p.add_argument("--captured",   required=True,
                   help="recaptured image path or directory of captured images")
    p.add_argument("--input-json", required=True,
                   help="alignment JSON produced by align_quad.py (mandatory)")
    p.add_argument("--loss", default="ms-ssim", metavar="FUNC",
                   help="loss function: ms-ssim or dists")
    p.add_argument("--greyscale", action="store_true",
                   help="convert both images to greyscale before loss")
    p.add_argument("--iters", type=int, default=300,
                   help="number of optimisation steps")
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate (default 0.005 for adam, 0.5 for lbfgs)")
    p.add_argument("--optimizer", choices=("adam", "lbfgs"), default="adam")
    p.add_argument("--max-size", type=int, default=512,
                   help="longest edge of the working resolution for loss computation")
    p.add_argument("--output", default=None,
                   help="output JSON path (single-image mode) or output directory "
                        "(directory mode).  Default: <captured_stem>.align.opt.json "
                        "in the current directory.")
    p.add_argument("--report-every", type=int, default=50,
                   help="print loss every N iterations")
    p.add_argument("--device", default=None,
                   help="torch device (default: cuda if available, else cpu)")
    g = p.add_argument_group("comparison output")
    g.add_argument("--compare-before", default=None, metavar="PATH",
                   help="save pre-optimisation comparison image to this path "
                        "(directory in batch mode)")
    g.add_argument("--compare-after", default=None, metavar="PATH",
                   help="save post-optimisation comparison image to this path "
                        "(directory in batch mode)")
    g.add_argument("--compare-blend", nargs="+", default=["normal"],
                   choices=["normal", "gcap"], metavar="MODE",
                   help="blend mode(s) for comparison: normal gcap (default: normal)")
    g.add_argument("--compare-opacity", type=float, default=0.5,
                   help="overlay opacity for normal blend (default: 0.5)")
    g.add_argument("--compare-size", type=int, default=0,
                   help="max resolution of comparison images in pixels "
                        "(default: full reference resolution)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()

    ref_input  = Path(args.reference).resolve()
    cap_input  = Path(args.captured).resolve()
    json_path  = Path(args.input_json).resolve()

    if not json_path.is_file():
        print(f"error: file not found: {json_path}", file=sys.stderr)
        return 2
    json_data = json.loads(json_path.read_text(encoding="utf-8"))

    # ---- device ----
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ---- loss ----
    try:
        loss_name, loss_fn = build_loss(args.loss)  # type: ignore[assignment]
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    loss_fn = loss_fn.to(device)
    print(f"loss:   {loss_name}")

    # ---- single-image mode ----
    if ref_input.is_file() and cap_input.is_file():
        out_json = (
            Path(args.output).resolve()
            if args.output
            else Path.cwd() / f"{cap_input.stem}.align.opt.json"
        )
        compare_before = Path(args.compare_before).resolve() if args.compare_before else None
        compare_after  = Path(args.compare_after).resolve()  if args.compare_after  else None
        return process_pair(
            ref_path=ref_input,
            cap_path=cap_input,
            json_data=json_data,
            args=args,
            device=device,
            loss_name=loss_name,
            loss_fn=loss_fn,
            out_json=out_json,
            compare_before=compare_before,
            compare_after=compare_after,
        )

    # ---- directory mode ----
    if not ref_input.is_dir():
        print(f"error: not a file or directory: {ref_input}", file=sys.stderr)
        return 2
    if not cap_input.is_dir():
        print(f"error: not a file or directory: {cap_input}", file=sys.stderr)
        return 2

    out_dir = Path(args.output).resolve() if args.output else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    cmp_before_dir = Path(args.compare_before).resolve() if args.compare_before else None
    cmp_after_dir  = Path(args.compare_after).resolve()  if args.compare_after  else None
    if cmp_before_dir:
        cmp_before_dir.mkdir(parents=True, exist_ok=True)
    if cmp_after_dir:
        cmp_after_dir.mkdir(parents=True, exist_ok=True)

    pairs = _build_pairs(ref_input, cap_input)
    if not pairs:
        print("error: no matching image pairs found", file=sys.stderr)
        return 2

    print(f"\nfound {len(pairs)} pair(s) to process")

    # When a single captured image maps to multiple reference files we need to
    # disambiguate the output names with the reference file extension.
    from collections import Counter
    cap_counts = Counter(cap for _, cap in pairs)

    errors = 0
    for ref_path, cap_path in pairs:
        if cap_counts[cap_path] > 1:
            out_stem = f"{cap_path.stem}_{ref_path.suffix.lstrip('.')}"
        else:
            out_stem = cap_path.stem
        out_json = out_dir / f"{out_stem}.align.opt.json"

        cmp_before = (cmp_before_dir / f"{out_stem}.png") if cmp_before_dir else None
        cmp_after  = (cmp_after_dir  / f"{out_stem}.png") if cmp_after_dir  else None

        rc = process_pair(
            ref_path=ref_path,
            cap_path=cap_path,
            json_data=json_data,
            args=args,
            device=device,
            loss_name=loss_name,
            loss_fn=loss_fn,
            out_json=out_json,
            compare_before=cmp_before,
            compare_after=cmp_after,
        )
        if rc != 0:
            errors += 1

    print(f"\n{'='*60}")
    print(f"done: {len(pairs) - errors}/{len(pairs)} pairs succeeded")
    return 0 if errors == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
