#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Iterable

from PIL import Image


def clamp_int(x: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, x))


def iter_images(input_dir: Path) -> Iterable[Path]:
    for p in sorted(input_dir.iterdir()):
        if p.is_file():
            yield p


def compute_overlay_size(
    base_w: int,
    base_h: int,
    checker_w: int,
    checker_h: int,
) -> tuple[int, int]:
    max_w = base_w // 2
    max_h = base_h // 3

    if max_w <= 0 or max_h <= 0:
        return 0, 0

    if checker_w <= 0 or checker_h <= 0:
        return 0, 0

    scale_w = max_w / float(checker_w)
    scale_h = max_h / float(checker_h)
    scale = min(scale_w, scale_h)

    out_w = int(checker_w * scale)
    out_h = int(checker_h * scale)

    out_w = max(1, min(out_w, max_w))
    out_h = max(1, min(out_h, max_h))
    return out_w, out_h


def paste_checker(
    base: Image.Image,
    checker_rgba: Image.Image,
    y_jitter_px: int,
    rng: random.Random,
) -> Image.Image:
    base_rgba = base.convert("RGBA")
    bw, bh = base_rgba.size
    cw, ch = checker_rgba.size

    out_cw, out_ch = compute_overlay_size(bw, bh, cw, ch)
    if out_cw <= 0 or out_ch <= 0:
        return base_rgba

    checker = checker_rgba.resize((out_cw, out_ch), Image.Resampling.LANCZOS)

    x = (bw - out_cw) // 2
    jitter = rng.randint(-y_jitter_px, y_jitter_px) if y_jitter_px else 0
    y = (bh // 2) + jitter
    y = clamp_int(y, 0, bh - out_ch)

    layer = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    layer.paste(checker, (x, y), mask=checker)
    out = Image.alpha_composite(base_rgba, layer)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Overlay a color-checker image onto every image in a directory."
        )
    )
    p.add_argument(
        "--input-dir",
        required=True,
        help="directory containing input images",
    )
    p.add_argument(
        "--output-dir",
        required=True,
        help="directory to write output images",
    )
    p.add_argument(
        "--checker",
        required=True,
        help="path to the color-checker image",
    )
    p.add_argument(
        "--y-jitter-px",
        type=int,
        default=10,
        help=(
            "random +/- pixel jitter added to the default y position (h/2)"
        ),
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="optional RNG seed for reproducible placement",
    )
    p.add_argument(
        "--format",
        default=None,
        help=(
            "optional output format override (e.g. PNG). "
            "If omitted, saves using the input file extension."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    checker_path = Path(args.checker)

    if not input_dir.is_dir():
        raise SystemExit(f"input dir is not a directory: {input_dir}")
    if not checker_path.is_file():
        raise SystemExit(f"checker is not a file: {checker_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    checker_rgba = Image.open(checker_path).convert("RGBA")

    processed = 0
    failed = 0
    for p in iter_images(input_dir):
        try:
            with Image.open(p) as im:
                out = paste_checker(
                    base=im,
                    checker_rgba=checker_rgba,
                    y_jitter_px=int(args.y_jitter_px),
                    rng=rng,
                )

                out_path = output_dir / p.name
                save_kwargs: dict[str, object] = {}
                if args.format:
                    save_kwargs["format"] = str(args.format)

                fmt = save_kwargs.get("format")
                if fmt is None:
                    ext = (out_path.suffix or "").lower().lstrip(".")
                    fmt = ext.upper() if ext else None

                if isinstance(fmt, str) and fmt.upper() in {"JPG", "JPEG"}:
                    out = out.convert("RGB")
                    save_kwargs["quality"] = 95

                save_kwargs2 = {
                    k: v for k, v in save_kwargs.items() if v is not None
                }
                out.save(out_path, **save_kwargs2)

            processed += 1
        except Exception as exc:
            failed += 1
            print(f"[fail] {p}: {exc}")

    print(f"[ok] processed={processed} failed={failed} output_dir={output_dir}")
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
