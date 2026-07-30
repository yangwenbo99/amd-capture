#!/usr/bin/env python3
"""
Organize manually-downloaded capture results into a renamed directory tree
plus a CSV mapping filenames to the simulated lighting conditions used at
capture time.

Inputs:
- A ``steps.tsv`` log produced by ``control_capture_session.py``. Each
  successful ``capture`` row records the simulation parameters and
  ``capture_filenames`` (the BMP filename(s) the server returned).
- ``--bmp-dir`` (optional): a directory containing only the BMP files of
  interest. When provided, it is the canonical list of captures to organize.
  BMPs are sorted by file creation time (falling back to mtime when birth
  time is not available) and paired one-to-one (in order) with successful
  capture rows from the log; the log's recorded filenames are ignored in
  this mode. When ``--bmp-dir`` is omitted, the BMP filenames recorded in
  the log (``capture_filenames``) are used instead.
- ``--results-dir``: a directory containing the BMPs and their sidecar
  files (may also contain unrelated files which will be ignored). Defaults
  to ``--bmp-dir`` if not given.

Output naming:
    <source_index>-<repeat_index>-<source_filestem><tail>
where ``source_index`` is assigned per unique source image (in the order of
first appearance) and ``repeat_index`` counts repeated captures of that
image.

Two modes:
- ``flat``: every renamed file is placed directly in the output directory.
- ``per-capture``: each capture's files go in their own subdirectory, and a
  ``params.txt`` describing the capture parameters is written alongside.

Missing files (capture rows whose BMP is not in ``--bmp-dir``, or BMPs in
``--bmp-dir`` not referenced by the log) are reported.
"""
from __future__ import annotations

import argparse
import csv
import json
import ntpath
import os
import re
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class CaptureRow:
    ts_start_iso: str
    ts_end_iso: str
    duration_sec: float
    step: str
    ok: bool
    request_id: str
    image: str
    brightness_scale: str
    target_kelvin: str
    error: str
    extra: dict = field(default_factory=dict)


STEP_LOG_BASE_COLS = 10


def parse_steps_log(path: Path) -> list[CaptureRow]:
    rows: list[CaptureRow] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < STEP_LOG_BASE_COLS:
                continue
            (
                ts_start,
                ts_end,
                duration,
                step,
                ok_s,
                request_id,
                image,
                brightness,
                kelvin,
                error,
            ) = parts[:STEP_LOG_BASE_COLS]
            extra: dict = {}
            if len(parts) >= STEP_LOG_BASE_COLS + 1:
                extra_str = parts[STEP_LOG_BASE_COLS].strip()
                if extra_str:
                    try:
                        loaded = json.loads(extra_str)
                        if isinstance(loaded, dict):
                            extra = loaded
                    except json.JSONDecodeError:
                        pass
            try:
                duration_sec = float(duration) if duration else 0.0
            except ValueError:
                duration_sec = 0.0
            rows.append(
                CaptureRow(
                    ts_start_iso=ts_start,
                    ts_end_iso=ts_end,
                    duration_sec=duration_sec,
                    step=step,
                    ok=(ok_s.strip().lower() == "true"),
                    request_id=request_id,
                    image=image,
                    brightness_scale=brightness,
                    target_kelvin=kelvin,
                    error=error,
                    extra=extra,
                )
            )
    return rows


def filter_capture_rows(rows: list[CaptureRow]) -> list[CaptureRow]:
    return [r for r in rows if r.step == "capture" and r.ok and r.image]


def capture_filenames(row: CaptureRow) -> list[str]:
    raw = row.extra.get("capture_filenames")
    if isinstance(raw, list):
        return [str(x) for x in raw if isinstance(x, str) and x]
    if isinstance(raw, str) and raw:
        return [raw]
    return []


def _creation_time(p: Path) -> float:
    """
    Return the file's creation time when available, falling back to mtime.

    On Linux, ``st_birthtime`` exists only on Python 3.12+ and only when the
    filesystem supports it; otherwise ``st_ctime`` is inode change time, not
    creation time, so mtime is a more useful fallback for our use case.
    """
    st = p.stat()
    bt = getattr(st, "st_birthtime", None)
    if bt is not None:
        return float(bt)
    return float(st.st_mtime)


def list_bmp_files(bmp_dir: Path, suffix: str) -> list[Path]:
    suf = suffix.lower()
    files: list[Path] = []
    for p in bmp_dir.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() == suf:
            files.append(p)
    files.sort(key=lambda p: (_creation_time(p), p.name))
    return files


def find_files_for_stem(results_dir: Path, stem: str) -> list[Path]:
    """
    Return all files in ``results_dir`` whose name starts with ``stem`` and
    whose next character (if any) is a separator (``.``/``_``/``-``). This
    avoids accidentally matching ``IMG_0010.bmp`` when the stem is
    ``IMG_001``.
    """
    pattern = re.compile(r"^" + re.escape(stem) + r"([._\-].*)?$")
    matches: list[Path] = []
    for p in results_dir.iterdir():
        if not p.is_file():
            continue
        if pattern.match(p.name):
            matches.append(p)
    matches.sort(key=lambda p: p.name)
    return matches


def link_or_copy(src: Path, dst: Path, prefer_hardlink: bool) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if prefer_hardlink:
        try:
            os.link(src, dst)
            return "hardlink"
        except OSError:
            pass
    shutil.copy2(src, dst)
    return "copy"


def source_image_stem(image: str) -> str:
    base = ntpath.basename(image) or image
    if "." in base:
        return base.rsplit(".", 1)[0]
    return base


def format_params_txt(
    cap: CaptureRow,
    source_index: int,
    repeat_index: int,
    bmp_stem: str,
) -> str:
    lines = [
        f"source_index      = {source_index}",
        f"repeat_index      = {repeat_index}",
        f"source_image      = {cap.image}",
        f"source_image_stem = {source_image_stem(cap.image)}",
        f"capture_filestem  = {bmp_stem}",
        f"request_id        = {cap.request_id}",
        f"brightness_scale  = {cap.brightness_scale}",
        f"target_kelvin     = {cap.target_kelvin}",
        f"ts_start_iso      = {cap.ts_start_iso}",
        f"ts_end_iso        = {cap.ts_end_iso}",
        f"duration_sec      = {cap.duration_sec:.3f}",
        "",
    ]
    return "\n".join(lines)


CSV_FIELDS = [
    "output_prefix",
    "source_index",
    "repeat_index",
    "source_image",
    "source_image_stem",
    "capture_filestem",
    "capture_bmp",
    "request_id",
    "brightness_scale",
    "target_kelvin",
    "ts_start_iso",
    "num_files",
    "output_subdir",
    "filenames",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--steps-log",
        required=True,
        type=Path,
        help="Path to steps.tsv produced by control_capture_session.py.",
    )
    p.add_argument(
        "--bmp-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing the BMP files of interest. When "
            "given, BMPs sorted by name are paired one-to-one (in order) "
            "with successful capture rows; the log's recorded filenames are "
            "ignored. When omitted, the filenames recorded in the log are "
            "used."
        ),
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing BMPs and sidecar files (may include "
            "unrelated extras). Defaults to --bmp-dir; required if "
            "--bmp-dir is not given."
        ),
    )
    p.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Destination directory for renamed/linked files and the CSV.",
    )
    p.add_argument(
        "--mode",
        choices=("flat", "per-capture"),
        default="flat",
        help=(
            "'flat' puts all renamed files in --output-dir. 'per-capture' "
            "puts each capture's files in its own subdirectory and writes a "
            "params.txt alongside."
        ),
    )
    p.add_argument(
        "--bmp-suffix",
        default=".bmp",
        help="File suffix used to enumerate BMPs in --bmp-dir.",
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Force file copies instead of trying hardlinks first.",
    )
    p.add_argument(
        "--manifest-name",
        default="manifest.csv",
        help="Filename for the CSV written under --output-dir.",
    )
    p.add_argument(
        "--source-index-width",
        type=int,
        default=4,
        help="Zero-pad width for source_index in output names.",
    )
    p.add_argument(
        "--repeat-index-width",
        type=int,
        default=3,
        help="Zero-pad width for repeat_index in output names.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero if any warnings (missing files, mismatches) occur.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    results_dir = args.results_dir if args.results_dir is not None else args.bmp_dir
    if results_dir is None:
        raise SystemExit(
            "either --results-dir or --bmp-dir must be provided so sidecar "
            "files can be located"
        )

    if not args.steps_log.is_file():
        raise SystemExit(f"steps log does not exist: {args.steps_log}")
    if args.bmp_dir is not None and not args.bmp_dir.is_dir():
        raise SystemExit(f"bmp dir is not a directory: {args.bmp_dir}")
    if not results_dir.is_dir():
        raise SystemExit(f"results dir is not a directory: {results_dir}")

    rows = parse_steps_log(args.steps_log)
    captures = filter_capture_rows(rows)
    if not captures:
        raise SystemExit(
            f"No successful 'capture' rows in {args.steps_log}; nothing to organize."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefer_hardlink = not args.copy

    pairs: list[tuple[CaptureRow, str]] = []
    warnings: list[str] = []

    if args.bmp_dir is not None:
        bmps = list_bmp_files(args.bmp_dir, args.bmp_suffix)
        if not bmps:
            raise SystemExit(
                f"No '*{args.bmp_suffix}' files in {args.bmp_dir}."
            )
        n = min(len(bmps), len(captures))
        if len(bmps) != len(captures):
            warnings.append(
                f"count mismatch: {len(captures)} successful capture rows vs "
                f"{len(bmps)} '*{args.bmp_suffix}' files in {args.bmp_dir}; "
                f"pairing the first {n} in order."
            )
        for cap, bmp in zip(captures[:n], bmps[:n]):
            pairs.append((cap, bmp.name))
    else:
        for cap in captures:
            names = capture_filenames(cap)
            if not names:
                warnings.append(
                    f"capture {cap.request_id or '?'} (image={cap.image}): "
                    f"no capture_filenames recorded in log; skipping"
                )
                continue
            bmp_name = next(
                (n for n in names if n.lower().endswith(args.bmp_suffix.lower())),
                names[0],
            )
            pairs.append((cap, bmp_name))

    source_indices: dict[str, int] = {}
    repeat_counters: dict[str, int] = {}
    csv_rows: list[dict[str, object]] = []

    for cap, bmp_name in pairs:
        bmp_stem = bmp_name.rsplit(".", 1)[0] if "." in bmp_name else bmp_name

        if cap.image not in source_indices:
            source_indices[cap.image] = len(source_indices) + 1
            repeat_counters[cap.image] = 0
        repeat_counters[cap.image] += 1

        src_idx = source_indices[cap.image]
        rep_idx = repeat_counters[cap.image]

        prefix = (
            f"{src_idx:0{args.source_index_width}d}"
            f"-{rep_idx:0{args.repeat_index_width}d}"
            f"-{source_image_stem(cap.image)}"
        )

        target_dir = (
            args.output_dir / prefix if args.mode == "per-capture" else args.output_dir
        )

        matches = find_files_for_stem(results_dir, bmp_stem)
        if not matches:
            warnings.append(
                f"capture {cap.request_id or '?'} (bmp={bmp_name}): "
                f"no files in {results_dir} match stem {bmp_stem!r}"
            )
            continue

        out_filenames: list[str] = []
        for src in matches:
            tail = src.name[len(bmp_stem):]
            new_name = f"{prefix}{tail}"
            dst = target_dir / new_name
            link_or_copy(src, dst, prefer_hardlink=prefer_hardlink)
            out_filenames.append(new_name)

        if args.mode == "per-capture":
            (target_dir / "params.txt").write_text(
                format_params_txt(cap, src_idx, rep_idx, bmp_stem),
                encoding="utf-8",
            )

        csv_rows.append(
            {
                "output_prefix": prefix,
                "source_index": src_idx,
                "repeat_index": rep_idx,
                "source_image": cap.image,
                "source_image_stem": source_image_stem(cap.image),
                "capture_filestem": bmp_stem,
                "capture_bmp": bmp_name,
                "request_id": cap.request_id,
                "brightness_scale": cap.brightness_scale,
                "target_kelvin": cap.target_kelvin,
                "ts_start_iso": cap.ts_start_iso,
                "num_files": len(matches),
                "output_subdir": prefix if args.mode == "per-capture" else "",
                "filenames": ";".join(out_filenames),
            }
        )

    csv_path = args.output_dir / args.manifest_name
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for row in csv_rows:
            w.writerow(row)

    print(
        f"Organized {len(csv_rows)} captures into {args.output_dir} "
        f"(mode={args.mode}); manifest: {csv_path}",
        file=sys.stderr,
    )

    if warnings:
        print(f"Warnings ({len(warnings)}):", file=sys.stderr)
        for w_msg in warnings:
            print(f"  {w_msg}", file=sys.stderr)
        if args.strict:
            return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
