#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence
from urllib.parse import urljoin

import requests


def _sleep(seconds: float) -> None:
    if seconds <= 0:
        return
    time.sleep(seconds)


def _join(base: str, path: str) -> str:
    base2 = base.rstrip("/") + "/"
    return urljoin(base2, path.lstrip("/"))


def http_get_json(url: str, timeout: float) -> dict[str, Any]:
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected JSON payload type: {type(payload)}")
    return payload


def http_post_json(
    url: str,
    data: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    resp = requests.post(url, json=data, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected JSON payload type: {type(payload)}")
    return payload


def ensure_ok(payload: dict[str, Any], what: str) -> None:
    ok = payload.get("ok", True)
    if ok is True:
        return
    raise RuntimeError(f"{what} failed: {payload}")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class StepLog:
    ts_start_iso: str
    ts_end_iso: str
    duration_sec: float
    step: str
    request_id: str | None = None
    image: str | None = None
    brightness_scale: float | None = None
    target_kelvin: int | None = None
    ok: bool = True
    error: str | None = None
    payload: dict[str, Any] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


def append_step_log(path: Path | None, entry: StepLog) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    line = (
        f"{entry.ts_start_iso}\t{entry.ts_end_iso}\t"
        f"{entry.duration_sec:.3f}\t{entry.step}\t"
        f"{entry.ok}\t{entry.request_id or ''}\t"
        f"{entry.image or ''}\t{entry.brightness_scale or ''}\t"
        f"{entry.target_kelvin or ''}\t{entry.error or ''}\n"
    )
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)


def timed_step(
    log_path: Path | None,
    step: str,
    fn,
    *,
    request_id: str | None = None,
    image: str | None = None,
    brightness_scale: float | None = None,
    target_kelvin: int | None = None,
    extra: dict[str, Any] | None = None,
):
    t0 = time.time()
    ts0 = now_iso()
    try:
        result = fn()
        ts1 = now_iso()
        append_step_log(
            log_path,
            StepLog(
                ts_start_iso=ts0,
                ts_end_iso=ts1,
                duration_sec=time.time() - t0,
                step=step,
                request_id=request_id,
                image=image,
                brightness_scale=brightness_scale,
                target_kelvin=target_kelvin,
                ok=True,
                extra=extra or {},
            ),
        )
        return result
    except Exception as exc:
        ts1 = now_iso()
        append_step_log(
            log_path,
            StepLog(
                ts_start_iso=ts0,
                ts_end_iso=ts1,
                duration_sec=time.time() - t0,
                step=step,
                request_id=request_id,
                image=image,
                brightness_scale=brightness_scale,
                target_kelvin=target_kelvin,
                ok=False,
                error=str(exc),
                extra=extra or {},
            ),
        )
        raise


def _is_image_path(p: Path, suffixes: set[str]) -> bool:
    if not p.is_file():
        return False
    return p.suffix.lower() in suffixes


def list_images_in_dir(
    image_dir: Path,
    suffixes: Sequence[str],
) -> list[str]:
    suffix_set = {s.lower() for s in suffixes}
    if not image_dir.exists():
        raise SystemExit(f"image dir does not exist: {image_dir}")
    if not image_dir.is_dir():
        raise SystemExit(f"image dir is not a directory: {image_dir}")

    images: list[Path] = []
    for p in sorted(image_dir.iterdir()):
        if _is_image_path(p, suffix_set):
            images.append(p)

    if not images:
        suf = ",".join(sorted(suffix_set))
        raise SystemExit(f"no images in {image_dir} with suffixes: {suf}")

    # Send paths relative to image_dir so they can map to display media-root
    # when the media root is configured to the same directory.
    return [p.name for p in images]


def linspace(lo: float, hi: float, n: int) -> list[float]:
    if n <= 0:
        raise ValueError("n must be >= 1")
    if n == 1:
        return [lo]
    step = (hi - lo) / float(n - 1)
    return [lo + step * i for i in range(n)]


def display_simulate(
    display_base: str,
    brightness_scale: float,
    target_kelvin: int,
    timeout: float,
) -> None:
    url = _join(display_base, "/simulate")
    payload = http_post_json(
        url,
        {
            "brightness_scale": brightness_scale,
            "target_kelvin": target_kelvin,
        },
        timeout=timeout,
    )
    ensure_ok(payload, "display simulate")


def parse_images(args: argparse.Namespace) -> list[str]:
    images: list[str] = []

    if args.image:
        images.extend(args.image)

    if args.image_dir:
        suffixes = args.image_suffixes.split(",")
        suffixes = [s.strip() for s in suffixes if s.strip()]
        images.extend(
            list_images_in_dir(Path(args.image_dir), suffixes=suffixes)
        )

    if args.image_list:
        p = Path(args.image_list)
        text = p.read_text(encoding="utf-8")
        for raw in text.splitlines():
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            images.append(s)

    if not images:
        raise SystemExit("No images provided. Use --image or --image-list.")
    return images


def display_load(
    display_base: str,
    path: str,
    timeout: float,
) -> None:
    url = _join(display_base, "/load")
    payload = http_post_json(url, {"path": path}, timeout=timeout)
    ensure_ok(payload, "display load")


def capture_trigger(
    capture_base: str,
    token: str | None,
    request_id: str | None,
    timeout: float,
    expected_glob: str | None,
    expected_count: int | None,
    watch_folder: str | None,
    capture_timeout_sec: float | None,
) -> dict[str, Any]:
    url = _join(capture_base, "/capture")
    body: dict[str, Any] = {}

    if token:
        body["token"] = token
    if request_id:
        body["request_id"] = request_id
    if expected_glob:
        body["expected_glob"] = expected_glob
    if expected_count is not None:
        body["expected_count"] = expected_count
    if watch_folder:
        body["watch_folder"] = watch_folder
    if capture_timeout_sec is not None:
        body["timeout_sec"] = capture_timeout_sec

    payload = http_post_json(url, body, timeout=timeout)
    # /capture returns a pydantic model; ok is always present.
    if payload.get("ok") is not True:
        raise RuntimeError(f"capture failed: {payload}")
    return payload


def is_capture_timeout(payload: dict[str, Any]) -> bool:
    if payload.get("ok") is True:
        return False
    msg = payload.get("message")
    if not isinstance(msg, str):
        return False
    return "Timed out waiting" in msg or "Timed out" in msg


def capture_with_retries(
    *,
    capture_base: str,
    token: str | None,
    request_id: str | None,
    timeout: float,
    expected_glob: str | None,
    expected_count: int | None,
    watch_folder: str | None,
    capture_timeout_sec: float | None,
    retries: int,
    retry_delay_sec: float,
) -> dict[str, Any]:
    attempts = 0
    last_exc: Exception | None = None
    while attempts <= retries:
        try:
            return capture_trigger(
                capture_base=capture_base,
                token=token,
                request_id=request_id,
                timeout=timeout,
                expected_glob=expected_glob,
                expected_count=expected_count,
                watch_folder=watch_folder,
                capture_timeout_sec=capture_timeout_sec,
            )
        except RuntimeError as exc:
            last_exc = exc
            text = str(exc)
            is_timeout = "Timed out waiting" in text or "Timed out" in text
            if attempts < retries and is_timeout:
                _sleep(float(retry_delay_sec))
                attempts += 1
                continue
            raise
        except requests.RequestException as exc:
            last_exc = exc
            if attempts >= retries:
                raise
            _sleep(float(retry_delay_sec))
            attempts += 1

    if last_exc is not None:
        raise last_exc
    raise RuntimeError("capture retry loop ended unexpectedly")


def download_captured_files(
    capture_base: str,
    token: str | None,
    watch_folder: str | None,
    files_found: Iterable[str],
    output_dir: Path,
    timeout: float,
    overwrite: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pulled: list[Path] = []

    for full in files_found:
        name = os.path.basename(full)
        if not name:
            continue

        stem = Path(name).stem
        if not stem:
            continue

        out_path = output_dir / stem
        if out_path.exists() and not overwrite:
            pulled.append(out_path)
            continue

        params: dict[str, str] = {"name": name}
        if token:
            params["token"] = token
        if watch_folder:
            params["watch_folder"] = watch_folder

        url = _join(capture_base, "/pull")
        with requests.get(
            url,
            params=params,
            stream=True,
            timeout=timeout,
        ) as resp:
            resp.raise_for_status()
            with open(out_path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)

        pulled.append(out_path)

    return pulled


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Control an mpv display server and a capture server in one session."
        )
    )
    p.add_argument(
        "--display",
        required=True,
        help="mpv display server base URL, e.g. http://192.168.1.2:8080",
    )
    p.add_argument(
        "--capture",
        required=True,
        help="capture server base URL, e.g. http://192.168.1.10:48765",
    )
    p.add_argument(
        "--token",
        default=None,
        help="optional capture server token",
    )

    p.add_argument(
        "--image",
        action="append",
        default=None,
        help=(
            "image path to send to the display server (repeatable). "
            "May be relative to the display server's media-root."
        ),
    )
    p.add_argument(
        "--image-dir",
        default=None,
        help=(
            "directory of images; iterates each file (non-recursive). "
            "Use --image-suffixes to control which files are included."
        ),
    )
    p.add_argument(
        "--image-suffixes",
        default=".png,.bmp,.jpg,.jpeg,.webp,.tif,.tiff,.exr",
        help=(
            "comma-separated file suffixes when using --image-dir, "
            "example: .bmp,.png"
        ),
    )
    p.add_argument(
        "--image-list",
        default=None,
        help="text file with one image path per line",
    )

    p.add_argument(
        "-k",
        "--k-captures",
        type=int,
        default=1,
        help="number of captures per image with different settings",
    )
    p.add_argument(
        "--brightness-scale-min",
        type=float,
        default=1.0,
        help="minimum brightness_scale for display /simulate",
    )
    p.add_argument(
        "--brightness-scale-max",
        type=float,
        default=1.0,
        help="maximum brightness_scale for display /simulate",
    )
    p.add_argument(
        "--kelvin-min",
        type=int,
        default=6500,
        help="minimum target_kelvin for display /simulate",
    )
    p.add_argument(
        "--kelvin-max",
        type=int,
        default=6500,
        help="maximum target_kelvin for display /simulate",
    )
    p.add_argument(
        "--delay-after-simulate",
        type=float,
        default=0.0,
        help="delay after /simulate before capture (seconds)",
    )

    p.add_argument(
        "--delay-before-load",
        type=float,
        default=0.0,
        help="delay before loading each image (seconds)",
    )
    p.add_argument(
        "--delay-after-load",
        type=float,
        default=0.0,
        help="delay after loading each image (seconds)",
    )
    p.add_argument(
        "--delay-before-capture",
        type=float,
        default=0.75,
        help=(
            "delay between showing the image and triggering capture (seconds)"
        ),
    )
    p.add_argument(
        "--delay-after-capture",
        type=float,
        default=0.0,
        help="delay after capture completes (seconds)",
    )

    p.add_argument(
        "--expected-glob",
        default=None,
        help="override capture server expected_glob (e.g. *.bmp)",
    )
    p.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="override capture server expected_count",
    )
    p.add_argument(
        "--watch-folder",
        default=None,
        help="override capture server watch_folder",
    )
    p.add_argument(
        "--capture-timeout-sec",
        type=float,
        default=None,
        help="override capture server timeout_sec (waiting for file)",
    )
    p.add_argument(
        "--http-timeout",
        type=float,
        default=180.0,
        help="HTTP request timeout (seconds)",
    )
    p.add_argument(
        "--capture-retries",
        type=int,
        default=2,
        help="retry capture on timeout/error this many times",
    )
    p.add_argument(
        "--retry-delay-sec",
        type=float,
        default=3.0,
        help="sleep this many seconds before retrying capture",
    )
    p.add_argument(
        "--step-log",
        default=None,
        help=(
            "optional TSV log file for step timestamps; "
            "default: <download-dir>/steps.tsv when downloading"
        ),
    )

    p.add_argument(
        "--download-dir",
        default=None,
        help=(
            "if set, download every captured file into this local directory"
        ),
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="overwrite existing files in --download-dir",
    )

    p.add_argument(
        "--request-id-prefix",
        default="cap",
        help="prefix for capture request_id values",
    )

    return p.parse_args()


def main() -> int:
    args = parse_args()
    images = parse_images(args)

    download_dir = Path(args.download_dir) if args.download_dir else None
    step_log = Path(args.step_log) if args.step_log else None
    if step_log is None and download_dir is not None:
        step_log = download_dir / "steps.tsv"

    if args.k_captures < 1:
        raise SystemExit("--k-captures must be >= 1")
    if args.capture_retries < 0:
        raise SystemExit("--capture-retries must be >= 0")

    b_vals = linspace(
        float(args.brightness_scale_min),
        float(args.brightness_scale_max),
        int(args.k_captures),
    )
    k_vals_f = linspace(
        float(args.kelvin_min),
        float(args.kelvin_max),
        int(args.k_captures),
    )
    k_vals = [int(round(x)) for x in k_vals_f]

    # Quick connectivity checks (fail fast with readable errors).
    try:
        timed_step(
            step_log,
            "display_ping",
            lambda: ensure_ok(
                http_get_json(_join(args.display, "/ping"), args.http_timeout),
                "display ping",
            ),
        )
    except Exception:
        # Older versions may not have /ping; fall back to /status.
        timed_step(
            step_log,
            "display_status",
            lambda: ensure_ok(
                http_get_json(_join(args.display, "/status"), args.http_timeout),
                "display status",
            ),
        )

    timed_step(
        step_log,
        "capture_health",
        lambda: ensure_ok(
            http_get_json(_join(args.capture, "/health"), args.http_timeout),
            "capture health",
        ),
    )

    for img_idx, img in enumerate(images, start=1):
        timed_step(
            step_log,
            "delay_before_load",
            lambda: _sleep(float(args.delay_before_load)),
            image=img,
        )
        timed_step(
            step_log,
            "display_load",
            lambda: display_load(args.display, img, timeout=args.http_timeout),
            image=img,
        )
        timed_step(
            step_log,
            "delay_after_load",
            lambda: _sleep(float(args.delay_after_load)),
            image=img,
        )

        for sweep_idx in range(int(args.k_captures)):
            request_id = (
                f"{args.request_id_prefix}-{img_idx:04d}-{sweep_idx:03d}"
            )
            brightness_scale = float(b_vals[sweep_idx])
            target_kelvin = int(k_vals[sweep_idx])

            timed_step(
                step_log,
                "display_simulate",
                lambda: display_simulate(
                    args.display,
                    brightness_scale=brightness_scale,
                    target_kelvin=target_kelvin,
                    timeout=args.http_timeout,
                ),
                request_id=request_id,
                image=img,
                brightness_scale=brightness_scale,
                target_kelvin=target_kelvin,
            )
            timed_step(
                step_log,
                "delay_after_simulate",
                lambda: _sleep(float(args.delay_after_simulate)),
                request_id=request_id,
                image=img,
                brightness_scale=brightness_scale,
                target_kelvin=target_kelvin,
            )

            timed_step(
                step_log,
                "delay_before_capture",
                lambda: _sleep(float(args.delay_before_capture)),
                request_id=request_id,
                image=img,
                brightness_scale=brightness_scale,
                target_kelvin=target_kelvin,
            )
            cap = timed_step(
                step_log,
                "capture",
                lambda: capture_with_retries(
                    capture_base=args.capture,
                    token=args.token,
                    request_id=request_id,
                    timeout=args.http_timeout,
                    expected_glob=args.expected_glob,
                    expected_count=args.expected_count,
                    watch_folder=args.watch_folder,
                    capture_timeout_sec=args.capture_timeout_sec,
                    retries=int(args.capture_retries),
                    retry_delay_sec=float(args.retry_delay_sec),
                ),
                request_id=request_id,
                image=img,
                brightness_scale=brightness_scale,
                target_kelvin=target_kelvin,
            )

            files_found = cap.get("files_found", [])
            if not isinstance(files_found, list):
                raise RuntimeError(f"Malformed capture response: {cap}")

            if download_dir is not None and files_found:
                timed_step(
                    step_log,
                    "download",
                    lambda: download_captured_files(
                        capture_base=args.capture,
                        token=args.token,
                        watch_folder=args.watch_folder,
                        files_found=files_found,
                        output_dir=download_dir,
                        timeout=args.http_timeout,
                        overwrite=bool(args.overwrite),
                    ),
                    request_id=request_id,
                    image=img,
                    brightness_scale=brightness_scale,
                    target_kelvin=target_kelvin,
                    extra={"files_found_count": len(files_found)},
                )

            timed_step(
                step_log,
                "delay_after_capture",
                lambda: _sleep(float(args.delay_after_capture)),
                request_id=request_id,
                image=img,
                brightness_scale=brightness_scale,
                target_kelvin=target_kelvin,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
