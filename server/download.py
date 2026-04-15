#!/usr/bin/env python3
"""
download_all_files.py

Download each file returned by the capture server's /files endpoint.

Example:
    python download_all_files.py ^
        --server http://192.168.1.10:8765 ^
        --output-dir downloads

With filters:
    python download_all_files.py ^
        --server http://192.168.1.10:8765 ^
        --output-dir downloads ^
        --pattern "*.png" ^
        --modified-after "2026-04-08T12:30:00" ^
        --modified-before "2026-04-08T13:00:00"

With token:
    python download_all_files.py ^
        --server http://192.168.1.10:8765 ^
        --output-dir downloads ^
        --token my-lan-secret
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import requests


def get_file_list(
    server: str,
    token: str | None,
    watch_folder: str | None,
    pattern: str | None,
    modified_after: str | None,
    modified_before: str | None,
    timeout: float,
) -> list[dict[str, Any]]:
    params: dict[str, str] = {}

    if token:
        params["token"] = token
    if watch_folder:
        params["watch_folder"] = watch_folder
    if pattern:
        params["pattern"] = pattern
    if modified_after:
        params["modified_after"] = modified_after
    if modified_before:
        params["modified_before"] = modified_before

    url = f"{server.rstrip('/')}/files"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()

    payload = resp.json()
    if not payload.get("ok", False):
        raise RuntimeError(f"Server returned failure payload: {payload}")

    files = payload.get("files")
    if not isinstance(files, list):
        raise RuntimeError("Malformed server response: 'files' is missing or not a list.")

    return files


def download_file(
    server: str,
    filename: str,
    output_dir: Path,
    token: str | None,
    watch_folder: str | None,
    timeout: float,
    overwrite: bool,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / filename

    if out_path.exists() and not overwrite:
        print(f"[skip] {filename} already exists")
        return out_path

    params: dict[str, str] = {"name": filename}
    if token:
        params["token"] = token
    if watch_folder:
        params["watch_folder"] = watch_folder

    url = f"{server.rstrip('/')}/pull"
    with requests.get(url, params=params, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)

    print(f"[ok] downloaded {filename}")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Download all files listed by the capture server.")
    parser.add_argument("--server", required=True, help="Server base URL, e.g. http://192.168.1.10:8765")
    parser.add_argument("--output-dir", required=True, help="Local directory to save downloaded files")
    parser.add_argument("--token", default=None, help="Optional shared secret token")
    parser.add_argument("--watch-folder", default=None, help="Optional server-side watch folder override")
    parser.add_argument("--pattern", default=None, help='Optional filename glob, e.g. "*.png"')
    parser.add_argument("--modified-after", default=None, help="Optional time filter passed to /files")
    parser.add_argument("--modified-before", default=None, help="Optional time filter passed to /files")
    parser.add_argument("--timeout", type=float, default=60.0, help="HTTP timeout in seconds")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing local files")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    try:
        files = get_file_list(
            server=args.server,
            token=args.token,
            watch_folder=args.watch_folder,
            pattern=args.pattern,
            modified_after=args.modified_after,
            modified_before=args.modified_before,
            timeout=args.timeout,
        )
    except Exception as exc:
        print(f"Failed to get file list: {exc}", file=sys.stderr)
        return 1

    if not files:
        print("No files matched.")
        return 0

    print(f"Found {len(files)} file(s).")

    failed = 0
    for entry in files:
        filename = entry.get("name")
        if not filename:
            print(f"[warn] skipping malformed entry: {entry}")
            failed += 1
            continue

        try:
            download_file(
                server=args.server,
                filename=filename,
                output_dir=output_dir,
                token=args.token,
                watch_folder=args.watch_folder,
                timeout=args.timeout,
                overwrite=args.overwrite,
            )
        except Exception as exc:
            print(f"[fail] {filename}: {exc}", file=sys.stderr)
            failed += 1

    if failed:
        print(f"Completed with {failed} failure(s).", file=sys.stderr)
        return 2

    print("All files downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
