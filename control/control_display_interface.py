#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from typing import Any
from urllib.parse import urljoin

import requests


def _join(base: str, path: str) -> str:
    base2 = base.rstrip("/") + "/"
    return urljoin(base2, path.lstrip("/"))


def _http_get_json(url: str, timeout: float) -> dict[str, Any]:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected JSON payload type: {type(payload)}")
    return payload


def _http_post_json(
    url: str,
    body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    response = requests.post(url, json=body, timeout=timeout)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"Unexpected JSON payload type: {type(payload)}")
    return payload


def _display_get(
    display_base: str,
    path: str,
    timeout: float,
) -> dict[str, Any]:
    return _http_get_json(_join(display_base, path), timeout=timeout)


def _display_post(
    display_base: str,
    path: str,
    body: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    return _http_post_json(_join(display_base, path), body=body, timeout=timeout)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Call display server HTTP endpoints individually.",
    )
    parser.add_argument(
        "--display",
        required=True,
        help="display server base URL, e.g. http://127.0.0.1:8080",
    )
    parser.add_argument(
        "--http-timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ping", help="GET /ping")
    sub.add_parser("status", help="GET /status")

    p_load = sub.add_parser("load", help="POST /load")
    p_load.add_argument(
        "--path",
        required=True,
        help="path passed to /load",
    )

    p_simulate = sub.add_parser("simulate", help="POST /simulate")
    p_simulate.add_argument("--brightness-scale", type=float, default=None)
    p_simulate.add_argument("--target-kelvin", type=int, default=None)
    p_simulate.add_argument("--gamma", type=float, default=None)
    p_simulate.add_argument("--saturation", type=float, default=None)
    p_simulate.add_argument("--augmentation-mode", default=None)
    p_simulate.add_argument(
        "--crop-enabled",
        dest="crop_enabled",
        action="store_true",
    )
    p_simulate.add_argument(
        "--crop-disabled",
        dest="crop_enabled",
        action="store_false",
    )
    p_simulate.set_defaults(crop_enabled=None)
    p_simulate.add_argument("--crop-ratio", default=None)
    p_simulate.add_argument(
        "--crop-mode",
        choices=("crop", "reflect_pad"),
        default=None,
    )
    p_simulate.add_argument(
        "--enabled",
        dest="enabled",
        action="store_true",
        help="set simulation enabled=true",
    )
    p_simulate.add_argument(
        "--disabled",
        dest="enabled",
        action="store_false",
        help="set simulation enabled=false",
    )
    p_simulate.set_defaults(enabled=None)

    p_load_black = sub.add_parser("load-black", help="POST /load-black")
    p_load_black.add_argument(
        "--duration-sec",
        type=float,
        default=0.0,
        help="duration_sec passed to /load-black",
    )

    sub.add_parser("reset-simulation", help="POST /reset-simulation")

    p_pause = sub.add_parser("pause", help="POST /pause")
    p_pause.add_argument(
        "--paused",
        dest="paused",
        action="store_true",
        default=True,
        help="set paused=true (default)",
    )
    p_pause.add_argument(
        "--resumed",
        dest="paused",
        action="store_false",
        help="set paused=false",
    )

    p_tv = sub.add_parser("tv-baseline", help="POST /tv-baseline")
    p_tv.add_argument("--brightness", default=None)
    p_tv.add_argument("--color-temperature", default=None)
    p_tv.add_argument("--notes", default=None)

    return parser.parse_args()


def run_command(args: argparse.Namespace) -> dict[str, Any]:
    timeout = float(args.http_timeout)
    command = str(args.command)

    if command == "ping":
        return _display_get(args.display, "/ping", timeout)
    if command == "status":
        return _display_get(args.display, "/status", timeout)
    if command == "load":
        return _display_post(
            args.display,
            "/load",
            {"path": str(args.path)},
            timeout,
        )
    if command == "simulate":
        body: dict[str, Any] = {}
        if args.brightness_scale is not None:
            body["brightness_scale"] = float(args.brightness_scale)
        if args.target_kelvin is not None:
            body["target_kelvin"] = int(args.target_kelvin)
        if args.gamma is not None:
            body["gamma"] = float(args.gamma)
        if args.saturation is not None:
            body["saturation"] = float(args.saturation)
        if args.enabled is not None:
            body["enabled"] = bool(args.enabled)
        if args.augmentation_mode is not None:
            body["augmentation_mode"] = str(args.augmentation_mode)
        if args.crop_enabled is not None:
            body["crop_enabled"] = bool(args.crop_enabled)
        if args.crop_ratio is not None:
            body["crop_ratio"] = str(args.crop_ratio)
        if args.crop_mode is not None:
            body["crop_mode"] = str(args.crop_mode)
        return _display_post(args.display, "/simulate", body, timeout)
    if command == "load-black":
        return _display_post(
            args.display,
            "/load-black",
            {"duration_sec": float(args.duration_sec)},
            timeout,
        )
    if command == "reset-simulation":
        return _display_post(args.display, "/reset-simulation", {}, timeout)
    if command == "pause":
        return _display_post(
            args.display,
            "/pause",
            {"paused": bool(args.paused)},
            timeout,
        )
    if command == "tv-baseline":
        body: dict[str, Any] = {}
        if args.brightness is not None:
            body["brightness"] = str(args.brightness)
        if args.color_temperature is not None:
            body["color_temperature"] = str(args.color_temperature)
        if args.notes is not None:
            body["notes"] = str(args.notes)
        if not body:
            raise SystemExit(
                "tv-baseline requires at least one of: "
                "--brightness, --color-temperature, --notes",
            )
        return _display_post(args.display, "/tv-baseline", body, timeout)

    raise RuntimeError(f"Unknown command: {command}")


def main() -> int:
    args = parse_args()
    try:
        payload = run_command(args)
    except requests.RequestException as exc:
        print(f"request failed: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
