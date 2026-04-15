"""
capture_server.py

Windows LAN capture automation server.

Features:
- Accept LAN HTTP requests
- Click a button in a Windows GUI
- Wait for files to appear in a folder
- Return JSON status
- List files in monitored folder
- Download files from monitored folder

Install:
    py -m pip install fastapi uvicorn pywinauto pydantic

Run:
    py capture_server.py

Examples:
    curl http://HOST_IP:8765/health
    curl http://HOST_IP:8765/files
    curl -X POST http://HOST_IP:8765/capture ^
         -H "Content-Type: application/json" ^
         -d "{\"request_id\":\"test1\"}"

Download a file:
    curl -OJ "http://HOST_IP:8765/pull?name=img001.png"

Notes:
- Run in an interactive Windows desktop session.
- The target app should already be open.
- Prefer automation_id over screen coordinates.
"""

from __future__ import annotations

import fnmatch
import logging
import mimetypes
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pywinauto import Application
from pywinauto.findwindows import ElementNotFoundError
from pywinauto import mouse
import time

# =========================
# Configuration
# =========================

HOST = "0.0.0.0"
PORT = 48765

WATCH_FOLDER = Path(r"C:\Users\Administrator\Documents\EFFP_new\EFFP_ISProj\RawCapture")

EXPECTED_FILE_GLOB = "*.bmp"
EXPECTED_FILE_COUNT = 1
CAPTURE_TIMEOUT_SEC = 20.0
POLL_INTERVAL_SEC = 0.25

BACKEND = "uia"

WINDOW_TITLE_REGEX = r".*Image-Studio.*"
WINDOW_TITLE = None

BUTTON_AUTOMATION_ID = None
BUTTON_TITLE = None # "Capture"
BUTTON_BEST_MATCH = "Capture"
BUTTON_CONTROL_TYPE = "Button"

PRE_CLICK_DELAY_SEC = 0.1
POST_CLICK_DELAY_SEC = 0.1

API_TOKEN = None  # e.g. "my-lan-secret"

LOG_LEVEL = logging.INFO


# =========================
# Logging
# =========================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("capture_server")


# =========================
# Data models
# =========================

class CaptureRequest(BaseModel):
    request_id: Optional[str] = Field(default=None)
    token: Optional[str] = Field(default=None)
    timeout_sec: Optional[float] = Field(default=None, ge=0.5, le=300.0)
    expected_glob: Optional[str] = Field(default=None)
    expected_count: Optional[int] = Field(default=None, ge=1, le=100)
    watch_folder: Optional[str] = Field(default=None)


class CaptureResponse(BaseModel):
    ok: bool
    message: str
    request_id: Optional[str]
    started_at_unix: float
    ended_at_unix: float
    elapsed_sec: float
    files_found: list[str]
    watch_folder: str
    expected_glob: str
    expected_count: int


class FileEntry(BaseModel):
    name: str
    size_bytes: int
    modified_unix: float
    modified_iso: str


@dataclass
class LastStatus:
    ok: bool
    message: str
    request_id: Optional[str]
    started_at_unix: float
    ended_at_unix: float
    elapsed_sec: float
    files_found: list[str]
    watch_folder: str
    expected_glob: str
    expected_count: int


# =========================
# Global state
# =========================

app = FastAPI(title="LAN Capture Server")
capture_lock = threading.Lock()
last_status: Optional[LastStatus] = None


# =========================
# Helpers
# =========================

def check_token(token: Optional[str]) -> None:
    if API_TOKEN is None:
        return
    if token != API_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid token")


def resolve_watch_folder(folder_override: Optional[str]) -> Path:
    folder = Path(folder_override) if folder_override else WATCH_FOLDER
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Folder does not exist: {folder}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Path is not a folder: {folder}")
    return folder


def safe_child_path(base: Path, name: str) -> Path:
    """
    Resolve a file path under base and reject path traversal.
    """
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid file name.")

    candidate = (base / name).resolve()
    base_resolved = base.resolve()

    try:
        candidate.relative_to(base_resolved)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Path escapes monitored directory.") from exc

    return candidate


def parse_time_value(value: Optional[str], field_name: str) -> Optional[float]:
    """
    Accept either:
    - unix timestamp as string, e.g. "1775620000"
    - ISO datetime, e.g. "2026-04-08T12:30:00"

    Returns unix timestamp as float.
    """
    if value is None:
        return None

    text = value.strip()
    if not text:
        return None

    # Try unix timestamp first
    try:
        return float(text)
    except ValueError:
        pass

    # Then ISO datetime
    try:
        dt = datetime.fromisoformat(text)
        return dt.timestamp()
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Invalid {field_name}. Use unix timestamp or ISO datetime, "
                f"for example 1775620000 or 2026-04-08T12:30:00"
            ),
        ) from exc


def list_files_in_folder(
    folder: Path,
    pattern: Optional[str] = None,
    modified_after: Optional[float] = None,
    modified_before: Optional[float] = None,
) -> list[FileEntry]:
    entries: list[FileEntry] = []

    for p in folder.iterdir():
        if not p.is_file():
            continue
        if pattern and not fnmatch.fnmatch(p.name, pattern):
            continue

        stat = p.stat()
        mtime = stat.st_mtime

        if modified_after is not None and mtime < modified_after:
            continue
        if modified_before is not None and mtime > modified_before:
            continue

        entries.append(
            FileEntry(
                name=p.name,
                size_bytes=stat.st_size,
                modified_unix=mtime,
                modified_iso=datetime.fromtimestamp(mtime).isoformat(),
            )
        )

    entries.sort(key=lambda x: x.modified_unix, reverse=True)
    return entries


@app.get("/files")
def files(
    token: Optional[str] = Query(default=None),
    watch_folder: Optional[str] = Query(default=None),
    pattern: Optional[str] = Query(default=None),
    modified_after: Optional[str] = Query(
        default=None,
        description="Only include files modified at or after this time. Unix timestamp or ISO datetime.",
    ),
    modified_before: Optional[str] = Query(
        default=None,
        description="Only include files modified at or before this time. Unix timestamp or ISO datetime.",
    ),
) -> dict[str, Any]:
    """
    List files in the monitored directory.

    Optional filters:
    - pattern=*.png
    - modified_after=2026-04-08T12:30:00
    - modified_before=2026-04-08T13:00:00

    You can also use unix timestamps.
    """
    check_token(token)
    folder = resolve_watch_folder(watch_folder)

    after_ts = parse_time_value(modified_after, "modified_after")
    before_ts = parse_time_value(modified_before, "modified_before")

    if after_ts is not None and before_ts is not None and after_ts > before_ts:
        raise HTTPException(
            status_code=400,
            detail="modified_after cannot be later than modified_before.",
        )

    entries = list_files_in_folder(
        folder,
        pattern=pattern,
        modified_after=after_ts,
        modified_before=before_ts,
    )

    return {
        "ok": True,
        "watch_folder": str(folder),
        "count": len(entries),
        "filters": {
            "pattern": pattern,
            "modified_after": after_ts,
            "modified_before": before_ts,
        },
        "files": [entry.model_dump() for entry in entries],
    }

def snapshot_matching_files(folder: Path, pattern: str) -> set[Path]:
    if not folder.exists():
        return set()
    return {
        p.resolve()
        for p in folder.iterdir()
        if p.is_file() and fnmatch.fnmatch(p.name, pattern)
    }


def wait_for_new_files(
    folder: Path,
    pattern: str,
    count_needed: int,
    timeout_sec: float,
    baseline_files: set[Path],
) -> list[Path]:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        current = snapshot_matching_files(folder, pattern)
        new_files = sorted(current - baseline_files, key=lambda p: p.stat().st_mtime)
        if len(new_files) >= count_needed:
            return new_files[:count_needed]
        time.sleep(POLL_INTERVAL_SEC)
    return []


def connect_to_window():
    app_obj = Application(backend=BACKEND).connect(
        title_re=WINDOW_TITLE_REGEX if WINDOW_TITLE_REGEX else None,
        title=WINDOW_TITLE if WINDOW_TITLE else None,
    )

    if WINDOW_TITLE_REGEX:
        window = app_obj.window(title_re=WINDOW_TITLE_REGEX)
    elif WINDOW_TITLE:
        window = app_obj.window(title=WINDOW_TITLE)
    else:
        raise RuntimeError("No window selector configured.")

    return app_obj, window


def resolve_button(window):
    if BUTTON_AUTOMATION_ID:
        try:
            return window.child_window(
                auto_id=BUTTON_AUTOMATION_ID,
                control_type=BUTTON_CONTROL_TYPE,
            )
        except Exception:
            pass

    if BUTTON_TITLE:
        try:
            return window.child_window(
                title=BUTTON_TITLE,
                control_type=BUTTON_CONTROL_TYPE,
            )
        except Exception:
            pass

    if BUTTON_BEST_MATCH:
        try:
            return window[BUTTON_BEST_MATCH]
        except Exception:
            pass

    raise ElementNotFoundError("Could not find the capture button.")



def perform_capture_click() -> None:
    _, window = connect_to_window()

    logger.info("Connected to target window.")

    # For UIA backend, restore() is not always reliable, so ignore failures.
    try:
        window.set_focus()
    except Exception:
        try:
            window.minimize()
            time.sleep(0.2)
            window.maximize()
            time.sleep(0.3)
            window.set_focus()
        except Exception as exc:
            raise RuntimeError(f"Unable to focus target window: {exc}") from exc

    time.sleep(PRE_CLICK_DELAY_SEC)

    spec = resolve_button(window)

    # Resolve WindowSpecification -> wrapper
    try:
        button = spec.wrapper_object()
    except Exception as exc:
        raise ElementNotFoundError(f"Could not resolve capture button wrapper: {exc}") from exc

    # Existence/visibility checks
    try:
        spec.wait("exists", timeout=10)
    except Exception as exc:
        raise RuntimeError(f"Capture button does not exist: {exc}") from exc

    # Debug info
    try:
        rect = button.rectangle()
        logger.info(
            "Button resolved: class=%s, rect=(%d,%d,%d,%d), visible=%s, enabled=%s",
            button.friendly_class_name(),
            rect.left, rect.top, rect.right, rect.bottom,
            button.is_visible(),
            button.is_enabled(),
        )
    except Exception:
        logger.info("Button resolved, but failed to read debug properties.")

    # Attempt 1: normal click_input()
    try:
        button.click_input()
        logger.info("Capture button clicked with click_input().")
        time.sleep(POST_CLICK_DELAY_SEC)
        return
    except Exception as exc:
        logger.warning("click_input() failed: %r", exc)

    # Attempt 2: invoke/select/click if available
    for method_name in ("invoke", "select", "click"):
        method = getattr(button, method_name, None)
        if callable(method):
            try:
                method()
                logger.info("Capture button activated with %s().", method_name)
                time.sleep(POST_CLICK_DELAY_SEC)
                return
            except Exception as exc:
                logger.warning("%s() failed: %r", method_name, exc)

    # Attempt 3: coordinate click on the center of the element rectangle
    try:
        rect = button.rectangle()
        if rect.width() <= 0 or rect.height() <= 0:
            raise RuntimeError("Button rectangle is empty.")
        center = rect.mid_point()
        mouse.click(coords=(center.x, center.y))
        logger.info("Capture button clicked by screen coordinates at (%d, %d).", center.x, center.y)
        time.sleep(POST_CLICK_DELAY_SEC)
        return
    except Exception as exc:
        raise RuntimeError(f"All button click methods failed. Last error: {exc}") from exc

def run_capture(
    request_id: Optional[str],
    watch_folder: Path,
    expected_glob: str,
    expected_count: int,
    timeout_sec: float,
) -> CaptureResponse:
    global last_status

    started = time.time()

    baseline_files = snapshot_matching_files(watch_folder, expected_glob)
    logger.info(
        "Capture started. request_id=%s folder=%s pattern=%s count=%d baseline=%d",
        request_id, watch_folder, expected_glob, expected_count, len(baseline_files)
    )

    try:
        perform_capture_click()
    except ElementNotFoundError as exc:
        ended = time.time()
        resp = CaptureResponse(
            ok=False,
            message=f"UI element not found: {exc}",
            request_id=request_id,
            started_at_unix=started,
            ended_at_unix=ended,
            elapsed_sec=ended - started,
            files_found=[],
            watch_folder=str(watch_folder),
            expected_glob=expected_glob,
            expected_count=expected_count,
        )
        last_status = LastStatus(**resp.model_dump())
        return resp
    except Exception as exc:
        ended = time.time()
        resp = CaptureResponse(
            ok=False,
            message=f"GUI automation failed: {exc}",
            request_id=request_id,
            started_at_unix=started,
            ended_at_unix=ended,
            elapsed_sec=ended - started,
            files_found=[],
            watch_folder=str(watch_folder),
            expected_glob=expected_glob,
            expected_count=expected_count,
        )
        last_status = LastStatus(**resp.model_dump())
        return resp

    found_files = wait_for_new_files(
        folder=watch_folder,
        pattern=expected_glob,
        count_needed=expected_count,
        timeout_sec=timeout_sec,
        baseline_files=baseline_files,
    )

    ended = time.time()
    ok = len(found_files) >= expected_count
    message = (
        "Capture success."
        if ok
        else f"Timed out waiting for {expected_count} new file(s) matching {expected_glob}."
    )

    resp = CaptureResponse(
        ok=ok,
        message=message,
        request_id=request_id,
        started_at_unix=started,
        ended_at_unix=ended,
        elapsed_sec=ended - started,
        files_found=[str(p) for p in found_files],
        watch_folder=str(watch_folder),
        expected_glob=expected_glob,
        expected_count=expected_count,
    )
    last_status = LastStatus(**resp.model_dump())
    return resp


# =========================
# API routes
# =========================

@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "service": "capture_server",
        "busy": capture_lock.locked(),
    }


@app.get("/last-status")
def get_last_status() -> dict[str, Any]:
    if last_status is None:
        return {"ok": True, "last_status": None}
    return {"ok": True, "last_status": asdict(last_status)}


@app.post("/capture", response_model=CaptureResponse)
def capture(req: CaptureRequest) -> CaptureResponse:
    check_token(req.token)

    if not capture_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Capture already in progress.")

    try:
        watch_folder = resolve_watch_folder(req.watch_folder)
        expected_glob = req.expected_glob or EXPECTED_FILE_GLOB
        expected_count = req.expected_count or EXPECTED_FILE_COUNT
        timeout_sec = req.timeout_sec or CAPTURE_TIMEOUT_SEC

        return run_capture(
            request_id=req.request_id,
            watch_folder=watch_folder,
            expected_glob=expected_glob,
            expected_count=expected_count,
            timeout_sec=timeout_sec,
        )
    finally:
        capture_lock.release()


@app.get("/files")
def files(
    token: Optional[str] = Query(default=None),
    watch_folder: Optional[str] = Query(default=None),
    pattern: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    """
    List files in the monitored directory.
    Optional pattern example: *.png
    """
    check_token(token)
    folder = resolve_watch_folder(watch_folder)
    entries = list_files_in_folder(folder, pattern=pattern)
    return {
        "ok": True,
        "watch_folder": str(folder),
        "count": len(entries),
        "files": [entry.model_dump() for entry in entries],
    }


@app.get("/pull")
def pull_file(
    name: str = Query(..., description="File name inside the monitored directory"),
    token: Optional[str] = Query(default=None),
    watch_folder: Optional[str] = Query(default=None),
):
    """
    Download a file from the monitored directory.
    Example:
        GET /pull?name=img001.png
    """
    check_token(token)
    folder = resolve_watch_folder(watch_folder)
    file_path = safe_child_path(folder, name)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Requested path is not a file.")

    media_type, _ = mimetypes.guess_type(str(file_path))
    if media_type is None:
        media_type = "application/octet-stream"

    return FileResponse(
        path=str(file_path),
        media_type=media_type,
        filename=file_path.name,
    )


@app.delete("/files/{name}")
def delete_file(
    name: str,
    token: Optional[str] = Query(default=None),
    watch_folder: Optional[str] = Query(default=None),
) -> dict[str, Any]:
    """
    Optional cleanup endpoint.
    """
    check_token(token)
    folder = resolve_watch_folder(watch_folder)
    file_path = safe_child_path(folder, name)

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    if not file_path.is_file():
        raise HTTPException(status_code=400, detail="Requested path is not a file.")

    file_path.unlink()
    return {
        "ok": True,
        "deleted": file_path.name,
        "watch_folder": str(folder),
    }


# =========================
# Main
# =========================

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting server on %s:%d", HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)
