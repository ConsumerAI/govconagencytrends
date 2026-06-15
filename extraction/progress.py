from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Callable

PROGRESS_ENV = "GOVCON_EXTRACTION_PROGRESS_FILE"

ProgressCallback = Callable[[str, float], None]


def _replace_progress_file(tmp: Path, target: Path) -> None:
    for attempt in range(6):
        try:
            tmp.replace(target)
            return
        except PermissionError:
            if attempt == 5:
                break
            time.sleep(0.05)
    tmp.unlink(missing_ok=True)


def progress_file_from_env() -> Path | None:
    value = os.getenv(PROGRESS_ENV, "").strip()
    return Path(value) if value else None


def write_extraction_progress(
    label: str,
    pct: float,
    *,
    doc_index: int | None = None,
    doc_total: int | None = None,
    filename: str | None = None,
    phase: str | None = None,
    path: Path | None = None,
) -> None:
    target = path or progress_file_from_env()
    if not target:
        return
    payload = {
        "label": label,
        "pct": max(0.0, min(1.0, float(pct))),
        "docIndex": doc_index,
        "docTotal": doc_total,
        "filename": filename,
        "phase": phase,
        "updatedAt": time.time(),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(f"{target.suffix}.tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    _replace_progress_file(tmp, target)


def read_extraction_progress(path: Path) -> dict[str, Any] | None:
    try:
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def merge_progress_callbacks(*callbacks: ProgressCallback | None) -> ProgressCallback | None:
    active = [callback for callback in callbacks if callback]
    if not active:
        return None

    def merged(label: str, pct: float) -> None:
        for callback in active:
            callback(label, pct)

    return merged


def file_progress_callback(path: Path) -> ProgressCallback:
    latest: dict[str, Any] = {}

    def callback(label: str, pct: float) -> None:
        latest["label"] = label
        latest["pct"] = pct
        write_extraction_progress(
            label,
            pct,
            doc_index=latest.get("docIndex"),
            doc_total=latest.get("docTotal"),
            filename=latest.get("filename"),
            phase=latest.get("phase"),
            path=path,
        )

    callback.set_meta = lambda **kwargs: latest.update({k: v for k, v in kwargs.items() if v is not None})  # type: ignore[attr-defined]
    return callback


def emit_progress(
    progress: ProgressCallback | None,
    label: str,
    pct: float,
    *,
    doc_index: int | None = None,
    doc_total: int | None = None,
    filename: str | None = None,
    phase: str | None = None,
) -> None:
    write_extraction_progress(
        label,
        pct,
        doc_index=doc_index,
        doc_total=doc_total,
        filename=filename,
        phase=phase,
    )
    if progress:
        progress(label, pct)
