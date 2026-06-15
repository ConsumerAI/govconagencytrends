from __future__ import annotations

from pathlib import Path

from extraction.config import sources_dir


def normalize_storage_ext(ext: str | None) -> str:
    trimmed = (ext or "").strip().lower().lstrip(".")
    if not trimmed or trimmed == "unknown":
        return "bin"
    return trimmed


def store_bytes_to_run_sources(
    run_id: str,
    content_hash: str,
    data: bytes,
    ext: str,
) -> Path:
    """Store bytes at data/runs/{run_id}/sources/{hash}.{ext} (exclusive create)."""
    directory = sources_dir(run_id)
    directory.mkdir(parents=True, exist_ok=True)
    storage_ext = normalize_storage_ext(ext)
    stored_path = directory / f"{content_hash}.{storage_ext}"
    if not stored_path.exists():
        stored_path.write_bytes(data)
    return stored_path.resolve()
