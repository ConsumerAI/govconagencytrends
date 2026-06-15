from __future__ import annotations

import mimetypes
import uuid
from datetime import datetime, timezone
from pathlib import Path

from extraction.config import RUN_ID_PREFIX, source_storage_key
from extraction.ingest.hash import sha256_hex
from extraction.ingest.store import normalize_storage_ext, store_bytes_to_run_sources
from extraction.types import Finding, IngestResult, ProcessSourcesV1, RunManifest, SourceRecord


def create_run_id() -> str:
    return f"{RUN_ID_PREFIX}{uuid.uuid4()}"


def _file_ext(path: Path) -> str:
    ext = path.suffix.lstrip(".").lower()
    return ext if ext else "bin"


def _guess_mime(path: Path) -> str | None:
    mime, _ = mimetypes.guess_type(str(path))
    return mime


def ingest_files(
    file_paths: list[Path],
    run_id: str | None = None,
) -> IngestResult:
    """Copy/hash uploaded files into the run sources folder."""
    resolved_run_id = run_id or create_run_id()
    findings: list[Finding] = []
    sources: list[SourceRecord] = []
    seen_hashes: dict[str, SourceRecord] = {}

    if not file_paths:
        findings.append(Finding("error", "INGEST_NO_FILES", "No input files were provided."))
        return IngestResult(run_id=resolved_run_id, sources=[], findings=findings)

    for file_path in file_paths:
        path = file_path.resolve()
        if not path.exists():
            findings.append(
                Finding(
                    "error",
                    "INGEST_FILE_NOT_FOUND",
                    f"File not found: {path}",
                    {"path": str(path)},
                )
            )
            continue
        if not path.is_file():
            findings.append(
                Finding(
                    "error",
                    "INGEST_NOT_A_FILE",
                    f"Not a regular file: {path}",
                    {"path": str(path)},
                )
            )
            continue

        data = path.read_bytes()
        if not data:
            findings.append(
                Finding(
                    "warn",
                    "INGEST_EMPTY_FILE",
                    f"Skipping empty file: {path.name}",
                    {"path": str(path)},
                )
            )
            continue

        content_hash = sha256_hex(data)
        ext = normalize_storage_ext(_file_ext(path))
        did_dedupe = content_hash in seen_hashes

        if did_dedupe:
            existing = seen_hashes[content_hash]
            sources.append(
                SourceRecord(
                    key=existing.key,
                    ext=existing.ext,
                    bytes=len(data),
                    sha256=content_hash,
                    original_filename=path.name,
                    abs_path=existing.abs_path,
                    did_dedupe=True,
                )
            )
            findings.append(
                Finding(
                    "info",
                    "INGEST_DEDUPE",
                    f"Duplicate content skipped: {path.name}",
                    {"sha256": content_hash, "original_filename": path.name},
                )
            )
            continue

        stored_path = store_bytes_to_run_sources(resolved_run_id, content_hash, data, ext)
        key = source_storage_key(resolved_run_id, content_hash, ext)
        record = SourceRecord(
            key=key,
            ext=ext,
            bytes=len(data),
            sha256=content_hash,
            original_filename=path.name,
            abs_path=str(stored_path),
            did_dedupe=False,
        )
        seen_hashes[content_hash] = record
        sources.append(record)

    sources.sort(key=lambda s: s.sha256)
    if not sources:
        findings.append(Finding("error", "INGEST_NO_SOURCES_STORED", "No sources were stored for this run."))

    return IngestResult(run_id=resolved_run_id, sources=sources, findings=findings)


def build_process_sources_v1(run_id: str, sources: list[SourceRecord]) -> ProcessSourcesV1:
    return ProcessSourcesV1(
        version=1,
        run_id=run_id,
        sources=[s.to_manifest_entry() for s in sources],
    )


def build_run_manifest(run_id: str, sources: list[SourceRecord]) -> RunManifest:
    created_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return RunManifest(
        version=1,
        run_id=run_id,
        created_at=created_at,
        sources=[s.to_manifest_entry() for s in sources],
    )


def process_sources_base_signal(process_sources: ProcessSourcesV1) -> dict:
    """Shape used later as process_sources_v1 resolved signal canonical_value."""
    import json

    meta_json = json.dumps(process_sources.to_dict(), indent=2)
    return {
        "id": "process_sources_v1",
        "value": meta_json,
        "confidence": "high",
        "evidence": [],
        "findings": [],
    }
