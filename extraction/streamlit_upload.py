from __future__ import annotations

import hashlib
import re
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from extraction.config import get_data_dir, get_extraction_mode, load_project_env, resolved_signals_json_path
from extraction.ingest.hash import sha256_hex
from extraction.ingest.upload import ingest_files
from extraction.persist import read_json
from extraction.streamlit_isolated import run_extraction_subprocess
from extraction.types import Finding, PipelineResult

ProgressCallback = Callable[[str, float], None]

ALLOWED_UPLOAD_EXTENSIONS = frozenset({"pdf", "docx", "xlsx", "xls", "csv", "txt"})
REJECTED_UPLOAD_EXTENSIONS = frozenset({"exe", "dll", "bat", "cmd", "ps1", "sh", "js", "zip", "rar", "7z", "tar", "gz"})
MAX_SINGLE_FILE_BYTES = 50 * 1024 * 1024
MAX_PACKAGE_BYTES = 200 * 1024 * 1024


@dataclass
class UploadValidation:
    paths: list[Path] = field(default_factory=list)
    findings: list[Finding] = field(default_factory=list)
    fingerprint: str = ""
    total_bytes: int = 0
    file_types: dict[str, int] = field(default_factory=dict)


@dataclass
class ExtractionUIResult:
    success: bool
    pipeline: PipelineResult | None
    resolved_signals: dict[str, Any] | None
    summary: dict[str, Any]
    findings: list[Finding] = field(default_factory=list)
    error_message: str | None = None


def sanitize_upload_filename(name: str) -> str:
    base = Path(name).name
    cleaned = re.sub(r"[^A-Za-z0-9._\- ]+", "_", base).strip("._ ")
    return cleaned or "upload.bin"


def ocr_environment_status() -> dict[str, Any]:
    try:
        from extraction.ocr.ocr_fallback import _tesseract_available

        tesseract = _tesseract_available()
    except ImportError:
        tesseract = False
    pypdfium2 = False
    try:
        import pypdfium2  # noqa: F401

        pypdfium2 = True
    except ImportError:
        pypdfium2 = False
    return {
        "tesseractAvailable": tesseract,
        "pypdfium2Available": pypdfium2,
        "ocrReady": tesseract and pypdfium2,
    }


def validate_and_stage_uploads(uploaded_files: list[Any]) -> UploadValidation:
    result = UploadValidation()
    if not uploaded_files:
        result.findings.append(Finding("error", "UPLOAD_NO_FILES", "No files were uploaded."))
        return result

    temp_root = Path(tempfile.mkdtemp(prefix="govcon-upload-"))
    fingerprint_parts: list[str] = []

    for uploaded in uploaded_files:
        raw_name = sanitize_upload_filename(getattr(uploaded, "name", "upload.bin"))
        ext = Path(raw_name).suffix.lstrip(".").lower()
        if ext in REJECTED_UPLOAD_EXTENSIONS:
            result.findings.append(Finding("error", "UPLOAD_REJECTED_TYPE", f"Rejected file type: {raw_name}"))
            continue
        if ext not in ALLOWED_UPLOAD_EXTENSIONS:
            result.findings.append(Finding("error", "UPLOAD_UNSUPPORTED_TYPE", f"Unsupported file type: {raw_name}"))
            continue
        data = uploaded.getvalue()
        if len(data) > MAX_SINGLE_FILE_BYTES:
            result.findings.append(Finding("error", "UPLOAD_FILE_TOO_LARGE", f"File exceeds size limit: {raw_name}"))
            continue
        if result.total_bytes + len(data) > MAX_PACKAGE_BYTES:
            result.findings.append(Finding("error", "UPLOAD_PACKAGE_TOO_LARGE", "Total upload package exceeds size limit."))
            break
        dest = temp_root / raw_name
        dest.write_bytes(data)
        result.paths.append(dest)
        result.total_bytes += len(data)
        result.file_types[ext] = result.file_types.get(ext, 0) + 1
        fingerprint_parts.append(f"{raw_name}:{sha256_hex(data)}")

    result.fingerprint = hashlib.sha256("|".join(sorted(fingerprint_parts)).encode()).hexdigest()
    if not result.paths and not any(f.level == "error" for f in result.findings):
        result.findings.append(Finding("error", "UPLOAD_NO_VALID_FILES", "No valid files remained after validation."))
    return result


def summarize_pipeline_result(pipeline: PipelineResult, resolved: dict[str, Any] | None) -> dict[str, Any]:
    findings = pipeline.findings
    prominent_codes = {
        "SOLICITATION_IDENTITY_AMBIGUOUS",
        "SIGNAL_CONFLICT",
        "NAICS_CONFLICT",
        "PSC_CONFLICT",
        "DOCUMENT_NEEDS_OCR",
        "OCR_FALLBACK_NOT_AVAILABLE",
        "OCR_FALLBACK_EMPTY",
        "DOCUMENT_PDF_PARSE_FAILED",
        "SOLICITATION_SOURCE_EXCLUDED",
        "CONTROLLING_DOCUMENT_UNREADABLE",
        "OCR_FEEDBACK_RERUN",
        "OPENAI_API_KEY_MISSING",
        "PACKAGE_CACHE_HIT",
        "EXTRACTION_BUILD_INFO",
    }
    prominent = [f for f in findings if f.code in prominent_codes or f.level == "error"]
    signals = (resolved or {}).get("signals") or []
    summary_meta = (resolved or {}).get("summary") or {}
    profile = str(summary_meta.get("profile") or summary_meta.get("extractionProfile") or "package_llm_full")
    requested_count = int(summary_meta.get("requestedSignalCount") or 0)
    total_signals = requested_count or int(summary_meta.get("total_signal_ids") or len(signals) or 0)
    countable = [
        s
        for s in signals
        if str(s.get("id") or "") != "process_sources_v1"
    ]
    confirmed_statuses = {
        "validated_model_extraction",
        "validated_model_extraction_with_alternates",
        "passthrough",
        "canonical_explicit",
        "resolved_equivalent_candidates",
    }
    review_statuses = {
        "review_required",
        "withheld_evidence_mismatch",
        "withheld_semantic_validation",
        "unresolved_conflict",
        "resolved_with_conflict",
    }
    confirmed = sum(
        1
        for s in countable
        if str(s.get("resolution_status") or "") in confirmed_statuses and s.get("canonical_value") is not None
    )
    review = sum(1 for s in countable if str(s.get("resolution_status") or "") in review_statuses)
    not_found = sum(1 for s in countable if str(s.get("resolution_status") or "") == "not_found")
    other = max(0, len(countable) - confirmed - review - not_found)
    sol_number = next(
        (
            s.get("canonical_value")
            for s in signals
            if str(s.get("id") or "") in {"rfp_solicitation_id_v1", "rfp_solicitation_number_v1"}
            and s.get("canonical_value")
        ),
        None,
    )
    excluded_files = [
        f.to_dict()
        for f in findings
        if f.code == "SOLICITATION_SOURCE_EXCLUDED" or f.code == "UPLOAD_UNSUPPORTED_TYPE" or f.code == "UPLOAD_REJECTED_TYPE"
    ]
    package_diagnostics = {}
    try:
        from extraction.persist import read_json as _read_json
        from extraction.config import diagnostics_path

        diag = _read_json(diagnostics_path(pipeline.run_id))
        package_diagnostics = diag.get("packageDiagnostics") or {}
    except Exception:
        package_diagnostics = {}
    return {
        "runId": pipeline.run_id,
        "elapsedSec": pipeline.elapsed_sec,
        "documentCount": pipeline.document_count,
        "amendmentsDetected": pipeline.amendments_detected,
        "resolvedSignalCount": pipeline.resolved_signal_count,
        "confirmedFilterSignals": confirmed,
        "unresolvedConflicts": pipeline.unresolved_amendment_conflicts,
        "solicitationNumber": sol_number,
        "prominentFindings": [f.to_dict() for f in prominent[:12]],
        "excludedFiles": excluded_files,
        "optionalFieldsMissingCount": not_found,
        "citationReviewSignals": review,
        "signalStatusSummary": {
            "confirmed": confirmed,
            "reviewRequired": review,
            "notFound": not_found,
            "other": other,
            "totalSignals": len(countable),
            "extractionProfile": profile,
        },
        "confirmedSignals": confirmed,
        "reviewRequiredSignals": review,
        "ocrEnvironment": ocr_environment_status(),
        "ocrDocumentsRequiring": pipeline.ocr_documents_requiring,
        "ocrPagesProcessed": pipeline.ocr_pages_processed,
        "ocrDocumentsRerun": pipeline.ocr_documents_rerun,
        "ocrFailures": pipeline.ocr_failures,
        "controllingDocumentsUnreadable": pipeline.controlling_documents_unreadable,
        "ocrDiagnostics": pipeline.ocr_diagnostics,
        "packageCacheHit": bool((package_diagnostics or {}).get("cacheHit")),
        "packageDiagnostics": package_diagnostics,
        "extractionProfile": profile,
        "marketScopeFast": profile == "market_scope_fast",
        "totalSignalCount": requested_count or total_signals,
        "blockingFailure": any(
            f.code
            in {
                "OCR_FALLBACK_EMPTY",
                "DOCUMENT_PDF_PARSE_FAILED",
                "SOLICITATION_IDENTITY_AMBIGUOUS",
                "CONTROLLING_DOCUMENT_UNREADABLE",
                "OPENAI_API_KEY_MISSING",
            }
            and f.level in {"error", "warning", "warn"}
            for f in findings
        ),
    }


def run_extraction_from_paths(
    file_paths: list[Path],
    *,
    progress: ProgressCallback | None = None,
    progress_detail: dict[str, Any] | None = None,
    force_refresh: bool = False,
) -> ExtractionUIResult:
    file_count = len(file_paths)
    started_at = time.time()
    try:
        load_project_env()
        extraction_mode = get_extraction_mode()
        if progress:
            progress("Reading documents", 0.05)
        if extraction_mode == "package_llm":
            if progress:
                progress("Reading documents", 0.02)
            pipeline, _cli_summary = run_extraction_subprocess(
                file_paths,
                force_refresh=force_refresh,
                progress=progress,
                progress_detail=progress_detail,
            )
            if progress:
                progress("Matching dashboard filters", 0.95)
        else:
            from extraction.pipeline import run_ingest_pipeline

            pipeline = run_ingest_pipeline(
                file_paths,
                extract_text=True,
                extract_signals=True,
                resolve_signals=True,
                progress=progress,
                force_refresh=force_refresh,
            )
            if progress:
                progress("Matching dashboard filters", 0.95)

        resolved_path = resolved_signals_json_path(pipeline.run_id)
        resolved = read_json(resolved_path) if resolved_path.exists() else None
        summary = summarize_pipeline_result(pipeline, resolved if isinstance(resolved, dict) else None)
        summary["extractionMode"] = extraction_mode
        summary["isolatedSubprocess"] = extraction_mode == "package_llm"
        success = bool(resolved) and pipeline.resolved_signal_count > 0 and not summary.get("blockingFailure")
        return ExtractionUIResult(
            success=success,
            pipeline=pipeline,
            resolved_signals=resolved if isinstance(resolved, dict) else None,
            summary=summary,
            findings=pipeline.findings,
            error_message=None if success else "Extraction completed with blocking issues; review warnings before applying filters.",
        )
    except Exception as exc:
        salvaged = _latest_completed_resolved_artifact(started_at)
        if salvaged:
            run_id, resolved = salvaged
            summary = {
                "runId": run_id,
                "documentCount": file_count,
                "recoveredFromPostExtractionFailure": True,
                "postExtractionFailure": f"{type(exc).__name__}: {exc}",
            }
            return ExtractionUIResult(
                success=True,
                pipeline=None,
                resolved_signals=resolved,
                summary=summary,
                findings=[Finding("warn", "POST_EXTRACTION_HANDOFF_RECOVERED", str(exc))],
                error_message=None,
            )
        return ExtractionUIResult(
            success=False,
            pipeline=None,
            resolved_signals=None,
            summary={"documentCount": file_count},
            findings=[Finding("error", "EXTRACTION_FAILED", str(exc))],
            error_message=str(exc),
        )


def _latest_completed_resolved_artifact(started_at: float) -> tuple[str, dict[str, Any]] | None:
    runs_dir = get_data_dir() / "runs"
    if not runs_dir.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for path in runs_dir.glob("run_*/signals/resolved_signals.json"):
        try:
            modified = path.stat().st_mtime
        except OSError:
            continue
        if modified >= started_at - 2:
            candidates.append((modified, path))
    for _modified, path in sorted(candidates, reverse=True):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if isinstance(payload, dict) and payload.get("signals"):
            return path.parents[1].name, payload
    return None


def cleanup_staged_paths(paths: list[Path]) -> None:
    for path in paths:
        parent = path.parent
        if parent.name.startswith("govcon-upload-") and parent.exists():
            shutil.rmtree(parent, ignore_errors=True)
            break
