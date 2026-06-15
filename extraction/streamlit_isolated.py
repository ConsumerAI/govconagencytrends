from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from extraction.config import (
    diagnostics_path,
    get_data_dir,
    get_extraction_subprocess_timeout_sec,
    get_project_root,
    load_project_env,
    resolved_signals_json_path,
)
from extraction.persist import read_json
from extraction.progress import PROGRESS_ENV, ProgressCallback, read_extraction_progress
from extraction.types import Finding, PipelineResult

RESOLVED_SIGNALS_VERSION = "resolved_signals.v1"


def _finding_from_dict(payload: dict[str, Any]) -> Finding:
    return Finding(
        str(payload.get("level") or "info"),
        str(payload.get("code") or ""),
        str(payload.get("message") or ""),
        payload.get("details") if isinstance(payload.get("details"), dict) else None,
    )


def _pipeline_result_from_cli_summary(summary: dict[str, Any]) -> PipelineResult:
    run_id = str(summary.get("runId") or "")
    diagnostics = read_json(diagnostics_path(run_id)) if run_id else {}
    extra = diagnostics if isinstance(diagnostics, dict) else {}
    findings = [_finding_from_dict(item) for item in summary.get("findings") or [] if isinstance(item, dict)]
    package_diagnostics = extra.get("packageDiagnostics") if isinstance(extra.get("packageDiagnostics"), dict) else {}
    return PipelineResult(
        run_id=run_id,
        manifest_path=str(summary.get("manifest") or ""),
        diagnostics_path=str(summary.get("diagnostics") or diagnostics_path(run_id)),
        process_sources_path=str(summary.get("processSourcesSignal") or ""),
        corpus_path=summary.get("corpusV1"),
        full_text_path=summary.get("fullCorpusText"),
        signals_path=summary.get("signalsJson"),
        signal_count=int(summary.get("rawSignalCount") or 0),
        resolved_signals_path=summary.get("resolvedSignalsJson"),
        resolved_signal_count=int(summary.get("resolvedSignalCount") or 0),
        alternates_count=int(summary.get("alternatesCount") or 0),
        derived_signal_count=int(summary.get("derivedSignalCount") or 0),
        model_used=summary.get("modelUsed"),
        elapsed_sec=float(summary.get("elapsedSec") or 0.0),
        corpus_char_count=summary.get("corpusCharCount"),
        findings=findings,
        solicitation_set_detected=bool(summary.get("solicitationSetDetected")),
        base_solicitation_filename=summary.get("baseSolicitationFilename"),
        amendments_detected=list(summary.get("amendmentsDetectedInOrder") or []),
        document_count=int(summary.get("documentCount") or 0),
        superseded_candidate_count=int(summary.get("supersededCandidateCount") or 0),
        unresolved_amendment_conflicts=0,
        docset_manifest_path=summary.get("docsetManifest"),
        ocr_documents_requiring=int((package_diagnostics or {}).get("ocrDocumentsRequiring") or extra.get("ocrDocumentsRequiring") or 0),
        ocr_pages_processed=int((package_diagnostics or {}).get("ocrPagesProcessed") or extra.get("ocrPagesProcessed") or 0),
        ocr_documents_rerun=int(extra.get("ocrDocumentsRerun") or 0),
        ocr_failures=int(extra.get("ocrFailures") or 0),
        controlling_documents_unreadable=int(extra.get("controllingDocumentsUnreadable") or 0),
        ocr_diagnostics=list(extra.get("ocrDiagnostics") or []),
    )


def _valid_resolved_payload(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != RESOLVED_SIGNALS_VERSION:
        return False
    signals = payload.get("signals")
    if not isinstance(signals, list) or not signals:
        return False
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    requested = int(summary.get("requestedSignalCount") or 0)
    if requested:
        countable = [item for item in signals if isinstance(item, dict) and item.get("id") != "process_sources_v1"]
        return len(countable) >= requested
    return True


def _latest_valid_resolved_artifact_since(started_at: float) -> tuple[str, Path, dict[str, Any]] | None:
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
        if _valid_resolved_payload(payload):
            return path.parents[1].name, path, payload
    return None


def _pipeline_result_from_resolved_artifact(run_id: str, path: Path, payload: dict[str, Any]) -> PipelineResult:
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    diagnostics = read_json(diagnostics_path(run_id)) if diagnostics_path(run_id).exists() else {}
    package_diagnostics = diagnostics.get("packageDiagnostics") if isinstance(diagnostics, dict) else {}
    countable = [item for item in payload.get("signals") or [] if isinstance(item, dict) and item.get("id") != "process_sources_v1"]
    return PipelineResult(
        run_id=run_id,
        manifest_path=str((get_data_dir() / "runs" / run_id / "manifest.json")),
        diagnostics_path=str(diagnostics_path(run_id)),
        resolved_signals_path=str(path),
        resolved_signal_count=int(summary.get("resolved_signal_count") or len(payload.get("signals") or [])),
        signal_count=len(countable),
        derived_signal_count=int(summary.get("derived_signal_count") or 0),
        alternates_count=int(summary.get("alternates_count") or 0),
        model_used=summary.get("model"),
        elapsed_sec=float((package_diagnostics or {}).get("totalElapsedSec") or 0.0),
        findings=[],
        document_count=int((package_diagnostics or {}).get("documentCount") or 0),
        ocr_documents_requiring=int((package_diagnostics or {}).get("ocrDocumentsRequiring") or 0),
        ocr_pages_processed=int((package_diagnostics or {}).get("ocrPagesProcessed") or 0),
    )


def _preserve_subprocess_logs(run_id: str | None, stdout_file: Path, stderr_file: Path, result_file: Path) -> None:
    if not run_id:
        return
    target_dir = get_data_dir() / "runs" / run_id / "package_extraction"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        if stdout_file.exists():
            (target_dir / "subprocess_stdout.log").write_text(stdout_file.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        if stderr_file.exists():
            (target_dir / "subprocess_stderr.log").write_text(stderr_file.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
        if result_file.exists():
            (target_dir / "subprocess_result.json").write_text(result_file.read_text(encoding="utf-8", errors="replace"), encoding="utf-8")
    except Exception:
        return


def run_extraction_subprocess(
    file_paths: list[Path],
    *,
    force_refresh: bool = False,
    profile: str | None = None,
    merge_run_id: str | None = None,
    progress: ProgressCallback | None = None,
    progress_detail: dict[str, Any] | None = None,
) -> tuple[PipelineResult, dict[str, Any]]:
    """Run extraction in a fresh Python process so Streamlit cannot use stale pipeline code."""
    load_project_env()
    from extraction.config import get_extraction_profile

    cmd = [sys.executable, "-m", "extraction.cli", "--full-extraction", *[str(path) for path in file_paths]]
    env = os.environ.copy()
    root = str(get_project_root())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env["GOVCON_SUBPROCESS_EXTRACTION"] = "1"

    fd, progress_path = tempfile.mkstemp(prefix="govcon-extraction-progress-", suffix=".json")
    os.close(fd)
    progress_file = Path(progress_path)
    env[PROGRESS_ENV] = str(progress_file)
    result_fd, result_path = tempfile.mkstemp(prefix="govcon-extraction-result-", suffix=".json")
    os.close(result_fd)
    result_file = Path(result_path)
    result_file.unlink(missing_ok=True)
    env["GOVCON_EXTRACTION_RESULT_FILE"] = str(result_file)
    stdout_fd, stdout_path = tempfile.mkstemp(prefix="govcon-extraction-stdout-", suffix=".log")
    stderr_fd, stderr_path = tempfile.mkstemp(prefix="govcon-extraction-stderr-", suffix=".log")
    os.close(stdout_fd)
    os.close(stderr_fd)
    stdout_file = Path(stdout_path)
    stderr_file = Path(stderr_path)

    if force_refresh:
        env["GOVCON_FORCE_PACKAGE_REFRESH"] = "1"
    if profile:
        env["GOVCON_EXTRACTION_PROFILE"] = profile
    elif get_extraction_profile():
        env.setdefault("GOVCON_EXTRACTION_PROFILE", get_extraction_profile())
    if merge_run_id:
        env["GOVCON_MERGE_RUN_ID"] = merge_run_id

    stdout_handle = stdout_file.open("w", encoding="utf-8")
    stderr_handle = stderr_file.open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, stdout=stdout_handle, stderr=stderr_handle, text=True, env=env, cwd=root)
    last_label = ""
    started = time.monotonic()
    timeout_sec = get_extraction_subprocess_timeout_sec()
    try:
        while proc.poll() is None:
            if result_file.exists() and result_file.stat().st_size > 0:
                if progress:
                    progress("Market scope extracted successfully", 1.0)
                _terminate_process_tree(proc)
                break
            completed_artifact = _latest_valid_resolved_artifact_since(started)
            if completed_artifact:
                if progress:
                    progress("Market scope extracted successfully", 1.0)
                _terminate_process_tree(proc)
                break
            if time.monotonic() - started > timeout_sec:
                if progress:
                    progress("Extraction subprocess timed out; loading any saved scope artifacts", 0.99)
                _terminate_process_tree(proc)
                if not (result_file.exists() and result_file.stat().st_size > 0):
                    raise TimeoutError(f"Extraction subprocess exceeded {timeout_sec:.0f}s timeout")
                break
            state = read_extraction_progress(progress_file)
            if state:
                if progress_detail is not None:
                    progress_detail.clear()
                    progress_detail.update(state)
                label = str(state.get("label") or "")
                pct = float(state.get("pct") or 0.0)
                if progress and label and (label != last_label or pct >= 0):
                    progress(label, pct)
                    last_label = label
            time.sleep(0.12)
    finally:
        stdout_handle.close()
        stderr_handle.close()
        progress_file.unlink(missing_ok=True)
    stdout = stdout_file.read_text(encoding="utf-8", errors="replace") if stdout_file.exists() else ""
    stderr = stderr_file.read_text(encoding="utf-8", errors="replace") if stderr_file.exists() else ""
    accepted_artifact = None if result_file.exists() and result_file.stat().st_size > 0 else _latest_valid_resolved_artifact_since(started)

    if proc.returncode not in {0, 1, None} and not (stdout or "").strip() and not accepted_artifact:
        raise RuntimeError((stderr or "").strip() or f"Extraction subprocess failed with code {proc.returncode}")
    if result_file.exists() and result_file.stat().st_size > 0:
        summary_text = result_file.read_text(encoding="utf-8")
        try:
            summary = json.loads(summary_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Extraction subprocess returned invalid JSON.") from exc
        if not isinstance(summary, dict):
            raise RuntimeError("Extraction subprocess returned invalid JSON.")
        run_id = str(summary.get("runId") or "")
        _preserve_subprocess_logs(run_id, stdout_file, stderr_file, result_file)
        stdout_file.unlink(missing_ok=True)
        stderr_file.unlink(missing_ok=True)
        result_file.unlink(missing_ok=True)
        return _pipeline_result_from_cli_summary(summary), summary
    else:
        completed_artifact = accepted_artifact
        if completed_artifact:
            run_id, resolved_path, resolved_payload = completed_artifact
            summary = {
                "runId": run_id,
                "resolvedSignalsJson": str(resolved_path),
                "resolvedSignalCount": len(resolved_payload.get("signals") or []),
                "rawSignalCount": len([item for item in resolved_payload.get("signals") or [] if isinstance(item, dict) and item.get("id") != "process_sources_v1"]),
                "findings": [
                    {
                        "level": "info",
                        "code": "RESOLVED_ARTIFACT_ACCEPTED",
                        "message": "Accepted resolved_signals.json before CLI result-file handoff completed.",
                    }
                ],
                "artifactBoundaryAccepted": True,
            }
            _preserve_subprocess_logs(run_id, stdout_file, stderr_file, result_file)
            stdout_file.unlink(missing_ok=True)
            stderr_file.unlink(missing_ok=True)
            result_file.unlink(missing_ok=True)
            return _pipeline_result_from_resolved_artifact(run_id, resolved_path, resolved_payload), summary
        summary_text = (stdout or "").strip()
    stdout_file.unlink(missing_ok=True)
    stderr_file.unlink(missing_ok=True)
    result_file.unlink(missing_ok=True)
    if not summary_text:
        raise RuntimeError((stderr or "").strip() or "Extraction subprocess produced no output.")
    summary = json.loads(summary_text)
    if not isinstance(summary, dict):
        raise RuntimeError("Extraction subprocess returned invalid JSON.")
    return _pipeline_result_from_cli_summary(summary), summary


def _terminate_process_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
        else:
            proc.kill()
    except Exception:
        proc.kill()
