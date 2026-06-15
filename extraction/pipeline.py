from __future__ import annotations

import time
from pathlib import Path

from extraction.config import (
    corpus_v1_path,
    diagnostics_path,
    docset_manifest_path,
    documents_dir,
    full_corpus_text_path,
    get_extraction_mode,
    has_openai_api_key,
    manifest_path,
    package_corpus_text_path,
    process_sources_signal_path,
    resolved_signals_json_path,
    signals_json_path,
)
from extraction.corpus.build import build_corpus_v1, build_full_corpus_text
from extraction.diagnostics_events import artifact_written, pipeline_branch_completed, pipeline_branch_entered
from extraction.docset.run import run_docset_phase
from extraction.ingest.upload import (
    build_process_sources_v1,
    build_run_manifest,
    ingest_files,
    process_sources_base_signal,
)
from extraction.llm.openai_extract import extract_signals_from_corpus
from extraction.persist import read_json, write_json, write_text
from extraction.resolve.run import resolve_run_signals
from extraction.types import Finding, PipelineResult


def _run_legacy_extraction(
    ingest_result,
    *,
    extract_text: bool,
    extract_signals: bool,
    resolve_signals: bool,
    base_signal: dict,
    findings: list[Finding],
) -> tuple[
    Path | None,
    Path | None,
    str,
    int | None,
    Path | None,
    int,
    str | None,
    int,
    int,
    int,
    dict,
    object | None,
]:
    """Legacy deterministic/docset extraction path (GOVCON_EXTRACTION_MODE=legacy)."""
    findings.append(pipeline_branch_entered("legacy"))
    corpus_file: Path | None = None
    full_text_file: Path | None = None
    signals_file: Path | None = None
    full_text = ""
    corpus_char_count: int | None = None
    signal_count = 0
    resolved_signals_path: str | None = None
    resolved_signal_count = 0
    alternates_count = 0
    derived_signal_count = 0
    resolved_summary: dict = {}
    model_used: str | None = None
    docset_result = None
    active_source_count = sum(1 for source in ingest_result.sources if not source.did_dedupe)

    if extract_text and ingest_result.sources:
        corpus = build_corpus_v1(ingest_result.run_id, ingest_result.sources)
        findings.extend(corpus.findings)
        corpus_file = corpus_v1_path(ingest_result.run_id)
        write_json(corpus_file, corpus.to_dict())

        full_text, text_findings = build_full_corpus_text(
            ingest_result.run_id,
            ingest_result.sources,
            corpus,
        )
        findings.extend(text_findings)
        corpus_char_count = len(full_text)
        full_text_file = full_corpus_text_path(ingest_result.run_id)
        write_text(full_text_file, full_text + ("\n" if full_text else ""))

    if extract_signals:
        if not has_openai_api_key():
            findings.append(
                Finding(
                    "warn",
                    "OPENAI_API_KEY_MISSING",
                    "OPENAI_API_KEY is not set; LLM signal extraction skipped; deterministic extraction only.",
                )
            )
        if active_source_count == 0:
            findings.append(
                Finding("error", "SIGNAL_EXTRACTION_NO_SOURCES", "Signal extraction requires at least one source file.")
            )
        else:
            full_corpus_llm_signals: list[dict] = []
            if full_text.strip() and has_openai_api_key():
                fallback_source = (
                    ingest_result.sources[0].original_filename if ingest_result.sources else None
                )
                full_corpus_llm_signals, llm_findings, model_used = extract_signals_from_corpus(
                    ingest_result.run_id,
                    full_text,
                    fallback_source_file=fallback_source,
                    base_signals=[],
                )
                findings.extend(llm_findings)

            docset_result = run_docset_phase(
                ingest_result.run_id,
                ingest_result.sources,
                use_llm=has_openai_api_key(),
                full_corpus_llm_signals=full_corpus_llm_signals if full_corpus_llm_signals else None,
                base_signals=[base_signal],
            )
            findings.extend(docset_result.findings)

            merged_by_id: dict[str, dict] = {str(base_signal["id"]): base_signal}
            for signal in docset_result.merged_signals:
                signal_id = str(signal.get("id") or "").strip()
                if signal_id:
                    merged_by_id[signal_id] = signal
            merged_signals = sorted(merged_by_id.values(), key=lambda item: str(item.get("id") or ""))

            signals_file, signal_count = _write_signals_if_present(
                ingest_result.run_id,
                merged_signals,
                model_used,
                force=True,
            )

    if resolve_signals:
        if signal_count == 0 and signals_json_path(ingest_result.run_id).exists():
            signal_count = len(read_json(signals_json_path(ingest_result.run_id)))
        (
            resolved_signals_path,
            resolved_artifact,
            resolved_signal_count,
            alternates_count,
            derived_signal_count,
            resolved_summary,
        ) = _apply_resolution(ingest_result.run_id, findings, full_text)
        _ = resolved_artifact
        if resolved_signals_path:
            findings.append(
                artifact_written(
                    resolved_signals_path,
                    producer="legacy",
                    mode="legacy",
                )
            )

    findings.append(pipeline_branch_completed("legacy", elapsed_sec=0.0))
    return (
        corpus_file,
        full_text_file,
        full_text,
        corpus_char_count,
        signals_file,
        signal_count,
        resolved_signals_path,
        resolved_signal_count,
        alternates_count,
        derived_signal_count,
        resolved_summary,
        docset_result,
        model_used,
    )


def _run_package_llm_extraction(
    ingest_result,
    *,
    extract_signals: bool,
    resolve_signals: bool,
    base_signal: dict,
    findings: list[Finding],
    progress=None,
    force_refresh: bool = False,
) -> tuple[
    Path | None,
    Path | None,
    str,
    int | None,
    Path | None,
    int,
    str | None,
    int,
    int,
    int,
    dict,
    object | None,
    str | None,
]:
    from extraction.package_llm.run import run_package_llm_extraction

    findings.append(pipeline_branch_entered("package_llm"))
    corpus_file: Path | None = None
    full_text_file: Path | None = None
    signals_file: Path | None = None
    full_text = ""
    corpus_char_count: int | None = None
    signal_count = 0
    resolved_signals_path: str | None = None
    resolved_signal_count = 0
    alternates_count = 0
    derived_signal_count = 0
    resolved_summary: dict = {}
    model_used: str | None = None
    package_result = None

    if extract_signals and resolve_signals:
        package_result = run_package_llm_extraction(
            ingest_result.run_id,
            ingest_result.sources,
            process_sources_payload=base_signal,
            progress=progress,
            force_refresh=force_refresh,
        )
        findings.extend(package_result.findings)
        model_used = package_result.model_used
        signal_count = package_result.signal_count
        resolved_signal_count = package_result.resolved_signal_count
        alternates_count = package_result.alternates_count
        corpus_char_count = package_result.corpus_char_count
        resolved_summary = package_result.resolved_signals.get("summary") or {}
        resolved_signals_path = str(resolved_signals_json_path(ingest_result.run_id))
        signals_path = signals_json_path(ingest_result.run_id)
        signals_file = signals_path if signals_path.exists() else None
        corpus_path = package_corpus_text_path(ingest_result.run_id)
        if corpus_path.exists():
            full_text_file = corpus_path
            full_text = corpus_path.read_text(encoding="utf-8")
        if resolved_signals_path:
            findings.append(
                artifact_written(
                    resolved_signals_path,
                    producer="package_llm",
                    mode="package_llm",
                )
            )
    elif extract_signals:
        findings.append(
            Finding("warn", "PACKAGE_LLM_PARTIAL", "Package LLM mode expects resolve_signals=True; skipping extraction.")
        )

    findings.append(pipeline_branch_completed("package_llm", elapsed_sec=0.0))
    return (
        corpus_file,
        full_text_file,
        full_text,
        corpus_char_count,
        signals_file,
        signal_count,
        resolved_signals_path,
        resolved_signal_count,
        alternates_count,
        derived_signal_count,
        resolved_summary,
        package_result,
        model_used,
    )


def _validate_package_mode_artifacts(run_id: str, findings: list[Finding]) -> None:
    """Fail fast when package mode produced legacy-shaped artifacts."""
    docs_dir = documents_dir(run_id)
    if docs_dir.exists() and any(docs_dir.iterdir()):
        findings.append(
            Finding(
                "error",
                "PACKAGE_LEGACY_ARTIFACT_LEAK",
                f"Legacy documents/ artifacts present after package extraction: {docs_dir}",
            )
        )
    docset_path = docset_manifest_path(run_id)
    if docset_path.exists():
        findings.append(
            Finding(
                "error",
                "PACKAGE_LEGACY_ARTIFACT_LEAK",
                f"Legacy docset manifest present after package extraction: {docset_path}",
            )
        )
    resolved_path = resolved_signals_json_path(run_id)
    if not resolved_path.exists():
        return
    resolved = read_json(resolved_path)
    if not isinstance(resolved, dict):
        return
    for signal in resolved.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        signal_id = str(signal.get("id") or "")
        if signal_id == "process_sources_v1":
            continue
        if signal.get("docset_provenance"):
            findings.append(
                Finding(
                    "error",
                    "PACKAGE_LEGACY_SIGNAL_SHAPE",
                    f"Legacy docset_provenance on signal {signal_id}.",
                )
            )
        if str(signal.get("resolution_status") or "") == "passthrough":
            findings.append(
                Finding(
                    "error",
                    "PACKAGE_LEGACY_SIGNAL_SHAPE",
                    f"Legacy passthrough resolution on signal {signal_id}.",
                )
            )
        source_summary = signal.get("source_summary") if isinstance(signal.get("source_summary"), dict) else {}
        tiers = source_summary.get("authority_tiers") or []
        if any(str(item) == "GLOBAL_SCAN" for item in tiers):
            findings.append(
                Finding(
                    "error",
                    "PACKAGE_LEGACY_SIGNAL_SHAPE",
                    f"GLOBAL_SCAN authority on signal {signal_id}.",
                )
            )


def _write_signals_if_present(
    run_id: str,
    merged_signals: list[dict],
    model_used: str | None,
    *,
    force: bool = False,
) -> tuple[Path | None, int]:
    if not merged_signals:
        return None, 0
    if not force and not model_used:
        return None, 0
    if not any(item.get("id") != "process_sources_v1" for item in merged_signals):
        return None, 0
    signals_file = signals_json_path(run_id)
    write_json(signals_file, merged_signals)
    return signals_file, len(merged_signals)


def _write_diagnostics(
    run_id: str,
    *,
    stage: str,
    findings: list[Finding],
    elapsed_sec: float,
    artifacts: dict[str, str | None],
    extra: dict | None = None,
) -> Path:
    payload = {
        "runId": run_id,
        "stage": stage,
        "openaiApiKeyPresent": has_openai_api_key(),
        "elapsedSec": elapsed_sec,
        "findings": [finding.to_dict() for finding in findings],
        "artifacts": artifacts,
    }
    if extra:
        payload.update(extra)
    diagnostics_file = diagnostics_path(run_id)
    write_json(diagnostics_file, payload)
    return diagnostics_file


def _apply_resolution(
    run_id: str,
    findings: list[Finding],
    full_text: str,
    *,
    signals: list[dict] | None = None,
) -> tuple[str | None, dict | None, int, int, int, dict]:
    artifact, resolve_findings, resolved_path = resolve_run_signals(
        run_id,
        signals=signals,
        corpus_text=full_text,
    )
    findings.extend(resolve_findings)
    if not artifact:
        return None, None, 0, 0, 0, {}
    summary = artifact.get("summary") if isinstance(artifact, dict) else {}
    return (
        resolved_path,
        artifact,
        int(summary.get("resolved_signal_count") or len(artifact.get("signals") or [])),
        int(summary.get("signals_with_alternates") or 0),
        int(summary.get("derived_signal_count") or 0),
        summary if isinstance(summary, dict) else {},
    )


def run_ingest_pipeline(
    file_paths: list[Path],
    run_id: str | None = None,
    *,
    extract_text: bool = True,
    extract_signals: bool = False,
    resolve_signals: bool = False,
    progress=None,
    force_refresh: bool = False,
) -> PipelineResult:
    """Ingest/corpus plus optional signal extraction (package_llm default or legacy docset)."""
    import os

    from extraction.config import load_project_env

    load_project_env()
    started = time.perf_counter()
    extraction_mode = get_extraction_mode()
    force_refresh = force_refresh or os.getenv("GOVCON_FORCE_PACKAGE_REFRESH", "").strip().lower() in {"1", "true", "yes"}
    ingest_result = ingest_files(file_paths, run_id=run_id)
    findings = list(ingest_result.findings)

    manifest = build_run_manifest(ingest_result.run_id, ingest_result.sources)
    manifest_file = manifest_path(ingest_result.run_id)
    write_json(manifest_file, manifest.to_dict())

    process_sources = build_process_sources_v1(ingest_result.run_id, ingest_result.sources)
    process_sources_file = process_sources_signal_path(ingest_result.run_id)
    base_signal = process_sources_base_signal(process_sources)
    write_json(process_sources_file, base_signal)

    docset_result = None
    package_result = None
    model_used: str | None = None
    active_source_count = sum(1 for source in ingest_result.sources if not source.did_dedupe)
    corpus_file: Path | None = None
    full_text_file: Path | None = None
    signals_file: Path | None = None
    full_text = ""
    corpus_char_count: int | None = None
    signal_count = 0
    resolved_signals_path: str | None = None
    resolved_signal_count = 0
    alternates_count = 0
    derived_signal_count = 0
    resolved_summary: dict = {}

    if extraction_mode == "legacy":
        (
            corpus_file,
            full_text_file,
            full_text,
            corpus_char_count,
            signals_file,
            signal_count,
            resolved_signals_path,
            resolved_signal_count,
            alternates_count,
            derived_signal_count,
            resolved_summary,
            docset_result,
            model_used,
        ) = _run_legacy_extraction(
            ingest_result,
            extract_text=extract_text,
            extract_signals=extract_signals,
            resolve_signals=resolve_signals,
            base_signal=base_signal,
            findings=findings,
        )
    else:
        corpus_file = None
        if extract_text and ingest_result.sources and not extract_signals:
            corpus = build_corpus_v1(ingest_result.run_id, ingest_result.sources)
            findings.extend(corpus.findings)
            corpus_file = corpus_v1_path(ingest_result.run_id)
            write_json(corpus_file, corpus.to_dict())
            full_text, text_findings = build_full_corpus_text(
                ingest_result.run_id,
                ingest_result.sources,
                corpus,
            )
            findings.extend(text_findings)
            full_text_file = full_corpus_text_path(ingest_result.run_id)
            write_text(full_text_file, full_text + ("\n" if full_text else ""))
            corpus_char_count = len(full_text)
        else:
            full_text_file = None
            full_text = ""
            corpus_char_count = None

        if extract_signals and force_refresh:
            findings.append(Finding("info", "PACKAGE_CACHE_BYPASS", "Forced refresh; bypassing package extraction cache."))

        (
            _corpus_file,
            full_text_file,
            full_text,
            corpus_char_count,
            signals_file,
            signal_count,
            resolved_signals_path,
            resolved_signal_count,
            alternates_count,
            derived_signal_count,
            resolved_summary,
            package_result,
            model_used,
        ) = _run_package_llm_extraction(
            ingest_result,
            extract_signals=extract_signals,
            resolve_signals=resolve_signals,
            base_signal=base_signal,
            findings=findings,
            progress=progress,
            force_refresh=force_refresh,
        )
        if extract_signals and resolve_signals:
            _validate_package_mode_artifacts(ingest_result.run_id, findings)

    elapsed_sec = round(time.perf_counter() - started, 3)
    if extraction_mode == "package_llm" and extract_signals:
        stage = "package_llm"
    elif resolve_signals:
        stage = "milestone_5" if docset_result and getattr(docset_result, "solicitation_set_detected", False) else "milestone_4"
    elif extract_signals and docset_result:
        stage = "milestone_5" if docset_result.solicitation_set_detected else "milestone_3"
    elif extract_signals and extract_text:
        stage = "milestone_3"
    elif extract_text:
        stage = "milestone_2"
    else:
        stage = "milestone_1"

    package_diagnostics = package_result.diagnostics if package_result else {}
    diagnostics_file = _write_diagnostics(
        ingest_result.run_id,
        stage=stage,
        findings=findings,
        elapsed_sec=elapsed_sec,
        artifacts={
            "manifest": str(manifest_file),
            "processSourcesSignal": str(process_sources_file),
            "corpusV1": str(corpus_file) if corpus_file else None,
            "fullCorpusText": str(full_text_file) if full_text_file else None,
            "signalsJson": str(signals_file) if signals_file else None,
            "resolvedSignalsJson": resolved_signals_path,
            "docsetManifest": str(docset_manifest_path(ingest_result.run_id))
            if docset_result
            else None,
        },
        extra={
            "extractionMode": extraction_mode,
            "sourceCount": len(ingest_result.sources),
            "sourceDocumentCount": active_source_count,
            "corpusCharCount": corpus_char_count,
            "rawSignalCount": signal_count,
            "resolvedSignalCount": resolved_signal_count,
            "signalsWithAlternates": alternates_count,
            "derivedSignalCount": derived_signal_count,
            "modelUsed": model_used,
            "statusCounts": resolved_summary.get("status_counts") if resolve_signals else None,
            "confidenceCounts": resolved_summary.get("confidence_counts") if resolve_signals else None,
            "solicitationSetDetected": docset_result.solicitation_set_detected if docset_result else bool(package_result),
            "baseSolicitationFilename": (
                docset_result.base_solicitation_filename if docset_result else (package_result.base_solicitation_filename if package_result else None)
            ),
            "detectedAmendmentsInOrder": (
                docset_result.amendments_in_order if docset_result else (package_result.amendments_in_order if package_result else [])
            ),
            "classifiedDocumentCounts": docset_result.classified_counts if docset_result else {},
            "unclassifiedSources": docset_result.unclassified_count if docset_result else 0,
            "perDocumentSignalCounts": docset_result.per_document_signal_counts if docset_result else {},
            "mergedSignalCount": signal_count,
            "supersededCandidateCount": docset_result.superseded_candidate_count if docset_result else 0,
            "unresolvedAmendmentConflicts": docset_result.unresolved_amendment_conflicts if docset_result else 0,
            "ocrDocumentsRequiring": (
                docset_result.ocr_documents_requiring if docset_result else (package_result.ocr_documents_requiring if package_result else 0)
            ),
            "ocrPagesProcessed": (
                docset_result.ocr_pages_processed if docset_result else (package_result.ocr_pages_processed if package_result else 0)
            ),
            "ocrDocumentsRerun": docset_result.ocr_documents_rerun if docset_result else 0,
            "ocrFailures": docset_result.ocr_failures if docset_result else 0,
            "controllingDocumentsUnreadable": docset_result.controlling_documents_unreadable if docset_result else 0,
            "ocrDiagnostics": docset_result.ocr_diagnostics if docset_result else [],
            "packageCacheHit": package_result.cache_hit if package_result else False,
            "packageDiagnostics": package_diagnostics,
        },
    )

    return PipelineResult(
        run_id=ingest_result.run_id,
        manifest_path=str(manifest_file),
        diagnostics_path=str(diagnostics_file),
        process_sources_path=str(process_sources_file),
        corpus_path=str(corpus_file) if corpus_file else None,
        full_text_path=str(full_text_file) if full_text_file else None,
        signals_path=str(signals_file) if signals_file else None,
        signal_count=signal_count,
        resolved_signals_path=resolved_signals_path,
        resolved_signal_count=resolved_signal_count,
        alternates_count=alternates_count,
        derived_signal_count=derived_signal_count,
        model_used=model_used,
        elapsed_sec=elapsed_sec,
        corpus_char_count=corpus_char_count,
        findings=findings,
        solicitation_set_detected=docset_result.solicitation_set_detected if docset_result else bool(package_result),
        base_solicitation_filename=(
            docset_result.base_solicitation_filename if docset_result else (package_result.base_solicitation_filename if package_result else None)
        ),
        amendments_detected=(
            docset_result.amendments_in_order if docset_result else (package_result.amendments_in_order if package_result else [])
        ),
        document_count=active_source_count,
        superseded_candidate_count=docset_result.superseded_candidate_count if docset_result else 0,
        unresolved_amendment_conflicts=docset_result.unresolved_amendment_conflicts if docset_result else 0,
        docset_manifest_path=str(docset_manifest_path(ingest_result.run_id))
        if docset_result
        else None,
        ocr_documents_requiring=(
            docset_result.ocr_documents_requiring if docset_result else (package_result.ocr_documents_requiring if package_result else 0)
        ),
        ocr_pages_processed=(
            docset_result.ocr_pages_processed if docset_result else (package_result.ocr_pages_processed if package_result else 0)
        ),
        ocr_documents_rerun=docset_result.ocr_documents_rerun if docset_result else 0,
        ocr_failures=docset_result.ocr_failures if docset_result else 0,
        controlling_documents_unreadable=docset_result.controlling_documents_unreadable if docset_result else 0,
        ocr_diagnostics=docset_result.ocr_diagnostics if docset_result else [],
    )


def run_signal_extraction_for_existing_run(run_id: str) -> PipelineResult:
    """Run Milestone 3 against an existing run folder that already has corpus text."""
    started = time.perf_counter()
    findings: list[Finding] = []

    full_text_file = full_corpus_text_path(run_id)
    if not full_text_file.exists():
        findings.append(
            Finding("error", "SIGNAL_EXTRACTION_NO_CORPUS", f"Missing corpus text: {full_text_file}")
        )
        elapsed_sec = round(time.perf_counter() - started, 3)
        diagnostics_file = diagnostics_path(run_id)
        write_json(
            diagnostics_file,
            {
                "runId": run_id,
                "stage": "milestone_3",
                "openaiApiKeyPresent": has_openai_api_key(),
                "findings": [item.to_dict() for item in findings],
                "elapsedSec": elapsed_sec,
            },
        )
        return PipelineResult(
            run_id=run_id,
            manifest_path=str(manifest_path(run_id)),
            diagnostics_path=str(diagnostics_file),
            findings=findings,
            elapsed_sec=elapsed_sec,
        )

    manifest = read_json(manifest_path(run_id))
    sources = manifest.get("sources") if isinstance(manifest, dict) else []
    fallback_source = None
    if isinstance(sources, list) and sources:
        first = sources[0]
        if isinstance(first, dict):
            fallback_source = first.get("original_filename")

    process_sources_file = process_sources_signal_path(run_id)
    base_signals = [read_json(process_sources_file)] if process_sources_file.exists() else []

    full_text = full_text_file.read_text(encoding="utf-8")
    merged_signals, llm_findings, model_used = extract_signals_from_corpus(
        run_id,
        full_text,
        fallback_source_file=fallback_source,
        base_signals=base_signals,
    )
    findings.extend(llm_findings)

    signals_file, signal_count = _write_signals_if_present(run_id, merged_signals, model_used)

    elapsed_sec = round(time.perf_counter() - started, 3)
    diagnostics_file = diagnostics_path(run_id)
    write_json(
        diagnostics_file,
        {
            "runId": run_id,
            "stage": "milestone_3",
            "corpusCharCount": len(full_text),
            "signalCount": signal_count,
            "modelUsed": model_used,
            "openaiApiKeyPresent": has_openai_api_key(),
            "elapsedSec": elapsed_sec,
            "findings": [item.to_dict() for item in findings],
            "artifacts": {
                "fullCorpusText": str(full_text_file),
                "signalsJson": str(signals_file) if signals_file else None,
            },
        },
    )

    return PipelineResult(
        run_id=run_id,
        manifest_path=str(manifest_path(run_id)),
        diagnostics_path=str(diagnostics_file),
        full_text_path=str(full_text_file),
        signals_path=str(signals_file) if signals_file else None,
        signal_count=signal_count,
        model_used=model_used,
        elapsed_sec=elapsed_sec,
        corpus_char_count=len(full_text),
        findings=findings,
    )


def run_resolve_for_existing_run(run_id: str) -> PipelineResult:
    started = time.perf_counter()
    findings: list[Finding] = []
    full_text_file = full_corpus_text_path(run_id)
    full_text = full_text_file.read_text(encoding="utf-8") if full_text_file.exists() else ""
    raw_count = 0
    signals_path = signals_json_path(run_id)
    if signals_path.exists():
        raw = read_json(signals_path)
        if isinstance(raw, list):
            raw_count = len(raw)

    (
        resolved_signals_path,
        resolved_artifact,
        resolved_signal_count,
        alternates_count,
        derived_signal_count,
        resolved_summary,
    ) = _apply_resolution(run_id, findings, full_text)
    _ = resolved_artifact

    elapsed_sec = round(time.perf_counter() - started, 3)
    diagnostics_file = _write_diagnostics(
        run_id,
        stage="milestone_4",
        findings=findings,
        elapsed_sec=elapsed_sec,
        artifacts={
            "fullCorpusText": str(full_text_file) if full_text_file.exists() else None,
            "signalsJson": str(signals_path) if signals_path.exists() else None,
            "resolvedSignalsJson": resolved_signals_path,
        },
        extra={
            "rawSignalCount": raw_count,
            "resolvedSignalCount": resolved_signal_count,
            "signalsWithAlternates": alternates_count,
            "derivedSignalCount": derived_signal_count,
            "statusCounts": resolved_summary.get("status_counts"),
            "confidenceCounts": resolved_summary.get("confidence_counts"),
        },
    )

    return PipelineResult(
        run_id=run_id,
        manifest_path=str(manifest_path(run_id)),
        diagnostics_path=str(diagnostics_file),
        full_text_path=str(full_text_file) if full_text_file.exists() else None,
        signals_path=str(signals_path) if signals_path.exists() else None,
        signal_count=raw_count,
        resolved_signals_path=resolved_signals_path,
        resolved_signal_count=resolved_signal_count,
        alternates_count=alternates_count,
        derived_signal_count=derived_signal_count,
        elapsed_sec=elapsed_sec,
        corpus_char_count=len(full_text),
        findings=findings,
    )
