from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from extraction.config import (
    get_extraction_profile,
    get_merge_run_id,
    has_openai_api_key,
    package_corpus_text_path,
    package_extraction_raw_path,
    package_extraction_validated_path,
    package_extraction_dir,
    resolved_signals_json_path,
    signals_json_path,
    validation_findings_path,
)
from extraction.ingest.upload import build_process_sources_v1, process_sources_base_signal
from extraction.package_llm.build_info import build_info_finding, collect_build_info
from extraction.package_llm.cache import (
    cache_dir_for_key,
    cache_key,
    copy_cache_to_run,
    load_cached_artifacts,
    write_cache_artifacts,
    write_run_artifacts,
)
from extraction.package_llm.corpus import PackageCorpus, assemble_package_corpus, extract_package_source_text
from extraction.package_llm.extract import call_package_extraction
from extraction.package_llm.merge import merge_resolved_signals
from extraction.package_llm.resolve import resolve_validated_package_signals
from extraction.package_llm.scope_corpus import assemble_scope_corpus
from extraction.package_llm.validate import validate_package_extraction
from extraction.persist import read_json, write_json, write_text
from extraction.progress import ProgressCallback, emit_progress
from extraction.types import Finding, SourceRecord


@dataclass
class PackageLLMResult:
    resolved_signals: dict[str, Any]
    findings: list[Finding] = field(default_factory=list)
    model_used: str | None = None
    signal_count: int = 0
    resolved_signal_count: int = 0
    alternates_count: int = 0
    corpus_char_count: int = 0
    cache_hit: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)
    base_solicitation_filename: str | None = None
    amendments_in_order: list[str] = field(default_factory=list)
    document_count: int = 0
    ocr_pages_processed: int = 0
    ocr_documents_requiring: int = 0


def _emit(
    progress: ProgressCallback | None,
    label: str,
    pct: float,
    *,
    doc_index: int | None = None,
    doc_total: int | None = None,
    filename: str | None = None,
    phase: str | None = None,
) -> None:
    emit_progress(
        progress,
        label,
        pct,
        doc_index=doc_index,
        doc_total=doc_total,
        filename=filename,
        phase=phase,
    )


def _stage_event(name: str, status: str, started_at: float, *, function: str, exc: Exception | None = None) -> dict[str, Any]:
    ended_at = time.time()
    event = {
        "stage": name,
        "status": status,
        "function": function,
        "startTime": started_at,
        "endTime": ended_at,
        "elapsedSeconds": round(ended_at - started_at, 3),
        "heartbeatAt": ended_at,
    }
    if exc is not None:
        event["exceptionClass"] = type(exc).__name__
        event["exceptionMessage"] = str(exc)
    return event


def _write_package_stage_diagnostics(run_id: str, diagnostics: dict[str, Any], stage_events: list[dict[str, Any]]) -> None:
    package_extraction_dir(run_id).mkdir(parents=True, exist_ok=True)
    write_json(package_extraction_dir(run_id) / "diagnostics.json", {**diagnostics, "stageEvents": stage_events})


def run_package_llm_extraction(
    run_id: str,
    sources: list[SourceRecord],
    *,
    process_sources_payload: dict[str, Any] | None = None,
    force_refresh: bool = False,
    progress: ProgressCallback | None = None,
    profile: str | None = None,
) -> PackageLLMResult:
    started = time.perf_counter()
    profile = profile or get_extraction_profile()
    merge_run_id = get_merge_run_id()
    findings: list[Finding] = [build_info_finding(profile=profile)]
    build_info = collect_build_info(profile=profile)
    active_sources = [source for source in sources if not source.did_dedupe]
    key = cache_key(active_sources, profile=profile)
    timings: dict[str, float] = {}
    stage_events: list[dict[str, Any]] = []

    if not force_refresh:
        cached = load_cached_artifacts(key, profile=profile)
        if cached:
            _emit(progress, "Using cached extraction result", 0.92, phase="cache")
            copy_cache_to_run(key, run_id)
            resolved = cached["resolved"]
            if isinstance(resolved, dict):
                resolved = dict(resolved)
                resolved["runId"] = run_id
            if merge_run_id:
                base = read_json(resolved_signals_json_path(merge_run_id))
                if isinstance(base, dict):
                    resolved = merge_resolved_signals(base, resolved)
                    resolved["runId"] = run_id
            write_json(resolved_signals_json_path(run_id), resolved)
            diagnostics = cached.get("diagnostics") if isinstance(cached.get("diagnostics"), dict) else {}
            diagnostics = dict(diagnostics)
            diagnostics["cacheHit"] = True
            diagnostics["cacheKey"] = key
            diagnostics["cacheDir"] = str(cache_dir_for_key(key))
            diagnostics["buildInfo"] = build_info
            diagnostics["totalElapsedSec"] = round(time.perf_counter() - started, 3)
            summary = resolved.get("summary") if isinstance(resolved, dict) else {}
            package_summary = (cached.get("validated") or {}).get("package_summary") or {}
            return PackageLLMResult(
                resolved_signals=resolved if isinstance(resolved, dict) else {},
                findings=[Finding("info", "PACKAGE_CACHE_HIT", "Reused cached package extraction result.")],
                model_used=diagnostics.get("model"),
                signal_count=int(summary.get("total_signal_ids") or 0),
                resolved_signal_count=int(summary.get("resolved_signal_count") or 0),
                alternates_count=int(summary.get("alternates_count") or 0),
                cache_hit=True,
                diagnostics=diagnostics,
                base_solicitation_filename=package_summary.get("base_solicitation_filename"),
                amendments_in_order=list(package_summary.get("amendments_in_order") or []),
                document_count=len(active_sources),
            )

    if not has_openai_api_key():
        findings.append(
            Finding(
                "error",
                "OPENAI_API_KEY_MISSING",
                "OPENAI_API_KEY is required for package-level GPT extraction.",
            )
        )
        return PackageLLMResult(resolved_signals={}, findings=findings, document_count=len(active_sources))

    parse_started = time.perf_counter()
    stage_started_at = time.time()
    total = max(len(active_sources), 1)
    package_sources = []
    for index, source in enumerate(active_sources, start=1):
        short_name = source.original_filename
        _emit(
            progress,
            f"Analyzing document {index} of {total} — {short_name}",
            0.05 + (0.15 * index / total),
            doc_index=index,
            doc_total=total,
            filename=short_name,
            phase="reading",
        )
        package_source, source_findings = extract_package_source_text(source, apply_ocr=True)
        findings.extend(source_findings)
        if package_source.ocr_pages:
            _emit(
                progress,
                f"Extracting scanned pages from {short_name}",
                0.05 + (0.15 * (index - 0.35) / total),
                doc_index=index,
                doc_total=total,
                filename=short_name,
                phase="ocr",
            )
        package_sources.append(package_source)
    timings["documentParsingSec"] = round(time.perf_counter() - parse_started, 3)
    stage_events.append(_stage_event("Reading documents", "completed", stage_started_at, function="extract_package_source_text"))

    corpus_started = time.perf_counter()
    stage_started_at = time.time()
    full_corpus_text = assemble_package_corpus(package_sources)
    corpus_stats: dict[str, Any] = {}
    if profile == "market_scope_fast":
        corpus_text, corpus_stats = assemble_scope_corpus(package_sources)
    else:
        corpus_text = full_corpus_text
    corpus = PackageCorpus(sources=package_sources, corpus_text=corpus_text, findings=findings)
    timings["corpusAssemblySec"] = round(time.perf_counter() - corpus_started, 3)
    stage_events.append(_stage_event("Saving extracted scope", "completed", stage_started_at, function="assemble_scope_corpus"))
    ocr_pages = sum(len(source.ocr_pages) for source in package_sources)
    ocr_docs = sum(1 for source in package_sources if source.ocr_pages)

    package_extraction_dir(run_id).mkdir(parents=True, exist_ok=True)
    write_text(package_corpus_text_path(run_id), corpus.corpus_text + ("\n" if corpus.corpus_text else ""))
    if profile == "market_scope_fast":
        write_text(package_extraction_dir(run_id) / "full_package_corpus.txt", full_corpus_text)

    _emit(progress, "Assembling solicitation package", 0.25, phase="corpus")
    _emit(
        progress,
        f"Extracting market scope from {len(active_sources)} documents with GPT-5.5…",
        0.45,
        doc_total=len(active_sources),
        phase="gpt",
    )

    if not process_sources_payload:
        process_sources = build_process_sources_v1(run_id, sources)
        process_sources_payload = process_sources_base_signal(process_sources)

    gpt_started = time.perf_counter()
    stage_started_at = time.time()
    try:
        raw_payload, gpt_diagnostics, gpt_findings = call_package_extraction(corpus.corpus_text, profile=profile)
    except Exception as exc:
        stage_events.append(
            _stage_event("Analyzing market scope with GPT-5.5", "failed", stage_started_at, function="call_package_extraction", exc=exc)
        )
        _write_package_stage_diagnostics(run_id, {"stage": "gpt_failed", "buildInfo": build_info}, stage_events)
        raise
    findings.extend(gpt_findings)
    timings["gptRequestSec"] = round(time.perf_counter() - gpt_started, 3)
    write_json(package_extraction_raw_path(run_id), raw_payload)
    stage_events.append(
        _stage_event("Analyzing market scope with GPT-5.5", "completed", stage_started_at, function="call_package_extraction")
    )

    _emit(progress, "Validating citations", 0.75, phase="validate")
    validation_started = time.perf_counter()
    stage_started_at = time.time()
    try:
        validated_payload, validation_findings = validate_package_extraction(raw_payload, corpus)
    except Exception as exc:
        stage_events.append(
            _stage_event("Validating citations", "failed", stage_started_at, function="validate_package_extraction", exc=exc)
        )
        _write_package_stage_diagnostics(run_id, {"stage": "validation_failed", "buildInfo": build_info}, stage_events)
        raise
    findings.extend(validation_findings)
    timings["validationSec"] = round(time.perf_counter() - validation_started, 3)
    write_json(package_extraction_validated_path(run_id), validated_payload)
    validation_records = []
    for signal in validated_payload.get("signals") or []:
        if isinstance(signal, dict) and signal.get("validation_findings"):
            validation_records.append(
                {
                    "id": signal.get("id"),
                    "findings": signal.get("validation_findings"),
                }
            )
    write_json(validation_findings_path(run_id), validation_records)
    stage_events.append(_stage_event("Validating citations", "completed", stage_started_at, function="validate_package_extraction"))

    _emit(progress, "Saving extracted scope", 0.9, phase="resolve")
    resolve_started = time.perf_counter()
    stage_started_at = time.time()
    try:
        resolved = resolve_validated_package_signals(
            run_id,
            validated_payload,
            process_sources_signal=process_sources_payload,
            profile=profile,
        )
    except Exception as exc:
        stage_events.append(
            _stage_event("Saving extracted scope", "failed", stage_started_at, function="resolve_validated_package_signals", exc=exc)
        )
        _write_package_stage_diagnostics(run_id, {"stage": "resolution_failed", "buildInfo": build_info}, stage_events)
        raise
    if merge_run_id:
        base = read_json(resolved_signals_json_path(merge_run_id))
        if isinstance(base, dict):
            resolved = merge_resolved_signals(base, resolved)
            resolved["runId"] = run_id
    timings["resolutionSec"] = round(time.perf_counter() - resolve_started, 3)
    write_json(resolved_signals_json_path(run_id), resolved)
    stage_events.append(
        _stage_event("Saving extracted scope", "completed", stage_started_at, function="resolve_validated_package_signals")
    )
    _emit(progress, "Market scope extracted successfully", 0.93, phase="saved")

    diagnostics = {
        **gpt_diagnostics,
        **corpus_stats,
        "buildInfo": build_info,
        "cacheHit": False,
        "cacheKey": key,
        "cacheDir": str(cache_dir_for_key(key)),
        "extractionProfile": profile,
        "timings": timings,
        "documentParsingSec": timings.get("documentParsingSec"),
        "corpusAssemblySec": timings.get("corpusAssemblySec"),
        "textExtractionSec": timings.get("documentParsingSec"),
        "ocrPagesProcessed": ocr_pages,
        "ocrDocumentsRequiring": ocr_docs,
        "corpusCharCount": len(corpus.corpus_text),
        "documentCount": len(active_sources),
        "totalElapsedSec": round(time.perf_counter() - started, 3),
        "stageEvents": stage_events,
    }

    write_run_artifacts(
        run_id,
        raw=raw_payload,
        validated=validated_payload,
        validation_findings=validation_records,
        diagnostics=diagnostics,
    )
    signals_json = []
    for signal in validated_payload.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        signals_json.append(
            {
                "id": signal.get("id"),
                "value": signal.get("value"),
                "confidence": signal.get("confidence"),
                "evidence": signal.get("evidence") or [],
                "notes": signal.get("reasoning_summary") or "",
            }
        )
    signals_json.insert(0, process_sources_payload)
    write_json(signals_json_path(run_id), signals_json)

    write_cache_artifacts(
        key,
        raw=raw_payload,
        validated=validated_payload,
        resolved=resolved,
        validation_findings=validation_records,
        diagnostics=diagnostics,
    )

    package_summary = validated_payload.get("package_summary") if isinstance(validated_payload, dict) else {}
    summary = resolved.get("summary") if isinstance(resolved, dict) else {}

    return PackageLLMResult(
        resolved_signals=resolved,
        findings=findings,
        model_used=gpt_diagnostics.get("model"),
        signal_count=int(summary.get("total_signal_ids") or len(resolved.get("signals") or [])),
        resolved_signal_count=int(summary.get("resolved_signal_count") or 0),
        alternates_count=int(summary.get("alternates_count") or 0),
        corpus_char_count=len(corpus.corpus_text),
        cache_hit=False,
        diagnostics=diagnostics,
        base_solicitation_filename=(package_summary or {}).get("base_solicitation_filename")
        if isinstance(package_summary, dict)
        else None,
        amendments_in_order=list((package_summary or {}).get("amendments_in_order") or [])
        if isinstance(package_summary, dict)
        else [],
        document_count=len(active_sources),
        ocr_pages_processed=ocr_pages,
        ocr_documents_requiring=ocr_docs,
    )
