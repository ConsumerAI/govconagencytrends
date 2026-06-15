from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from extraction.config import (
    package_extraction_raw_path,
    package_extraction_validated_path,
    process_sources_signal_path,
    resolved_signals_json_path,
    validation_findings_path,
)
from extraction.package_llm.corpus import build_package_corpus
from extraction.package_llm.resolve import resolve_validated_package_signals
from extraction.package_llm.validate import validate_package_extraction
from extraction.persist import read_json, write_json
from extraction.types import SourceRecord


def _sources_from_manifest(manifest: dict[str, Any]) -> list[SourceRecord]:
    records: list[SourceRecord] = []
    for item in manifest.get("sources") or []:
        if not isinstance(item, dict):
            continue
        records.append(
            SourceRecord(
                key=str(item.get("key") or ""),
                ext=str(item.get("ext") or ""),
                bytes=int(item.get("bytes") or 0),
                sha256=str(item.get("sha256") or ""),
                original_filename=str(item.get("original_filename") or item.get("key") or ""),
                abs_path=str(item.get("absPath") or ""),
                did_dedupe=bool(item.get("didDedupe")),
            )
        )
    return records


def summarize_validation(validated: dict[str, Any]) -> dict[str, int]:
    counts = {"confirmed": 0, "review_required": 0, "not_found": 0, "not_applicable": 0}
    evidence_mismatch = 0
    for signal in validated.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        status = str(signal.get("status") or "not_found")
        counts[status] = counts.get(status, 0) + 1
        for finding in signal.get("validation_findings") or []:
            if isinstance(finding, dict) and finding.get("code") == "EVIDENCE_QUOTE_NOT_FOUND":
                evidence_mismatch += 1
    counts["evidence_quote_not_found_findings"] = evidence_mismatch
    return counts


def summarize_resolution(resolved: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    confirmed_filters = 0
    for signal in resolved.get("signals") or []:
        if not isinstance(signal, dict):
            continue
        status = str(signal.get("resolution_status") or "")
        counts[status] = counts.get(status, 0) + 1
        if status.startswith("validated_model_extraction") and signal.get("canonical_value") is not None:
            confirmed_filters += 1
    counts["confirmed_market_filters"] = confirmed_filters
    return counts


def revalidate_package_run(run_id: str, *, write_artifacts: bool = True) -> dict[str, Any]:
    run_dir = Path("data/runs") / run_id
    raw_path = package_extraction_raw_path(run_id)
    if not raw_path.exists():
        raw_path = run_dir / "package_extraction" / "package_extraction.raw.json"
    raw = read_json(raw_path)
    manifest = read_json(run_dir / "manifest.json")
    process_sources = read_json(process_sources_signal_path(run_id))
    process_sources_signal = process_sources if isinstance(process_sources, dict) else {}

    corpus = build_package_corpus(_sources_from_manifest(manifest), apply_ocr=False)
    before_validated = read_json(package_extraction_validated_path(run_id)) if package_extraction_validated_path(run_id).exists() else {}
    before_resolved = read_json(resolved_signals_json_path(run_id)) if resolved_signals_json_path(run_id).exists() else {}

    validated, findings = validate_package_extraction(raw, corpus)
    resolved = resolve_validated_package_signals(run_id, validated, process_sources_signal=process_sources_signal)

    validation_records = []
    for signal in validated.get("signals") or []:
        if isinstance(signal, dict) and signal.get("validation_findings"):
            validation_records.append({"id": signal.get("id"), "findings": signal.get("validation_findings")})

    if write_artifacts:
        write_json(package_extraction_validated_path(run_id), validated)
        write_json(validation_findings_path(run_id), validation_records)
        write_json(resolved_signals_json_path(run_id), resolved)

    return {
        "runId": run_id,
        "openaiCalled": False,
        "before": {
            "validation": summarize_validation(before_validated if isinstance(before_validated, dict) else {}),
            "resolution": summarize_resolution(before_resolved if isinstance(before_resolved, dict) else {}),
        },
        "after": {
            "validation": summarize_validation(validated),
            "resolution": summarize_resolution(resolved),
        },
        "pipelineFindings": [item.to_dict() for item in findings],
        "resolved": resolved,
        "validated": validated,
    }


if __name__ == "__main__":
    import sys

    run_id = sys.argv[1] if len(sys.argv) > 1 else "run_4bdc5c49-bdba-49e0-9327-6e55bc48ddeb"
    report = revalidate_package_run(run_id)
    print(json.dumps({k: v for k, v in report.items() if k not in {"resolved", "validated"}}, indent=2))
