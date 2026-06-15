from __future__ import annotations

from pathlib import Path
from typing import Any

from extraction.config import get_run_dir, parity_report_json_path, parity_report_md_path
from extraction.persist import read_json, write_json, write_text
from extraction.parity.compare import compare_resolved_signals
from extraction.parity.scope_review import check_scope_review_compatibility


def _render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Parity Report",
        "",
        f"- Golden run: `{report.get('goldenRunId')}`",
        f"- Local run: `{report.get('localRunId')}`",
        f"- Passed: **{report.get('passed')}**",
        "",
        "## Summary",
        "",
    ]
    summary = report.get("summary") or {}
    counts = summary.get("counts") or {}
    for key, value in counts.items():
        lines.append(f"- {key}: {value}")
    mismatches = summary.get("materialMismatches") or []
    if mismatches:
        lines.append(f"- material mismatch IDs: {', '.join(mismatches)}")
    scope = report.get("scopeReview") or {}
    lines.extend(
        [
            "",
            "## Scope Review Compatibility",
            "",
            f"- loader accepted: {scope.get('loaderAccepted')}",
            f"- preview constructed: {scope.get('previewConstructed')}",
            f"- validation error: {scope.get('validationError')}",
            f"- confirmed mappings: {len(scope.get('confirmedMappings') or [])}",
            f"- suggested mappings: {len(scope.get('suggestedMappings') or [])}",
            f"- unmapped values: {len(scope.get('unmappedValues') or [])}",
            f"- withheld conflicts: {len(scope.get('withheldConflicts') or [])}",
        ]
    )
    lines.extend(["", "## Signals", ""])
    for row in report.get("signals") or []:
        lines.extend(
            [
                f"### {row.get('signalId')}",
                f"- comparison: {row.get('comparison')}",
                f"- golden: `{row.get('goldenCanonicalValue')}` ({row.get('goldenConfidence')})",
                f"- local: `{row.get('localCanonicalValue')}` ({row.get('localConfidence')})",
                f"- local disposition: {row.get('localUiDisposition')}",
                f"- controlling: {row.get('controllingFilename')} / amendment {row.get('controllingAmendment')}",
                f"- authority tier: {row.get('authorityTier')}",
                "",
            ]
        )
    return "\n".join(lines)


def run_parity_comparison(
    *,
    golden_path: Path,
    local_path: Path,
    run_id: str,
) -> dict[str, Any]:
    golden_payload = read_json(golden_path)
    local_payload = read_json(local_path)
    report = compare_resolved_signals(golden_payload=golden_payload, local_payload=local_payload)
    report["scopeReview"] = check_scope_review_compatibility(local_payload)
    report["runId"] = run_id
    report["goldenPath"] = str(golden_path)
    report["localPath"] = str(local_path)

    json_path = parity_report_json_path(run_id)
    md_path = parity_report_md_path(run_id)
    write_json(json_path, report)
    write_text(md_path, _render_markdown(report))
    report["parityReportJson"] = str(json_path)
    report["parityReportMd"] = str(md_path)
    return report


def run_parity_for_run(
    *,
    golden_path: Path,
    run_id: str,
    local_path: Path | None = None,
) -> dict[str, Any]:
    from extraction.config import resolved_signals_json_path

    resolved = local_path or resolved_signals_json_path(run_id)
    if not resolved.is_file():
        raise FileNotFoundError(f"Local resolved signals not found: {resolved}")
    return run_parity_comparison(golden_path=golden_path, local_path=resolved, run_id=run_id)
