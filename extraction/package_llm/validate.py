from __future__ import annotations

from typing import Any

from extraction.package_llm.corpus import PackageCorpus
from extraction.package_llm.validators.common import validate_signal_record
from extraction.types import Finding


def validate_package_extraction(raw: dict[str, Any], corpus: PackageCorpus) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    signals = raw.get("signals") if isinstance(raw, dict) else None
    if not isinstance(signals, list):
        return {"signals": [], "package_summary": raw.get("package_summary") if isinstance(raw, dict) else {}}, [
            Finding("error", "PACKAGE_VALIDATION_NO_SIGNALS", "No signals to validate")
        ]

    validated_signals: list[dict[str, Any]] = []
    for signal in signals:
        if not isinstance(signal, dict):
            continue
        validated, signal_findings = validate_signal_record(signal, corpus)
        validated_signals.append(validated)
        findings.extend(signal_findings)

    payload = {
        "signals": validated_signals,
        "package_summary": raw.get("package_summary") if isinstance(raw, dict) else {},
    }
    findings.append(
        Finding(
            "info",
            "PACKAGE_VALIDATION_COMPLETE",
            "Package extraction validation completed",
            {"signalCount": len(validated_signals), "findingCount": len(findings)},
        )
    )
    return payload, findings
