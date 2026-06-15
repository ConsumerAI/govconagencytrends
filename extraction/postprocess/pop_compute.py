from __future__ import annotations

import re
from typing import Any

from extraction.types import Finding

POP_COMPOSITE_ID = "rfp_period_of_performance_v1"
POP_START_ID = "rfp_pop_start_v1"
POP_END_ID = "rfp_pop_end_v1"
POP_YEARS_ID = "rfp_pop_years_v1"

POP_YEARS_NOISE_RE = re.compile(r"^(\d+(?:\.\d+)?)\s*(?:a|y|yr|yrs|year|years)\b", re.IGNORECASE)


def _read_trimmed(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def normalize_pop_years_label(raw: str | None) -> str | None:
    text = _read_trimmed(raw)
    if not text:
        return None
    match = POP_YEARS_NOISE_RE.match(text)
    if match:
        number = float(match.group(1))
        unit = "Year" if number == 1 else "Years"
        return f"{match.group(1)} {unit}"
    if re.fullmatch(r"\d+(?:\.\d+)?", text):
        number = float(text)
        unit = "Year" if number == 1 else "Years"
        return f"{text} {unit}"
    return text


def compute_period_of_performance_summary(
    pop_start: str | None,
    pop_end: str | None,
    pop_years: str | None,
) -> str | None:
    if not pop_start and not pop_end and not pop_years:
        return None
    pop_years = normalize_pop_years_label(pop_years)
    if pop_start and pop_end:
        range_text = f"{pop_start} to {pop_end}"
    elif pop_start:
        range_text = f"from {pop_start}"
    elif pop_end:
        range_text = f"through {pop_end}"
    else:
        range_text = None
    if range_text and pop_years:
        return f"Period of Performance: {range_text} ({pop_years})"
    if range_text:
        return f"Period of Performance: {range_text}"
    if pop_years:
        return f"Period of Performance: {pop_years}"
    return None


def _read_signal_value(signals: list[dict[str, Any]], signal_id: str) -> str | None:
    for signal in signals:
        if str(signal.get("id") or "") == signal_id:
            return _read_trimmed(signal.get("value"))
    return None


def _pick_evidence_source(signals: list[dict[str, Any]]) -> dict[str, Any] | None:
    for signal_id in (POP_START_ID, POP_END_ID, POP_YEARS_ID):
        for signal in signals:
            if str(signal.get("id") or "") == signal_id and _read_trimmed(signal.get("value")):
                return signal
    return None


def apply_computed_period_of_performance(signals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    without_composite = [signal for signal in signals if str(signal.get("id") or "") != POP_COMPOSITE_ID]

    pop_start = _read_signal_value(signals, POP_START_ID)
    pop_end = _read_signal_value(signals, POP_END_ID)
    pop_years = normalize_pop_years_label(_read_signal_value(signals, POP_YEARS_ID))
    summary = compute_period_of_performance_summary(pop_start, pop_end, pop_years)
    if not summary:
        return without_composite, findings

    if any(str(signal.get("id") or "") == POP_COMPOSITE_ID for signal in signals):
        findings.append(
            Finding(
                "info",
                "POP_COMPOSITE_REPLACED",
                "Replaced extracted rfp_period_of_performance_v1 with composite derived from pop start/end/years signals",
            )
        )

    evidence_source = _pick_evidence_source(signals)
    composite = {
        "id": POP_COMPOSITE_ID,
        "value": summary,
        "confidence": _read_trimmed(evidence_source.get("confidence") if evidence_source else None) or "high",
        "evidence": evidence_source.get("evidence") if evidence_source else [],
        "evidence_v1": evidence_source.get("evidence_v1") if evidence_source else None,
        "findings": [
            Finding(
                "info",
                "DERIVED_PERIOD_OF_PERFORMANCE_V1",
                "Period of performance composite derived from rfp_pop_start_v1, rfp_pop_end_v1, and rfp_pop_years_v1",
            ).to_dict()
        ],
    }
    merged = sorted([*without_composite, composite], key=lambda item: str(item.get("id") or ""))
    return merged, findings
