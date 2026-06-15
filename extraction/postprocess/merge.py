from __future__ import annotations

import re
from typing import Any

from extraction.resolve.phrase_decomposition import decompose_competition_set_aside_phrase
from extraction.types import Finding


def sort_signals_by_id(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(signals, key=lambda item: str(item.get("id") or ""))


def ensure_solicitation_number_alias(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(signal.get("id") or ""): signal for signal in signals}
    source = by_id.get("rfp_solicitation_id_v1")
    if not source or "rfp_solicitation_number_v1" in by_id:
        return signals
    alias = {**source, "id": "rfp_solicitation_number_v1"}
    return sort_signals_by_id([*signals, alias])


def fix_mislabeled_competition_type(signals: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    updated: list[dict[str, Any]] = []
    for signal in signals:
        if str(signal.get("id") or "") != "rfp_competition_type_v1":
            updated.append(signal)
            continue
        value = str(signal.get("value") or "").strip()
        if not value or value.lower() == "competitive":
            updated.append(signal)
            continue
        if not re.search(r"(?:set-aside|set aside|8\s*\(\s*a\s*\))", value, flags=re.IGNORECASE):
            updated.append(signal)
            continue

        excerpt = ""
        ev1 = signal.get("evidence_v1")
        if isinstance(ev1, dict):
            excerpt = str(ev1.get("excerpt") or "")
        if not excerpt and isinstance(signal.get("evidence"), list):
            excerpt = " ".join(str(item.get("snippet") or "") for item in signal["evidence"] if isinstance(item, dict))

        decomposed = decompose_competition_set_aside_phrase(excerpt or value)
        if not decomposed:
            updated.append(signal)
            continue

        updated.append({**signal, "value": decomposed["competition_type"]})
        findings.append(
            Finding(
                "info",
                "COMPETITION_TYPE_DECOMPOSED",
                "Corrected mislabeled competition type from combined procurement phrase",
            )
        )

        if decomposed.get("set_aside") and not any(s.get("id") == "rfp_set_aside_v1" for s in signals):
            updated.append(
                {
                    "id": "rfp_set_aside_v1",
                    "value": decomposed["set_aside"],
                    "confidence": signal.get("confidence") or "high",
                    "evidence": signal.get("evidence") or [],
                    "evidence_v1": signal.get("evidence_v1"),
                    "findings": [],
                }
            )
    return sort_signals_by_id(updated), findings
