from __future__ import annotations

import re
from typing import Any

from extraction.signals.authority import authority_rank

SUBMISSION_IDS = (
    "rfp_submission_instructions_v1",
    "rfp_submission_method_v1",
    "rfp_submission_destination_v1",
    "rfp_submission_format_v1",
)

SF_BOILERPLATE = re.compile(r"\b(sealed offers|item\s+8|standard form\s+33|sf\s*33)\b", re.I)


def _confidence_rank(value: str | None) -> int:
    return {"high": 3, "medium": 2, "low": 1}.get(str(value or "").lower(), 0)


def _source_rank(signal: dict[str, Any]) -> int:
    docset = signal.get("_docset") if isinstance(signal.get("_docset"), dict) else {}
    amendment = docset.get("amendment_order")
    if amendment:
        return 100
    hint = ""
    ev1 = signal.get("evidence_v1")
    if isinstance(ev1, dict):
        excerpt = str(ev1.get("excerpt") or "")
        if excerpt.startswith("[FULLTEXT:L]"):
            return 90
        if excerpt.startswith("[FULLTEXT:M]"):
            return 80
    authority = signal.get("authority")
    if isinstance(authority, dict):
        return 50 + authority_rank(authority)
    return 10


def _submission_score(signal: dict[str, Any]) -> int:
    value = str(signal.get("value") or "")
    score = _confidence_rank(str(signal.get("confidence") or "")) * 10 + _source_rank(signal)
    if re.search(r"DoD SAFE|electronically|portal|upload", value, flags=re.IGNORECASE):
        score += 8
    if re.search(r"@", value):
        score += 3
    if SF_BOILERPLATE.search(value):
        score -= 25
    return score


def canonicalize_submission_signals(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id: dict[str, list[dict[str, Any]]] = {}
    others: list[dict[str, Any]] = []
    for signal in signals:
        signal_id = str(signal.get("id") or "")
        if signal_id in SUBMISSION_IDS:
            by_id.setdefault(signal_id, []).append(signal)
        else:
            others.append(signal)

    chosen: list[dict[str, Any]] = []
    for signal_id in SUBMISSION_IDS:
        candidates = by_id.get(signal_id) or []
        if not candidates:
            continue
        best = max(candidates, key=_submission_score)
        chosen.append(best)
    return sorted([*others, *chosen], key=lambda item: str(item.get("id") or ""))
