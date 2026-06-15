from __future__ import annotations

import re
from typing import Any

from extraction.signals.evaluation_authority import (
    evaluation_authority_score,
    is_marketing_summary,
    is_section_l_submission_context,
)
from extraction.types import Finding

EVALUATION_IDS = (
    "rfp_evaluation_criteria_v1",
    "rfp_eval_method_v1",
    "rfp_eval_weights_v1",
    "rfp_tech_factors_v1",
)

STRONG_CUES = re.compile(
    r"\b(evaluation factors|basis for award|LPTA|lowest price technically acceptable|tradeoff|best value|pass/fail)\b",
    re.I,
)
BOILERPLATE = re.compile(r"\b(table of contents|see section l|instructions to offerors)\b", re.I)
LPTA_CUE = re.compile(r"\b(LPTA|lowest price technically acceptable|price only)\b", re.I)
TRADEOFF_CUE = re.compile(r"\b(trade[\s-]?off|best value trade[\s-]?off)\b", re.I)


def _authority_rank(signal: dict[str, Any]) -> float:
    ev1 = signal.get("evidence_v1") or {}
    excerpt = str(ev1.get("excerpt") or signal.get("value") or "")
    source_hint = None
    auth = signal.get("authority") or {}
    reason = str(auth.get("reason") or "")
    if "Section M" in reason:
        source_hint = "sectionMFulltext"
    elif "SF-30" in reason or "SF30" in reason:
        source_hint = "sf30"
    elif "attachment" in reason.lower():
        source_hint = "attachment"
    elif "Questions" in reason:
        source_hint = "qa"
    amendment = str(ev1.get("amendmentNumber") or "")
    logical = str(ev1.get("logicalStructureType") or ev1.get("normalizedMethod") or "")
    return evaluation_authority_score(
        source_hint=source_hint,
        excerpt=excerpt,
        amendment_number=amendment or None,
        logical_structure_type=logical if logical.startswith("SECTION") else None,
    )


def _score(signal: dict[str, Any]) -> float:
    value = str(signal.get("value") or "")
    excerpt = str((signal.get("evidence_v1") or {}).get("excerpt") or value)
    if is_section_l_submission_context(None, excerpt):
        return -100.0
    if is_marketing_summary(excerpt):
        return 5.0
    score = _authority_rank(signal)
    score += {"high": 30, "medium": 20, "low": 5}.get(str(signal.get("confidence") or "").lower(), 0)
    if STRONG_CUES.search(value) or STRONG_CUES.search(excerpt):
        score += 15
    if BOILERPLATE.search(value) or BOILERPLATE.search(excerpt):
        score -= 20
    if LPTA_CUE.search(value):
        score += 25
    if TRADEOFF_CUE.search(value):
        score += 20
    if "Review Required" in value:
        score -= 50
    return score


def _has_method_conflict(signals: list[dict[str, Any]]) -> bool:
    methods: set[str] = set()
    for signal in signals:
        if str(signal.get("id") or "") not in {"rfp_eval_method_v1", "rfp_evaluation_criteria_v1"}:
            continue
        blob = f"{signal.get('value')} {(signal.get('evidence_v1') or {}).get('excerpt')}"
        if LPTA_CUE.search(blob):
            methods.add("LPTA")
        if TRADEOFF_CUE.search(blob):
            methods.add("Tradeoff")
    return len(methods) > 1


def canonicalize_evaluation_signals(
    signals: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    by_id: dict[str, list[dict[str, Any]]] = {}
    others: list[dict[str, Any]] = []
    for signal in signals:
        signal_id = str(signal.get("id") or "")
        if signal_id in EVALUATION_IDS:
            by_id.setdefault(signal_id, []).append(signal)
        else:
            others.append(signal)

    eval_pool = [item for group in by_id.values() for item in group]
    if _has_method_conflict(eval_pool):
        findings.append(
            Finding(
                "warn",
                "EVALUATION_CANONICALIZATION_AMBIGUOUS",
                "Conflicting LPTA and tradeoff candidates; review required",
            )
        )

    chosen: list[dict[str, Any]] = []
    for signal_id in EVALUATION_IDS:
        candidates = by_id.get(signal_id) or []
        if not candidates:
            continue
        ranked = sorted(candidates, key=_score, reverse=True)
        winner = ranked[0]
        if len(ranked) > 1 and _score(ranked[0]) == _score(ranked[1]):
            findings.append(
                Finding(
                    "warn",
                    "EVALUATION_CANONICALIZATION_TIE",
                    f"Tied top candidates for {signal_id}; deterministic tie-break applied",
                    {"signalId": signal_id},
                )
            )
        if ranked[1:]:
            ev1 = winner.setdefault("evidence_v1", {})
            if isinstance(ev1, dict):
                ev1["evaluationAlternates"] = [
                    {
                        "value": alt.get("value"),
                        "confidence": alt.get("confidence"),
                        "excerpt": str((alt.get("evidence_v1") or {}).get("excerpt") or "")[:160],
                    }
                    for alt in ranked[1:6]
                ]
        if signal_id == "rfp_eval_method_v1" and "Review Required" in str(winner.get("value") or ""):
            winner = dict(winner)
            winner["confidence"] = "low"
            winner.setdefault("findings", []).append(
                {"level": "warn", "code": "EVALUATION_REVIEW_REQUIRED", "message": "Ambiguous evaluation method"}
            )
        chosen.append(winner)

    if chosen:
        findings.append(Finding("info", "EVALUATION_CANONICALIZED", "Evaluation cluster canonicalized"))
    return sorted([*others, *chosen], key=lambda item: str(item.get("id") or "")), findings
