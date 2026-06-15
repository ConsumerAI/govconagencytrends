from __future__ import annotations

import json
import re
from typing import Any

from extraction.resolve.types import SignalConfidence


def normalize_scalar(value: object | None) -> str | int | float | bool | None:
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        if isinstance(value, str) and not value.strip():
            return None
        return value
    return None


def normalize_confidence(value: object | None) -> SignalConfidence | None:
    if value in {"high", "medium", "low"}:
        return value
    return None


def rank_confidence(value: SignalConfidence | None) -> int:
    if value == "high":
        return 3
    if value == "medium":
        return 2
    if value == "low":
        return 1
    return 0


def cap_confidence(a: SignalConfidence | None, b: SignalConfidence | None) -> SignalConfidence | None:
    ra = rank_confidence(a)
    rb = rank_confidence(b)
    weaker = a if ra <= rb else b
    return weaker if ra and rb else a or b


def read_legacy_evidence(signal: dict[str, Any]) -> list[dict[str, Any]]:
    evidence = signal.get("evidence")
    if not isinstance(evidence, list):
        return []
    out: list[dict[str, Any]] = []
    for item in evidence:
        if not isinstance(item, dict):
            continue
        if all(key in item for key in ("sourceId", "artifact", "locator", "snippet")):
            out.append(item)
    return out


def read_evidence_v1(signal: dict[str, Any]) -> dict[str, Any] | None:
    evidence = signal.get("evidence_v1")
    if not isinstance(evidence, dict):
        return None
    source = evidence.get("source")
    excerpt = evidence.get("excerpt")
    if not isinstance(source, str) or not source.strip():
        return None
    span_hashes = evidence.get("spanHashes")
    if not isinstance(span_hashes, list):
        span_hashes = []
    return {
        "source": source,
        "excerpt": str(excerpt or ""),
        "spanHashes": [str(item) for item in span_hashes if isinstance(item, str)],
    }


def has_usable_evidence(signal: dict[str, Any]) -> bool:
    if str(signal.get("id") or "") == "process_sources_v1":
        return True
    if read_legacy_evidence(signal):
        return True
    ev1 = read_evidence_v1(signal)
    return bool(ev1 and str(ev1.get("excerpt") or "").strip())


def read_authority_tier(signal: dict[str, Any]) -> int:
    authority = signal.get("authority")
    if isinstance(authority, dict):
        tier = authority.get("tier")
        if isinstance(tier, (int, float)):
            return int(tier)
    return 99


def has_derived_finding(signal: dict[str, Any]) -> bool:
    findings = signal.get("findings")
    if not isinstance(findings, list):
        return False
    for finding in findings:
        if isinstance(finding, dict):
            code = str(finding.get("code") or "")
            if code.startswith("DERIVED_") or code.startswith("PIID_") or code.startswith("POP_"):
                return True
    return False


def read_findings(signals: list[dict[str, Any]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for signal in signals:
        findings = signal.get("findings")
        if not isinstance(findings, list):
            continue
        for finding in findings:
            if isinstance(finding, dict) and finding.get("code"):
                out.append({"code": str(finding["code"])})
    return out


def build_evidence(signal: dict[str, Any]) -> dict[str, Any]:
    return {
        "legacy": read_legacy_evidence(signal),
        "evidence_v1": read_evidence_v1(signal),
    }


def build_source_summary(signals: list[dict[str, Any]]) -> dict[str, Any]:
    confidences = sorted(
        {c for c in (normalize_confidence(s.get("confidence")) for s in signals) if c},
        key=lambda item: (-rank_confidence(item), item),
    )
    legacy_source_ids = sorted(
        {
            str(item.get("sourceId"))
            for signal in signals
            for item in read_legacy_evidence(signal)
            if item.get("sourceId")
        }
    )
    evidence_v1_sources = sorted(
        {
            str(read_evidence_v1(signal).get("source"))
            for signal in signals
            if read_evidence_v1(signal)
        }
    )
    authority_tiers = sorted(
        {read_authority_tier(signal) for signal in signals if read_authority_tier(signal) < 99}
    )
    finding_codes = sorted({item["code"] for item in read_findings(signals)})
    return {
        "candidate_count": len(signals),
        "confidence_values": confidences,
        "legacy_source_ids": legacy_source_ids,
        "evidence_v1_sources": evidence_v1_sources,
        "authority_tiers": authority_tiers,
        "finding_codes": finding_codes,
    }


def stable_scalar_key(value: object | None) -> str:
    return json.dumps([type(value).__name__, value], sort_keys=True)


def normalize_comparable_value(value: object | None) -> str:
    if value is None:
        return ""
    text = str(value).strip().lower()
    text = re.sub(r"\s+", " ", text)
    return text


def evidence_specificity_score(signal: dict[str, Any]) -> int:
    score = 0
    legacy = read_legacy_evidence(signal)
    score += len(legacy) * 2
    ev1 = read_evidence_v1(signal)
    if ev1 and ev1.get("excerpt"):
        score += 3
    snippet_text = " ".join(str(item.get("snippet") or "") for item in legacy)
    labeled_patterns = [
        r"\bNAICS\b",
        r"\bPSC\b",
        r"\bSOLICITATION NUMBER\b",
        r"\bISSUED BY\b",
        r"\bCONTRACT TYPE\b",
        r"\bPERIOD OF PERFORMANCE\b",
    ]
    for pattern in labeled_patterns:
        if re.search(pattern, snippet_text, flags=re.IGNORECASE):
            score += 2
    return score
