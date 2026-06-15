from __future__ import annotations

import re
from typing import Any

from extraction.documents.amendment import DELETION_PHRASE_RE, is_explicit_deletion
from extraction.types import Finding

FIELD_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "rfp_due_date_v1": [
        re.compile(r"(?:proposal|offer(?:s)?)\s+due\s+(?:date|datetime)?\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
        re.compile(r"due\s+(?:date|datetime)\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_questions_due_v1": [
        re.compile(r"questions?\s+due\s+(?:date|datetime)?\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_primary_naics_v1": [
        re.compile(r"naics\s*(?:code)?\s*[:\-]?\s*(\d{6})", re.IGNORECASE),
    ],
    "rfp_primary_psc_v1": [
        re.compile(r"psc\s*(?:code)?\s*[:\-]?\s*([A-Z0-9]{4,10})", re.IGNORECASE),
    ],
    "rfp_set_aside_v1": [
        re.compile(r"((?:8\s*\(\s*a\s*\)|small\s+business|hubzone|sdvosb)[^\n\r;]{0,80}set[\s-]?aside)", re.IGNORECASE),
    ],
    "rfp_competition_type_v1": [
        re.compile(r"\b(competitive|full\s+and\s+open\s+competition|sole\s+source)\b", re.IGNORECASE),
    ],
    "rfp_contract_type_v1": [
        re.compile(r"\b(firm\s+fixed\s+price|\bffp\b|cost[\s-]?plus|time\s+and\s+materials|\bt\s*&\s*m\b)", re.IGNORECASE),
    ],
    "rfp_submission_method_v1": [
        re.compile(r"\b(electronically|email|portal|physical\s+delivery|dod\s+safe)\b", re.IGNORECASE),
    ],
    "rfp_submission_destination_v1": [
        re.compile(r"([A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})", re.IGNORECASE),
        re.compile(r"(https?://[^\s]+)", re.IGNORECASE),
    ],
    "rfp_primary_poc_v1": [
        re.compile(r"(?:primary\s+poc|point\s+of\s+contact)\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_issuing_office_v1": [
        re.compile(r"(?:issued\s+by|contracting\s+office)\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_pop_start_v1": [
        re.compile(r"(?:pop\s+start|period\s+of\s+performance\s+start)\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_pop_end_v1": [
        re.compile(r"(?:pop\s+end|period\s+of\s+performance\s+end)\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_pop_years_v1": [
        re.compile(r"(?:pop\s+years|duration)\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_place_of_performance_v1": [
        re.compile(r"place\s+of\s+performance\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_eval_method_v1": [
        re.compile(r"(?:evaluation\s+method(?:ology)?)\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_page_limits_v1": [
        re.compile(r"page\s+limit\s*:\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_pricing_constraints_v1": [
        re.compile(r"pricing\s+instructions?\s*[:\-]?\s*([^\n\r;]+)", re.IGNORECASE),
    ],
    "rfp_prior_contract_piid_v1": [
        re.compile(r"(?:prior\s+contract\s+piid|contract\s+number)\s*[:\-]?\s*([A-Z0-9-]+)", re.IGNORECASE),
    ],
}

DELETION_FIELD_HINTS: dict[str, str] = {
    "rfp_page_limits_v1": "page limit",
    "rfp_pricing_constraints_v1": "pricing",
    "rfp_primary_naics_v1": "naics",
}


def _make_evidence(source_filename: str, snippet: str, locator: str = "page:0") -> tuple[list[dict], dict]:
    legacy = [
        {
            "sourceId": source_filename,
            "artifact": "text",
            "locator": locator,
            "snippet": snippet[:500],
        }
    ]
    ev1 = {
        "source": source_filename,
        "excerpt": snippet[:280],
        "spanHashes": [],
    }
    return legacy, ev1


def _tag_signal(
    signal: dict[str, Any],
    *,
    source_id: str,
    file_name: str,
    document_type: str,
    amendment_order: str | None,
) -> dict[str, Any]:
    tagged = dict(signal)
    tagged["_docset"] = {
        "source_id": source_id,
        "filename": file_name,
        "document_type": document_type,
        "amendment_order": amendment_order,
    }
    return tagged


def extract_deterministic_signals(
    text: str,
    *,
    source_id: str,
    file_name: str,
    document_type: str,
    amendment_order: str | None,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    signals: list[dict[str, Any]] = []
    if not text.strip():
        return signals, findings

    for signal_id, patterns in FIELD_PATTERNS.items():
        hint = DELETION_FIELD_HINTS.get(signal_id, "")
        if hint and is_explicit_deletion(text, hint):
            excerpt_match = re.search(re.escape(hint), text, flags=re.IGNORECASE)
            start = max(0, (excerpt_match.start() if excerpt_match else 0) - 40)
            end = min(len(text), (excerpt_match.end() if excerpt_match else 0) + 120)
            excerpt = text[start:end].strip()
            legacy, ev1 = _make_evidence(file_name, excerpt)
            signals.append(
                _tag_signal(
                    {
                        "id": signal_id,
                        "value": None,
                        "confidence": "high",
                        "evidence": legacy,
                        "evidence_v1": ev1,
                        "findings": [{"level": "info", "code": "FIELD_DELETED", "message": "Explicit deletion language detected"}],
                        "_deleted": True,
                    },
                    source_id=source_id,
                    file_name=file_name,
                    document_type=document_type,
                    amendment_order=amendment_order,
                )
            )
            findings.append(
                Finding(
                    "info",
                    "FIELD_EXPLICITLY_DELETED",
                    f"{signal_id} explicitly deleted in {file_name}",
                    {"signalId": signal_id, "sourceId": source_id},
                )
            )
            continue

        for pattern in patterns:
            match = pattern.search(text)
            if not match:
                continue
            raw_value = match.group(1).strip() if match.lastindex else match.group(0).strip()
            excerpt = text[max(0, match.start() - 40) : min(len(text), match.end() + 80)].strip()
            if hint and is_explicit_deletion(excerpt, hint):
                findings.append(
                    Finding(
                        "info",
                        "FIELD_EXPLICITLY_DELETED",
                        f"{signal_id} explicitly deleted in {file_name}",
                        {"signalId": signal_id, "sourceId": source_id},
                    )
                )
                legacy, ev1 = _make_evidence(file_name, excerpt)
                signals.append(
                    _tag_signal(
                        {
                            "id": signal_id,
                            "value": None,
                            "confidence": "high",
                            "evidence": legacy,
                            "evidence_v1": ev1,
                            "findings": [{"level": "info", "code": "FIELD_DELETED", "message": "Explicit deletion language detected"}],
                            "_deleted": True,
                        },
                        source_id=source_id,
                        file_name=file_name,
                        document_type=document_type,
                        amendment_order=amendment_order,
                    )
                )
                break

            if not raw_value or DELETION_PHRASE_RE.fullmatch(raw_value):
                continue

            legacy, ev1 = _make_evidence(file_name, excerpt)
            signals.append(
                _tag_signal(
                    {
                        "id": signal_id,
                        "value": raw_value,
                        "confidence": "high" if "due" in signal_id else "medium",
                        "evidence": legacy,
                        "evidence_v1": ev1,
                        "findings": [],
                    },
                    source_id=source_id,
                    file_name=file_name,
                    document_type=document_type,
                    amendment_order=amendment_order,
                )
            )
            break

    return signals, findings
