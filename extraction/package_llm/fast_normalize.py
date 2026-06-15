from __future__ import annotations

from typing import Any

from extraction.package_llm.fast_schema import MARKET_SCOPE_FAST_SIGNAL_IDS


def _compact_not_found(signal_id: str) -> dict[str, Any]:
    return {
        "id": signal_id,
        "value": None,
        "confidence": "low",
        "status": "not_found",
        "source_id": None,
        "filename": None,
        "page": None,
        "sheet": None,
        "quote": "",
        "review_note": "",
        "alternates": [],
    }


def _compact_to_standard(compact: dict[str, Any]) -> dict[str, Any]:
    signal_id = str(compact.get("id") or "")
    status = str(compact.get("status") or "not_found")
    value = compact.get("value")
    if status == "not_found":
        value = None

    source_id = compact.get("source_id")
    filename = compact.get("filename")
    page = compact.get("page")
    sheet = compact.get("sheet")
    quote = str(compact.get("quote") or "").strip()

    evidence = []
    if quote and (source_id or filename):
        evidence.append(
            {
                "source_id": source_id or "",
                "filename": filename or "",
                "page": page,
                "sheet": sheet,
                "quote": quote,
            }
        )

    controlling_source = None
    if source_id or filename:
        controlling_source = {
            "source_id": source_id,
            "filename": filename,
            "page": page,
            "sheet": sheet,
            "amendment_number": None,
        }

    alternates = []
    for alt in compact.get("alternates") or []:
        if not isinstance(alt, dict):
            continue
        alternates.append(
            {
                "value": alt.get("value"),
                "status": alt.get("status") or "review_required",
                "source_id": alt.get("source_id"),
                "filename": alt.get("filename"),
                "page": alt.get("page"),
                "sheet": alt.get("sheet"),
                "quote": alt.get("quote"),
                "reason": alt.get("reason") or "alternate retained",
            }
        )

    review_note = str(compact.get("review_note") or "").strip()
    reasoning_summary = review_note if status == "review_required" and review_note else ""

    return {
        "id": signal_id,
        "value": value,
        "confidence": str(compact.get("confidence") or "low"),
        "status": status,
        "controlling_source": controlling_source,
        "evidence": evidence,
        "reasoning_summary": reasoning_summary,
        "alternates": alternates,
    }


def normalize_fast_extraction_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Convert compact fast-scope GPT output into the standard package validation shape."""
    signals_in = payload.get("signals") if isinstance(payload, dict) else None
    if not isinstance(signals_in, list):
        return {"signals": [], "package_summary": payload.get("package_summary") if isinstance(payload, dict) else {}}

    by_id = {str(item.get("id") or ""): item for item in signals_in if isinstance(item, dict)}
    normalized = []
    for signal_id in MARKET_SCOPE_FAST_SIGNAL_IDS:
        compact = by_id.get(signal_id)
        if not isinstance(compact, dict):
            compact = _compact_not_found(signal_id)
        normalized.append(_compact_to_standard(compact))

    return {
        "signals": normalized,
        "package_summary": payload.get("package_summary") if isinstance(payload, dict) else {},
    }
