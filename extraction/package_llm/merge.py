from __future__ import annotations

from typing import Any

from extraction.package_llm.fast_schema import MARKET_SCOPE_FAST_SIGNAL_IDS
from extraction.package_llm.schema import PACKAGE_SIGNAL_IDS

_FAST_VALIDATED = frozenset(
    {
        "validated_model_extraction",
        "validated_model_extraction_with_alternates",
    }
)


def _signal_by_id(resolved: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(item.get("id") or ""): item
        for item in resolved.get("signals") or []
        if isinstance(item, dict)
    }


def _is_stronger(existing: dict[str, Any], incoming: dict[str, Any]) -> bool:
    existing_status = str(existing.get("resolution_status") or "")
    incoming_status = str(incoming.get("resolution_status") or "")
    if existing_status in _FAST_VALIDATED and incoming_status not in _FAST_VALIDATED:
        return True
    if existing_status in _FAST_VALIDATED and incoming_status in _FAST_VALIDATED:
        existing_conf = str(existing.get("canonical_confidence") or "")
        incoming_conf = str(incoming.get("canonical_confidence") or "")
        rank = {"high": 3, "medium": 2, "low": 1}
        return rank.get(existing_conf, 0) >= rank.get(incoming_conf, 0)
    return False


def merge_resolved_signals(
    base: dict[str, Any],
    incoming: dict[str, Any],
    *,
    preserve_signal_ids: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Merge full-detail extraction into an existing fast-scope run without overwriting stronger values."""
    preserve = preserve_signal_ids or frozenset(MARKET_SCOPE_FAST_SIGNAL_IDS)
    base_by_id = _signal_by_id(base)
    incoming_by_id = _signal_by_id(incoming)
    merged_ids: list[str] = []
    for signal_id in PACKAGE_SIGNAL_IDS:
        if signal_id == "process_sources_v1":
            merged_ids.append(signal_id)
            continue
        existing = base_by_id.get(signal_id)
        new_signal = incoming_by_id.get(signal_id)
        if signal_id in preserve and existing and _is_stronger(existing, new_signal or {}):
            continue
        if new_signal:
            merged_ids.append(signal_id)

    signals_out: list[dict[str, Any]] = []
    for signal_id in PACKAGE_SIGNAL_IDS:
        existing = base_by_id.get(signal_id)
        new_signal = incoming_by_id.get(signal_id)
        if signal_id in preserve and existing and new_signal and _is_stronger(existing, new_signal):
            signals_out.append(existing)
        elif new_signal:
            signals_out.append(new_signal)
        elif existing:
            signals_out.append(existing)
        else:
            signals_out.append(
                {
                    "id": signal_id,
                    "canonical_value": None,
                    "canonical_confidence": "low",
                    "resolution_status": "not_found",
                    "resolution_basis": "signal absent after merge",
                    "source_summary": {
                        "candidate_count": 0,
                        "confidence_values": [],
                        "legacy_source_ids": [],
                        "evidence_v1_sources": [],
                        "authority_tiers": [],
                        "finding_codes": [],
                    },
                    "evidence": {"legacy": [], "evidence_v1": None},
                    "alternates": [],
                    "notes": "",
                }
            )

    summary = dict(incoming.get("summary") or base.get("summary") or {})
    summary["mergeSourceRunId"] = base.get("runId")
    summary["extractionProfile"] = "package_llm_full"
    summary["mergedFromFastScope"] = True
    summary["total_signal_ids"] = len(PACKAGE_SIGNAL_IDS)
    return {
        "version": incoming.get("version") or base.get("version"),
        "runId": incoming.get("runId") or base.get("runId"),
        "signals": signals_out,
        "summary": summary,
    }
