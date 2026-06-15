from __future__ import annotations

from functools import cmp_to_key
from typing import Any

from extraction.resolve.helpers import (
    build_evidence,
    build_source_summary,
    cap_confidence,
    evidence_specificity_score,
    has_derived_finding,
    has_usable_evidence,
    normalize_comparable_value,
    normalize_confidence,
    normalize_scalar,
    rank_confidence,
    read_authority_tier,
    read_evidence_v1,
    read_legacy_evidence,
    stable_scalar_key,
)
from extraction.resolve.types import RESOLVED_SIGNALS_VERSION_V1


def _is_valid_candidate(signal: dict[str, Any]) -> bool:
    signal_id = str(signal.get("id") or "").strip()
    if not signal_id:
        return False
    if normalize_scalar(signal.get("value")) is None:
        return False
    return has_usable_evidence(signal)


def _compare_candidates(a: dict[str, Any], b: dict[str, Any]) -> int:
    a_value = normalize_scalar(a.get("value"))
    b_value = normalize_scalar(b.get("value"))
    if (a_value is None) != (b_value is None):
        return -1 if a_value is not None else 1

    conf_rank = rank_confidence(normalize_confidence(b.get("confidence"))) - rank_confidence(
        normalize_confidence(a.get("confidence"))
    )
    if conf_rank:
        return conf_rank

    spec_rank = evidence_specificity_score(b) - evidence_specificity_score(a)
    if spec_rank:
        return spec_rank

    ev1_rank = int(read_evidence_v1(b) is not None) - int(read_evidence_v1(a) is not None)
    if ev1_rank:
        return ev1_rank

    legacy_rank = len(read_legacy_evidence(b)) - len(read_legacy_evidence(a))
    if legacy_rank:
        return legacy_rank

    tier_rank = read_authority_tier(a) - read_authority_tier(b)
    if tier_rank:
        return tier_rank

    a_key = stable_scalar_key(a_value)
    b_key = stable_scalar_key(b_value)
    if a_key != b_key:
        return -1 if a_key < b_key else 1

    return int(a.get("_index", 0)) - int(b.get("_index", 0))


def _resolve_status(
    candidate_count: int,
    chosen: dict[str, Any],
    has_conflict: bool,
    equivalent_only: bool,
) -> str:
    if has_derived_finding(chosen):
        return "canonical_derived"
    if has_conflict:
        return "resolved_with_conflict"
    if equivalent_only and candidate_count > 1:
        return "resolved_equivalent_candidates"
    if candidate_count == 1:
        return "passthrough"
    return "canonical_explicit"


def _resolve_basis(status: str) -> str:
    mapping = {
        "passthrough": "single supported candidate",
        "resolved_equivalent_candidates": "duplicate equivalent candidates collapsed",
        "resolved_with_conflict": "highest-confidence supported candidate selected; alternates preserved",
        "canonical_derived": "single final derived signal",
        "canonical_explicit": "duplicate_candidates_same_value",
        "resolved_with_promotion": "promoted_from_authoritative_evidence",
    }
    return mapping.get(status, "resolved_from_final_signal_set")


def _build_alternate(signal: dict[str, Any], reason: str) -> dict[str, Any]:
    return {
        "value": normalize_scalar(signal.get("value")),
        "confidence": normalize_confidence(signal.get("confidence")),
        "evidence": build_evidence(signal),
        "source_summary": build_source_summary([signal]),
        "reason_not_selected": reason,
    }


def _normalize_canonical_value(signal_id: str, value: object | None, all_signals: list[dict[str, Any]]) -> object | None:
    scalar = normalize_scalar(value)
    if signal_id == "rfp_period_of_performance_v1" and scalar is None:
        from extraction.postprocess.pop_compute import compute_period_of_performance_summary

        pop_start = next(
            (normalize_scalar(s.get("value")) for s in all_signals if s.get("id") == "rfp_pop_start_v1"),
            None,
        )
        pop_end = next(
            (normalize_scalar(s.get("value")) for s in all_signals if s.get("id") == "rfp_pop_end_v1"),
            None,
        )
        pop_years = next(
            (normalize_scalar(s.get("value")) for s in all_signals if s.get("id") == "rfp_pop_years_v1"),
            None,
        )
        return compute_period_of_performance_summary(
            str(pop_start) if pop_start is not None else None,
            str(pop_end) if pop_end is not None else None,
            str(pop_years) if pop_years is not None else None,
        )
    return scalar


def resolve_one_signal(
    signal_id: str,
    candidates: list[dict[str, Any]],
    all_signals: list[dict[str, Any]],
) -> dict[str, Any] | None:
    valid = [signal for signal in candidates if _is_valid_candidate(signal)]
    if not valid:
        return None

    indexed = [{**signal, "_index": index} for index, signal in enumerate(valid)]
    indexed.sort(key=cmp_to_key(_compare_candidates))
    chosen = indexed[0]

    normalized_values = [_normalize_canonical_value(signal_id, item.get("value"), all_signals) for item in indexed]
    distinct = {
        normalize_comparable_value(value)
        for value in normalized_values
        if value is not None and normalize_comparable_value(value)
    }
    has_conflict = len(distinct) > 1
    equivalent_only = not has_conflict and len(indexed) > 1

    chosen_value = _normalize_canonical_value(signal_id, chosen.get("value"), all_signals)
    if chosen_value is None:
        return None

    chosen_confidence = normalize_confidence(chosen.get("confidence"))
    if has_conflict:
        chosen_confidence = cap_confidence(chosen_confidence, "medium")

    status = _resolve_status(len(indexed), chosen, has_conflict, equivalent_only)
    alternates = [
        _build_alternate(
            item,
            "lower-ranked conflicting candidate" if has_conflict else "equivalent duplicate not selected",
        )
        for item in indexed[1:]
    ]

    notes_parts: list[str] = []
    if len(indexed) == 1:
        notes_parts.append("Resolved from one supported candidate with direct evidence.")
    if equivalent_only:
        notes_parts.append(f"Collapsed {len(indexed) - 1} equivalent duplicate candidate(s).")
    if has_conflict:
        notes_parts.append("Conflicting candidates preserved in alternates.")

    record: dict[str, Any] = {
        "id": signal_id,
        "canonical_value": chosen_value,
        "canonical_confidence": chosen_confidence,
        "resolution_status": status,
        "resolution_basis": _resolve_basis(status),
        "source_summary": build_source_summary(indexed),
        "evidence": build_evidence(chosen),
        "alternates": alternates,
        "notes": " ".join(notes_parts).strip(),
    }
    if isinstance(chosen.get("docset_provenance"), dict):
        record["docset_provenance"] = chosen["docset_provenance"]
    elif isinstance(chosen.get("_docset"), dict):
        docset = chosen["_docset"]
        record["docset_provenance"] = {
            "controlling_source_id": docset.get("source_id"),
            "controlling_filename": docset.get("filename"),
            "controlling_document_type": docset.get("document_type"),
            "controlling_amendment": docset.get("amendment_order"),
            "supersedes": [],
        }
    if status in {"passthrough", "resolved_equivalent_candidates", "canonical_derived"}:
        record["preservation"] = {
            "immutable": True,
            "contract": "passthrough_single_final_or_promoted",
        }
    return record


def build_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    derived_count = 0
    resolved_count = 0
    alternates_count = 0

    for record in records:
        status = str(record.get("resolution_status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        confidence = str(record.get("canonical_confidence") or "low")
        confidence_counts[confidence] = confidence_counts.get(confidence, 0) + 1
        if record.get("canonical_value") is not None:
            resolved_count += 1
        if status in {"canonical_derived", "resolved_with_promotion"} or record.get("promotion"):
            derived_count += 1
        alternates_count += len(record.get("alternates") or [])

    return {
        "total_signal_ids": len(records),
        "resolved_signal_count": resolved_count,
        "signals_with_alternates": sum(1 for record in records if record.get("alternates")),
        "status_counts": status_counts,
        "confidence_counts": confidence_counts,
        "derived_signal_count": derived_count,
        "alternates_count": alternates_count,
    }


def resolve_signals_v1(run_id: str, signals: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for index, signal in enumerate(signals):
        if not isinstance(signal, dict):
            continue
        signal_id = str(signal.get("id") or "").strip()
        if not signal_id:
            continue
        grouped.setdefault(signal_id, []).append({**signal, "_index": index})

    resolved_records: list[dict[str, Any]] = []
    for signal_id in sorted(grouped.keys()):
        record = resolve_one_signal(signal_id, grouped[signal_id], signals)
        if record is not None:
            resolved_records.append(record)

    artifact = {
        "version": RESOLVED_SIGNALS_VERSION_V1,
        "runId": run_id,
        "signals": resolved_records,
        "summary": build_summary(resolved_records),
    }
    from extraction.resolve.enhance_resolved import enhance_resolved_signals_v1

    return enhance_resolved_signals_v1(artifact)
