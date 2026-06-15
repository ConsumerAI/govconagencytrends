from __future__ import annotations

from typing import Any

from extraction.parity.critical_signals import CRITICAL_SIGNAL_IDS
from extraction.resolve.helpers import normalize_comparable_value


def _index_signals(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in payload.get("signals") or []:
        if isinstance(item, dict):
            signal_id = str(item.get("id") or "").strip()
            if signal_id:
                out[signal_id] = item
    return out


def _excerpt(record: dict[str, Any]) -> str:
    evidence = record.get("evidence") or {}
    ev1 = evidence.get("evidence_v1") if isinstance(evidence, dict) else None
    if isinstance(ev1, dict) and ev1.get("excerpt"):
        return str(ev1["excerpt"])[:280]
    legacy = evidence.get("legacy") if isinstance(evidence, dict) else None
    if isinstance(legacy, list) and legacy:
        return str(legacy[0].get("snippet") or "")[:280]
    return ""


def _controlling_filename(record: dict[str, Any]) -> str | None:
    prov = record.get("docset_provenance")
    if isinstance(prov, dict) and prov.get("controlling_filename"):
        return str(prov["controlling_filename"])
    summary = record.get("source_summary") or {}
    sources = summary.get("evidence_v1_sources") or summary.get("legacy_source_ids") or []
    if sources:
        return str(sources[0])
    return None


def _controlling_amendment(record: dict[str, Any]) -> str | None:
    prov = record.get("docset_provenance")
    if isinstance(prov, dict) and prov.get("controlling_amendment"):
        return str(prov["controlling_amendment"])
    ev = record.get("evidence") or {}
    ev1 = ev.get("evidence_v1") if isinstance(ev, dict) else None
    if isinstance(ev1, dict) and ev1.get("amendmentNumber"):
        return str(ev1["amendmentNumber"])
    return None


def _authority_tier(record: dict[str, Any]) -> int | None:
    summary = record.get("source_summary") or {}
    tiers = summary.get("authority_tiers") or []
    if tiers:
        return int(tiers[0])
    return None


def ui_disposition(record: dict[str, Any] | None) -> str:
    if not record:
        return "unmapped"
    status = str(record.get("resolution_status") or "")
    confidence = str(record.get("canonical_confidence") or record.get("confidence") or "").lower()
    value = record.get("canonical_value") if "canonical_value" in record else record.get("value")
    findings = record.get("findings") or []
    codes = {str(item.get("code") or "") for item in findings if isinstance(item, dict)}

    if status in {"unresolved_conflict"} or record.get("_withheld"):
        return "withheld"
    if "EVALUATION_REVIEW_REQUIRED" in codes or "Review Required" in str(value or ""):
        return "review required"
    if value is None or str(value).strip() == "":
        return "unmapped"
    if status in {"resolved_with_conflict"}:
        return "review required"
    if status in {"resolved_with_promotion"} or confidence == "medium":
        return "suggested"
    if status in {"passthrough", "resolved_equivalent_candidates", "canonical_explicit", "canonical_derived"} and confidence == "high":
        return "confirmed"
    if confidence == "low":
        return "review required"
    return "suggested"


def classify_comparison(
    *,
    golden: dict[str, Any] | None,
    local: dict[str, Any] | None,
) -> str:
    if golden is None and local is None:
        return "unmapped"
    if golden is None:
        return "extra locally"
    if local is None:
        return "missing locally"

    local_status = str(local.get("resolution_status") or "")
    if local_status == "unresolved_conflict" or local.get("_withheld"):
        return "withheld due to conflict"

    g_val = golden.get("canonical_value")
    l_val = local.get("canonical_value")
    if g_val is None and l_val is None:
        return "exact match"
    if g_val is None or l_val is None:
        return "material mismatch"

    if str(g_val) == str(l_val):
        return "exact match"
    if normalize_comparable_value(g_val) == normalize_comparable_value(l_val):
        return "normalized-equivalent match"
    return "material mismatch"


def compare_resolved_signals(
    *,
    golden_payload: dict[str, Any],
    local_payload: dict[str, Any],
    signal_ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    ids = signal_ids or CRITICAL_SIGNAL_IDS
    golden_by_id = _index_signals(golden_payload)
    local_by_id = _index_signals(local_payload)

    rows: list[dict[str, Any]] = []
    counts = {
        "exact match": 0,
        "normalized-equivalent match": 0,
        "material mismatch": 0,
        "missing locally": 0,
        "extra locally": 0,
        "withheld due to conflict": 0,
    }
    material_mismatches: list[str] = []
    missing_critical: list[str] = []

    for signal_id in ids:
        golden = golden_by_id.get(signal_id)
        local = local_by_id.get(signal_id)
        comparison = classify_comparison(golden=golden, local=local)
        counts[comparison] = counts.get(comparison, 0) + 1
        if comparison == "material mismatch":
            material_mismatches.append(signal_id)
        if comparison == "missing locally":
            missing_critical.append(signal_id)

        rows.append(
            {
                "signalId": signal_id,
                "comparison": comparison,
                "goldenCanonicalValue": golden.get("canonical_value") if golden else None,
                "localCanonicalValue": local.get("canonical_value") if local else None,
                "goldenConfidence": golden.get("canonical_confidence") if golden else None,
                "localConfidence": local.get("canonical_confidence") if local else None,
                "localResolutionStatus": local.get("resolution_status") if local else None,
                "controllingFilename": _controlling_filename(local) if local else _controlling_filename(golden or {}),
                "controllingAmendment": _controlling_amendment(local) if local else _controlling_amendment(golden or {}),
                "authorityTier": _authority_tier(local) if local else _authority_tier(golden or {}),
                "goldenEvidenceExcerpt": _excerpt(golden) if golden else "",
                "localEvidenceExcerpt": _excerpt(local) if local else "",
                "localAlternates": local.get("alternates") if local else [],
                "localUiDisposition": ui_disposition(local),
                "goldenUiDisposition": ui_disposition(golden),
            }
        )

    extra_local = sorted(set(local_by_id) - set(ids))
    for signal_id in extra_local:
        local = local_by_id[signal_id]
        rows.append(
            {
                "signalId": signal_id,
                "comparison": "extra locally",
                "goldenCanonicalValue": None,
                "localCanonicalValue": local.get("canonical_value"),
                "goldenConfidence": None,
                "localConfidence": local.get("canonical_confidence"),
                "localResolutionStatus": local.get("resolution_status"),
                "controllingFilename": _controlling_filename(local),
                "controllingAmendment": _controlling_amendment(local),
                "authorityTier": _authority_tier(local),
                "goldenEvidenceExcerpt": "",
                "localEvidenceExcerpt": _excerpt(local),
                "localAlternates": local.get("alternates") or [],
                "localUiDisposition": ui_disposition(local),
                "goldenUiDisposition": "unmapped",
            }
        )
        counts["extra locally"] += 1

    critical_pass = not material_mismatches and not missing_critical
    unsupported_confirmed = [
        row["signalId"]
        for row in rows
        if row.get("localUiDisposition") == "confirmed"
        and row.get("comparison") in {"material mismatch", "missing locally"}
        and row.get("localEvidenceExcerpt") == ""
    ]

    return {
        "version": 1,
        "goldenRunId": golden_payload.get("runId"),
        "localRunId": local_payload.get("runId"),
        "criticalSignalIds": list(ids),
        "passed": critical_pass and not unsupported_confirmed,
        "summary": {
            "counts": counts,
            "materialMismatches": material_mismatches,
            "missingCriticalLocally": missing_critical,
            "unsupportedConfirmed": unsupported_confirmed,
        },
        "signals": rows,
    }
