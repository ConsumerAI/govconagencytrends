from __future__ import annotations

import json
from typing import Any

from extraction.config import get_extraction_model, get_extraction_profile, get_extraction_reasoning_effort
from extraction.package_llm.canonicalize import canonical_scalar_for_signal
from extraction.package_llm.fast_schema import MARKET_SCOPE_FAST_SIGNAL_COUNT, MARKET_SCOPE_FAST_SIGNAL_IDS
from extraction.package_llm.schema import PACKAGE_SIGNAL_IDS
from extraction.package_llm.versions import PROFILE_VERSIONS
from extraction.resolve.types import RESOLVED_SIGNALS_VERSION_V1


def _profile_signal_ids(profile: str) -> tuple[str, ...]:
    if profile == "market_scope_fast":
        return ("process_sources_v1",) + MARKET_SCOPE_FAST_SIGNAL_IDS
    return PACKAGE_SIGNAL_IDS


def _legacy_evidence_item(item: dict[str, Any]) -> dict[str, Any]:
    page = item.get("page")
    sheet = item.get("sheet")
    locator_parts = []
    if page is not None:
        locator_parts.append(f"page:{page}")
    if sheet:
        locator_parts.append(f"sheet:{sheet}")
    return {
        "sourceId": item.get("filename") or item.get("source_id") or "",
        "artifact": "text",
        "locator": " · ".join(locator_parts),
        "snippet": str(item.get("quote") or "")[:500],
    }


def _confidence_from_signal(signal: dict[str, Any], resolution_status: str) -> str:
    if resolution_status in {"not_found", "review_required", "withheld_evidence_mismatch", "withheld_semantic_validation"}:
        return "low" if resolution_status == "not_found" else str(signal.get("confidence") or "medium")
    return str(signal.get("confidence") or "medium")


def resolve_validated_package_signals(
    run_id: str,
    validated: dict[str, Any],
    *,
    process_sources_signal: dict[str, Any],
    profile: str | None = None,
) -> dict[str, Any]:
    profile = profile or get_extraction_profile()
    target_signal_ids = _profile_signal_ids(profile)
    profile_meta = PROFILE_VERSIONS.get(profile, PROFILE_VERSIONS["package_llm_full"])
    signals_out: list[dict[str, Any]] = []

    process_record = {
        "id": "process_sources_v1",
        "canonical_value": process_sources_signal.get("value") or json.dumps(process_sources_signal.get("value")),
        "canonical_confidence": "high",
        "resolution_status": "passthrough",
        "resolution_basis": "process sources artifact",
        "source_summary": {
            "candidate_count": 1,
            "confidence_values": ["high"],
            "legacy_source_ids": [],
            "evidence_v1_sources": [],
            "authority_tiers": [],
            "finding_codes": [],
        },
        "evidence": {"legacy": [], "evidence_v1": None},
        "alternates": [],
        "notes": "Package-level extraction process sources.",
    }
    if isinstance(process_sources_signal.get("value"), str):
        process_record["canonical_value"] = process_sources_signal["value"]
    elif process_sources_signal.get("value") is not None:
        process_record["canonical_value"] = json.dumps(process_sources_signal["value"])
    else:
        process_record["canonical_value"] = json.dumps(process_sources_signal)
    signals_out.append(process_record)

    validated_signals = validated.get("signals") if isinstance(validated, dict) else []
    if not isinstance(validated_signals, list):
        validated_signals = []

    by_id = {str(item.get("id") or ""): item for item in validated_signals if isinstance(item, dict)}

    for signal_id in target_signal_ids:
        if signal_id == "process_sources_v1":
            continue
        signal = by_id.get(signal_id)
        if not signal:
            signals_out.append(
                {
                    "id": signal_id,
                    "canonical_value": None,
                    "canonical_confidence": "low",
                    "resolution_status": "not_found",
                    "resolution_basis": "signal absent from validated package extraction",
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
                    "notes": "No package extraction result for this signal id.",
                }
            )
            continue

        status = str(signal.get("status") or "not_found")
        validation_findings = signal.get("validation_findings") or []
        has_validation_issues = bool(validation_findings)
        alternates_raw = signal.get("alternates") or []
        alternates = []
        for alt in alternates_raw:
            if not isinstance(alt, dict):
                continue
            alternates.append(
                {
                    "value": alt.get("value"),
                    "confidence": "low",
                    "evidence": {"legacy": [], "evidence_v1": None},
                    "source_summary": {
                        "candidate_count": 1,
                        "confidence_values": ["low"],
                        "legacy_source_ids": [alt.get("filename") or alt.get("source_id") or ""],
                        "evidence_v1_sources": [],
                        "authority_tiers": [],
                        "finding_codes": [],
                    },
                    "reason_not_selected": alt.get("reason") or alt.get("status") or "alternate retained",
                }
            )

        validation_codes = [
            str(item.get("code") or "")
            for item in validation_findings
            if isinstance(item, dict)
        ]
        evidence_validation = signal.get("evidence_validation") if isinstance(signal.get("evidence_validation"), dict) else {}
        validated_evidence_count = int(evidence_validation.get("validatedEvidenceCount") or 0)
        authoritative_evidence_validated = bool(evidence_validation.get("authoritativeEvidenceValidated"))
        semantic_fail_codes = {
            "NAICS_CLAUSE_EXAMPLE_REJECTED",
            "SET_ASIDE_CHECKBOX_ONLY",
            "COMPETITION_CHECKBOX_ONLY",
            "DUE_DATE_AMENDMENT_EFFECTIVE",
            "SUBMISSION_POC_EMAIL",
            "SOLICITATION_ID_AMENDMENT_SUFFIX",
            "SOLICITATION_ID_PRIOR_PIID",
        }
        if any(code in semantic_fail_codes for code in validation_codes):
            resolution_status = "withheld_semantic_validation"
        elif status == "not_found":
            resolution_status = "not_found"
        elif authoritative_evidence_validated or status in {"confirmed", "not_applicable"}:
            resolution_status = "validated_model_extraction_with_alternates" if alternates else "validated_model_extraction"
        elif validated_evidence_count > 0:
            resolution_status = "review_required"
        elif any(code in {"EVIDENCE_QUOTE_NOT_FOUND", "EVIDENCE_SOURCE_MISSING"} for code in validation_codes):
            resolution_status = "withheld_evidence_mismatch"
        elif status == "review_required":
            resolution_status = "review_required"
        elif has_validation_issues:
            resolution_status = "withheld_semantic_validation"
        else:
            resolution_status = "review_required"

        value = signal.get("value")
        structured_detail = None
        keep_canonical = resolution_status in {
            "validated_model_extraction",
            "validated_model_extraction_with_alternates",
            "review_required",
        } and (
            resolution_status.startswith("validated_model_extraction")
            or validated_evidence_count > 0
            or authoritative_evidence_validated
        )
        if keep_canonical and value is not None:
            canonical_value, structured_detail = canonical_scalar_for_signal(signal_id, value)
        else:
            canonical_value = None

        evidence_items = signal.get("evidence") or []
        legacy_evidence = [_legacy_evidence_item(item) for item in evidence_items if isinstance(item, dict)]
        controlling = signal.get("controlling_source") if isinstance(signal.get("controlling_source"), dict) else {}
        evidence_v1 = None
        if legacy_evidence:
            evidence_v1 = {
                "source": legacy_evidence[0].get("sourceId") or "",
                "excerpt": legacy_evidence[0].get("snippet") or "",
                "controllingSource": controlling,
                "reasoningSummary": signal.get("reasoning_summary") or "",
            }

        signals_out.append(
            {
                "id": signal_id,
                "canonical_value": canonical_value,
                "canonical_confidence": _confidence_from_signal(signal, resolution_status),
                "resolution_status": resolution_status,
                "resolution_basis": signal.get("reasoning_summary") or f"package_llm status={status}",
                "source_summary": {
                    "candidate_count": 1,
                    "confidence_values": [str(signal.get("confidence") or "low")],
                    "legacy_source_ids": [item.get("sourceId") for item in legacy_evidence if item.get("sourceId")],
                    "evidence_v1_sources": [controlling.get("filename") or controlling.get("source_id") or ""]
                    if controlling
                    else [],
                    "authority_tiers": ["package_llm"],
                    "finding_codes": validation_codes,
                },
                "evidence": {"legacy": legacy_evidence, "evidence_v1": evidence_v1},
                "alternates": alternates,
                "notes": signal.get("reasoning_summary") or "",
                "package_llm": {
                    "status": status,
                    "modelValue": value,
                    "structuredDetail": structured_detail,
                    "controllingSource": controlling,
                    "validationFindings": validation_findings,
                    "evidenceValidation": evidence_validation,
                },
            }
        )

    status_counts: dict[str, int] = {}
    confidence_counts: dict[str, int] = {}
    alternates_count = 0
    resolved_count = 0
    for signal in signals_out:
        status = str(signal.get("resolution_status") or "")
        status_counts[status] = status_counts.get(status, 0) + 1
        conf = str(signal.get("canonical_confidence") or "low")
        confidence_counts[conf] = confidence_counts.get(conf, 0) + 1
        if signal.get("alternates"):
            alternates_count += 1
        if signal.get("canonical_value") is not None:
            resolved_count += 1

    requested_count = MARKET_SCOPE_FAST_SIGNAL_COUNT if profile == "market_scope_fast" else len(PACKAGE_SIGNAL_IDS) - 1
    return {
        "version": RESOLVED_SIGNALS_VERSION_V1,
        "runId": run_id,
        "signals": signals_out,
        "summary": {
            "profile": profile,
            "requestedSignalCount": requested_count,
            "total_signal_ids": len(target_signal_ids),
            "resolved_signal_count": resolved_count,
            "signals_with_alternates": alternates_count,
            "status_counts": status_counts,
            "confidence_counts": confidence_counts,
            "derived_signal_count": 0,
            "alternates_count": alternates_count,
            "producer": "package_llm",
            "extractionMode": "package_llm",
            "extractionProfile": profile,
            "model": get_extraction_model(),
            "reasoningEffort": get_extraction_reasoning_effort(),
            "promptVersion": profile_meta.get("promptVersion"),
            "schemaVersion": profile_meta.get("schemaVersion"),
            "corpusBuilderVersion": profile_meta.get("corpusBuilderVersion"),
            "formExtractorVersion": profile_meta.get("formExtractorVersion"),
            "validatorVersion": profile_meta.get("validatorVersion"),
            "canonicalizerVersion": profile_meta.get("canonicalizerVersion"),
            "resolverVersion": profile_meta.get("resolverVersion"),
        },
    }
