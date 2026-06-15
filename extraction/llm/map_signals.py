from __future__ import annotations

import json
from typing import Any

from extraction.types import Finding

GEMINI_TO_AEGIS_ID: list[tuple[str, str]] = [
    ("issuing_office", "rfp_issuing_office_v1"),
    ("issuing_agency", "rfp_issuing_agency_v1"),
    ("issuing_office_aac", "rfp_office_aac_v1"),
    ("solicitation_title", "rfp_title_v1"),
    ("solicitation_id", "rfp_solicitation_id_v1"),
    ("prior_contract_piid", "rfp_prior_contract_piid_v1"),
    ("solicitation_description", "rfp_description_v1"),
    ("requirement_name", "rfp_requirement_name_v1"),
    ("contracting_officer_name", "rfp_ko_name_v1"),
    ("contract_specialist_name", "rfp_contract_specialist_name_v1"),
    ("primary_poc", "rfp_primary_poc_v1"),
    ("primary_naics_code", "rfp_primary_naics_v1"),
    ("psc_code", "rfp_primary_psc_v1"),
    ("set_aside_type", "rfp_set_aside_v1"),
    ("competition_type", "rfp_competition_type_v1"),
    ("contract_type", "rfp_contract_type_v1"),
    ("proposal_due_date", "rfp_due_date_v1"),
    ("period_of_performance_start", "rfp_pop_start_v1"),
    ("period_of_performance_end", "rfp_pop_end_v1"),
    ("total_period_of_performance", "rfp_pop_years_v1"),
    ("places_of_performance", "rfp_place_of_performance_v1"),
    ("total_ftes", "rfp_total_ftes_v1"),
    ("odc_and_travel_plugs", "rfp_odc_plugs_v1"),
    ("productive_hours_per_fte", "rfp_productive_hours_v1"),
    ("fee_and_burden_constraints", "rfp_fee_constraints_v1"),
    ("clearance_requirements", "rfp_clearances_v1"),
    ("incumbent_data", "rfp_incumbent_data_v1"),
    ("evaluation_methodology", "rfp_eval_method_v1"),
    ("technical_factors", "rfp_tech_factors_v1"),
    ("questions_due_date", "rfp_questions_due_v1"),
    ("past_performance_requirements", "rfp_past_perf_reqs_v1"),
    ("pricing_instructions", "rfp_pricing_constraints_v1"),
    ("key_personnel_specs", "rfp_key_personnel_v1"),
    ("submission_format_instructions", "rfp_submission_format_v1"),
    ("submission_method", "rfp_submission_method_v1"),
    ("submission_destination", "rfp_submission_destination_v1"),
    ("evaluation_factors_relative_importance", "rfp_eval_weights_v1"),
    ("uncompensated_overtime_policy", "rfp_uncompensated_ot_v1"),
    ("material_markup_ceiling", "rfp_material_cap_v1"),
    ("labor_mapping_mandate", "rfp_labor_mapping_v1"),
    ("security_clearance_depth", "rfp_security_depth_v1"),
    ("page_limit_constraints", "rfp_page_limits_v1"),
    ("contract_vehicle", "rfp_contract_vehicle_v1"),
]


def _cmp_lex(a: str, b: str) -> int:
    return (a > b) - (a < b)


def _normalize_evidence(raw: object, fallback_source_file: str) -> list[dict[str, str]]:
    if isinstance(raw, list) and raw:
        citations: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            source_file = item.get("source_file")
            quote = item.get("quote")
            if isinstance(source_file, str) and isinstance(quote, str) and quote.strip():
                citations.append({"source_file": source_file, "quote": quote})
        return citations
    if isinstance(raw, str) and raw.strip():
        return [{"source_file": fallback_source_file, "quote": raw.strip()}]
    return []


def _serialize_odc_plugs_value(raw_value: object) -> str | None:
    if raw_value is None:
        return None
    if isinstance(raw_value, str):
        trimmed = raw_value.strip()
        return trimmed or None
    if isinstance(raw_value, dict):
        return json.dumps(raw_value, sort_keys=True)
    return str(raw_value).strip() or None


def _derive_confidence(status: str, confidence: str | None, has_evidence: bool) -> str:
    if confidence in {"high", "medium", "low"}:
        return confidence
    if status == "explicit" and has_evidence:
        return "high"
    if status == "derived_from_explicit_inputs":
        return "medium"
    if status.startswith("null_"):
        return "low"
    return "medium" if has_evidence else "low"


def _build_legacy_evidence(citations: list[dict[str, str]], page: int) -> list[dict[str, str]]:
    return [
        {
            "sourceId": citation["source_file"],
            "artifact": "text",
            "locator": f"page:{page}",
            "snippet": citation["quote"][:200],
        }
        for citation in citations
    ]


def _build_evidence_v1(citations: list[dict[str, str]]) -> dict[str, object]:
    excerpt = " | ".join(citation["quote"] for citation in citations)[:280]
    source = citations[0]["source_file"] if citations else "unknown"
    return {"spanHashes": [], "excerpt": excerpt, "source": source}


def map_extraction_to_signals(
    run_id: str,
    extraction_result: dict[str, Any],
    *,
    fallback_source_file: str | None = None,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    fallback = fallback_source_file or f"runs/{run_id}/sources/unknown"
    findings: list[Finding] = []
    by_id: dict[str, dict[str, Any]] = {}

    for gemini_key, aegis_id in GEMINI_TO_AEGIS_ID:
        field = extraction_result.get(gemini_key)
        raw_value = field.get("value") if isinstance(field, dict) else None
        if aegis_id == "rfp_odc_plugs_v1":
            value = _serialize_odc_plugs_value(raw_value)
        elif raw_value is not None:
            value = str(raw_value).strip() or None
        else:
            value = None

        page = field.get("page") if isinstance(field, dict) and isinstance(field.get("page"), int) else 0
        citations = _normalize_evidence(field.get("evidence") if isinstance(field, dict) else None, fallback)
        status = field.get("status") if isinstance(field, dict) and isinstance(field.get("status"), str) else "explicit"
        confidence = _derive_confidence(status, field.get("confidence") if isinstance(field, dict) else None, bool(citations))

        if value and citations:
            signal = {
                "id": aegis_id,
                "value": value,
                "confidence": confidence,
                "evidence": _build_legacy_evidence(citations, page),
                "findings": [],
                "evidence_v1": _build_evidence_v1(citations),
            }
            by_id[aegis_id] = signal
        else:
            findings.append(
                Finding(
                    "info",
                    "LLM_FIELD_MISSING",
                    f"Could not extract a value with evidence for {gemini_key}",
                )
            )

    solicitation = by_id.get("rfp_solicitation_id_v1")
    if solicitation and "rfp_solicitation_number_v1" not in by_id:
        alias = dict(solicitation)
        alias["id"] = "rfp_solicitation_number_v1"
        by_id["rfp_solicitation_number_v1"] = alias

    signals = sorted(by_id.values(), key=lambda item: str(item.get("id", "")))
    findings.sort(key=lambda item: (_cmp_lex(item.code, ""), _cmp_lex(item.message, "")))
    return signals, findings
