from __future__ import annotations

PACKAGE_SIGNAL_IDS: tuple[str, ...] = (
    "process_sources_v1",
    "rfp_clearances_v1",
    "rfp_competition_type_v1",
    "rfp_contract_specialist_name_v1",
    "rfp_contract_type_v1",
    "rfp_description_v1",
    "rfp_eval_method_v1",
    "rfp_eval_weights_v1",
    "rfp_fee_constraints_v1",
    "rfp_incumbent_data_v1",
    "rfp_issuing_agency_v1",
    "rfp_issuing_office_v1",
    "rfp_key_personnel_v1",
    "rfp_ko_name_v1",
    "rfp_labor_mapping_v1",
    "rfp_odc_plugs_v1",
    "rfp_office_aac_v1",
    "rfp_past_perf_reqs_v1",
    "rfp_period_of_performance_v1",
    "rfp_place_of_performance_v1",
    "rfp_pop_end_v1",
    "rfp_pop_start_v1",
    "rfp_pop_years_v1",
    "rfp_pricing_constraints_v1",
    "rfp_primary_naics_v1",
    "rfp_primary_poc_v1",
    "rfp_primary_psc_v1",
    "rfp_prior_contract_piid_v1",
    "rfp_questions_due_v1",
    "rfp_requirement_name_v1",
    "rfp_set_aside_v1",
    "rfp_signals_findings_v1",
    "rfp_solicitation_id_v1",
    "rfp_strategic_intent_v1",
    "rfp_submission_destination_v1",
    "rfp_submission_format_v1",
    "rfp_submission_instructions_v1",
    "rfp_submission_method_v1",
    "rfp_tech_factors_v1",
    "rfp_title_v1",
    "solicitation_is_stalled_v1",
    "solicitation_status_alert_v1",
)

PACKAGE_EXTRACTION_SIGNAL_IDS: tuple[str, ...] = tuple(
    sid for sid in PACKAGE_SIGNAL_IDS if sid != "process_sources_v1"
)

from extraction.package_llm.versions import (  # noqa: E402
    CANONICALIZER_VERSION,
    CORPUS_BUILDER_VERSION,
    FORM_EXTRACTOR_VERSION,
    PROMPT_VERSION,
    RESOLVER_VERSION,
    SCHEMA_VERSION,
    VALIDATOR_VERSION,
)

SIGNAL_STATUSES = ("confirmed", "review_required", "not_found", "superseded", "not_applicable")
CONFIDENCE_LEVELS = ("high", "medium", "low")

SOURCE_REF_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source_id": {"type": ["string", "null"]},
        "filename": {"type": ["string", "null"]},
        "page": {"type": ["integer", "null"]},
        "sheet": {"type": ["string", "null"]},
        "amendment_number": {"type": ["string", "null"]},
    },
    "required": ["source_id", "filename", "page", "sheet", "amendment_number"],
}

EVIDENCE_ITEM_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "source_id": {"type": "string"},
        "filename": {"type": "string"},
        "page": {"type": ["integer", "null"]},
        "sheet": {"type": ["string", "null"]},
        "quote": {"type": "string"},
    },
    "required": ["source_id", "filename", "page", "sheet", "quote"],
}

ALTERNATE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "value": {},
        "status": {"type": "string", "enum": list(SIGNAL_STATUSES)},
        "source_id": {"type": ["string", "null"]},
        "filename": {"type": ["string", "null"]},
        "page": {"type": ["integer", "null"]},
        "sheet": {"type": ["string", "null"]},
        "quote": {"type": ["string", "null"]},
        "reason": {"type": ["string", "null"]},
    },
    "required": ["value", "status", "source_id", "filename", "page", "sheet", "quote", "reason"],
}

PACKAGE_SIGNAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "enum": list(PACKAGE_EXTRACTION_SIGNAL_IDS)},
        "value": {},
        "confidence": {"type": "string", "enum": list(CONFIDENCE_LEVELS)},
        "status": {"type": "string", "enum": list(SIGNAL_STATUSES)},
        "controlling_source": {"anyOf": [SOURCE_REF_SCHEMA, {"type": "null"}]},
        "evidence": {"type": "array", "items": EVIDENCE_ITEM_SCHEMA},
        "reasoning_summary": {"type": "string"},
        "alternates": {"type": "array", "items": ALTERNATE_SCHEMA},
    },
    "required": [
        "id",
        "value",
        "confidence",
        "status",
        "controlling_source",
        "evidence",
        "reasoning_summary",
        "alternates",
    ],
}

PACKAGE_EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "signals": {
            "type": "array",
            "items": PACKAGE_SIGNAL_SCHEMA,
            "minItems": len(PACKAGE_EXTRACTION_SIGNAL_IDS),
            "maxItems": len(PACKAGE_EXTRACTION_SIGNAL_IDS),
        },
        "package_summary": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "base_solicitation_filename": {"type": ["string", "null"]},
                "amendments_in_order": {"type": "array", "items": {"type": "string"}},
                "controlling_amendment": {"type": ["string", "null"]},
            },
            "required": ["base_solicitation_filename", "amendments_in_order", "controlling_amendment"],
        },
    },
    "required": ["signals", "package_summary"],
}


def build_openai_json_schema() -> dict:
    return PACKAGE_EXTRACTION_RESPONSE_SCHEMA
