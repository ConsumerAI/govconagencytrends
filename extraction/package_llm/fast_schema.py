from __future__ import annotations

from typing import Any

from extraction.package_llm.versions import FAST_SCHEMA_VERSION

MARKET_SCOPE_FAST_SIGNAL_IDS: tuple[str, ...] = (
    "rfp_solicitation_id_v1",
    "rfp_title_v1",
    "rfp_issuing_agency_v1",
    "rfp_issuing_office_v1",
    "rfp_office_aac_v1",
    "rfp_primary_naics_v1",
    "rfp_primary_psc_v1",
    "rfp_contract_type_v1",
    "rfp_set_aside_v1",
    "rfp_place_of_performance_v1",
    "solicitation_status_alert_v1",
)

MARKET_SCOPE_FAST_SIGNAL_COUNT = len(MARKET_SCOPE_FAST_SIGNAL_IDS)

FAST_SIGNAL_STATUSES = ("confirmed", "review_required", "not_found")
FAST_CONFIDENCE_LEVELS = ("high", "medium", "low")

FAST_COMPACT_SIGNAL_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string", "enum": list(MARKET_SCOPE_FAST_SIGNAL_IDS)},
        "value": {"type": ["string", "number", "boolean", "null"]},
        "confidence": {"type": "string", "enum": list(FAST_CONFIDENCE_LEVELS)},
        "status": {"type": "string", "enum": list(FAST_SIGNAL_STATUSES)},
        "source_id": {"type": ["string", "null"]},
        "filename": {"type": ["string", "null"]},
        "page": {"type": ["integer", "null"]},
        "sheet": {"type": ["string", "null"]},
        "quote": {"type": "string", "maxLength": 320},
        "review_note": {"type": "string", "maxLength": 200},
        "alternates": {
            "type": "array",
            "maxItems": 2,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "value": {"type": ["string", "number", "boolean", "null"]},
                    "status": {"type": "string", "enum": list(FAST_SIGNAL_STATUSES)},
                    "source_id": {"type": ["string", "null"]},
                    "filename": {"type": ["string", "null"]},
                    "page": {"type": ["integer", "null"]},
                    "sheet": {"type": ["string", "null"]},
                    "quote": {"type": "string", "maxLength": 320},
                    "reason": {"type": "string", "maxLength": 160},
                },
                "required": ["value", "status", "source_id", "filename", "page", "sheet", "quote", "reason"],
            },
        },
    },
    "required": [
        "id",
        "value",
        "confidence",
        "status",
        "source_id",
        "filename",
        "page",
        "sheet",
        "quote",
        "review_note",
        "alternates",
    ],
}

FAST_EXTRACTION_RESPONSE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "signals": {
            "type": "array",
            "items": FAST_COMPACT_SIGNAL_SCHEMA,
            "minItems": MARKET_SCOPE_FAST_SIGNAL_COUNT,
            "maxItems": MARKET_SCOPE_FAST_SIGNAL_COUNT,
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


def build_fast_openai_json_schema() -> dict:
    return FAST_EXTRACTION_RESPONSE_SCHEMA
