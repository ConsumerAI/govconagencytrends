from __future__ import annotations

import re
from typing import Any, Callable

from extraction.package_llm.fast_schema import MARKET_SCOPE_FAST_SIGNAL_COUNT, MARKET_SCOPE_FAST_SIGNAL_IDS

ALL_PRODUCT_SERVICE_CODES = "All Product Service Codes"
OPTION_SEPARATOR = "||"

AUTO_MAPPED_DASHBOARD_FILTER_FIELDS = frozenset(
    {
        "Agency",
        "Subagency / Bureau",
        "Contracting Office",
        "NAICS",
        "PSC",
        "Contract Type",
        "Set-Aside",
    }
)

FAST_SCOPE_DETAIL_LABELS: dict[str, str] = {
    "rfp_solicitation_id_v1": "Solicitation ID",
    "rfp_title_v1": "Title",
    "rfp_issuing_agency_v1": "Issuing Agency",
    "rfp_issuing_office_v1": "Issuing Office",
    "rfp_office_aac_v1": "Office AAC",
    "rfp_primary_naics_v1": "Primary NAICS",
    "rfp_primary_psc_v1": "Primary PSC",
    "rfp_contract_type_v1": "Contract Type",
    "rfp_set_aside_v1": "Set-Aside",
    "rfp_place_of_performance_v1": "Place of Performance",
    "solicitation_status_alert_v1": "Status Alert",
}


def extract_psc_code(value: object | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    match = re.fullmatch(r"[A-Z0-9]{4}", text)
    if match:
        return match.group(0)
    token = re.search(r"\b([A-Z0-9]{4})\b", text)
    return token.group(1) if token else None


def match_psc_to_options(
    extracted_value: object | None,
    psc_options: list[str],
    default_option: str = ALL_PRODUCT_SERVICE_CODES,
) -> dict[str, Any]:
    code = extract_psc_code(extracted_value)
    attempts = [{"strategy": "exact_psc_code", "candidate": extracted_value, "code": code, "matches": []}]
    if not code:
        return {
            "filter_option": None,
            "mapping_status": "Unmapped",
            "preselect": False,
            "attempts": attempts,
            "extracted_code": "",
        }
    matches = []
    for option in psc_options:
        if option == default_option:
            continue
        option_code, _label = _decode_option(option)
        if option_code.upper() == code:
            matches.append(option)
    attempts[0]["matches"] = matches
    if len(matches) == 1:
        return {
            "filter_option": matches[0],
            "mapping_status": "Exact match",
            "preselect": True,
            "attempts": attempts,
            "extracted_code": code,
        }
    return {
        "filter_option": None,
        "mapping_status": "Unmapped",
        "preselect": False,
        "attempts": attempts,
        "extracted_code": code,
    }


def _decode_option(option: str) -> tuple[str, str]:
    if OPTION_SEPARATOR in option:
        code, label = option.split(OPTION_SEPARATOR, 1)
        return code.strip(), label.strip()
    if " — " in option:
        code, label = option.split(" — ", 1)
        return code.strip(), label.strip()
    return option.strip(), option.strip()


def count_auto_mapped_dashboard_filters(
    rows: list[dict],
    *,
    is_confirmed_filter: Callable[[dict, dict], bool],
    available_filter_options: dict,
) -> int:
    count = 0
    for row in rows:
        field = row.get("field")
        if field not in AUTO_MAPPED_DASHBOARD_FILTER_FIELDS:
            continue
        if is_confirmed_filter(row, available_filter_options):
            count += 1
    return count


def solicitation_scope_review_visible(
    *,
    has_resolved_signals: bool,
    scope_applied: bool,
    review_open: bool,
) -> bool:
    if not has_resolved_signals:
        return False
    if review_open:
        return True
    return not scope_applied


def solicitation_status_alert_text(resolved_signals: dict) -> str | None:
    for signal in resolved_signals.get("signals") or []:
        if not isinstance(signal, dict) or signal.get("id") != "solicitation_status_alert_v1":
            continue
        value = signal.get("canonical_value")
        if value is None and isinstance(signal.get("package_llm"), dict):
            value = signal["package_llm"].get("modelValue")
        if value is None:
            continue
        text = str(value).strip()
        return text or None
    return None


def build_fast_scope_detail_rows(resolved_signals: dict) -> list[dict[str, Any]]:
    by_id = {
        str(item.get("id") or ""): item
        for item in resolved_signals.get("signals") or []
        if isinstance(item, dict)
    }
    rows: list[dict[str, Any]] = []
    for signal_id in MARKET_SCOPE_FAST_SIGNAL_IDS:
        signal = by_id.get(signal_id) or {}
        value = signal.get("canonical_value")
        if value is None and isinstance(signal.get("package_llm"), dict):
            raw = signal["package_llm"].get("modelValue")
            if raw is not None and not isinstance(raw, dict):
                value = raw
        if isinstance(value, dict):
            value = None
        evidence = signal.get("evidence") or {}
        legacy = evidence.get("legacy") or []
        legacy_item = legacy[0] if legacy and isinstance(legacy[0], dict) else {}
        rows.append(
            {
                "signal_id": signal_id,
                "label": FAST_SCOPE_DETAIL_LABELS.get(signal_id, signal_id),
                "value": value,
                "confidence": signal.get("canonical_confidence") or "",
                "validation_status": signal.get("resolution_status") or "",
                "source": legacy_item.get("sourceId") or "",
                "page": legacy_item.get("locator") or "",
                "quote": legacy_item.get("snippet") or "",
            }
        )
    return rows


def dedupe_extraction_findings(findings: list[dict]) -> list[dict]:
    seen: set[tuple[str, str]] = set()
    deduped: list[dict] = []
    for item in findings:
        if not isinstance(item, dict):
            continue
        key = (str(item.get("code") or ""), str(item.get("message") or ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def should_show_blocking_ocr_error(summary: dict, findings: list[dict]) -> bool:
    if int(summary.get("controllingDocumentsUnreadable") or 0) > 0:
        return True
    blocking_codes = {
        "CONTROLLING_DOCUMENT_UNREADABLE",
        "OCR_FALLBACK_EMPTY",
        "DOCUMENT_PDF_PARSE_FAILED",
    }
    for item in findings:
        if str(item.get("code") or "") in blocking_codes and str(item.get("level") or "") in {"error", "warn", "warning"}:
            return True
    return False


def is_ocr_environment_notice(message: str) -> bool:
    lowered = message.lower()
    return any(
        token in lowered
        for token in (
            "pypdfium2",
            "tesseract",
            "ocr requires",
            "ocr fallback is available",
            "ocr environment",
        )
    )


def fast_scope_requested_count(resolved_signals: dict | None) -> int:
    if not isinstance(resolved_signals, dict):
        return MARKET_SCOPE_FAST_SIGNAL_COUNT
    summary = resolved_signals.get("summary") or {}
    requested = int(summary.get("requestedSignalCount") or 0)
    if requested:
        return requested
    signal_ids = {
        str(item.get("id") or "")
        for item in resolved_signals.get("signals") or []
        if isinstance(item, dict)
    }
    signal_ids.discard("process_sources_v1")
    if signal_ids:
        return len(signal_ids)
    return MARKET_SCOPE_FAST_SIGNAL_COUNT


OPTIONAL_ADDITIONAL_FILTER_SPECS: list[tuple[str, str]] = [
    ("Contracting Office", "contracting_office"),
    ("Funding Office", "funding_office"),
    ("NAICS", "naics_code"),
    ("PSC", "psc_code"),
    ("Contract Type", "contract_type"),
    ("Set-Aside", "set_aside_type"),
    ("Place of Performance", "pop_state"),
]


def is_solicitation_filter_removed(field_name: str, removed_fields: set[str] | frozenset[str]) -> bool:
    return field_name in removed_fields


def is_solicitation_auto_applied_row(
    row: dict,
    *,
    is_confirmed_filter: Callable[[dict, dict], bool],
    available_filter_options: dict,
) -> bool:
    return bool(row.get("filter_key") and is_confirmed_filter(row, available_filter_options))


def should_show_solicitation_additional_filter(
    row: dict,
    *,
    removed_fields: set[str] | frozenset[str],
    funding_office_extracted: bool,
    is_confirmed_filter: Callable[[dict, dict], bool],
    requires_user_confirmation: Callable[[dict], bool],
    available_filter_options: dict,
) -> bool:
    field_name = row.get("field") or ""
    if is_solicitation_filter_removed(field_name, removed_fields):
        return True
    if requires_user_confirmation(row):
        return True
    if is_solicitation_auto_applied_row(
        row,
        is_confirmed_filter=is_confirmed_filter,
        available_filter_options=available_filter_options,
    ):
        return False
    if field_name == "Funding Office":
        return True
    if row.get("extracted_value") is not None:
        return True
    if row.get("filter_option") and row.get("mapping_status") in {"Suggested match", "Unmapped"}:
        return True
    if row.get("mapping_status") == "Suggested match":
        return True
    if str(row.get("confidence") or "").strip().lower() in {"medium", "low"} and row.get("filter_key"):
        return True
    if field_name == "Funding Office" and not funding_office_extracted:
        return True
    return False


def build_solicitation_additional_filter_rows(
    rows: list[dict],
    *,
    removed_fields: set[str] | frozenset[str],
    funding_office_extracted: bool,
    is_confirmed_filter: Callable[[dict, dict], bool],
    requires_user_confirmation: Callable[[dict], bool],
    available_filter_options: dict,
) -> list[dict]:
    rows_by_field = {
        str(row.get("field") or ""): row
        for row in rows
        if isinstance(row, dict) and row.get("field")
    }
    additional_rows: list[dict] = []
    for field_name, filter_key in OPTIONAL_ADDITIONAL_FILTER_SPECS:
        row = rows_by_field.get(field_name)
        if not row:
            row = {
                "field": field_name,
                "filter_key": filter_key,
                "extracted_value": None,
                "filter_option": None,
                "mapping_status": "Unmapped",
                "confidence": None,
                "evidence_snippet": "",
            }
        if should_show_solicitation_additional_filter(
            row,
            removed_fields=removed_fields,
            funding_office_extracted=funding_office_extracted,
            is_confirmed_filter=is_confirmed_filter,
            requires_user_confirmation=requires_user_confirmation,
            available_filter_options=available_filter_options,
        ):
            additional_rows.append(row)
    return additional_rows


def resolve_solicitation_filter_pending_value(
    row: dict,
    *,
    removed_fields: set[str] | frozenset[str],
    user_override: str | None,
    keep_current_token: str,
    keep_removed_token: str,
    is_confirmed_filter: Callable[[dict, dict], bool],
    option_is_valid: Callable[[str | None, str | None, dict], bool],
    available_filter_options: dict,
) -> str | None:
    field_name = row.get("field") or ""
    filter_key = row.get("filter_key")
    if not filter_key:
        return None
    if user_override == keep_removed_token:
        return None
    if user_override and user_override != keep_current_token:
        if option_is_valid(filter_key, user_override, available_filter_options):
            return user_override
        return None
    if is_solicitation_filter_removed(field_name, removed_fields):
        return None
    if is_confirmed_filter(row, available_filter_options):
        selected = row.get("filter_option")
        if option_is_valid(filter_key, selected, available_filter_options):
            return selected
    return None


def begin_comparable_market_scope(current_office: str, default_all_offices: str) -> dict[str, object]:
    if current_office != default_all_offices:
        return {
            "active_contracting_office": default_all_offices,
            "saved_contracting_office": current_office,
            "comparable_market": True,
        }
    return {
        "active_contracting_office": current_office,
        "saved_contracting_office": None,
        "comparable_market": True,
    }


def restore_exact_market_scope(saved_office: str | None, default_all_offices: str) -> dict[str, object]:
    if saved_office:
        return {
            "active_contracting_office": saved_office,
            "comparable_market": False,
        }
    return {
        "active_contracting_office": default_all_offices,
        "comparable_market": False,
    }
