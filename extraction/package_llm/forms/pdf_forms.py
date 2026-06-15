from __future__ import annotations

import io
import re
from typing import Any

from extraction.package_llm.forms.labels import humanize_field_name
from extraction.types import Finding

SELECTED_CHECKBOX_VALUES = {"/yes", "/on", "yes", "on", "true", "1"}
SF_FORM_PAGE_HINT = re.compile(r"page(\d+)", re.I)
SIX_DIGIT_NAICS = re.compile(r"\b(\d{6})\b")


def _field_page_index(field_name: str, default: int = 0) -> int:
    match = SF_FORM_PAGE_HINT.search(field_name)
    if match:
        return max(0, int(match.group(1)) - 1)
    return default


def _normalize_field_value(value: Any) -> tuple[str | None, bool | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if text.startswith("/"):
        lowered = text.lower()
        if lowered in {"/off", "/no"}:
            return None, False
        if lowered in SELECTED_CHECKBOX_VALUES:
            return "selected", True
        return text.lstrip("/"), True
    return text, None


def _field_type_name(field_obj: dict[str, Any]) -> str:
    ft = str(field_obj.get("/FT") or "")
    if ft == "/Btn":
        return "checkbox"
    if ft == "/Ch":
        return "choice"
    return "text"


def extract_pdf_form_fields(
    pdf_bytes: bytes,
    *,
    source_id: str,
    filename: str,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    records: list[dict[str, Any]] = []
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(pdf_bytes))
        fields = reader.get_fields() or {}
    except Exception as exc:
        findings.append(
            Finding(
                "warn",
                "PDF_FORM_EXTRACTION_FAILED",
                f"Could not read PDF form fields for {filename}",
                {"sourceId": source_id, "error": str(exc)},
            )
        )
        return records, findings

    for field_name, field_obj in fields.items():
        if not isinstance(field_obj, dict):
            continue
        field_type = _field_type_name(field_obj)
        raw_value = field_obj.get("/V")
        field_value, selected = _normalize_field_value(raw_value)
        page_index = _field_page_index(str(field_name))
        label = humanize_field_name(str(field_name))

        if field_type == "checkbox":
            if selected is not True and str(raw_value or "").lower() not in SELECTED_CHECKBOX_VALUES:
                continue
            display_value = label if field_value in {None, "selected"} else str(field_value)
            records.append(
                {
                    "source_id": source_id,
                    "filename": filename,
                    "page": page_index + 1,
                    "page_index": page_index,
                    "field_name": str(field_name),
                    "field_label": label,
                    "field_value": display_value,
                    "field_type": field_type,
                    "selected": True,
                    "provenance": "acroform",
                    "include_in_corpus": True,
                }
            )
            continue

        if not field_value:
            continue

        records.append(
            {
                "source_id": source_id,
                "filename": filename,
                "page": page_index + 1,
                "page_index": page_index,
                "field_name": str(field_name),
                "field_label": label,
                "field_value": field_value,
                "field_type": field_type,
                "selected": selected,
                "provenance": "acroform",
                "include_in_corpus": True,
            }
        )

    if records:
        findings.append(
            Finding(
                "info",
                "PDF_FORM_FIELDS_RECOVERED",
                f"Recovered {len(records)} PDF form field value(s) from {filename}",
                {"sourceId": source_id, "fieldCount": len(records)},
            )
        )
    return records, findings


def format_form_fields_corpus_block(records: list[dict[str, Any]]) -> str:
    lines = ["--- PDF FORM FIELDS ---"]
    for record in records:
        if not record.get("include_in_corpus", True):
            continue
        lines.append(f"field: {record.get('field_label') or record.get('field_name')}")
        lines.append(f"value: {record.get('field_value')}")
        lines.append(f"page: {record.get('page')}")
        lines.append(f"field_name: {record.get('field_name')}")
        lines.append(f"provenance: {record.get('provenance')}")
        lines.append("")
    return "\n".join(lines).strip()


def merge_form_fields_into_pages(
    records: list[dict[str, Any]],
    normalized_text_by_page: dict[int, str],
) -> None:
    if not records:
        return
    by_page: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        if not record.get("include_in_corpus", True):
            continue
        page_index = int(record.get("page_index") or 0)
        by_page.setdefault(page_index, []).append(record)
    for page_index, page_records in by_page.items():
        block = format_form_fields_corpus_block(page_records)
        existing = normalized_text_by_page.get(page_index, "")
        if block and block not in existing:
            normalized_text_by_page[page_index] = (existing + "\n\n" + block).strip()


def is_federal_form_page(text: str) -> bool:
    upper = (text or "").upper()
    return any(
        marker in upper
        for marker in (
            "SOLICITATION/CONTRACT/ORDER FOR COMMERCIAL",
            "STANDARD FORM 1449",
            "STANDARD FORM 30",
            "AMENDMENT OF SOLICITATION",
            "SF 30",
            "SF1449",
        )
    )


def form_page_needs_render_fallback(
    *,
    native_page_text: str,
    form_records: list[dict[str, Any]],
) -> bool:
    if form_records:
        return False
    if not is_federal_form_page(native_page_text):
        return False
    if "NAICS" in native_page_text.upper() and not SIX_DIGIT_NAICS.search(native_page_text):
        return True
    if "SOLICITATION NUMBER" in native_page_text.upper() and not re.search(r"[A-Z]{2}\d{4}[A-Z0-9]{2}[A-Z]\d{4}", native_page_text.upper()):
        return True
    return False
