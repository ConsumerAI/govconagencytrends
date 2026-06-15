from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

AMENDMENT_ORDER_SOURCES = ("sf30", "body", "form", "filename", "unknown")


@dataclass
class Sf30FieldEvidence:
    field: str
    value: str
    excerpt: str
    locator: str


@dataclass
class Sf30ParseResult:
    is_sf30: bool = False
    solicitation_number: str | None = None
    amendment_number: str | None = None
    amendment_order: str | None = None
    amendment_order_source: str = "unknown"
    amendment_order_confidence: str = "unknown"
    effective_date: str | None = None
    acknowledgement_required: bool | None = None
    modification_language: list[str] = field(default_factory=list)
    revised_attachments: list[str] = field(default_factory=list)
    deleted_fields: list[str] = field(default_factory=list)
    field_evidence: list[Sf30FieldEvidence] = field(default_factory=list)


SF30_MARKERS = (
    re.compile(r"\bstandard\s+form\s+30\b", re.I),
    re.compile(r"\bsf\s*[-]?\s*30\b", re.I),
    re.compile(r"\bamendment\s+of\s+contract\b", re.I),
)

AMENDMENT_NUM_RE = re.compile(
    r"(?:amendment|modification)\s*(?:no\.?|number|#)?\s*(\d{1,4})",
    re.I,
)
SOLICITATION_RE = re.compile(
    r"(?:solicitation\s*(?:no\.?|number)?|contract\s*no\.?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-./]{4,})",
    re.I,
)
EFFECTIVE_DATE_RE = re.compile(
    r"(?:effective|issue)\s+date\s*[:\-]?\s*([^\n;]{4,40})",
    re.I,
)
ACK_RE = re.compile(r"\backnowledg(?:e|ement)\s+(?:is\s+)?required\b", re.I)
REPLACE_ATTACH_RE = re.compile(
    r"\b(?:replace|revis(?:e|es|ed)|supersed(?:e|es|ed))\s+(?:attachment|exhibit)\s*(?:no\.?|#)?\s*(\d+|[A-Z])",
    re.I,
)
DELETE_RE = re.compile(r"\b(?:delete|remove|cancel)\s+(?:item|block|section|requirement|page\s+limit|naics|psc)\b[^\n]{0,80}", re.I)


def _normalize_order(raw: str | None) -> str | None:
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    try:
        value = int(digits)
    except ValueError:
        return None
    if value < 0 or value > 9999:
        return None
    return f"{value:04d}"


def _record(result: Sf30ParseResult, field: str, value: str, excerpt: str, locator: str) -> None:
    result.field_evidence.append(Sf30FieldEvidence(field=field, value=value, excerpt=excerpt[:280], locator=locator))


def parse_sf30_metadata(text: str, *, filename: str = "") -> Sf30ParseResult:
    result = Sf30ParseResult()
    normalized = text or ""
    head = normalized[:6000]
    if not any(marker.search(head) for marker in SF30_MARKERS):
        return result

    result.is_sf30 = True
    _record(result, "form_type", "SF-30", head[:120], "sf30:header")

    sol_match = SOLICITATION_RE.search(head)
    if sol_match:
        result.solicitation_number = sol_match.group(1).upper()
        _record(result, "solicitation_number", result.solicitation_number, sol_match.group(0), "sf30:solicitation")

    amend_match = AMENDMENT_NUM_RE.search(head)
    if amend_match:
        result.amendment_number = amend_match.group(0).strip()
        result.amendment_order = _normalize_order(amend_match.group(1))
        result.amendment_order_source = "sf30"
        result.amendment_order_confidence = "authoritative"
        _record(result, "amendment_number", result.amendment_number, amend_match.group(0), "sf30:amendment_number")

    eff = EFFECTIVE_DATE_RE.search(head)
    if eff:
        result.effective_date = eff.group(1).strip()
        _record(result, "effective_date", result.effective_date, eff.group(0), "sf30:effective_date")

    if ACK_RE.search(head):
        result.acknowledgement_required = True
        _record(result, "acknowledgement_required", "true", ACK_RE.search(head).group(0), "sf30:acknowledgement")

    for match in REPLACE_ATTACH_RE.finditer(normalized):
        token = match.group(1).upper()
        label = f"Attachment {token}"
        result.revised_attachments.append(label)
        result.modification_language.append(match.group(0).strip())
        _record(result, "revised_attachment", label, match.group(0), "sf30:revised_attachment")

    for match in DELETE_RE.finditer(normalized):
        result.deleted_fields.append(match.group(0).strip())
        result.modification_language.append(match.group(0).strip())
        _record(result, "deletion_language", match.group(0).strip(), match.group(0), "sf30:deletion")

    if not result.amendment_order and filename:
        from extraction.documents.amendment import detect_amendment_from_filename

        raw, order = detect_amendment_from_filename(filename)
        if order:
            result.amendment_order = order
            result.amendment_number = raw
            result.amendment_order_source = "filename"
            result.amendment_order_confidence = "provisional"

    return result


def resolve_amendment_identity(
    *,
    text: str,
    filename: str,
    sf30: Sf30ParseResult | None = None,
) -> dict[str, Any]:
    parsed = sf30 or parse_sf30_metadata(text, filename=filename)
    if parsed.is_sf30 and parsed.amendment_order and parsed.amendment_order_confidence == "authoritative":
        return {
            "amendmentRaw": parsed.amendment_number,
            "amendmentOrder": parsed.amendment_order,
            "amendmentOrderSource": "sf30",
            "amendmentOrderConfidence": "authoritative",
            "solicitationNumber": parsed.solicitation_number,
            "sf30Evidence": [item.__dict__ for item in parsed.field_evidence],
        }

    from extraction.documents.amendment import detect_amendment_from_text, detect_amendment_from_filename

    body_raw, body_order = detect_amendment_from_text(text)
    if body_order:
        return {
            "amendmentRaw": body_raw,
            "amendmentOrder": body_order,
            "amendmentOrderSource": "body",
            "amendmentOrderConfidence": "authoritative",
            "solicitationNumber": parsed.solicitation_number,
            "sf30Evidence": [item.__dict__ for item in parsed.field_evidence],
        }

    file_raw, file_order = detect_amendment_from_filename(filename)
    if file_order:
        return {
            "amendmentRaw": file_raw,
            "amendmentOrder": file_order,
            "amendmentOrderSource": "filename",
            "amendmentOrderConfidence": "provisional",
            "solicitationNumber": parsed.solicitation_number,
            "sf30Evidence": [item.__dict__ for item in parsed.field_evidence],
        }

    if parsed.is_sf30 and not parsed.amendment_order:
        return {
            "amendmentRaw": parsed.amendment_number,
            "amendmentOrder": None,
            "amendmentOrderSource": "sf30",
            "amendmentOrderConfidence": "unknown",
            "solicitationNumber": parsed.solicitation_number,
            "sf30Evidence": [item.__dict__ for item in parsed.field_evidence],
        }

    return {
        "amendmentRaw": None,
        "amendmentOrder": None,
        "amendmentOrderSource": "unknown",
        "amendmentOrderConfidence": "unknown",
        "solicitationNumber": parsed.solicitation_number,
        "sf30Evidence": [item.__dict__ for item in parsed.field_evidence],
    }
