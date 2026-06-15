from __future__ import annotations

import re

DocumentType = str

TYPE_BASE_SOLICITATION = "base_solicitation"
TYPE_AMENDMENT = "amendment"
TYPE_AMENDMENT_CONTINUATION = "amendment_continuation"
TYPE_QA = "questions_and_answers"
TYPE_PWS_SOW = "pws_sow_soo"
TYPE_SECTION_L = "section_l"
TYPE_SECTION_M = "section_m"
TYPE_PRICING = "pricing_workbook"
TYPE_ATTACHMENT = "attachment_exhibit"
TYPE_UNKNOWN = "unknown"

CLASS_A = "A"
CLASS_B = "B"
CLASS_C = "C"
CLASS_UNKNOWN = "unknown"

CLASS_A_RE = re.compile(
    r"\b(solicitation|request\s+for\s+proposal|\brfp\b|\brfq\b|\bsf[\s-]?33\b|"
    r"\bsf[\s-]?1449\b|\bsf[\s-]?30\b|modification|amendment|cover\s+page|"
    r"instructions\s+to\s+offerors)\b",
    re.IGNORECASE,
)
CLASS_B_RE = re.compile(
    r"\b(\bpws\b|\bsow\b|statement\s+of\s+work|performance\s+work\s+statement|"
    r"technical\s+requirements|schedule\s+[a-c]\b)\b",
    re.IGNORECASE,
)
CLASS_C_RE = re.compile(
    r"\b(\bcba\b|collective\s+bargaining|wage\s+determination|labor\s+agreement|"
    r"attachment\s+\d+|exhibit\s+[a-z0-9]|incumbent|historical\s+performance)\b",
    re.IGNORECASE,
)

SECTION_L_RE = re.compile(r"\b(section\s+l|instructions\s+to\s+offerors|proposal\s+instructions)\b", re.IGNORECASE)
SECTION_M_RE = re.compile(r"\b(section\s+m|evaluation\s+factors|evaluation\s+criteria)\b", re.IGNORECASE)
QA_RE = re.compile(r"\b(q\s*&\s*a|questions?\s+and\s+answers?|q\s+and\s+a)\b", re.IGNORECASE)
PRICING_RE = re.compile(r"\b(pricing|price\s+schedule|clin|boq|bill\s+of\s+quantities|worksheet)\b", re.IGNORECASE)
ATTACHMENT_RE = re.compile(r"\b(attachment|exhibit|appendix|annex)\b", re.IGNORECASE)
CONTINUATION_RE = re.compile(r"\b(continuation|continued|con't|cont\.)\b", re.IGNORECASE)


def _normalize_label(value: str) -> str:
    return re.sub(r"[_\-.]+", " ", value.strip()).lower()


def classify_document_class(file_name: str, text: str = "") -> str:
    label = _normalize_label(file_name)
    excerpt = _normalize_label(text[:2000])
    combined = f"{label} {excerpt}"
    if CLASS_C_RE.search(combined) and not CLASS_A_RE.search(combined):
        return CLASS_C
    if CLASS_A_RE.search(combined):
        return CLASS_A
    if CLASS_B_RE.search(combined):
        return CLASS_B
    if CLASS_C_RE.search(combined):
        return CLASS_C
    return CLASS_UNKNOWN


def classify_document_type(
    file_name: str,
    text: str = "",
    *,
    is_amendment: bool = False,
) -> DocumentType:
    label = _normalize_label(file_name)
    excerpt = _normalize_label(text[:3000])
    combined = f"{label} {excerpt}"

    if is_amendment:
        if CONTINUATION_RE.search(combined):
            return TYPE_AMENDMENT_CONTINUATION
        return TYPE_AMENDMENT

    if QA_RE.search(combined):
        return TYPE_QA
    if SECTION_L_RE.search(combined):
        return TYPE_SECTION_L
    if SECTION_M_RE.search(combined):
        return TYPE_SECTION_M
    if PRICING_RE.search(combined) or label.endswith((".xlsx", ".xls", ".xlsm")):
        return TYPE_PRICING
    if CLASS_B_RE.search(combined):
        return TYPE_PWS_SOW
    if ATTACHMENT_RE.search(combined):
        return TYPE_ATTACHMENT
    if CLASS_A_RE.search(combined):
        return TYPE_BASE_SOLICITATION
    return TYPE_UNKNOWN
