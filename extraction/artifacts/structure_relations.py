from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from extraction.signals.authority import infer_authority_tier
from extraction.types import CorpusPage, Finding

EXCERPT_MAX = 420
WINDOW_MAX = 1200

LOGICAL_TYPES = (
    "SF30",
    "SF30_CONTINUATION",
    "SF1449",
    "SF33",
    "FORM_CONTINUATION",
    "SECTION_L",
    "SECTION_M",
    "PWS",
    "SOW",
    "SOO",
    "REVISED_PWS",
    "REVISED_SOW",
    "QA",
    "PRICING",
    "ATTACHMENT",
    "EXHIBIT",
    "CLAUSE",
    "GENERIC",
)


@dataclass
class StructureItem:
    item_id: str
    logical_type: str
    heading: str
    source_sha256: str
    filename: str
    page_index: int
    page_end_index: int
    span_hashes: list[str]
    excerpt: str
    full_text: str
    authority_tier: int
    authority_label: str
    continuation_status: str = "primary"
    parent_item_id: str | None = None
    amendment_number: str | None = None
    attachment_number: str | None = None
    superseded_item_id: str | None = None
    revised_relationship: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "itemId": self.item_id,
            "logicalType": self.logical_type,
            "heading": self.heading,
            "sourceSha256": self.source_sha256,
            "filename": self.filename,
            "pageIndex": self.page_index,
            "pageEndIndex": self.page_end_index,
            "pageRange": [self.page_index, self.page_end_index],
            "spanHashes": self.span_hashes,
            "excerpt": self.excerpt,
            "fullText": self.full_text,
            "authorityTier": self.authority_tier,
            "authorityLabel": self.authority_label,
            "continuationStatus": self.continuation_status,
            "parentItemId": self.parent_item_id,
            "amendmentNumber": self.amendment_number,
            "attachmentNumber": self.attachment_number,
            "supersededItemId": self.superseded_item_id,
            "revisedRelationship": self.revised_relationship,
        }
        if self.logical_type == "SECTION_L":
            payload.update({"kind": "SECTION", "id": "L", "title": self.heading})
        elif self.logical_type == "SECTION_M":
            payload.update({"kind": "SECTION", "id": "M", "title": self.heading})
        elif self.logical_type in {"ATTACHMENT", "EXHIBIT"}:
            payload.update({"kind": "ATTACHMENT", "id": self.logical_type, "title": self.heading})
        return payload


def _truncate(text: str, limit: int = EXCERPT_MAX) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _span_hash(source_sha256: str, page_index: int, start: int, end: int) -> str:
    payload = f"{source_sha256}:{page_index}:{start}:{end}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_amendment(raw: str | None) -> str | None:
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    try:
        value = int(digits)
    except ValueError:
        return None
    return f"{value:04d}"


HEADING_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("SF30_CONTINUATION", re.compile(r"\b(?:SF\s*[-]?\s*30\s+)?CONTINUATION\s+(?:OF\s+)?(?:SOLICITATION|CONTRACT|AMENDMENT)\b", re.I), "continuation"),
    ("SF30", re.compile(r"\b(?:STANDARD\s+FORM\s+30|SF\s*[-]?\s*30)\b", re.I), "primary"),
    ("FORM_CONTINUATION", re.compile(r"\bCONTINUATION\s+(?:SHEET|PAGE|OF\s+OFFER)\b", re.I), "continuation"),
    ("SF1449", re.compile(r"\b(?:STANDARD\s+FORM\s+1449|SF\s*[-]?\s*1449)\b", re.I), "primary"),
    ("SF33", re.compile(r"\b(?:STANDARD\s+FORM\s+33|SF\s*[-]?\s*33)\b", re.I), "primary"),
    ("SECTION_L", re.compile(r"^\s*(?:SECTION|S\s*E\s*C\s*T\s*I\s*O\s*N)\s+L\b", re.I | re.M), "primary"),
    ("SECTION_M", re.compile(r"^\s*(?:SECTION|S\s*E\s*C\s*T\s*I\s*O\s*N)\s+M\b", re.I | re.M), "primary"),
    ("REVISED_PWS", re.compile(r"\bREVISED\s+(?:PERFORMANCE\s+WORK\s+STATEMENT|PWS)\b", re.I), "primary"),
    ("REVISED_SOW", re.compile(r"\bREVISED\s+(?:STATEMENT\s+OF\s+WORK|SOW)\b", re.I), "primary"),
    ("PWS", re.compile(r"\bPERFORMANCE\s+WORK\s+STATEMENT\b", re.I), "primary"),
    ("SOW", re.compile(r"\bSTATEMENT\s+OF\s+WORK\b", re.I), "primary"),
    ("SOO", re.compile(r"\bSTATEMENT\s+OF\s+OBJECTIVES\b", re.I), "primary"),
    ("QA", re.compile(r"\b(?:QUESTIONS?\s+(?:AND|&)\s+ANSWERS?|Q\s*&\s*A)\b", re.I), "primary"),
    ("PRICING", re.compile(r"\b(?:PRICE\s+PROPOSAL|PRICING\s+INSTRUCTIONS?|COST\s+PROPOSAL)\b", re.I), "primary"),
    ("ATTACHMENT", re.compile(r"^\s*(?:ATTACHMENT|EXHIBIT|APPENDIX|ANNEX)\b\s*[-:–—#]?\s*(.*)$", re.I | re.M), "primary"),
]

REVISION_PATTERNS = [
    re.compile(r"\bREPLACE\s+(?:IN\s+)?(?:ITS\s+)?ENTIRETY\b", re.I),
    re.compile(r"\bDELETE\s+AND\s+SUBSTITUTE\b", re.I),
    re.compile(r"\bINCORPORATED\s+BY\s+(?:AMENDMENT|MODIFICATION)\b", re.I),
    re.compile(r"\bREVISED\b", re.I),
    re.compile(r"\bSUPERSED(?:E|ES|ED)\b", re.I),
]


def _detect_heading(text: str, page_index: int) -> list[tuple[str, str, re.Match[str], str]]:
    hits: list[tuple[str, str, re.Match[str], str]] = []
    for logical_type, pattern, continuation in HEADING_PATTERNS:
        for match in pattern.finditer(text):
            hits.append((logical_type, continuation, match, match.group(0).strip()))
    hits.sort(key=lambda item: item[2].start())
    return hits


def _authority_for_type(logical_type: str, excerpt: str) -> tuple[int, str]:
    source_hint = None
    if logical_type.startswith("SF30"):
        source_hint = "sf30"
    elif logical_type in {"SECTION_L", "SECTION_M"}:
        source_hint = "sectionLFulltext" if logical_type == "SECTION_L" else "sectionMFulltext"
    elif logical_type in {"ATTACHMENT", "EXHIBIT"}:
        source_hint = "attachment"
    elif logical_type == "QA":
        source_hint = "qa"
    auth = infer_authority_tier(signal_id="structure", excerpt=excerpt, source_hint=source_hint)
    return int(auth["tier"]), str(auth["label"])


def build_structure_relations_v1(
    run_id: str,
    pages: list[CorpusPage],
    *,
    filename: str = "",
    amendment_number: str | None = None,
) -> dict[str, Any]:
    findings: list[Finding] = []
    items: list[StructureItem] = []
    open_parents: dict[str, str] = {}
    counter = 0

    for page in pages:
        hits = _detect_heading(page.text, page.page_index)
        for logical_type, continuation_status, match, heading in hits:
            counter += 1
            item_id = f"struct-{page.source_sha256[:8]}-{page.page_index}-{counter}"
            start = match.start()
            end = min(len(page.text), start + WINDOW_MAX)
            excerpt = _truncate(page.text[start:end].replace("\n", " ").strip())
            full_text = page.text[start:end]
            span_hashes = [_span_hash(page.source_sha256, page.page_index, start, end)]
            tier, label = _authority_for_type(logical_type, excerpt)

            parent_id: str | None = None
            if continuation_status == "continuation":
                if logical_type == "SF30_CONTINUATION":
                    parent_id = open_parents.get("SF30")
                elif logical_type == "FORM_CONTINUATION":
                    parent_id = open_parents.get("SF1449") or open_parents.get("SF33") or open_parents.get("SF30")
                if not parent_id:
                    parent_id = open_parents.get(logical_type.replace("_CONTINUATION", ""))
            elif logical_type in {"SF30", "SF1449", "SF33", "SECTION_L", "SECTION_M", "PWS", "SOW", "SOO", "QA"}:
                open_parents[logical_type.split("_")[0] if logical_type.startswith("SF") else logical_type] = item_id
                if logical_type == "SF30":
                    open_parents["SF30"] = item_id

            attach_num = None
            if logical_type == "ATTACHMENT":
                attach_match = re.search(r"(?:ATTACHMENT|EXHIBIT)\s*(?:NO\.?|#)?\s*(\d+|[A-Z])", heading, re.I)
                if attach_match:
                    attach_num = attach_match.group(1).upper()

            amend_num = amendment_number
            if logical_type.startswith("SF30"):
                amend_match = re.search(r"(?:amendment|modification)\s*(?:no\.?|number|#)?\s*(\d{1,4})", page.text[:2000], re.I)
                if amend_match:
                    amend_num = _normalize_amendment(amend_match.group(1))

            revised_relationship = None
            superseded_id = None
            context = page.text[max(0, start - 120) : min(len(page.text), end + 120)]
            if any(pat.search(context) for pat in REVISION_PATTERNS):
                revised_relationship = next(pat.pattern for pat in REVISION_PATTERNS if pat.search(context))
                if logical_type in {"REVISED_PWS", "REVISED_SOW"}:
                    base_key = "PWS" if "PWS" in logical_type else "SOW"
                    superseded_id = open_parents.get(base_key)

            item = StructureItem(
                item_id=item_id,
                logical_type=logical_type,
                heading=heading,
                source_sha256=page.source_sha256,
                filename=filename,
                page_index=page.page_index,
                page_end_index=page.page_index,
                span_hashes=span_hashes,
                excerpt=excerpt,
                full_text=full_text,
                authority_tier=tier,
                authority_label=label,
                continuation_status=continuation_status,
                parent_item_id=parent_id,
                amendment_number=amend_num,
                attachment_number=attach_num,
                superseded_item_id=superseded_id,
                revised_relationship=revised_relationship,
            )
            items.append(item)
            if logical_type in {"PWS", "SOW", "REVISED_PWS", "REVISED_SOW"}:
                open_parents["PWS" if "PWS" in logical_type else "SOW"] = item_id

    if not items:
        findings.append(Finding("info", "STRUCTURE_NO_HEADINGS", "No structure headings detected"))

    return {
        "version": 1,
        "runId": run_id,
        "items": [item.to_dict() for item in items],
        "relationships": [
            {
                "itemId": item.item_id,
                "parentItemId": item.parent_item_id,
                "supersededItemId": item.superseded_item_id,
                "revisedRelationship": item.revised_relationship,
            }
            for item in items
            if item.parent_item_id or item.superseded_item_id
        ],
        "findings": [f.to_dict() for f in findings],
    }


def structure_items_for_extraction(structure: dict[str, Any] | None) -> list[dict[str, Any]]:
    return list((structure or {}).get("items") or [])


def source_hint_for_structure_item(item: dict[str, Any]) -> str | None:
    logical = str(item.get("logicalType") or "")
    if logical.startswith("SF30"):
        return "sf30"
    if logical == "SECTION_L":
        return "sectionLFulltext"
    if logical == "SECTION_M":
        return "sectionMFulltext"
    if logical == "QA":
        return "qa"
    if logical in {"ATTACHMENT", "EXHIBIT", "PWS", "SOW", "SOO", "REVISED_PWS", "REVISED_SOW"}:
        return "attachment"
    if logical in {"SF1449", "SF33", "FORM_CONTINUATION"}:
        return "form_field"
    return "globalScan"


def is_superseded_structure_item(item: dict[str, Any], structure: dict[str, Any] | None) -> bool:
    item_id = str(item.get("itemId") or "")
    if not item_id or not structure:
        return False
    for rel in structure.get("relationships") or []:
        if str(rel.get("supersededItemId") or "") == item_id:
            return True
    return False
