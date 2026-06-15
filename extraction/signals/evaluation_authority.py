from __future__ import annotations

import re
from typing import Any

EVALUATION_AUTHORITY_ORDER: dict[str, int] = {
    "sf30": 600,
    "sectionMFulltext": 500,
    "structureMWindow": 480,
    "form_field": 400,
    "attachment": 200,
    "qa": 100,
    "globalScan": 50,
}

SECTION_L_HINTS = {"sectionLFulltext", "structureLWindow", "submissionWindow"}
MARKETING_CUES = (
    "executive summary",
    "marketing",
    "capability statement",
    "company overview",
    "win strategy",
)


def is_section_l_submission_context(source_hint: str | None, excerpt: str) -> bool:
    if source_hint in SECTION_L_HINTS:
        return True
    if excerpt.startswith("[FULLTEXT:L]"):
        return True
    if re.search(r"\bSECTION\s+L\b", excerpt, re.I) and re.search(
        r"\b(?:submission|proposal|offer|instructions?\s+to\s+offerors)\b", excerpt, re.I
    ):
        return True
    return False


def is_marketing_summary(excerpt: str) -> bool:
    lower = excerpt.lower()
    return any(cue in lower for cue in MARKETING_CUES)


def evaluation_authority_score(
    *,
    source_hint: str | None,
    excerpt: str,
    amendment_number: str | None = None,
    logical_structure_type: str | None = None,
) -> float:
    if is_section_l_submission_context(source_hint, excerpt):
        return -100.0
    if is_marketing_summary(excerpt):
        return 10.0
    hint = str(source_hint or "globalScan")
    if logical_structure_type in {"SECTION_M", "REVISED_SECTION_M"}:
        hint = "sectionMFulltext"
    score = float(EVALUATION_AUTHORITY_ORDER.get(hint, 20))
    if logical_structure_type == "QA":
        score = float(EVALUATION_AUTHORITY_ORDER["qa"])
    if amendment_number and amendment_number.isdigit():
        score += int(amendment_number) * 0.01
    if re.search(r"\b(?:replace\s+in\s+its\s+entirety|delete\s+and\s+substitute|incorporated\s+by\s+amendment)\b", excerpt, re.I):
        score += 25.0
    return score


def source_hint_for_evaluation_item(item: dict[str, Any]) -> str | None:
    logical = str(item.get("logicalType") or "")
    if logical.startswith("SF30"):
        return "sf30"
    if logical == "SECTION_M":
        return "sectionMFulltext"
    if logical == "QA":
        return "qa"
    if logical in {"ATTACHMENT", "EXHIBIT"}:
        return "attachment"
    return "globalScan"
