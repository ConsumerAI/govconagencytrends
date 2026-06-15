from __future__ import annotations

import re
from typing import Literal, TypedDict

AuthorityLabel = Literal["FORM_FIELD", "SECTION_LM", "COVER_LETTER", "GLOBAL_SCAN", "SF30", "ATTACHMENT", "QA"]
SourceHint = Literal[
    "sectionLFulltext",
    "sectionMFulltext",
    "structureLWindow",
    "structureMWindow",
    "globalScan",
    "submissionWindow",
    "sf30",
    "attachment",
    "qa",
]


class SignalAuthority(TypedDict):
    tier: int
    label: str
    reason: str


def _has_form_field_cues(excerpt: str) -> tuple[bool, str]:
    if re.search(r"\bSF\s*[-]?\s*18\b", excerpt, re.I) and re.search(r"\bBlock\s+10\b", excerpt, re.I):
        return True, "SF-18 Block 10"
    if re.search(r"\bStandard\s+Form\s+18\b", excerpt, re.I) and re.search(r"\bBlock\s+10\b", excerpt, re.I):
        return True, "Standard Form 18 Block 10"
    if re.search(r"\bSF\s*[-]?\s*33\b", excerpt, re.I) and re.search(r"\bItem\s+9\b", excerpt, re.I):
        return True, "SF-33 Item 9"
    if re.search(r"\bStandard\s+Form\s+33\b", excerpt, re.I) and re.search(r"\bItem\s+9\b", excerpt, re.I):
        return True, "Standard Form 33 Item 9"
    if re.search(r"\bSF\s*[-]?\s*30\b", excerpt, re.I) or re.search(r"\bStandard\s+Form\s+30\b", excerpt, re.I):
        return True, "SF-30 amendment form"
    if re.search(r"\b(?:Block|Item)\s+\d+\b", excerpt, re.I) and re.search(r"\bSF\s*[-]?\s*\d+\b", excerpt, re.I):
        return True, "Form block/item reference"
    return False, ""


def _has_section_lm_cues(excerpt: str, source_hint: str | None) -> tuple[bool, str]:
    if excerpt.startswith("[FULLTEXT:L] "):
        return True, "Section L fulltext"
    if excerpt.startswith("[FULLTEXT:M] "):
        return True, "Section M fulltext"
    if re.search(r"\bSECTION\s+L\b", excerpt, re.I):
        return True, "Section L window"
    if re.search(r"\bSECTION\s+M\b", excerpt, re.I):
        return True, "Section M window"
    if source_hint in {"sectionLFulltext", "structureLWindow"}:
        return True, "Section L fulltext"
    if source_hint in {"sectionMFulltext", "structureMWindow"}:
        return True, "Section M fulltext"
    return False, ""


def infer_authority_tier(*, signal_id: str, excerpt: str, source_hint: str | None = None) -> SignalAuthority:
    _ = signal_id
    form_match, form_reason = _has_form_field_cues(excerpt)
    if form_match:
        label: AuthorityLabel = "SF30" if "30" in form_reason else "FORM_FIELD"
        return {"tier": 1, "label": label, "reason": form_reason}

    if source_hint == "sf30":
        return {"tier": 1, "label": "SF30", "reason": "SF-30 amendment form"}

    sec_match, sec_reason = _has_section_lm_cues(excerpt, source_hint)
    if sec_match:
        return {"tier": 2, "label": "SECTION_LM", "reason": sec_reason}

    if source_hint == "qa":
        return {"tier": 4, "label": "QA", "reason": "Questions and answers"}

    if source_hint == "attachment":
        return {"tier": 4, "label": "ATTACHMENT", "reason": "Attachment or exhibit prose"}

    if re.search(r"\b(?:offers?\s+must\s+be\s+received|on\s+or\s+before|offers?\s+due\s+by)\b", excerpt, re.I):
        return {"tier": 3, "label": "COVER_LETTER", "reason": "Cover letter cue"}

    if source_hint in {"globalScan", "submissionWindow"}:
        return {"tier": 4, "label": "GLOBAL_SCAN", "reason": "Global scan window"}
    return {"tier": 4, "label": "GLOBAL_SCAN", "reason": "Global scan window"}


def authority_rank(authority: SignalAuthority | None) -> int:
    if not authority:
        return 0
    tier = int(authority.get("tier") or 0)
    label_boost = {
        "SF30": 5,
        "FORM_FIELD": 4,
        "SECTION_LM": 3,
        "COVER_LETTER": 2,
        "ATTACHMENT": 1,
        "QA": 0,
        "GLOBAL_SCAN": 0,
    }.get(str(authority.get("label") or ""), 0)
    return tier * 10 + label_boost
