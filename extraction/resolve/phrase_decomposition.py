from __future__ import annotations

import re
from typing import TypedDict


class DecomposedProcurementPhrase(TypedDict):
    competition_type: str | None
    set_aside: str | None
    full_phrase: str


def decompose_competition_set_aside_phrase(text: str) -> DecomposedProcurementPhrase | None:
    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return None

    full_match = re.search(
        r"this is a\s+(competitive\s*,?\s*.+?(?:set-aside|set aside)[^.]{0,80})",
        normalized,
        flags=re.IGNORECASE,
    )
    phrase = (full_match.group(1) if full_match else normalized).strip()
    if not re.search(r"competitive", phrase, flags=re.IGNORECASE):
        return None
    if not re.search(r"(?:set-aside|set aside|8\s*\(\s*a\s*\))", phrase, flags=re.IGNORECASE):
        return None

    set_aside_match = re.search(
        r"(?:,\s*)?((?:8\s*\(\s*a\s*\)|sdvosb|hubzone|wosb|edwosb|small business)[^,.]{0,120}?(?:set-aside|set aside))",
        phrase,
        flags=re.IGNORECASE,
    )
    set_aside = set_aside_match.group(1).strip() if set_aside_match else None
    if not set_aside:
        return None
    return {
        "competition_type": "Competitive",
        "set_aside": set_aside,
        "full_phrase": phrase,
    }
