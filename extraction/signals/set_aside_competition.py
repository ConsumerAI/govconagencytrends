from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ProcurementClassification:
    set_aside: str | None
    competition_type: str | None
    raw_phrase: str
    normalized_set_aside: str | None = None
    normalized_competition: str | None = None


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


# Canonical display labels for set-aside (Scope Review / filter mapping input).
SET_ASIDE_CANONICAL: dict[str, str] = {
    "8a_competed": "8(a) Small Business Set-Aside",
    "8a": "8(a)",
    "small_business": "Small Business Set-Aside",
    "total_small_business": "Total Small Business Set-Aside",
    "partial_small_business": "Partial Small Business Set-Aside",
    "hubzone": "HUBZone Set-Aside",
    "hubzone_sole": "HUBZone Sole Source",
    "sdvosb": "Service-Disabled Veteran-Owned Small Business Set-Aside",
    "sdvosb_sole": "Service-Disabled Veteran-Owned Small Business Sole Source",
    "wosb": "Women-Owned Small Business Set-Aside",
    "edwosb": "Economically Disadvantaged WOSB Set-Aside",
    "local_area": "Local Area Set-Aside",
    "unrestricted": "Unrestricted",
}

COMPETITION_CANONICAL: dict[str, str] = {
    "competitive": "Competitive",
    "full_and_open": "Full and Open Competition",
    "full_and_open_after_exclusion": "Full and Open Competition After Exclusion of Sources",
    "sole_source": "Sole Source",
    "limited": "Limited Competition",
}


def classify_procurement_phrase(raw: str) -> ProcurementClassification | None:
    """Deterministically split combined procurement restriction language."""
    phrase = _normalize_ws(raw)
    if not phrase:
        return None
    lower = phrase.lower()

    competition: str | None = None
    set_aside_key: str | None = None

    if re.search(r"\b8\s*\(\s*a\s*\)\s+sole[\s-]?source\b", lower):
        set_aside_key, competition = "8a", "sole_source"
    elif re.search(r"\bcompetitive\s+8\s*\(\s*a\s*\)", lower) or re.search(
        r"\b8\s*\(\s*a\s*\)\s+(?:competitive|competed)\b", lower
    ):
        set_aside_key, competition = "8a_competed", "competitive"
    elif re.search(r"\bhubzone\s+sole[\s-]?source\b", lower):
        set_aside_key, competition = "hubzone_sole", "sole_source"
    elif re.search(r"\bhubzone\s+set[\s-]?aside\b", lower):
        set_aside_key, competition = "hubzone", "competitive"
    elif re.search(r"\bsdvosb\s+sole[\s-]?source\b", lower) or re.search(
        r"\bservice[\s-]?disabled\s+veteran[\s-]?owned\s+small\s+business\s+sole[\s-]?source\b", lower
    ):
        set_aside_key, competition = "sdvosb_sole", "sole_source"
    elif re.search(r"\bsdvosb\s+set[\s-]?aside\b", lower) or re.search(
        r"\bservice[\s-]?disabled\s+veteran[\s-]?owned\s+small\s+business\s+set[\s-]?aside\b", lower
    ):
        set_aside_key, competition = "sdvosb", "competitive"
    elif re.search(r"\bedwosb\s+set[\s-]?aside\b", lower):
        set_aside_key, competition = "edwosb", "competitive"
    elif re.search(r"\bwosb\s+set[\s-]?aside\b", lower):
        set_aside_key, competition = "wosb", "competitive"
    elif re.search(r"\btotal\s+small\s+business\s+set[\s-]?aside\b", lower):
        set_aside_key, competition = "total_small_business", "competitive"
    elif re.search(r"\bpartial\s+small\s+business\s+set[\s-]?aside\b", lower):
        set_aside_key, competition = "partial_small_business", "competitive"
    elif re.search(r"\bsmall\s+business\s+set[\s-]?aside\b", lower):
        set_aside_key, competition = "small_business", "competitive"
    elif re.search(r"\blocal[\s-]?area\s+set[\s-]?aside\b", lower):
        set_aside_key, competition = "local_area", "competitive"
    elif re.search(r"\bfull\s+and\s+open\s+competition\s+after\s+exclusion\b", lower):
        set_aside_key, competition = None, "full_and_open_after_exclusion"
    elif re.search(r"\bfull\s+and\s+open\s+competition\b", lower) or lower == "full and open":
        set_aside_key, competition = None, "full_and_open"
    elif re.search(r"\bunrestricted\b", lower):
        set_aside_key, competition = "unrestricted", "full_and_open"
    elif re.search(r"\bsole[\s-]?source\b", lower):
        competition = "sole_source"
        if re.search(r"\b8\s*\(\s*a\s*\)\b", lower):
            set_aside_key = "8a"
    elif re.search(r"\blimited\s+competition\b", lower):
        competition = "limited"
    elif re.search(r"\bcompetitive\b", lower) and re.search(r"\bset[\s-]?aside\b", lower):
        competition = "competitive"
        if re.search(r"\b8\s*\(\s*a\s*\)\b", lower):
            set_aside_key = "8a_competed"
        elif re.search(r"\bhubzone\b", lower):
            set_aside_key = "hubzone"
        elif re.search(r"\bsdvosb\b", lower):
            set_aside_key = "sdvosb"
        elif re.search(r"\bsmall\s+business\b", lower):
            set_aside_key = "small_business"
    elif re.search(r"\bcompetitive\b", lower):
        competition = "competitive"
    elif re.search(r"\bset[\s-]?aside\b", lower):
        competition = "competitive"
        if re.search(r"\b8\s*\(\s*a\s*\)\b", lower):
            set_aside_key = "8a_competed"

    if not competition and not set_aside_key:
        return None

    set_aside_val = SET_ASIDE_CANONICAL.get(set_aside_key) if set_aside_key else None
    competition_val = COMPETITION_CANONICAL.get(competition) if competition else None
    return ProcurementClassification(
        set_aside=set_aside_val,
        competition_type=competition_val,
        raw_phrase=phrase,
        normalized_set_aside=set_aside_val,
        normalized_competition=competition_val,
    )


def extract_procurement_from_text(text: str) -> list[ProcurementClassification]:
    """Find labeled procurement restriction phrases in document text."""
    results: list[ProcurementClassification] = []
    patterns = [
        re.compile(
            r"\b(?:set[\s-]?aside|competition|type\s+of\s+set[\s-]?aside)\s*[:\-–—]\s*([^\n\r;]{4,120})",
            re.I,
        ),
        re.compile(
            r"\b(total\s+small\s+business\s+set[\s-]?aside|full\s+and\s+open\s+competition(?:\s+after\s+exclusion\s+of\s+sources)?|competitive\s+8\s*\(\s*a\s*\)[^,\n]{0,80}|8\s*\(\s*a\s*\)\s+sole[\s-]?source[^,\n]{0,40}|hubzone\s+set[\s-]?aside|sdvosb\s+set[\s-]?aside|sole[\s-]?source|unrestricted)\b",
            re.I,
        ),
    ]
    seen: set[str] = set()
    for pattern in patterns:
        for match in pattern.finditer(text or ""):
            phrase = _normalize_ws(match.group(1) if match.lastindex else match.group(0))
            if not phrase or phrase.lower() in seen:
                continue
            seen.add(phrase.lower())
            classified = classify_procurement_phrase(phrase)
            if classified:
                results.append(classified)
    return results


def signals_from_classification(
    classified: ProcurementClassification,
    *,
    evidence: dict[str, Any],
    run_id: str,
    source_hint: str | None = None,
) -> list[dict[str, Any]]:
    from extraction.signals.authority import infer_authority_tier

    out: list[dict[str, Any]] = []
    excerpt = str(evidence.get("excerpt") or classified.raw_phrase)

    if classified.competition_type:
        out.append(
            {
                "id": "rfp_competition_type_v1",
                "value": classified.competition_type,
                "confidence": "high",
                "evidence": [
                    {
                        "sourceId": f"runs/{run_id}/corpus/corpus.v1.json",
                        "artifact": "text",
                        "locator": "procurement_classification",
                        "snippet": classified.raw_phrase[:200],
                    }
                ],
                "evidence_v1": {**evidence, "excerpt": excerpt, "sourcePhrase": classified.raw_phrase},
                "authority": infer_authority_tier(
                    signal_id="rfp_competition_type_v1", excerpt=excerpt, source_hint=source_hint
                ),
            }
        )
    if classified.set_aside:
        out.append(
            {
                "id": "rfp_set_aside_v1",
                "value": classified.set_aside,
                "confidence": "high",
                "evidence": [
                    {
                        "sourceId": f"runs/{run_id}/corpus/corpus.v1.json",
                        "artifact": "text",
                        "locator": "procurement_classification",
                        "snippet": classified.raw_phrase[:200],
                    }
                ],
                "evidence_v1": {**evidence, "excerpt": excerpt, "sourcePhrase": classified.raw_phrase},
                "authority": infer_authority_tier(
                    signal_id="rfp_set_aside_v1", excerpt=excerpt, source_hint=source_hint
                ),
            }
        )
    return out
