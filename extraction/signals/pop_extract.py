from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PopExtraction:
    start: str | None = None
    end: str | None = None
    years: str | None = None
    composite: str | None = None
    base_period: str | None = None
    option_periods: list[str] | None = None


DATE_TOKEN = re.compile(
    r"\b(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4})\b",
    re.I,
)


def extract_pop_from_text(text: str) -> PopExtraction:
    result = PopExtraction()
    normalized = text or ""

    start_match = re.search(
        r"\b(?:pop\s+start|period\s+of\s+performance\s+start|start\s+date\s+of\s+performance)\s*[:\-–—]?\s*([^\n;]{4,60})",
        normalized,
        re.I,
    )
    if start_match:
        date = DATE_TOKEN.search(start_match.group(1))
        result.start = (date.group(1) if date else start_match.group(1)).strip()

    end_match = re.search(
        r"\b(?:pop\s+end|period\s+of\s+performance\s+end|end\s+date\s+of\s+performance)\s*[:\-–—]?\s*([^\n;]{4,60})",
        normalized,
        re.I,
    )
    if end_match:
        date = DATE_TOKEN.search(end_match.group(1))
        result.end = (date.group(1) if date else end_match.group(1)).strip()

    base_match = re.search(r"\bbase\s+period\s*[:\-–—]?\s*([^\n;]{4,120})", normalized, re.I)
    if base_match:
        result.base_period = base_match.group(1).strip()

    options = re.findall(r"\boption\s+(?:period\s+)?(?:year\s+)?(\d+)\s*[:\-–—]?\s*([^\n;]{4,80})", normalized, re.I)
    if options:
        result.option_periods = [f"Option {num}: {desc.strip()}" for num, desc in options]

    years_match = re.search(r"\b(?:pop\s+years|total\s+duration|contract\s+duration)\s*[:\-–—]?\s*([^\n;]{2,60})", normalized, re.I)
    if years_match and not (result.start and result.end):
        result.years = years_match.group(1).strip()
    elif re.search(r"\bfive[\s-]year\b|\b5[\s-]year\b", normalized, re.I) and not (result.start and result.end):
        result.years = "5 years"

    parts: list[str] = []
    if result.base_period:
        parts.append(f"Base: {result.base_period}")
    if result.option_periods:
        parts.extend(result.option_periods)
    if result.start:
        parts.append(f"Start: {result.start}")
    if result.end:
        parts.append(f"End: {result.end}")
    if result.years and not (result.start and result.end):
        parts.append(f"Duration: {result.years}")
    if parts:
        result.composite = "; ".join(parts)
    return result
