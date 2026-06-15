from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class CodeCandidate:
    value: str
    score: int
    role: str
    excerpt: str


PRIMARY_NAICS_CUES = (
    re.compile(r"\bprimary\s+naics\b", re.I),
    re.compile(r"\bnaics\s+code\b", re.I),
    re.compile(r"\bnaics\s*[:\-]", re.I),
)
SECONDARY_NAICS_CUES = (
    re.compile(r"\bsecondary\s+naics\b", re.I),
    re.compile(r"\bsubcontract(?:ing|or)\s+naics\b", re.I),
    re.compile(r"\bsize\s+standard\b", re.I),
    re.compile(r"\bprior\s+contract\b", re.I),
)
PRIMARY_PSC_CUES = (
    re.compile(r"\bprimary\s+psc\b", re.I),
    re.compile(r"\bproduct\s+or\s+service\s+code\b", re.I),
    re.compile(r"\bpsc\s+code\b", re.I),
    re.compile(r"\bpsc\s*[:\-]", re.I),
)
SECONDARY_PSC_CUES = (
    re.compile(r"\bexample\s+psc\b", re.I),
    re.compile(r"\blegacy\s+code\b", re.I),
    re.compile(r"\bprior\s+contract\b", re.I),
)


def _window_score(text: str, primary: tuple[re.Pattern[str], ...], secondary: tuple[re.Pattern[str], ...]) -> tuple[int, str]:
    primary_hits = sum(len(p.findall(text)) for p in primary)
    secondary_hits = sum(len(p.findall(text)) for p in secondary)
    if primary_hits:
        return primary_hits * 30, "primary"
    if secondary_hits:
        return secondary_hits * 5, "secondary"
    return 0, "context"


def extract_naics_candidates(text: str, *, excerpt: str = "") -> list[CodeCandidate]:
    candidates: list[CodeCandidate] = []
    for match in re.finditer(r"\b(?:naics|naics\s*code)\s*[:\-–—]?\s*(\d{6}|\d{2}[-\s]?\d{4})\b", text, re.I):
        raw = re.sub(r"\D", "", match.group(1))
        if len(raw) != 6:
            continue
        window = text[max(0, match.start() - 120) : min(len(text), match.end() + 120)]
        score, role = _window_score(window, PRIMARY_NAICS_CUES, SECONDARY_NAICS_CUES)
        if role == "context" and re.search(r"\battachment\b", window, re.I):
            score = 1
        candidates.append(CodeCandidate(value=raw, score=score, role=role, excerpt=excerpt or window.strip()))
    return candidates


def extract_psc_candidates(text: str, *, excerpt: str = "") -> list[CodeCandidate]:
    candidates: list[CodeCandidate] = []
    for match in re.finditer(r"\b(?:psc|product\s*or\s*service\s*code)\s*[:\-–—]?\s*([A-Z0-9]{4,10})\b", text, re.I):
        raw = match.group(1).upper()
        window = text[max(0, match.start() - 120) : min(len(text), match.end() + 120)]
        score, role = _window_score(window, PRIMARY_PSC_CUES, SECONDARY_PSC_CUES)
        if role == "context" and re.search(r"\battachment\b", window, re.I):
            score = 1
        candidates.append(CodeCandidate(value=raw, score=score, role=role, excerpt=excerpt or window.strip()))
    return candidates


def pick_primary_code(candidates: list[CodeCandidate]) -> tuple[CodeCandidate | None, list[CodeCandidate]]:
    if not candidates:
        return None, []
    primary = [c for c in candidates if c.role == "primary"]
    pool = primary or [c for c in candidates if c.role != "secondary"]
    if not pool:
        return None, candidates
    pool.sort(key=lambda c: (-c.score, c.value))
    winner = pool[0]
    conflicts = [c for c in pool if c.value != winner.value and c.score >= winner.score - 5]
    return winner, conflicts
