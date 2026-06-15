from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class IncumbentCandidate:
    value: str
    confidence: str
    role: str
    excerpt: str


EXPLICIT_INCUMBENT = re.compile(
    r"\b(?:incumbent(?:\s+contractor)?|current\s+contractor|current\s+service\s+provider)\s*[:\-–—]?\s*([A-Za-z0-9][A-Za-z0-9 .,&'\-]{2,120})",
    re.I,
)
PRIOR_HOLDER = re.compile(
    r"\b(?:prior\s+contract\s+holder|current\s+holder|awarded\s+to)\s*[:\-–—]?\s*([A-Za-z0-9][A-Za-z0-9 .,&'\-]{2,120})",
    re.I,
)
CONTEXTUAL = re.compile(
    r"\b(?:past\s+performance|unsuccessful\s+offeror|competitor|subcontractor)\b",
    re.I,
)
PIID_NEAR = re.compile(
    r"\b(?:prior\s+contract\s+piid|contract\s+number|current\s+contract\s+number)\s*[:\-–—]?\s*([A-Z0-9-]{8,20})",
    re.I,
)


def extract_incumbent_candidates(text: str) -> list[IncumbentCandidate]:
    results: list[IncumbentCandidate] = []
    for match in EXPLICIT_INCUMBENT.finditer(text or ""):
        excerpt = text[max(0, match.start() - 40) : min(len(text), match.end() + 40)]
        results.append(
            IncumbentCandidate(
                value=match.group(1).strip().rstrip(".,;"),
                confidence="high",
                role="explicit_incumbent",
                excerpt=excerpt.strip(),
            )
        )
    for match in PRIOR_HOLDER.finditer(text or ""):
        excerpt = text[max(0, match.start() - 40) : min(len(text), match.end() + 40)]
        if CONTEXTUAL.search(excerpt):
            continue
        results.append(
            IncumbentCandidate(
                value=match.group(1).strip().rstrip(".,;"),
                confidence="medium",
                role="prior_holder",
                excerpt=excerpt.strip(),
            )
        )
    return results


def extract_prior_piid(text: str) -> str | None:
    match = PIID_NEAR.search(text or "")
    if not match:
        return None
    return match.group(1).upper()
