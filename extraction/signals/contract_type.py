from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class ContractTypeClassification:
    value: str
    is_hybrid: bool
    components: list[str]
    raw_phrase: str


TYPE_PATTERNS: list[tuple[str, re.Pattern[str], bool]] = [
    ("Firm Fixed Price (FFP)", re.compile(r"\b(?:firm\s+fixed\s+price|\bffp\b)\b", re.I), False),
    ("Fixed Price with Economic Price Adjustment (FP-EPA)", re.compile(r"\bfp[\s-]?epa\b|\beconomic\s+price\s+adjustment\b", re.I), False),
    ("Time and Materials (T&M)", re.compile(r"\btime\s+and\s+materials\b|\bt\s*&\s*m\b", re.I), False),
    ("Labor Hour (LH)", re.compile(r"\blabor\s+hour\b|\blh\b", re.I), False),
    ("Cost Plus Fixed Fee (CPFF)", re.compile(r"\bcost\s+plus\s+fixed\s+fee\b|\bcpff\b", re.I), False),
    ("Cost Plus Award Fee (CPAF)", re.compile(r"\bcost\s+plus\s+award\s+fee\b|\bcpaf\b", re.I), False),
    ("Cost Plus Incentive Fee (CPIF)", re.compile(r"\bcost\s+plus\s+incentive\s+fee\b|\bcpif\b", re.I), False),
    ("Indefinite Delivery / Indefinite Quantity (IDIQ)", re.compile(r"\bidiq\b|\bindefinite\s+delivery\b", re.I), False),
]


def classify_contract_type(text: str) -> ContractTypeClassification | None:
    phrase = re.sub(r"\s+", " ", text or "").strip()
    if not phrase:
        return None
    found: list[str] = []
    for label, pattern, _ in TYPE_PATTERNS:
        if pattern.search(phrase):
            found.append(label)
    if not found:
        labeled = re.search(r"\bcontract\s+type\s*[:\-–—]\s*([^\n;]{2,80})", phrase, re.I)
        if labeled:
            inner = labeled.group(1).strip()
            return ContractTypeClassification(value=inner, is_hybrid=False, components=[inner], raw_phrase=phrase)
        return None
    is_hybrid = len(found) > 1 or bool(re.search(r"\b(?:hybrid|mixed|combination)\b", phrase, re.I))
    if is_hybrid:
        value = "Hybrid / Mixed: " + "; ".join(found)
    else:
        value = found[0]
    return ContractTypeClassification(value=value, is_hybrid=is_hybrid, components=found, raw_phrase=phrase)
