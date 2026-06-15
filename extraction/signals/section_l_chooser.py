from __future__ import annotations

import re

HEADING_RES = [
    re.compile(r"^L\.\d+(\.\d+)?\b", re.IGNORECASE),
    re.compile(r"\bSECTION\s+L\b", re.IGNORECASE),
    re.compile(r"\bINSTRUCTIONS\b", re.IGNORECASE),
    re.compile(r"\bPROPOSAL\b", re.IGNORECASE),
    re.compile(r"\bSUBMISSION\b", re.IGNORECASE),
    re.compile(r"\bOFFERS?\s+MUST\s+BE\s+RECEIVED\b", re.IGNORECASE),
    re.compile(r"\bPOINT\s+OF\s+CONTACT\b|\bPOC\b", re.IGNORECASE),
]

STRONG = [
    "offers must be received",
    "proposal submission",
    "submit proposals",
    "proposals shall be submitted",
    "submission instructions",
    "deliver to",
    "electronic submission",
    "email proposals",
]
WEAK = [
    "offeror",
    "submission",
    "volume",
    "attachment",
    "received at",
    "deadline",
    "closing date",
    "contracting officer",
    "contract specialist",
    "point of contact",
]


def repair_wrapped_tlds(text: str) -> str:
    return (
        text.replace("\n", " ")
        .replace(".mi l", ".mil")
        .replace(".go v", ".gov")
        .replace(".co m", ".com")
    )


def _count_phrase_hits(haystack_lower: str, phrase_lower: str, cap: int) -> int:
    hits = 0
    idx = 0
    while hits < cap:
        at = haystack_lower.find(phrase_lower, idx)
        if at < 0:
            break
        hits += 1
        idx = at + max(1, len(phrase_lower))
    return hits


def _count_clause_ids(text: str) -> int:
    return len(re.findall(r"\b(?:52|252)\.\d{3}-\d{1,4}\b", text, flags=re.IGNORECASE)) + len(
        re.findall(r"\b\d{2}\.\d{3}-\d{1,4}\b", text)
    )


def _has_payment_noise(window_lower: str) -> bool:
    return bool(
        re.search(
            r"\b(wawf|irapt|invoice|payment request|dfas|receiving report|wide area workflow|acceptance)\b",
            window_lower,
            flags=re.IGNORECASE,
        )
    )


def _has_carrier(window_text: str, window_lower: str) -> bool:
    if re.search(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", window_text, flags=re.IGNORECASE):
        return True
    if "http://" in window_lower or "https://" in window_lower or "portal" in window_lower:
        return True
    return bool(re.search(r"\bdeliver\s+to\b[^\n]{10,}", window_text, flags=re.IGNORECASE))


def choose_section_l_instruction_window(full_text: str) -> dict[str, object] | None:
    text = str(full_text or "")
    if not text:
        return None

    starts: list[int] = []
    line_start = 0
    for i, ch in enumerate(text + "\n"):
        if ch != "\n":
            continue
        line = text[line_start:i]
        if 0 < len(line) <= 120 and any(re.search(pat, line) for pat in HEADING_RES):
            starts.append(line_start)
        line_start = i + 1
    if not starts:
        return None

    best: dict[str, object] | None = None
    for si, start in enumerate(starts):
        hard_end = min(len(text), start + 1800)
        next_start = starts[si + 1] if si + 1 < len(starts) else -1
        end = next_start if next_start > start and next_start <= hard_end else hard_end
        if end <= start:
            continue
        window_text = text[start:end]
        window_lower = window_text.lower()

        strong_hits_raw = sum(_count_phrase_hits(window_lower, p, 99) for p in STRONG)
        strong_hits_capped = min(6, strong_hits_raw)
        if _has_payment_noise(window_lower) and strong_hits_raw < 2:
            continue

        weak_hits = min(10, sum(_count_phrase_hits(window_lower, p, 10) for p in WEAK))
        carrier_boost = 30 if _has_carrier(window_text, window_lower) else 0
        clause_penalty = 40 if _count_clause_ids(window_text) >= 3 else 0
        score = strong_hits_capped * 25 + weak_hits * 10 + carrier_boost - clause_penalty

        candidate = {"windowText": window_text, "windowStart": start, "score": score}
        if not best or score > best["score"] or (score == best["score"] and start < int(best["windowStart"])):
            best = candidate

    if not best or int(best["score"]) <= 0:
        return None
    return best
