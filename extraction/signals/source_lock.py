from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from extraction.types import CorpusPage


@dataclass
class SolicitationSourceLock:
    source_sha256: str | None
    confidence: str
    evidence: list[str] = field(default_factory=list)


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _count_matches(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def pick_solicitation_source_sha256(
    pages: list[CorpusPage],
    *,
    section_l_fulltext: dict[str, Any] | None = None,
    section_m_fulltext: dict[str, Any] | None = None,
) -> SolicitationSourceLock:
    sha_l = str((section_l_fulltext or {}).get("evidence_v1", {}).get("sourceSha256") or "").strip()
    if sha_l:
        return SolicitationSourceLock(source_sha256=sha_l, confidence="authoritative", evidence=["section_l_fulltext"])

    sha_m = str((section_m_fulltext or {}).get("evidence_v1", {}).get("sourceSha256") or "").strip()
    if sha_m:
        return SolicitationSourceLock(source_sha256=sha_m, confidence="authoritative", evidence=["section_m_fulltext"])

    scores: dict[str, tuple[int, int]] = {}
    section_re = re.compile(r"\b(?:SECTION|S\s*E\s*C\s*T\s*I\s*O\s*N)\s*[A-M]\b", re.I)
    for index, page in enumerate(pages):
        sha = str(page.source_sha256 or "").strip()
        if not sha:
            continue
        text = _normalize_ws(page.text)
        upper = text.upper()
        score = 0
        if "SOLICITATION" in upper:
            score += 30
        if "TABLE OF CONTENTS" in upper:
            score += 25
        if re.search(r"\bOFFER[, ]+AND\s+AWARD\b", text, re.I) or re.search(r"\bOFFER\s+AND\s+AWARD\b", text, re.I):
            score += 30
        if _count_matches(text, section_re) >= 6:
            score += 40
        if score <= 0:
            continue
        prev_score, first_index = scores.get(sha, (0, index))
        if score > prev_score:
            scores[sha] = (score, first_index)

    if not scores:
        return SolicitationSourceLock(source_sha256=None, confidence="unknown", evidence=[])

    best_sha = max(scores.items(), key=lambda item: (item[1][0], -item[1][1]))[0]
    return SolicitationSourceLock(
        source_sha256=best_sha,
        confidence="provisional",
        evidence=["corpus_page_scoring"],
    )


def lock_windows(windows: list[Any], source_sha256: str | None) -> list[Any]:
    if not source_sha256:
        return windows
    locked = []
    for window in windows:
        loc = getattr(window, "loc", None)
        if loc and str(getattr(loc, "source_sha256", "") or "") == source_sha256:
            locked.append(window)
    return locked if locked else windows
