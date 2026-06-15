from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable

from extraction.signals.section_l_chooser import repair_wrapped_tlds


@dataclass
class SubmissionCandidate:
    value: str
    evidence: dict[str, Any]
    score: float
    page_ordinal: int
    source_sha256: str
    page_index: int
    start: int
    source_hint: str | None = None
    from_dod_safe_fallback: bool = False
    method: str | None = None
    destination: str | None = None


def _normalize_ws(raw: str) -> str:
    s = raw.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def _looks_like_email(text: str) -> bool:
    return bool(re.search(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", text, re.I))


def _count_matches(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def _build_evidence(source: str, span_hashes: list[str], window_text: str, match_index: int, prefix: str | None) -> dict[str, Any] | None:
    if not span_hashes:
        return None
    start = max(0, match_index - 800)
    end = min(len(window_text), match_index + 800)
    raw = re.sub(r"\s+", " ", window_text[start:end]).strip()
    excerpt = raw if len(raw) <= 280 else f"{raw[:279]}…"
    if prefix:
        excerpt = f"{prefix} {excerpt}".strip()
    return {"spanHashes": span_hashes, "excerpt": excerpt, "source": source}


def extract_submission_from_windows(windows: list[Any]) -> SubmissionCandidate | None:
    candidates: list[SubmissionCandidate] = []
    kw_re = re.compile(
        r"\b(submit|submission|proposals?|offer|offers|email|portal|points\s+of\s+contact|received|deliver|send|electronically)\b",
        re.I,
    )

    def carrier_in_text(s: str) -> bool:
        return (
            _looks_like_email(s)
            or bool(re.search(r"\bhttps?://", s, re.I))
            or bool(re.search(r"\bwww\.", s, re.I))
            or bool(re.search(r"\bportal\b", s, re.I))
            or bool(re.search(r"\b(deliver\s+to|address\s+offer\s+to|send\s+to|points\s+of\s+contact)\b", s, re.I))
        )

    for window in windows:
        text = window.text
        if not re.search(r"\b(proposal|offeror|offer\b|quote\b|quotes\b|submit|submission|email|portal|electronic|received|deliver|send)\b", text, re.I):
            continue
        pay_hit = bool(re.search(r"\b(wawf|irapt|invoice|payment\s+request|receiving\s+report|wide\s+area\s+workflow|acceptance|dfas)\b", text, re.I))
        strong_count = (
            _count_matches(
                text,
                re.compile(
                    r"\b(proposal submission|offers must be received|submit proposals|email proposals|no alternate method of proposal submission)\b",
                    re.I,
                ),
            )
            if pay_hit
            else 0
        )
        if pay_hit and strong_count < 2:
            continue
        if not carrier_in_text(text):
            continue
        for match in kw_re.finditer(text):
            chunk = text[match.start() : min(len(text), match.start() + 1800)]
            chunk = repair_wrapped_tlds(chunk)
            ev = _build_evidence(window.source, window.span_hashes, text, match.start(), window.provenance_prefix)
            if not ev or not chunk.strip():
                continue
            score = 50 + strong_count * 10
            if re.search(r"\boffers?\s+must\s+be\s+received\b", chunk, re.I):
                score += 20
            candidates.append(
                SubmissionCandidate(
                    value=chunk.strip(),
                    evidence=ev,
                    score=float(score),
                    page_ordinal=window.loc.page_ordinal,
                    source_sha256=window.loc.source_sha256,
                    page_index=window.loc.page_index,
                    start=window.loc.start + match.start(),
                    source_hint=window.source_hint,
                )
            )
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c.score, c.page_ordinal, c.start))
    return candidates[0]


def scan_corpus_submission(
    pages: list[Any],
    *,
    source_sha256: str | None,
    source_filename: str,
    span_loc: dict[str, Any],
) -> SubmissionCandidate | None:
    candidates: list[SubmissionCandidate] = []
    for page_ordinal, page in enumerate(pages):
        if source_sha256 and str(page.source_sha256 or "") != source_sha256:
            continue
        text = _normalize_ws(page.text)
        if not re.search(r"\b(proposal|offeror|offer\b|submit|submission|volume|attachment|email|portal|electronic|received at)\b", text, re.I):
            continue
        pay_hit = bool(re.search(r"\b(wawf|irapt|invoice|payment\s+request|receiving\s+report|wide\s+area\s+workflow|acceptance|dfas)\b", text, re.I))
        strong_count = (
            _count_matches(
                text,
                re.compile(
                    r"\b(proposal submission|offers must be received|submit proposals|email proposals|no alternate method of proposal submission)\b",
                    re.I,
                ),
            )
            if pay_hit
            else 0
        )
        if pay_hit and strong_count < 2:
            continue
        boost = 10
        if re.search(r"\boffers?\s+must\s+be\s+received\b", text, re.I):
            boost += 15
        span_hashes = [s.sha256 for s in page.spans[:4]] if page.spans else []
        loc = span_loc.get(span_hashes[0]) if span_hashes else None
        if not loc:
            continue
        ev = _build_evidence(source_filename, span_hashes, text, 0, "[CORPUS_SCAN]")
        if not ev:
            continue
        candidates.append(
            SubmissionCandidate(
                value=text[:1800],
                evidence=ev,
                score=float(boost + strong_count * 5),
                page_ordinal=page_ordinal,
                source_sha256=page.source_sha256,
                page_index=page.page_index,
                start=0,
                source_hint="globalScan",
            )
        )
    if not candidates:
        return None
    candidates.sort(key=lambda c: (-c.score, c.page_ordinal))
    return candidates[0]


def ok_sub_domain_check(candidate: SubmissionCandidate | None, section_l_fulltext: str) -> SubmissionCandidate | None:
    if not candidate:
        return None
    emails = re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", section_l_fulltext, re.I)
    domains = {e.split("@")[1].lower() for e in emails if "@" in e}
    if not domains:
        return candidate
    cand_emails = re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", candidate.value, re.I)
    if cand_emails:
        for email in cand_emails:
            domain = email.split("@")[1].lower()
            if domain and domain not in domains:
                return None
        return candidate
    lower = candidate.value.lower()
    if any(token in lower for token in ("http://", "https://", "portal", "www.", "deliver to", "address offer to", "send to")):
        return candidate
    return None


def dod_safe_fallback(text: str, *, evidence: dict[str, Any], loc_meta: dict[str, int]) -> SubmissionCandidate | None:
    match = re.search(
        r"(DoD\s+SAFE[^.\n]{0,200}(?:government[\s-]initiated\s+drop[\s-]off|drop[\s-]off)[^.\n]{0,200})",
        text,
        re.I,
    )
    if not match:
        return None
    chunk = text[match.start() : min(len(text), match.start() + 2000)]
    return SubmissionCandidate(
        value=chunk.strip(),
        evidence=evidence,
        score=999.0,
        page_ordinal=loc_meta.get("page_ordinal", 0),
        source_sha256=loc_meta.get("source_sha256", ""),
        page_index=loc_meta.get("page_index", 0),
        start=match.start(),
        source_hint="sectionLFulltext",
        from_dod_safe_fallback=True,
        method="PORTAL",
        destination="DoD SAFE (government-initiated drop-off)",
    )


def derive_submission_method_destination(value: str) -> tuple[str | None, str | None]:
    if re.search(r"DoD\s+SAFE", value, re.I):
        return "PORTAL", "DoD SAFE (government-initiated drop-off)"
    emails = re.findall(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", value, re.I)
    if emails:
        return "EMAIL", emails[0]
    urls = re.findall(r"https?://[^\s)]+", value, re.I)
    if urls:
        return "PORTAL", urls[0]
    if re.search(r"\bportal\b", value, re.I):
        return "PORTAL", None
    if re.search(r"\b(physically|hand[\s-]deliver|mail to)\b", value, re.I):
        return "PHYSICAL", None
    return None, None


def choose_submission_candidate(
    *,
    section_l_fulltext: str,
    chosen_l_window: dict[str, Any] | None,
    l_locator_windows: list[Any],
    fulltext_windows_l: list[Any],
    structure_windows_l: list[Any],
    fulltext_windows_m: list[Any],
    structure_windows_m: list[Any],
    section_windows: list[Any],
    locked_windows_fn: Callable[[list[Any]], list[Any]],
    pages: list[Any],
    source_sha256: str | None,
    source_filename: str,
    span_loc: dict[str, Any],
) -> SubmissionCandidate | None:
    l_full = repair_wrapped_tlds(section_l_fulltext) if section_l_fulltext else ""

    sub_chosen: SubmissionCandidate | None = None
    if chosen_l_window and l_full:
        start = int(chosen_l_window.get("windowStart") or 0)
        chunk = repair_wrapped_tlds(str(chosen_l_window.get("windowText") or ""))
        if chunk.strip():
            ev = {"excerpt": chunk[:280], "source": source_filename, "spanHashes": []}
            sub_chosen = SubmissionCandidate(
                value=chunk.strip(),
                evidence=ev,
                score=9999.0,
                page_ordinal=0,
                source_sha256=source_sha256 or "",
                page_index=0,
                start=start,
                source_hint="sectionLFulltext",
            )

    cascade: SubmissionCandidate | None = None
    if l_full.strip():
        cascade = ok_sub_domain_check(sub_chosen, l_full)
        if not cascade:
            wins = l_locator_windows if l_locator_windows else fulltext_windows_l
            cascade = ok_sub_domain_check(extract_submission_from_windows(locked_windows_fn(wins)), l_full)
    else:
        cascade = (
            sub_chosen
            or extract_submission_from_windows(locked_windows_fn(fulltext_windows_l))
            or extract_submission_from_windows(locked_windows_fn(structure_windows_l))
            or extract_submission_from_windows(locked_windows_fn(fulltext_windows_m))
            or extract_submission_from_windows(locked_windows_fn(structure_windows_m))
            or extract_submission_from_windows(locked_windows_fn(section_windows))
            or scan_corpus_submission(
                pages,
                source_sha256=source_sha256,
                source_filename=source_filename,
                span_loc=span_loc,
            )
        )

    if cascade and l_full:
        dod = dod_safe_fallback(l_full, evidence=cascade.evidence, loc_meta={"page_ordinal": cascade.page_ordinal, "source_sha256": cascade.source_sha256, "page_index": cascade.page_index})
        if dod:
            return dod
    return cascade
