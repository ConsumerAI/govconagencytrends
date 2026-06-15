from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from extraction.signals.authority import authority_rank, infer_authority_tier
from extraction.signals.due_date_validator import parse_date_to_iso_or_null

DATE_TOKEN_RE = re.compile(
    r"\b(\d{1,2}/\d{1,2}/\d{4}|\d{1,2}/\d{1,2}/\d{2}|"
    r"\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t|tember)?|"
    r"Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},?\s+\d{4})\b",
    re.I,
)

TIME_RE = re.compile(r"\b(\d{1,2}:\d{2}\s*(?:AM|PM)?|\d{1,2}\.\d{2}\s*(?:AM|PM)?)\b", re.I)

TIMEZONE_PATTERNS: list[tuple[str, str | None]] = [
    (r"\b(Eastern\s+(?:Standard|Daylight)\s+Time|Eastern\s+Time|\bET\b|\bEST\b|\bEDT\b)\b", None),
    (r"\b(Central\s+(?:Standard|Daylight)\s+Time|Central\s+Time|\bCT\b|\bCST\b|\bCDT\b)\b", None),
    (r"\b(Mountain\s+(?:Standard|Daylight)\s+Time|Mountain\s+Time|\bMT\b|\bMST\b|\bMDT\b)\b", None),
    (r"\b(Pacific\s+(?:Standard|Daylight)\s+Time|Pacific\s+Time|\bPT\b|\bPST\b|\bPDT\b)\b", None),
    (r"\bAlaska\s+(?:Standard|Daylight)?\s*Time\b", None),
    (r"\bHawaii(?:-Aleutian)?\s+Time\b", None),
    (r"\b(UTC|Zulu|\bZ\b)\b", "UTC"),
    (r"\b(?:local\s+time|prevailing\s+local\s+time)\b", None),
]

KEYWORD_SCORES: dict[str, int] = {
    "offers must be received": 110,
    "proposal due": 100,
    "proposals due": 100,
    "response due": 95,
    "closing date": 85,
    "due date": 80,
    "deadline": 75,
    "questions due": 70,
    "site visit": 40,
    "project start": 20,
    "schedule": 15,
    "amendment issue date": 60,
}

NON_PROPOSAL_CUES = [
    "period of performance",
    "project start",
    "project commencement",
    "site visit",
    "performance work statement",
    "statement of work",
    "pop start",
    "pop end",
    "place of performance",
]

AUTHORITY_ORDER = {
    "sf30": 600,
    "form_field": 500,
    "sectionLFulltext": 400,
    "structureLWindow": 380,
    "submissionWindow": 350,
    "sectionMFulltext": 200,
    "attachment": 100,
    "qa": 50,
    "globalScan": 30,
}


class CandidateType(str, Enum):
    PROPOSAL_DUE = "proposal_due"
    QUESTIONS_DUE = "questions_due"
    SITE_VISIT = "site_visit"
    SCHEDULE = "schedule"
    AMENDMENT_ISSUE = "amendment_issue"
    OTHER = "other"


class CandidateStatus(str, Enum):
    FIRM = "firm"
    ANTICIPATED = "anticipated"
    DRAFT = "draft"
    ESTIMATED = "estimated"
    TENTATIVE = "tentative"
    TBD = "tbd"
    SUPERSEDED = "superseded"
    DELETED = "deleted"


@dataclass
class DueDateCandidate:
    date_iso: str | None
    date_raw: str
    time_text: str | None = None
    timezone_text: str | None = None
    timezone_iana: str | None = None
    candidate_type: CandidateType = CandidateType.OTHER
    status: CandidateStatus = CandidateStatus.FIRM
    source_authority: str = "globalScan"
    source_document: str = ""
    amendment_number: str | None = None
    evidence_excerpt: str = ""
    span_hashes: list[str] = field(default_factory=list)
    page_index: int = 0
    source_sha256: str = ""
    confidence: str = "medium"
    score: float = 0.0
    logical_structure_type: str | None = None
    continuation_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date_iso,
            "dateRaw": self.date_raw,
            "time": self.time_text,
            "timezoneText": self.timezone_text,
            "timezoneIana": self.timezone_iana,
            "candidateType": self.candidate_type.value,
            "status": self.status.value,
            "sourceAuthority": self.source_authority,
            "sourceDocument": self.source_document,
            "amendmentNumber": self.amendment_number,
            "evidence": self.evidence_excerpt,
            "confidence": self.confidence,
            "pageIndex": self.page_index,
            "logicalStructureType": self.logical_structure_type,
            "continuationOf": self.continuation_of,
        }


def extract_timezone(text: str) -> tuple[str | None, str | None]:
    for pattern, iana in TIMEZONE_PATTERNS:
        match = re.search(pattern, text, re.I)
        if not match:
            continue
        raw = match.group(1).strip() if match.lastindex else match.group(0).strip()
        if pattern.startswith(r"\b(?:local"):
            return "local time", None
        if iana:
            return raw, iana
        if re.search(r"\bCentral\s+Time\b", raw, re.I) or raw.upper() == "CT":
            return "Central Time", None
        return raw, None
    return None, None


def classify_status(excerpt: str) -> CandidateStatus:
    lower = excerpt.lower()
    for sentinel, status in (
        ("tbd", CandidateStatus.TBD),
        ("to be determined", CandidateStatus.TBD),
        ("draft", CandidateStatus.DRAFT),
        ("anticipated", CandidateStatus.ANTICIPATED),
        ("estimated", CandidateStatus.ESTIMATED),
        ("tentative", CandidateStatus.TENTATIVE),
        ("projected", CandidateStatus.ESTIMATED),
        ("on or about", CandidateStatus.ANTICIPATED),
        ("example", CandidateStatus.DRAFT),
        ("placeholder", CandidateStatus.DRAFT),
        ("subject to change", CandidateStatus.ANTICIPATED),
        ("previous due date", CandidateStatus.SUPERSEDED),
        ("superseded", CandidateStatus.SUPERSEDED),
        ("deleted", CandidateStatus.DELETED),
        ("revised elsewhere", CandidateStatus.SUPERSEDED),
    ):
        if sentinel in lower:
            return status
    return CandidateStatus.FIRM


def classify_candidate_type(excerpt: str) -> CandidateType:
    lower = excerpt.lower()
    if re.search(r"\bquestions?\s+due\b", lower):
        return CandidateType.QUESTIONS_DUE
    if re.search(r"\bsite\s+visit\b", lower):
        return CandidateType.SITE_VISIT
    if re.search(r"\b(?:project\s+start|commencement|schedule)\b", lower) and not re.search(
        r"\b(?:proposal|offer|submission|closing)\s+due\b", lower
    ):
        return CandidateType.SCHEDULE
    if re.search(r"\bamendment\s+(?:issue|effective)\s+date\b", lower):
        return CandidateType.AMENDMENT_ISSUE
    if re.search(r"\b(?:proposal|offer|offers|response|closing)\s+due\b|\boffers?\s+must\s+be\s+received\b|\bdue\s+date\b", lower):
        return CandidateType.PROPOSAL_DUE
    return CandidateType.OTHER


def _status_score(status: CandidateStatus) -> int:
    return {
        CandidateStatus.FIRM: 100,
        CandidateStatus.ANTICIPATED: 40,
        CandidateStatus.ESTIMATED: 35,
        CandidateStatus.TENTATIVE: 30,
        CandidateStatus.DRAFT: 20,
        CandidateStatus.TBD: 10,
        CandidateStatus.SUPERSEDED: 0,
        CandidateStatus.DELETED: 0,
    }.get(status, 0)


def _keyword_score(excerpt: str) -> int:
    lower = excerpt.lower()
    score = 0
    for phrase, weight in KEYWORD_SCORES.items():
        if phrase in lower:
            score = max(score, weight)
    return score


def _reject_non_proposal_context(excerpt: str, candidate_type: CandidateType) -> bool:
    lower = excerpt.lower()
    if candidate_type == CandidateType.PROPOSAL_DUE:
        if any(cue in lower for cue in NON_PROPOSAL_CUES) and not re.search(
            r"\b(?:proposal|offer|submission|closing)\s+due\b|\boffers?\s+must\s+be\s+received\b", lower
        ):
            return True
    return False


def build_due_date_candidate(
    *,
    date_raw: str,
    excerpt: str,
    source_hint: str | None,
    source_document: str,
    span_hashes: list[str],
    page_index: int,
    source_sha256: str,
    amendment_number: str | None = None,
    logical_structure_type: str | None = None,
    continuation_of: str | None = None,
) -> DueDateCandidate | None:
    status = classify_status(excerpt)
    candidate_type = classify_candidate_type(excerpt)
    if _reject_non_proposal_context(excerpt, candidate_type):
        return None
    if candidate_type not in {CandidateType.PROPOSAL_DUE, CandidateType.QUESTIONS_DUE, CandidateType.AMENDMENT_ISSUE}:
        return None

    date_iso = parse_date_to_iso_or_null(date_raw)
    if not date_iso and status == CandidateStatus.FIRM and candidate_type == CandidateType.PROPOSAL_DUE:
        return None
    if status in {CandidateStatus.TBD, CandidateStatus.DELETED, CandidateStatus.SUPERSEDED} and candidate_type == CandidateType.PROPOSAL_DUE:
        return None

    time_match = TIME_RE.search(excerpt)
    time_text = time_match.group(1).strip() if time_match else None
    tz_text, tz_iana = extract_timezone(excerpt)

    auth = infer_authority_tier(signal_id="rfp_due_date_v1", excerpt=excerpt, source_hint=source_hint)
    authority_key = str(source_hint or "globalScan")
    score = (
        AUTHORITY_ORDER.get(authority_key, 20)
        + _status_score(status)
        + _keyword_score(excerpt)
        + authority_rank(auth)
    )

    confidence = "high" if status == CandidateStatus.FIRM and date_iso and score >= 400 else "medium"
    if status != CandidateStatus.FIRM:
        confidence = "low"

    return DueDateCandidate(
        date_iso=date_iso,
        date_raw=date_raw,
        time_text=time_text,
        timezone_text=tz_text,
        timezone_iana=tz_iana,
        candidate_type=candidate_type,
        status=status,
        source_authority=authority_key,
        source_document=source_document,
        amendment_number=amendment_number,
        evidence_excerpt=excerpt[:280],
        span_hashes=span_hashes,
        page_index=page_index,
        source_sha256=source_sha256,
        confidence=confidence,
        score=float(score),
        logical_structure_type=logical_structure_type,
        continuation_of=continuation_of,
    )


def extract_due_date_candidates_from_text(
    text: str,
    *,
    source_document: str,
    source_sha256: str,
    page_index: int,
    span_hashes: list[str],
    source_hint: str | None = None,
    amendment_number: str | None = None,
    logical_structure_type: str | None = None,
    continuation_of: str | None = None,
    provenance_prefix: str | None = None,
) -> list[DueDateCandidate]:
    candidates: list[DueDateCandidate] = []
    kw_re = re.compile(
        r"\b(offers?\s+must\s+be\s+received|proposal\s+due|proposals?\s+due|response\s+due|closing\s+date|"
        r"due\s+date|deadline|questions?\s+due|site\s+visit|project\s+start|schedule)\b",
        re.I,
    )

    for match in DATE_TOKEN_RE.finditer(text):
        raw_date = match.group(1)
        window_start = max(0, match.start() - 180)
        window_end = min(len(text), match.end() + 180)
        excerpt = text[window_start:window_end].replace("\n", " ").strip()
        if provenance_prefix:
            excerpt = f"{provenance_prefix} {excerpt}".strip()

        nearest_kw = 9999
        for kw in kw_re.finditer(text):
            nearest_kw = min(nearest_kw, abs(kw.start() - match.start()))
        if nearest_kw > 400:
            continue

        candidate = build_due_date_candidate(
            date_raw=raw_date,
            excerpt=excerpt,
            source_hint=source_hint,
            source_document=source_document,
            span_hashes=span_hashes,
            page_index=page_index,
            source_sha256=source_sha256,
            amendment_number=amendment_number,
            logical_structure_type=logical_structure_type,
            continuation_of=continuation_of,
        )
        if candidate:
            candidates.append(candidate)

    label_re = re.compile(
        r"(?:proposal|offer(?:s)?|questions?)\s+due\s+(?:date|datetime)?\s*[:\-–—]?\s*([^\n\r;]{4,120})",
        re.I,
    )
    for match in label_re.finditer(text):
        line = match.group(1).strip()
        date_match = DATE_TOKEN_RE.search(line)
        if not date_match:
            continue
        window_start = max(0, match.start() - 40)
        window_end = min(len(text), match.end() + 120)
        excerpt = text[window_start:window_end].replace("\n", " ").strip()
        if provenance_prefix:
            excerpt = f"{provenance_prefix} {excerpt}".strip()
        candidate = build_due_date_candidate(
            date_raw=date_match.group(1),
            excerpt=excerpt,
            source_hint=source_hint,
            source_document=source_document,
            span_hashes=span_hashes,
            page_index=page_index,
            source_sha256=source_sha256,
            amendment_number=amendment_number,
            logical_structure_type=logical_structure_type,
            continuation_of=continuation_of,
        )
        if candidate:
            candidates.append(candidate)

    return candidates


def select_proposal_due_candidate(candidates: list[DueDateCandidate]) -> tuple[DueDateCandidate | None, list[DueDateCandidate]]:
    pool = [
        c
        for c in candidates
        if c.candidate_type == CandidateType.PROPOSAL_DUE
        and c.status not in {CandidateStatus.DELETED, CandidateStatus.SUPERSEDED}
    ]
    if not pool:
        return None, []

    firm = [c for c in pool if c.status == CandidateStatus.FIRM and c.date_iso]
    chosen_pool = firm or [c for c in pool if c.date_iso]
    if not chosen_pool:
        return None, pool

    chosen_pool.sort(
        key=lambda c: (
            -c.score,
            -(int(c.amendment_number) if c.amendment_number and c.amendment_number.isdigit() else -1),
            c.page_index,
        )
    )
    winner = chosen_pool[0]
    alternates = [c for c in pool if c is not winner]
    return winner, alternates


def select_questions_due_candidate(candidates: list[DueDateCandidate]) -> DueDateCandidate | None:
    pool = [c for c in candidates if c.candidate_type == CandidateType.QUESTIONS_DUE and c.date_iso]
    if not pool:
        return None
    pool.sort(key=lambda c: (-c.score, c.page_index))
    return pool[0]


def format_due_datetime_local(candidate: DueDateCandidate) -> str | None:
    if not candidate.date_iso:
        return None
    parts = [candidate.date_iso]
    if candidate.time_text:
        parts.append(candidate.time_text)
    if candidate.timezone_text:
        parts.append(candidate.timezone_text)
    elif re.search(r"\blocal\s+time\b", candidate.evidence_excerpt, re.I):
        parts.append("local time")
    return " ".join(parts).strip()
