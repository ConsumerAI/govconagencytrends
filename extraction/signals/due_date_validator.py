from __future__ import annotations

import re

POP_CUES = [
    "section f - deliveries",
    "section f – deliveries",
    "deliveries or performance",
    "period of performance",
    "place of performance",
]
SUBMISSION_CUES = [
    "sealed offers in original",
    "proposals shall be submitted",
    "cutoff for proposals",
    "will be received",
    "offers must be received",
    "offers due",
    "submit proposals",
    "submit to ",
    "due date and time",
    "due date for receipt of offers",
    "closing date",
]
MONTH_MAP = {
    "jan": "01", "january": "01", "feb": "02", "february": "02", "mar": "03", "march": "03",
    "apr": "04", "april": "04", "may": "05", "jun": "06", "june": "06", "jul": "07", "july": "07",
    "aug": "08", "august": "08", "sep": "09", "sept": "09", "september": "09",
    "oct": "10", "october": "10", "nov": "11", "november": "11", "dec": "12", "december": "12",
}


def parse_date_to_iso_or_null(raw: str) -> str | None:
    s = raw.strip()
    m1 = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", s)
    if m1:
        y, mo, d = int(m1.group(1)), int(m1.group(2)), int(m1.group(3))
        if 1900 <= y <= 2100 and 1 <= mo <= 12 and 1 <= d <= 31:
            return f"{y}-{mo:02d}-{d:02d}"
        return None
    m2 = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{4})\b", s)
    if m2:
        a, b, year = int(m2.group(1)), int(m2.group(2)), int(m2.group(3))
        if 1 <= a <= 12 and 1 <= b <= 31 and 1900 <= year <= 2100:
            return f"{year}-{a:02d}-{b:02d}"
    m2b = re.search(r"\b(\d{1,2})/(\d{1,2})/(\d{2})\b", s)
    if m2b:
        a, b, yy = int(m2b.group(1)), int(m2b.group(2)), int(m2b.group(3))
        year = 1900 + yy if yy >= 50 else 2000 + yy
        if 1 <= a <= 12 and 1 <= b <= 31:
            return f"{year}-{a:02d}-{b:02d}"
    m3 = re.search(r"\b(\d{1,2})\s+([A-Za-z]{3,9})\s+(\d{4})\b", s)
    if m3:
        dd = int(m3.group(1))
        mon = MONTH_MAP.get(m3.group(2).lower())
        yy = int(m3.group(3))
        if mon and 1 <= dd <= 31 and 1900 <= yy <= 2100:
            return f"{yy}-{mon}-{dd:02d}"
    m4 = re.search(r"\b([A-Za-z]{3,9})\s+(\d{1,2}),?\s+(\d{4})\b", s)
    if m4:
        mon = MONTH_MAP.get(m4.group(1).lower())
        dd = int(m4.group(2))
        yy = int(m4.group(3))
        if mon and 1 <= dd <= 31 and 1900 <= yy <= 2100:
            return f"{yy}-{mon}-{dd:02d}"
    return None


def validate_due_date_candidate(candidate_date: str, raw_evidence: str) -> str | None:
    ev = (raw_evidence or "").lower()
    for cue in POP_CUES:
        if cue in ev:
            return None
    if re.search(r"\bpop\s", ev):
        return None

    has_submission = any(cue in ev for cue in SUBMISSION_CUES) or re.search(
        r"date\s*[:\s]*[^\n]*\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b", ev, re.I
    ) or ("sealed offers" in ev and "will be received" in ev)
    if not has_submission:
        return None

    parsed = parse_date_to_iso_or_null(candidate_date.strip())
    if parsed:
        return parsed
    local = re.search(r"until\s+[\d:]+\s*(?:AM|PM)?\s*local\s+time\s+(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})", raw_evidence, re.I)
    if local:
        return parse_date_to_iso_or_null(local.group(1).strip())
    any_date = re.search(r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4})\b", raw_evidence, re.I)
    if any_date:
        return parse_date_to_iso_or_null(any_date.group(1).strip())
    return None
