from __future__ import annotations

import re

AMENDMENT_FILENAME_TOKENS = (
    "amend",
    "amnd",
    "amendment",
    "a000",
    "0001",
    "0002",
    "0003",
    "0004",
    "0005",
    "0006",
    "0007",
    "0008",
    "0009",
)

AMENDMENT_CONTENT_RE = re.compile(
    r"(?:amendment\s*(?:no\.?|number|#)?\s*(\d{1,4})|"
    r"modification\s*(?:no\.?|number|#)?\s*(\d{1,4})|"
    r"sf[\s-]?30|standard\s+form\s+30|"
    r"\ba(\d{3,4})\b)",
    re.IGNORECASE,
)

SOLICITATION_NUMBER_RE = re.compile(
    r"\b([A-Z]{2}\d{4}[A-Z0-9]{2,}[A-Z0-9]*)\b",
    re.IGNORECASE,
)

DELETION_PHRASE_RE = re.compile(
    r"\b(?:removed|deleted|cancelled|canceled|no\s+longer\s+(?:required|applicable|valid)|"
    r"is\s+hereby\s+(?:removed|deleted|cancelled|canceled))\b",
    re.IGNORECASE,
)


def is_amendment_filename(file_name: str) -> bool:
    lower = str(file_name or "").lower()
    return any(token in lower for token in AMENDMENT_FILENAME_TOKENS)


def normalize_amendment_order(raw: str | None) -> str | None:
    if not raw:
        return None
    digits = re.sub(r"\D", "", raw)
    if not digits:
        return None
    try:
        value = int(digits)
    except ValueError:
        return None
    if value < 0 or value > 9999:
        return None
    return f"{value:04d}"


def detect_amendment_from_filename(file_name: str) -> tuple[str | None, str | None]:
    lower = str(file_name or "").lower()
    if not is_amendment_filename(lower):
        return None, None

    for match in re.finditer(r"(?:amendment|amend|modification|mod)[_\s-]*(\d{1,4})", lower):
        order = normalize_amendment_order(match.group(1))
        if order:
            return match.group(0).upper(), order

    for match in re.finditer(r"\b0{0,1}(\d{4})\b", lower):
        token = match.group(1)
        if token in {"0001", "0002", "0003", "0004", "0005", "0006", "0007", "0008", "0009"}:
            return f"Amendment {token}", token

    for match in re.finditer(r"\ba(\d{3,4})\b", lower):
        order = normalize_amendment_order(match.group(1))
        if order:
            return f"A{match.group(1).upper()}", order

    if any(token in lower for token in ("amend", "amnd", "amendment", "sf-30", "sf30")):
        return "Amendment", None
    return None, None


def detect_amendment_from_text(text: str) -> tuple[str | None, str | None]:
    if not text.strip():
        return None, None
    head = text[:4000]
    best: tuple[str | None, str | None] = (None, None)
    for match in AMENDMENT_CONTENT_RE.finditer(head):
        raw_num = next((group for group in match.groups() if group), None)
        if raw_num:
            order = normalize_amendment_order(raw_num)
            if order:
                return match.group(0).strip(), order
        if "sf" in match.group(0).lower():
            best = ("SF-30", best[1])
    return best


def detect_solicitation_number(text: str, file_name: str = "") -> str | None:
    for source in (file_name, text[:3000]):
        if not source:
            continue
        for match in SOLICITATION_NUMBER_RE.finditer(source):
            candidate = match.group(1).upper()
            if re.search(r"R\d{4}$", candidate, flags=re.IGNORECASE):
                return candidate
        for match in SOLICITATION_NUMBER_RE.finditer(source):
            return match.group(1).upper()
    return None


def index_of_highest_precedence(documents: list[dict]) -> int:
    if not documents:
        return -1
    last_amendment_idx = -1
    for index, doc in enumerate(documents):
        if doc.get("isAmendment") or is_amendment_filename(str(doc.get("fileName") or "")):
            last_amendment_idx = index
    return last_amendment_idx if last_amendment_idx >= 0 else len(documents) - 1


def is_explicit_deletion(text: str, field_hint: str = "") -> bool:
    if not text.strip():
        return False
    if not DELETION_PHRASE_RE.search(text):
        return False
    if not field_hint:
        return True
    hint = field_hint.lower()
    window = text.lower()
    hint_pos = window.find(hint)
    if hint_pos < 0:
        return bool(DELETION_PHRASE_RE.search(text))
    start = max(0, hint_pos - 120)
    end = min(len(window), hint_pos + len(hint) + 120)
    return bool(DELETION_PHRASE_RE.search(window[start:end]))
