from __future__ import annotations

import os
import re
from typing import Any

from extraction.types import CorpusPage, Finding

EXCERPT_MAX = 420
SECTION_HEADING_RE = re.compile(
    r"^\s*(?:SECTION|S\s*E\s*C\s*T\s*I\s*O\s*N)\s+([A-N])\b",
    re.IGNORECASE | re.MULTILINE,
)
SECTION_M_CONTINUATION_RE = re.compile(
    r"\b(?:CONTINUATION\s+OF\s+(?:SECTION\s+)?M|SECTION\s+M\s+CONTINUATION)\b",
    re.I,
)
REPLACE_ENTIRETY_RE = re.compile(r"\b(?:REPLACE\s+(?:IN\s+)?(?:ITS\s+)?ENTIRETY|DELETE\s+AND\s+SUBSTITUTE)\b", re.I)
ATTACHMENT_BREAK_RE = re.compile(
    r"^\s*(?:ATTACHMENT|EXHIBIT|APPENDIX|ANNEX)\b",
    re.IGNORECASE | re.MULTILINE,
)
INCORPORATED_EVAL_RE = re.compile(
    r"\b(?:INCORPORATED\s+BY\s+(?:REFERENCE|AMENDMENT)|ATTACHMENT\s+\d+.*EVALUATION)\b",
    re.I,
)
CLAUSE_ID_RE = re.compile(r"\b(?:FAR|DFARS)\s+\d{2,3}\.\d{3,4}(?:-\d{1,4})?\b|\b(?:52|252)\.\d{3,4}(?:-\d{1,4})?\b", re.I)


def _section_char_cap(section: str) -> int:
    key = "GOVCON_SECTION_M_CHAR_CAP" if section.upper() == "M" else "GOVCON_SECTION_L_CHAR_CAP"
    default = "120000" if section.upper() == "M" else "80000"
    try:
        return max(8000, int(os.getenv(key, default)))
    except ValueError:
        return int(default)


def _truncate(text: str, limit: int = EXCERPT_MAX) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _span_hash(source_sha256: str, page_index: int, start: int, end: int) -> str:
    import hashlib

    payload = f"{source_sha256}:{page_index}:{start}:{end}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _line_start(text: str, idx: int) -> int:
    if idx <= 0:
        return 0
    pos = text.rfind("\n", 0, idx)
    return 0 if pos == -1 else pos + 1


def _line_text(text: str, idx: int) -> str:
    start = _line_start(text, idx)
    end = text.find("\n", start)
    return text[start : end if end != -1 else len(text)]


def _extract_title(section: str, line: str) -> str | None:
    match = re.match(
        rf"^\s*(?:SECTION|S\s*E\s*C\s*T\s*I\s*O\s*N)\s+{section.upper()}\s*(?:[-–:]\s*)?(.*)$",
        line,
        re.I,
    )
    if not match:
        return None
    title = (match.group(1) or "").strip()
    return title or None


def _count_clause_ids(snippet: str) -> int:
    return len(CLAUSE_ID_RE.findall(snippet))


def _spine_pages(pages: list[CorpusPage]) -> dict[str, set[int]]:
    spine: dict[str, set[int]] = {}
    for page in pages:
        if SECTION_HEADING_RE.search(page.text):
            spine.setdefault(page.source_sha256, set()).add(page.page_index)
    return spine


def _has_spine_near(spine: set[int], page_index: int) -> bool:
    return any(abs(page_index - idx) <= 3 for idx in spine)


def _find_next_heading(
    pages: list[CorpusPage],
    *,
    source_sha256: str,
    after_page_index: int,
    after_offset: int,
    line_pattern: re.Pattern[str],
) -> tuple[int, int] | None:
    source_pages = sorted(
        [page for page in pages if page.source_sha256 == source_sha256],
        key=lambda item: item.page_index,
    )
    for page in source_pages:
        if page.page_index < after_page_index:
            continue
        offset = 0
        for line in page.text.split("\n"):
            line_start = offset
            line_end = line_start + len(line)
            after = page.page_index > after_page_index or line_start >= after_offset
            if after and line_pattern.search(line):
                return page.page_index, line_start
            offset = line_end + 1
    return None


def _score_heading_candidate(
    *,
    section: str,
    page: CorpusPage,
    offset: int,
    line: str,
    spine: dict[str, set[int]],
) -> int:
    keyword = "INSTRUCTIONS" if section.upper() == "L" else "EVALUATION"
    score = 0
    if offset == 0 or (offset > 0 and page.text[offset - 1] == "\n"):
        score += 40
    upper = line.upper()
    if keyword in upper:
        score += 25
    if _has_spine_near(spine.get(page.source_sha256, set()), page.page_index):
        score += 15
    around = page.text[max(0, offset - 600) : min(len(page.text), offset + 600)]
    if _count_clause_ids(around) >= 3 and keyword not in upper:
        score -= 30
    if REPLACE_ENTIRETY_RE.search(around):
        score += 20
    return score


def _collect_section_slices(
    pages: list[CorpusPage],
    *,
    source_sha256: str,
    start_page_index: int,
    start_offset: int,
    end_page_index: int,
    end_offset: int,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    slices: list[str] = []
    span_hashes: list[str] = []
    page_ranges: list[dict[str, Any]] = []
    source_pages = sorted(
        [page for page in pages if page.source_sha256 == source_sha256],
        key=lambda item: item.page_index,
    )
    for page in source_pages:
        if page.page_index < start_page_index or page.page_index > end_page_index:
            continue
        win_start = start_offset if page.page_index == start_page_index else 0
        win_end = end_offset if page.page_index == end_page_index else len(page.text)
        chunk = page.text[win_start:win_end]
        if chunk.strip():
            slices.append(chunk)
            span_hashes.append(_span_hash(page.source_sha256, page.page_index, win_start, win_end))
            page_ranges.append(
                {
                    "sourceSha256": page.source_sha256,
                    "pageIndex": page.page_index,
                    "start": win_start,
                    "end": win_end,
                }
            )
    return slices, span_hashes, page_ranges


def _incorporated_evaluation_attachment_text(
    structure: dict[str, Any] | None,
    pages: list[CorpusPage],
) -> list[tuple[str, list[str], list[dict[str, Any]]]]:
    if not structure:
        return []
    incorporated: list[tuple[str, list[str], list[dict[str, Any]]]] = []
    for item in structure.get("items") or []:
        logical = str(item.get("logicalType") or "")
        if logical not in {"ATTACHMENT", "EXHIBIT"}:
            continue
        title = str(item.get("title") or item.get("heading") or "").lower()
        excerpt = str(item.get("excerpt") or item.get("fullText") or "")
        if "evaluation" not in title and not re.search(r"\bevaluation\s+factors?\b", excerpt, re.I):
            continue
        if item.get("parentItemId") and not INCORPORATED_EVAL_RE.search(excerpt):
            continue
        if logical == "ATTACHMENT" and not (
            INCORPORATED_EVAL_RE.search(excerpt)
            or "evaluation factor" in title
            or re.search(r"\battachment\s+\d+.*evaluation\b", title, re.I)
            or re.search(r"\bevaluation\s+factors?\b", text, re.I)
        ):
            continue
        text = str(item.get("fullText") or item.get("excerpt") or "")
        if not text.strip():
            continue
        span_hashes = list(item.get("spanHashes") or [])
        page_ranges = [
            {
                "sourceSha256": item.get("sourceSha256"),
                "pageIndex": item.get("pageIndex"),
                "attachmentNumber": item.get("attachmentNumber"),
            }
        ]
        incorporated.append((text, span_hashes, page_ranges))
    return incorporated


def extract_section_fulltext_v1(
    run_id: str,
    pages: list[CorpusPage],
    *,
    section: str,
    structure: dict[str, Any] | None = None,
    filename: str = "",
    amendment_number: str | None = None,
) -> dict[str, Any] | None:
    letter = section.upper()
    heading_token = rf"(?:SECTION|S\s*E\s*C\s*T\s*I\s*O\s*N)\s+{letter}\b"
    heading_re = re.compile(heading_token, re.IGNORECASE)
    findings: list[Finding] = []
    spine = _spine_pages(pages)

    candidates: list[dict[str, Any]] = []
    for page in pages:
        for match in heading_re.finditer(page.text):
            offset = match.start()
            line = _line_text(page.text, offset)
            candidates.append(
                {
                    "sourceSha256": page.source_sha256,
                    "pageIndex": page.page_index,
                    "offset": offset,
                    "lineStartOffset": _line_start(page.text, offset),
                    "line": line,
                    "title": _extract_title(letter, line),
                    "score": _score_heading_candidate(
                        section=letter,
                        page=page,
                        offset=offset,
                        line=line,
                        spine=spine,
                    ),
                    "amendmentNumber": amendment_number,
                    "filename": filename,
                }
            )

    candidates.sort(
        key=lambda item: (
            -int(item["score"]),
            -(int(amendment_number) if amendment_number and str(amendment_number).isdigit() else -1),
            item["sourceSha256"],
            item["pageIndex"],
            item["offset"],
        )
    )

    superseded_segments: list[dict[str, Any]] = []
    chosen = candidates[0] if candidates else None

    if letter == "M" and len(candidates) > 1:
        top = candidates[0]
        if top and REPLACE_ENTIRETY_RE.search(str(top.get("line") or "")):
            for stale in candidates[1:]:
                superseded_segments.append(
                    {
                        "reason": "replaced_in_entirety",
                        "sourceSha256": stale["sourceSha256"],
                        "pageIndex": stale["pageIndex"],
                        "filename": stale.get("filename"),
                        "amendmentNumber": stale.get("amendmentNumber"),
                    }
                )
        elif top and candidates[1] and top["score"] > candidates[1]["score"]:
            second = candidates[1]
            superseded_segments.append(
                {
                    "reason": "lower_authority_section_m",
                    "sourceSha256": second["sourceSha256"],
                    "pageIndex": second["pageIndex"],
                    "filename": second.get("filename"),
                    "amendmentNumber": second.get("amendmentNumber"),
                }
            )

    if not chosen:
        code = f"SECTION_{letter}_NOT_FOUND"
        findings.append(Finding("warn", code, f"Section {letter} heading not found"))
        return None

    source_sha256 = str(chosen["sourceSha256"])
    start_page_index = int(chosen["pageIndex"])
    start_offset = int(chosen["lineStartOffset"])

    end_line_re = (
        re.compile(r"^\s*(?:SECTION|S\s*E\s*C\s*T\s*I\s*O\s*N)\s+M\b", re.I)
        if letter == "L"
        else SECTION_HEADING_RE
    )
    end_heading = _find_next_heading(
        pages,
        source_sha256=source_sha256,
        after_page_index=start_page_index,
        after_offset=start_offset + 1,
        line_pattern=end_line_re,
    )
    source_pages = sorted(
        [page for page in pages if page.source_sha256 == source_sha256],
        key=lambda item: item.page_index,
    )
    last_page = source_pages[-1] if source_pages else None
    end_page_index = last_page.page_index if last_page else start_page_index
    end_offset = len(last_page.text) if last_page else 0
    if letter == "M" and not end_heading:
        for page in source_pages:
            if page.page_index <= start_page_index:
                continue
            attach_match = ATTACHMENT_BREAK_RE.search(page.text)
            if attach_match and not re.search(r"\bevaluation\b", page.text[: attach_match.start() + 120], re.I):
                end_page_index = page.page_index
                end_offset = attach_match.start()
                break

    if end_heading:
        end_page_index = end_heading[0]
        end_offset = end_heading[1]

    slices, span_hashes, page_ranges = _collect_section_slices(
        pages,
        source_sha256=source_sha256,
        start_page_index=start_page_index,
        start_offset=start_offset,
        end_page_index=end_page_index,
        end_offset=end_offset,
    )

    if letter == "M":
        for page in source_pages:
            if page.page_index <= start_page_index:
                continue
            if page.page_index > end_page_index:
                break
            if SECTION_M_CONTINUATION_RE.search(page.text) and page.page_index not in {r["pageIndex"] for r in page_ranges}:
                extra_slices, extra_hashes, extra_ranges = _collect_section_slices(
                    pages,
                    source_sha256=source_sha256,
                    start_page_index=page.page_index,
                    start_offset=0,
                    end_page_index=page.page_index,
                    end_offset=len(page.text),
                )
                slices.extend(extra_slices)
                span_hashes.extend(extra_hashes)
                page_ranges.extend(extra_ranges)

        for attach_text, attach_hashes, attach_ranges in _incorporated_evaluation_attachment_text(structure, pages):
            slices.append(attach_text)
            span_hashes.extend(attach_hashes)
            page_ranges.extend(attach_ranges)

    full_text = "\n\n".join(slices).strip()
    truncated = False
    cap = _section_char_cap(letter)
    if len(full_text) > cap:
        full_text = full_text[: cap - 1] + "…"
        truncated = True
        findings.append(
            Finding(
                "warn",
                f"SECTION_{letter}_TRUNCATED",
                f"Section {letter} fulltext exceeded cap and was truncated",
                {"cap": cap, "filename": filename},
            )
        )

    if not full_text.strip():
        findings.append(Finding("warn", "SECTION_TEXT_EMPTY", f"Section {letter} window produced empty text"))
        return None

    evidence: dict[str, Any] = {
        "sourceSha256": source_sha256,
        "spanHashes": span_hashes,
        "pageIndexStart": start_page_index,
        "pageIndexEnd": end_page_index,
        "pageRanges": page_ranges,
        "filename": filename or None,
        "amendmentNumber": amendment_number,
    }
    if superseded_segments:
        evidence["supersededSegments"] = superseded_segments

    return {
        "version": 1,
        "runId": run_id,
        "section": letter,
        "heading": f"SECTION {letter}",
        "title": chosen.get("title"),
        "fullText": full_text,
        "evidence_v1": evidence,
        "excerpt": _truncate(full_text.replace("\n", " ")),
        "findings": [item.to_dict() for item in findings],
        "truncated": truncated,
    }
