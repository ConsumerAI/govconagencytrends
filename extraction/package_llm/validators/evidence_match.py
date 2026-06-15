from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

from extraction.package_llm.corpus import PackageSourceText

LIGATURES = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "ﬅ": "ft",
    "ﬆ": "st",
}

ELLIPSIS_RE = re.compile(r"(?:\.\.\.|…)+")
TOKEN_RE = re.compile(r"[a-z0-9]+")
CONTROL_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")

EXACT_MATCH = "exact_normalized"
NEARBY_PAGE_EXACT = "nearby_page_exact"
SOURCE_EXACT = "source_exact"
SEGMENT_ORDERED = "segment_ordered"
TOKEN_WINDOW = "token_window"
FUZZY_WINDOW = "fuzzy_window"

TOKEN_COVERAGE_THRESHOLD = 0.82
FUZZY_TOKEN_THRESHOLD = 0.88
NEARBY_PAGE_TOLERANCE = 1


@dataclass
class EvidenceMatchResult:
    matched: bool
    method: str | None = None
    score: float = 0.0
    matched_page: int | None = None
    matched_page_index: int | None = None
    matched_source_id: str | None = None
    matched_sheet: str | None = None
    normalized_excerpt: str | None = None
    cited_page: int | None = None
    cited_page_index: int | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def page_number_to_index(page: int | None) -> int | None:
    if page is None:
        return None
    if page > 0:
        return page - 1
    return page


def page_index_to_number(page_index: int | None) -> int | None:
    if page_index is None:
        return None
    return page_index + 1


def normalize_evidence_text(text: str) -> str:
    """Normalize text for conservative evidence matching."""
    value = unicodedata.normalize("NFKC", text or "")
    for src, dst in LIGATURES.items():
        value = value.replace(src, dst)
    value = (
        value.replace("\ufffd", "'")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("“", '"')
        .replace("”", '"')
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u00ad", "")
    )
    value = CONTROL_RE.sub(" ", value)
    value = value.lower()
    value = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", value)
    value = re.sub(r"\s+-\s+", " ", value)
    value = re.sub(r"(\w)\s+-\s*(\w)", r"\1\2", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def relax_alnum(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", normalize_evidence_text(text))


def meaningful_tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(normalize_evidence_text(text)) if len(token) >= 2]


def split_ellipsis_segments(quote: str) -> list[str]:
    parts = [part.strip() for part in ELLIPSIS_RE.split(quote or "") if part.strip()]
    return parts or [quote.strip()]


def _page_haystack(source: PackageSourceText, page_index: int) -> str:
    return source.normalized_text_by_page.get(page_index, "")


def _sheet_haystack(source: PackageSourceText, sheet: str | None) -> str:
    if not sheet:
        return ""
    if sheet in source.normalized_text_by_sheet:
        return source.normalized_text_by_sheet[sheet]
    lowered = sheet.lower()
    for name, text in source.normalized_text_by_sheet.items():
        if name.lower() == lowered:
            return text
    return ""


def _source_haystack(source: PackageSourceText) -> str:
    return "\n".join(source.normalized_text_by_page.values()) + "\n" + "\n".join(
        source.normalized_text_by_sheet.values()
    )


def _exact_in_haystack(quote: str, haystack: str) -> bool:
    normalized_quote = normalize_evidence_text(quote)
    if not normalized_quote:
        return False
    normalized_haystack = normalize_evidence_text(haystack)
    if normalized_quote in normalized_haystack:
        return True
    return relax_alnum(quote) in relax_alnum(haystack)


def _ordered_token_coverage(quote_tokens: list[str], haystack_tokens: list[str], *, max_gap: int = 12) -> float:
    if not quote_tokens:
        return 0.0
    hi = 0
    matched = 0
    for token in quote_tokens:
        found_at = None
        search_end = min(len(haystack_tokens), hi + max_gap + 1)
        for idx in range(hi, search_end):
            if haystack_tokens[idx] == token:
                found_at = idx
                break
        if found_at is None:
            for idx in range(hi, len(haystack_tokens)):
                if haystack_tokens[idx] == token:
                    found_at = idx
                    break
        if found_at is None:
            continue
        matched += 1
        hi = found_at + 1
    return matched / len(quote_tokens)


def _segment_ordered_match(quote: str, haystack: str) -> tuple[bool, float]:
    segments = split_ellipsis_segments(quote)
    if len(segments) <= 1:
        return False, 0.0
    cursor = 0
    normalized_haystack = normalize_evidence_text(haystack)
    scores: list[float] = []
    for segment in segments:
        normalized_segment = normalize_evidence_text(segment)
        if not normalized_segment:
            continue
        pos = normalized_haystack.find(normalized_segment, cursor)
        if pos >= 0:
            scores.append(1.0)
            cursor = pos + len(normalized_segment)
            continue
        relaxed = relax_alnum(segment)
        relaxed_hay = relax_alnum(normalized_haystack[cursor:])
        pos = relaxed_hay.find(relaxed)
        if pos >= 0:
            coverage = _ordered_token_coverage(meaningful_tokens(segment), meaningful_tokens(haystack))
            scores.append(max(coverage, 0.75))
            cursor += pos + len(relaxed)
            continue
        coverage = _ordered_token_coverage(meaningful_tokens(segment), meaningful_tokens(haystack))
        if coverage < TOKEN_COVERAGE_THRESHOLD:
            return False, coverage
        scores.append(coverage)
    if not scores:
        return False, 0.0
    return True, sum(scores) / len(scores)


def _fuzzy_window_match(quote: str, haystack: str) -> tuple[bool, float, str | None]:
    quote_tokens = meaningful_tokens(quote)
    haystack_tokens = meaningful_tokens(haystack)
    if len(quote_tokens) < 4:
        return False, 0.0, None
    best_score = 0.0
    best_excerpt = None
    window = max(len(quote_tokens) + 8, int(len(quote_tokens) * 1.5))
    for start in range(0, max(len(haystack_tokens) - 3, 1)):
        end = min(len(haystack_tokens), start + window)
        window_tokens = haystack_tokens[start:end]
        score = _ordered_token_coverage(quote_tokens, window_tokens, max_gap=16)
        if score > best_score:
            best_score = score
            best_excerpt = " ".join(window_tokens[: min(len(window_tokens), len(quote_tokens) + 6)])
        if best_score >= FUZZY_TOKEN_THRESHOLD:
            break
    return best_score >= FUZZY_TOKEN_THRESHOLD, best_score, best_excerpt


def _match_in_haystack(
    quote: str,
    haystack: str,
    *,
    method_prefix: str,
    page: int | None = None,
    page_index: int | None = None,
    sheet: str | None = None,
    source_id: str | None = None,
) -> EvidenceMatchResult:
    cited_page_index = page_number_to_index(page)
    if _exact_in_haystack(quote, haystack):
        excerpt = normalize_evidence_text(quote)[:240]
        return EvidenceMatchResult(
            matched=True,
            method=method_prefix if method_prefix.endswith("_exact") else EXACT_MATCH,
            score=1.0,
            matched_page=page,
            matched_page_index=page_index if page_index is not None else cited_page_index,
            matched_source_id=source_id,
            matched_sheet=sheet,
            normalized_excerpt=excerpt,
            cited_page=page,
            cited_page_index=cited_page_index,
        )

    segment_ok, segment_score = _segment_ordered_match(quote, haystack)
    if segment_ok:
        return EvidenceMatchResult(
            matched=True,
            method=SEGMENT_ORDERED,
            score=segment_score,
            matched_page=page,
            matched_page_index=page_index if page_index is not None else cited_page_index,
            matched_source_id=source_id,
            matched_sheet=sheet,
            normalized_excerpt=normalize_evidence_text(split_ellipsis_segments(quote)[0])[:240],
            cited_page=page,
            cited_page_index=cited_page_index,
        )

    quote_tokens = meaningful_tokens(quote)
    haystack_tokens = meaningful_tokens(haystack)
    token_score = _ordered_token_coverage(quote_tokens, haystack_tokens)
    if token_score >= TOKEN_COVERAGE_THRESHOLD:
        return EvidenceMatchResult(
            matched=True,
            method=TOKEN_WINDOW,
            score=token_score,
            matched_page=page,
            matched_page_index=page_index if page_index is not None else cited_page_index,
            matched_source_id=source_id,
            matched_sheet=sheet,
            normalized_excerpt=" ".join(haystack_tokens[: min(len(haystack_tokens), len(quote_tokens) + 4)])[:240],
            cited_page=page,
            cited_page_index=cited_page_index,
            diagnostics={"tokenCoverage": round(token_score, 3)},
        )

    fuzzy_ok, fuzzy_score, excerpt = _fuzzy_window_match(quote, haystack)
    if fuzzy_ok:
        return EvidenceMatchResult(
            matched=True,
            method=FUZZY_WINDOW,
            score=fuzzy_score,
            matched_page=page,
            matched_page_index=page_index if page_index is not None else cited_page_index,
            matched_source_id=source_id,
            matched_sheet=sheet,
            normalized_excerpt=(excerpt or "")[:240],
            cited_page=page,
            cited_page_index=cited_page_index,
            diagnostics={"tokenCoverage": round(fuzzy_score, 3)},
        )

    return EvidenceMatchResult(
        matched=False,
        method=None,
        score=max(token_score, fuzzy_score, segment_score),
        matched_source_id=source_id,
        cited_page=page,
        cited_page_index=cited_page_index,
        diagnostics={
            "tokenCoverage": round(token_score, 3),
            "segmentScore": round(segment_score, 3),
            "fuzzyScore": round(fuzzy_score, 3),
        },
    )


def match_evidence_quote(
    source: PackageSourceText,
    *,
    page: int | None,
    sheet: str | None,
    quote: str,
    source_id: str | None = None,
) -> EvidenceMatchResult:
    if not str(quote or "").strip():
        return EvidenceMatchResult(matched=False, matched_source_id=source_id, cited_page=page, cited_page_index=page_number_to_index(page))

    if sheet:
        haystack = _sheet_haystack(source, sheet)
        result = _match_in_haystack(
            quote,
            haystack,
            method_prefix=EXACT_MATCH,
            page=page,
            sheet=sheet,
            source_id=source_id or source.source_id,
        )
        if result.matched:
            return result

    cited_page_index = page_number_to_index(page)
    candidate_pages: list[int] = []
    if cited_page_index is not None:
        for offset in (0, -1, 1):
            idx = cited_page_index + offset
            if 0 <= idx < len(source.pages) and abs(offset) <= NEARBY_PAGE_TOLERANCE:
                candidate_pages.append(idx)
    elif source.pages:
        candidate_pages = [page.page_index for page in source.pages]

    best_failed: EvidenceMatchResult | None = None
    for page_index in candidate_pages:
        haystack = _page_haystack(source, page_index)
        page_number = page_index_to_number(page_index)
        method_prefix = EXACT_MATCH if page_index == cited_page_index else NEARBY_PAGE_EXACT
        result = _match_in_haystack(
            quote,
            haystack,
            method_prefix=method_prefix,
            page=page_number,
            page_index=page_index,
            source_id=source_id or source.source_id,
        )
        if result.matched:
            if page_index != cited_page_index and cited_page_index is not None:
                result.diagnostics["pageAdjustedFrom"] = page_index_to_number(cited_page_index)
                result.diagnostics["pageAdjustedTo"] = page_number
            return result
        if best_failed is None or result.score > best_failed.score:
            best_failed = result

    source_haystack = _source_haystack(source)
    result = _match_in_haystack(
        quote,
        source_haystack,
        method_prefix=SOURCE_EXACT,
        page=page,
        page_index=cited_page_index,
        source_id=source_id or source.source_id,
    )
    if result.matched:
        if cited_page_index is not None:
            result.diagnostics["matchedInFullSourceFallback"] = True
        return result

    return best_failed or result
