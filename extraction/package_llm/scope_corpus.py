from __future__ import annotations

import re
from typing import Any

from extraction.package_llm.corpus import PackageSourceText, assemble_package_corpus
from extraction.package_llm.versions import FAST_CORPUS_VERSION

_SECTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "set_aside": re.compile(r"\bset[- ]aside\b|\bsmall business\b|\b8\s*\(\s*a\s*\)|\bunrestricted\b|\bfull and open\b", re.I),
    "psc_clin": re.compile(r"\b(?:clin|psc|product service code|schedule of supplies|pricing arrangement)\b", re.I),
    "contract_type": re.compile(r"\b(?:contract type|ffp|firm[- ]fixed|firm fixed price|cost plus|t&m)\b", re.I),
    "issuing_office": re.compile(r"\b(?:issuing office|contracting office|aac|activity address code|325 cons)\b", re.I),
    "status_stay": re.compile(r"\b(?:protest|stay|stalled|indefinite|gao|solicitation status|amendment of solicitation)\b", re.I),
    "pop_scope": re.compile(r"\b(?:place of performance|tyndall|performance location|city/state)\b", re.I),
    "sf30": re.compile(r"\b(?:sf\s*30|standard form 30|amendment of solicitation|modification)\b", re.I),
}

_MAX_EXCERPT = 3200
_MAX_POP_EXCERPT = 1800
_MAX_STATUS_EXCERPT = 2200


def _excerpt(text: str, *, limit: int = _MAX_EXCERPT) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[:limit].rstrip() + "\n[... excerpt truncated ...]"


def _page_matches(text: str, patterns: list[re.Pattern[str]]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def _select_pages(source: PackageSourceText) -> list[tuple[int, str, str]]:
    """Return (page_index, marker_suffix, text) tuples for fast market-scope corpus."""
    selected: list[tuple[int, str, str]] = []
    seen: set[int] = set()

    def add_page(page_index: int, suffix: str, text: str) -> None:
        if page_index in seen:
            return
        seen.add(page_index)
        selected.append((page_index, suffix, _excerpt(text)))

    # Base solicitation cover / form pages.
    for page in source.pages[:3]:
        text = source.normalized_text_by_page.get(page.page_index, page.text)
        add_page(page.page_index, "COVER/FORM", text)

    for page in source.pages:
        text = source.normalized_text_by_page.get(page.page_index, page.text)
        if not text.strip():
            continue
        if _page_matches(text, [_SECTION_PATTERNS["sf30"]]):
            add_page(page.page_index, "SF30 AMENDMENT", text)
        if source.document_type_hint in {"amendment", "amendment continuation"}:
            add_page(page.page_index, "AMENDMENT", text)
        if _page_matches(text, [_SECTION_PATTERNS["issuing_office"]]):
            add_page(page.page_index, "ISSUING OFFICE", text)
        if _page_matches(text, [_SECTION_PATTERNS["set_aside"]]):
            add_page(page.page_index, "SET-ASIDE", text)
        if _page_matches(text, [_SECTION_PATTERNS["psc_clin"], _SECTION_PATTERNS["contract_type"]]):
            add_page(page.page_index, "CLIN / PSC / CONTRACT TYPE", text)
        if _page_matches(text, [_SECTION_PATTERNS["status_stay"]]):
            add_page(page.page_index, "STATUS / STAY", _excerpt(text, limit=_MAX_STATUS_EXCERPT))
        if _page_matches(text, [_SECTION_PATTERNS["pop_scope"]]):
            add_page(page.page_index, "PLACE OF PERFORMANCE", _excerpt(text, limit=_MAX_POP_EXCERPT))

    return sorted(selected, key=lambda item: item[0])


def assemble_scope_corpus(sources: list[PackageSourceText]) -> tuple[str, dict[str, Any]]:
    blocks: list[str] = []
    for source in sources:
        lines = [
            "===== SOURCE BEGIN =====",
            f"source_id: {source.source_id}",
            f"filename: {source.filename}",
            f"file_type: {source.file_type}",
            f"document_type_hint: {source.document_type_hint}",
            "",
        ]
        if source.sheets:
            # Pricing workbook: only first sheet header rows for PSC/contract type hints.
            for item in source.sheets[:1]:
                sheet_name = str(item["sheet_name"])
                text = _excerpt(str(item.get("text") or ""), limit=1800)
                lines.append(f"--- SHEET: {sheet_name} ---")
                lines.append(text)
                lines.append("")
        else:
            if source.form_fields:
                from extraction.package_llm.forms.pdf_forms import format_form_fields_corpus_block

                lines.append(format_form_fields_corpus_block(source.form_fields))
                lines.append("")
            for page_index, suffix, text in _select_pages(source):
                marker = f"--- PAGE {page_index + 1}"
                if suffix:
                    marker += f" ({suffix})"
                marker += " ---"
                if page_index in source.ocr_pages:
                    marker = marker.replace(" ---", " (OCR) ---")
                lines.append(marker)
                lines.append(text)
                lines.append("")
        lines.append("===== SOURCE END =====")
        blocks.append("\n".join(lines))

    full_text = assemble_package_corpus(sources)
    fast_text = "\n\n".join(blocks)
    full_chars = len(full_text)
    fast_chars = len(fast_text)
    reduction_pct = round((1 - (fast_chars / full_chars)) * 100, 1) if full_chars else 0.0
    stats = {
        "corpusBuilderVersion": FAST_CORPUS_VERSION,
        "fullCorpusCharCount": full_chars,
        "fastCorpusCharCount": fast_chars,
        "corpusReductionPct": reduction_pct,
        "fullCorpusApproxTokens": max(1, full_chars // 4),
        "fastCorpusApproxTokens": max(1, fast_chars // 4),
    }
    return fast_text, stats
