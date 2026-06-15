from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass

from pypdf import PdfReader

# Align with AEGIS extract/pdf.ts scanned detection thresholds.
SCANNED_TOTAL_CHAR_THRESHOLD = 200
SCANNED_AVG_CHAR_PER_PAGE_THRESHOLD = 50


@dataclass
class PdfExtractResult:
    text: str
    page_texts: list[str]
    page_count: int
    total_char_count: int
    avg_char_count_per_page: float
    is_probably_scanned: bool
    low_confidence_reason: str | None


def _parse_page_markers(text: str) -> list[str]:
    pattern = re.compile(r"\n\n--- PAGE (\d+) ---\n\n")
    matches = list(pattern.finditer(text))
    if not matches:
        return [text]
    pages: list[tuple[int, str]] = []
    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages.append((max(0, page_number - 1), text[start:end]))
    pages.sort(key=lambda item: item[0])
    return [page_text for _, page_text in pages]


def extract_pdf_bytes(data: bytes) -> PdfExtractResult:
    reader = PdfReader(io.BytesIO(data))
    page_count = len(reader.pages)
    page_chunks: list[str] = []
    char_counts: list[int] = []

    for page_number in range(1, page_count + 1):
        page = reader.pages[page_number - 1]
        page_text = (page.extract_text() or "").strip()
        char_counts.append(len(page_text))
        page_chunks.append(f"\n\n--- PAGE {page_number} ---\n\n{page_text}")

    text = "".join(page_chunks)
    page_texts = _parse_page_markers(text) if page_chunks else []
    total_char_count = sum(char_counts)
    avg_char_count_per_page = total_char_count / page_count if page_count else 0.0
    is_probably_scanned = (
        total_char_count < SCANNED_TOTAL_CHAR_THRESHOLD
        or avg_char_count_per_page < SCANNED_AVG_CHAR_PER_PAGE_THRESHOLD
    )
    low_confidence_reason = None
    if is_probably_scanned:
        low_confidence_reason = (
            f"Likely scanned: totalCharCount={total_char_count}, "
            f"avgCharCountPerPage={avg_char_count_per_page:.1f}"
        )

    return PdfExtractResult(
        text=text,
        page_texts=page_texts,
        page_count=page_count,
        total_char_count=total_char_count,
        avg_char_count_per_page=avg_char_count_per_page,
        is_probably_scanned=is_probably_scanned,
        low_confidence_reason=low_confidence_reason,
    )


def build_spans(source_sha256: str, page_index: int, text: str, chunk_size: int = 240, overlap: int = 40) -> list[dict]:
    spans: list[dict] = []
    step = chunk_size - overlap
    length = len(text)
    start = 0
    while start < length:
        end = min(length, start + chunk_size)
        slice_text = text[start:end]
        to_hash = f"v1|{source_sha256}|{page_index}|{start}|{end}|{slice_text}"
        span_hash = hashlib.sha256(to_hash.encode("utf-8")).hexdigest()
        spans.append({"start": start, "end": end, "sha256": span_hash})
        if end >= length:
            break
        start += step
    return spans
