from __future__ import annotations

from extraction.corpus.pdf_extract import build_spans
from extraction.types import CorpusPage, CorpusSpan, DocumentPage


def document_pages_to_corpus_pages(source_sha256: str, pages: list[DocumentPage]) -> list[CorpusPage]:
    corpus_pages: list[CorpusPage] = []
    for page in pages:
        span_dicts = build_spans(source_sha256, page.page_index, page.text)
        spans = [CorpusSpan(start=item["start"], end=item["end"], sha256=item["sha256"]) for item in span_dicts]
        if not spans and page.text:
            from hashlib import sha256 as _sha256

            end = len(page.text)
            span_hash = _sha256(f"v1|{source_sha256}|{page.page_index}|0|{end}|{page.text}".encode()).hexdigest()
            spans = [CorpusSpan(start=0, end=end, sha256=span_hash)]
        corpus_pages.append(
            CorpusPage(
                source_sha256=source_sha256,
                page_index=page.page_index,
                text=page.text,
                spans=spans,
            )
        )
    return corpus_pages
