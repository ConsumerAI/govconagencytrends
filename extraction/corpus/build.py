from __future__ import annotations

import re
from pathlib import Path

from extraction.config import source_storage_key
from extraction.corpus.pdf_extract import build_spans, extract_pdf_bytes
from extraction.sources.supplemental import build_supplemental_text_for_sources
from extraction.types import CorpusPage, CorpusSource, CorpusSpan, CorpusV1, Finding, SourceRecord


def normalize_corpus_text(raw: str) -> str:
    normalized_newlines = raw.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized_newlines.split("\n")
    out: list[str] = []
    for line in lines:
        trimmed_trailing = re.sub(r"[ \t]+$", "", line)
        collapsed = re.sub(r"[ \t\f\v]+", " ", trimmed_trailing)
        out.append(collapsed)
    return "\n".join(out)


def _is_effectively_empty(text: str) -> bool:
    return len(re.sub(r"[ \t\n\r\f\v]+", "", text)) == 0


def build_corpus_v1(run_id: str, sources: list[SourceRecord]) -> CorpusV1:
    findings: list[Finding] = []
    corpus_sources: list[CorpusSource] = []
    pages: list[CorpusPage] = []

    for source in sources:
        if source.did_dedupe:
            continue
        path = Path(source.abs_path)
        if not path.exists():
            findings.append(
                Finding(
                    "error",
                    "CORPUS_SOURCE_READ_FAILED",
                    "Failed to read source bytes",
                    {"storageKey": source.key},
                )
            )
            continue

        corpus_sources.append(
            CorpusSource(
                source_key=source_storage_key(run_id, source.sha256, source.ext),
                sha256=source.sha256,
                filename=source.original_filename,
                mime=None,
            )
        )

        if source.ext != "pdf":
            continue

        try:
            pdf_result = extract_pdf_bytes(path.read_bytes())
        except Exception as exc:
            findings.append(
                Finding(
                    "error",
                    "PDF_PARSE_FAILED",
                    "PDF extraction failed",
                    {"sourceSha256": source.sha256, "error": str(exc)},
                )
            )
            continue

        if pdf_result.is_probably_scanned:
            findings.append(
                Finding(
                    "warn",
                    "PDF_NEEDS_OCR",
                    pdf_result.low_confidence_reason or "PDF appears scanned / text-sparse",
                    {
                        "sourceSha256": source.sha256,
                        "totalCharCount": pdf_result.total_char_count,
                        "avgCharCountPerPage": pdf_result.avg_char_count_per_page,
                    },
                )
            )

        normalized_pages = [normalize_corpus_text(page_text) for page_text in pdf_result.page_texts]
        if normalized_pages and all(_is_effectively_empty(page) for page in normalized_pages):
            findings.append(
                Finding(
                    "warn",
                    "PDF_TEXT_EMPTY",
                    "PDF produced empty text",
                    {"sourceSha256": source.sha256},
                )
            )

        for page_index, page_text in enumerate(normalized_pages):
            span_dicts = build_spans(source.sha256, page_index, page_text)
            spans = [CorpusSpan(start=s["start"], end=s["end"], sha256=s["sha256"]) for s in span_dicts]
            pages.append(
                CorpusPage(
                    source_sha256=source.sha256,
                    page_index=page_index,
                    text=page_text,
                    spans=spans,
                )
            )

    corpus_sources.sort(key=lambda item: item.sha256)
    pages.sort(key=lambda item: (item.source_sha256, item.page_index))

    return CorpusV1(
        version=1,
        run_id=run_id,
        sources=corpus_sources,
        pages=pages,
        findings=findings,
    )


def build_full_corpus_text(run_id: str, sources: list[SourceRecord], corpus: CorpusV1) -> tuple[str, list[Finding]]:
    findings: list[Finding] = []
    chunks: list[str] = []

    for page in corpus.pages:
        if page.text.strip():
            chunks.append(page.text)

    supplemental, supplemental_findings = build_supplemental_text_for_sources(sources)
    findings.extend(supplemental_findings)
    if supplemental.strip():
        chunks.append(supplemental)

    full_text = "\n\n".join(chunks).strip()
    if not full_text:
        findings.append(
            Finding(
                "warn",
                "CORPUS_TEXT_EMPTY",
                "No extractable text was produced from run sources",
                {"runId": run_id},
            )
        )
    return full_text, findings
