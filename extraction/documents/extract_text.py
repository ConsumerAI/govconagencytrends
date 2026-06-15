from __future__ import annotations

from pathlib import Path

from extraction.config import source_storage_key
from extraction.corpus.build import normalize_corpus_text
from extraction.corpus.pdf_extract import extract_pdf_bytes
from extraction.documents.amendment import detect_solicitation_number, is_amendment_filename
from extraction.documents.classify import classify_document_class, classify_document_type
from extraction.sf30.parse import parse_sf30_metadata, resolve_amendment_identity
from extraction.sources.supplemental import extract_supplemental_text
from extraction.types import DocumentPage, DocumentRecord, DocumentTextResult, Finding, PdfDocumentContext, SourceRecord


def pages_to_full_text(pages: list[DocumentPage]) -> str:
    return "\n\n".join(page.text for page in pages if page.text.strip()).strip()


def extract_document_text(source: SourceRecord) -> DocumentTextResult:
    """Native text extraction only. OCR feedback pass runs later in the docset pipeline."""
    findings: list[Finding] = []
    path = Path(source.abs_path)
    ext = source.ext.lower()
    pages: list[DocumentPage] = []
    pdf_context: PdfDocumentContext | None = None

    if ext == "pdf":
        pdf_bytes = path.read_bytes()
        try:
            pdf_result = extract_pdf_bytes(pdf_bytes)
        except Exception as exc:
            findings.append(
                Finding("error", "DOCUMENT_PDF_PARSE_FAILED", f"PDF parse failed: {source.original_filename}", {"error": str(exc)})
            )
            return DocumentTextResult(text="", pages=[], findings=findings)

        pdf_context = PdfDocumentContext(
            pdf_bytes=pdf_bytes,
            page_char_counts=[len(text) for text in pdf_result.page_texts],
            total_char_count=pdf_result.total_char_count,
            is_probably_scanned=pdf_result.is_probably_scanned,
            low_confidence_reason=pdf_result.low_confidence_reason,
        )

        for page_index, page_text in enumerate(pdf_result.page_texts):
            normalized = normalize_corpus_text(page_text)
            pages.append(
                DocumentPage(
                    page_index=page_index,
                    text=normalized,
                    char_count=len(normalized),
                    text_provenance="native",
                )
            )

        if pdf_result.is_probably_scanned:
            findings.append(
                Finding(
                    "info",
                    "DOCUMENT_LOW_TEXT_LAYER",
                    pdf_result.low_confidence_reason or "PDF appears scanned; OCR feedback may run after initial extraction",
                    {
                        "sourceId": source.sha256,
                        "filename": source.original_filename,
                        "totalCharCount": pdf_result.total_char_count,
                    },
                )
            )

        return DocumentTextResult(
            text=pages_to_full_text(pages),
            pages=pages,
            findings=findings,
            pdf_context=pdf_context,
        )

    supplemental, supplemental_findings = extract_supplemental_text(source)
    findings.extend(supplemental_findings)
    normalized = normalize_corpus_text(supplemental)
    if normalized.strip():
        pages.append(
            DocumentPage(page_index=0, text=normalized, char_count=len(normalized), text_provenance="native")
        )
    return DocumentTextResult(text=normalized.strip(), pages=pages, findings=findings, pdf_context=None)


def build_document_record(
    run_id: str,
    source: SourceRecord,
    *,
    text: str,
    pages: list[DocumentPage],
    findings: list[Finding],
) -> DocumentRecord:
    warnings = [finding.message for finding in findings if finding.level in {"warn", "error"}]

    sf30 = parse_sf30_metadata(text, filename=source.original_filename)
    identity = resolve_amendment_identity(text=text, filename=source.original_filename, sf30=sf30)

    amendment_raw = identity.get("amendmentRaw")
    amendment_order = identity.get("amendmentOrder")
    amendment_order_source = identity.get("amendmentOrderSource")
    amendment_order_confidence = identity.get("amendmentOrderConfidence")
    sf30_evidence = identity.get("sf30Evidence") or []

    is_amendment = bool(
        amendment_order
        or amendment_raw
        or sf30.is_sf30
        or is_amendment_filename(source.original_filename)
    )

    document_type = classify_document_type(source.original_filename, text, is_amendment=is_amendment)
    document_class = classify_document_class(source.original_filename, text)
    solicitation_number = identity.get("solicitationNumber") or detect_solicitation_number(text, source.original_filename)

    return DocumentRecord(
        source_id=source.sha256,
        sha256=source.sha256,
        original_filename=source.original_filename,
        ext=source.ext,
        source_key=source_storage_key(run_id, source.sha256, source.ext),
        document_type=document_type,
        document_class=document_class,
        is_amendment=is_amendment,
        amendment_raw=amendment_raw,
        amendment_order=amendment_order,
        solicitation_number=solicitation_number,
        amendment_order_source=amendment_order_source,
        amendment_order_confidence=amendment_order_confidence,
        revised_attachments=list(sf30.revised_attachments),
        sf30_evidence=list(sf30_evidence) if isinstance(sf30_evidence, list) else [],
        char_count=len(text),
        page_count=len(pages) if pages else (1 if text.strip() else 0),
        warnings=warnings,
    )


def page_boundaries_for_document(pages: list[DocumentPage]) -> list[dict]:
    boundaries: list[dict] = []
    offset = 0
    for page in pages:
        boundaries.append(
            {
                "pageIndex": page.page_index,
                "startChar": offset,
                "endChar": offset + page.char_count,
                "textProvenance": page.text_provenance,
            }
        )
        offset += page.char_count + 2
    return boundaries
