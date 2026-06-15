from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from extraction.documents.extract_text import build_document_record, pages_to_full_text
from extraction.ocr.ocr_fallback import (
    build_ocr_artifact,
    document_is_sparse,
    merge_native_and_ocr_pages,
    ocr_environment_ready,
    pages_requiring_ocr,
    run_ocr_on_page_indices,
    should_run_ocr_fallback,
)
from extraction.sf30.parse import parse_sf30_metadata
from extraction.types import DocumentPage, DocumentRecord, Finding, PdfDocumentContext, SourceRecord

CRITICAL_SIGNAL_IDS = (
    "rfp_due_date_v1",
    "rfp_submission_instructions_v1",
    "rfp_solicitation_id_v1",
)


@dataclass
class OcrFeedbackResult:
    rerun: bool = False
    text: str = ""
    pages: list[DocumentPage] = field(default_factory=list)
    record: DocumentRecord | None = None
    findings: list[Finding] = field(default_factory=list)
    ocr_artifact: dict[str, Any] | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


def _signal_populated(signals: list[dict[str, Any]], signal_id: str) -> bool:
    for signal in signals:
        if str(signal.get("id") or "") != signal_id:
            continue
        value = signal.get("value")
        return value is not None and str(value).strip() != "" and not signal.get("_deleted")
    return False


def _is_controlling_document(record: DocumentRecord) -> bool:
    if record.document_type == "base_solicitation":
        return True
    if record.is_amendment and (record.amendment_order_source == "sf30" or record.document_type == "amendment"):
        return True
    return record.document_type in {"amendment", "sf30"}


def evaluate_ocr_need(
    *,
    pages: list[DocumentPage],
    pdf_context: PdfDocumentContext | None,
    signals: list[dict[str, Any]],
    record: DocumentRecord,
) -> tuple[bool, list[int], str]:
    if not pdf_context or not pdf_context.pdf_bytes:
        return False, [], "not_pdf"

    sparse_page_indices = pages_requiring_ocr(pages, page_char_counts=pdf_context.page_char_counts)
    doc_sparse = document_is_sparse(
        total_char_count=pdf_context.total_char_count,
        is_probably_scanned=pdf_context.is_probably_scanned,
    )
    has_due = _signal_populated(signals, "rfp_due_date_v1")
    has_submission = _signal_populated(signals, "rfp_submission_instructions_v1")

    if sparse_page_indices:
        if doc_sparse or not has_due or (record.is_amendment and not has_due):
            return True, sparse_page_indices, "sparse_pages"
        if not has_submission and any(idx > 0 for idx in sparse_page_indices):
            return True, sparse_page_indices, "sparse_pages_with_missing_submission"

    if should_run_ocr_fallback(
        has_due_date=has_due,
        has_submission_instructions=has_submission,
        corpus_char_count=pdf_context.total_char_count,
        no_text_layer=pdf_context.is_probably_scanned and pdf_context.total_char_count == 0,
    ):
        indices = sparse_page_indices or list(range(len(pdf_context.page_char_counts or pages or [])))
        return True, indices, "document_sparse_missing_critical_signals"

    if record.is_amendment and doc_sparse and not _signal_populated(signals, "rfp_solicitation_id_v1"):
        return True, sparse_page_indices or [0], "amendment_sparse"

    return False, [], "not_needed"


def run_ocr_feedback_pass(
    *,
    run_id: str,
    source: SourceRecord,
    record: DocumentRecord,
    text: str,
    pages: list[DocumentPage],
    signals: list[dict[str, Any]],
    pdf_context: PdfDocumentContext | None,
    text_findings: list[Finding],
    feedback_already_applied: bool = False,
) -> OcrFeedbackResult:
    result = OcrFeedbackResult(text=text, pages=pages, record=record, findings=list(text_findings))
    if feedback_already_applied:
        result.diagnostics = {"skipped": "ocr_feedback_already_applied"}
        return result

    should_run, page_indices, reason = evaluate_ocr_need(
        pages=pages,
        pdf_context=pdf_context,
        signals=signals,
        record=record,
    )
    controlling = _is_controlling_document(record)
    result.diagnostics = {
        "sourceId": record.source_id,
        "filename": record.original_filename,
        "ocrRequired": should_run,
        "ocrReason": reason,
        "pagesRequiringOcr": page_indices,
        "controllingDocument": controlling,
        "rerunApplied": False,
    }

    if not should_run:
        return result

    if not ocr_environment_ready():
        result.findings.append(
            Finding(
                "warn" if not controlling else "error",
                "OCR_FALLBACK_NOT_AVAILABLE",
                f"OCR required for {record.original_filename} but OCR dependencies are unavailable",
                {
                    "sourceId": record.source_id,
                    "filename": record.original_filename,
                    "pagesRequiringOcr": page_indices,
                    "controllingDocument": controlling,
                },
            )
        )
        if controlling:
            result.findings.append(
                Finding(
                    "error",
                    "CONTROLLING_DOCUMENT_UNREADABLE",
                    f"Controlling document may be unreadable without OCR: {record.original_filename}",
                    {"sourceId": record.source_id, "filename": record.original_filename},
                )
            )
        return result

    assert pdf_context and pdf_context.pdf_bytes
    ocr_results, ocr_findings = run_ocr_on_page_indices(
        pdf_context.pdf_bytes,
        page_indices,
        source_sha256=record.source_id,
        original_filename=record.original_filename,
    )
    result.findings.extend(ocr_findings)

    if not ocr_results or not any(item.text.strip() for item in ocr_results):
        if controlling:
            result.findings.append(
                Finding(
                    "error",
                    "CONTROLLING_DOCUMENT_UNREADABLE",
                    f"Controlling document OCR produced no usable text: {record.original_filename}",
                    {"sourceId": record.source_id, "filename": record.original_filename},
                )
            )
        return result

    merged_pages, evidence_pages = merge_native_and_ocr_pages(pages, ocr_results)
    merged_text = pages_to_full_text(merged_pages)
    merged_findings = list(text_findings)
    updated_record = build_document_record(
        run_id,
        source,
        text=merged_text,
        pages=merged_pages,
        findings=merged_findings,
    )

    artifact = build_ocr_artifact(
        source_sha256=record.source_id,
        original_filename=record.original_filename,
        merged_pages=merged_pages,
        evidence_pages=evidence_pages,
        rerun_applied=True,
    )

    sf30 = parse_sf30_metadata(merged_text, filename=record.original_filename)
    if controlling and sf30.is_sf30 and not updated_record.amendment_order and record.is_amendment:
        result.findings.append(
            Finding(
                "warning",
                "CONTROLLING_SF30_PARTIAL",
                f"SF30 detected after OCR but amendment metadata remains incomplete: {record.original_filename}",
                {"sourceId": record.source_id},
            )
        )

    result.rerun = True
    result.text = merged_text
    result.pages = merged_pages
    result.record = updated_record
    result.ocr_artifact = artifact
    result.diagnostics.update(
        {
            "rerunApplied": True,
            "ocrPagesProcessed": len(ocr_results),
            "ocrPageIndices": [item.page_index for item in ocr_results],
            "mergedOcrPages": sum(1 for page in merged_pages if page.text_provenance == "ocr"),
        }
    )
    result.findings.append(
        Finding(
            "info",
            "OCR_FEEDBACK_RERUN",
            f"Extraction rerun after OCR for {record.original_filename}",
            result.diagnostics,
        )
    )
    return result
