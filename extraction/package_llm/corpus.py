from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from extraction.documents.classify import classify_document_type
from extraction.documents.extract_text import build_document_record, extract_document_text, pages_to_full_text
from extraction.documents.amendment import is_amendment_filename
from extraction.ocr.ocr_fallback import merge_native_and_ocr_pages, run_ocr_on_page_indices
from extraction.sources.pandas_parser import parse_excel  # noqa: F401 - re-export parity
from extraction.types import DocumentPage, Finding, SourceRecord

DOCUMENT_TYPE_HINTS = {
    "base_solicitation": "base solicitation",
    "amendment": "amendment",
    "amendment_continuation": "amendment continuation",
    "questions_and_answers": "qa",
    "pws_sow_soo": "pws",
    "section_l": "section_l",
    "section_m": "section_m",
    "pricing_workbook": "pricing",
    "attachment_exhibit": "attachment",
    "unknown": "unknown",
}


@dataclass
class PackageSourceText:
    source_id: str
    filename: str
    file_type: str
    document_type_hint: str
    pages: list[DocumentPage] = field(default_factory=list)
    sheets: list[dict[str, Any]] = field(default_factory=list)
    form_fields: list[dict[str, Any]] = field(default_factory=list)
    normalized_text_by_page: dict[int, str] = field(default_factory=dict)
    normalized_text_by_sheet: dict[str, str] = field(default_factory=dict)
    ocr_pages: list[int] = field(default_factory=list)
    char_count: int = 0


@dataclass
class PackageCorpus:
    sources: list[PackageSourceText]
    corpus_text: str
    findings: list[Finding] = field(default_factory=list)


def _normalize_for_match(text: str) -> str:
    from extraction.package_llm.validators.evidence_match import normalize_evidence_text

    return normalize_evidence_text(text)


def _document_type_hint(filename: str, text: str, *, is_amendment: bool) -> str:
    doc_type = classify_document_type(filename, text, is_amendment=is_amendment)
    return DOCUMENT_TYPE_HINTS.get(doc_type, "unknown")


def _extract_excel_sheets(source: SourceRecord) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    sheets: list[dict[str, Any]] = []
    try:
        import pandas as pd

        workbook = pd.read_excel(source.abs_path, sheet_name=None, header=None)
        for sheet_name, frame in workbook.items():
            cleaned = frame.dropna(how="all", axis=0).dropna(how="all", axis=1)
            cleaned = cleaned.ffill(axis=0).ffill(axis=1).fillna("")
            text = cleaned.to_csv(index=False, header=False).strip() if not cleaned.empty else ""
            sheets.append({"sheet_name": str(sheet_name), "text": text})
    except Exception as exc:
        findings.append(
            Finding(
                "warn",
                "EXCEL_EXTRACTION_FAILED",
                f"Excel extraction failed for {source.original_filename}",
                {"sourceId": source.sha256, "error": str(exc)},
            )
        )
    return sheets, findings


def extract_package_source_text(
    source: SourceRecord,
    *,
    apply_ocr: bool = True,
) -> tuple[PackageSourceText, list[Finding]]:
    findings: list[Finding] = []
    ext = source.ext.lower()
    is_amendment = is_amendment_filename(source.original_filename)
    package_source = PackageSourceText(
        source_id=source.sha256,
        filename=source.original_filename,
        file_type=ext,
        document_type_hint="unknown",
    )

    if ext in {"xlsx", "xls", "xlsm", "xlsb"}:
        sheets, sheet_findings = _extract_excel_sheets(source)
        findings.extend(sheet_findings)
        preview = "\n".join(item["text"][:2000] for item in sheets if item.get("text"))
        package_source.document_type_hint = _document_type_hint(source.original_filename, preview, is_amendment=is_amendment)
        package_source.sheets = sheets
        for item in sheets:
            name = str(item["sheet_name"])
            text = str(item.get("text") or "")
            package_source.normalized_text_by_sheet[name] = text
            package_source.char_count += len(text)
        return package_source, findings

    text_result = extract_document_text(source)
    findings.extend(text_result.findings)
    pages = list(text_result.pages)
    ocr_pages: list[int] = []

    if apply_ocr and ext == "pdf" and text_result.pdf_context is not None:
        sparse_pages = [page.page_index for page in pages if len(page.text.strip()) < 50]
        needs_ocr = bool(text_result.pdf_context.is_probably_scanned)
        page_indices: list[int] = []
        if needs_ocr:
            page_indices = list(range(min(len(pages), 12)))
        elif sparse_pages and text_result.pdf_context.total_char_count < 800:
            needs_ocr = True
            page_indices = sparse_pages[:12]
        if needs_ocr and page_indices:
            ocr_results, ocr_findings = run_ocr_on_page_indices(
                text_result.pdf_context.pdf_bytes or Path(source.abs_path).read_bytes(),
                page_indices,
                source_sha256=source.sha256,
                original_filename=source.original_filename,
            )
            findings.extend(ocr_findings)
            if ocr_results:
                pages, _ = merge_native_and_ocr_pages(pages, ocr_results)
                ocr_pages = [item.page_index for item in ocr_results if item.text.strip()]

    package_source.pages = pages
    package_source.ocr_pages = ocr_pages
    full_text = pages_to_full_text(pages)
    package_source.document_type_hint = _document_type_hint(source.original_filename, full_text, is_amendment=is_amendment)
    for page in pages:
        package_source.normalized_text_by_page[page.page_index] = page.text
        package_source.char_count += len(page.text)

    if ext == "pdf":
        from extraction.package_llm.forms.pdf_forms import (
            extract_pdf_form_fields,
            form_page_needs_render_fallback,
            is_federal_form_page,
            merge_form_fields_into_pages,
        )

        pdf_bytes = Path(source.abs_path).read_bytes()
        form_records, form_findings = extract_pdf_form_fields(
            pdf_bytes,
            source_id=source.sha256,
            filename=source.original_filename,
        )
        findings.extend(form_findings)
        package_source.form_fields = form_records
        merge_form_fields_into_pages(form_records, package_source.normalized_text_by_page)

        if apply_ocr and text_result.pdf_context is not None:
            render_pages: list[int] = []
            for page in pages[:3]:
                page_index = page.page_index
                native = package_source.normalized_text_by_page.get(page_index, page.text)
                page_form_records = [item for item in form_records if int(item.get("page_index") or 0) == page_index]
                if form_page_needs_render_fallback(native_page_text=native, form_records=page_form_records):
                    render_pages.append(page_index)
                elif is_federal_form_page(native) and page_index == 0 and not page_form_records:
                    render_pages.append(page_index)
            if render_pages:
                ocr_results, ocr_findings = run_ocr_on_page_indices(
                    pdf_bytes,
                    render_pages,
                    source_sha256=source.sha256,
                    original_filename=source.original_filename,
                )
                findings.extend(ocr_findings)
                if ocr_results:
                    for item in ocr_results:
                        if not item.text.strip():
                            continue
                        page_index = item.page_index
                        existing = package_source.normalized_text_by_page.get(page_index, "")
                        if item.text.strip() not in existing:
                            merged = (existing + "\n\n--- OCR FORM RECOVERY ---\n" + item.text.strip()).strip()
                            package_source.normalized_text_by_page[page_index] = merged
                            if page_index not in package_source.ocr_pages:
                                package_source.ocr_pages.append(page_index)

        package_source.char_count = sum(
            len(package_source.normalized_text_by_page.get(page.page_index, page.text))
            for page in package_source.pages
        )

    _ = build_document_record  # imported for parity with legacy metadata helpers if needed later
    return package_source, findings


def assemble_package_corpus(sources: list[PackageSourceText]) -> str:
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
            for item in source.sheets:
                sheet_name = str(item["sheet_name"])
                text = str(item.get("text") or "")
                lines.append(f"--- SHEET: {sheet_name} ---")
                lines.append(text)
                lines.append("")
        else:
            if source.form_fields:
                from extraction.package_llm.forms.pdf_forms import format_form_fields_corpus_block

                lines.append(format_form_fields_corpus_block(source.form_fields))
                lines.append("")
            for page in source.pages:
                marker = f"--- PAGE {page.page_index + 1} ---"
                if page.page_index in source.ocr_pages or page.text_provenance == "ocr":
                    marker = f"--- PAGE {page.page_index + 1} (OCR) ---"
                lines.append(marker)
                lines.append(page.text)
                lines.append("")
        lines.append("===== SOURCE END =====")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def build_package_corpus(sources: list[SourceRecord], *, apply_ocr: bool = True) -> PackageCorpus:
    findings: list[Finding] = []
    package_sources: list[PackageSourceText] = []
    active = [source for source in sources if not source.did_dedupe]
    for source in active:
        package_source, source_findings = extract_package_source_text(source, apply_ocr=apply_ocr)
        findings.extend(source_findings)
        package_sources.append(package_source)
    corpus_text = assemble_package_corpus(package_sources)
    return PackageCorpus(sources=package_sources, corpus_text=corpus_text, findings=findings)


def source_lookup(corpus: PackageCorpus) -> dict[str, PackageSourceText]:
    return {source.source_id: source for source in corpus.sources}


def page_text(source: PackageSourceText, page: int | None) -> str:
    if page is None:
        return ""
    index = page - 1 if page > 0 else page
    return source.normalized_text_by_page.get(index, "")


def sheet_text(source: PackageSourceText, sheet: str | None) -> str:
    if not sheet:
        return ""
    if sheet in source.normalized_text_by_sheet:
        return source.normalized_text_by_sheet[sheet]
    lowered = sheet.lower()
    for name, text in source.normalized_text_by_sheet.items():
        if name.lower() == lowered:
            return text
    return ""


def quote_in_source(source: PackageSourceText, *, page: int | None, sheet: str | None, quote: str) -> bool:
    from extraction.package_llm.validators.evidence_match import match_evidence_quote

    return match_evidence_quote(source, page=page, sheet=sheet, quote=quote, source_id=source.source_id).matched
