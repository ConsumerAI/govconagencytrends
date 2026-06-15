from __future__ import annotations

from typing import Any

from extraction.artifacts.build import build_document_artifacts
from extraction.config import docset_manifest_path, docset_diagnostics_path, merged_docset_signals_path, resolved_signals_json_path
from extraction.docset.solicitation_lock import build_package_solicitation_lock, filter_documents_for_merge
from extraction.documents.amendment import is_amendment_filename
from extraction.documents.corpus_pages import document_pages_to_corpus_pages
from extraction.documents.persist import persist_document_artifacts
from extraction.documents.extract_text import build_document_record, extract_document_text
from extraction.docset.merge import merge_llm_and_docset_signals, merge_solicitation_set_signals
from extraction.llm.openai_extract import extract_signals_from_corpus
from extraction.ocr.feedback import run_ocr_feedback_pass
from extraction.persist import write_json
from extraction.signals.extract_signals_v1 import extract_signals_v1
from extraction.types import DocsetResult, DocumentPage, DocumentRecord, Finding, SourceRecord


def _tag_signals(
    signals: list[dict[str, Any]],
    *,
    source_id: str,
    file_name: str,
    document_type: str,
    amendment_order: str | None,
) -> list[dict[str, Any]]:
    tagged: list[dict[str, Any]] = []
    for signal in signals:
        item = dict(signal)
        for ev in item.get("evidence") or []:
            if isinstance(ev, dict) and not ev.get("sourceId"):
                ev["sourceId"] = file_name
        ev1 = item.get("evidence_v1")
        if isinstance(ev1, dict) and not ev1.get("source"):
            ev1["source"] = file_name
        item["_docset"] = {
            "source_id": source_id,
            "filename": file_name,
            "document_type": document_type,
            "amendment_order": amendment_order,
        }
        tagged.append(item)
    return tagged


def _merge_signals_for_document(
    deterministic: list[dict[str, Any]],
    llm_signals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    for signal in deterministic:
        signal_id = str(signal.get("id") or "").strip()
        if signal_id:
            by_id[signal_id] = signal
    for signal in llm_signals:
        signal_id = str(signal.get("id") or "").strip()
        if not signal_id:
            continue
        existing = by_id.get(signal_id)
        value = signal.get("value")
        populated = value is not None and str(value).strip() != ""
        if populated or existing is None:
            by_id[signal_id] = signal
    return sorted(by_id.values(), key=lambda item: str(item.get("id") or ""))


def extract_document_signals(
    run_id: str,
    record: DocumentRecord,
    text: str,
    pages: list[DocumentPage],
    *,
    use_llm: bool = True,
) -> tuple[list[dict[str, Any]], list[Finding], str | None, dict[str, Any]]:
    findings: list[Finding] = []
    if not pages and text.strip():
        pages = [DocumentPage(page_index=0, text=text, char_count=len(text))]

    corpus_pages = document_pages_to_corpus_pages(record.source_id, pages)
    structure_artifacts = build_document_artifacts(
        run_id,
        corpus_pages,
        filename=record.original_filename,
        amendment_number=record.amendment_order,
    )

    deterministic, det_findings = extract_signals_v1(
        run_id=run_id,
        pages=corpus_pages,
        structure=structure_artifacts["structure"],
        sections=structure_artifacts["sections"],
        clauses=structure_artifacts["clauses"],
        section_l_fulltext=structure_artifacts["sectionLFulltext"],
        section_m_fulltext=structure_artifacts["sectionMFulltext"],
        source_filename=record.original_filename,
    )
    findings.extend(det_findings)
    deterministic = _tag_signals(
        deterministic,
        source_id=record.source_id,
        file_name=record.original_filename,
        document_type=record.document_type,
        amendment_order=record.amendment_order,
    )

    llm_signals: list[dict[str, Any]] = []
    model_used: str | None = None
    if use_llm and text.strip():
        llm_signals, llm_findings, model_used = extract_signals_from_corpus(
            run_id,
            text,
            fallback_source_file=record.original_filename,
            base_signals=[],
        )
        findings.extend(llm_findings)
        llm_signals = _tag_signals(
            llm_signals,
            source_id=record.source_id,
            file_name=record.original_filename,
            document_type=record.document_type,
            amendment_order=record.amendment_order,
        )

    merged = _merge_signals_for_document(deterministic, llm_signals)
    return merged, findings, model_used, structure_artifacts


def run_docset_phase(
    run_id: str,
    sources: list[SourceRecord],
    *,
    use_llm: bool = True,
    full_corpus_llm_signals: list[dict[str, Any]] | None = None,
    base_signals: list[dict[str, Any]] | None = None,
) -> DocsetResult:
    findings: list[Finding] = []
    active_sources = [source for source in sources if not source.did_dedupe]
    documents: list[DocumentRecord] = []
    doc_entries: list[dict[str, Any]] = []
    per_document_signal_counts: dict[str, int] = {}
    classified_counts: dict[str, int] = {}
    model_used: str | None = None
    ocr_diagnostics: list[dict[str, Any]] = []
    ocr_documents_requiring = 0
    ocr_pages_processed = 0
    ocr_documents_rerun = 0
    ocr_failures = 0
    controlling_documents_unreadable = 0

    for source in active_sources:
        text_result = extract_document_text(source)
        text = text_result.text
        pages = text_result.pages
        text_findings = list(text_result.findings)
        findings.extend(text_findings)
        record = build_document_record(run_id, source, text=text, pages=pages, findings=text_findings)
        doc_signals, doc_findings, doc_model, structure_artifacts = extract_document_signals(
            run_id,
            record,
            text,
            pages,
            use_llm=use_llm,
        )
        findings.extend(doc_findings)
        ocr_artifact: dict[str, Any] | None = None

        feedback = run_ocr_feedback_pass(
            run_id=run_id,
            source=source,
            record=record,
            text=text,
            pages=pages,
            signals=doc_signals,
            pdf_context=text_result.pdf_context,
            text_findings=text_findings,
        )
        findings.extend(feedback.findings)
        ocr_diagnostics.append(feedback.diagnostics)
        if feedback.diagnostics.get("ocrRequired"):
            ocr_documents_requiring += 1
        if feedback.diagnostics.get("rerunApplied"):
            ocr_documents_rerun += 1
            ocr_pages_processed += int(feedback.diagnostics.get("ocrPagesProcessed") or 0)
        if any(item.code == "OCR_FALLBACK_EMPTY" for item in feedback.findings):
            ocr_failures += 1
        if any(item.code == "CONTROLLING_DOCUMENT_UNREADABLE" for item in feedback.findings):
            controlling_documents_unreadable += 1

        if feedback.rerun and feedback.record is not None:
            record = feedback.record
            text = feedback.text
            pages = feedback.pages
            ocr_artifact = feedback.ocr_artifact
            doc_signals, doc_findings, doc_model, structure_artifacts = extract_document_signals(
                run_id,
                record,
                text,
                pages,
                use_llm=use_llm,
            )
            findings.extend(doc_findings)

        if doc_model:
            model_used = doc_model

        persist_document_artifacts(
            run_id,
            record,
            text=text,
            pages=pages,
            signals=doc_signals,
            findings=doc_findings,
            structure_artifacts=structure_artifacts,
            ocr_artifact=ocr_artifact,
        )

        documents.append(record)
        classified_counts[record.document_type] = classified_counts.get(record.document_type, 0) + 1
        per_document_signal_counts[record.source_id] = len(doc_signals)
        doc_entries.append(
            {
                "docKey": record.source_id,
                "sourceId": record.source_id,
                "fileName": record.original_filename,
                "originalFilename": record.original_filename,
                "documentType": record.document_type,
                "documentClass": record.document_class,
                "isAmendment": record.is_amendment,
                "amendmentOrder": record.amendment_order,
                "amendmentOrderSource": record.amendment_order_source,
                "amendmentOrderConfidence": record.amendment_order_confidence,
                "amendmentRaw": record.amendment_raw,
                "solicitationNumber": record.solicitation_number,
                "revisedAttachments": record.revised_attachments,
                "sf30Evidence": record.sf30_evidence,
                "signals": doc_signals,
                "findings": [item.to_dict() for item in doc_findings],
            }
        )

    unclassified_count = classified_counts.get("unknown", 0)
    solicitation_set_detected = len(active_sources) >= 2

    merged_signals: list[dict[str, Any]] = []
    superseded_count = 0
    unresolved_conflicts = 0
    package_lock = None

    if solicitation_set_detected:
        package_lock = build_package_solicitation_lock(doc_entries)
        active_entries, excluded = filter_documents_for_merge(doc_entries, package_lock)
        if package_lock.confidence == "ambiguous":
            findings.append(
                Finding(
                    "warning",
                    "SOLICITATION_IDENTITY_AMBIGUOUS",
                    "Solicitation identity could not be locked confidently",
                    {"conflicts": package_lock.conflicts},
                )
            )
        if excluded:
            findings.append(
                Finding(
                    "info",
                    "SOLICITATION_SOURCE_EXCLUDED",
                    f"Excluded {len(excluded)} document(s) from merge due to solicitation mismatch",
                    {"excluded": excluded, "lockedId": package_lock.locked_solicitation_id},
                )
            )
        merged_signals, merge_findings, superseded_count, unresolved_conflicts = merge_solicitation_set_signals(
            active_entries
        )
        findings.extend(merge_findings)
    elif doc_entries:
        merged_signals = doc_entries[0]["signals"]
        for signal in merged_signals:
            if "docset_provenance" not in signal:
                signal["docset_provenance"] = {
                    "controlling_source_id": doc_entries[0]["sourceId"],
                    "controlling_filename": doc_entries[0]["fileName"],
                    "controlling_document_type": doc_entries[0]["documentType"],
                    "controlling_amendment": doc_entries[0]["amendmentOrder"],
                    "supersedes": [],
                }

    if full_corpus_llm_signals is not None:
        merged_signals = merge_llm_and_docset_signals(
            base_signals or [],
            full_corpus_llm_signals,
            merged_signals,
        )

    manifest = {
        "version": "docset.manifest.v1",
        "runId": run_id,
        "solicitationLock": {
            "lockedSolicitationId": package_lock.locked_solicitation_id,
            "confidence": package_lock.confidence,
            "lockingEvidence": package_lock.locking_evidence,
            "associatedDocuments": package_lock.associated_documents,
            "excludedDocuments": package_lock.excluded_documents,
            "conflicts": package_lock.conflicts,
        }
        if package_lock
        else None,
        "documents": [
            {
                "docKey": doc.source_id,
                "sourceId": doc.source_id,
                "fileName": doc.original_filename,
                "documentType": doc.document_type,
                "documentClass": doc.document_class,
                "isAmendment": doc.is_amendment,
                "amendmentOrder": doc.amendment_order,
                "amendmentOrderSource": doc.amendment_order_source,
                "amendmentOrderConfidence": doc.amendment_order_confidence,
                "amendmentRaw": doc.amendment_raw,
                "solicitationNumber": doc.solicitation_number,
                "revisedAttachments": doc.revised_attachments,
                "charCount": doc.char_count,
                "pageCount": doc.page_count,
            }
            for doc in documents
        ],
    }
    manifest_file = docset_manifest_path(run_id)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    write_json(manifest_file, manifest)
    write_json(merged_docset_signals_path(run_id), merged_signals)

    amendments_in_order = [
        doc.amendment_order
        for doc in documents
        if doc.is_amendment and doc.amendment_order
    ]
    base_filename = next(
        (doc.original_filename for doc in documents if not doc.is_amendment and doc.document_type == "base_solicitation"),
        next((doc.original_filename for doc in documents if not doc.is_amendment), None),
    )

    write_json(
        docset_diagnostics_path(run_id),
        {
            "runId": run_id,
            "stage": "milestone_5_1_docset",
            "sourceDocumentCount": len(active_sources),
            "classifiedDocumentCounts": classified_counts,
            "unclassifiedSources": unclassified_count,
            "detectedAmendmentsInOrder": amendments_in_order,
            "perDocumentSignalCounts": per_document_signal_counts,
            "mergedSignalCount": len(merged_signals),
            "supersededCandidateCount": superseded_count,
            "unresolvedAmendmentConflicts": unresolved_conflicts,
            "modelUsed": model_used,
            "ocrDocumentsRequiring": ocr_documents_requiring,
            "ocrPagesProcessed": ocr_pages_processed,
            "ocrDocumentsRerun": ocr_documents_rerun,
            "ocrFailures": ocr_failures,
            "controllingDocumentsUnreadable": controlling_documents_unreadable,
            "ocrDiagnostics": ocr_diagnostics,
        },
    )

    return DocsetResult(
        documents=documents,
        merged_signals=merged_signals,
        findings=findings,
        solicitation_set_detected=solicitation_set_detected,
        base_solicitation_filename=base_filename,
        amendments_in_order=amendments_in_order,
        superseded_candidate_count=superseded_count,
        unresolved_amendment_conflicts=unresolved_conflicts,
        per_document_signal_counts=per_document_signal_counts,
        classified_counts=classified_counts,
        unclassified_count=unclassified_count,
        ocr_documents_requiring=ocr_documents_requiring,
        ocr_pages_processed=ocr_pages_processed,
        ocr_documents_rerun=ocr_documents_rerun,
        ocr_failures=ocr_failures,
        controlling_documents_unreadable=controlling_documents_unreadable,
        ocr_diagnostics=ocr_diagnostics,
    )
