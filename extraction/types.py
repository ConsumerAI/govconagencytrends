from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

FindingLevel = Literal["info", "warn", "error"]


@dataclass
class Finding:
    level: FindingLevel
    code: str
    message: str
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"level": self.level, "code": self.code, "message": self.message}
        if self.details:
            out["details"] = self.details
        return out


@dataclass
class SourceRecord:
    key: str
    ext: str
    bytes: int
    sha256: str
    original_filename: str
    abs_path: str
    did_dedupe: bool = False

    def to_manifest_entry(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "ext": self.ext,
            "bytes": self.bytes,
            "sha256": self.sha256,
            "original_filename": self.original_filename,
            "absPath": self.abs_path,
            "didDedupe": self.did_dedupe,
        }


@dataclass
class ProcessSourcesV1:
    version: int
    run_id: str
    sources: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return {"version": self.version, "runId": self.run_id, "sources": self.sources}


@dataclass
class RunManifest:
    version: int
    run_id: str
    created_at: str
    sources: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "runId": self.run_id,
            "createdAt": self.created_at,
            "sources": self.sources,
        }


@dataclass
class CorpusSpan:
    start: int
    end: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "sha256": self.sha256}


@dataclass
class CorpusPage:
    source_sha256: str
    page_index: int
    text: str
    spans: list[CorpusSpan]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceSha256": self.source_sha256,
            "pageIndex": self.page_index,
            "text": self.text,
            "spans": [s.to_dict() for s in self.spans],
        }


@dataclass
class CorpusSource:
    source_key: str
    sha256: str
    filename: str | None
    mime: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceKey": self.source_key,
            "sha256": self.sha256,
            "filename": self.filename,
            "mime": self.mime,
        }


@dataclass
class CorpusV1:
    version: int
    run_id: str
    sources: list[CorpusSource]
    pages: list[CorpusPage]
    findings: list[Finding]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "runId": self.run_id,
            "sources": [s.to_dict() for s in self.sources],
            "pages": [p.to_dict() for p in self.pages],
            "findings": [f.to_dict() for f in self.findings],
        }


@dataclass
class IngestResult:
    run_id: str
    sources: list[SourceRecord]
    findings: list[Finding] = field(default_factory=list)


@dataclass
class DocumentPage:
    page_index: int
    text: str
    char_count: int
    text_provenance: str = "native"
    ocr_engine: str | None = None
    ocr_confidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        payload = {"pageIndex": self.page_index, "text": self.text, "charCount": self.char_count}
        if self.text_provenance != "native":
            payload["textProvenance"] = self.text_provenance
        if self.ocr_engine:
            payload["ocrEngine"] = self.ocr_engine
        if self.ocr_confidence is not None:
            payload["ocrConfidence"] = self.ocr_confidence
        return payload


@dataclass
class PdfDocumentContext:
    pdf_bytes: bytes | None = None
    page_char_counts: list[int] = field(default_factory=list)
    total_char_count: int = 0
    is_probably_scanned: bool = False
    low_confidence_reason: str | None = None


@dataclass
class DocumentTextResult:
    text: str
    pages: list[DocumentPage]
    findings: list[Finding]
    pdf_context: PdfDocumentContext | None = None


@dataclass
class DocumentRecord:
    source_id: str
    sha256: str
    original_filename: str
    ext: str
    source_key: str
    document_type: str
    document_class: str
    is_amendment: bool
    amendment_raw: str | None
    amendment_order: str | None
    solicitation_number: str | None
    char_count: int
    page_count: int
    amendment_order_source: str | None = None
    amendment_order_confidence: str | None = None
    revised_attachments: list[str] = field(default_factory=list)
    sf30_evidence: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sourceId": self.source_id,
            "sha256": self.sha256,
            "originalFilename": self.original_filename,
            "ext": self.ext,
            "sourceKey": self.source_key,
            "documentType": self.document_type,
            "documentClass": self.document_class,
            "isAmendment": self.is_amendment,
            "amendmentRaw": self.amendment_raw,
            "amendmentOrder": self.amendment_order,
            "solicitationNumber": self.solicitation_number,
            "amendmentOrderSource": self.amendment_order_source,
            "amendmentOrderConfidence": self.amendment_order_confidence,
            "revisedAttachments": self.revised_attachments,
            "sf30Evidence": self.sf30_evidence,
            "charCount": self.char_count,
            "pageCount": self.page_count,
            "warnings": self.warnings,
        }


@dataclass
class DocsetResult:
    documents: list[DocumentRecord]
    merged_signals: list[dict[str, Any]]
    findings: list[Finding] = field(default_factory=list)
    solicitation_set_detected: bool = False
    base_solicitation_filename: str | None = None
    amendments_in_order: list[str] = field(default_factory=list)
    superseded_candidate_count: int = 0
    unresolved_amendment_conflicts: int = 0
    per_document_signal_counts: dict[str, int] = field(default_factory=dict)
    classified_counts: dict[str, int] = field(default_factory=dict)
    unclassified_count: int = 0
    ocr_documents_requiring: int = 0
    ocr_pages_processed: int = 0
    ocr_documents_rerun: int = 0
    ocr_failures: int = 0
    controlling_documents_unreadable: int = 0
    ocr_diagnostics: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class PipelineResult:
    run_id: str
    manifest_path: str
    diagnostics_path: str
    process_sources_path: str | None = None
    corpus_path: str | None = None
    full_text_path: str | None = None
    signals_path: str | None = None
    signal_count: int = 0
    resolved_signals_path: str | None = None
    resolved_signal_count: int = 0
    alternates_count: int = 0
    derived_signal_count: int = 0
    model_used: str | None = None
    elapsed_sec: float | None = None
    corpus_char_count: int | None = None
    findings: list[Finding] = field(default_factory=list)
    # Milestone 5 docset
    solicitation_set_detected: bool = False
    base_solicitation_filename: str | None = None
    amendments_detected: list[str] = field(default_factory=list)
    document_count: int = 0
    superseded_candidate_count: int = 0
    unresolved_amendment_conflicts: int = 0
    docset_manifest_path: str | None = None
    ocr_documents_requiring: int = 0
    ocr_pages_processed: int = 0
    ocr_documents_rerun: int = 0
    ocr_failures: int = 0
    controlling_documents_unreadable: int = 0
    ocr_diagnostics: list[dict[str, Any]] = field(default_factory=list)
