from __future__ import annotations

from pathlib import Path
from typing import Any

from extraction.config import document_dir
from extraction.documents.extract_text import page_boundaries_for_document
from extraction.persist import write_json, write_text
from extraction.types import DocumentPage, DocumentRecord, Finding


def persist_document_artifacts(
    run_id: str,
    record: DocumentRecord,
    *,
    text: str,
    pages: list[DocumentPage],
    signals: list[dict[str, Any]],
    findings: list[Finding],
    structure_artifacts: dict[str, Any] | None = None,
    ocr_artifact: dict[str, Any] | None = None,
) -> Path:
    root = document_dir(run_id, record.source_id)
    root.mkdir(parents=True, exist_ok=True)

    document_payload = {
        **record.to_dict(),
        "pageBoundaries": page_boundaries_for_document(pages),
    }
    write_json(root / "document.json", document_payload)
    write_text(root / "text.txt", text + ("\n" if text else ""))
    write_json(root / "signals.json", {"signalsJson": signals, "findings": [item.to_dict() for item in findings]})
    write_json(
        root / "diagnostics.json",
        {
            "sourceId": record.source_id,
            "originalFilename": record.original_filename,
            "documentType": record.document_type,
            "charCount": record.char_count,
            "pageCount": record.page_count,
            "signalCount": len(signals),
            "findings": [item.to_dict() for item in findings],
            "stage": "milestone_5_1_docset",
            "ocrArtifactPresent": ocr_artifact is not None,
        },
    )

    if structure_artifacts:
        if structure_artifacts.get("structure") is not None:
            write_json(root / "structure.v1.json", structure_artifacts["structure"])
        if structure_artifacts.get("sections") is not None:
            write_json(root / "sections.v1.json", structure_artifacts["sections"])
        if structure_artifacts.get("sectionLFulltext") is not None:
            write_json(root / "section-l.fulltext.v1.json", structure_artifacts["sectionLFulltext"])
        if structure_artifacts.get("sectionMFulltext") is not None:
            write_json(root / "section-m.fulltext.v1.json", structure_artifacts["sectionMFulltext"])
        if structure_artifacts.get("clauses") is not None:
            write_json(root / "clauses.v1.json", structure_artifacts["clauses"])

    if ocr_artifact is not None:
        write_json(root / "pages.ocr.v1.json", ocr_artifact)

    return root
