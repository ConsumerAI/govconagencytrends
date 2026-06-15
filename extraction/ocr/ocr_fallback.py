from __future__ import annotations

from pathlib import Path
import hashlib
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any

from extraction.corpus.build import normalize_corpus_text
from extraction.corpus.pdf_extract import SCANNED_AVG_CHAR_PER_PAGE_THRESHOLD, SCANNED_TOTAL_CHAR_THRESHOLD
from extraction.types import DocumentPage, Finding

OCR_ENGINE_NAME = "tesseract"
OCR_FALLBACK_MAX_PAGES = 12
OCR_FALLBACK_LOW_TEXT_CHAR_THRESHOLD = 400
PAGE_NATIVE_SUFFICIENT_CHARS = SCANNED_AVG_CHAR_PER_PAGE_THRESHOLD


@dataclass
class OcrPageResult:
    page_index: int
    text: str
    engine: str = OCR_ENGINE_NAME
    confidence: float | None = None
    excerpt: str = ""


def _tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def _pypdfium2_available() -> bool:
    try:
        import pypdfium2  # noqa: F401

        return True
    except ImportError:
        return False


def ocr_environment_ready() -> bool:
    return _tesseract_available() and _pypdfium2_available()


def _truncate(text: str, limit: int = 280) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _ocr_image_bytes(image_bytes: bytes) -> tuple[str, float | None, list[Finding]]:
    findings: list[Finding] = []
    if not _tesseract_available():
        findings.append(
            Finding("warn", "OCR_FALLBACK_NOT_AVAILABLE", "OCR not available in environment (tesseract missing)")
        )
        return "", None, findings
    with tempfile.TemporaryDirectory(prefix="govcon-ocr-") as tmp:
        image_path = Path(tmp) / "page.png"
        image_path.write_bytes(image_bytes)
        try:
            completed = subprocess.run(
                ["tesseract", str(image_path), "stdout"],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            findings.append(Finding("warn", "OCR_FALLBACK_FAILED", f"OCR failed: {exc}"))
            return "", None, findings
        if completed.returncode != 0:
            findings.append(Finding("warn", "OCR_FALLBACK_FAILED", completed.stderr.strip() or "OCR failed"))
            return "", None, findings
        text = completed.stdout or ""
        confidence: float | None = None
        try:
            tsv = subprocess.run(
                ["tesseract", str(image_path), "stdout", "tsv"],
                capture_output=True,
                text=True,
                check=False,
            )
            if tsv.returncode == 0 and tsv.stdout:
                scores: list[float] = []
                for line in tsv.stdout.splitlines()[1:]:
                    parts = line.split("\t")
                    if len(parts) >= 12:
                        try:
                            conf = float(parts[10])
                            if conf >= 0:
                                scores.append(conf)
                        except ValueError:
                            continue
                if scores:
                    confidence = round(sum(scores) / len(scores), 2)
        except OSError:
            confidence = None
        return text, confidence, findings


def run_ocr_on_page_indices(
    pdf_bytes: bytes,
    page_indices: list[int],
    *,
    source_sha256: str,
    original_filename: str,
) -> tuple[list[OcrPageResult], list[Finding]]:
    findings: list[Finding] = []
    results: list[OcrPageResult] = []
    if not page_indices:
        return results, findings
    if not _pypdfium2_available():
        findings.append(Finding("warn", "OCR_FALLBACK_NOT_AVAILABLE", "OCR requires pypdfium2 for PDF rendering"))
        return results, findings
    if not _tesseract_available():
        findings.append(Finding("warn", "OCR_FALLBACK_NOT_AVAILABLE", "OCR not available in environment (tesseract missing)"))
        return results, findings

    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(pdf_bytes)
    for page_index in sorted(set(page_indices)):
        if page_index < 0 or page_index >= len(doc):
            continue
        page = doc[page_index]
        bitmap = page.render(scale=2.0)
        pil_image = bitmap.to_pil()
        buf = BytesIO()
        pil_image.save(buf, format="PNG")
        text, confidence, page_findings = _ocr_image_bytes(buf.getvalue())
        findings.extend(page_findings)
        normalized = normalize_corpus_text(text.replace("\r\n", "\n"))
        results.append(
            OcrPageResult(
                page_index=page_index,
                text=normalized,
                engine=OCR_ENGINE_NAME,
                confidence=confidence,
                excerpt=_truncate(normalized.replace("\n", " ")),
            )
        )
    if results and any(item.text.strip() for item in results):
        findings.append(
            Finding(
                "info",
                "OCR_FALLBACK_APPLIED",
                f"OCR applied to {len(results)} page(s) in {original_filename}",
                {
                    "sourceId": source_sha256,
                    "filename": original_filename,
                    "pageIndices": [item.page_index for item in results],
                },
            )
        )
    elif page_indices:
        findings.append(
            Finding(
                "error",
                "OCR_FALLBACK_EMPTY",
                f"OCR produced no usable text for {original_filename}",
                {"sourceId": source_sha256, "filename": original_filename, "pageIndices": page_indices},
            )
        )
    return results, findings


def merge_native_and_ocr_pages(
    native_pages: list[DocumentPage],
    ocr_results: list[OcrPageResult],
) -> tuple[list[DocumentPage], list[dict[str, Any]]]:
    ocr_by_index = {item.page_index: item for item in ocr_results}
    merged: list[DocumentPage] = []
    evidence_pages: list[dict[str, Any]] = []
    max_index = max([page.page_index for page in native_pages] + list(ocr_by_index.keys()), default=-1)

    for page_index in range(max_index + 1):
        native = next((page for page in native_pages if page.page_index == page_index), None)
        ocr = ocr_by_index.get(page_index)
        native_text = (native.text if native else "").strip()
        ocr_text = (ocr.text if ocr else "").strip()

        if len(native_text) >= PAGE_NATIVE_SUFFICIENT_CHARS:
            chosen_text = native_text
            provenance = "native"
            engine = None
            confidence = None
        elif ocr_text and len(ocr_text) > len(native_text):
            chosen_text = ocr_text
            provenance = "ocr"
            engine = ocr.engine if ocr else OCR_ENGINE_NAME
            confidence = ocr.confidence if ocr else None
        elif native_text:
            chosen_text = native_text
            provenance = "native"
            engine = None
            confidence = None
        else:
            chosen_text = ocr_text
            provenance = "ocr" if ocr_text else "native"
            engine = ocr.engine if ocr and ocr_text else None
            confidence = ocr.confidence if ocr and ocr_text else None

        if native_text and ocr_text and _normalize_for_dedupe(native_text) == _normalize_for_dedupe(ocr_text):
            chosen_text = native_text
            provenance = "native"
            engine = None
            confidence = None

        merged.append(
            DocumentPage(
                page_index=page_index,
                text=chosen_text,
                char_count=len(chosen_text),
                text_provenance=provenance,
                ocr_engine=engine,
                ocr_confidence=confidence,
            )
        )
        if provenance == "ocr" and ocr is not None:
            evidence_pages.append(
                {
                    "sourceId": page_index,
                    "pageIndex": page_index,
                    "pageNumber": page_index + 1,
                    "provenance": "ocr",
                    "ocrEngine": engine,
                    "ocrConfidence": confidence,
                    "excerpt": ocr.excerpt or _truncate(ocr_text.replace("\n", " ")),
                }
            )
    return merged, evidence_pages


def _normalize_for_dedupe(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip().lower()


def build_ocr_artifact(
    *,
    source_sha256: str,
    original_filename: str,
    merged_pages: list[DocumentPage],
    evidence_pages: list[dict[str, Any]],
    rerun_applied: bool,
) -> dict[str, Any]:
    return {
        "version": "pages.ocr.v1",
        "sourceSha256": source_sha256,
        "originalFilename": original_filename,
        "ocrEngine": OCR_ENGINE_NAME,
        "rerunApplied": rerun_applied,
        "pages": [
            {
                "pageIndex": page.page_index,
                "pageNumber": page.page_index + 1,
                "textProvenance": page.text_provenance,
                "ocrEngine": page.ocr_engine,
                "ocrConfidence": page.ocr_confidence,
                "charCount": page.char_count,
                "excerpt": _truncate(page.text.replace("\n", " ")),
            }
            for page in merged_pages
            if page.text_provenance == "ocr"
        ],
        "evidence": evidence_pages,
    }


def pages_requiring_ocr(pages: list[DocumentPage], *, page_char_counts: list[int] | None = None) -> list[int]:
    indices: list[int] = []
    for page in pages:
        native_count = page.char_count
        if page_char_counts and page.page_index < len(page_char_counts):
            native_count = page_char_counts[page.page_index]
        if native_count < PAGE_NATIVE_SUFFICIENT_CHARS:
            indices.append(page.page_index)
    return indices


def document_is_sparse(*, total_char_count: int, is_probably_scanned: bool) -> bool:
    if is_probably_scanned:
        return True
    return total_char_count < OCR_FALLBACK_LOW_TEXT_CHAR_THRESHOLD


def should_run_ocr_fallback(
    *,
    has_due_date: bool,
    has_submission_instructions: bool,
    corpus_char_count: int,
    no_text_layer: bool = False,
) -> bool:
    if no_text_layer:
        return True
    if corpus_char_count < OCR_FALLBACK_LOW_TEXT_CHAR_THRESHOLD:
        return not (has_due_date and has_submission_instructions)
    return False


def run_ocr_fallback_on_pdf(
    pdf_bytes: bytes,
    *,
    source_sha256: str,
    max_pages: int = OCR_FALLBACK_MAX_PAGES,
    original_filename: str = "",
) -> tuple[list[DocumentPage], list[Finding], dict[str, Any] | None]:
    page_indices = list(range(max_pages))
    ocr_results, findings = run_ocr_on_page_indices(
        pdf_bytes,
        page_indices,
        source_sha256=source_sha256,
        original_filename=original_filename or source_sha256,
    )
    if not ocr_results or not any(item.text.strip() for item in ocr_results):
        return [], findings, None
    native_pages = [DocumentPage(page_index=item.page_index, text="", char_count=0) for item in ocr_results]
    merged, evidence = merge_native_and_ocr_pages(native_pages, ocr_results)
    artifact = build_ocr_artifact(
        source_sha256=source_sha256,
        original_filename=original_filename or source_sha256,
        merged_pages=merged,
        evidence_pages=evidence,
        rerun_applied=True,
    )
    return merged, findings, artifact
