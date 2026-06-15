from __future__ import annotations

from pathlib import Path

from extraction.sources.pandas_parser import parse_excel
from extraction.sources.word_parser import extract_word
from extraction.types import Finding, SourceRecord

EXCEL_EXTENSIONS = {"xlsx", "xls", "xlsm", "xlsb"}
WORD_EXTENSIONS = {"docx", "doc"}
TEXT_EXTENSIONS = {"csv", "txt"}


def _append_heading(heading: str, body: str) -> str:
    cleaned = body.strip()
    if not cleaned:
        return ""
    return f"\n\n--- {heading} ---\n\n{cleaned}\n\n"


def extract_supplemental_text(source: SourceRecord) -> tuple[str, list[Finding]]:
    findings: list[Finding] = []
    path = Path(source.abs_path)
    ext = source.ext.lower()

    try:
        if ext in EXCEL_EXTENSIONS:
            parsed = parse_excel(str(path))
            return _append_heading("EXCEL PRICING MODEL DATA", parsed), findings
        if ext in WORD_EXTENSIONS:
            parsed = extract_word(str(path))
            return _append_heading("WORD DOCUMENT", parsed), findings
        if ext in TEXT_EXTENSIONS:
            content = path.read_text(encoding="utf-8", errors="replace").strip()
            return _append_heading("TEXT/CSV DATA", content), findings
        content = path.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            label = path.name
            return _append_heading(f"FILE: {label}", content), findings
    except Exception as exc:
        code = "EXCEL_EXTRACTION_FAILED"
        if ext in WORD_EXTENSIONS:
            code = "WORD_EXTRACTION_FAILED"
        elif ext in TEXT_EXTENSIONS:
            code = "TEXT_SOURCE_READ_FAILED"
        else:
            code = "SUPPLEMENTAL_EXTRACTION_FAILED"
        findings.append(
            Finding(
                "warn",
                code,
                f"Supplemental extraction failed for {source.original_filename}",
                {"key": source.key, "error": str(exc)},
            )
        )
    return "", findings


def build_supplemental_text_for_sources(sources: list[SourceRecord]) -> tuple[str, list[Finding]]:
    findings: list[Finding] = []
    chunks: list[str] = []
    processed_exts = {"pdf", *EXCEL_EXTENSIONS, *WORD_EXTENSIONS, *TEXT_EXTENSIONS}

    for source in sources:
        if source.did_dedupe:
            continue
        ext = source.ext.lower()
        if ext == "pdf":
            continue
        text, source_findings = extract_supplemental_text(source)
        findings.extend(source_findings)
        if text:
            chunks.append(text)

    for source in sources:
        if source.did_dedupe:
            continue
        ext = source.ext.lower()
        if ext in processed_exts:
            continue
        text, source_findings = extract_supplemental_text(source)
        findings.extend(source_findings)
        if text:
            chunks.append(text)

    return "".join(chunks), findings
