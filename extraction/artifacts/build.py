from __future__ import annotations

import hashlib
import re
from typing import Any

from extraction.artifacts.section_fulltext import extract_section_fulltext_v1
from extraction.artifacts.structure_relations import build_structure_relations_v1
from extraction.types import CorpusPage, Finding

EXCERPT_MAX = 420
WINDOW_MAX = 1200


def _truncate(text: str, limit: int = EXCERPT_MAX) -> str:
    return text if len(text) <= limit else f"{text[: limit - 1]}…"


def _span_hash(source_sha256: str, page_index: int, start: int, end: int) -> str:
    payload = f"{source_sha256}:{page_index}:{start}:{end}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_span_loc_map(pages: list[CorpusPage]) -> dict[str, dict[str, Any]]:
    mapping: dict[str, dict[str, Any]] = {}
    for page_ordinal, page in enumerate(pages):
        for span in page.spans:
            mapping[span.sha256] = {
                "sourceSha256": page.source_sha256,
                "pageIndex": page.page_index,
                "pageOrdinal": page_ordinal,
                "start": span.start,
                "end": span.end,
            }
    return mapping


def build_structure_index_v1(
    run_id: str,
    pages: list[CorpusPage],
    *,
    filename: str = "",
    amendment_number: str | None = None,
) -> dict[str, Any]:
    enhanced = build_structure_relations_v1(
        run_id,
        pages,
        filename=filename,
        amendment_number=amendment_number,
    )
    items: list[dict[str, Any]] = list(enhanced.get("items") or [])

    section_re = re.compile(r"^\s*(?:SECTION|S\s*E\s*C\s*T\s*I\s*O\s*N)\s+([A-M])\b\s*[-:–—]?\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    attach_re = re.compile(r"^\s*(ATTACHMENT|EXHIBIT|APPENDIX|ANNEX)\b\s*[-:–—#]?\s*(.*)$", re.IGNORECASE | re.MULTILINE)
    for page in pages:
        for match in section_re.finditer(page.text):
            letter = match.group(1).upper()
            if letter in {"L", "M"} and any(
                str(item.get("logicalType") or "") == f"SECTION_{letter}" and item.get("pageIndex") == page.page_index
                for item in items
            ):
                continue
            start = match.start()
            end = min(len(page.text), start + WINDOW_MAX)
            excerpt = _truncate(page.text[start:end].replace("\n", " ").strip())
            full_text = page.text[start:end]
            span_hashes = [_span_hash(page.source_sha256, page.page_index, start, end)]
            items.append(
                {
                    "kind": "SECTION",
                    "id": letter,
                    "title": (match.group(2) or f"Section {letter}").strip(),
                    "logicalType": f"SECTION_{letter}",
                    "sourceSha256": page.source_sha256,
                    "pageIndex": page.page_index,
                    "spanHashes": span_hashes,
                    "excerpt": excerpt,
                    "fullText": full_text,
                }
            )
        for match in attach_re.finditer(page.text):
            start = match.start()
            end = min(len(page.text), start + WINDOW_MAX)
            excerpt = _truncate(page.text[start:end].replace("\n", " ").strip())
            items.append(
                {
                    "kind": "ATTACHMENT",
                    "id": match.group(1).upper(),
                    "title": (match.group(2) or match.group(1)).strip(),
                    "logicalType": "ATTACHMENT",
                    "sourceSha256": page.source_sha256,
                    "pageIndex": page.page_index,
                    "spanHashes": [_span_hash(page.source_sha256, page.page_index, start, end)],
                    "excerpt": excerpt,
                    "fullText": page.text[start:end],
                }
            )

    findings = list(enhanced.get("findings") or [])
    if not items:
        findings.append(Finding("info", "STRUCTURE_NO_HEADINGS", "No structure headings detected").to_dict())
    return {
        "version": 1,
        "runId": run_id,
        "items": items,
        "relationships": enhanced.get("relationships") or [],
        "findings": findings,
    }


def build_sections_index_v1(run_id: str, pages: list[CorpusPage]) -> dict[str, Any]:
    anchors: dict[str, list[re.Pattern[str]]] = {
        "opportunity": [
            re.compile(r"\bstatement of work\b", re.I),
            re.compile(r"\bsection\s+l\b", re.I),
            re.compile(r"\bsection\s+m\b", re.I),
            re.compile(r"\binstructions?\s+to\s+offerors\b", re.I),
        ],
        "entities": [
            re.compile(r"\bissued by\b", re.I),
            re.compile(r"\bcontracting officer\b", re.I),
            re.compile(r"\bpoint of contact\b", re.I),
            re.compile(r"\bsubmission instructions\b", re.I),
        ],
        "pricing": [re.compile(r"\bprice proposal\b", re.I), re.compile(r"\bclins?\b", re.I)],
        "compliance": [re.compile(r"\brepresentations and certifications\b", re.I), re.compile(r"\bfar\b", re.I)],
        "risks": [re.compile(r"\bsecurity\b", re.I), re.compile(r"\bcyber\b", re.I)],
    }
    sections: dict[str, dict[str, Any]] = {}
    for page_ordinal, page in enumerate(pages):
        for name, patterns in anchors.items():
            score = sum(len(pat.findall(page.text)) for pat in patterns)
            if score <= 0:
                continue
            start = 0
            end = min(len(page.text), WINDOW_MAX)
            entry = {
                "title": name.title(),
                "spanHashes": [_span_hash(page.source_sha256, page.page_index, start, end)],
                "excerpt": _truncate(page.text[start:end].replace("\n", " ").strip()),
            }
            existing = sections.get(name)
            if not existing or score > existing.get("_score", 0):
                entry["_score"] = score
                sections[name] = entry
    for entry in sections.values():
        entry.pop("_score", None)
    return {"version": 1, "runId": run_id, "sections": sections, "findings": []}


def extract_clause_index_v1(run_id: str, pages: list[CorpusPage]) -> dict[str, Any]:
    clauses: list[dict[str, Any]] = []
    seen: set[str] = set()
    clause_re = re.compile(r"\b((?:52|252)\.\d{3}-\d{1,4}|\d{2}\.\d{3}-\d{1,4})\b", re.I)
    for page in pages:
        for match in clause_re.finditer(page.text):
            clause_id = match.group(1).upper()
            if clause_id in seen:
                continue
            seen.add(clause_id)
            family = "DFARS" if clause_id.startswith("252.") else "FAR" if clause_id.startswith("52.") else "OTHER"
            start = max(0, match.start() - 40)
            end = min(len(page.text), match.end() + 80)
            clauses.append(
                {
                    "clauseId": clause_id,
                    "family": family,
                    "mentions": [
                        {
                            "sourceSha256": page.source_sha256,
                            "pageIndex": page.page_index,
                            "spanHash": _span_hash(page.source_sha256, page.page_index, start, end),
                            "excerpt": _truncate(page.text[start:end].replace("\n", " ")),
                        }
                    ],
                }
            )
    return {"version": 1, "runId": run_id, "clauses": clauses, "findings": []}


def build_document_artifacts(
    run_id: str,
    pages: list[CorpusPage],
    *,
    filename: str = "",
    amendment_number: str | None = None,
) -> dict[str, Any]:
    structure = build_structure_index_v1(run_id, pages, filename=filename, amendment_number=amendment_number)
    return {
        "structure": structure,
        "sections": build_sections_index_v1(run_id, pages),
        "sectionLFulltext": extract_section_fulltext_v1(
            run_id,
            pages,
            section="L",
            structure=structure,
            filename=filename,
            amendment_number=amendment_number,
        ),
        "sectionMFulltext": extract_section_fulltext_v1(
            run_id,
            pages,
            section="M",
            structure=structure,
            filename=filename,
            amendment_number=amendment_number,
        ),
        "clauses": extract_clause_index_v1(run_id, pages),
    }
