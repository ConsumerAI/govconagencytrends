from __future__ import annotations

from typing import Any

from extraction.artifacts.structure_relations import (
    is_superseded_structure_item,
    source_hint_for_structure_item,
    structure_items_for_extraction,
)
from extraction.signals.due_date_candidates import (
    DueDateCandidate,
    extract_due_date_candidates_from_text,
    format_due_datetime_local,
    select_proposal_due_candidate,
    select_questions_due_candidate,
)
from extraction.types import Finding


def extract_and_select_due_dates(
    *,
    pages: list[Any],
    structure: dict[str, Any] | None,
    source_filename: str,
    span_loc: dict[str, Any],
    section_l_fulltext: dict[str, Any] | None = None,
    section_m_fulltext: dict[str, Any] | None = None,
    amendment_number: str | None = None,
) -> tuple[DueDateCandidate | None, DueDateCandidate | None, list[DueDateCandidate], list[Finding]]:
    findings: list[Finding] = []
    all_candidates: list[DueDateCandidate] = []

    for label, fulltext, hint in (
        ("L", section_l_fulltext, "sectionLFulltext"),
        ("M", section_m_fulltext, "sectionMFulltext"),
    ):
        if not fulltext:
            continue
        text = str(fulltext.get("fullText") or "")
        if not text.strip():
            continue
        ev = fulltext.get("evidence_v1") or {}
        all_candidates.extend(
            extract_due_date_candidates_from_text(
                text,
                source_document=source_filename,
                source_sha256=str(ev.get("sourceSha256") or ""),
                page_index=int(ev.get("pageIndexStart") or 0),
                span_hashes=[str(x) for x in (ev.get("spanHashes") or []) if x],
                source_hint=hint,
                amendment_number=amendment_number,
                logical_structure_type=f"SECTION_{label}",
                provenance_prefix=f"[FULLTEXT:{label}]",
            )
        )

    structure_items = structure_items_for_extraction(structure)
    for item in structure_items:
        if is_superseded_structure_item(item, structure):
            continue
        text = str(item.get("fullText") or item.get("excerpt") or "")
        if not text.strip():
            continue
        source_hint = source_hint_for_structure_item(item)
        all_candidates.extend(
            extract_due_date_candidates_from_text(
                text,
                source_document=source_filename,
                source_sha256=str(item.get("sourceSha256") or ""),
                page_index=int(item.get("pageIndex") or 0),
                span_hashes=list(item.get("spanHashes") or []),
                source_hint=source_hint,
                amendment_number=str(item.get("amendmentNumber") or amendment_number or "") or None,
                logical_structure_type=str(item.get("logicalType") or ""),
                continuation_of=str(item.get("parentItemId") or "") or None,
                provenance_prefix=f"[{item.get('logicalType')}]",
            )
        )

    for page_ordinal, page in enumerate(pages):
        span_hashes = [span.sha256 for span in page.spans[:4]] if page.spans else []
        page_index = int(getattr(page, "page_index", page_ordinal))
        all_candidates.extend(
            extract_due_date_candidates_from_text(
                page.text,
                source_document=source_filename,
                source_sha256=str(page.source_sha256 or ""),
                page_index=page_index,
                span_hashes=span_hashes,
                source_hint="globalScan",
                amendment_number=amendment_number,
            )
        )

    proposal, alternates = select_proposal_due_candidate(all_candidates)
    questions = select_questions_due_candidate(all_candidates)

    if proposal and alternates:
        findings.append(
            Finding(
                "info",
                "DUE_DATE_ALTERNATES_RETAINED",
                "Non-winning due date candidates retained as alternates",
                {"alternates": [item.to_dict() for item in alternates[:8]]},
            )
        )

    return proposal, questions, all_candidates, findings


def due_signals_from_candidates(
    proposal: DueDateCandidate | None,
    questions: DueDateCandidate | None,
    all_candidates: list[DueDateCandidate],
    *,
    run_id: str,
    source_filename: str,
) -> list[dict[str, Any]]:
    signals: list[dict[str, Any]] = []

    def _legacy_evidence(candidate: DueDateCandidate) -> list[dict[str, str]]:
        return [
            {
                "sourceId": source_filename,
                "artifact": "text",
                "locator": f"spanHashes:{','.join(candidate.span_hashes[:3])}",
                "snippet": candidate.evidence_excerpt[:200],
            }
        ]

    if proposal and proposal.date_iso:
        ev1 = {
            "spanHashes": proposal.span_hashes,
            "excerpt": proposal.evidence_excerpt,
            "source": source_filename,
            "dueDateCandidates": [c.to_dict() for c in all_candidates[:12]],
            "timezoneText": proposal.timezone_text,
            "timezoneIana": proposal.timezone_iana,
            "time": proposal.time_text,
        }
        signals.append(
            {
                "id": "rfp_due_date_v1",
                "value": proposal.date_iso,
                "confidence": proposal.confidence,
                "evidence": _legacy_evidence(proposal),
                "evidence_v1": ev1,
                "findings": [],
            }
        )
        dt_local = format_due_datetime_local(proposal)
        if dt_local:
            signals.append(
                {
                    "id": "rfp_due_datetime_local_v1",
                    "value": dt_local,
                    "confidence": proposal.confidence,
                    "evidence": _legacy_evidence(proposal),
                    "evidence_v1": ev1,
                    "findings": [],
                }
            )

    if questions and questions.date_iso:
        ev1 = {
            "spanHashes": questions.span_hashes,
            "excerpt": questions.evidence_excerpt,
            "source": source_filename,
        }
        signals.append(
            {
                "id": "rfp_questions_due_v1",
                "value": questions.date_raw if not questions.date_iso else questions.date_iso,
                "confidence": questions.confidence,
                "evidence": _legacy_evidence(questions),
                "evidence_v1": ev1,
                "findings": [],
            }
        )

    return signals
