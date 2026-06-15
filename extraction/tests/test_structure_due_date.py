from __future__ import annotations

import hashlib
import unittest

from extraction.artifacts.build import build_document_artifacts
from extraction.artifacts.structure_relations import build_structure_relations_v1
from extraction.signals.due_date_candidates import (
    CandidateType,
    build_due_date_candidate,
    extract_due_date_candidates_from_text,
    extract_timezone,
    select_proposal_due_candidate,
)
from extraction.signals.extract_signals_v1 import extract_signals_v1
from extraction.types import CorpusPage, CorpusSpan


def _span_hash(source_sha256: str, page_index: int, start: int, end: int) -> str:
    payload = f"{source_sha256}:{page_index}:{start}:{end}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _page(text: str, *, page_index: int = 0, source_sha256: str = "abc123") -> CorpusPage:
    start, end = 0, len(text)
    return CorpusPage(
        source_sha256=source_sha256,
        page_index=page_index,
        text=text,
        spans=[CorpusSpan(start=start, end=end, sha256=_span_hash(source_sha256, page_index, start, end))],
    )


def _signal_value(signals: list[dict], signal_id: str) -> str | None:
    for item in signals:
        if item.get("id") == signal_id:
            return item.get("value")
    return None


def _run_extract(text: str, *, filename: str = "solicitation.pdf", amendment_number: str | None = None) -> list[dict]:
    pages = [_page(text)]
    artifacts = build_document_artifacts("run_test", pages, filename=filename, amendment_number=amendment_number)
    signals, _ = extract_signals_v1(
        run_id="run_test",
        pages=pages,
        structure=artifacts["structure"],
        sections=artifacts["sections"],
        clauses=artifacts["clauses"],
        section_l_fulltext=artifacts["sectionLFulltext"],
        section_m_fulltext=artifacts["sectionMFulltext"],
        source_filename=filename,
    )
    return signals


class StructureDueDateTests(unittest.TestCase):
    def test_firm_section_l_beats_anticipated(self) -> None:
        text = (
            "Anticipated proposal due date: 01 July 2026\n"
            "SECTION L\n"
            "Proposal Due Date: 15 May 2026\n"
            "Offers must be received by the deadline above."
        )
        signals = _run_extract(text)
        self.assertEqual(_signal_value(signals, "rfp_due_date_v1"), "2026-05-15")
        ev1 = next(item["evidence_v1"] for item in signals if item["id"] == "rfp_due_date_v1")
        candidates = ev1.get("dueDateCandidates") or []
        statuses = {item.get("status") for item in candidates if item.get("candidateType") == "proposal_due"}
        self.assertIn("firm", statuses)
        self.assertIn("anticipated", statuses)

    def test_latest_amendment_updates_date_time_timezone(self) -> None:
        text = (
            "STANDARD FORM 30\n"
            "Amendment No. 0004\n"
            "Offers must be received by 30 June 2026 at 3:00 PM Eastern Time."
        )
        signals = _run_extract(text, filename="amendment_0004.pdf", amendment_number="0004")
        self.assertEqual(_signal_value(signals, "rfp_due_date_v1"), "2026-06-30")
        dt_local = _signal_value(signals, "rfp_due_datetime_local_v1")
        self.assertIsNotNone(dt_local)
        assert dt_local is not None
        self.assertIn("3:00 PM", dt_local)
        self.assertIn("Eastern", dt_local)
        ev1 = next(item["evidence_v1"] for item in signals if item["id"] == "rfp_due_date_v1")
        self.assertEqual(ev1.get("timezoneText"), "Eastern Time")

    def test_pws_schedule_date_not_proposal_due(self) -> None:
        text = (
            "PERFORMANCE WORK STATEMENT\n"
            "Project start date is 01 May 2026. The proposal narrative should describe staffing."
        )
        candidates = extract_due_date_candidates_from_text(
            text,
            source_document="pws.pdf",
            source_sha256="abc123",
            page_index=0,
            span_hashes=[],
            source_hint="attachment",
            logical_structure_type="PWS",
        )
        proposal = [c for c in candidates if c.candidate_type == CandidateType.PROPOSAL_DUE]
        self.assertEqual(proposal, [])
        signals = _run_extract(text, filename="pws.pdf")
        self.assertIsNone(_signal_value(signals, "rfp_due_date_v1"))

    def test_timezone_ambiguity_preserves_central_time(self) -> None:
        excerpt = "Offers must be received by 2:00 PM Central Time on 15 May 2026."
        tz_text, tz_iana = extract_timezone(excerpt)
        self.assertEqual(tz_text, "Central Time")
        self.assertIsNone(tz_iana)
        candidate = build_due_date_candidate(
            date_raw="15 May 2026",
            excerpt=excerpt,
            source_hint="sectionLFulltext",
            source_document="solicitation.pdf",
            span_hashes=[],
            page_index=0,
            source_sha256="abc123",
        )
        assert candidate is not None
        self.assertEqual(candidate.timezone_text, "Central Time")
        self.assertIsNone(candidate.timezone_iana)

    def test_sf30_continuation_associated_with_parent(self) -> None:
        page1 = _page("STANDARD FORM 30\nAmendment No. 0004\nItem 8: See continuation sheet.", page_index=0)
        page2 = _page(
            "SF 30 CONTINUATION OF SOLICITATION\n"
            "Proposal Due Date: 30 June 2026 at 3:00 PM Eastern Time.",
            page_index=1,
        )
        structure = build_structure_relations_v1("run_test", [page1, page2], filename="amendment_0004.pdf", amendment_number="0004")
        items = structure.get("items") or []
        cont = next(item for item in items if item.get("logicalType") == "SF30_CONTINUATION")
        self.assertIsNotNone(cont.get("parentItemId"))
        parent = next(item for item in items if item.get("itemId") == cont.get("parentItemId"))
        self.assertEqual(parent.get("logicalType"), "SF30")

        artifacts = build_document_artifacts("run_test", [page1, page2], filename="amendment_0004.pdf", amendment_number="0004")
        signals, _ = extract_signals_v1(
            run_id="run_test",
            pages=[page1, page2],
            structure=artifacts["structure"],
            sections=artifacts["sections"],
            clauses=artifacts["clauses"],
            section_l_fulltext=artifacts["sectionLFulltext"],
            section_m_fulltext=artifacts["sectionMFulltext"],
            source_filename="amendment_0004.pdf",
        )
        self.assertEqual(_signal_value(signals, "rfp_due_date_v1"), "2026-06-30")

    def test_revised_pws_supersedes_pws_not_due_date(self) -> None:
        base_text = "SECTION L\nProposal Due Date: 15 May 2026\nOffers must be received by the deadline."
        amend_text = (
            "STANDARD FORM 30\nAmendment No. 0004\n"
            "REVISED PERFORMANCE WORK STATEMENT incorporated by amendment.\n"
            "REPLACE IN ITS ENTIRETY Attachment 3 PWS."
        )
        base_signals = _run_extract(base_text, filename="solicitation.pdf")
        amend_signals = _run_extract(amend_text, filename="amendment_0004.pdf", amendment_number="0004")
        self.assertEqual(_signal_value(base_signals, "rfp_due_date_v1"), "2026-05-15")
        self.assertIsNone(_signal_value(amend_signals, "rfp_due_date_v1"))

        pages = [_page(amend_text)]
        structure = build_structure_relations_v1("run_test", pages, filename="amendment_0004.pdf", amendment_number="0004")
        revised = next(item for item in structure["items"] if item.get("logicalType") == "REVISED_PWS")
        self.assertTrue(revised.get("revisedRelationship"))

    def test_qa_clarification_does_not_override_formal_deadline(self) -> None:
        text = (
            "SECTION L\nProposal Due Date: 15 May 2026\nOffers must be received by the deadline.\n"
            "QUESTIONS AND ANSWERS\n"
            "Q: Can the due date move to 01 June 2026?\nA: No change to the formal deadline."
        )
        signals = _run_extract(text)
        self.assertEqual(_signal_value(signals, "rfp_due_date_v1"), "2026-05-15")

    def test_amendment_supersedes_earlier_deadline_in_candidate_pool(self) -> None:
        base = build_due_date_candidate(
            date_raw="15 May 2026",
            excerpt="SECTION L Proposal Due Date: 15 May 2026",
            source_hint="sectionLFulltext",
            source_document="solicitation.pdf",
            span_hashes=[],
            page_index=0,
            source_sha256="abc123",
        )
        amend = build_due_date_candidate(
            date_raw="30 June 2026",
            excerpt="STANDARD FORM 30 Amendment 0004 Offers must be received by 30 June 2026 at 3:00 PM Eastern Time",
            source_hint="sf30",
            source_document="amendment_0004.pdf",
            span_hashes=[],
            page_index=0,
            source_sha256="def456",
            amendment_number="0004",
            logical_structure_type="SF30",
        )
        assert base is not None and amend is not None
        winner, alternates = select_proposal_due_candidate([base, amend])
        assert winner is not None
        self.assertEqual(winner.date_iso, "2026-06-30")
        self.assertEqual(winner.amendment_number, "0004")
        self.assertTrue(alternates)


if __name__ == "__main__":
    unittest.main()
