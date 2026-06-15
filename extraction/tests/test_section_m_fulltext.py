from __future__ import annotations

import unittest
from unittest import mock

from extraction.postprocess.run import run_postprocess
from extraction.resolve.resolve_signals import resolve_signals_v1
from extraction.signals.evaluation_extract import extract_evaluation_signals_v1
from extraction.tests.test_structure_due_date import _page, _signal_value


def _pages_section_m_four_page() -> list:
    return [
        _page("SECTION M - Evaluation Factors\nFactor 1 Technical Approach", page_index=0),
        _page("Factor 2 Management Approach continues on this page.", page_index=1),
        _page("Factor 3 Past Performance considerations.", page_index=2),
        _page("Factor 4 Price. Relative importance: technical is significantly more important than price.", page_index=3),
    ]


class SectionMFulltextTests(unittest.TestCase):
    def test_section_m_spans_four_pages(self) -> None:
        from extraction.artifacts.build import build_document_artifacts

        pages = _pages_section_m_four_page()
        artifacts = build_document_artifacts("run_sm", pages, filename="solicitation.pdf")
        fulltext = artifacts["sectionMFulltext"]
        assert fulltext is not None
        self.assertIn("Factor 1", fulltext["fullText"])
        self.assertIn("Factor 4", fulltext["fullText"])
        self.assertEqual(fulltext["evidence_v1"]["pageIndexStart"], 0)
        self.assertEqual(fulltext["evidence_v1"]["pageIndexEnd"], 3)
        self.assertGreaterEqual(len(fulltext["evidence_v1"]["pageRanges"]), 4)

    def test_amendment_replaces_subsection_only(self) -> None:
        from extraction.artifacts.build import build_document_artifacts

        base_pages = _pages_section_m_four_page()
        amend_pages = [
            _page(
                "STANDARD FORM 30 Amendment 0002\nSECTION M Factor 2 is revised to include cybersecurity staffing.",
                page_index=0,
            )
        ]
        base = build_document_artifacts("run_base", base_pages, filename="solicitation.pdf")
        amend = build_document_artifacts(
            "run_amend",
            amend_pages,
            filename="amendment_0002.pdf",
            amendment_number="0002",
        )
        assert base["sectionMFulltext"] is not None
        assert amend["sectionMFulltext"] is not None
        self.assertIn("Factor 1", base["sectionMFulltext"]["fullText"])
        self.assertIn("Factor 2 is revised", amend["sectionMFulltext"]["fullText"])
        self.assertNotIn("Factor 1", amend["sectionMFulltext"]["fullText"])

    def test_amendment_replaces_section_m_entirety(self) -> None:
        from extraction.artifacts.build import build_document_artifacts

        base_pages = [_page("SECTION M\nFactor 1 Legacy\nFactor 2 Legacy", page_index=0)]
        amend_pages = [
            _page(
                "STANDARD FORM 30 Amendment 0004\nSECTION M is replaced in its entirety.\nFactor A Technical\nFactor B Price",
                page_index=0,
            )
        ]
        base = build_document_artifacts("run_base", base_pages, filename="solicitation.pdf")
        amend = build_document_artifacts(
            "run_amend",
            amend_pages,
            filename="amendment_0004.pdf",
            amendment_number="0004",
        )
        assert amend["sectionMFulltext"] is not None
        self.assertIn("Factor A", amend["sectionMFulltext"]["fullText"])
        self.assertNotIn("Legacy", amend["sectionMFulltext"]["fullText"])
        superseded = (amend["sectionMFulltext"]["evidence_v1"] or {}).get("supersededSegments") or []
        self.assertTrue(superseded or "replaced" in amend["sectionMFulltext"]["fullText"].lower())

    def test_continuation_page_factor_hierarchy(self) -> None:
        from extraction.artifacts.build import build_document_artifacts

        pages = [
            _page("SECTION M\nFactor 1 Technical", page_index=0),
            _page("CONTINUATION OF SECTION M\nSubfactor 1.1 staffing\nSubfactor 1.2 methodology", page_index=1),
        ]
        artifacts = build_document_artifacts("run_sm", pages, filename="solicitation.pdf")
        fulltext = artifacts["sectionMFulltext"]
        assert fulltext is not None
        self.assertIn("Subfactor 1.1", fulltext["fullText"])
        self.assertIn("Subfactor 1.2", fulltext["fullText"])

    def test_generic_attachment_not_in_section_m(self) -> None:
        from extraction.artifacts.build import build_document_artifacts

        pages = [
            _page("SECTION M\nLowest price technically acceptable evaluation applies.", page_index=0),
            _page("ATTACHMENT 5 - Company Brochure\nMarketing summary for contractor capabilities.", page_index=1),
        ]
        artifacts = build_document_artifacts("run_sm", pages, filename="solicitation.pdf")
        fulltext = artifacts["sectionMFulltext"]
        assert fulltext is not None
        self.assertNotIn("Marketing summary", fulltext["fullText"])
        self.assertNotIn("Company Brochure", fulltext["fullText"])


if __name__ == "__main__":
    unittest.main()
