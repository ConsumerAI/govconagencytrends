from __future__ import annotations

import unittest

from extraction.package_llm.corpus import PackageCorpus, PackageSourceText, quote_in_source
from extraction.package_llm.validators.common import validate_signal_record
from extraction.package_llm.validators.evidence_match import (
    NEARBY_PAGE_TOLERANCE,
    TOKEN_COVERAGE_THRESHOLD,
    match_evidence_quote,
    normalize_evidence_text,
)
from extraction.types import DocumentPage


def _source(pages: list[str], *, source_id: str = "src1", filename: str = "Doc.pdf") -> PackageSourceText:
    package_source = PackageSourceText(
        source_id=source_id,
        filename=filename,
        file_type="pdf",
        document_type_hint="base solicitation",
        pages=[DocumentPage(page_index=i, text=text, char_count=len(text)) for i, text in enumerate(pages)],
    )
    for page in package_source.pages:
        package_source.normalized_text_by_page[page.page_index] = page.text
    return package_source


class EvidenceNormalizationTests(unittest.TestCase):
    def test_whitespace_and_line_breaks(self) -> None:
        text = "This   is\na competitive,\n8(a) small business set-aside"
        normalized = normalize_evidence_text(text)
        self.assertIn("8(a) small business set-aside", normalized)

    def test_hyphenated_line_wrap(self) -> None:
        page = "This is a competitive, 8(a) small business set-\naside conducted in accordance with FAR."
        quote = "8(a) small business set-aside conducted in accordance"
        source = _source([page])
        result = match_evidence_quote(source, page=1, sheet=None, quote=quote)
        self.assertTrue(result.matched)

    def test_curly_quotes(self) -> None:
        page = "Offeror’s Total Evaluated Price shall be used for ordering."
        quote = "Offeror's Total Evaluated Price shall be used"
        source = _source(["intro", page, "tail"])
        result = match_evidence_quote(source, page=2, sheet=None, quote=quote)
        self.assertTrue(result.matched)

    def test_ligatures(self) -> None:
        page = "The efﬁcient offeror submitted a proposal."
        quote = "efficient offeror submitted"
        source = _source([page])
        result = match_evidence_quote(source, page=1, sheet=None, quote=quote)
        self.assertTrue(result.matched)

    def test_model_ellipsis(self) -> None:
        page = (
            "Step 2: Order all responsive proposals based upon offeror Total Evaluated Price. "
            "Step 6: the Government will make a Best-Value Determination based on those proposals."
        )
        quote = "Step 2: Order all responsive proposals ... Step 6: the Government will make a Best-Value Determination"
        source = _source(["section m intro", page])
        result = match_evidence_quote(source, page=2, sheet=None, quote=quote)
        self.assertTrue(result.matched)
        self.assertEqual(result.method, "segment_ordered")

    def test_truncated_quote(self) -> None:
        page = "This is a competitive, 8(a) small business set-aside conducted in accordance with FAR parts 12 and 15."
        quote = "competitive, 8(a) small business set-aside conducted"
        source = _source([page])
        result = match_evidence_quote(source, page=1, sheet=None, quote=quote)
        self.assertTrue(result.matched)

    def test_page_off_by_one(self) -> None:
        target = "Proposals must be received by 325 CONS/PKB via the Solicitation Module."
        source = _source(["cover", target, "tail"])
        result = match_evidence_quote(source, page=1, sheet=None, quote=target)
        self.assertTrue(result.matched)
        self.assertIn(result.method, {"nearby_page_exact", "token_window", "source_exact", "exact_normalized"})

    def test_spreadsheet_sheet_match(self) -> None:
        source = PackageSourceText(
            source_id="xlsx1",
            filename="Pricing.xlsx",
            file_type="xlsx",
            document_type_hint="pricing",
        )
        source.normalized_text_by_sheet["Sheet1"] = "Product Service Code: S203\nPricing Arrangement: Firm Fixed Price"
        quote = "Product Service Code: S203"
        result = match_evidence_quote(source, page=None, sheet="Sheet1", quote=quote)
        self.assertTrue(result.matched)

    def test_unrelated_text_with_superficial_overlap_fails(self) -> None:
        page = "The contractor shall provide food services and support to the dining facility."
        quote = "competitive 8(a) small business set-aside conducted in accordance with FAR"
        source = _source([page])
        result = match_evidence_quote(source, page=1, sheet=None, quote=quote)
        self.assertFalse(result.matched)
        self.assertLess(result.score, TOKEN_COVERAGE_THRESHOLD)


class EvidenceValidationPolicyTests(unittest.TestCase):
    def test_one_of_three_quotes_failing_still_confirms(self) -> None:
        source = _source(
            [
                "Primary evidence: 8(a) small business set-aside is selected for this acquisition.",
                "Secondary page without the failed quote.",
            ]
        )
        corpus = PackageCorpus(sources=[source], corpus_text="")
        signal = {
            "id": "rfp_set_aside_v1",
            "value": "8(a) small business set-aside",
            "confidence": "high",
            "status": "confirmed",
            "controlling_source": {"source_id": "src1", "page": 1},
            "evidence": [
                {
                    "source_id": "src1",
                    "filename": "Doc.pdf",
                    "page": 1,
                    "quote": "8(a) small business set-aside is selected for this acquisition",
                },
                {"source_id": "src1", "filename": "Doc.pdf", "page": 2, "quote": "this quote is completely absent from corpus"},
                {"source_id": "src1", "filename": "Doc.pdf", "page": 2, "quote": "another absent quote for warning only"},
            ],
            "alternates": [],
        }
        validated, findings = validate_signal_record(signal, corpus)
        self.assertEqual(validated["status"], "confirmed")
        self.assertTrue(validated["evidence_validation"]["authoritativeEvidenceValidated"])
        self.assertEqual(validated["evidence_validation"]["validatedEvidenceCount"], 1)
        self.assertEqual(validated["evidence_validation"]["failedEvidenceCount"], 2)
        self.assertTrue(any(item.code == "EVIDENCE_QUOTE_NOT_FOUND" for item in findings))

    def test_all_quotes_failing_marks_review_required(self) -> None:
        source = _source(["Unrelated page content only."])
        corpus = PackageCorpus(sources=[source], corpus_text="")
        signal = {
            "id": "rfp_set_aside_v1",
            "value": "8(a) small business set-aside",
            "confidence": "high",
            "status": "confirmed",
            "evidence": [
                {"source_id": "src1", "filename": "Doc.pdf", "page": 1, "quote": "missing quote one"},
                {"source_id": "src1", "filename": "Doc.pdf", "page": 1, "quote": "missing quote two"},
            ],
            "alternates": [],
        }
        validated, _ = validate_signal_record(signal, corpus)
        self.assertEqual(validated["status"], "review_required")
        self.assertFalse(validated["evidence_validation"]["authoritativeEvidenceValidated"])

    def test_quote_in_source_delegates_to_matcher(self) -> None:
        source = _source(["This acquisition is an 8(a) small business set -aside for commercial services."])
        self.assertTrue(quote_in_source(source, page=1, sheet=None, quote="8(a) small business set-aside"))


if __name__ == "__main__":
    unittest.main()
