from __future__ import annotations

import unittest
from unittest.mock import patch

from extraction.docset.merge import merge_solicitation_set_signals
from extraction.documents.classify import TYPE_AMENDMENT, TYPE_BASE_SOLICITATION
from extraction.documents.deterministic import extract_deterministic_signals
from extraction.ocr.feedback import run_ocr_feedback_pass
from extraction.ocr.ocr_fallback import (
    OcrPageResult,
    merge_native_and_ocr_pages,
    ocr_environment_ready,
    pages_requiring_ocr,
    run_ocr_on_page_indices,
)
from extraction.signals.extract_signals_v1 import extract_signals_v1
from extraction.documents.corpus_pages import document_pages_to_corpus_pages
from extraction.artifacts.build import build_document_artifacts
from extraction.types import DocumentPage, DocumentRecord, Finding, PdfDocumentContext, SourceRecord


def _source(**kwargs) -> SourceRecord:
    defaults = {
        "key": "runs/test/sources/abc.pdf",
        "ext": "pdf",
        "bytes": 100,
        "sha256": "abc123",
        "original_filename": "file.pdf",
        "abs_path": "C:/tmp/file.pdf",
        "did_dedupe": False,
    }
    defaults.update(kwargs)
    return SourceRecord(**defaults)


def _record(**kwargs) -> DocumentRecord:
    defaults = {
        "source_id": "abc123",
        "sha256": "abc123",
        "original_filename": "amendment_0004.pdf",
        "ext": "pdf",
        "source_key": "runs/test/sources/abc123.pdf",
        "document_type": TYPE_AMENDMENT,
        "document_class": "amendment",
        "is_amendment": True,
        "amendment_raw": None,
        "amendment_order": None,
        "solicitation_number": None,
        "char_count": 0,
        "page_count": 1,
        "amendment_order_source": None,
        "amendment_order_confidence": None,
    }
    defaults.update(kwargs)
    return DocumentRecord(**defaults)


class OcrFeedbackScannedAmendmentTests(unittest.TestCase):
    @patch("extraction.ocr.feedback.ocr_environment_ready", return_value=True)
    @patch("extraction.ocr.feedback.run_ocr_on_page_indices")
    def test_scanned_amendment_rerun_controls_due_date(self, mock_ocr: object, _mock_ready: object) -> None:
        mock_ocr.return_value = (
            [
                OcrPageResult(
                    page_index=0,
                    text="STANDARD FORM 30\nAmendment 0004\nPROPOSAL DUE DATE: 30 May 2026",
                    excerpt="Amendment 0004 PROPOSAL DUE DATE: 30 May 2026",
                )
            ],
            [],
        )

        native_pages = [DocumentPage(page_index=0, text="", char_count=0, text_provenance="native")]
        pdf_context = PdfDocumentContext(
            pdf_bytes=b"%PDF-1.4",
            page_char_counts=[0],
            total_char_count=0,
            is_probably_scanned=True,
        )
        record = _record(is_amendment=True)
        feedback = run_ocr_feedback_pass(
            run_id="run_test",
            source=_source(original_filename="amendment_0004.pdf"),
            record=record,
            text="",
            pages=native_pages,
            signals=[{"id": "rfp_due_date_v1", "value": None}],
            pdf_context=pdf_context,
            text_findings=[],
        )
        self.assertTrue(feedback.rerun)
        assert feedback.record is not None
        corpus_pages = document_pages_to_corpus_pages(feedback.record.source_id, feedback.pages)
        artifacts = build_document_artifacts("run_test", corpus_pages)
        ocr_signals, _ = extract_signals_v1(
            run_id="run_test",
            pages=corpus_pages,
            structure=artifacts["structure"],
            sections=artifacts["sections"],
            clauses=artifacts["clauses"],
            section_l_fulltext=artifacts["sectionLFulltext"],
            section_m_fulltext=artifacts["sectionMFulltext"],
            source_filename=feedback.record.original_filename,
        )
        due = next(item for item in ocr_signals if item["id"] == "rfp_due_date_v1")
        self.assertTrue(due.get("value"))

        base_signals, _ = extract_deterministic_signals(
            "PROPOSAL DUE DATE: 01 May 2026",
            source_id="base",
            file_name="solicitation.pdf",
            document_type=TYPE_BASE_SOLICITATION,
            amendment_order=None,
        )
        amend_signals = ocr_signals
        documents = [
            {
                "docKey": "base",
                "sourceId": "base",
                "fileName": "solicitation.pdf",
                "documentType": TYPE_BASE_SOLICITATION,
                "isAmendment": False,
                "amendmentOrder": None,
                "amendmentOrderSource": None,
                "amendmentOrderConfidence": None,
                "revisedAttachments": [],
                "signals": base_signals,
                "findings": [],
            },
            {
                "docKey": "abc123",
                "sourceId": "abc123",
                "fileName": "amendment_0004.pdf",
                "documentType": TYPE_AMENDMENT,
                "isAmendment": True,
                "amendmentOrder": "0004",
                "amendmentOrderSource": "body",
                "amendmentOrderConfidence": "authoritative",
                "revisedAttachments": [],
                "signals": amend_signals,
                "findings": [],
            },
        ]
        merged, _, _, _ = merge_solicitation_set_signals(documents)
        by_id = {item["id"]: item for item in merged}
        self.assertIn("rfp_due_date_v1", by_id)
        self.assertEqual(by_id["rfp_due_date_v1"]["docset_provenance"]["controlling_amendment"], "0004")


class OcrMixedNativeTests(unittest.TestCase):
    def test_ocr_only_sparse_page_preserves_native(self) -> None:
        native_pages = [
            DocumentPage(page_index=0, text="SECTION L native instructions with enough text " * 3, char_count=120),
            DocumentPage(page_index=1, text="", char_count=0),
        ]
        ocr_results = [
            OcrPageResult(page_index=1, text="PAGE LIMIT: 50 pages", excerpt="PAGE LIMIT: 50 pages"),
        ]
        merged, evidence = merge_native_and_ocr_pages(native_pages, ocr_results)
        self.assertEqual(merged[0].text_provenance, "native")
        self.assertEqual(merged[1].text_provenance, "ocr")
        self.assertIn("SECTION L", merged[0].text)
        self.assertEqual(merged[1].text, "PAGE LIMIT: 50 pages")
        self.assertEqual(evidence[0]["pageNumber"], 2)


class OcrUnavailableTests(unittest.TestCase):
    @patch("extraction.ocr.feedback.ocr_environment_ready", return_value=False)
    def test_controlling_amendment_blocks_when_ocr_unavailable(self, _mock_ready: object) -> None:
        feedback = run_ocr_feedback_pass(
            run_id="run_test",
            source=_source(),
            record=_record(is_amendment=True, document_type=TYPE_AMENDMENT),
            text="",
            pages=[DocumentPage(page_index=0, text="", char_count=0)],
            signals=[],
            pdf_context=PdfDocumentContext(pdf_bytes=b"%PDF", page_char_counts=[0], total_char_count=0, is_probably_scanned=True),
            text_findings=[],
        )
        self.assertFalse(feedback.rerun)
        self.assertTrue(any(item.code == "OCR_FALLBACK_NOT_AVAILABLE" for item in feedback.findings))
        self.assertTrue(any(item.code == "CONTROLLING_DOCUMENT_UNREADABLE" for item in feedback.findings))


class OcrNoUnnecessaryTests(unittest.TestCase):
    def test_readable_pdf_does_not_require_ocr(self) -> None:
        pages = [
            DocumentPage(page_index=0, text="PROPOSAL DUE DATE: 01 May 2026\nNAICS CODE: 541511", char_count=80),
        ]
        self.assertEqual(pages_requiring_ocr(pages), [])
        with patch("extraction.ocr.feedback.run_ocr_on_page_indices") as mock_ocr:
            feedback = run_ocr_feedback_pass(
                run_id="run_test",
                source=_source(original_filename="readable.pdf"),
                record=_record(document_type=TYPE_BASE_SOLICITATION, is_amendment=False, original_filename="readable.pdf"),
                text=pages[0].text,
                pages=pages,
                signals=[{"id": "rfp_due_date_v1", "value": "2026-05-01"}],
                pdf_context=PdfDocumentContext(
                    pdf_bytes=b"%PDF",
                    page_char_counts=[80],
                    total_char_count=80,
                    is_probably_scanned=False,
                ),
                text_findings=[],
            )
            mock_ocr.assert_not_called()
            self.assertFalse(feedback.rerun)


class OcrDedupeTests(unittest.TestCase):
    def test_duplicate_native_and_ocr_prefers_native(self) -> None:
        text = "PROPOSAL DUE DATE: 01 May 2026"
        native_pages = [DocumentPage(page_index=0, text=text, char_count=len(text))]
        ocr_results = [OcrPageResult(page_index=0, text=text, excerpt=text)]
        merged, evidence = merge_native_and_ocr_pages(native_pages, ocr_results)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].text_provenance, "native")
        self.assertEqual(evidence, [])


if __name__ == "__main__":
    unittest.main()
