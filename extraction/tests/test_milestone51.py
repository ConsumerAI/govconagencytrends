from __future__ import annotations

import unittest
from unittest.mock import patch

from extraction.docset.merge import merge_solicitation_set_signals
from extraction.documents.deterministic import extract_deterministic_signals
from extraction.documents.classify import TYPE_AMENDMENT, TYPE_BASE_SOLICITATION
from extraction.ocr.ocr_fallback import should_run_ocr_fallback
from extraction.sf30.parse import parse_sf30_metadata, resolve_amendment_identity
from extraction.signals.authority import infer_authority_tier


def _signal(
    signal_id: str,
    value: str,
    *,
    source_hint: str | None = None,
    excerpt: str = "",
) -> dict:
    authority = infer_authority_tier(signal_id=signal_id, excerpt=excerpt or value, source_hint=source_hint)
    return {
        "id": signal_id,
        "value": value,
        "confidence": "high",
        "evidence_v1": {"excerpt": excerpt or value, "source": "test"},
        "authority": authority,
    }


def _doc(
    *,
    doc_key: str,
    file_name: str,
    document_type: str,
    is_amendment: bool,
    amendment_order: str | None = None,
    amendment_order_source: str | None = None,
    amendment_order_confidence: str | None = None,
    revised_attachments: list[str] | None = None,
    signals: list[dict],
) -> dict:
    return {
        "docKey": doc_key,
        "sourceId": doc_key,
        "fileName": file_name,
        "documentType": document_type,
        "isAmendment": is_amendment,
        "amendmentOrder": amendment_order,
        "amendmentOrderSource": amendment_order_source,
        "amendmentOrderConfidence": amendment_order_confidence,
        "revisedAttachments": revised_attachments or [],
        "signals": signals,
        "findings": [],
    }


class Milestone51Sf30Tests(unittest.TestCase):
    def test_sf30_amendment_number_from_content_generic_filename(self) -> None:
        text = (
            "STANDARD FORM 30 (Rev. 11/2016)\n"
            "Amendment of Solicitation/Modification of Contract\n"
            "Amendment No. 0004\n"
            "Solicitation No. FA1234-26-R-0001\n"
        )
        parsed = parse_sf30_metadata(text, filename="document.pdf")
        identity = resolve_amendment_identity(text=text, filename="document.pdf", sf30=parsed)

        self.assertTrue(parsed.is_sf30)
        self.assertEqual(parsed.amendment_order, "0004")
        self.assertEqual(identity["amendmentOrder"], "0004")
        self.assertEqual(identity["amendmentOrderSource"], "sf30")
        self.assertEqual(identity["amendmentOrderConfidence"], "authoritative")
        self.assertTrue(any(item.field == "amendment_number" for item in parsed.field_evidence))


class Milestone51ConflictTests(unittest.TestCase):
    def test_ambiguous_amendment_order_emits_conflict_without_canonical(self) -> None:
        documents = [
            _doc(
                doc_key="a2",
                file_name="amendment_0002.pdf",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0002",
                amendment_order_source="filename",
                amendment_order_confidence="provisional",
                signals=[_signal("rfp_due_date_v1", "15 May 2026", excerpt="PROPOSAL DUE DATE: 15 May 2026")],
            ),
            _doc(
                doc_key="a4",
                file_name="amendment_0004.pdf",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0004",
                amendment_order_source="filename",
                amendment_order_confidence="provisional",
                signals=[_signal("rfp_due_date_v1", "30 May 2026", excerpt="PROPOSAL DUE DATE: 30 May 2026")],
            ),
        ]
        merged, findings, _, conflicts = merge_solicitation_set_signals(documents)

        self.assertEqual(conflicts, 1)
        self.assertNotIn("rfp_due_date_v1", {item["id"] for item in merged})
        conflict = next(item for item in findings if item.code == "SIGNAL_CONFLICT")
        candidates = (conflict.details or {}).get("candidates") or []
        self.assertEqual(len(candidates), 2)
        values = {item["value"] for item in candidates}
        self.assertEqual(values, {"15 May 2026", "30 May 2026"})


class Milestone51RevisedAttachmentTests(unittest.TestCase):
    def test_revised_attachment_supersedes_old_attachment_values(self) -> None:
        documents = [
            _doc(
                doc_key="base",
                file_name="solicitation.pdf",
                document_type=TYPE_BASE_SOLICITATION,
                is_amendment=False,
                signals=[
                    _signal("rfp_primary_naics_v1", "541511", excerpt="NAICS CODE: 541511"),
                    _signal("rfp_page_limits_v1", "40 pages", excerpt="PAGE LIMIT: 40 pages"),
                ],
            ),
            _doc(
                doc_key="attach2",
                file_name="attachment_2.pdf",
                document_type="attachment_exhibit",
                is_amendment=False,
                signals=[_signal("rfp_page_limits_v1", "30 pages", excerpt="PAGE LIMIT: 30 pages", source_hint="attachment")],
            ),
            _doc(
                doc_key="a3",
                file_name="amendment_0003.pdf",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0003",
                amendment_order_source="sf30",
                amendment_order_confidence="authoritative",
                revised_attachments=["Attachment 2"],
                signals=[_signal("rfp_page_limits_v1", "50 pages", excerpt="PAGE LIMIT: 50 pages", source_hint="sf30")],
            ),
        ]
        merged, _, _, _ = merge_solicitation_set_signals(documents)
        by_id = {item["id"]: item for item in merged}

        self.assertEqual(by_id["rfp_page_limits_v1"]["value"], "50 pages")
        self.assertEqual(by_id["rfp_primary_naics_v1"]["value"], "541511")


class Milestone51SectionAuthorityTests(unittest.TestCase):
    def test_section_l_beats_generic_attachment_prose(self) -> None:
        documents = [
            _doc(
                doc_key="attach",
                file_name="attachment_context.pdf",
                document_type="attachment_exhibit",
                is_amendment=False,
                signals=[
                    _signal(
                        "rfp_due_date_v1",
                        "10 May 2026",
                        source_hint="attachment",
                        excerpt="Offers may be due around 10 May 2026 based on draft schedule.",
                    )
                ],
            ),
            _doc(
                doc_key="base",
                file_name="solicitation.pdf",
                document_type=TYPE_BASE_SOLICITATION,
                is_amendment=False,
                signals=[
                    _signal(
                        "rfp_due_date_v1",
                        "25 May 2026",
                        source_hint="sectionLFulltext",
                        excerpt="[FULLTEXT:L] SECTION L Instructions. PROPOSAL DUE DATE: 25 May 2026",
                    )
                ],
            ),
        ]
        merged, _, _, _ = merge_solicitation_set_signals(documents)
        by_id = {item["id"]: item for item in merged}
        self.assertEqual(by_id["rfp_due_date_v1"]["value"], "25 May 2026")
        self.assertEqual(by_id["rfp_due_date_v1"]["docset_provenance"]["controlling_filename"], "solicitation.pdf")


class Milestone51OcrTests(unittest.TestCase):
    def test_sparse_pdf_triggers_ocr_decision(self) -> None:
        self.assertTrue(
            should_run_ocr_fallback(
                has_due_date=False,
                has_submission_instructions=False,
                corpus_char_count=12,
                no_text_layer=True,
            )
        )

    @patch("extraction.ocr.ocr_fallback._tesseract_available", return_value=False)
    def test_ocr_unavailable_emits_diagnostic(self, _mock_tesseract: object) -> None:
        from extraction.ocr.ocr_fallback import run_ocr_fallback_on_pdf

        pages, findings, artifact = run_ocr_fallback_on_pdf(b"%PDF-1.4 minimal", source_sha256="abc")
        self.assertEqual(pages, [])
        self.assertIsNone(artifact)
        self.assertTrue(any(item.code == "OCR_FALLBACK_NOT_AVAILABLE" for item in findings))


class Milestone51TyndallParityTests(unittest.TestCase):
    def test_tyndall_package_parity_skipped_when_files_unavailable(self) -> None:
        from pathlib import Path

        package_dir = Path(__file__).resolve().parents[2] / "data" / "tyndall_package"
        golden_path = Path(__file__).resolve().parents[2] / "data" / "runs" / "run_m3-dotenv-test2" / "signals" / "resolved_signals.json"
        if not package_dir.exists():
            self.skipTest("Tyndall four-amendment package files not available locally")
        if not golden_path.is_file():
            self.skipTest("Golden resolved_signals.json not available")
        from extraction.parity.report import run_parity_comparison
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            local_path = Path(tmp) / "local_resolved.json"
            local_path.write_text('{"version":"resolved_signals.v1","runId":"run_local","signals":[]}', encoding="utf-8")
            report = run_parity_comparison(golden_path=golden_path, local_path=local_path, run_id="run_tyndall_parity_probe")
            self.assertIn("signals", report)
            self.assertIsNotNone(report.get("parityReportJson"))


if __name__ == "__main__":
    unittest.main()
