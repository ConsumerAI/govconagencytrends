from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extraction.docset.merge import merge_solicitation_set_signals
from extraction.documents.deterministic import extract_deterministic_signals
from extraction.documents.classify import TYPE_BASE_SOLICITATION, TYPE_AMENDMENT, TYPE_QA
from extraction.pipeline import run_ingest_pipeline


def _doc_entry(
    *,
    doc_key: str,
    file_name: str,
    text: str,
    document_type: str,
    is_amendment: bool,
    amendment_order: str | None,
    amendment_order_source: str | None = None,
    amendment_order_confidence: str | None = None,
    revised_attachments: list[str] | None = None,
) -> dict:
    signals, findings = extract_deterministic_signals(
        text,
        source_id=doc_key,
        file_name=file_name,
        document_type=document_type,
        amendment_order=amendment_order,
    )
    if is_amendment and amendment_order and not amendment_order_confidence:
        amendment_order_source = amendment_order_source or "body"
        amendment_order_confidence = amendment_order_confidence or "authoritative"
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
        "findings": [item.to_dict() for item in findings],
    }


class Milestone5MergeTests(unittest.TestCase):
    def test_a_base_plus_four_amendments_due_date_wins(self) -> None:
        documents = [
            _doc_entry(
                doc_key="base",
                file_name="solicitation_base.pdf",
                text="PROPOSAL DUE DATE: 01 May 2026\nNAICS CODE: 722310",
                document_type=TYPE_BASE_SOLICITATION,
                is_amendment=False,
                amendment_order=None,
            ),
            _doc_entry(
                doc_key="a1",
                file_name="amendment_0001.pdf",
                text="Amendment 0001 updates page limit.\nPAGE LIMIT: 50 pages",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0001",
            ),
            _doc_entry(
                doc_key="a2",
                file_name="amendment_0002.pdf",
                text="Amendment 0002\nPROPOSAL DUE DATE: 15 May 2026",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0002",
            ),
            _doc_entry(
                doc_key="a3",
                file_name="amendment_0003.pdf",
                text="Amendment 0003 updates place of performance.\nPLACE OF PERFORMANCE: Tyndall AFB, FL",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0003",
            ),
            _doc_entry(
                doc_key="a4",
                file_name="amendment_0004.pdf",
                text="Amendment 0004\nPROPOSAL DUE DATE: 30 May 2026",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0004",
            ),
        ]
        merged, findings, superseded, conflicts = merge_solicitation_set_signals(documents)
        by_id = {item["id"]: item for item in merged}

        self.assertEqual(by_id["rfp_due_date_v1"]["value"], "30 May 2026")
        self.assertEqual(by_id["rfp_primary_naics_v1"]["value"], "722310")
        self.assertEqual(by_id["rfp_page_limits_v1"]["value"], "50 pages")
        self.assertGreater(superseded, 0)
        self.assertEqual(conflicts, 0)
        due_prov = by_id["rfp_due_date_v1"]["docset_provenance"]
        self.assertEqual(due_prov["controlling_amendment"], "0004")
        self.assertTrue(any(item.get("amendment") == "0002" for item in due_prov.get("supersedes") or []))
        self.assertTrue(any(item.code == "AMENDMENT_CONTROLLING_VALUE" for item in findings))

    def test_b_later_amendment_omission_preserves_naics(self) -> None:
        documents = [
            _doc_entry(
                doc_key="base",
                file_name="solicitation.pdf",
                text="NAICS CODE: 541511",
                document_type=TYPE_BASE_SOLICITATION,
                is_amendment=False,
                amendment_order=None,
            ),
            _doc_entry(
                doc_key="a4",
                file_name="amendment_0004.pdf",
                text="Amendment 0004 updates due date only.\nPROPOSAL DUE DATE: 01 Jun 2026",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0004",
            ),
        ]
        merged, _, _, _ = merge_solicitation_set_signals(documents)
        by_id = {item["id"]: item for item in merged}
        self.assertEqual(by_id["rfp_primary_naics_v1"]["value"], "541511")

    def test_c_explicit_deletion_removes_active_value(self) -> None:
        documents = [
            _doc_entry(
                doc_key="base",
                file_name="solicitation.pdf",
                text="PAGE LIMIT: 40 pages",
                document_type=TYPE_BASE_SOLICITATION,
                is_amendment=False,
                amendment_order=None,
            ),
            _doc_entry(
                doc_key="a2",
                file_name="amendment_0002.pdf",
                text="Amendment 0002: PAGE LIMIT is hereby removed and no longer applicable.",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0002",
            ),
            _doc_entry(
                doc_key="a4",
                file_name="amendment_0004.pdf",
                text="Amendment 0004 administrative note only.",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0004",
            ),
        ]
        merged, findings, _, _ = merge_solicitation_set_signals(documents)
        self.assertNotIn("rfp_page_limits_v1", {item["id"] for item in merged})
        self.assertTrue(any(item.code == "AMENDMENT_FIELD_DELETED" for item in findings))

    def test_d_sf30_amendment_wins_over_attachment(self) -> None:
        documents = [
            _doc_entry(
                doc_key="attach",
                file_name="attachment_context.pdf",
                text="Offers may be due around 10 May 2026 based on draft schedule.",
                document_type="attachment_exhibit",
                is_amendment=False,
                amendment_order=None,
            ),
            _doc_entry(
                doc_key="sf30",
                file_name="amendment_0003_sf30.pdf",
                text="STANDARD FORM 30 Amendment 0003\nPROPOSAL DUE DATE: 20 May 2026",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0003",
            ),
        ]
        merged, _, _, _ = merge_solicitation_set_signals(documents)
        by_id = {item["id"]: item for item in merged}
        self.assertEqual(by_id["rfp_due_date_v1"]["value"], "20 May 2026")
        self.assertEqual(by_id["rfp_due_date_v1"]["docset_provenance"]["controlling_amendment"], "0003")

    def test_e_qa_does_not_override_amendment(self) -> None:
        documents = [
            _doc_entry(
                doc_key="base",
                file_name="solicitation.pdf",
                text="PROPOSAL DUE DATE: 01 May 2026",
                document_type=TYPE_BASE_SOLICITATION,
                is_amendment=False,
                amendment_order=None,
            ),
            _doc_entry(
                doc_key="a2",
                file_name="amendment_0002.pdf",
                text="Amendment 0002\nPROPOSAL DUE DATE: 15 May 2026",
                document_type=TYPE_AMENDMENT,
                is_amendment=True,
                amendment_order="0002",
            ),
            _doc_entry(
                doc_key="qa",
                file_name="questions_and_answers.pdf",
                text="Q&A clarification: vendors asked if due date could be 25 May 2026.",
                document_type=TYPE_QA,
                is_amendment=False,
                amendment_order=None,
            ),
        ]
        merged, _, _, _ = merge_solicitation_set_signals(documents)
        by_id = {item["id"]: item for item in merged}
        self.assertEqual(by_id["rfp_due_date_v1"]["value"], "15 May 2026")


class Milestone5PipelineTests(unittest.TestCase):
    @patch("extraction.pipeline.get_extraction_mode", return_value="legacy")
    @patch("extraction.pipeline.has_openai_api_key", return_value=False)
    def test_docset_artifacts_written_for_multi_file_package(self, _mock_key: object, _mock_mode: object) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            base = root / "solicitation_base.txt"
            amend = root / "amendment_0001.txt"
            base.write_text("PROPOSAL DUE DATE: 01 May 2026\nNAICS CODE: 722310", encoding="utf-8")
            amend.write_text("Amendment 0001\nPROPOSAL DUE DATE: 15 May 2026", encoding="utf-8")

            result = run_ingest_pipeline(
                [base, amend],
                extract_text=True,
                extract_signals=True,
                resolve_signals=False,
            )
            self.assertTrue(result.solicitation_set_detected)
            self.assertEqual(result.document_count, 2)
            self.assertGreater(result.signal_count, 0)

            run_dir = Path(result.manifest_path).parent
            docs_dir = run_dir / "documents"
            self.assertTrue(docs_dir.exists())
            doc_dirs = [path for path in docs_dir.iterdir() if path.is_dir()]
            self.assertEqual(len(doc_dirs), 2)
            for doc_dir in doc_dirs:
                self.assertTrue((doc_dir / "document.json").exists())
                self.assertTrue((doc_dir / "text.txt").exists())
                self.assertTrue((doc_dir / "signals.json").exists())


if __name__ == "__main__":
    unittest.main()
