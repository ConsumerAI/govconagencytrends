from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extraction.package_llm.schema import PACKAGE_EXTRACTION_SIGNAL_IDS
from extraction.ingest.hash import sha256_hex
from extraction.pipeline import run_ingest_pipeline


def _tyndall_mock_response(source_id: str, filename: str) -> dict:
    def signal(
        sid: str,
        value,
        *,
        quote: str = "",
        status: str = "confirmed",
    ) -> dict:
        return {
            "id": sid,
            "value": value,
            "confidence": "high",
            "status": status,
            "controlling_source": {
                "source_id": source_id,
                "filename": filename,
                "page": 1,
                "sheet": None,
                "amendment_number": None,
            },
            "evidence": [
                {
                    "source_id": source_id,
                    "filename": filename,
                    "page": 1,
                    "sheet": None,
                    "quote": quote or str(value or ""),
                }
            ]
            if value is not None
            else [],
            "reasoning_summary": "Mock Tyndall package extraction",
            "alternates": [],
        }

    signals = [
        signal("rfp_primary_naics_v1", "722310", quote="NAICS: 722310"),
        signal("rfp_primary_psc_v1", "S203", quote="PSC: S203"),
        signal("rfp_set_aside_v1", "8(a) small business set-aside", quote="8(a) small business set-aside"),
        signal("rfp_competition_type_v1", "competitive", quote="This is a competitive, 8(a) small business set-aside"),
        signal("rfp_solicitation_id_v1", "FA481926R0001", quote="FA481926R0001"),
        signal("rfp_office_aac_v1", "FA4819", quote="FA4819"),
        signal("rfp_issuing_office_v1", "325 CONS PKP", quote="325 CONS PKP"),
        signal("rfp_contract_type_v1", "Firm Fixed Price (FFP)", quote="Firm Fixed Price (FFP)"),
        signal("rfp_submission_method_v1", "DoD SAFE", quote="Submit proposals electronically via DoD SAFE"),
        signal("rfp_submission_destination_v1", "submissions.tyndall@example.mil", quote="submissions.tyndall@example.mil"),
        signal("rfp_incumbent_data_v1", "Trogon Group Services, LLC", quote="Trogon Group Services, LLC"),
        signal("rfp_prior_contract_piid_v1", "FA481924C0006", quote="Prior Contract PIID: FA481924C0006"),
        signal("rfp_pop_start_v1", "2026-10-01", quote="Start: 01 Oct 2026"),
        signal("rfp_pop_years_v1", "5 Years", quote="Total: 5 Years"),
        signal("solicitation_is_stalled_v1", None, status="not_found"),
        signal("solicitation_status_alert_v1", None, status="not_found"),
    ]
    present = {item["id"] for item in signals}
    for sid in PACKAGE_EXTRACTION_SIGNAL_IDS:
        if sid not in present:
            signals.append(signal(sid, None, status="not_found"))
    return {
        "signals": signals,
        "package_summary": {
            "base_solicitation_filename": "tyndall_sample.txt",
            "amendments_in_order": ["0005"],
            "controlling_amendment": "0005",
        },
    }


class PackageLLMPipelineTests(unittest.TestCase):
    @patch("extraction.package_llm.run.load_cached_artifacts", return_value=None)
    @patch("extraction.package_llm.run.call_package_extraction")
    @patch("extraction.package_llm.run.has_openai_api_key", return_value=True)
    def test_package_llm_pipeline_tyndall_fixture(self, _mock_key: object, mock_extract: object, _mock_cache: object) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "tyndall_sample.txt"
        source_id = sha256_hex(fixture.read_bytes())
        mock_extract.return_value = (
            _tyndall_mock_response(source_id, fixture.name),
            {"model": "gpt-5.5", "inputTokens": 100, "outputTokens": 200},
            [],
        )

        with patch("extraction.pipeline.get_extraction_mode", return_value="package_llm"):
            with patch("extraction.package_llm.run.get_extraction_profile", return_value="package_llm_full"):
                result = run_ingest_pipeline(
                    [fixture],
                    extract_text=True,
                    extract_signals=True,
                    resolve_signals=True,
                )

        self.assertTrue(result.resolved_signals_path)
        from extraction.persist import read_json
        from extraction.config import resolved_signals_json_path

        resolved = read_json(resolved_signals_json_path(result.run_id))
        by_id = {item["id"]: item for item in resolved.get("signals") or []}
        self.assertEqual(by_id["rfp_primary_naics_v1"]["canonical_value"], "722310")
        self.assertEqual(by_id["rfp_primary_psc_v1"]["canonical_value"], "S203")
        self.assertEqual(by_id["rfp_solicitation_id_v1"]["canonical_value"], "FA481926R0001")
        self.assertEqual(by_id["rfp_incumbent_data_v1"]["canonical_value"], "Trogon Group Services, LLC")
        self.assertEqual(resolved.get("summary", {}).get("producer"), "package_llm")
        codes = {finding.code for finding in result.findings}
        self.assertIn("PACKAGE_LLM_ENTERED", codes)
        self.assertIn("PACKAGE_LLM_COMPLETED", codes)
        self.assertIn("ARTIFACT_WRITTEN", codes)
        from extraction.config import documents_dir, docset_manifest_path, package_extraction_dir

        self.assertFalse(documents_dir(result.run_id).exists())
        self.assertFalse(docset_manifest_path(result.run_id).exists())
        self.assertTrue(package_extraction_dir(result.run_id).exists())


if __name__ == "__main__":
    unittest.main()
