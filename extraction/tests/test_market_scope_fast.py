from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

from extraction.package_llm.corpus import PackageCorpus, PackageSourceText, assemble_package_corpus
from extraction.package_llm.fast_normalize import normalize_fast_extraction_payload
from extraction.package_llm.fast_schema import MARKET_SCOPE_FAST_SIGNAL_COUNT, MARKET_SCOPE_FAST_SIGNAL_IDS
from extraction.package_llm.forms.pdf_forms import extract_pdf_form_fields, format_form_fields_corpus_block
from extraction.package_llm.resolve import resolve_validated_package_signals
from extraction.package_llm.scope_corpus import assemble_scope_corpus
from extraction.package_llm.validate import validate_package_extraction
from extraction.pipeline import run_ingest_pipeline
from extraction.types import DocumentPage, SourceRecord


def _tyndall_pdf_path() -> Path | None:
    candidate = Path("data/runs/run_990fe1d2-af4e-4d10-add8-84d103160a2d/sources/4f25a7b3ec56735cc9a5e9ff1517af2f1f8fe07fd1d2736a20b553b0e44ffc6b.pdf")
    return candidate if candidate.is_file() else None


def _compact_signal(
    signal_id: str,
    value,
    *,
    quote: str = "",
    source_id: str = "src1",
    filename: str = "Solicitation.pdf",
) -> dict:
    return {
        "id": signal_id,
        "value": value,
        "confidence": "high",
        "status": "confirmed" if value is not None else "not_found",
        "source_id": source_id if value is not None else None,
        "filename": filename if value is not None else None,
        "page": 1 if value is not None else None,
        "sheet": None,
        "quote": quote or (str(value) if value is not None else ""),
        "review_note": "",
        "alternates": [],
    }


class FastSchemaTests(unittest.TestCase):
    def test_fast_schema_has_exactly_eleven_signals(self) -> None:
        self.assertEqual(len(MARKET_SCOPE_FAST_SIGNAL_IDS), 11)
        self.assertEqual(MARKET_SCOPE_FAST_SIGNAL_COUNT, 11)
        self.assertEqual(len(set(MARKET_SCOPE_FAST_SIGNAL_IDS)), 11)
        self.assertNotIn("rfp_competition_type_v1", MARKET_SCOPE_FAST_SIGNAL_IDS)
        self.assertNotIn("rfp_funding_office_v1", MARKET_SCOPE_FAST_SIGNAL_IDS)


class FastNormalizeTests(unittest.TestCase):
    def test_compact_payload_normalizes_to_scalar_standard_shape(self) -> None:
        compact = {
            "signals": [
                _compact_signal("rfp_primary_naics_v1", "722310", quote="NAICS: 722310"),
            ],
            "package_summary": {},
        }
        normalized = normalize_fast_extraction_payload(compact)
        self.assertEqual(len(normalized["signals"]), 11)
        naics = next(item for item in normalized["signals"] if item["id"] == "rfp_primary_naics_v1")
        self.assertEqual(naics["value"], "722310")
        self.assertIsInstance(naics["value"], str)
        self.assertEqual(len(naics["evidence"]), 1)


class FastCorpusTests(unittest.TestCase):
    def test_form_naics_reaches_scope_corpus(self) -> None:
        pdf_path = _tyndall_pdf_path()
        if not pdf_path:
            self.skipTest("Tyndall base PDF not available locally")
        pdf_bytes = pdf_path.read_bytes()
        records, _ = extract_pdf_form_fields(
            pdf_bytes,
            source_id="4f25a7b3ec56735cc9a5e9ff1517af2f1f8fe07fd1d2736a20b553b0e44ffc6b",
            filename=pdf_path.name,
        )
        block = format_form_fields_corpus_block(records)
        self.assertIn("722310", block)
        source = PackageSourceText(
            source_id="4f25a7b3ec56735cc9a5e9ff1517af2f1f8fe07fd1d2736a20b553b0e44ffc6b",
            filename=pdf_path.name,
            file_type="pdf",
            document_type_hint="base solicitation",
            pages=[DocumentPage(page_index=0, text="cover", char_count=5)],
            form_fields=records,
        )
        source.normalized_text_by_page[0] = "cover"
        fast_text, stats = assemble_scope_corpus([source])
        self.assertIn("722310", fast_text)
        self.assertIn("--- PDF FORM FIELDS ---", fast_text)
        self.assertGreater(stats["fastCorpusCharCount"], 0)


class FastWorkflowTests(unittest.TestCase):
    def _mock_compact_response(self, source_id: str, filename: str) -> dict:
        signals = [
            _compact_signal("rfp_solicitation_id_v1", "FA481926R0001", quote="FA481926R0001", source_id=source_id, filename=filename),
            _compact_signal("rfp_title_v1", "Tyndall mess attendant recompete", quote="mess attendant", source_id=source_id, filename=filename),
            _compact_signal("rfp_issuing_agency_v1", "Department of the Air Force", quote="Department of the Air Force", source_id=source_id, filename=filename),
            _compact_signal("rfp_issuing_office_v1", "325 CONS PKP", quote="325 CONS PKP", source_id=source_id, filename=filename),
            _compact_signal("rfp_office_aac_v1", "FA4819", quote="FA4819", source_id=source_id, filename=filename),
            _compact_signal("rfp_primary_naics_v1", "722310", quote="NAICS: 722310", source_id=source_id, filename=filename),
            _compact_signal("rfp_primary_psc_v1", "S203", quote="PSC S203", source_id=source_id, filename=filename),
            _compact_signal("rfp_contract_type_v1", "Firm-Fixed-Price", quote="Firm Fixed Price", source_id=source_id, filename=filename),
            _compact_signal("rfp_set_aside_v1", "8(a) Small Business Set-Aside", quote="8(a) small business set-aside", source_id=source_id, filename=filename),
            _compact_signal("rfp_place_of_performance_v1", "Tyndall AFB, Florida", quote="Tyndall AFB, Florida", source_id=source_id, filename=filename),
            _compact_signal(
                "solicitation_status_alert_v1",
                "Solicitation stayed indefinitely due to GAO protest",
                quote="GAO protest stay",
                source_id=source_id,
                filename=filename,
            ),
        ]
        return {"signals": signals, "package_summary": {"base_solicitation_filename": filename, "amendments_in_order": [], "controlling_amendment": None}}

    @patch("extraction.package_llm.run.load_cached_artifacts", return_value=None)
    @patch("extraction.package_llm.run.call_package_extraction")
    @patch("extraction.package_llm.run.has_openai_api_key", return_value=True)
    def test_fast_workflow_resolves_eleven_signals_only(self, _mock_key: object, mock_extract: object, _mock_cache: object) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "tyndall_sample.txt"
        from extraction.ingest.hash import sha256_hex

        source_id = sha256_hex(fixture.read_bytes())
        compact = self._mock_compact_response(source_id, fixture.name)
        normalized = normalize_fast_extraction_payload(compact)
        mock_extract.return_value = (normalized, {"model": "gpt-5.5", "openaiRequestCount": 1}, [])

        with patch("extraction.pipeline.get_extraction_mode", return_value="package_llm"):
            with patch("extraction.package_llm.run.get_extraction_profile", return_value="market_scope_fast"):
                result = run_ingest_pipeline([fixture], extract_text=True, extract_signals=True, resolve_signals=True)

        from extraction.persist import read_json
        from extraction.config import resolved_signals_json_path

        resolved = read_json(resolved_signals_json_path(result.run_id))
        signals = [s for s in resolved.get("signals") or [] if s.get("id") != "process_sources_v1"]
        self.assertEqual(len(signals), 11)
        summary = resolved.get("summary") or {}
        self.assertEqual(summary.get("profile"), "market_scope_fast")
        self.assertEqual(summary.get("requestedSignalCount"), 11)
        mock_extract.assert_called_once()

    @patch("extraction.package_llm.run.load_cached_artifacts")
    @patch("extraction.package_llm.run.has_openai_api_key", return_value=True)
    def test_cache_hit_avoids_gpt(self, _mock_key: object, mock_cache: object) -> None:
        fixture = Path(__file__).resolve().parent / "fixtures" / "tyndall_sample.txt"
        from extraction.package_llm.build_info import collect_build_info

        build_info = collect_build_info(profile="market_scope_fast")
        cached_resolved = {
            "signals": [{"id": "rfp_solicitation_id_v1", "canonical_value": "FA481926R0001", "resolution_status": "validated_model_extraction"}],
            "summary": {"profile": "market_scope_fast", "requestedSignalCount": 11, "total_signal_ids": 12},
        }
        mock_cache.return_value = {
            "raw": {"signals": []},
            "validated": {"signals": []},
            "resolved": cached_resolved,
            "validation_findings": [],
            "diagnostics": {"cacheHit": True, "buildInfo": build_info},
        }
        with patch.dict("os.environ", {"GOVCON_FORCE_PACKAGE_REFRESH": ""}, clear=False):
            with patch("extraction.package_llm.run.call_package_extraction") as mock_extract:
                with patch("extraction.pipeline.get_extraction_mode", return_value="package_llm"):
                    with patch("extraction.package_llm.run.get_extraction_profile", return_value="market_scope_fast"):
                        run_ingest_pipeline([fixture], extract_text=True, extract_signals=True, resolve_signals=True)
                mock_extract.assert_not_called()


class FastScopeReviewPolicyTests(unittest.TestCase):
    def test_pop_remains_review_only_filter(self) -> None:
        from app import SOLICITATION_USER_CONFIRMATION_FILTER_KEYS

        self.assertIn("pop_state", SOLICITATION_USER_CONFIRMATION_FILTER_KEYS)

    def test_funding_office_not_in_fast_signals(self) -> None:
        self.assertNotIn("rfp_funding_office_v1", MARKET_SCOPE_FAST_SIGNAL_IDS)


class FastCountTests(unittest.TestCase):
    def test_summary_counts_sum_to_eleven(self) -> None:
        validated = normalize_fast_extraction_payload(
            {
                "signals": [_compact_signal("rfp_primary_naics_v1", "722310", quote="722310")],
                "package_summary": {},
            }
        )
        corpus = PackageCorpus(
            sources=[
                PackageSourceText(
                    source_id="src1",
                    filename="Solicitation.pdf",
                    file_type="txt",
                    document_type_hint="base solicitation",
                    pages=[DocumentPage(page_index=0, text="NAICS: 722310", char_count=14)],
                )
            ],
            corpus_text="NAICS: 722310",
        )
        corpus.sources[0].normalized_text_by_page[0] = "NAICS: 722310"
        validated_payload, _ = validate_package_extraction(validated, corpus)
        resolved = resolve_validated_package_signals(
            "run_test",
            validated_payload,
            process_sources_signal={"value": "ok"},
            profile="market_scope_fast",
        )
        countable = [s for s in resolved["signals"] if s["id"] != "process_sources_v1"]
        self.assertEqual(len(countable), 11)
        statuses = [str(s.get("resolution_status") or "") for s in countable]
        confirmed = sum(1 for s in statuses if s.startswith("validated_model_extraction"))
        review = sum(1 for s in statuses if "review" in s or "withheld" in s)
        not_found = sum(1 for s in statuses if s == "not_found")
        self.assertEqual(confirmed + review + not_found, 11)


if __name__ == "__main__":
    unittest.main()
