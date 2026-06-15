from __future__ import annotations

import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extraction.package_llm.corpus import build_package_corpus, extract_package_source_text
from extraction.package_llm.forms.pdf_forms import (
    extract_pdf_form_fields,
    format_form_fields_corpus_block,
    form_page_needs_render_fallback,
)
from extraction.package_llm.solicitation_id import canonical_base_solicitation_id, strip_amendment_suffix
from extraction.package_llm.validators.common import validate_signal_record
from extraction.package_llm.corpus import PackageCorpus, PackageSourceText
from extraction.pipeline import run_ingest_pipeline
from extraction.types import SourceRecord


def _tyndall_base_pdf() -> Path | None:
    candidates = list(Path("data/runs").glob("**/sources/4f25a7b3ec56735cc9a5e9ff1517af2f1f8fe07fd1d2736a20b553b0e44ffc6b.pdf"))
    return candidates[0] if candidates else None


class PdfFormExtractionTests(unittest.TestCase):
    def test_acroform_naics_field_extraction(self) -> None:
        pdf_path = _tyndall_base_pdf()
        if not pdf_path:
            self.skipTest("Tyndall base solicitation PDF not available locally")
        records, _ = extract_pdf_form_fields(
            pdf_path.read_bytes(),
            source_id="base",
            filename="Solicitation___FA481926R0001.pdf",
        )
        naics = next((item for item in records if "naics" in str(item.get("field_name")).lower()), None)
        self.assertIsNotNone(naics)
        self.assertEqual(naics["field_value"], "722310")
        self.assertEqual(naics["provenance"], "acroform")
        self.assertEqual(naics["page"], 1)

    def test_selected_checkbox_only_in_corpus(self) -> None:
        pdf_path = _tyndall_base_pdf()
        if not pdf_path:
            self.skipTest("Tyndall base solicitation PDF not available locally")
        records, _ = extract_pdf_form_fields(
            pdf_path.read_bytes(),
            source_id="base",
            filename="Solicitation___FA481926R0001.pdf",
        )
        selected = [item for item in records if item.get("field_type") == "checkbox"]
        self.assertTrue(selected)
        self.assertTrue(all(item.get("selected") is True for item in selected))
        block = format_form_fields_corpus_block(records)
        self.assertIn("722310", block)
        self.assertNotIn("/Off", block)

    def test_form_fields_preserve_provenance(self) -> None:
        pdf_path = _tyndall_base_pdf()
        if not pdf_path:
            self.skipTest("Tyndall base solicitation PDF not available locally")
        records, _ = extract_pdf_form_fields(
            pdf_path.read_bytes(),
            source_id="sha-base",
            filename="Solicitation___FA481926R0001.pdf",
        )
        sol = next(item for item in records if "solicitationnumber" in item["field_name"].lower())
        self.assertEqual(sol["source_id"], "sha-base")
        self.assertEqual(sol["filename"], "Solicitation___FA481926R0001.pdf")
        self.assertEqual(sol["field_value"], "FA481926R0001")

    def test_package_corpus_includes_form_block_and_naics(self) -> None:
        pdf_path = _tyndall_base_pdf()
        if not pdf_path:
            self.skipTest("Tyndall base solicitation PDF not available locally")
        source = SourceRecord(
            key="runs/test/sources/base.pdf",
            ext="pdf",
            bytes=pdf_path.stat().st_size,
            sha256="4f25a7b3ec56735cc9a5e9ff1517af2f1f8fe07fd1d2736a20b553b0e44ffc6b",
            original_filename="Solicitation___FA481926R0001.pdf",
            abs_path=str(pdf_path.resolve()),
            did_dedupe=False,
        )
        package_source, _ = extract_package_source_text(source, apply_ocr=False)
        corpus = build_package_corpus([source], apply_ocr=False)
        self.assertIn("--- PDF FORM FIELDS ---", corpus.corpus_text)
        self.assertIn("722310", corpus.corpus_text)
        self.assertIn("NORTH AMERICAN INDUSTRY CLASSIFICATION STANDARD (NAICS)", corpus.corpus_text)
        self.assertGreater(len(package_source.form_fields), 0)

    def test_native_form_preferred_over_ocr_when_form_present(self) -> None:
        self.assertFalse(
            form_page_needs_render_fallback(
                native_page_text="SOLICITATION/CONTRACT/ORDER FOR COMMERCIAL PRODUCTS",
                form_records=[{"field_name": "10naics", "field_value": "722310"}],
            )
        )


class NaicsValidationTests(unittest.TestCase):
    def test_clause_naics_562910_rejected(self) -> None:
        source = PackageSourceText(
            source_id="src1",
            filename="Clause.pdf",
            file_type="pdf",
            document_type_hint="attachment",
        )
        source.normalized_text_by_page[0] = "FAR 52.212-1 environmental remediation NAICS 562910 prior contract example"
        corpus = PackageCorpus(sources=[source], corpus_text="")
        signal = {
            "id": "rfp_primary_naics_v1",
            "value": "562910",
            "confidence": "high",
            "status": "confirmed",
            "evidence": [
                {
                    "source_id": "src1",
                    "filename": "Clause.pdf",
                    "page": 1,
                    "quote": "environmental remediation NAICS 562910 prior contract",
                }
            ],
            "alternates": [],
        }
        validated, findings = validate_signal_record(signal, corpus)
        self.assertTrue(any(item.code == "NAICS_CLAUSE_EXAMPLE_REJECTED" for item in findings))
        self.assertEqual(validated["status"], "review_required")

    def test_form_naics_722310_validates(self) -> None:
        source = PackageSourceText(
            source_id="base",
            filename="Solicitation.pdf",
            file_type="pdf",
            document_type_hint="base solicitation",
        )
        source.normalized_text_by_page[0] = (
            "--- PDF FORM FIELDS ---\n"
            "field: NORTH AMERICAN INDUSTRY CLASSIFICATION STANDARD (NAICS)\n"
            "value: 722310\npage: 1\nfield_name: 10naics\nprovenance: acroform"
        )
        corpus = PackageCorpus(sources=[source], corpus_text="")
        signal = {
            "id": "rfp_primary_naics_v1",
            "value": "722310",
            "confidence": "high",
            "status": "confirmed",
            "evidence": [
                {
                    "source_id": "base",
                    "filename": "Solicitation.pdf",
                    "page": 1,
                    "quote": "NORTH AMERICAN INDUSTRY CLASSIFICATION STANDARD (NAICS) value: 722310",
                }
            ],
            "alternates": [],
        }
        validated, findings = validate_signal_record(signal, corpus)
        self.assertEqual(validated["status"], "confirmed")
        self.assertFalse(any(item.code == "NAICS_CLAUSE_EXAMPLE_REJECTED" for item in findings))


class SolicitationIdCanonicalizationTests(unittest.TestCase):
    def test_base_id_with_amendment_variants(self) -> None:
        base, structured = canonical_base_solicitation_id(
            {
                "base_solicitation_id": "FA481926R0001",
                "variants": ["FA4819-26-R-0001", "FA481926R00010005"],
            }
        )
        self.assertEqual(base, "FA481926R0001")
        self.assertIsNotNone(structured)

    def test_hyphenated_variant_canonicalizes(self) -> None:
        base, _ = canonical_base_solicitation_id("FA4819-26-R-0001")
        self.assertEqual(base, "FA481926R0001")

    def test_amendment_suffix_stripped(self) -> None:
        self.assertEqual(strip_amendment_suffix("FA481926R00010005"), "FA481926R0001")

    def test_amendment_suffix_cannot_become_canonical_base(self) -> None:
        base, _ = canonical_base_solicitation_id("FA481926R00010005")
        self.assertEqual(base, "FA481926R0001")

    def test_solicitation_id_with_variant_validates(self) -> None:
        source = PackageSourceText(
            source_id="base",
            filename="Solicitation.pdf",
            file_type="pdf",
            document_type_hint="base solicitation",
        )
        source.normalized_text_by_page[0] = "field: SOLICITATION NUMBER\nvalue: FA481926R0001"
        corpus = PackageCorpus(sources=[source], corpus_text="")
        signal = {
            "id": "rfp_solicitation_id_v1",
            "value": "FA4819-26-R-0001",
            "confidence": "high",
            "status": "confirmed",
            "evidence": [
                {
                    "source_id": "base",
                    "filename": "Solicitation.pdf",
                    "page": 1,
                    "quote": "SOLICITATION NUMBER value: FA481926R0001",
                }
            ],
            "alternates": [],
        }
        validated, findings = validate_signal_record(signal, corpus)
        self.assertEqual(validated["status"], "confirmed")
        self.assertFalse(any(item.code == "SOLICITATION_ID_AMENDMENT_SUFFIX" for item in findings))

    def test_prior_piid_not_used_as_solicitation_id(self) -> None:
        source = PackageSourceText(
            source_id="src1",
            filename="Doc.pdf",
            file_type="pdf",
            document_type_hint="attachment",
        )
        source.normalized_text_by_page[0] = "Prior Contract PIID: FA481924C0006"
        corpus = PackageCorpus(sources=[source], corpus_text="")
        signal = {
            "id": "rfp_solicitation_id_v1",
            "value": "FA481924C0006",
            "confidence": "high",
            "status": "confirmed",
            "evidence": [
                {
                    "source_id": "src1",
                    "filename": "Doc.pdf",
                    "page": 1,
                    "quote": "Prior Contract PIID: FA481924C0006",
                }
            ],
            "alternates": [],
        }
        validated, findings = validate_signal_record(signal, corpus)
        self.assertTrue(any(item.code == "SOLICITATION_ID_PRIOR_PIID" for item in findings))


class CacheKeyTests(unittest.TestCase):
    def test_corpus_version_in_cache_key(self) -> None:
        from extraction.package_llm.cache import cache_key
        from extraction.package_llm.versions import PROFILE_VERSIONS

        source = SourceRecord(
            key="k",
            ext="txt",
            bytes=1,
            sha256="abc",
            original_filename="a.txt",
            abs_path="a.txt",
        )
        original = PROFILE_VERSIONS["package_llm_full"]["corpusBuilderVersion"]
        PROFILE_VERSIONS["package_llm_full"]["corpusBuilderVersion"] = "package_corpus.v1"
        try:
            with patch.dict("os.environ", {"GOVCON_EXTRACTION_PROFILE": "package_llm_full"}, clear=False):
                old = cache_key([source])
        finally:
            PROFILE_VERSIONS["package_llm_full"]["corpusBuilderVersion"] = original
        PROFILE_VERSIONS["package_llm_full"]["corpusBuilderVersion"] = "package_corpus.v2"
        try:
            with patch.dict("os.environ", {"GOVCON_EXTRACTION_PROFILE": "package_llm_full"}, clear=False):
                new = cache_key([source])
        finally:
            PROFILE_VERSIONS["package_llm_full"]["corpusBuilderVersion"] = original
        self.assertNotEqual(old, new)


class TyndallFormCorpusPipelineTests(unittest.TestCase):
    @patch("extraction.package_llm.run.load_cached_artifacts", return_value=None)
    @patch("extraction.package_llm.run.call_package_extraction")
    @patch("extraction.package_llm.run.has_openai_api_key", return_value=True)
    def test_tyndall_base_pdf_form_fields_flow_to_validated_naics(self, _mock_key: object, mock_extract: object, _mock_cache: object) -> None:
        pdf_path = _tyndall_base_pdf()
        if not pdf_path:
            self.skipTest("Tyndall base solicitation PDF not available locally")
        source_id = "4f25a7b3ec56735cc9a5e9ff1517af2f1f8fe07fd1d2736a20b553b0e44ffc6b"
        filename = "Solicitation___FA481926R0001.pdf"

        def _response(*_args, **_kwargs):
            return (
                {
                    "signals": [
                        {
                            "id": "rfp_primary_naics_v1",
                            "value": "722310",
                            "confidence": "high",
                            "status": "confirmed",
                            "controlling_source": {"source_id": source_id, "filename": filename, "page": 1},
                            "evidence": [
                                {
                                    "source_id": source_id,
                                    "filename": filename,
                                    "page": 1,
                                    "quote": "field: NORTH AMERICAN INDUSTRY CLASSIFICATION STANDARD (NAICS) value: 722310",
                                }
                            ],
                            "reasoning_summary": "Primary NAICS from SF1449 form field",
                            "alternates": [],
                        },
                        {
                            "id": "rfp_solicitation_id_v1",
                            "value": "FA4819-26-R-0001",
                            "confidence": "high",
                            "status": "confirmed",
                            "controlling_source": {"source_id": source_id, "filename": filename, "page": 1},
                            "evidence": [
                                {
                                    "source_id": source_id,
                                    "filename": filename,
                                    "page": 1,
                                    "quote": "field: SOLICITATION NUMBER value: FA481926R0001",
                                }
                            ],
                            "reasoning_summary": "Base solicitation number from SF1449 form field",
                            "alternates": [],
                        },
                    ],
                    "package_summary": {"base_solicitation_filename": filename, "amendments_in_order": [], "controlling_amendment": None},
                },
                {"model": "gpt-5.5", "inputTokens": 100, "outputTokens": 100},
                [],
            )

        mock_extract.side_effect = _response
        with patch("extraction.pipeline.get_extraction_mode", return_value="package_llm"):
            result = run_ingest_pipeline([pdf_path], extract_text=True, extract_signals=True, resolve_signals=True)

        from extraction.persist import read_json
        from extraction.config import resolved_signals_json_path

        resolved = read_json(resolved_signals_json_path(result.run_id))
        by_id = {item["id"]: item for item in resolved.get("signals") or []}
        self.assertEqual(by_id["rfp_primary_naics_v1"]["canonical_value"], "722310")
        self.assertEqual(by_id["rfp_primary_naics_v1"]["resolution_status"], "validated_model_extraction")
        self.assertEqual(by_id["rfp_solicitation_id_v1"]["canonical_value"], "FA481926R0001")
        self.assertEqual(by_id["rfp_solicitation_id_v1"]["resolution_status"], "validated_model_extraction")


if __name__ == "__main__":
    unittest.main()
