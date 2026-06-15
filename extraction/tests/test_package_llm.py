from __future__ import annotations

import unittest
from pathlib import Path

from extraction.package_llm.corpus import (
    PackageCorpus,
    PackageSourceText,
    assemble_package_corpus,
    quote_in_source,
)
from extraction.package_llm.validate import validate_package_extraction
from extraction.types import DocumentPage, SourceRecord


def _source_record(source_id: str, filename: str, text: str) -> SourceRecord:
    return SourceRecord(
        key=f"runs/test/sources/{source_id}.txt",
        ext="txt",
        bytes=len(text.encode()),
        sha256=source_id,
        original_filename=filename,
        abs_path=str(Path(filename)),
    )


def _package_corpus_from_pages(source_id: str, filename: str, pages: list[str]) -> PackageCorpus:
    package_source = PackageSourceText(
        source_id=source_id,
        filename=filename,
        file_type="txt",
        document_type_hint="base solicitation",
        pages=[DocumentPage(page_index=i, text=text, char_count=len(text)) for i, text in enumerate(pages)],
    )
    for page in package_source.pages:
        package_source.normalized_text_by_page[page.page_index] = page.text
    package_source.char_count = sum(page.char_count for page in package_source.pages)
    corpus_text = assemble_package_corpus([package_source])
    return PackageCorpus(sources=[package_source], corpus_text=corpus_text)


class PackageCorpusTests(unittest.TestCase):
    def test_corpus_preserves_file_and_page_boundaries(self) -> None:
        corpus = _package_corpus_from_pages("abc123", "Solicitation.pdf", ["Page one NAICS 722310", "Page two"])
        self.assertIn("===== SOURCE BEGIN =====", corpus.corpus_text)
        self.assertIn("filename: Solicitation.pdf", corpus.corpus_text)
        self.assertIn("--- PAGE 1 ---", corpus.corpus_text)
        self.assertIn("722310", corpus.corpus_text)

    def test_quote_lookup_matches_page_text(self) -> None:
        corpus = _package_corpus_from_pages("abc123", "Solicitation.pdf", ["NORTH AMERICAN INDUSTRY CLASSIFICATION STANDARD (NAICS): 722310"])
        source = corpus.sources[0]
        self.assertTrue(quote_in_source(source, page=1, sheet=None, quote="NAICS): 722310"))


class PackageValidatorTests(unittest.TestCase):
    def _validate(self, corpus: PackageCorpus, signal: dict) -> dict:
        payload = {"signals": [signal], "package_summary": {}}
        validated, _ = validate_package_extraction(payload, corpus)
        return validated["signals"][0]

    def test_naics_clause_language_rejected(self) -> None:
        corpus = _package_corpus_from_pages(
            "src1",
            "Clause.pdf",
            ["FAR 52.212-1 environmental remediation NAICS 562910 prior contract example"],
        )
        signal = {
            "id": "rfp_primary_naics_v1",
            "value": "562910",
            "confidence": "high",
            "status": "confirmed",
            "controlling_source": {"source_id": "src1", "filename": "Clause.pdf", "page": 1, "sheet": None, "amendment_number": None},
            "evidence": [
                {
                    "source_id": "src1",
                    "filename": "Clause.pdf",
                    "page": 1,
                    "sheet": None,
                    "quote": "environmental remediation NAICS 562910 prior contract",
                }
            ],
            "reasoning_summary": "Found NAICS in clause",
            "alternates": [],
        }
        validated = self._validate(corpus, signal)
        self.assertEqual(validated["status"], "review_required")
        codes = {item["code"] for item in validated.get("validation_findings") or []}
        self.assertIn("NAICS_CLAUSE_EXAMPLE_REJECTED", codes)

    def test_unselected_unrestricted_label_rejected(self) -> None:
        corpus = _package_corpus_from_pages(
            "src2",
            "SF1449.pdf",
            ["UNRESTRICTED  FULL AND OPEN  PARTIAL SMALL BUSINESS"],
        )
        signal = {
            "id": "rfp_set_aside_v1",
            "value": "Full and Open",
            "confidence": "high",
            "status": "confirmed",
            "controlling_source": {"source_id": "src2", "filename": "SF1449.pdf", "page": 1, "sheet": None, "amendment_number": None},
            "evidence": [
                {
                    "source_id": "src2",
                    "filename": "SF1449.pdf",
                    "page": 1,
                    "sheet": None,
                    "quote": "UNRESTRICTED  FULL AND OPEN",
                }
            ],
            "reasoning_summary": "Checkbox label",
            "alternates": [],
        }
        validated = self._validate(corpus, signal)
        self.assertEqual(validated["status"], "review_required")
        codes = {item["code"] for item in validated.get("validation_findings") or []}
        self.assertIn("SET_ASIDE_CHECKBOX_ONLY", codes)

    def test_amendment_effective_date_rejected_as_due_date(self) -> None:
        corpus = _package_corpus_from_pages(
            "src3",
            "Amendment.pdf",
            ["Amendment effective date: 08 April 2026"],
        )
        signal = {
            "id": "rfp_questions_due_v1",
            "value": "2026-04-08",
            "confidence": "high",
            "status": "confirmed",
            "controlling_source": {"source_id": "src3", "filename": "Amendment.pdf", "page": 1, "sheet": None, "amendment_number": "0005"},
            "evidence": [
                {
                    "source_id": "src3",
                    "filename": "Amendment.pdf",
                    "page": 1,
                    "sheet": None,
                    "quote": "Amendment effective date: 08 April 2026",
                }
            ],
            "reasoning_summary": "Effective date",
            "alternates": [],
        }
        validated = self._validate(corpus, signal)
        self.assertEqual(validated["status"], "review_required")
        codes = {item["code"] for item in validated.get("validation_findings") or []}
        self.assertIn("DUE_DATE_AMENDMENT_EFFECTIVE", codes)

    def test_poc_email_rejected_as_submission_destination(self) -> None:
        corpus = _package_corpus_from_pages(
            "src4",
            "Solicitation.pdf",
            ["Primary POC: contracting.tyndall@example.mil"],
        )
        signal = {
            "id": "rfp_submission_destination_v1",
            "value": "contracting.tyndall@example.mil",
            "confidence": "high",
            "status": "confirmed",
            "controlling_source": {"source_id": "src4", "filename": "Solicitation.pdf", "page": 1, "sheet": None, "amendment_number": None},
            "evidence": [
                {
                    "source_id": "src4",
                    "filename": "Solicitation.pdf",
                    "page": 1,
                    "sheet": None,
                    "quote": "Primary POC: contracting.tyndall@example.mil",
                }
            ],
            "reasoning_summary": "Email found",
            "alternates": [],
        }
        validated = self._validate(corpus, signal)
        self.assertEqual(validated["status"], "review_required")
        codes = {item["code"] for item in validated.get("validation_findings") or []}
        self.assertIn("SUBMISSION_POC_EMAIL", codes)

    def test_prior_piid_rejected_as_solicitation_id(self) -> None:
        corpus = _package_corpus_from_pages(
            "src5",
            "Solicitation.pdf",
            ["Prior contract FA481924C0006 under this requirement"],
        )
        signal = {
            "id": "rfp_solicitation_id_v1",
            "value": "FA481924C0006",
            "confidence": "high",
            "status": "confirmed",
            "controlling_source": {"source_id": "src5", "filename": "Solicitation.pdf", "page": 1, "sheet": None, "amendment_number": None},
            "evidence": [
                {
                    "source_id": "src5",
                    "filename": "Solicitation.pdf",
                    "page": 1,
                    "sheet": None,
                    "quote": "Prior contract FA481924C0006",
                }
            ],
            "reasoning_summary": "Prior PIID",
            "alternates": [],
        }
        validated = self._validate(corpus, signal)
        self.assertEqual(validated["status"], "review_required")
        codes = {item["code"] for item in validated.get("validation_findings") or []}
        self.assertIn("SOLICITATION_ID_PRIOR_PIID", codes)

    def test_valid_naics_confirmed(self) -> None:
        corpus = _package_corpus_from_pages(
            "src6",
            "Solicitation.pdf",
            ["NORTH AMERICAN INDUSTRY CLASSIFICATION STANDARD (NAICS): 722310"],
        )
        signal = {
            "id": "rfp_primary_naics_v1",
            "value": "722310",
            "confidence": "high",
            "status": "confirmed",
            "controlling_source": {"source_id": "src6", "filename": "Solicitation.pdf", "page": 1, "sheet": None, "amendment_number": None},
            "evidence": [
                {
                    "source_id": "src6",
                    "filename": "Solicitation.pdf",
                    "page": 1,
                    "sheet": None,
                    "quote": "NORTH AMERICAN INDUSTRY CLASSIFICATION STANDARD (NAICS): 722310",
                }
            ],
            "reasoning_summary": "Acquisition NAICS on solicitation form",
            "alternates": [],
        }
        validated = self._validate(corpus, signal)
        self.assertEqual(validated["status"], "confirmed")


class PackageLLMResolveTests(unittest.TestCase):
    def test_resolve_preserves_validated_model_value(self) -> None:
        from extraction.package_llm.resolve import resolve_validated_package_signals

        validated = {
            "signals": [
                {
                    "id": "rfp_primary_naics_v1",
                    "value": "722310",
                    "confidence": "high",
                    "status": "confirmed",
                    "controlling_source": None,
                    "evidence": [],
                    "reasoning_summary": "Acquisition NAICS",
                    "alternates": [],
                    "validation_findings": [],
                }
            ],
            "package_summary": {},
        }
        resolved = resolve_validated_package_signals(
            "run_test",
            validated,
            process_sources_signal={"id": "process_sources_v1", "value": "{}", "confidence": "high", "evidence": []},
        )
        by_id = {item["id"]: item for item in resolved["signals"]}
        self.assertEqual(by_id["rfp_primary_naics_v1"]["canonical_value"], "722310")
        self.assertEqual(by_id["rfp_primary_naics_v1"]["resolution_status"], "validated_model_extraction")

    def test_resolve_canonicalizes_structured_office_dict(self) -> None:
        from extraction.package_llm.resolve import resolve_validated_package_signals

        validated = {
            "signals": [
                {
                    "id": "rfp_issuing_office_v1",
                    "value": {"office": "325 CONS PKP", "address": "Tyndall AFB, FL 32403"},
                    "confidence": "high",
                    "status": "confirmed",
                    "controlling_source": None,
                    "evidence": [],
                    "reasoning_summary": "Issuing office block",
                    "alternates": [],
                    "validation_findings": [],
                }
            ],
            "package_summary": {},
        }
        resolved = resolve_validated_package_signals(
            "run_test",
            validated,
            process_sources_signal={"id": "process_sources_v1", "value": "{}", "confidence": "high", "evidence": []},
        )
        by_id = {item["id"]: item for item in resolved["signals"]}
        self.assertEqual(by_id["rfp_issuing_office_v1"]["canonical_value"], "325 CONS PKP — Tyndall AFB, FL 32403")
        self.assertEqual(
            by_id["rfp_issuing_office_v1"]["package_llm"]["structuredDetail"],
            {"office": "325 CONS PKP", "address": "Tyndall AFB, FL 32403"},
        )

    def test_resolve_summary_tags_package_producer(self) -> None:
        from extraction.package_llm.resolve import resolve_validated_package_signals

        resolved = resolve_validated_package_signals(
            "run_test",
            {"signals": [], "package_summary": {}},
            process_sources_signal={"id": "process_sources_v1", "value": "{}", "confidence": "high", "evidence": []},
        )
        self.assertEqual(resolved["summary"]["producer"], "package_llm")
        self.assertEqual(resolved["summary"]["extractionMode"], "package_llm")


class PackageCacheKeyTests(unittest.TestCase):
    def test_cache_key_includes_extraction_mode(self) -> None:
        from unittest.mock import patch

        from extraction.package_llm.cache import cache_key

        source = _source_record("abc123", "sample.txt", "hello")
        with patch("extraction.package_llm.cache.get_extraction_mode", return_value="package_llm"):
            package_key = cache_key([source])
        with patch("extraction.package_llm.cache.get_extraction_mode", return_value="legacy"):
            legacy_key = cache_key([source])
        self.assertNotEqual(package_key, legacy_key)


if __name__ == "__main__":
    unittest.main()
