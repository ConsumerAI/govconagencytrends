from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

from extraction.persist import read_json
from extraction.postprocess.piid_backstop import inject_prior_contract_piid_backstop
from extraction.postprocess.run import run_postprocess
from extraction.resolve.resolve_signals import resolve_one_signal, resolve_signals_v1

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TYNDALL_SIGNALS = Path(__file__).resolve().parents[2] / "data" / "runs" / "run_m3-dotenv-test2" / "signals" / "signals.json"


def _load_tyndall_signals() -> list[dict]:
    if not TYNDALL_SIGNALS.exists():
        raise unittest.SkipTest(f"Missing Tyndall signals fixture: {TYNDALL_SIGNALS}")
    payload = read_json(TYNDALL_SIGNALS)
    if not isinstance(payload, list):
        raise AssertionError("signals.json must be a list")
    return payload


class Milestone4ResolverTests(unittest.TestCase):
    def test_tyndall_resolved_core_fields(self) -> None:
        raw = _load_tyndall_signals()
        processed, _ = run_postprocess(raw, corpus_text="")
        artifact = resolve_signals_v1("run_test", processed)
        by_id = {item["id"]: item for item in artifact["signals"]}

        self.assertEqual(artifact["version"], "resolved_signals.v1")
        self.assertEqual(by_id["rfp_solicitation_id_v1"]["canonical_value"], "FA481926R0001")
        self.assertEqual(by_id["rfp_solicitation_number_v1"]["canonical_value"], "FA481926R0001")
        self.assertEqual(by_id["rfp_issuing_agency_v1"]["canonical_value"], "Department of the Air Force")
        self.assertEqual(by_id["rfp_office_aac_v1"]["canonical_value"], "FA4819")
        self.assertEqual(by_id["rfp_primary_naics_v1"]["canonical_value"], "722310")
        self.assertEqual(by_id["rfp_primary_psc_v1"]["canonical_value"], "S203")
        self.assertEqual(by_id["rfp_contract_type_v1"]["canonical_value"], "Firm Fixed Price (FFP)")
        self.assertIn("Tyndall AFB", str(by_id["rfp_place_of_performance_v1"]["canonical_value"]))
        self.assertEqual(by_id["rfp_pop_start_v1"]["canonical_value"], "01 Oct 2026")
        self.assertEqual(by_id["rfp_pop_end_v1"]["canonical_value"], "30 Sep 2031")
        self.assertIn("Period of Performance", str(by_id["rfp_period_of_performance_v1"]["canonical_value"]))
        self.assertEqual(by_id["rfp_prior_contract_piid_v1"]["canonical_value"], "FA481924C0006")
        self.assertEqual(by_id["rfp_competition_type_v1"]["canonical_value"], "Competitive")

    def test_conflict_resolution_prefers_labeled_evidence(self) -> None:
        candidates = [
            {
                "id": "rfp_primary_naics_v1",
                "value": "722310",
                "confidence": "high",
                "evidence": [
                    {
                        "sourceId": "form",
                        "artifact": "text",
                        "locator": "page:1",
                        "snippet": "NAICS: 722310",
                    }
                ],
                "evidence_v1": {"source": "form", "excerpt": "NAICS: 722310", "spanHashes": []},
            },
            {
                "id": "rfp_primary_naics_v1",
                "value": "541511",
                "confidence": "medium",
                "evidence": [
                    {
                        "sourceId": "narrative",
                        "artifact": "text",
                        "locator": "page:9",
                        "snippet": "support services similar to NAICS 541511",
                    }
                ],
                "evidence_v1": {"source": "narrative", "excerpt": "support services similar to NAICS 541511", "spanHashes": []},
            },
        ]
        record = resolve_one_signal("rfp_primary_naics_v1", candidates, candidates)
        assert record is not None
        self.assertEqual(record["canonical_value"], "722310")
        self.assertEqual(record["resolution_status"], "resolved_with_conflict")
        self.assertEqual(len(record["alternates"]), 1)

    def test_equivalent_duplicate_collapse(self) -> None:
        candidates = [
            {
                "id": "rfp_solicitation_id_v1",
                "value": "FA481926R0001",
                "confidence": "high",
                "evidence": [{"sourceId": "a", "artifact": "text", "locator": "page:1", "snippet": "FA481926R0001"}],
                "evidence_v1": {"source": "a", "excerpt": "FA481926R0001", "spanHashes": []},
            },
            {
                "id": "rfp_solicitation_id_v1",
                "value": "FA481926R0001",
                "confidence": "medium",
                "evidence": [{"sourceId": "b", "artifact": "text", "locator": "page:2", "snippet": "SOLICITATION NUMBER FA481926R0001"}],
                "evidence_v1": {"source": "b", "excerpt": "SOLICITATION NUMBER FA481926R0001", "spanHashes": []},
            },
        ]
        record = resolve_one_signal("rfp_solicitation_id_v1", candidates, candidates)
        assert record is not None
        self.assertEqual(record["resolution_status"], "resolved_equivalent_candidates")
        self.assertEqual(record["canonical_value"], "FA481926R0001")

    def test_piid_backstop_from_incumbent(self) -> None:
        signals = [
            {
                "id": "rfp_incumbent_data_v1",
                "value": "Trogon Group Services, LLC; FA481924C0006",
                "confidence": "high",
                "evidence": [
                    {
                        "sourceId": "incumbent",
                        "artifact": "text",
                        "locator": "page:1",
                        "snippet": "Prior Contract PIID: FA481924C0006",
                    }
                ],
                "evidence_v1": {"source": "incumbent", "excerpt": "Prior Contract PIID: FA481924C0006", "spanHashes": []},
            }
        ]
        merged, _ = inject_prior_contract_piid_backstop(signals, corpus_text="")
        artifact = resolve_signals_v1("run_test", merged)
        by_id = {item["id"]: item for item in artifact["signals"]}
        self.assertEqual(by_id["rfp_prior_contract_piid_v1"]["canonical_value"], "FA481924C0006")
        self.assertIn(by_id["rfp_prior_contract_piid_v1"]["canonical_confidence"], {"high", "medium", "low"})

    def test_scope_review_compatibility(self) -> None:
        raw = _load_tyndall_signals()
        processed, _ = run_postprocess(raw, corpus_text="")
        artifact = resolve_signals_v1("run_test", processed)
        repo_root = Path(__file__).resolve().parents[2]
        sys.path.insert(0, str(repo_root))
        import app

        loaded = app.load_resolved_signals_json(artifact)
        preview = app.build_solicitation_scope_preview(loaded)
        self.assertIn("Primary NAICS", preview)
        self.assertIn("722310", str(preview["Primary NAICS"]["value"]))


    def test_no_unsupported_values_invented(self) -> None:
        raw = [
            {
                "id": "rfp_solicitation_id_v1",
                "value": "FA481926R0001",
                "confidence": "high",
                "evidence": [{"sourceId": "form", "artifact": "text", "locator": "page:1", "snippet": "FA481926R0001"}],
                "evidence_v1": {"source": "form", "excerpt": "FA481926R0001", "spanHashes": []},
            }
        ]
        artifact = resolve_signals_v1("run_test", raw)
        ids = {item["id"] for item in artifact["signals"]}
        self.assertIn("rfp_solicitation_id_v1", ids)
        self.assertNotIn("rfp_prior_contract_piid_v1", ids)
        self.assertNotIn("rfp_period_of_performance_v1", ids)
        self.assertNotIn("rfp_primary_naics_v1", ids)


if __name__ == "__main__":
    unittest.main()
