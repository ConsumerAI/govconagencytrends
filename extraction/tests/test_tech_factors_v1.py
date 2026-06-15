from __future__ import annotations

import unittest

from extraction.docset.merge import merge_solicitation_set_signals
from extraction.documents.classify import TYPE_AMENDMENT, TYPE_BASE_SOLICITATION
from extraction.parity.compare import compare_resolved_signals
from extraction.postprocess.run import run_postprocess
from extraction.resolve.resolve_signals import resolve_signals_v1
from extraction.signals.evaluation_extract import extract_evaluation_signals_v1
from extraction.tests.test_milestone5 import _doc_entry
from extraction.tests.test_structure_due_date import _page, _signal_value


def _tech_signal(value: str, *, source_hint: str = "sectionMFulltext", amendment: str | None = None) -> dict:
    ev1 = {"excerpt": value, "source": "solicitation.pdf", "spanHashes": []}
    if amendment:
        ev1["amendmentNumber"] = amendment
    return {
        "id": "rfp_tech_factors_v1",
        "value": value,
        "confidence": "high",
        "evidence": [{"sourceId": "solicitation.pdf", "artifact": "text", "locator": "p:0", "snippet": value[:120]}],
        "evidence_v1": ev1,
        "authority": {"tier": 2, "label": "SECTION_LM", "reason": "Section M fulltext"},
    }


class TechFactorsV1Tests(unittest.TestCase):
    def test_deterministic_extraction(self) -> None:
        text = "SECTION M\nFactor 1 Technical Approach includes staffing and methodology."
        pages = [_page(text)]
        from extraction.artifacts.build import build_document_artifacts

        artifacts = build_document_artifacts("run_tf", pages, filename="solicitation.pdf")
        signals, _ = extract_evaluation_signals_v1(
            windows=[],
            structure=artifacts["structure"],
            source_filename="solicitation.pdf",
            run_id="run_tf",
        )
        self.assertIsNotNone(_signal_value(signals, "rfp_tech_factors_v1"))

    def test_resolver_preservation(self) -> None:
        raw = [_tech_signal("Factor 1 Technical Approach")]
        processed, _ = run_postprocess(raw, corpus_text="")
        artifact = resolve_signals_v1("run_tf", processed)
        by_id = {item["id"]: item for item in artifact["signals"]}
        self.assertIn("rfp_tech_factors_v1", by_id)
        self.assertIn("Technical Approach", str(by_id["rfp_tech_factors_v1"]["canonical_value"]))

    def test_amendment_replacement(self) -> None:
        documents = [
            _doc_entry(
                doc_key="base",
                file_name="solicitation.pdf",
                text="SECTION M Factor 1 Technical legacy",
                document_type=TYPE_BASE_SOLICITATION,
                is_amendment=False,
                amendment_order=None,
            ),
        ]
        documents[0]["signals"].append(_tech_signal("Factor 1 Technical legacy"))
        documents.append(
            {
                **_doc_entry(
                    doc_key="a4",
                    file_name="amendment_0004.pdf",
                    text="Amendment 0004 Factor 1 Technical revised",
                    document_type=TYPE_AMENDMENT,
                    is_amendment=True,
                    amendment_order="0004",
                ),
                "signals": [_tech_signal("Factor 1 Technical revised", amendment="0004")],
            }
        )
        merged, _, _, _ = merge_solicitation_set_signals(documents)
        by_id = {item["id"]: item for item in merged}
        self.assertIn("revised", str(by_id["rfp_tech_factors_v1"]["value"]).lower())

    def test_ambiguity_conflict_review_required(self) -> None:
        raw = [
            _tech_signal("Factor 1 Technical"),
            {**_tech_signal("Factor 1 Management"), "confidence": "high"},
        ]
        artifact = resolve_signals_v1("run_tf", raw)
        record = next(item for item in artifact["signals"] if item["id"] == "rfp_tech_factors_v1")
        self.assertEqual(record["resolution_status"], "resolved_with_conflict")
        self.assertTrue(record.get("alternates"))

    def test_parity_comparison_includes_tech_factors(self) -> None:
        golden = {
            "version": "resolved_signals.v1",
            "runId": "golden",
            "signals": [
                {
                    "id": "rfp_tech_factors_v1",
                    "canonical_value": "Factor 1 Technical Approach",
                    "canonical_confidence": "high",
                    "resolution_status": "passthrough",
                    "evidence": {"legacy": [], "evidence_v1": {"excerpt": "Factor 1 Technical Approach", "source": "s.pdf"}},
                    "alternates": [],
                }
            ],
        }
        local = dict(golden)
        local["runId"] = "local"
        report = compare_resolved_signals(
            golden_payload=golden,
            local_payload=local,
            signal_ids=("rfp_tech_factors_v1",),
        )
        self.assertEqual(report["signals"][0]["comparison"], "exact match")


if __name__ == "__main__":
    unittest.main()
