from __future__ import annotations

import unittest

from extraction.artifacts.build import build_document_artifacts
from extraction.postprocess.evaluation_canonicalize import canonicalize_evaluation_signals
from extraction.signals.evaluation_extract import (
    classify_eval_method,
    extract_evaluation_signals_v1,
)
from extraction.signals.extract_signals_v1 import extract_signals_v1
from extraction.tests.test_structure_due_date import _page, _signal_value


def _run_extract(text: str, *, filename: str = "solicitation.pdf") -> list[dict]:
    pages = [_page(text)]
    artifacts = build_document_artifacts("run_eval", pages, filename=filename)
    signals, _ = extract_signals_v1(
        run_id="run_eval",
        pages=pages,
        structure=artifacts["structure"],
        sections=artifacts["sections"],
        clauses=artifacts["clauses"],
        section_l_fulltext=artifacts["sectionLFulltext"],
        section_m_fulltext=artifacts["sectionMFulltext"],
        source_filename=filename,
    )
    return signals


class EvaluationCanonicalizationTests(unittest.TestCase):
    def test_lpta_beats_vague_best_value(self) -> None:
        text = (
            "SECTION M\n"
            "Award will be made on a best value basis in accordance with FAR 15.\n"
            "This acquisition uses lowest price technically acceptable procedures."
        )
        method, ambiguous, _ = classify_eval_method(text)
        self.assertEqual(method, "LPTA")
        self.assertFalse(ambiguous)
        signals = _run_extract(text)
        self.assertEqual(_signal_value(signals, "rfp_eval_method_v1"), "LPTA")

    def test_lpta_tradeoff_conflict_review_required(self) -> None:
        text = (
            "SECTION M\n"
            "The Government will conduct a tradeoff among offerors.\n"
            "Lowest price technically acceptable evaluation applies."
        )
        method, ambiguous, _ = classify_eval_method(text)
        self.assertTrue(ambiguous)
        canonicalized, findings = canonicalize_evaluation_signals(
            [
                {"id": "rfp_eval_method_v1", "value": "LPTA", "confidence": "high", "evidence_v1": {"excerpt": text, "source": "a.pdf"}},
                {"id": "rfp_eval_method_v1", "value": "Tradeoff", "confidence": "high", "evidence_v1": {"excerpt": text, "source": "b.pdf"}},
            ]
        )
        codes = {item.code for item in findings}
        self.assertIn("EVALUATION_CANONICALIZATION_AMBIGUOUS", codes)

    def test_amended_section_m_replaces_original(self) -> None:
        base = (
            "SECTION M\nEvaluation Factor 1 Technical\nEvaluation Factor 2 Management\n"
            "Relative importance: technical is significantly more important than price."
        )
        amend = (
            "STANDARD FORM 30 Amendment 0004\n"
            "SECTION M is replaced in its entirety.\n"
            "Evaluation Factor 1 Technical\nEvaluation Factor 3 Past Performance\n"
            "Relative importance: all factors are of equal importance."
        )
        base_signals, _ = extract_evaluation_signals_v1(
            windows=[],
            structure=build_document_artifacts("run_eval", [_page(base)])["structure"],
            source_filename="solicitation.pdf",
            run_id="run_eval",
        )
        amend_signals, _ = extract_evaluation_signals_v1(
            windows=[],
            structure=build_document_artifacts("run_eval", [_page(amend)], amendment_number="0004")["structure"],
            source_filename="amendment_0004.pdf",
            run_id="run_eval",
            amendment_number="0004",
        )
        base_weights = _signal_value(base_signals, "rfp_eval_weights_v1")
        amend_weights = _signal_value(amend_signals, "rfp_eval_weights_v1")
        self.assertIn("significantly more important", str(base_weights).lower())
        self.assertIn("equal importance", str(amend_weights).lower())

    def test_past_performance_advisory_not_evaluated(self) -> None:
        text = (
            "SECTION M\nPast performance is advisory and not evaluated.\n"
            "Evaluation Factor 1 Technical Approach."
        )
        signals = _run_extract(text)
        criteria = next(item for item in signals if item["id"] == "rfp_evaluation_criteria_v1")
        meta = (criteria.get("evidence_v1") or {}).get("evaluationMetadata") or {}
        self.assertIn("pastPerformanceRole", meta)
        self.assertIn("advisory", meta["pastPerformanceRole"].lower())

    def test_pass_fail_factor_separation(self) -> None:
        text = (
            "SECTION M\n"
            "Factor 1 Technical Approach is pass/fail.\n"
            "Factor 2 Management is rated using adjectival ratings."
        )
        signals = _run_extract(text)
        criteria = next(item for item in signals if item["id"] == "rfp_evaluation_criteria_v1")
        meta = (criteria.get("evidence_v1") or {}).get("evaluationMetadata") or {}
        self.assertIn("passFailLanguage", meta)
        self.assertIn("ratingStyle", meta)

    def test_relative_importance_ambiguity_low_confidence(self) -> None:
        text = "SECTION M\nEvaluation factors will be assessed. Importance may vary by subfactor."
        signals = _run_extract(text)
        method = next((item for item in signals if item["id"] == "rfp_eval_method_v1"), None)
        if method:
            self.assertIn(method.get("confidence"), {"low", "medium"})

    def test_qa_does_not_override_section_m(self) -> None:
        text = (
            "SECTION M\nLowest price technically acceptable evaluation applies.\n"
            "QUESTIONS AND ANSWERS\nQ: Is this a tradeoff? A: Offerors should assume best value tradeoff."
        )
        signals = _run_extract(text)
        self.assertEqual(_signal_value(signals, "rfp_eval_method_v1"), "LPTA")


if __name__ == "__main__":
    unittest.main()
