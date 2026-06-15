from __future__ import annotations

import unittest

from extraction.docset.solicitation_lock import build_package_solicitation_lock
from extraction.signals.contract_type import classify_contract_type
from extraction.signals.incumbent import extract_incumbent_candidates
from extraction.signals.naics_psc import extract_naics_candidates, pick_primary_code
from extraction.signals.pop_extract import extract_pop_from_text
from extraction.signals.set_aside_competition import classify_procurement_phrase


class SetAsideCompetitionMatrixTests(unittest.TestCase):
    def _assert_pair(self, source: str, *, set_aside: str | None, competition: str | None) -> None:
        result = classify_procurement_phrase(source)
        self.assertIsNotNone(result, source)
        assert result is not None
        self.assertEqual(result.set_aside, set_aside, source)
        self.assertEqual(result.competition_type, competition, source)

    def test_competitive_8a_set_aside(self) -> None:
        self._assert_pair(
            "Competitive 8(a) set-aside",
            set_aside="8(a) Small Business Set-Aside",
            competition="Competitive",
        )

    def test_8a_sole_source(self) -> None:
        self._assert_pair("8(a) sole-source award", set_aside="8(a)", competition="Sole Source")

    def test_full_and_open(self) -> None:
        self._assert_pair("Full and Open Competition", set_aside=None, competition="Full and Open Competition")

    def test_hubzone_set_aside(self) -> None:
        self._assert_pair("HUBZone set-aside", set_aside="HUBZone Set-Aside", competition="Competitive")

    def test_sdvosb_sole_source(self) -> None:
        self._assert_pair(
            "SDVOSB sole source",
            set_aside="Service-Disabled Veteran-Owned Small Business Sole Source",
            competition="Sole Source",
        )

    def test_total_small_business(self) -> None:
        self._assert_pair(
            "Total Small Business Set-Aside",
            set_aside="Total Small Business Set-Aside",
            competition="Competitive",
        )


class SourceLockingTests(unittest.TestCase):
    def test_unrelated_attachment_does_not_control(self) -> None:
        documents = [
            {
                "sourceId": "base",
                "fileName": "solicitation.pdf",
                "documentType": "base_solicitation",
                "isAmendment": False,
                "solicitationNumber": "FA1234-26-R-0001",
                "signals": [{"id": "rfp_solicitation_id_v1", "value": "FA1234-26-R-0001"}],
            },
            {
                "sourceId": "attach",
                "fileName": "other_solicitation_attachment.pdf",
                "documentType": "attachment_exhibit",
                "isAmendment": False,
                "signals": [{"id": "rfp_solicitation_id_v1", "value": "W912DY-20-R-0009"}],
            },
            {
                "sourceId": "piid",
                "fileName": "incumbent.pdf",
                "documentType": "attachment_exhibit",
                "isAmendment": False,
                "signals": [{"id": "rfp_prior_contract_piid_v1", "value": "FA8671-19-D-0001"}],
            },
        ]
        lock = build_package_solicitation_lock(documents)
        self.assertEqual(lock.locked_solicitation_id, "FA1234-26-R-0001")
        self.assertTrue(any(item.get("reason") == "solicitation_mismatch" for item in lock.excluded_documents))


class NaicsPscTests(unittest.TestCase):
    def test_primary_naics_wins_over_attachment_example(self) -> None:
        text = (
            "NAICS CODE: 541511 primary requirement.\n"
            "Attachment A example NAICS CODE: 999999 for illustration only."
        )
        candidates = extract_naics_candidates(text)
        winner, conflicts = pick_primary_code(candidates)
        self.assertIsNotNone(winner)
        assert winner is not None
        self.assertEqual(winner.value, "541511")

    def test_conflicting_authoritative_naics(self) -> None:
        text = "Primary NAICS CODE: 541511\nPrimary NAICS CODE: 541512"
        candidates = extract_naics_candidates(text)
        winner, conflicts = pick_primary_code(candidates)
        self.assertIsNotNone(winner)
        self.assertGreaterEqual(len(conflicts), 1)


class ContractTypeHybridTests(unittest.TestCase):
    def test_hybrid_not_flattened_to_ffp(self) -> None:
        result = classify_contract_type("Contract Type: FFP for CLIN 0001 and T&M for CLIN 0002 hybrid acquisition")
        self.assertIsNotNone(result)
        assert result is not None
        self.assertTrue(result.is_hybrid)
        self.assertIn("Hybrid", result.value)


class PopTests(unittest.TestCase):
    def test_base_and_option_periods(self) -> None:
        text = (
            "Base Period: 12 months beginning 1 Oct 2026\n"
            "Option Period 1: 12 months\n"
            "POP start: 1 Oct 2026\n"
            "POP end: 30 Sep 2027"
        )
        pop = extract_pop_from_text(text)
        self.assertIsNotNone(pop.start)
        self.assertIsNotNone(pop.end)
        self.assertTrue(pop.option_periods)


class IncumbentAmbiguityTests(unittest.TestCase):
    def test_explicit_incumbent_only(self) -> None:
        text = "Incumbent contractor: Acme Federal Services LLC"
        explicit = [item for item in extract_incumbent_candidates(text) if item.role == "explicit_incumbent"]
        self.assertEqual(len(explicit), 1)
        self.assertIn("Acme", explicit[0].value)

    def test_past_performance_not_incumbent(self) -> None:
        text = "Past performance discussion mentions Beta Systems as an unsuccessful offeror."
        explicit = [item for item in extract_incumbent_candidates(text) if item.role == "explicit_incumbent"]
        self.assertEqual(len(explicit), 0)


if __name__ == "__main__":
    unittest.main()
