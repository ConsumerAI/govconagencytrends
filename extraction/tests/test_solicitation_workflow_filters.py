from __future__ import annotations

import unittest

from solicitation_workflow import (
    build_solicitation_additional_filter_rows,
    is_solicitation_filter_removed,
)


def _confirmed_filter(row: dict, _available_filter_options: dict) -> bool:
    return bool(row.get("filter_key") and row.get("filter_option") and row.get("preselect"))


def _requires_user_confirmation(row: dict) -> bool:
    return row.get("filter_key") == "pop_state"


def _confirmed_row(field: str, filter_key: str, option: str) -> dict:
    return {
        "field": field,
        "filter_key": filter_key,
        "filter_option": option,
        "mapping_status": "Exact match",
        "preselect": True,
        "confidence": "high",
        "extracted_value": option,
    }


def _additional_fields(rows: list[dict], removed_fields: set[str] | None = None) -> set[str]:
    additional = build_solicitation_additional_filter_rows(
        rows,
        removed_fields=removed_fields or set(),
        funding_office_extracted=False,
        is_confirmed_filter=_confirmed_filter,
        requires_user_confirmation=_requires_user_confirmation,
        available_filter_options={},
    )
    return {row["field"] for row in additional}


class SolicitationAdditionalFilterRowsTests(unittest.TestCase):
    def test_auto_mapped_filter_is_excluded(self) -> None:
        fields = _additional_fields([_confirmed_row("NAICS", "naics_code", "722310")])

        self.assertNotIn("NAICS", fields)

    def test_unmapped_filter_is_included(self) -> None:
        fields = _additional_fields(
            [
                {
                    "field": "PSC",
                    "filter_key": "psc_code",
                    "filter_option": None,
                    "mapping_status": "Unmapped",
                    "preselect": False,
                    "confidence": "high",
                    "extracted_value": "S203",
                }
            ]
        )

        self.assertIn("PSC", fields)

    def test_review_only_place_of_performance_is_included(self) -> None:
        fields = _additional_fields([_confirmed_row("Place of Performance", "pop_state", "FL")])

        self.assertIn("Place of Performance", fields)

    def test_manual_funding_office_is_included(self) -> None:
        fields = _additional_fields([])

        self.assertIn("Funding Office", fields)

    def test_user_removed_mapped_filter_is_included(self) -> None:
        fields = _additional_fields(
            [_confirmed_row("Contracting Office", "contracting_office", "FA4819")],
            removed_fields={"Contracting Office"},
        )

        self.assertIn("Contracting Office", fields)

    def test_restored_filter_is_excluded_again(self) -> None:
        row = _confirmed_row("PSC", "psc_code", "S203")

        self.assertIn("PSC", _additional_fields([row], removed_fields={"PSC"}))
        self.assertNotIn("PSC", _additional_fields([row], removed_fields=set()))

    def test_no_duplicate_field_in_applied_and_additional_sections(self) -> None:
        rows = [
            _confirmed_row("NAICS", "naics_code", "722310"),
            _confirmed_row("PSC", "psc_code", "S203"),
        ]
        applied_fields = {
            row["field"]
            for row in rows
            if _confirmed_filter(row, {})
            and not is_solicitation_filter_removed(row["field"], set())
        }
        additional_fields = _additional_fields(rows)

        self.assertFalse(applied_fields.intersection(additional_fields))


if __name__ == "__main__":
    unittest.main()
