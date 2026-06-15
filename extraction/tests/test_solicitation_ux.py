from __future__ import annotations

import inspect
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

import app
from streamlit.errors import StreamlitAPIException
from app import (
    ALL_BUREAUS,
    ALL_CONTRACTING_OFFICES,
    ALL_CONTRACT_TYPES,
    ALL_NAICS_CODES,
    ALL_PRODUCT_SERVICE_CODES,
    ALL_SET_ASIDE_TYPES,
    KEEP_CURRENT_SOLICITATION_FILTER,
    KEEP_REMOVED_SOLICITATION_FILTER,
    SET_ASIDE_TYPE_OPTIONS,
    SOLICITATION_USER_CONFIRMATION_FILTER_KEYS,
    _build_pending_filters_from_review_rows,
    _build_reviewed_pending_filters,
    _build_solicitation_audit_state,
    _solicitation_review_rows,
    _solicitation_source_completeness,
    _solicitation_optional_filter_rows,
    _solicitation_requires_user_confirmation,
    apply_solicitation_pending_filters,
    build_solicitation_scope_preview,
    degraded_solicitation_mapping_result,
    encode_contracting_office,
    encode_option,
    is_solicitation_confirmed_filter,
    load_resolved_signals_json,
    map_solicitation_market_filters,
    map_solicitation_organization,
    match_contracting_office_to_options,
)
from extraction.persist import read_json
from solicitation_workflow import (
    AUTO_MAPPED_DASHBOARD_FILTER_FIELDS,
    MARKET_SCOPE_FAST_SIGNAL_COUNT,
    begin_comparable_market_scope,
    build_fast_scope_detail_rows,
    build_solicitation_additional_filter_rows,
    count_auto_mapped_dashboard_filters,
    dedupe_extraction_findings,
    extract_psc_code,
    fast_scope_requested_count,
    is_ocr_environment_notice,
    is_solicitation_filter_removed,
    match_psc_to_options,
    resolve_solicitation_filter_pending_value,
    restore_exact_market_scope,
    should_show_blocking_ocr_error,
    should_show_solicitation_additional_filter,
    solicitation_scope_review_visible,
    solicitation_status_alert_text,
)

TYNDALL_FAST_RESOLVED = (
    REPO_ROOT
    / "data"
    / "package_cache"
    / "4f6f9087ee7f04416260ce4ef48c37d0f1d081e4cc98907c352af4fc01e56418"
    / "resolved_signals.json"
)
RECLAMATION_RESOLVED = (
    REPO_ROOT
    / "extraction"
    / "tests"
    / "fixtures"
    / "reclamation_140R2026Q0025_resolved_signals.json"
)


def _tyndall_available_options() -> dict:
    return {
        "agencies": [
            "Department of Defense",
            "Department of the Air Force",
            "National Aeronautics and Space Administration",
        ],
        "bureaus": [ALL_BUREAUS, "Department of the Air Force"],
        "contracting_offices": [
            ALL_CONTRACTING_OFFICES,
            encode_contracting_office("FA4819", "325 CONS PKP"),
        ],
        "funding_offices": [],
        "naics": [ALL_NAICS_CODES, encode_option("722310", "Food Service Contractors")],
        "psc": [ALL_PRODUCT_SERVICE_CODES, encode_option("S203", "HOUSEKEEPING- FOOD")],
        "contract_types": [ALL_CONTRACT_TYPES, encode_option("J", "FIRM FIXED PRICE")],
        "set_asides": [
            ALL_SET_ASIDE_TYPES,
            encode_option("8A", SET_ASIDE_TYPE_OPTIONS["8A"]),
        ],
        "pop_locations": [],
    }


def _reclamation_available_options() -> dict:
    return {
        "agencies": ["Department of the Interior", "Bureau of Reclamation"],
        "bureaus": [ALL_BUREAUS, "Bureau of Reclamation"],
        "contracting_offices": [
            ALL_CONTRACTING_OFFICES,
            encode_contracting_office("R20", "Division of Acquisition Services"),
        ],
        "funding_offices": [
            app.ALL_FUNDING_OFFICES,
            encode_option("R20", "Division of Acquisition Services"),
        ],
        "naics": [ALL_NAICS_CODES, encode_option("561210", "Facilities Support Services")],
        "psc": [ALL_PRODUCT_SERVICE_CODES, encode_option("S216", "Facilities Operations Support")],
        "contract_types": [
            ALL_CONTRACT_TYPES,
            encode_option("J", "FIRM FIXED PRICE"),
            encode_option("Y", "TIME AND MATERIALS"),
        ],
        "set_asides": [
            ALL_SET_ASIDE_TYPES,
            encode_option("NONE", "Unrestricted"),
        ],
        "pop_locations": [app.ALL_POP_LOCATIONS, encode_option("CA", "California")],
    }


class _SessionState(dict):
    def __getattr__(self, name: str):
        return self.get(name)

    def __setattr__(self, name: str, value) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
        else:
            self[name] = value


class _FakeStreamlitContext:
    def __init__(self, owner=None, *, kind: str | None = None, label: str | None = None):
        self._owner = owner
        self._kind = kind
        self._label = label
        if owner is not None:
            self.session_state = owner.session_state

    def __enter__(self):
        if self._owner is not None and self._kind == "expander":
            if self._owner._expander_depth:
                raise StreamlitAPIException("Expanders may not be nested inside other expanders.")
            self._owner._expander_depth += 1
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._owner is not None and self._kind == "expander":
            self._owner._expander_depth -= 1
        return False

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def checkbox(self, *args, key=None, **kwargs):
        return bool(key and self.session_state.get(key)) if hasattr(self, "session_state") else False

    def selectbox(self, label, options, key=None, **kwargs):
        value = self.session_state.get(key, options[0]) if hasattr(self, "session_state") else options[0]
        if key:
            self.session_state[key] = value
        return value


class _FakeStreamlit:
    def __init__(self, session_state: _SessionState):
        self.session_state = session_state
        self._expander_depth = 0
        self.expander_calls: list[dict] = []
        self.tab_calls: list[list[str]] = []

    def markdown(self, *args, **kwargs):
        return None

    def caption(self, *args, **kwargs):
        return None

    def warning(self, *args, **kwargs):
        return None

    def error(self, *args, **kwargs):
        return None

    def json(self, *args, **kwargs):
        return None

    def dataframe(self, *args, **kwargs):
        return None

    def divider(self, *args, **kwargs):
        return None

    def button(self, *args, **kwargs):
        return False

    def file_uploader(self, *args, **kwargs):
        return None

    def text_input(self, label, value="", **kwargs):
        return value

    def success(self, *args, **kwargs):
        return None

    def checkbox(self, *args, key=None, **kwargs):
        return bool(key and self.session_state.get(key))

    def selectbox(self, label, options, key=None, **kwargs):
        value = self.session_state.get(key, options[0])
        if key:
            self.session_state[key] = value
        return value

    def container(self, *args, **kwargs):
        return _FakeStreamlitContext(self)

    def expander(self, label, *args, **kwargs):
        self.expander_calls.append({"label": label, "expanded": kwargs.get("expanded")})
        return _FakeStreamlitContext(self, kind="expander", label=label)

    def tabs(self, labels):
        labels = list(labels)
        self.tab_calls.append(labels)
        return [_FakeStreamlitContext(self, kind="tab", label=label) for label in labels]

    def columns(self, spec, *args, **kwargs):
        columns = []
        count = spec if isinstance(spec, int) else len(spec)
        for _ in range(count):
            columns.append(_FakeStreamlitContext(self))
        return columns


class UploadSuccessNoticeTests(unittest.TestCase):
    def test_upload_panel_does_not_render_large_success_block(self) -> None:
        source = inspect.getsource(app.render_upload_solicitation_package_panel)
        self.assertNotIn("st.success", source)
        self.assertNotIn("confirmed market filter signal", source.lower())
        self.assertNotIn("12 confirmed", source)

    def test_scope_review_shows_muted_timing_not_success_card(self) -> None:
        source = inspect.getsource(app._render_solicitation_scope_heading)
        self.assertIn("Market scope extracted in", source)
        self.assertNotIn("st.success", source)


class DarkThemeReadabilityTests(unittest.TestCase):
    def test_semantic_text_color_variables_exist(self) -> None:
        source = inspect.getsource(app.inject_styles)
        for token in (
            "--text-primary:",
            "--text-secondary:",
            "--text-muted:",
            "--text-disabled:",
            "--text-link:",
            "--text-warning:",
            "--text-error:",
            "--text-success:",
        ):
            self.assertIn(token, source)

    def test_low_contrast_user_facing_text_colors_removed(self) -> None:
        source = inspect.getsource(app.inject_styles)
        for token in (
            "color: rgba(170, 180, 194",
            "color: rgba(148, 163, 184, 0.62)",
            "color: rgba(220, 229, 239, 0.58)",
            "color: #aab4c2",
        ):
            self.assertNotIn(token, source)

    def test_streamlit_captions_keep_readable_muted_color(self) -> None:
        source = inspect.getsource(app.inject_styles)
        self.assertIn('[data-testid="stCaptionContainer"]', source)
        self.assertIn("color: var(--text-muted) !important;", source)
        self.assertIn("opacity: 1 !important;", source)

    def test_scope_review_helpers_use_readable_classes(self) -> None:
        row_source = inspect.getsource(app._render_compact_solicitation_review_row)
        preview_source = inspect.getsource(app.render_solicitation_scope_preview)
        helper_source = inspect.getsource(app._render_scope_review_text)
        self.assertIn("scope-review-helper", helper_source)
        self.assertIn("scope-review-summary", helper_source)
        self.assertIn('_render_scope_review_text("Review before applying"', row_source)
        self.assertIn('_render_scope_review_text("Needs manual selection"', row_source)
        self.assertIn('_render_scope_review_text("Missing: "', inspect.getsource(app._render_compact_solicitation_review))
        self.assertIn("fields extracted", preview_source)
        self.assertNotIn('st.caption("Review before applying")', row_source)
        self.assertNotIn('st.caption("Needs manual selection")', row_source)

    def test_disabled_styles_are_reserved_for_disabled_controls(self) -> None:
        source = inspect.getsource(app.inject_styles)
        disabled_index = source.index(".stButton>button:disabled")
        disabled_block = source[disabled_index : source.index(".control-button-spacer", disabled_index)]
        self.assertIn("color: var(--text-disabled) !important;", disabled_block)
        self.assertIn("opacity: 1 !important;", disabled_block)
        self.assertEqual(source.count("var(--text-disabled)"), 1)
        row_source = inspect.getsource(app._render_compact_solicitation_review_row)
        helper_source = inspect.getsource(app._render_scope_review_text)
        self.assertNotIn("text-disabled", row_source + helper_source)

    def test_semantic_alert_text_colors_are_preserved(self) -> None:
        source = inspect.getsource(app.inject_styles)
        self.assertIn("color: var(--text-warning);", source)
        self.assertIn("color: var(--text-error);", source)
        self.assertIn("color: var(--text-success);", source)


class OCRNoticeTests(unittest.TestCase):
    def test_environment_notice_detection(self) -> None:
        self.assertTrue(is_ocr_environment_notice("OCR requires pypdfium2 for PDF rendering"))
        self.assertFalse(is_ocr_environment_notice("CONTROLLING_DOCUMENT_UNREADABLE"))

    def test_upload_panel_omits_primary_ocr_environment_caption(self) -> None:
        source = inspect.getsource(app.render_upload_solicitation_package_panel)
        self.assertNotIn("pypdfium2", source)

    def test_blocking_ocr_only_when_controlling_unreadable(self) -> None:
        self.assertTrue(should_show_blocking_ocr_error({"controllingDocumentsUnreadable": 1}, []))
        self.assertFalse(
            should_show_blocking_ocr_error(
                {},
                [{"code": "OCR_READY", "level": "info", "message": "OCR requires pypdfium2 for PDF rendering"}],
            )
        )

    def test_dedupe_findings(self) -> None:
        findings = [
            {"code": "A", "message": "same"},
            {"code": "A", "message": "same"},
            {"code": "B", "message": "other"},
        ]
        self.assertEqual(len(dedupe_extraction_findings(findings)), 2)


class ScopeReviewOrderTests(unittest.TestCase):
    def test_scope_review_renders_before_diagnostics(self) -> None:
        source = inspect.getsource(app.render_solicitation_scope_preview)
        filters_idx = source.index("_render_compact_solicitation_review")
        diagnostics_idx = source.index('st.expander("Developer diagnostics"')
        advanced_idx = source.index("_render_extraction_diagnostics_content")
        self.assertLess(filters_idx, diagnostics_idx)
        self.assertLess(diagnostics_idx, advanced_idx)


class ScopeReviewVisibilityTests(unittest.TestCase):
    def test_review_visible_after_extraction_before_apply(self) -> None:
        self.assertTrue(
            solicitation_scope_review_visible(
                has_resolved_signals=True,
                scope_applied=False,
                review_open=False,
            )
        )

    def test_review_hidden_after_apply_until_reopened(self) -> None:
        self.assertFalse(
            solicitation_scope_review_visible(
                has_resolved_signals=True,
                scope_applied=True,
                review_open=False,
            )
        )
        self.assertTrue(
            solicitation_scope_review_visible(
                has_resolved_signals=True,
                scope_applied=True,
                review_open=True,
            )
        )

    def test_sidebar_summary_only_after_application(self) -> None:
        source = inspect.getsource(app.render_solicitation_sidebar_scope)
        self.assertIn("solicitation_scope_applied", source)
        self.assertIn("Review / Edit Scope", source)
        self.assertNotIn("Filters Found in Solicitation", source)


class PSCMappingTests(unittest.TestCase):
    def test_extract_psc_code_requires_four_characters(self) -> None:
        self.assertEqual(extract_psc_code("S203"), "S203")
        self.assertEqual(extract_psc_code("PSC S203"), "S203")
        self.assertIsNone(extract_psc_code("S20"))

    def test_psc_not_mapped_when_absent_from_scoped_options(self) -> None:
        options = [ALL_PRODUCT_SERVICE_CODES, encode_option("S201", "JANITORIAL")]
        match = match_psc_to_options("S203", options, ALL_PRODUCT_SERVICE_CODES)
        self.assertIsNone(match["filter_option"])

    def test_psc_exact_match_only(self) -> None:
        options = [
            ALL_PRODUCT_SERVICE_CODES,
            encode_option("S203", "HOUSEKEEPING- FOOD"),
            encode_option("S201", "JANITORIAL"),
        ]
        match = match_psc_to_options("S203", options, ALL_PRODUCT_SERVICE_CODES)
        self.assertEqual(match["mapping_status"], "Exact match")
        self.assertEqual(match["filter_option"], encode_option("S203", "HOUSEKEEPING- FOOD"))


class OfficeFilterTests(unittest.TestCase):
    def test_exact_office_code_match_without_prefix_broadening(self) -> None:
        options = [
            ALL_CONTRACTING_OFFICES,
            encode_contracting_office("FA4819", "325 CONS PKP"),
            encode_contracting_office("FA481", "OTHER OFFICE"),
        ]
        match = match_contracting_office_to_options("FA4819", "325 CONS PKP", options)
        self.assertEqual(match["filter_option"], encode_contracting_office("FA4819", "325 CONS PKP"))
        self.assertEqual(match["mapping_status"], "Exact match")

    def test_office_code_with_prefixed_name_matches_exactly(self) -> None:
        options = [
            ALL_CONTRACTING_OFFICES,
            encode_contracting_office("FA4819", "FA4819 325 CONS PKP"),
        ]
        match = match_contracting_office_to_options("FA4819", "325 CONS PKP", options)
        self.assertEqual(match["mapping_status"], "Exact match")
        self.assertEqual(match["filter_option"], encode_contracting_office("FA4819", "FA4819 325 CONS PKP"))

    def test_multiple_code_matches_without_name_do_not_auto_apply(self) -> None:
        options = [
            ALL_CONTRACTING_OFFICES,
            encode_contracting_office("FA4819", "325 CONS PKP"),
            encode_contracting_office("FA4819", "OTHER OFFICE"),
        ]
        match = match_contracting_office_to_options("FA4819", "325 CONS PKP", options)
        self.assertEqual(match["mapping_status"], "Exact match")
        self.assertEqual(match["filter_option"], encode_contracting_office("FA4819", "325 CONS PKP"))

    def test_office_tooltip_uses_awarding_office_code(self) -> None:
        source = inspect.getsource(app._render_solicitation_filters_found)
        self.assertIn("awarding office code", source.lower())


class ComparableScopeTests(unittest.TestCase):
    def test_broader_market_removes_only_contracting_office(self) -> None:
        transition = begin_comparable_market_scope(
            encode_contracting_office("FA4819", "325 CONS PKP"),
            ALL_CONTRACTING_OFFICES,
        )
        self.assertEqual(transition["active_contracting_office"], ALL_CONTRACTING_OFFICES)
        self.assertEqual(
            transition["saved_contracting_office"],
            encode_contracting_office("FA4819", "325 CONS PKP"),
        )
        self.assertTrue(transition["comparable_market"])

    def test_restore_exact_scope_restores_saved_office(self) -> None:
        transition = restore_exact_market_scope(
            encode_contracting_office("FA4819", "325 CONS PKP"),
            ALL_CONTRACTING_OFFICES,
        )
        self.assertEqual(
            transition["active_contracting_office"],
            encode_contracting_office("FA4819", "325 CONS PKP"),
        )
        self.assertFalse(transition["comparable_market"])


class CountTests(unittest.TestCase):
    def test_auto_mapped_filter_count_excludes_pop_and_context(self) -> None:
        rows = [
            {
                "field": "Agency",
                "filter_key": "agency",
                "filter_option": "Department of the Air Force",
                "mapping_status": "Exact match",
                "preselect": True,
                "confidence": "high",
                "deterministic_mapping": True,
                "validation_status": "validated_model_extraction",
                "evidence_snippet": "Department of the Air Force",
            },
            {
                "field": "PSC",
                "filter_key": "psc_code",
                "filter_option": "S203 — HOUSEKEEPING- FOOD",
                "mapping_status": "Exact match",
                "preselect": True,
                "confidence": "high",
                "validation_status": "validated_model_extraction",
                "evidence_snippet": "S203",
            },
            {
                "field": "Place of Performance",
                "filter_key": "pop_state",
                "filter_option": "FL — Florida",
                "mapping_status": "Exact match",
                "preselect": True,
                "confidence": "high",
                "validation_status": "validated_model_extraction",
                "evidence_snippet": "Tyndall AFB, FL",
            },
        ]
        available = {
            "agencies": ["Department of the Air Force"],
            "psc": ["S203 — HOUSEKEEPING- FOOD"],
            "pop_locations": ["FL — Florida"],
        }
        count = count_auto_mapped_dashboard_filters(
            rows,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            available_filter_options=available,
        )
        self.assertEqual(count, 2)
        self.assertIn("Agency", AUTO_MAPPED_DASHBOARD_FILTER_FIELDS)
        self.assertNotIn("Place of Performance", AUTO_MAPPED_DASHBOARD_FILTER_FIELDS)

    def test_process_sources_not_counted_as_requested_fast_signals(self) -> None:
        payload = {
            "summary": {"requestedSignalCount": 11},
            "signals": [{"id": "process_sources_v1"}, {"id": "rfp_primary_naics_v1"}],
        }
        self.assertEqual(fast_scope_requested_count(payload), 11)

    def test_pop_remains_review_only(self) -> None:
        self.assertIn("pop_state", SOLICITATION_USER_CONFIRMATION_FILTER_KEYS)


class TyndallMappingIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not TYNDALL_FAST_RESOLVED.exists():
            raise unittest.SkipTest(f"Missing Tyndall fast resolved fixture: {TYNDALL_FAST_RESOLVED}")
        cls.resolved = read_json(TYNDALL_FAST_RESOLVED)
        cls.scope_preview = build_solicitation_scope_preview(load_resolved_signals_json(cls.resolved))

    def test_tyndall_six_green_mapped_filters_when_s203_present_contract_type_manual(self) -> None:
        agency_records = [
            {
                "agency_name": "Department of Defense",
                "toptier_code": "097",
                "abbreviation": "DOD",
            },
            {
                "agency_name": "Department of the Air Force",
                "toptier_code": "057",
                "abbreviation": "USAF",
            },
        ]
        available = _tyndall_available_options()
        org_mapping = map_solicitation_organization(self.scope_preview, agency_records, 2026)
        market_mapping = map_solicitation_market_filters(self.scope_preview, available)
        rows = org_mapping["rows"] + market_mapping["rows"]
        mapped_count = count_auto_mapped_dashboard_filters(
            rows,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            available_filter_options=available,
        )
        confirmed_fields = {
            row["field"]
            for row in rows
            if row.get("filter_key") and is_solicitation_confirmed_filter(row, available)
        }
        self.assertEqual(mapped_count, 6)
        self.assertEqual(
            confirmed_fields,
            {
                "Agency",
                "Subagency / Bureau",
                "Contracting Office",
                "NAICS",
                "PSC",
                "Set-Aside",
            },
        )
        contract_type_row = next(row for row in rows if row["field"] == "Contract Type")
        self.assertIsNone(contract_type_row["filter_option"])
        self.assertFalse(contract_type_row["preselect"])

    def test_status_alert_not_treated_as_filter(self) -> None:
        self.assertNotIn("Status Alert", AUTO_MAPPED_DASHBOARD_FILTER_FIELDS)


class ReclamationMappingRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not RECLAMATION_RESOLVED.exists():
            raise unittest.SkipTest(f"Missing Reclamation fixture: {RECLAMATION_RESOLVED}")
        cls.resolved = read_json(RECLAMATION_RESOLVED)
        cls.scope_preview = build_solicitation_scope_preview(load_resolved_signals_json(cls.resolved))

    def test_bureau_of_reclamation_maps_to_interior_parent_and_subagency(self) -> None:
        agency_records = [
            {"agency_name": "Department of the Interior", "toptier_code": "014"},
            {"agency_name": "Bureau of Reclamation", "toptier_code": "999"},
        ]
        with patch.object(app, "get_bureau_options", return_value=[ALL_BUREAUS, "Bureau of Reclamation"]):
            org = map_solicitation_organization(self.scope_preview, agency_records, 2026)
        by_field = {row["field"]: row for row in org["rows"]}
        self.assertEqual(by_field["Agency"]["filter_option"], "Department of the Interior")
        self.assertEqual(by_field["Subagency / Bureau"]["filter_option"], "Bureau of Reclamation")
        self.assertNotEqual(by_field["Agency"]["filter_option"], "Bureau of Reclamation")
        self.assertNotEqual(by_field["Subagency / Bureau"]["filter_option"], ALL_BUREAUS)
        self.assertTrue(org["hierarchy_rule_applied"])
        self.assertIn("matched_subagency", org["mapping_attempts"]["Agency"][0])

    def test_reclamation_acceptance_rows_map_expected_values(self) -> None:
        agency_records = [{"agency_name": "Department of the Interior", "toptier_code": "014"}]
        available = _reclamation_available_options()
        with (
            patch.object(app, "get_bureau_options", return_value=[ALL_BUREAUS, "Bureau of Reclamation"]),
            patch.object(app, "build_solicitation_available_filter_options", return_value=available),
        ):
            mapping = app.map_solicitation_signals_to_dashboard_filters(self.scope_preview, agency_records, 2026)
        rows = {row["field"]: row for row in mapping["rows"]}
        self.assertEqual(rows["Agency"]["filter_option"], "Department of the Interior")
        self.assertEqual(rows["Subagency / Bureau"]["filter_option"], "Bureau of Reclamation")
        self.assertEqual(rows["Contracting Office"]["filter_option"], encode_contracting_office("R20", "Division of Acquisition Services"))
        self.assertEqual(rows["NAICS"]["filter_option"], encode_option("561210", "Facilities Support Services"))
        self.assertEqual(rows["PSC"]["filter_option"], encode_option("S216", "Facilities Operations Support"))
        self.assertEqual(rows["Set-Aside"]["filter_option"], encode_option("NONE", "Unrestricted"))
        self.assertEqual(rows["Place of Performance"]["filter_option"], encode_option("CA", "California"))
        self.assertTrue(rows["Contract Type"]["is_hybrid"])
        self.assertFalse(rows["Contract Type"]["preselect"])
        self.assertIsNone(rows["Contract Type"]["filter_option"])
        self.assertEqual(rows["Contract Type"]["mapped_filter_display"], "Select manually")
        self.assertIsNone(rows["Contract Type"]["contract_type_filter_source"])

    def test_reclamation_review_defaults_pop_unchecked_and_hybrid_requires_selection(self) -> None:
        available = _reclamation_available_options()
        rows = [
            {"field": "Agency", "filter_key": "agency", "filter_option": "Department of the Interior", "mapping_status": "Exact match", "preselect": True, "confidence": "high", "extracted_value": "Bureau of Reclamation"},
            {"field": "Subagency / Bureau", "filter_key": "bureau", "filter_option": "Bureau of Reclamation", "mapping_status": "Exact match", "preselect": True, "confidence": "high", "extracted_value": "Bureau of Reclamation"},
            {"field": "Contract Type", "filter_key": "contract_type", "filter_option": None, "mapped_filter_display": "Select manually", "unmapped_extracted_display": "Select manually", "mapping_status": "Manual selection required", "preselect": False, "confidence": "high", "extracted_value": "Hybrid Time & Materials and Firm-Fixed Price", "is_hybrid": True, "contract_type_filter_source": None},
            {"field": "Place of Performance", "filter_key": "pop_state", "filter_option": encode_option("CA", "California"), "mapping_status": "Suggested match", "preselect": False, "confidence": "medium", "extracted_value": "Fresno, CA"},
        ]
        session = _SessionState()
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows(rows, self.scope_preview, available)
            pending = _build_pending_filters_from_review_rows(review_rows, available)
        by_field = {item["field"]: item for item in review_rows}
        self.assertFalse(by_field["Contract Type"]["analyst_selected"])
        self.assertFalse(by_field["Place of Performance"]["analyst_selected"])
        self.assertNotIn("contract_type", pending["market_filters"])
        self.assertNotIn("pop_state", pending["market_filters"])

    def test_contract_type_enters_pending_only_after_analyst_selection(self) -> None:
        selected = encode_option("Y", "TIME AND MATERIALS")
        row = {
            "field": "Contract Type",
            "filter_key": "contract_type",
            "filter_option": None,
            "mapping_status": "Manual selection required",
            "preselect": False,
            "confidence": "high",
            "extracted_value": "hybrid Time & Material and Firm Fixed Price",
            "is_hybrid": True,
            "contract_type_filter_source": None,
        }
        session = _SessionState(
            {
                app._solicitation_review_session_key("use", "Contract Type"): True,
                app._solicitation_review_session_key("value", "Contract Type"): selected,
                "solicitation_analyst_edited_fields": {"Contract Type"},
            }
        )
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([row], {}, _reclamation_available_options())
            pending = _build_pending_filters_from_review_rows(review_rows, _reclamation_available_options())
            audit = _build_solicitation_audit_state(review_rows)
        self.assertEqual(pending["market_filters"]["contract_type"], selected)
        self.assertEqual(next(item for item in audit if item["field"] == "Contract Type")["decision_status"], "edited_by_analyst")
        self.assertEqual(next(item for item in audit if item["field"] == "Contract Type")["contract_type_filter_source"], "analyst_selection")

    def test_extracted_pure_ffp_does_not_preselect_contract_type(self) -> None:
        available = _reclamation_available_options()
        row = {
            "field": "Contract Type",
            "filter_key": "contract_type",
            "filter_option": encode_option("J", "FIRM FIXED PRICE"),
            "mapping_status": "Exact match",
            "preselect": True,
            "confidence": "high",
            "extracted_value": "Firm Fixed Price",
        }
        session = _SessionState()
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([row], {}, available)
            pending = _build_pending_filters_from_review_rows(review_rows, available)
        contract_type = next(item for item in review_rows if item["field"] == "Contract Type")
        self.assertFalse(contract_type["preselected"])
        self.assertFalse(contract_type["analyst_selected"])
        self.assertNotIn("contract_type", pending["market_filters"])

    def test_existing_contract_type_is_retained_without_ai_pending_value(self) -> None:
        available = _reclamation_available_options()
        existing = encode_option("J", "FIRM FIXED PRICE")
        row = {
            "field": "Contract Type",
            "filter_key": "contract_type",
            "filter_option": None,
            "mapping_status": "Manual selection required",
            "preselect": False,
            "confidence": "high",
            "extracted_value": "Time and Materials",
        }
        session = _SessionState({"active_market_filters": {**app.default_market_filters(), "contract_type": existing}})
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([row], {}, available)
            pending = _build_pending_filters_from_review_rows(review_rows, available)
            audit = _build_solicitation_audit_state(review_rows)
            placeholder = app._solicitation_select_placeholder_for_item(next(item for item in review_rows if item["field"] == "Contract Type"))
        self.assertEqual(placeholder, KEEP_CURRENT_SOLICITATION_FILTER)
        self.assertNotIn("contract_type", pending["market_filters"])
        self.assertEqual(next(item for item in audit if item["field"] == "Contract Type")["contract_type_filter_source"], "existing_pending_state")

    def test_failed_mapping_broad_defaults_are_removed_from_pending(self) -> None:
        available = _reclamation_available_options()
        pending, diagnostics = app._sanitize_review_pending_filters(
            {
                "agency": "Department of the Interior",
                "bureau": ALL_BUREAUS,
                "contracting_office": ALL_CONTRACTING_OFFICES,
                "market_filters": {"psc_code": ALL_PRODUCT_SERVICE_CODES},
            },
            available,
        )
        self.assertIsNone(pending["bureau"])
        self.assertIsNone(pending["contracting_office"])
        self.assertNotIn("psc_code", pending["market_filters"])
        self.assertTrue(diagnostics)

    def test_audit_chain_preserves_raw_to_pending_fields(self) -> None:
        available = _reclamation_available_options()
        row = {
            "field": "NAICS",
            "filter_key": "naics_code",
            "filter_option": encode_option("561210", "Facilities Support Services"),
            "mapped_filter_display": "561210 - Facilities Support Services",
            "mapping_status": "Exact match",
            "preselect": True,
            "confidence": "high",
            "extracted_value": "561210",
        }
        session = _SessionState()
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([row], {}, available)
            audit = _build_solicitation_audit_state(review_rows)
        naics = next(item for item in audit if item["field"] == "NAICS")
        for key in (
            "raw_extracted_value",
            "validated_value",
            "resolved_value",
            "mapped_dashboard_value",
            "preselection_allowed",
            "preselection_reason",
            "analyst_value",
            "final_pending_value",
        ):
            self.assertIn(key, naics)

    def test_usarcent_synthetic_contract_type_manual_and_pop_unchecked(self) -> None:
        available = _tyndall_available_options()
        available["set_asides"].append(encode_option("SBA", "Small Business Set-Aside"))
        available["pop_locations"] = [app.ALL_POP_LOCATIONS, encode_option("SC", "South Carolina")]
        rows = [
            {"field": "Set-Aside", "filter_key": "set_aside_type", "filter_option": encode_option("SBA", "Small Business Set-Aside"), "mapping_status": "Exact match", "preselect": True, "confidence": "high", "extracted_value": "Total Small Business Set-Aside", "validation_status": "validated_model_extraction", "evidence_snippet": "Total Small Business Set-Aside"},
            {"field": "Place of Performance", "filter_key": "pop_state", "filter_option": encode_option("SC", "South Carolina"), "mapping_status": "Suggested match", "preselect": False, "confidence": "high", "extracted_value": "South Carolina"},
            {"field": "Contract Type", "filter_key": "contract_type", "filter_option": encode_option("J", "FIRM FIXED PRICE"), "mapping_status": "Exact match", "preselect": True, "confidence": "high", "extracted_value": "FFP"},
        ]
        session = _SessionState()
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows(rows, {}, available)
            pending = _build_pending_filters_from_review_rows(review_rows, available)
        by_field = {item["field"]: item for item in review_rows}
        self.assertEqual(by_field["Set-Aside"]["selected_value"], encode_option("SBA", "Small Business Set-Aside"))
        self.assertFalse(by_field["Place of Performance"]["analyst_selected"])
        self.assertFalse(by_field["Contract Type"]["analyst_selected"])
        self.assertNotIn("contract_type", pending["market_filters"])


class ApplyFiltersStateTests(unittest.TestCase):
    def test_apply_pending_filters_updates_active_not_analyzed(self) -> None:
        pending = {
            "agency": "Department of the Air Force",
            "bureau": ALL_BUREAUS,
            "contracting_office": encode_contracting_office("FA4819", "325 CONS PKP"),
            "market_filters": {
                "naics_code": encode_option("722310", "Food Service Contractors"),
                "psc_code": encode_option("S203", "HOUSEKEEPING- FOOD"),
                "contract_type": encode_option("J", "FIRM FIXED PRICE"),
                "set_aside_type": encode_option("8A", SET_ASIDE_TYPE_OPTIONS["8A"]),
            },
        }
        available = _tyndall_available_options()
        session = _SessionState(
            {
                "active_agency": "National Aeronautics and Space Administration",
                "active_bureau": ALL_BUREAUS,
                "active_contracting_office": ALL_CONTRACTING_OFFICES,
                "active_market_filters": app.default_market_filters(),
                "analyzed_agency": "National Aeronautics and Space Administration",
                "analyzed_bureau": ALL_BUREAUS,
                "analyzed_contracting_office": ALL_CONTRACTING_OFFICES,
                "analyzed_market_filters": app.default_market_filters(),
                "analyzed_year": 2026,
            }
        )

        with patch.object(app.st, "session_state", session):
            apply_solicitation_pending_filters(pending, available)

        self.assertEqual(session["active_agency"], "Department of the Air Force")
        self.assertEqual(session["active_contracting_office"], encode_contracting_office("FA4819", "325 CONS PKP"))
        self.assertEqual(session["analyzed_agency"], "National Aeronautics and Space Administration")
        self.assertEqual(session["analyzed_contracting_office"], ALL_CONTRACTING_OFFICES)

    def test_apply_and_run_uses_mark_analysis_started(self) -> None:
        source = inspect.getsource(app.render_solicitation_scope_preview)
        apply_idx = source.index('key="apply_solicitation_filters"')
        run_idx = source.index('key="apply_solicitation_filters_and_run"')
        mark_idx = source.index("mark_analysis_started(")
        self.assertLess(apply_idx, run_idx)
        self.assertLess(run_idx, mark_idx)
        self.assertIn("if apply_filters_and_run:", source[run_idx:mark_idx + 80])


class DashboardLayoutTests(unittest.TestCase):
    def test_main_dashboard_skips_scope_review_after_apply(self) -> None:
        source = inspect.getsource(app.main)
        self.assertIn("_should_show_solicitation_scope_review()", source)
        self.assertIn("render_solicitation_baseline_notice()", source)


class FastDetailTests(unittest.TestCase):
    def test_fast_detail_rows_are_scalar(self) -> None:
        resolved = {
            "summary": {"requestedSignalCount": 11},
            "signals": [
                {
                    "id": "rfp_primary_naics_v1",
                    "canonical_value": "722310",
                    "canonical_confidence": "high",
                    "resolution_status": "validated_model_extraction",
                    "evidence": {"legacy": [{"sourceId": "doc.pdf", "locator": "page:1", "snippet": "722310"}]},
                }
            ],
        }
        rows = build_fast_scope_detail_rows(resolved)
        self.assertEqual(len(rows), MARKET_SCOPE_FAST_SIGNAL_COUNT)
        naics = next(item for item in rows if item["signal_id"] == "rfp_primary_naics_v1")
        self.assertEqual(naics["value"], "722310")
        self.assertNotIsInstance(naics["value"], dict)


class StatusAlertTests(unittest.TestCase):
    def test_status_alert_scalar(self) -> None:
        resolved = {
            "signals": [
                {
                    "id": "solicitation_status_alert_v1",
                    "canonical_value": "Stayed indefinitely due to GAO protest B-424443.1",
                }
            ]
        }
        self.assertIn("GAO protest", solicitation_status_alert_text(resolved) or "")


class USAspendingCalculationIsolationTests(unittest.TestCase):
    def test_obligation_helpers_unmodified_by_ux_module(self) -> None:
        for symbol in (
            "fetch_transaction_pages",
            "award_scope_dataframe",
            "award_scope_totals",
            "dashboard_base_filters",
        ):
            self.assertTrue(hasattr(app, symbol), f"Missing {symbol}")


class ScopeFilterVisibilityTests(unittest.TestCase):
    def _confirmed_row(self, field: str, filter_key: str, option: str) -> dict:
        return {
            "field": field,
            "filter_key": filter_key,
            "filter_option": option,
            "mapping_status": "Exact match",
            "preselect": True,
            "confidence": "high",
            "extracted_value": option,
            "validation_status": "validated_model_extraction",
            "evidence_snippet": str(option),
        }

    def test_auto_mapped_fields_not_in_additional_section(self) -> None:
        available = _tyndall_available_options()
        rows = [
            self._confirmed_row("NAICS", "naics_code", encode_option("722310", "Food Service Contractors")),
            self._confirmed_row("PSC", "psc_code", encode_option("S203", "HOUSEKEEPING- FOOD")),
            self._confirmed_row("Contract Type", "contract_type", encode_option("J", "FIRM FIXED PRICE")),
            self._confirmed_row("Set-Aside", "set_aside_type", encode_option("8A", SET_ASIDE_TYPE_OPTIONS["8A"])),
        ]
        additional = build_solicitation_additional_filter_rows(
            rows,
            removed_fields=set(),
            funding_office_extracted=False,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            requires_user_confirmation=app._solicitation_requires_user_confirmation,
            available_filter_options=available,
        )
        additional_fields = {row["field"] for row in additional}
        self.assertNotIn("NAICS", additional_fields)
        self.assertNotIn("PSC", additional_fields)
        self.assertIn("Contract Type", additional_fields)
        self.assertNotIn("Set-Aside", additional_fields)

    def test_tyndall_additional_fields_only_funding_and_pop(self) -> None:
        if not TYNDALL_FAST_RESOLVED.exists():
            self.skipTest("Tyndall fixture missing")
        available = _tyndall_available_options()
        available["pop_locations"] = [app.ALL_POP_LOCATIONS, encode_option("FL", "Florida")]
        scope_preview = build_solicitation_scope_preview(load_resolved_signals_json(read_json(TYNDALL_FAST_RESOLVED)))
        org = map_solicitation_organization(
            scope_preview,
            [
                {"agency_name": "Department of Defense", "toptier_code": "097"},
                {"agency_name": "Department of the Air Force", "toptier_code": "057"},
            ],
            2026,
        )
        market = map_solicitation_market_filters(scope_preview, available)
        rows = org["rows"] + market["rows"]
        additional = build_solicitation_additional_filter_rows(
            rows,
            removed_fields=set(),
            funding_office_extracted=False,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            requires_user_confirmation=app._solicitation_requires_user_confirmation,
            available_filter_options=available,
        )
        self.assertEqual({row["field"] for row in additional}, {"Funding Office", "Place of Performance", "Contract Type"})

    def test_removed_field_appears_in_additional_section(self) -> None:
        available = _tyndall_available_options()
        row = self._confirmed_row("NAICS", "naics_code", encode_option("722310", "Food Service Contractors"))
        additional = build_solicitation_additional_filter_rows(
            [row],
            removed_fields={"NAICS"},
            funding_office_extracted=False,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            requires_user_confirmation=app._solicitation_requires_user_confirmation,
            available_filter_options=available,
        )
        additional_fields = {item["field"] for item in additional}
        self.assertIn("NAICS", additional_fields)

    def test_restored_field_leaves_additional_section(self) -> None:
        available = _tyndall_available_options()
        row = self._confirmed_row("NAICS", "naics_code", encode_option("722310", "Food Service Contractors"))
        removed = build_solicitation_additional_filter_rows(
            [row],
            removed_fields={"NAICS"},
            funding_office_extracted=False,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            requires_user_confirmation=app._solicitation_requires_user_confirmation,
            available_filter_options=available,
        )
        restored = build_solicitation_additional_filter_rows(
            [row],
            removed_fields=set(),
            funding_office_extracted=False,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            requires_user_confirmation=app._solicitation_requires_user_confirmation,
            available_filter_options=available,
        )
        self.assertIn("NAICS", {item["field"] for item in removed})
        self.assertNotIn("NAICS", {item["field"] for item in restored})

    def test_contracting_office_fa4819_maps_automatically(self) -> None:
        options = [
            ALL_CONTRACTING_OFFICES,
            encode_contracting_office("FA4819", "325 CONS PKP"),
        ]
        match = match_contracting_office_to_options("FA4819", "325 CONS PKP", options)
        self.assertEqual(match["mapping_status"], "Exact match")
        self.assertEqual(match["filter_option"], encode_contracting_office("FA4819", "325 CONS PKP"))

    def test_psc_hidden_from_additional_when_mapped(self) -> None:
        available = _tyndall_available_options()
        rows = [self._confirmed_row("PSC", "psc_code", encode_option("S203", "HOUSEKEEPING- FOOD"))]
        additional = build_solicitation_additional_filter_rows(
            rows,
            removed_fields=set(),
            funding_office_extracted=False,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            requires_user_confirmation=app._solicitation_requires_user_confirmation,
            available_filter_options=available,
        )
        self.assertNotIn("PSC", {row["field"] for row in additional})

    def test_restore_pending_value_after_removal(self) -> None:
        available = _tyndall_available_options()
        row = self._confirmed_row("NAICS", "naics_code", encode_option("722310", "Food Service Contractors"))
        removed = resolve_solicitation_filter_pending_value(
            row,
            removed_fields={"NAICS"},
            user_override=KEEP_REMOVED_SOLICITATION_FILTER,
            keep_current_token=KEEP_CURRENT_SOLICITATION_FILTER,
            keep_removed_token=KEEP_REMOVED_SOLICITATION_FILTER,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            option_is_valid=app._solicitation_option_is_valid,
            available_filter_options=available,
        )
        self.assertIsNone(removed)
        restored = resolve_solicitation_filter_pending_value(
            row,
            removed_fields=set(),
            user_override=row["filter_option"],
            keep_current_token=KEEP_CURRENT_SOLICITATION_FILTER,
            keep_removed_token=KEEP_REMOVED_SOLICITATION_FILTER,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            option_is_valid=app._solicitation_option_is_valid,
            available_filter_options=available,
        )
        self.assertEqual(restored, row["filter_option"])

    def test_removing_one_field_does_not_alter_other_pending_filters(self) -> None:
        available = _tyndall_available_options()
        rows = [
            self._confirmed_row("NAICS", "naics_code", encode_option("722310", "Food Service Contractors")),
            self._confirmed_row("PSC", "psc_code", encode_option("S203", "HOUSEKEEPING- FOOD")),
        ]
        pending = _build_reviewed_pending_filters(rows, {}, available, removed_fields={"NAICS"})
        self.assertIsNone(pending["market_filters"].get("naics_code"))
        self.assertEqual(pending["market_filters"].get("psc_code"), encode_option("S203", "HOUSEKEEPING- FOOD"))

    def test_no_duplicate_field_in_both_sections(self) -> None:
        available = _tyndall_available_options()
        rows = [
            self._confirmed_row("PSC", "psc_code", encode_option("S203", "HOUSEKEEPING- FOOD")),
        ]
        additional = _solicitation_optional_filter_rows(
            rows,
            available,
            removed_fields=set(),
            funding_office_extracted=False,
        )
        confirmed_fields = {
            row["field"]
            for row in rows
            if is_solicitation_confirmed_filter(row, available)
            and not is_solicitation_filter_removed(row["field"], set())
        }
        overlap = confirmed_fields.intersection({row["field"] for row in additional})
        self.assertFalse(overlap)

    def test_additional_section_collapses_when_no_unresolved_fields(self) -> None:
        source = inspect.getsource(app._render_solicitation_optional_filters)
        self.assertIn("All extracted market filters are represented in the review groups.", source)

    def test_pop_remains_review_only(self) -> None:
        self.assertIn("pop_state", SOLICITATION_USER_CONFIRMATION_FILTER_KEYS)
        row = {
            "field": "Place of Performance",
            "filter_key": "pop_state",
            "filter_option": encode_option("FL", "Florida"),
            "mapping_status": "Exact match",
            "preselect": True,
            "confidence": "high",
            "extracted_value": "Tyndall AFB, FL",
        }
        self.assertTrue(
            should_show_solicitation_additional_filter(
                row,
                removed_fields=set(),
                funding_office_extracted=False,
                is_confirmed_filter=is_solicitation_confirmed_filter,
                requires_user_confirmation=_solicitation_requires_user_confirmation,
                available_filter_options={"pop_locations": [encode_option("FL", "Florida")]},
            )
        )

    def test_funding_office_always_manual(self) -> None:
        row = {
            "field": "Funding Office",
            "filter_key": "funding_office",
            "filter_option": None,
            "mapping_status": "Unmapped",
            "extracted_value": None,
        }
        self.assertTrue(
            should_show_solicitation_additional_filter(
                row,
                removed_fields=set(),
                funding_office_extracted=False,
                is_confirmed_filter=is_solicitation_confirmed_filter,
                requires_user_confirmation=_solicitation_requires_user_confirmation,
                available_filter_options={"funding_offices": []},
            )
        )

    def test_optional_hint_does_not_include_already_included_copy(self) -> None:
        source = inspect.getsource(app._solicitation_optional_filter_hint)
        self.assertNotIn("already included", source.lower())


class SolicitationTrustReviewTests(unittest.TestCase):
    def _confirmed_row(self, field: str, filter_key: str, option: str) -> dict:
        return {
            "field": field,
            "filter_key": filter_key,
            "filter_option": option,
            "mapping_status": "Exact match",
            "preselect": True,
            "confidence": "high",
            "extracted_value": option,
            "validation_status": "validated_model_extraction",
            "evidence_snippet": str(option),
            "evidence_source": "source.pdf",
            "evidence_locator": "page:1",
        }

    def test_mapping_result_does_not_prefill_pending_payload(self) -> None:
        rows = [self._confirmed_row("NAICS", "naics_code", encode_option("722310", "Food Service Contractors"))]
        pending = app._auto_pending_filters_from_mapping_rows(rows)
        self.assertEqual(pending["market_filters"]["naics_code"], encode_option("722310", "Food Service Contractors"))
        source = inspect.getsource(app.map_solicitation_signals_to_dashboard_filters)
        self.assertIn('"pending_filters": pending_filters', source)
        self.assertIn('"market_filters": {}', source)

    def test_high_confidence_exact_mapping_is_prechecked_but_not_applied(self) -> None:
        row = self._confirmed_row("NAICS", "naics_code", encode_option("722310", "Food Service Contractors"))
        session = _SessionState({"active_market_filters": app.default_market_filters()})
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([row], {}, _tyndall_available_options())
        naics = next(item for item in review_rows if item["field"] == "NAICS")
        self.assertTrue(naics["preselected"])
        self.assertTrue(naics["analyst_selected"])
        self.assertEqual(session["active_market_filters"], app.default_market_filters())

    def test_review_required_values_are_not_prechecked(self) -> None:
        row = self._confirmed_row("NAICS", "naics_code", encode_option("722310", "Food Service Contractors"))
        row["confidence"] = "medium"
        session = _SessionState()
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([row], {}, _tyndall_available_options())
        naics = next(item for item in review_rows if item["field"] == "NAICS")
        self.assertFalse(naics["preselected"])
        self.assertEqual(naics["group"], "review")

    def test_missing_fields_and_manual_only_fields(self) -> None:
        session = _SessionState()
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([], {}, _tyndall_available_options())
        manual_fields = {item["field"] for item in review_rows if item["group"] == "manual"}
        self.assertIn("Contracting Office", manual_fields)
        self.assertIn("AAC", manual_fields)
        self.assertIn("NAICS", manual_fields)
        self.assertIn("PSC", manual_fields)
        self.assertIn("Set-Aside", manual_fields)
        self.assertIn("Funding Office", manual_fields)

    def test_pop_and_funding_office_remain_unchecked_by_default(self) -> None:
        pop = self._confirmed_row("Place of Performance", "pop_state", encode_option("FL", "Florida"))
        funding = self._confirmed_row("Funding Office", "funding_office", encode_option("FA4819", "325 CONS PKP"))
        available = _tyndall_available_options()
        available["pop_locations"] = [app.ALL_POP_LOCATIONS, encode_option("FL", "Florida")]
        available["funding_offices"] = [app.ALL_FUNDING_OFFICES, encode_option("FA4819", "325 CONS PKP")]
        session = _SessionState()
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([pop, funding], {}, available)
        by_field = {item["field"]: item for item in review_rows}
        self.assertFalse(by_field["Place of Performance"]["preselected"])
        self.assertFalse(by_field["Funding Office"]["preselected"])
        self.assertFalse(by_field["Place of Performance"]["analyst_selected"])
        self.assertFalse(by_field["Funding Office"]["analyst_selected"])

    def test_unchecking_excludes_and_replacement_uses_analyst_value(self) -> None:
        suggested = encode_option("722310", "Food Service Contractors")
        replacement = encode_option("541330", "Engineering Services")
        row = self._confirmed_row("NAICS", "naics_code", suggested)
        available = _tyndall_available_options()
        available["naics"].append(replacement)
        session = _SessionState(
            {
                app._solicitation_review_session_key("use", "NAICS"): True,
                app._solicitation_review_session_key("value", "NAICS"): replacement,
            }
        )
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([row], {}, available)
            pending = _build_pending_filters_from_review_rows(review_rows, available)
            audit = _build_solicitation_audit_state(review_rows)
        self.assertEqual(pending["market_filters"]["naics_code"], replacement)
        naics_audit = next(item for item in audit if item["field"] == "NAICS")
        self.assertEqual(naics_audit["decision_status"], "edited_by_analyst")
        self.assertEqual(naics_audit["analyst_replacement"], replacement)

        session[app._solicitation_review_session_key("use", "NAICS")] = False
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([row], {}, available)
            pending = _build_pending_filters_from_review_rows(review_rows, available)
        self.assertNotIn("naics_code", pending["market_filters"])

    def test_restoring_suggestion_removes_replacement(self) -> None:
        suggested = encode_option("722310", "Food Service Contractors")
        row = self._confirmed_row("NAICS", "naics_code", suggested)
        session = _SessionState(
            {
                app._solicitation_review_session_key("use", "NAICS"): True,
                app._solicitation_review_session_key("value", "NAICS"): suggested,
            }
        )
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows([row], {}, _tyndall_available_options())
            audit = _build_solicitation_audit_state(review_rows)
        naics_audit = next(item for item in audit if item["field"] == "NAICS")
        self.assertIsNone(naics_audit["analyst_replacement"])

    def test_no_duplicate_groups_and_partial_completeness(self) -> None:
        rows = [
            self._confirmed_row("Agency", "agency", "Department of Defense"),
            self._confirmed_row("Contract Type", "contract_type", encode_option("J", "FIRM FIXED PRICE")),
            self._confirmed_row("Set-Aside", "set_aside_type", encode_option("SBA", "Small Business Set-Aside")),
        ]
        available = _tyndall_available_options()
        available["agencies"] = ["Department of Defense"]
        available["set_asides"].append(encode_option("SBA", "Small Business Set-Aside"))
        session = _SessionState()
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows(rows, {}, available)
        fields = [item["field"] for item in review_rows]
        self.assertEqual(len(fields), len(set(fields)))
        completeness, missing = _solicitation_source_completeness(review_rows)
        self.assertEqual(completeness, "Partial market scope found")
        self.assertIn("Contracting Office", missing)
        self.assertIn("NAICS", missing)
        self.assertIn("PSC", missing)

    def test_degraded_mapping_still_builds_scope_review_rows(self) -> None:
        scope_preview = {
            "Issuing Agency": {
                "value": "Department of the Air Force",
                "confidence": "high",
                "evidence_snippet": "Department of the Air Force",
                "validation_status": "validated_model_extraction",
            },
            "Primary NAICS": {
                "value": "722310",
                "confidence": "high",
                "evidence_snippet": "NAICS 722310",
                "validation_status": "validated_model_extraction",
            },
            "Contract Type": {
                "value": "FFP",
                "confidence": "high",
                "evidence_snippet": "Firm fixed price",
                "validation_status": "validated_model_extraction",
            },
        }
        agency_records = [
            {"agency_name": "Department of Defense", "toptier_code": "097"},
            {"agency_name": "Department of the Air Force", "toptier_code": "057"},
        ]
        session = _SessionState()
        mapping = degraded_solicitation_mapping_result(scope_preview, agency_records, 2026)
        with patch.object(app.st, "session_state", session):
            review_rows = _solicitation_review_rows(rows=mapping["rows"], scope_preview=scope_preview, available_filter_options=mapping["available_filter_options"])
        by_field = {item["field"]: item for item in review_rows}
        self.assertEqual(mapping["debug"]["mapping_degraded"], True)
        self.assertIn("NAICS", mapping["available_filter_options"]["mapping_unavailable_fields"])
        self.assertIn(by_field["NAICS"]["group"], {"review", "manual"})
        self.assertEqual(by_field["Contract Type"]["group"], "review")

    def test_extraction_to_review_transition_has_durable_completed_status(self) -> None:
        source = inspect.getsource(app.render_upload_solicitation_package_panel)
        self.assertIn('solicitation_extraction_status = "completed"', source)
        self.assertIn("solicitation_extraction_transition_rerun_done", source)
        self.assertIn("status.empty()", source)
        self.assertIn("progress.empty()", source)
        self.assertNotIn("Load completed extraction", source)
        self.assertIn("recoveredExistingArtifact", source)

    def test_no_six_column_review_table_is_rendered(self) -> None:
        source = inspect.getsource(app._render_compact_solicitation_review)
        self.assertNotIn('"Use", "Filter", "Suggested value", "Confidence", "Evidence", "Status"', source)
        self.assertNotIn("st.columns([0.9, 1.6, 2.7, 1.1, 1.6, 1.6])", source)
        self.assertIn("st.container(border=True)", source)
        self.assertIn("Suggested filters", source)

    def test_compact_review_row_displays_plain_values_without_change_expander(self) -> None:
        source = inspect.getsource(app._render_compact_solicitation_review_row)
        self.assertIn("st.columns([0.08, 0.30, 0.62])", source)
        self.assertIn("st.selectbox(", source)
        self.assertIn("_solicitation_dropdown_options_for_item(item, available_filter_options)", source)
        self.assertNotIn('st.expander("Change selection"', source)
        self.assertNotIn("_solicitation_truncate", source)

    def test_evidence_control_is_not_in_normal_review_renderer(self) -> None:
        source = inspect.getsource(app._render_compact_solicitation_review_row)
        preview_source = inspect.getsource(app.render_solicitation_scope_preview)
        self.assertNotIn("cols[4]", source)
        self.assertNotIn("_solicitation_evidence_expander", source)
        self.assertNotIn("View evidence", preview_source)

    def test_review_widgets_do_not_mix_session_state_defaults_with_index(self) -> None:
        row_source = inspect.getsource(app._render_compact_solicitation_review_row)
        selected_source = inspect.getsource(app._solicitation_selected_review_value)
        self.assertIn("key=select_key", row_source)
        self.assertNotIn("index=", row_source)
        self.assertNotIn("st.session_state[select_key] =", selected_source)
        self.assertIn("_solicitation_initialize_widget_state(check_key", row_source)
        post_checkbox_source = row_source[row_source.index("st.checkbox(") :]
        self.assertNotIn("st.session_state[check_key] =", post_checkbox_source)

    def test_mapping_status_uses_compact_optional_filter_copy(self) -> None:
        source = inspect.getsource(app.render_solicitation_scope_preview)
        self.assertNotIn("Some dashboard filter options are temporarily unavailable", source)
        self.assertNotIn("Source coverage:", source)
        self.assertNotIn("Dashboard matching:", source)
        self.assertIn("Some optional filters were not available", source)
        self.assertNotIn("Retry filter matching", source)

    def test_removed_normal_solicitation_controls_are_absent(self) -> None:
        upload_source = inspect.getsource(app.render_upload_solicitation_package_panel)
        preview_source = inspect.getsource(app.render_solicitation_scope_preview)
        row_source = inspect.getsource(app._render_compact_solicitation_review_row)
        self.assertNotIn("Re-run Extraction", upload_source)
        self.assertNotIn("Load completed extraction", upload_source)
        self.assertNotIn("Active extraction engine", upload_source)
        self.assertNotIn("GOVCON_EXTRACTION_MODE", upload_source)
        self.assertNotIn("file(s) selected", upload_source)
        self.assertNotIn("Types:", upload_source)
        self.assertIn('st.caption(f"{len(uploaded_files)} files selected")', upload_source)
        self.assertNotIn("confirmed_filter_fields", preview_source)
        self.assertNotIn("Change selection", row_source)

    def test_developer_diagnostics_are_collapsed_and_separate(self) -> None:
        source = inspect.getsource(app.render_solicitation_scope_preview)
        self.assertIn('st.expander("Developer diagnostics", expanded=False)', source)
        self.assertIn('st.tabs(\n            ["Extracted details", "Diagnostics", "Advanced tools"]', source)
        self.assertIn("_render_extraction_diagnostics_content", source)
        self.assertIn("_render_solicitation_full_details_content", source)
        self.assertIn("_render_resolved_signals_upload_content", source)
        self.assertNotIn("_render_solicitation_full_details_expander", source)

    def test_diagnostic_content_helpers_do_not_create_expanders(self) -> None:
        for helper in (
            app._render_solicitation_full_details_content,
            app._render_extraction_diagnostics_content,
            app._render_resolved_signals_upload_content,
        ):
            source = inspect.getsource(helper)
            self.assertNotIn("st.expander", source)

    def test_scope_review_render_path_does_not_raise_confirmed_fields_name_error(self) -> None:
        option = encode_option("722310", "Food Service Contractors")
        row = self._confirmed_row("NAICS", "naics_code", option)
        scope_preview = {
            "Solicitation Number": {"value": "FA481926R0001"},
            "Title": {"value": "Tyndall Air Force Base Mess Attendant Recompete"},
            "Primary NAICS": {"value": "722310"},
        }
        mapping_result = {
            "rows": [row],
            "available_filter_options": _tyndall_available_options(),
            "debug": {},
            "pending_filters": {"market_filters": {}},
        }
        session = _SessionState(
            {
                "solicitation_resolved_signals": {
                    "version": app.RESOLVED_SIGNALS_VERSION,
                    "summary": {},
                    "signals": [],
                },
                "solicitation_scope_preview": scope_preview,
                "solicitation_scope_review_open": True,
                "solicitation_scope_applied": False,
                "solicitation_mapping_status": "not_started",
                "active_market_filters": app.default_market_filters(),
                "active_contracting_office": app.ALL_CONTRACTING_OFFICES,
            }
        )
        fake_st = _FakeStreamlit(session)
        with (
            patch.object(app, "st", fake_st),
            patch.object(app, "map_solicitation_signals_to_dashboard_filters", return_value=mapping_result),
            patch.object(app, "build_analysis_validation_metadata", return_value={}),
        ):
            rendered = app.render_solicitation_scope_preview(
                [{"agency_name": "Department of Defense", "toptier_code": "097"}],
                "Department of Defense",
                app.ALL_BUREAUS,
                2026,
                show_advanced_upload=False,
            )
        self.assertTrue(rendered)
        self.assertEqual(fake_st.expander_calls, [{"label": "Developer diagnostics", "expanded": False}])
        self.assertEqual(fake_st.tab_calls, [["Extracted details", "Diagnostics", "Advanced tools"]])

    def test_scope_review_diagnostics_tools_tab_keeps_single_expander(self) -> None:
        option = encode_option("722310", "Food Service Contractors")
        row = self._confirmed_row("NAICS", "naics_code", option)
        scope_preview = {
            "Solicitation Number": {"value": "FA481926R0001"},
            "Title": {"value": "Tyndall Air Force Base Mess Attendant Recompete"},
            "Primary NAICS": {"value": "722310"},
        }
        mapping_result = {
            "rows": [row],
            "available_filter_options": _tyndall_available_options(),
            "debug": {},
            "pending_filters": {"market_filters": {}},
        }
        session = _SessionState(
            {
                "solicitation_resolved_signals": {
                    "version": app.RESOLVED_SIGNALS_VERSION,
                    "summary": {},
                    "signals": [],
                },
                "solicitation_scope_preview": scope_preview,
                "solicitation_scope_review_open": True,
                "solicitation_scope_applied": False,
                "solicitation_mapping_status": "not_started",
                "active_market_filters": app.default_market_filters(),
                "active_contracting_office": app.ALL_CONTRACTING_OFFICES,
            }
        )
        fake_st = _FakeStreamlit(session)
        with (
            patch.object(app, "st", fake_st),
            patch.object(app, "map_solicitation_signals_to_dashboard_filters", return_value=mapping_result),
            patch.object(app, "build_analysis_validation_metadata", return_value={}),
        ):
            rendered = app.render_solicitation_scope_preview(
                [{"agency_name": "Department of Defense", "toptier_code": "097"}],
                "Department of Defense",
                app.ALL_BUREAUS,
                2026,
                show_advanced_upload=True,
            )
        self.assertTrue(rendered)
        self.assertEqual([call["label"] for call in fake_st.expander_calls], ["Developer diagnostics"])
        self.assertFalse(fake_st.expander_calls[0]["expanded"])
        self.assertEqual(fake_st.tab_calls, [["Extracted details", "Diagnostics", "Advanced tools"]])


class MarketConcentrationTests(unittest.TestCase):
    def _transaction_df(self, rows: list[tuple[str, float]]) -> pd.DataFrame:
        return pd.DataFrame(rows, columns=["Contractor Name", "Obligation Amount"])

    def test_positive_concentration_replaces_net_denominator_formula_that_can_exceed_100(self) -> None:
        transaction_df = self._transaction_df(
            [
                ("Alpha", 100.0),
                ("Bravo", 50.0),
                ("Charlie", 25.0),
                ("Delta", 20.0),
                ("Echo", 10.0),
                ("Negative Only", -100.0),
            ]
        )
        result = app.calculate_market_concentration(
            transaction_df,
            contractor_key_col="Contractor Name",
            obligation_col="Obligation Amount",
            top_n=5,
        )
        self.assertEqual(result["net_obligations"], 105.0)
        self.assertEqual(result["positive_obligations"], 205.0)
        self.assertEqual(result["negative_obligations"], -100.0)
        self.assertEqual(result["legacy_top_n_net_obligations"], 205.0)
        self.assertEqual(result["legacy_net_denominator"], 105.0)
        self.assertGreater(result["legacy_net_denominator_share"], 1.0)
        self.assertLessEqual(result["top_n_share"], 1.0)
        self.assertEqual(result["top_n_positive_obligations"], 205.0)

    def test_concentration_uses_positive_rows_and_excludes_zero_and_negative_only_contractors(self) -> None:
        transaction_df = self._transaction_df(
            [
                ("Alpha", 100.0),
                ("Alpha", -60.0),
                ("Bravo", 90.0),
                ("Charlie", 0.0),
                ("Negative Only", -10.0),
            ]
        )
        result = app.calculate_market_concentration(
            transaction_df,
            contractor_key_col="Contractor Name",
            obligation_col="Obligation Amount",
            top_n=5,
        )
        self.assertEqual(result["net_obligations"], 120.0)
        self.assertEqual(result["positive_obligations"], 190.0)
        self.assertEqual(result["negative_obligations"], -70.0)
        self.assertEqual([row["contractor"] for row in result["contractor_breakdown"]], ["Alpha", "Bravo"])
        self.assertAlmostEqual(result["contractor_breakdown"][0]["amount"], 100.0)
        self.assertAlmostEqual(result["contractor_breakdown"][0]["share"], 100.0 / 190.0)
        self.assertEqual(result["contractor_count_positive"], 2)

    def test_concentration_summary_breakdown_matches_positive_denominator(self) -> None:
        transaction_df = self._transaction_df(
            [
                ("Alpha", 100.0),
                ("Bravo", 80.0),
                ("Charlie", 20.0),
                ("Delta", -50.0),
            ]
        )
        summary, debug = app.market_concentration_summary(transaction_df, total_obligations=150.0)
        self.assertEqual(summary["subtitle"], "Top 5 share of positive obligations")
        self.assertEqual(summary["value"], "100.0%")
        self.assertEqual(debug["net_market_obligations"], 150.0)
        self.assertEqual(debug["gross_positive_obligations"], 200.0)
        self.assertEqual(debug["gross_negative_obligations"], -50.0)
        self.assertEqual(debug["top_5_positive_obligations"], 200.0)
        self.assertLessEqual(debug["top_5_positive_share"], 1.0)
        breakdown_pct = sum(
            row["percentage"]
            for row in summary["concentration_segments"]
            if row["contractor"] != "All Other Contractors"
        )
        self.assertAlmostEqual(breakdown_pct, debug["concentration_percentage"], places=1)
        self.assertIn("positive obligations", summary["supporting_text"])
        self.assertIn("Negative obligations remain included", summary["helper_text"])

    def test_no_positive_transactions_returns_na_safely(self) -> None:
        transaction_df = self._transaction_df(
            [
                ("Alpha", 0.0),
                ("Bravo", -10.0),
            ]
        )
        summary, debug = app.market_concentration_summary(transaction_df, total_obligations=-10.0)
        self.assertEqual(summary["value"], "N/A")
        self.assertEqual(summary["supporting_text"], "No positive obligation transactions in this scope.")
        self.assertIsNone(debug["top_5_positive_share"])
        self.assertIsNone(debug["concentration_percentage"])

    def test_negative_net_market_obligations_do_not_break_positive_concentration(self) -> None:
        transaction_df = self._transaction_df(
            [
                ("Alpha", 100.0),
                ("Bravo", -150.0),
            ]
        )
        summary, debug = app.market_concentration_summary(transaction_df, total_obligations=-50.0)
        self.assertEqual(debug["net_market_obligations"], -50.0)
        self.assertEqual(debug["gross_positive_obligations"], 100.0)
        self.assertEqual(summary["value"], "100.0%")
        self.assertEqual(debug["top_5_positive_share"], 1.0)

    def test_main_contractor_leaderboard_remains_net_based(self) -> None:
        transaction_df = self._transaction_df(
            [
                ("Alpha", 100.0),
                ("Alpha", -60.0),
                ("Bravo", 25.0),
            ]
        )
        leaderboard = app.transaction_vendor_dataframe(transaction_df)
        result = app.calculate_market_concentration(
            transaction_df,
            contractor_key_col="Contractor Name",
            obligation_col="Obligation Amount",
            top_n=5,
        )
        self.assertEqual(float(leaderboard.loc[leaderboard["recipient"] == "Alpha", "amount"].iloc[0]), 40.0)
        self.assertEqual(result["contractor_breakdown"][0]["contractor"], "Alpha")
        self.assertEqual(result["contractor_breakdown"][0]["amount"], 100.0)


if __name__ == "__main__":
    unittest.main()
