import csv
import hashlib
import html
import io
import json
import os
import re
import time
import zipfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from openai import OpenAI

from extraction.config import get_data_dir, get_extraction_mode, load_project_env
from extraction.streamlit_upload import (
    cleanup_staged_paths,
    ocr_environment_status,
    run_extraction_from_paths,
    validate_and_stage_uploads,
)
from extraction.persist import read_json
from solicitation_workflow import (
    AUTO_MAPPED_DASHBOARD_FILTER_FIELDS,
    build_fast_scope_detail_rows,
    build_solicitation_additional_filter_rows,
    count_auto_mapped_dashboard_filters,
    dedupe_extraction_findings,
    fast_scope_requested_count,
    is_ocr_environment_notice,
    is_solicitation_filter_removed,
    match_psc_to_options,
    resolve_solicitation_filter_pending_value,
    should_show_blocking_ocr_error,
    solicitation_scope_review_visible,
    solicitation_status_alert_text,
)
from pypdf import PdfReader


BASE_URL = "https://api.usaspending.gov"
APP_CACHE_VERSION = "v5_cache_reset_after_lane_mix_fields"
AWARD_TYPE_CODES = ["A", "B", "C", "D"]
AWARD_OR_IDV_FLAG = "AWARD"
ALL_BUREAUS = "All Bureaus"
NOT_APPLICABLE_BUREAU = "Not applicable"
OPENAI_MODEL = "gpt-4o-mini"
AGENCY_EXTRACTION_PROMPT = (
    "Extract the exact formal federal Awarding Agency from this text. "
    "Return strictly as valid JSON: {'agency': '...'}. If unknown, return null."
)
DEFAULT_AGENCY_NAME = "National Aeronautics and Space Administration"
DEFAULT_TOPTIER_CODE = "080"
DEFAULT_AGENCY_RECORD = {
    "agency_name": DEFAULT_AGENCY_NAME,
    "toptier_code": DEFAULT_TOPTIER_CODE,
    "abbreviation": "NASA",
}
CRIMSON = "#E11D48"
SYNC_ICON = "\U0001F504"
ALL_CONTRACTING_OFFICES = "All Contracting Offices"
ALL_FUNDING_OFFICES = "All Funding Offices"
CONTRACTING_OFFICE_SEPARATOR = "||"
ALL_NAICS_CODES = "All NAICS Codes"
ALL_CONTRACT_TYPES = "All Contract Types"
ALL_PRODUCT_SERVICE_CODES = "All Product / Service Codes"
ALL_SET_ASIDE_TYPES = "All Set-Aside Types"
ALL_POP_LOCATIONS = "All Locations"
LEGACY_ALL_POP_STATES = "All States"
OPTION_SEPARATOR = "||"
TIME_GRAIN_MONTH = "By Month"
TIME_GRAIN_FISCAL_QUARTER = "By Fiscal Quarter"
TIME_GRAIN_FISCAL_YEAR = "By Fiscal Year"
TIME_GRAIN_OPTIONS = [
    TIME_GRAIN_MONTH,
    TIME_GRAIN_FISCAL_QUARTER,
    TIME_GRAIN_FISCAL_YEAR,
]
LEADERBOARD_OBLIGATIONS = "Obligations This Period"
LEADERBOARD_CURRENT_VALUE = "Current Award Value"
LEADERBOARD_AWARD_CEILING = "Award Ceiling"
LEADERBOARD_OPTIONS = [
    LEADERBOARD_OBLIGATIONS,
    LEADERBOARD_CURRENT_VALUE,
    LEADERBOARD_AWARD_CEILING,
]
ANALYSIS_LOADING_MESSAGES = [
    "Connecting to USAspending...",
    "Pulling prime contract transactions...",
    "Reconciling FY contract obligations...",
    "Deduping award ceilings...",
    "Calculating current award value...",
    "Ranking top contractors...",
    "Building obligation trends...",
    "Finalizing market intelligence dashboard...",
]
AWARD_SCOPE_DOWNLOAD_COLUMNS = [
    "contract_award_unique_key",
    "award_id_piid",
    "modification_number",
    "transaction_number",
    "federal_action_obligation",
    "total_dollars_obligated",
    "total_outlayed_amount_for_overall_award",
    "current_total_value_of_award",
    "potential_total_value_of_award",
    "action_date",
    "recipient_name",
    "awarding_office_code",
    "awarding_office_name",
    "funding_office_code",
    "funding_office_name",
]
CONTRACT_TYPE_OPTIONS = {
    "3": "OTHER",
    "J": "FIRM FIXED PRICE",
    "U": "COST PLUS FIXED FEE",
    "S": "COST NO FEE",
    "R": "COST PLUS AWARD FEE",
    "T": "COST SHARING",
    "Y": "TIME AND MATERIALS",
    "K": "FIXED PRICE WITH ECONOMIC PRICE ADJUSTMENT",
    "L": "FIXED PRICE INCENTIVE",
    "M": "FIXED PRICE AWARD FEE",
    "V": "COST PLUS INCENTIVE FEE",
    "Z": "LABOR HOURS",
}
SET_ASIDE_TYPE_OPTIONS = {
    "NONE": "Unrestricted",
    "SBA": "Small Business Set-Aside",
    "SBP": "Small Business Partial Set-Aside",
    "8A": "8(a) Competed",
    "8AN": "8(a) Sole Source",
    "WOSB": "Women-Owned Small Business",
    "EDWOSB": "Economically Disadvantaged WOSB",
    "SDVOSBC": "Service-Disabled Veteran-Owned Small Business",
    "HZS": "HUBZone Sole Source",
    "HZC": "HUBZone Set-Aside",
}
STATE_OPTIONS = {
    "AL": "Alabama",
    "AK": "Alaska",
    "AZ": "Arizona",
    "AR": "Arkansas",
    "CA": "California",
    "CO": "Colorado",
    "CT": "Connecticut",
    "DC": "District of Columbia",
    "DE": "Delaware",
    "FL": "Florida",
    "GA": "Georgia",
    "HI": "Hawaii",
    "IA": "Iowa",
    "ID": "Idaho",
    "IL": "Illinois",
    "IN": "Indiana",
    "KS": "Kansas",
    "KY": "Kentucky",
    "LA": "Louisiana",
    "MA": "Massachusetts",
    "MD": "Maryland",
    "ME": "Maine",
    "MI": "Michigan",
    "MN": "Minnesota",
    "MO": "Missouri",
    "MS": "Mississippi",
    "MT": "Montana",
    "NC": "North Carolina",
    "ND": "North Dakota",
    "NE": "Nebraska",
    "NH": "New Hampshire",
    "NJ": "New Jersey",
    "NM": "New Mexico",
    "NV": "Nevada",
    "NY": "New York",
    "OH": "Ohio",
    "OK": "Oklahoma",
    "OR": "Oregon",
    "PA": "Pennsylvania",
    "RI": "Rhode Island",
    "SC": "South Carolina",
    "SD": "South Dakota",
    "TN": "Tennessee",
    "TX": "Texas",
    "UT": "Utah",
    "VA": "Virginia",
    "VT": "Vermont",
    "WA": "Washington",
    "WI": "Wisconsin",
    "WV": "West Virginia",
    "WY": "Wyoming",
}
POP_COUNTRY_OPTION_EXCLUSIONS = {"USA", "US", "FOREIGN"}
TERMINATION_ACTION_MAP = {
    "E": "Default",
    "F": "Convenience",
    "N": "Cancellation",
}
CORE_TRANSACTION_FIELDS = [
    "Award ID",
    "Mod",
    "Transaction Description",
    "Transaction Amount",
    "Action Date",
    "Recipient Name",
    "Action Type",
    "Awarding Office",
    "Awarding Office Code",
    "Awarding Office Name",
    "Funding Office",
    "Funding Office Code",
    "Funding Office Name",
]
TRANSACTION_FIELDS = [
    *CORE_TRANSACTION_FIELDS,
    "NAICS",
    "PSC",
    "Primary Place of Performance",
]
BASE_TRANSACTION_FIELDS = [
    field for field in CORE_TRANSACTION_FIELDS if "Office" not in field
]
CANCELLATION_TERMS = ("TERMINATION", "CANCEL", "CONVENIENCE", "DEFAULT")
EXPLICIT_TERMINATION_TERMS = ("TERMINAT",)
ADMINISTRATIVE_TERMS = (
    "ADMIN",
    "ADJUST",
    "CLOSEOUT",
    "CLOSE OUT",
    "CORRECT",
    "RECONCIL",
    "SETTLEMENT",
)
DEOBLIGATION_TERMS = ("DEOBLIG", "DE-OBLIG", "REDUCTION", "REDUCE", "DECREASE")
RESOLVED_SIGNALS_VERSION = "resolved_signals.v1"
KEEP_CURRENT_SOLICITATION_FILTER = "— Keep current —"
KEEP_REMOVED_SOLICITATION_FILTER = "— Keep removed —"
SOLICITATION_MAPPING_UNAVAILABLE_STATUS = "Mapping temporarily unavailable"
# Dashboard filters that require explicit user confirmation before apply.
SOLICITATION_USER_CONFIRMATION_FILTER_KEYS = frozenset({"pop_state", "funding_office", "contract_type"})
SOLICITATION_POP_CAUTION_NOTE = (
    "Place of Performance may include multiple locations or contractor/remote work. "
    "This filter is always confirmed by the analyst."
)
SOLICITATION_MARKET_REVIEW_FIELDS: tuple[tuple[str, str | None], ...] = (
    ("Agency", "agency"),
    ("Subagency / Bureau", "bureau"),
    ("Contracting Office", "contracting_office"),
    ("AAC", None),
    ("NAICS", "naics_code"),
    ("PSC", "psc_code"),
    ("Contract Type", "contract_type"),
    ("Set-Aside", "set_aside_type"),
    ("Place of Performance", "pop_state"),
    ("Funding Office", "funding_office"),
)
SOLICITATION_MISSING_MARKET_FIELDS = {"Contracting Office", "AAC", "NAICS", "PSC", "Set-Aside"}
SOLICITATION_SOURCE_COMPLETENESS_CRITICAL_FIELDS = {
    "Contracting Office",
    "NAICS",
    "PSC",
    "Contract Type",
    "Set-Aside",
}
SOLICITATION_CONFIRMED_VALIDATION_STATUSES = {
    "confirmed",
    "validated",
    "validated_model_extraction",
    "deterministic_hierarchy",
}
EXTRACTION_STATUSES = {"idle", "running", "completed", "timed_out", "failed"}
MAPPING_STATUSES = {"not_started", "running", "completed", "partial", "timed_out", "failed"}
REVIEW_STATUSES = {"not_ready", "ready", "displayed"}
# Deterministic top-tier agency + subagency/bureau pairs for common solicitation org values.
SOLICITATION_ORGANIZATION_HIERARCHY: dict[str, tuple[str, str]] = {
    "doi": ("Department of the Interior", ALL_BUREAUS),
    "department of interior": ("Department of the Interior", ALL_BUREAUS),
    "department of the interior": ("Department of the Interior", ALL_BUREAUS),
    "bureau of reclamation": ("Department of the Interior", "Bureau of Reclamation"),
    "reclamation": ("Department of the Interior", "Bureau of Reclamation"),
    "bor": ("Department of the Interior", "Bureau of Reclamation"),
    "usbr": ("Department of the Interior", "Bureau of Reclamation"),
    "department of the air force": ("Department of Defense", "Department of the Air Force"),
    "air force": ("Department of Defense", "Department of the Air Force"),
    "usaf": ("Department of Defense", "Department of the Air Force"),
    "department of the army": ("Department of Defense", "Department of the Army"),
    "army": ("Department of Defense", "Department of the Army"),
    "department of the navy": ("Department of Defense", "Department of the Navy"),
    "navy": ("Department of Defense", "Department of the Navy"),
    "marine corps": ("Department of Defense", "Department of the Navy"),
    "usmc": ("Department of Defense", "Department of the Navy"),
    "defense logistics agency": ("Department of Defense", "Defense Logistics Agency"),
    "dla": ("Department of Defense", "Defense Logistics Agency"),
    "department of state": ("Department of State", ALL_BUREAUS),
    "state department": ("Department of State", ALL_BUREAUS),
    "dos": ("Department of State", ALL_BUREAUS),
    "nasa": ("National Aeronautics and Space Administration", ALL_BUREAUS),
    "national aeronautics and space administration": (
        "National Aeronautics and Space Administration",
        ALL_BUREAUS,
    ),
}
SOLICITATION_CONTRACT_TYPE_ALIASES = {
    "ffp": "J",
    "firm fixed price": "J",
    "cpff": "U",
    "cost plus fixed fee": "U",
    "t&m": "Y",
    "time & materials": "Y",
    "time material": "Y",
    "time and materials": "Y",
    "labor hour": "Z",
    "labor hours": "Z",
}
AGENCY_ALIASES = {
    "doi": "Department of the Interior",
    "interior": "Department of the Interior",
    "dod": "Department of Defense",
    "defense": "Department of Defense",
    "va": "Department of Veterans Affairs",
    "veterans": "Department of Veterans Affairs",
    "hhs": "Department of Health and Human Services",
    "health": "Department of Health and Human Services",
    "dhs": "Department of Homeland Security",
    "homeland": "Department of Homeland Security",
    "nasa": "National Aeronautics and Space Administration",
    "national aeronautics and space administration": "National Aeronautics and Space Administration",
    "nsf": "National Science Foundation",
    "science foundation": "National Science Foundation",
    "nrc": "Nuclear Regulatory Commission",
    "nuclear regulatory": "Nuclear Regulatory Commission",
    "energy": "Department of Energy",
    "doe": "Department of Energy",
    "gsa": "General Services Administration",
    "usaid": "Agency for International Development",
    "epa": "Environmental Protection Agency",
}


st.set_page_config(
    page_title="GovCon Pulse",
    layout="wide",
    initial_sidebar_state="expanded",
)


def request_headers() -> dict:
    return {
        "Accept": "application/json",
        "User-Agent": "govcon-pulse-streamlit/1.0",
    }


def env_float(name: str, default: float, minimum: float) -> float:
    raw = (os.getenv(name) or "").strip()
    if not raw:
        return default
    try:
        return max(minimum, float(raw))
    except ValueError:
        return default


def solicitation_option_request_timeout_sec() -> float:
    return env_float("GOVCON_SOLICITATION_OPTION_TIMEOUT_SEC", 20.0, 5.0)


def solicitation_mapping_timeout_sec() -> float:
    return env_float("GOVCON_SOLICITATION_MAPPING_TIMEOUT_SEC", 45.0, 15.0)


class SolicitationMappingTimeout(TimeoutError):
    pass


def _solicitation_mapping_stage(
    stage: str,
    started_at: float,
    status: str,
    *,
    function: str,
    exc: Exception | None = None,
) -> dict:
    ended_at = time.time()
    event = {
        "stage": stage,
        "status": status,
        "function": function,
        "startTime": started_at,
        "endTime": ended_at,
        "elapsedSeconds": round(ended_at - started_at, 3),
        "heartbeatAt": ended_at,
    }
    if exc is not None:
        event["exceptionClass"] = type(exc).__name__
        event["exceptionMessage"] = str(exc)
    return event


def _check_solicitation_mapping_deadline(deadline: float, stage: str) -> None:
    if time.monotonic() > deadline:
        raise SolicitationMappingTimeout(f"{stage} exceeded solicitation filter matching timeout")


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_toptier_agencies(cache_version: str = APP_CACHE_VERSION) -> list[dict]:
    try:
        response = requests.get(
            f"{BASE_URL}/api/v2/references/toptier_agencies/",
            headers=request_headers(),
            timeout=18,
        )
        response.raise_for_status()
        results = response.json().get("results") or []
    except (requests.RequestException, ValueError):
        return []

    records = []
    seen = set()
    for item in results:
        if not isinstance(item, dict):
            continue
        name = str(item.get("agency_name") or "").strip()
        code = str(item.get("toptier_code") or "").strip()
        active_fy = item.get("active_fy")
        if not name or not code or not active_fy:
            continue
        key = name.lower()
        if key in seen:
            continue
        records.append(
            {
                "agency_name": name,
                "toptier_code": code,
                "abbreviation": str(item.get("abbreviation") or "").strip(),
            }
        )
        seen.add(key)

    return sorted(records, key=lambda record: record["agency_name"])


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_subagencies(
    toptier_code: str,
    fiscal_year: int,
    cache_version: str = APP_CACHE_VERSION,
) -> list[str]:
    if not toptier_code:
        return [ALL_BUREAUS]

    all_results = []
    page = 1
    try:
        while True:
            response = requests.get(
                f"{BASE_URL}/api/v2/agency/{toptier_code}/sub_agency/",
                params={"fiscal_year": int(fiscal_year), "page": page},
                headers=request_headers(),
                timeout=18,
            )
            response.raise_for_status()
            payload = response.json()
            all_results.extend(payload.get("results") or [])

            page_metadata = payload.get("page_metadata") or {}
            if not page_metadata.get("hasNext"):
                break
            page += 1
    except (requests.RequestException, ValueError, TypeError):
        return [ALL_BUREAUS]

    names = []
    for item in all_results:
        if isinstance(item, str):
            names.append(item.strip())
            continue
        if not isinstance(item, dict):
            continue
        nested = item.get("subtier_agency") or item.get("agency")
        nested_name = nested.get("name") if isinstance(nested, dict) else None
        name = (
            item.get("name")
            or item.get("agency_name")
            or item.get("subagency_name")
            or item.get("sub_agency_name")
            or item.get("subtier_name")
            or item.get("bureau_name")
            or nested_name
        )
        if name:
            names.append(str(name).strip())

    unique_names = sorted({name for name in names if name and name != ALL_BUREAUS})
    return [ALL_BUREAUS] + unique_names


def agency_names_from_records(agency_records: list[dict]) -> list[str]:
    return [record["agency_name"] for record in agency_records if record.get("agency_name")]


def agency_record_by_name(agency_records: list[dict], agency_name: str | None) -> dict:
    normalized_name = normalize_agency_name(agency_name)
    for record in agency_records:
        if record.get("agency_name", "").lower() == normalized_name.lower():
            return record
    return agency_records[0] if agency_records else DEFAULT_AGENCY_RECORD


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg: #0b0c10;
            --panel: #161922;
            --panel-2: #1f2430;
            --line: rgba(255, 255, 255, 0.11);
            --text-primary: #f4f7fb;
            --text-secondary: #dce5ef;
            --text-muted: #bac6d4;
            --text-disabled: #9aa7b8;
            --text-link: #67e8f9;
            --text-warning: #fcd34d;
            --text-error: #fda4af;
            --text-success: #86efac;
            --text: var(--text-primary);
            --muted: var(--text-muted);
            --teal: #2dd4bf;
            --cyan: #38bdf8;
            --amber: #f59e0b;
            --rose: #fb7185;
        }
        .stApp {
            background:
                linear-gradient(135deg, rgba(45, 212, 191, 0.10), transparent 28%),
                linear-gradient(315deg, rgba(245, 158, 11, 0.09), transparent 26%),
                var(--bg);
            color: var(--text);
        }
        html,
        body,
        [data-testid="stAppViewContainer"],
        .stApp {
            overflow-y: auto;
        }
        [data-testid="stWidgetLabel"],
        [data-testid="stWidgetLabel"] *,
        label,
        label * {
            color: var(--text-secondary) !important;
        }
        [data-testid="stCaptionContainer"],
        [data-testid="stCaptionContainer"] * {
            color: var(--text-muted) !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] {
            background: #11131a;
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
            padding: 24px 18px 28px;
        }
        [data-testid="stSidebar"] * {
            color: var(--text);
        }
        [data-testid="stSidebar"] .stRadio,
        [data-testid="stSidebar"] .stSelectbox,
        [data-testid="stSidebar"] .stFileUploader,
        [data-testid="stSidebar"] .stButton {
            margin-bottom: 14px;
        }
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] legend {
            color: var(--text-primary) !important;
            font-size: 13px !important;
            font-weight: 800 !important;
            letter-spacing: 0 !important;
        }
        div[data-baseweb="select"] > div,
        div[data-baseweb="select"] *,
        div[data-baseweb="input"] *,
        input,
        textarea {
            color: #1A1A1A !important;
        }
        div[data-baseweb="select"] > div,
        input,
        textarea {
            background: #ffffff !important;
            border-radius: 8px !important;
            min-height: 44px;
        }
        div[data-baseweb="select"] svg {
            color: #1A1A1A !important;
            fill: #1A1A1A !important;
        }
        input::placeholder,
        textarea::placeholder {
            color: #525252 !important;
            opacity: 1 !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] div[data-baseweb="select"] *,
        [data-testid="stSidebar"] div[data-baseweb="input"] *,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea,
        [data-testid="stSidebar"] .stFileUploader section,
        [data-testid="stSidebar"] .stFileUploader section * {
            color: #1A1A1A !important;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] > div,
        [data-testid="stSidebar"] input,
        [data-testid="stSidebar"] textarea {
            background: #ffffff !important;
            border-color: rgba(255, 255, 255, 0.42) !important;
            border-radius: 8px !important;
            min-height: 44px;
        }
        [data-testid="stSidebar"] div[data-baseweb="select"] svg {
            color: #1A1A1A !important;
            fill: #1A1A1A !important;
        }
        [data-testid="stDataFrame"] *,
        [data-testid="stTable"] *,
        div[data-testid="stDataFrame"] div,
        div[data-testid="stDataFrame"] span {
            color: var(--text-secondary) !important;
        }
        [data-testid="stDataFrame"],
        [data-testid="stTable"] {
            background: rgba(9, 14, 27, 0.60) !important;
            border-radius: 12px;
        }
        [data-testid="stSidebar"] input::placeholder,
        [data-testid="stSidebar"] textarea::placeholder {
            color: #525252 !important;
            opacity: 1 !important;
        }
        div[data-baseweb="popover"] div[role="listbox"],
        div[data-baseweb="popover"] div[role="option"],
        div[data-baseweb="popover"] div[role="option"] *,
        ul[data-testid="stVirtualDropdown"] li,
        ul[data-testid="stVirtualDropdown"] li * {
            color: #1A1A1A !important;
            background-color: #ffffff !important;
        }
        [data-testid="stSidebar"] .sidebar-title {
            color: var(--text-primary);
            font-size: 22px;
            font-weight: 850;
            line-height: 1.12;
            letter-spacing: 0;
            margin: 0 0 6px;
        }
        [data-testid="stSidebar"] .sidebar-subtitle {
            color: var(--text-muted);
            font-size: 13px;
            line-height: 1.35;
            margin: 0 0 18px;
        }
        [data-testid="stSidebar"] .sidebar-section {
            color: var(--teal);
            font-size: 11px;
            font-weight: 850;
            letter-spacing: 0;
            text-transform: uppercase;
            margin: 18px 0 8px;
        }
        .landing-shell {
            min-height: 72vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 56px 0 32px;
        }
        .landing-panel {
            width: 100%;
            max-width: 760px;
            text-align: center;
            padding: 42px 34px 38px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background:
                linear-gradient(145deg, rgba(22, 25, 34, 0.96), rgba(31, 36, 48, 0.90)),
                linear-gradient(90deg, rgba(45, 212, 191, 0.14), rgba(245, 158, 11, 0.08));
            box-shadow: 0 26px 70px rgba(0, 0, 0, 0.30);
        }
        .landing-panel h1 {
            color: var(--text);
            font-size: 42px;
            line-height: 1.08;
            font-weight: 850;
            letter-spacing: 0;
            margin: 0 0 14px;
        }
        .landing-panel p {
            color: var(--muted);
            font-size: 16px;
            line-height: 1.5;
            margin: 0 auto;
            max-width: 620px;
        }
        .landing-control-spacer {
            height: clamp(42px, 12vh, 110px);
        }
        .landing-title {
            color: var(--text);
            font-size: 40px;
            line-height: 1.1;
            font-weight: 850;
            letter-spacing: 0;
            margin: 0 auto 12px;
            text-align: center;
        }
        .landing-subtitle {
            color: var(--muted);
            font-size: 16px;
            line-height: 1.5;
            margin: 0 auto 22px;
            max-width: 620px;
            text-align: center;
        }
        .workflow-section-label {
            color: var(--text);
            font-size: 15px;
            font-weight: 850;
            margin: 0 0 4px;
        }
        .workflow-section-helper {
            color: var(--text-muted);
            font-size: 13px;
            line-height: 1.45;
            margin: 0 0 14px;
        }
        .workflow-or-divider {
            display: flex;
            align-items: center;
            gap: 14px;
            margin: 28px 0 24px;
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
        }
        .workflow-or-divider::before,
        .workflow-or-divider::after {
            content: "";
            flex: 1;
            height: 1px;
            background: rgba(148, 163, 184, 0.22);
        }
        .workflow-cta-title {
            color: var(--text);
            font-size: 18px;
            font-weight: 850;
            margin: 0 0 6px;
        }
        .workflow-cta-subtitle {
            color: var(--muted);
            font-size: 14px;
            line-height: 1.45;
            margin: 0 0 8px;
        }
        .workflow-cta-helper {
            color: var(--text-muted);
            font-size: 13px;
            line-height: 1.45;
            margin: 0 0 14px;
        }
        .workflow-test-mode-note {
            color: var(--text-muted);
            font-size: 11px;
            line-height: 1.4;
            margin-top: 8px;
        }
        .hero {
            padding: 34px 36px 30px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background:
                linear-gradient(120deg, rgba(22, 25, 34, 0.96), rgba(31, 36, 48, 0.86)),
                linear-gradient(90deg, rgba(45, 212, 191, 0.14), rgba(251, 113, 133, 0.10));
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.28);
            margin-bottom: 20px;
        }
        .eyebrow {
            color: var(--teal);
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0;
            text-transform: uppercase;
            margin-bottom: 10px;
        }
        .hero h1 {
            color: var(--text);
            font-size: 56px;
            line-height: 1.02;
            font-weight: 900;
            letter-spacing: 0;
            margin: 0 0 12px;
            max-width: 1120px;
        }
        .hero .brand-subtitle {
            color: var(--muted);
            font-size: 13px;
            font-weight: 700;
            line-height: 1.35;
            margin: 0;
        }
        @media (max-width: 760px) {
            .hero {
                padding: 26px 22px 24px;
            }
            .hero h1 {
                font-size: 34px;
                line-height: 1.08;
            }
        }
        .metric-card {
            min-height: 136px;
            padding: 22px 22px 20px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(31, 36, 48, 0.98), rgba(22, 25, 34, 0.98));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 42px rgba(0,0,0,0.20);
            position: relative;
            overflow: hidden;
        }
        .metric-card:before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--accent);
        }
        .metric-label {
            color: var(--muted);
            font-size: 13px;
            font-weight: 700;
            margin-bottom: 12px;
        }
        .metric-value {
            color: var(--metric-value-color, var(--text));
            font-size: 32px;
            line-height: 1.05;
            font-weight: 850;
            letter-spacing: 0;
        }
        .metric-sub {
            color: var(--muted);
            font-size: 12px;
            margin-top: 12px;
        }
        .metric-helper {
            color: var(--text-muted);
            font-size: 11px;
            line-height: 1.35;
            margin-top: 8px;
        }
        .source-chip {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 8px 12px;
            border: 1px solid var(--line);
            border-radius: 999px;
            background: rgba(22, 25, 34, 0.80);
            color: var(--muted);
            font-size: 12px;
            font-weight: 700;
            margin: 2px 0 16px;
        }
        .source-dot {
            width: 8px;
            height: 8px;
            border-radius: 999px;
            background: var(--teal);
            box-shadow: 0 0 18px var(--teal);
        }
        .analysis-loading-card {
            border: 1px solid rgba(45, 212, 191, 0.28);
            border-radius: 16px;
            padding: 22px 24px;
            margin: 14px 0 20px;
            background:
                linear-gradient(145deg, rgba(22, 25, 34, 0.98), rgba(31, 36, 48, 0.94)),
                linear-gradient(90deg, rgba(45, 212, 191, 0.14), rgba(56, 189, 248, 0.08));
            box-shadow: 0 22px 55px rgba(0, 0, 0, 0.26);
            position: relative;
            overflow: hidden;
        }
        .analysis-loading-card:before {
            content: "";
            position: absolute;
            inset: 0 auto 0 0;
            width: 4px;
            background: linear-gradient(180deg, var(--teal), var(--cyan));
        }
        .analysis-loading-body {
            color: var(--text);
            font-size: 16px;
            line-height: 1.55;
            max-width: 980px;
            margin: 0 0 16px;
        }
        .analysis-loading-detail {
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            align-items: center;
            color: var(--muted);
            font-size: 14px;
            font-weight: 700;
            margin-bottom: 14px;
        }
        .analysis-loading-spinner {
            width: 18px;
            height: 18px;
            border-radius: 999px;
            border: 2px solid rgba(170, 180, 194, 0.30);
            border-top-color: var(--teal);
            animation: analysisSpin 0.85s linear infinite;
            flex: 0 0 auto;
        }
        .analysis-loading-stage {
            position: relative;
            min-height: 20px;
            color: var(--teal);
            font-weight: 850;
        }
        .analysis-loading-stage span {
            display: block;
            opacity: 0;
            animation: analysisStageCycle 16s linear infinite;
        }
        .analysis-loading-stage span:not(:first-child) {
            position: absolute;
            inset: 0 auto auto 0;
        }
        .analysis-loading-counts {
            display: flex;
            flex-wrap: wrap;
            gap: 10px;
            margin: 12px 0 14px;
        }
        .analysis-loading-count {
            min-width: 210px;
            padding: 12px 14px;
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 12px;
            background: rgba(9, 14, 27, 0.54);
        }
        .analysis-loading-count-value {
            color: var(--text);
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
            font-size: 24px;
            font-weight: 900;
            letter-spacing: 0.02em;
            font-variant-numeric: tabular-nums;
            animation: analysisCountSettle 0.5s ease-out both;
        }
        .analysis-loading-count-label {
            color: var(--muted);
            font-size: 12px;
            font-weight: 800;
            margin-top: 3px;
            text-transform: uppercase;
        }
        .analysis-loading-footer {
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 700;
        }
        .extraction-loading-card .extraction-doc-name {
            font-size: 16px;
            line-height: 1.25;
            word-break: break-word;
        }
        @keyframes analysisSpin {
            to { transform: rotate(360deg); }
        }
        @keyframes analysisStageCycle {
            0%, 10% { opacity: 1; transform: translateY(0); }
            12%, 100% { opacity: 0; transform: translateY(-6px); }
        }
        @keyframes analysisCountSettle {
            from { opacity: 0; transform: translateY(8px); filter: blur(2px); }
            to { opacity: 1; transform: translateY(0); filter: blur(0); }
        }
        .market-scope {
            border: 1px solid var(--line);
            border-radius: 18px;
            padding: 18px 20px;
            margin: 10px 0 20px;
            background: rgba(9, 14, 27, 0.72);
            box-shadow: 0 18px 45px rgba(0, 0, 0, 0.22);
        }
        .market-scope-title {
            color: var(--text);
            font-size: 18px;
            font-weight: 850;
            margin-bottom: 8px;
        }
        .market-scope-line {
            color: var(--muted);
            font-size: 14px;
            line-height: 1.45;
        }
        .filter-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 8px 0 12px;
        }
        .filter-chip {
            display: inline-flex;
            align-items: center;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.35);
            padding: 5px 10px;
            color: var(--muted);
            background: rgba(148, 163, 184, 0.10);
            font-size: 12px;
            font-weight: 700;
        }
        .filter-chip.active {
            color: #061015;
            background: linear-gradient(135deg, #2dd4bf, #38bdf8);
            border-color: rgba(45, 212, 191, 0.75);
        }
        .applied-filter-heading {
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 800;
            margin: 14px 0 6px;
        }
        .applied-filter-chip-row {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin: 8px 0 12px;
        }
        .applied-filter-chip-row [data-testid="column"] {
            width: auto !important;
            flex: 0 1 auto !important;
            min-width: 0 !important;
        }
        .applied-filter-chip-row .stButton>button,
        [class*="st-key-remove-applied-filter-"] button {
            min-height: 0;
            width: auto;
            border-radius: 999px;
            border: 1px solid rgba(148, 163, 184, 0.30) !important;
            background: #20242f !important;
            color: var(--text-primary) !important;
            padding: 5px 10px !important;
            font-size: 12px !important;
            font-weight: 800 !important;
            line-height: 1.25;
            box-shadow: none;
            transform: none;
        }
        .other-filters {
            color: var(--muted);
            font-size: 13px;
            margin-top: 2px;
        }
        .audit-heading {
            color: var(--text);
            font-size: 22px;
            font-weight: 850;
            letter-spacing: 0;
            margin: 26px 0 12px;
        }
        .section-spacer {
            height: 8px;
        }
        .debug-section-note {
            color: var(--text-muted);
            font-size: 12px;
            margin: 8px 0 4px;
        }
        .market-summary-panel {
            border: 1px solid rgba(45, 212, 191, 0.18);
            border-radius: 14px;
            padding: 14px 16px 12px;
            margin: 14px 0 18px;
            background: rgba(9, 14, 27, 0.50);
        }
        .market-summary-title {
            color: var(--text);
            font-size: 16px;
            font-weight: 850;
            margin-bottom: 4px;
        }
        .market-summary-helper {
            color: var(--text-muted);
            font-size: 12px;
            margin-bottom: 10px;
        }
        .solicitation-status-badge {
            display: inline-block;
            padding: 3px 9px;
            border-radius: 999px;
            font-size: 11px;
            font-weight: 800;
            line-height: 1.2;
            white-space: nowrap;
        }
        .solicitation-status-exact {
            background: rgba(45, 212, 191, 0.18);
            color: #99f6e4;
            border: 1px solid rgba(45, 212, 191, 0.35);
        }
        .solicitation-status-suggested {
            background: rgba(56, 189, 248, 0.16);
            color: #7dd3fc;
            border: 1px solid rgba(56, 189, 248, 0.32);
        }
        .solicitation-status-confirm {
            background: rgba(245, 158, 11, 0.16);
            color: var(--text-warning);
            border: 1px solid rgba(245, 158, 11, 0.32);
        }
        .solicitation-status-unmapped {
            background: rgba(251, 113, 133, 0.14);
            color: var(--text-error);
            border: 1px solid rgba(251, 113, 133, 0.30);
        }
        .solicitation-status-context {
            background: rgba(100, 116, 139, 0.18);
            color: #cbd5e1;
            border: 1px solid rgba(100, 116, 139, 0.30);
        }
        .solicitation-status-auto {
            background: rgba(34, 197, 94, 0.14);
            color: var(--text-success);
            border: 1px solid rgba(34, 197, 94, 0.30);
        }
        .solicitation-status-manual {
            background: rgba(167, 139, 250, 0.16);
            color: #c4b5fd;
            border: 1px solid rgba(167, 139, 250, 0.30);
        }
        .solicitation-status-removed {
            background: rgba(244, 63, 94, 0.14);
            color: var(--text-error);
            border: 1px solid rgba(244, 63, 94, 0.28);
        }
        .solicitation-status-alert-card {
            border: 1px solid rgba(245, 158, 11, 0.35);
            background: rgba(245, 158, 11, 0.10);
            border-radius: 12px;
            padding: 10px 12px;
            margin: 8px 0 14px;
            color: var(--text-warning);
            font-size: 13px;
            font-weight: 650;
        }
        .solicitation-scope-muted {
            color: var(--text-muted);
            font-size: 12px;
            margin-bottom: 8px;
        }
        .solicitation-baseline-line {
            color: var(--text-muted);
            font-size: 12px;
            margin: 0 0 10px;
        }
        .text-secondary {
            color: var(--text-secondary);
        }
        .text-muted {
            color: var(--text-muted);
        }
        .scope-review-helper {
            color: var(--text-muted);
            font-size: 12px;
            line-height: 1.45;
            margin: 2px 0 8px;
        }
        .scope-review-summary {
            color: var(--text-secondary);
            font-size: 13px;
            line-height: 1.45;
            margin: 4px 0 12px;
        }
        .solicitation-summary-card {
            border: 1px solid rgba(45, 212, 191, 0.18);
            border-radius: 12px;
            padding: 12px 14px;
            margin: 10px 0 16px;
            background: rgba(9, 14, 27, 0.42);
        }
        .solicitation-trust-note {
            color: var(--text-muted);
            font-size: 12px;
            line-height: 1.45;
            margin: 10px 0 8px;
            padding: 10px 12px;
            border-radius: 10px;
            border: 1px solid rgba(148, 163, 184, 0.18);
            background: rgba(9, 14, 27, 0.36);
        }
        .solicitation-preview-table {
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            margin: 8px 0 14px;
        }
        .solicitation-preview-table th,
        .solicitation-preview-table td {
            padding: 8px 10px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.16);
            vertical-align: top;
        }
        .solicitation-preview-table th {
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }
        .solicitation-preview-table td {
            color: var(--text-secondary);
        }
        .solicitation-review-note {
            color: var(--text-muted);
            font-size: 12px;
            margin: 4px 0 10px;
        }
        .award-drilldown-table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid rgba(148, 163, 184, 0.20);
            border-radius: 14px;
            background: rgba(9, 14, 27, 0.74);
            margin-top: 10px;
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.035), 0 18px 44px rgba(0,0,0,0.22);
        }
        .award-drilldown-table-wrap--scroll {
            max-height: 430px;
            overflow-y: auto;
        }
        .award-drilldown-table {
            width: 100%;
            border-collapse: collapse;
            color: var(--text-secondary);
            font-size: 13px;
            line-height: 1.35;
        }
        .award-drilldown-table th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #111827;
            color: var(--text-primary);
            font-weight: 850;
            text-align: left;
            white-space: nowrap;
            padding: 11px 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.20);
        }
        .award-drilldown-table td {
            max-width: 260px;
            padding: 9px 12px;
            border-bottom: 1px solid rgba(148, 163, 184, 0.13);
            vertical-align: top;
            color: var(--text-secondary);
            background: rgba(15, 23, 42, 0.32);
        }
        .award-drilldown-table tr:nth-child(even) td {
            background: rgba(15, 23, 42, 0.46);
        }
        .award-drilldown-table tr:hover td {
            background: rgba(45, 212, 191, 0.08);
        }
        .award-drilldown-table td:nth-child(3),
        .award-drilldown-table td:nth-child(8),
        .award-drilldown-table td:nth-child(9) {
            min-width: 220px;
        }
        .award-drilldown-table a {
            color: var(--text-link);
            font-weight: 800;
            text-decoration: none;
            white-space: nowrap;
        }
        .award-drilldown-table a:hover {
            text-decoration: underline;
        }
        .market-intel-card,
        .market-intel-note {
            min-height: 136px;
            padding: 18px 18px 16px;
            border: 1px solid var(--line);
            border-radius: 8px;
            background: linear-gradient(180deg, rgba(31, 36, 48, 0.98), rgba(22, 25, 34, 0.98));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 18px 42px rgba(0,0,0,0.20);
            position: relative;
            overflow: hidden;
        }
        .market-intel-card:before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: var(--accent, var(--teal));
        }
        .market-intel-label {
            color: var(--muted);
            font-size: 13px;
            font-weight: 800;
            margin-bottom: 8px;
        }
        .market-intel-value {
            color: var(--text);
            font-size: 30px;
            line-height: 1.05;
            font-weight: 900;
        }
        .market-intel-subtitle {
            color: var(--muted);
            font-size: 12px;
            font-weight: 800;
            margin-top: 6px;
        }
        .market-intel-helper {
            color: var(--text-muted);
            font-size: 12px;
            line-height: 1.35;
            margin-top: 8px;
        }
        .market-concentration-bar {
            display: flex;
            width: 100%;
            height: 14px;
            margin: 12px 0 8px;
            overflow: hidden;
            border-radius: 999px;
            background: rgba(148, 163, 184, 0.20);
        }
        .market-concentration-segment {
            min-width: 2px;
            height: 100%;
        }
        .market-concentration-legend {
            margin-top: 10px;
        }
        .market-concentration-legend-heading {
            color: var(--text-muted);
            font-size: 12px;
            font-weight: 600;
            margin-bottom: 8px;
        }
        .market-concentration-legend-row {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 12px;
            padding: 8px 12px 8px 10px;
            margin-bottom: 7px;
            border-radius: 9px;
            border-left: 4px solid;
        }
        .market-concentration-legend-name {
            flex: 1 1 auto;
            min-width: 0;
            color: var(--text-primary);
            font-size: 13px;
            font-weight: 750;
            line-height: 1.35;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }
        .market-concentration-legend-metrics {
            display: flex;
            align-items: center;
            justify-content: flex-end;
            gap: 8px;
            flex-shrink: 0;
        }
        .market-concentration-legend-badge {
            padding: 4px 10px;
            border-radius: 8px;
            border: 1px solid;
            background: rgba(9, 14, 27, 0.58);
            font-size: 12px;
            font-weight: 750;
            line-height: 1.2;
            white-space: nowrap;
        }
        .market-intel-note {
            display: flex;
            align-items: center;
            min-height: 84px;
            color: var(--text-muted);
            font-size: 13px;
            line-height: 1.4;
            background: rgba(9, 14, 27, 0.52);
        }
        h2, h3, label, .stMarkdown, .stSelectbox, .stTextInput, .stRadio {
            letter-spacing: 0 !important;
        }
        .stButton>button {
            border-radius: 8px;
            border: 1px solid rgba(45, 212, 191, 0.45);
            background: linear-gradient(135deg, #2dd4bf, #38bdf8);
            color: #061015;
            font-weight: 800;
            min-height: 42px;
        }
        .stButton>button:hover {
            border-color: #f59e0b;
            color: #061015;
        }
        .stDownloadButton>button {
            border-radius: 8px;
            border: 1px solid rgba(45, 212, 191, 0.45);
            background: linear-gradient(135deg, #2dd4bf, #38bdf8);
            color: #061015;
            font-weight: 800;
            min-height: 42px;
        }
        .stButton>button:disabled,
        .stDownloadButton>button:disabled {
            background: rgba(148, 163, 184, 0.16) !important;
            border-color: rgba(148, 163, 184, 0.24) !important;
            color: var(--text-disabled) !important;
            opacity: 1 !important;
            box-shadow: none !important;
        }
        .control-button-spacer {
            height: 28px;
        }
        [data-testid="stMetricValue"] {
            color: var(--text);
        }
        [data-testid="stExpander"] {
            background: rgba(9, 14, 27, 0.72) !important;
            border: 1px solid rgba(148, 163, 184, 0.20) !important;
            border-radius: 12px !important;
            overflow: hidden;
        }
        [data-testid="stExpander"] details {
            background: transparent !important;
            border: none !important;
        }
        [data-testid="stExpander"] summary,
        [data-testid="stExpander"] summary:hover,
        [data-testid="stExpander"] summary:focus {
            background: rgba(15, 23, 42, 0.92) !important;
            color: var(--text) !important;
            border: none !important;
            border-radius: 12px !important;
        }
        [data-testid="stExpander"] summary *,
        [data-testid="stExpander"] summary p,
        [data-testid="stExpander"] summary span,
        [data-testid="stExpander"] summary svg {
            color: var(--text) !important;
            fill: var(--text) !important;
        }
        [data-testid="stExpander"] [data-testid="stExpanderDetails"],
        [data-testid="stExpander"] .streamlit-expanderContent {
            background: rgba(9, 14, 27, 0.50) !important;
            border-top: 1px solid rgba(148, 163, 184, 0.14) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def normalize_agency_name(value: str | None) -> str:
    if not value:
        return DEFAULT_AGENCY_NAME
    cleaned = " ".join(str(value).strip().split())
    alias = AGENCY_ALIASES.get(cleaned.lower())
    if alias:
        return alias
    return cleaned


def comparable_org_name(value: str | None) -> str:
    text = normalize_agency_name(value).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\b(the|agency|department|office|bureau)\b", " ", text)
    return " ".join(text.split())


def meaningful_bureau_options(agency_name: str, bureau_options: list[str]) -> list[str]:
    agency_key = comparable_org_name(agency_name)
    meaningful = []
    seen = set()
    for option in bureau_options:
        if not option or option == ALL_BUREAUS:
            continue
        option_key = comparable_org_name(option)
        if not option_key or option_key == agency_key:
            continue
        if option_key in seen:
            continue
        seen.add(option_key)
        meaningful.append(option)
    return [ALL_BUREAUS] + meaningful


def canonical_bureau_name(bureau_name: str | None) -> str:
    bureau_name = clean_office_value(bureau_name)
    if not bureau_name or bureau_name.lower() == NOT_APPLICABLE_BUREAU.lower():
        return ALL_BUREAUS
    return bureau_name


def bureau_is_ui_only_not_applicable(bureau_name: str | None) -> bool:
    return clean_office_value(bureau_name).lower() == NOT_APPLICABLE_BUREAU.lower()


def resolve_bureau_filter_name(bureau_name: str | None) -> str | None:
    bureau_name = canonical_bureau_name(bureau_name)
    if not bureau_name or bureau_name == ALL_BUREAUS:
        return None
    return bureau_name


def first_present(mapping: dict, keys: list[str]):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def encode_option(code: str, description: str = "") -> str:
    return f"{str(code).strip()}{OPTION_SEPARATOR}{str(description or '').strip()}"


def decode_option(option: str | None) -> tuple[str, str]:
    if not option or OPTION_SEPARATOR not in str(option):
        return str(option or "").strip(), ""
    code, description = str(option).split(OPTION_SEPARATOR, 1)
    return code.strip(), description.strip()


def format_code_description_option(option: str) -> str:
    code, description = decode_option(option)
    if description:
        return f"{code} — {description}"
    return code


def default_market_filters() -> dict:
    return {
        "naics_code": ALL_NAICS_CODES,
        "contract_type": ALL_CONTRACT_TYPES,
        "psc_code": ALL_PRODUCT_SERVICE_CODES,
        "set_aside_type": ALL_SET_ASIDE_TYPES,
        "funding_office": ALL_FUNDING_OFFICES,
        "pop_state": ALL_POP_LOCATIONS,
    }


def is_us_pop_state_code(code: str | None) -> bool:
    return str(code or "").strip().upper() in STATE_OPTIONS


def build_place_of_performance_location_filter(location_code: str) -> dict:
    normalized = str(location_code or "").strip().upper()
    if is_us_pop_state_code(normalized):
        return {"country": "USA", "state": normalized}
    return {"country": normalized}


def place_of_performance_filter_matches(location_code: str, locations: list[dict]) -> bool:
    if not location_code or location_code in (ALL_POP_LOCATIONS, LEGACY_ALL_POP_STATES):
        return True
    expected = build_place_of_performance_location_filter(location_code)
    for location in locations:
        if not isinstance(location, dict):
            continue
        if location.get("country") != expected.get("country"):
            continue
        expected_state = expected.get("state")
        if expected_state:
            if location.get("state") == expected_state:
                return True
        else:
            return True
    return False


def load_resolved_signals_json(uploaded_file) -> dict:
    if uploaded_file is None:
        raise ValueError("No resolved_signals.json file was provided.")
    if isinstance(uploaded_file, dict):
        payload = uploaded_file
    elif isinstance(uploaded_file, (str, Path)):
        path = Path(uploaded_file)
        if not path.exists():
            raise ValueError(f"resolved_signals.json not found: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
    elif hasattr(uploaded_file, "read"):
        raw = uploaded_file.read()
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        payload = json.loads(raw)
    else:
        raise ValueError("Unsupported resolved_signals.json input type.")

    if not isinstance(payload, dict):
        raise ValueError("resolved_signals.json must be a JSON object.")
    version = str(payload.get("version") or "").strip()
    if version and version != RESOLVED_SIGNALS_VERSION:
        raise ValueError(f"Unsupported resolved_signals version: {version}")
    signals = payload.get("signals")
    if not isinstance(signals, list):
        raise ValueError("resolved_signals.json must contain a signals[] array.")
    for index, signal in enumerate(signals):
        if not isinstance(signal, dict):
            raise ValueError(f"signals[{index}] must be an object.")
        if not str(signal.get("id") or "").strip():
            raise ValueError(f"signals[{index}] is missing id.")
    return payload


def get_signal(resolved_signals: dict, signal_id: str) -> dict | None:
    if not isinstance(resolved_signals, dict):
        return None
    for signal in resolved_signals.get("signals") or []:
        if isinstance(signal, dict) and signal.get("id") == signal_id:
            return signal
    return None


def display_signal_scalar(value: object | None, signal_id: str | None = None) -> str | None:
    if value is None:
        return None
    if isinstance(value, dict):
        from extraction.package_llm.canonicalize import canonical_scalar_for_signal

        scalar, _ = canonical_scalar_for_signal(signal_id or "", value)
        if scalar is not None and not isinstance(scalar, dict):
            return str(scalar).strip() or None
        return None
    if isinstance(value, (list, tuple, set)):
        parts = [str(item).strip() for item in value if item not in (None, "")]
        return ", ".join(parts) if parts else None
    text = str(value).strip()
    return text or None


def get_signal_value(resolved_signals: dict, signal_id: str) -> tuple[object | None, str | None]:
    signal = get_signal(resolved_signals, signal_id)
    if not signal:
        return None, None
    value = signal.get("canonical_value")
    confidence = str(signal.get("canonical_confidence") or "").strip().lower() or None
    if value is None:
        package_llm = signal.get("package_llm") if isinstance(signal.get("package_llm"), dict) else {}
        model_value = package_llm.get("modelValue")
        if model_value is not None:
            value = display_signal_scalar(model_value, signal_id)
        if value is None:
            return None, confidence
    scalar = display_signal_scalar(value, signal_id)
    if scalar is None:
        return None, confidence
    return scalar, confidence


def signal_evidence_details(signal: dict | None) -> dict:
    if not isinstance(signal, dict):
        return {"source": "", "locator": "", "snippet": ""}
    evidence = signal.get("evidence") or {}
    legacy_items = evidence.get("legacy") or []
    for item in legacy_items:
        if isinstance(item, dict):
            return {
                "source": clean_office_value(item.get("sourceId")),
                "locator": clean_office_value(item.get("locator")),
                "snippet": clean_office_value(item.get("snippet")),
            }
    evidence_v1 = evidence.get("evidence_v1")
    if isinstance(evidence_v1, dict):
        return {
            "source": clean_office_value(evidence_v1.get("source")),
            "locator": "",
            "snippet": clean_office_value(evidence_v1.get("excerpt")),
        }
    return {"source": "", "locator": "", "snippet": ""}


def signal_evidence_snippet(signal: dict | None, max_length: int = 220) -> str:
    if not isinstance(signal, dict):
        return ""
    evidence = signal.get("evidence") or {}
    legacy_items = evidence.get("legacy") or []
    for item in legacy_items:
        if isinstance(item, dict):
            snippet = clean_office_value(item.get("snippet"))
            if snippet:
                return snippet[:max_length]
    evidence_v1 = evidence.get("evidence_v1")
    if isinstance(evidence_v1, dict):
        excerpt = clean_office_value(evidence_v1.get("excerpt"))
        if excerpt:
            return excerpt[:max_length]
    return ""


def first_signal_value(
    resolved_signals: dict,
    signal_ids: list[str],
) -> tuple[object | None, str | None, str | None, dict | None]:
    for signal_id in signal_ids:
        signal = get_signal(resolved_signals, signal_id)
        if not signal:
            continue
        value, confidence = get_signal_value(resolved_signals, signal_id)
        if value is not None:
            return value, confidence, signal_id, signal
    return None, None, None, None


def signal_review_model_value(signal: dict | None) -> object | None:
    if not isinstance(signal, dict):
        return None
    package_llm = signal.get("package_llm")
    if isinstance(package_llm, dict):
        return package_llm.get("modelValue")
    return None


def extract_category_code(value: object | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    match = re.search(r"\b([A-Z0-9]{1,6})\b", text.upper())
    return match.group(1) if match else text.upper()


def parse_pop_state_from_text(value: object | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None
    for code, name in STATE_OPTIONS.items():
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
            return code, name
    state_match = re.search(r",\s*([A-Z]{2})\b", text)
    if state_match and state_match.group(1).upper() in STATE_OPTIONS:
        code = state_match.group(1).upper()
        return code, STATE_OPTIONS[code]
    for code in STATE_OPTIONS:
        if re.search(rf"\b{code}\b", text.upper()):
            return code, STATE_OPTIONS[code]
    return None, None


def lookup_solicitation_organization_hierarchy(extracted_org: str | None) -> tuple[str | None, str | None]:
    cleaned = " ".join(str(extracted_org or "").strip().split())
    if not cleaned:
        return None, None
    return SOLICITATION_ORGANIZATION_HIERARCHY.get(cleaned.lower(), (None, None))


def exact_subagency_hierarchy_match(
    extracted_org: str | None,
    agency_records: list[dict],
    fiscal_year: int,
) -> dict:
    cleaned = canonical_bureau_name(extracted_org)
    attempts: list[dict] = []
    if not cleaned or cleaned == ALL_BUREAUS:
        return {"agency": None, "subagency": None, "method": None, "attempts": attempts}
    target_key = comparable_org_name(cleaned)
    for agency_record in agency_records:
        agency_name = agency_record.get("agency_name") or ""
        bureau_options = solicitation_bureau_options_for_agency(agency_records, agency_name, fiscal_year)
        exact_matches = [option for option in bureau_options if option != ALL_BUREAUS and option.lower() == cleaned.lower()]
        attempts.append(
            {
                "strategy": "exact_subagency_parent_scan",
                "candidate": cleaned,
                "agency": agency_name,
                "matches": exact_matches,
            }
        )
        if len(exact_matches) == 1:
            return {
                "agency": agency_name,
                "subagency": exact_matches[0],
                "method": "exact_subagency_parent_scan",
                "attempts": attempts,
            }
        normalized_matches = [
            option
            for option in bureau_options
            if option != ALL_BUREAUS and comparable_org_name(option) == target_key
        ]
        attempts.append(
            {
                "strategy": "exact_normalized_subagency_parent_scan",
                "candidate": cleaned,
                "agency": agency_name,
                "matches": normalized_matches,
            }
        )
        if len(normalized_matches) == 1:
            return {
                "agency": agency_name,
                "subagency": normalized_matches[0],
                "method": "exact_normalized_subagency_parent_scan",
                "attempts": attempts,
            }
    return {"agency": None, "subagency": None, "method": None, "attempts": attempts}


def validate_solicitation_hierarchy(
    agency: str | None,
    subagency: str | None,
    agency_records: list[dict],
    fiscal_year: int,
) -> dict:
    if not agency or not subagency or subagency == ALL_BUREAUS:
        return {"valid": True, "reason": ""}
    bureau_options = solicitation_bureau_options_for_agency(agency_records, agency, fiscal_year)
    valid = subagency in bureau_options
    return {
        "valid": valid,
        "reason": "" if valid else "selected_subagency.parent_agency != selected_agency",
    }


def solicitation_bureau_options_for_agency(
    agency_records: list[dict],
    agency_name: str,
    fiscal_year: int,
) -> list[str]:
    agency_record = agency_record_by_name(agency_records, agency_name)
    raw_bureau_options = get_bureau_options(agency_record["toptier_code"], int(fiscal_year))
    return meaningful_bureau_options(agency_record["agency_name"], raw_bureau_options)


def match_top_level_agency_to_options(extracted_agency: str, agency_options: list[str]) -> dict:
    cleaned = " ".join(str(extracted_agency or "").strip().split())
    attempts: list[dict] = []
    if not cleaned:
        return {
            "filter_option": None,
            "mapping_status": "Unmapped",
            "preselect": False,
            "attempts": attempts,
        }

    exact_matches = [agency for agency in agency_options if agency.lower() == cleaned.lower()]
    attempts.append({"strategy": "exact_normalized", "candidate": cleaned, "matches": exact_matches})
    if len(exact_matches) == 1:
        return {
            "filter_option": exact_matches[0],
            "mapping_status": "Exact match",
            "preselect": True,
            "attempts": attempts,
        }
    if len(exact_matches) > 1:
        return {
            "filter_option": None,
            "mapping_status": "Unmapped",
            "preselect": False,
            "attempts": attempts,
        }

    alias_target = AGENCY_ALIASES.get(cleaned.lower())
    if alias_target:
        alias_matches = [agency for agency in agency_options if agency.lower() == alias_target.lower()]
        attempts.append({"strategy": "known_alias", "candidate": alias_target, "matches": alias_matches})
        if len(alias_matches) == 1:
            return {
                "filter_option": alias_matches[0],
                "mapping_status": "Suggested match",
                "preselect": False,
                "attempts": attempts,
            }

    attempts.append(
        {
            "strategy": "no_fuzzy_hierarchy_match",
            "candidate": cleaned,
            "matches": [],
            "rejection_reason": "Federal hierarchy matching requires exact name or explicit alias.",
        }
    )
    return {
        "filter_option": None,
        "mapping_status": "Unmapped",
        "preselect": False,
        "attempts": attempts,
    }


def match_subagency_to_options(target_subagency: str | None, bureau_options: list[str]) -> dict:
    attempts: list[dict] = []
    target = canonical_bureau_name(target_subagency)
    if not target or target == ALL_BUREAUS:
        if ALL_BUREAUS in bureau_options:
            return {
                "filter_option": ALL_BUREAUS,
                "mapping_status": "Exact match",
                "preselect": True,
                "attempts": [{"strategy": "all_bureaus", "candidate": ALL_BUREAUS, "matches": [ALL_BUREAUS]}],
            }
        return {
            "filter_option": None,
            "mapping_status": "Unmapped",
            "preselect": False,
            "attempts": attempts,
        }

    exact_matches = [option for option in bureau_options if option.lower() == target.lower()]
    attempts.append({"strategy": "exact_subagency", "candidate": target, "matches": exact_matches})
    if len(exact_matches) == 1:
        return {
            "filter_option": exact_matches[0],
            "mapping_status": "Exact match",
            "preselect": True,
            "attempts": attempts,
        }

    needle = comparable_org_name(target)
    normalized_matches = [
        option
        for option in bureau_options
        if option != ALL_BUREAUS
        and comparable_org_name(option) == needle
    ]
    attempts.append({"strategy": "exact_normalized_subagency", "candidate": target, "matches": normalized_matches})
    if len(normalized_matches) == 1:
        return {
            "filter_option": normalized_matches[0],
            "mapping_status": "Exact match",
            "preselect": True,
            "attempts": attempts,
        }
    if len(normalized_matches) > 1:
        return {
            "filter_option": None,
            "mapping_status": "Unmapped",
            "preselect": False,
            "attempts": attempts,
        }
    attempts.append(
        {
            "strategy": "no_fuzzy_subagency_match",
            "candidate": target,
            "matches": [],
            "rejection_reason": "Subagency matching requires exact scoped name or explicit hierarchy alias.",
        }
    )
    return {
        "filter_option": None,
        "mapping_status": "Unmapped",
        "preselect": False,
        "attempts": attempts,
        "unmapped_extracted_display": target,
    }


def map_contract_type_text(value: object | None) -> tuple[str | None, str | None]:
    matches = map_contract_type_texts(value)
    if matches:
        return matches[0]
    return None, None


def map_contract_type_texts(value: object | None) -> list[tuple[str, str]]:
    if value is None:
        return []
    text = re.sub(r"[^\w\s&/+]", " ", str(value).lower())
    text = " ".join(text.split())
    if not text:
        return []
    matches: list[tuple[str, str]] = []
    for alias, code in SOLICITATION_CONTRACT_TYPE_ALIASES.items():
        if alias in text:
            pair = (code, CONTRACT_TYPE_OPTIONS.get(code, code))
            if pair not in matches:
                matches.append(pair)
    for code, label in CONTRACT_TYPE_OPTIONS.items():
        if label.lower() in text or text in label.lower():
            pair = (code, label)
            if pair not in matches:
                matches.append(pair)
    return matches


def contract_type_hybrid_match(value: object | None, contract_type_options: list[str]) -> dict | None:
    matches = map_contract_type_texts(value)
    if len(matches) < 2:
        return None
    valid_options = []
    for code, label in matches:
        option = encode_option(code, label or CONTRACT_TYPE_OPTIONS.get(code, code))
        if option in contract_type_options:
            valid_options.append(option)
    display = "Hybrid T&M / FFP" if {code for code, _label in matches} >= {"Y", "J"} else "Hybrid contract type"
    return {
        "filter_option": None,
        "mapping_status": "Suggested match",
        "preselect": False,
        "attempts": [
            {
                "strategy": "hybrid_contract_type_alias",
                "candidate": value,
                "contract_types": [label for _code, label in matches],
                "valid_options": valid_options,
                "requires_analyst_market_treatment": True,
            }
        ],
        "unmapped_extracted_display": f"{display} - select the market treatment",
        "contract_types": [label for _code, label in matches],
        "contract_type_options": valid_options,
        "is_hybrid": True,
    }


def map_set_aside_text(value: object | None) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    text = str(value).lower()
    if "8(a)" in text or "8a" in text:
        return "8A", SET_ASIDE_TYPE_OPTIONS["8A"]
    if "sdvosb" in text or "service-disabled veteran" in text:
        return "SDVOSBC", SET_ASIDE_TYPE_OPTIONS["SDVOSBC"]
    if "wosb" in text or "women-owned" in text:
        return "WOSB", SET_ASIDE_TYPE_OPTIONS["WOSB"]
    if "hubzone" in text:
        return "HZC", SET_ASIDE_TYPE_OPTIONS["HZC"]
    if "unrestricted" in text or "full and open" in text or "no set aside" in text or "not set aside" in text:
        return "NONE", SET_ASIDE_TYPE_OPTIONS["NONE"]
    if "small business set-aside" in text or "small business set aside" in text:
        return "SBA", SET_ASIDE_TYPE_OPTIONS["SBA"]
    return None, None


def build_solicitation_scope_preview(resolved_signals: dict) -> dict:
    field_specs = [
        ("Solicitation Number", ["rfp_solicitation_number_v1", "rfp_solicitation_id_v1"]),
        ("Title", ["rfp_title_v1"]),
        ("Issuing Agency", ["rfp_issuing_agency_v1"]),
        ("Issuing Office", ["rfp_issuing_office_v1"]),
        ("Office AAC", ["rfp_office_aac_v1"]),
        ("Funding Office", ["rfp_funding_office_v1", "rfp_funding_office_name_v1"]),
        ("Primary NAICS", ["rfp_primary_naics_v1"]),
        ("Primary PSC", ["rfp_primary_psc_v1"]),
        ("Contract Type", ["rfp_contract_type_v1"]),
        ("Competition / Set-Aside", ["rfp_set_aside_v1", "rfp_competition_type_v1"]),
        ("Place of Performance", ["rfp_place_of_performance_v1"]),
        ("Period of Performance", ["rfp_period_of_performance_v1"]),
        ("POP Start", ["rfp_pop_start_v1"]),
        ("POP End", ["rfp_pop_end_v1"]),
        ("Incumbent Data", ["rfp_incumbent_data_v1"]),
        ("Description", ["rfp_description_v1"]),
        ("Primary POC", ["rfp_primary_poc_v1"]),
        ("Contracting Officer (KO)", ["rfp_ko_name_v1"]),
        ("Contract Specialist", ["rfp_contract_specialist_name_v1"]),
    ]
    preview: dict[str, dict] = {}
    for field_name, signal_ids in field_specs:
        if field_name == "Competition / Set-Aside":
            competition_value, competition_confidence, competition_id, competition_signal = first_signal_value(
                resolved_signals,
                ["rfp_competition_type_v1"],
            )
            set_aside_value, set_aside_confidence, set_aside_id, set_aside_signal = first_signal_value(
                resolved_signals,
                ["rfp_set_aside_v1"],
            )
            if set_aside_value is None:
                set_aside_signal = get_signal(resolved_signals, "rfp_set_aside_v1")
                set_aside_value = signal_review_model_value(set_aside_signal)
                if set_aside_value is not None:
                    set_aside_confidence = (set_aside_signal or {}).get("canonical_confidence")
                    set_aside_id = "rfp_set_aside_v1"
            if set_aside_value is None and competition_value is None:
                continue
            combined_parts = []
            if competition_value is not None:
                combined_parts.append(str(competition_value))
            if set_aside_value is not None:
                combined_parts.append(str(set_aside_value))
            preview[field_name] = {
                "value": " — ".join(combined_parts),
                "confidence": set_aside_confidence or competition_confidence,
                "signal_id": set_aside_id or competition_id,
                "evidence_snippet": signal_evidence_snippet(set_aside_signal or competition_signal),
                "evidence_source": signal_evidence_details(set_aside_signal or competition_signal).get("source", ""),
                "evidence_locator": signal_evidence_details(set_aside_signal or competition_signal).get("locator", ""),
                "validation_status": (set_aside_signal or competition_signal or {}).get("resolution_status", ""),
                "set_aside_value": set_aside_value,
                "competition_value": competition_value,
            }
            continue

        value, confidence, signal_id, signal = first_signal_value(resolved_signals, signal_ids)
        if value is None:
            continue
        evidence_details = signal_evidence_details(signal)
        preview[field_name] = {
            "value": value,
            "confidence": confidence,
            "signal_id": signal_id,
            "evidence_snippet": signal_evidence_snippet(signal),
            "evidence_source": evidence_details.get("source", ""),
            "evidence_locator": evidence_details.get("locator", ""),
            "validation_status": (signal or {}).get("resolution_status", ""),
        }
    return preview


def _solicitation_mapping_row(
    field: str,
    extracted_value: object | None,
    *,
    confidence: str | None = None,
    evidence_snippet: str = "",
    evidence_source: str = "",
    evidence_locator: str = "",
    validation_status: str = "",
    filter_key: str | None = None,
    match: dict | None = None,
    context_only: bool = False,
    mapped_filter_display: str | None = None,
    deterministic_mapping: bool = False,
) -> dict:
    match = match or {}
    row = {
        "field": field,
        "extracted_value": extracted_value,
        "confidence": confidence,
        "evidence_snippet": evidence_snippet,
        "evidence_source": evidence_source,
        "evidence_locator": evidence_locator,
        "validation_status": validation_status,
        "mapped_filter": match.get("filter_option"),
        "mapped_filter_display": mapped_filter_display
        if mapped_filter_display is not None
        else (match.get("filter_option") or match.get("unmapped_extracted_display") or ""),
        "filter_key": filter_key,
        "mapping_status": "Context only" if context_only else match.get("mapping_status", "Unmapped"),
        "preselect": False if context_only else match.get("preselect", False),
        "filter_option": None if context_only else match.get("filter_option"),
        "unmapped_extracted_display": match.get("unmapped_extracted_display"),
        "deterministic_mapping": deterministic_mapping,
        "contract_types": match.get("contract_types"),
        "contract_type_options": match.get("contract_type_options"),
        "is_hybrid": bool(match.get("is_hybrid")),
        "rejection_reason": match.get("rejection_reason"),
        "contract_type_filter_source": match.get("contract_type_filter_source"),
    }
    if context_only or deterministic_mapping:
        return row
    return apply_confidence_to_mapping(row)


def map_solicitation_organization(
    scope_preview: dict,
    agency_records: list[dict],
    fiscal_year: int,
) -> dict:
    agency_options = agency_names_from_records(agency_records)
    issuing_agency_data = scope_preview.get("Issuing Agency", {})
    extracted_org = issuing_agency_data.get("value")
    signal_confidence = issuing_agency_data.get("confidence")
    evidence_snippet = issuing_agency_data.get("evidence_snippet", "")
    evidence_source = issuing_agency_data.get("evidence_source", "")
    evidence_locator = issuing_agency_data.get("evidence_locator", "")
    validation_status = issuing_agency_data.get("validation_status", "")
    mapping_attempts: dict[str, list] = {}

    rows: list[dict] = []
    rows.append(
        _solicitation_mapping_row(
        "Extracted Organization",
        extracted_org,
        confidence=signal_confidence,
        evidence_snippet=evidence_snippet,
        evidence_source=evidence_source,
        evidence_locator=evidence_locator,
        validation_status=validation_status,
        context_only=True,
    )
    )

    hierarchy_agency, hierarchy_subagency = lookup_solicitation_organization_hierarchy(
        str(extracted_org) if extracted_org is not None else None
    )
    hierarchy_method = "deterministic_alias" if hierarchy_agency else None
    subagency_hierarchy = (
        exact_subagency_hierarchy_match(extracted_org, agency_records, fiscal_year)
        if not hierarchy_agency
        else {"agency": None, "subagency": None, "method": None, "attempts": []}
    )
    if not hierarchy_agency and subagency_hierarchy.get("agency"):
        hierarchy_agency = subagency_hierarchy["agency"]
        hierarchy_subagency = subagency_hierarchy["subagency"]
        hierarchy_method = subagency_hierarchy["method"]
    if hierarchy_agency:
        agency_match = match_top_level_agency_to_options(hierarchy_agency, agency_options)
        if agency_match.get("filter_option"):
            agency_match["mapping_status"] = "Exact match"
            agency_match["preselect"] = True
        else:
            agency_match["mapping_status"] = "Suggested match"
            agency_match["preselect"] = False
            agency_match["unmapped_extracted_display"] = hierarchy_agency
        mapping_attempts["Agency"] = [
            {
                "strategy": "deterministic_hierarchy",
                "candidate": hierarchy_agency,
                "extracted_organization": extracted_org,
                "matched_hierarchy_level": "subagency" if hierarchy_subagency and hierarchy_subagency != ALL_BUREAUS else "agency",
                "matched_agency": agency_match.get("filter_option"),
                "matched_subagency": hierarchy_subagency,
                "match_method": hierarchy_method,
                "matches": [agency_match.get("filter_option")] if agency_match.get("filter_option") else [],
            },
            *subagency_hierarchy.get("attempts", []),
            *agency_match.get("attempts", []),
        ]
        bureau_options = solicitation_bureau_options_for_agency(
            agency_records,
            agency_match.get("filter_option") or hierarchy_agency,
            fiscal_year,
        )
        subagency_match = match_subagency_to_options(hierarchy_subagency, bureau_options)
        mapping_attempts["Subagency / Bureau"] = subagency_match.get("attempts", [])
        if subagency_match.get("mapping_status") == "Unmapped":
            subagency_match["unmapped_extracted_display"] = hierarchy_subagency
        if hierarchy_subagency and hierarchy_subagency != ALL_BUREAUS:
            if not subagency_match.get("filter_option"):
                if hierarchy_method == "deterministic_alias":
                    subagency_match["filter_option"] = hierarchy_subagency
                else:
                    subagency_match["unmapped_extracted_display"] = hierarchy_subagency
            subagency_match["mapping_status"] = (
                "Exact match"
                if subagency_match.get("filter_option")
                else "Suggested match"
            )
            subagency_match["preselect"] = bool(subagency_match.get("filter_option"))
        hierarchy_validation = validate_solicitation_hierarchy(
            agency_match.get("filter_option"),
            subagency_match.get("filter_option"),
            agency_records,
            fiscal_year,
        )
        if not hierarchy_validation["valid"] and hierarchy_method != "deterministic_alias":
            subagency_match["mapping_status"] = "Unmapped"
            subagency_match["preselect"] = False
            subagency_match["filter_option"] = None
            subagency_match["rejection_reason"] = hierarchy_validation["reason"]
            mapping_attempts["Subagency / Bureau"].append(
                {
                    "strategy": "hierarchy_validation",
                    "candidate": hierarchy_subagency,
                    "matched_agency": agency_match.get("filter_option"),
                    "matched_subagency": hierarchy_subagency,
                    "rejection_reason": hierarchy_validation["reason"],
                }
            )
    else:
        agency_match = match_top_level_agency_to_options(
            str(extracted_org) if extracted_org is not None else "",
            agency_options,
        )
        if agency_match.get("filter_option") and comparable_org_name(agency_match.get("filter_option")) != comparable_org_name(extracted_org):
            agency_match["filter_option"] = None
            agency_match["mapping_status"] = "Unmapped"
            agency_match["preselect"] = False
            agency_match["rejection_reason"] = "extracted organization did not exactly match a top-level agency"
        mapping_attempts["Agency"] = [
            *subagency_hierarchy.get("attempts", []),
            *agency_match.get("attempts", []),
        ]
        bureau_options = solicitation_bureau_options_for_agency(
            agency_records,
            agency_match.get("filter_option") or str(extracted_org or ""),
            fiscal_year,
        )
        subagency_match = {
            "filter_option": None,
            "mapping_status": "Unmapped",
            "preselect": False,
            "attempts": [
                {
                    "strategy": "no_silent_all_bureaus_fallback",
                    "candidate": extracted_org,
                    "matches": [],
                    "rejection_reason": "specific organization could not be placed in a valid hierarchy",
                }
            ],
        }
        mapping_attempts["Subagency / Bureau"] = subagency_match.get("attempts", [])

    agency_row = _solicitation_mapping_row(
        "Agency",
        hierarchy_agency or extracted_org,
        confidence=signal_confidence,
        evidence_snippet=evidence_snippet,
        evidence_source=evidence_source,
        evidence_locator=evidence_locator,
        validation_status=validation_status,
        filter_key="agency",
        match=agency_match,
        deterministic_mapping=bool(hierarchy_agency and agency_match.get("filter_option")),
    )
    agency_row["hierarchy_match_level"] = "top_level_agency" if agency_match.get("filter_option") else None
    subagency_display = (
        subagency_match["filter_option"]
        if subagency_match.get("filter_option")
        else subagency_match.get("unmapped_extracted_display")
        or hierarchy_subagency
        or ALL_BUREAUS
    )
    subagency_row = _solicitation_mapping_row(
        "Subagency / Bureau",
        hierarchy_subagency or extracted_org,
        confidence=signal_confidence,
        evidence_snippet=evidence_snippet,
        evidence_source=evidence_source,
        evidence_locator=evidence_locator,
        validation_status=validation_status,
        filter_key="bureau",
        match=subagency_match,
        mapped_filter_display=subagency_display,
        deterministic_mapping=bool(
            hierarchy_agency and hierarchy_subagency and subagency_match.get("filter_option")
        ),
    )
    subagency_row["hierarchy_match_level"] = (
        "subagency" if subagency_match.get("filter_option") and subagency_match.get("filter_option") != ALL_BUREAUS else None
    )
    rows.extend([agency_row, subagency_row])

    return {
        "rows": rows,
        "mapped_agency": agency_match.get("filter_option"),
        "mapped_bureau": subagency_match.get("filter_option"),
        "bureau_options": bureau_options,
        "mapping_attempts": mapping_attempts,
        "hierarchy_rule_applied": bool(hierarchy_agency),
    }


def map_solicitation_market_filters(
    scope_preview: dict,
    available_filter_options: dict,
) -> dict:
    rows: list[dict] = []
    mapping_attempts: dict[str, list] = {}
    unavailable_fields = set(available_filter_options.get("mapping_unavailable_fields") or [])

    def mark_unavailable(row: dict, field_name: str, extracted_value: object | None) -> dict:
        if field_name not in unavailable_fields:
            return row
        row["mapping_status"] = SOLICITATION_MAPPING_UNAVAILABLE_STATUS
        row["preselect"] = False
        row["filter_option"] = None
        row["mapped_filter"] = None
        row["mapped_filter_display"] = str(extracted_value or row.get("unmapped_extracted_display") or "")
        row["unmapped_extracted_display"] = str(extracted_value or row.get("unmapped_extracted_display") or "")
        row["mapping_unavailable"] = True
        return row

    office_aac = scope_preview.get("Office AAC", {}).get("value")
    issuing_office = scope_preview.get("Issuing Office", {}).get("value")
    if office_aac or issuing_office:
        office_match = match_contracting_office_to_options(
            str(office_aac) if office_aac is not None else None,
            str(issuing_office) if issuing_office is not None else None,
            available_filter_options["contracting_offices"],
        )
        mapping_attempts["Contracting Office"] = office_match["attempts"]
        office_row = _solicitation_mapping_row(
            "Contracting Office",
            f"{office_aac or ''} / {issuing_office or ''}".strip(" /"),
            confidence=scope_preview.get("Office AAC", {}).get("confidence")
            or scope_preview.get("Issuing Office", {}).get("confidence"),
            evidence_snippet=scope_preview.get("Office AAC", {}).get("evidence_snippet", "")
            or scope_preview.get("Issuing Office", {}).get("evidence_snippet", ""),
            evidence_source=scope_preview.get("Office AAC", {}).get("evidence_source", "")
            or scope_preview.get("Issuing Office", {}).get("evidence_source", ""),
            evidence_locator=scope_preview.get("Office AAC", {}).get("evidence_locator", "")
            or scope_preview.get("Issuing Office", {}).get("evidence_locator", ""),
            validation_status=scope_preview.get("Office AAC", {}).get("validation_status", "")
            or scope_preview.get("Issuing Office", {}).get("validation_status", ""),
            filter_key="contracting_office",
            match=office_match,
            mapped_filter_display=format_contracting_office_option(office_match["filter_option"])
            if office_match.get("filter_option")
            else f"{office_aac or ''} / {issuing_office or ''}".strip(" /"),
        )
        office_row = mark_unavailable(office_row, "Contracting Office", f"{office_aac or ''} / {issuing_office or ''}".strip(" /"))
        rows.append(office_row)

    funding_office_data = scope_preview.get("Funding Office", {})
    funding_extracted = funding_office_data.get("value")
    if funding_extracted is not None:
        funding_code = extract_category_code(funding_extracted)
        funding_name = clean_office_value(funding_extracted)
        if funding_code and funding_code == funding_name.upper():
            funding_name = ""
        funding_match = match_funding_office_to_options(
            funding_code,
            funding_name or str(funding_extracted),
            available_filter_options.get("funding_offices", [ALL_FUNDING_OFFICES]),
        )
        mapping_attempts["Funding Office"] = funding_match.get("attempts", [])
        funding_display = (
            format_funding_office_option(funding_match["filter_option"])
            if funding_match.get("filter_option")
            else str(funding_extracted)
        )
        funding_row = _solicitation_mapping_row(
                "Funding Office",
                funding_extracted,
                confidence=funding_office_data.get("confidence"),
                evidence_snippet=funding_office_data.get("evidence_snippet", ""),
                evidence_source=funding_office_data.get("evidence_source", ""),
                evidence_locator=funding_office_data.get("evidence_locator", ""),
                validation_status=funding_office_data.get("validation_status", ""),
                filter_key="funding_office",
                match=funding_match,
                mapped_filter_display=funding_display,
            )
        rows.append(funding_row)

    market_field_specs = [
        ("NAICS", "Primary NAICS", "naics_code", "naics", ALL_NAICS_CODES),
        ("PSC", "Primary PSC", "psc_code", "psc", ALL_PRODUCT_SERVICE_CODES),
        ("Contract Type", "Contract Type", "contract_type", "contract_types", ALL_CONTRACT_TYPES),
        ("Set-Aside", "Competition / Set-Aside", "set_aside_type", "set_asides", ALL_SET_ASIDE_TYPES),
        ("Place of Performance", "Place of Performance", "pop_state", "pop_locations", ALL_POP_LOCATIONS),
    ]
    for display_name, preview_key, filter_key, options_key, default_option in market_field_specs:
        field_data = scope_preview.get(preview_key, {})
        extracted_value = field_data.get("value")
        if extracted_value is None and preview_key != "Competition / Set-Aside":
            continue
        if preview_key == "Competition / Set-Aside":
            extracted_value = field_data.get("set_aside_value") or extracted_value
            if extracted_value is None:
                continue
            match = match_set_aside_to_options(extracted_value, available_filter_options[options_key])
        elif preview_key == "Primary NAICS":
            match = match_category_option_to_options(extracted_value, available_filter_options[options_key], default_option)
        elif preview_key == "Primary PSC":
            match = match_psc_to_options(extracted_value, available_filter_options[options_key], default_option)
        elif preview_key == "Contract Type":
            match = match_contract_type_to_options(extracted_value, available_filter_options[options_key])
        elif preview_key == "Place of Performance":
            match = match_pop_location_to_options(extracted_value, available_filter_options[options_key])
        else:
            continue

        mapping_attempts[display_name] = match.get("attempts", [])
        display = match.get("extracted_code") or str(extracted_value)
        mapped_display = (
            format_code_description_option(match["filter_option"])
            if match.get("filter_option")
            else display
        )
        if filter_key == "contract_type":
            mapped_display = match.get("unmapped_extracted_display") or "Select manually"
        if filter_key == "contracting_office":
            mapped_display = format_contracting_office_option(match["filter_option"]) if match.get("filter_option") else display
        if filter_key == "pop_state" and not match.get("filter_option"):
            mapped_display = str(extracted_value)

        market_row = _solicitation_mapping_row(
                display_name,
                extracted_value,
                confidence=field_data.get("confidence"),
                evidence_snippet=field_data.get("evidence_snippet", ""),
                evidence_source=field_data.get("evidence_source", ""),
                evidence_locator=field_data.get("evidence_locator", ""),
                validation_status=field_data.get("validation_status", ""),
                filter_key=filter_key,
                match=match,
                mapped_filter_display=mapped_display,
            )
        rows.append(mark_unavailable(market_row, display_name, extracted_value))

    context_field_specs = [
        ("Solicitation Number", "Solicitation Number"),
        ("Period of Performance", "Period of Performance"),
        ("POP Start", "POP Start"),
        ("POP End", "POP End"),
        ("Incumbent Data", "Incumbent Data"),
        ("Description", "Description"),
        ("Primary POC", "Primary POC"),
        ("Contracting Officer (KO)", "Contracting Officer (KO)"),
        ("Contract Specialist", "Contract Specialist"),
    ]
    context_rows = []
    for display_name, preview_key in context_field_specs:
        field_data = scope_preview.get(preview_key, {})
        if field_data.get("value") is None:
            continue
        context_rows.append(
            _solicitation_mapping_row(
                display_name,
                field_data.get("value"),
                confidence=field_data.get("confidence"),
                evidence_snippet=field_data.get("evidence_snippet", ""),
                context_only=True,
            )
        )

    return {
        "rows": rows,
        "context_rows": context_rows,
        "mapping_attempts": mapping_attempts,
    }


def match_contracting_office_to_options(
    office_aac: str | None,
    issuing_office: str | None,
    office_options: list[str],
) -> dict:
    attempts: list[dict] = []
    aac = clean_office_value(office_aac).upper()
    office_name = clean_office_value(issuing_office)

    def _option_parts(option: str) -> tuple[str, str]:
        code, name = decode_contracting_office(option)
        if code:
            return code.upper(), name
        if " — " in option:
            left, right = option.split(" — ", 1)
            return clean_office_value(left).upper(), clean_office_value(right)
        text = clean_office_value(option)
        token = text.split()[0] if text else ""
        if token and re.fullmatch(r"[A-Z0-9]{4,6}", token.upper()):
            return token.upper(), text
        return "", text

    def _names_equivalent(extracted: str, option_name: str) -> bool:
        left = comparable_org_name(extracted)
        right = comparable_org_name(option_name)
        if not left or not right:
            return False
        if left == right:
            return True
        return left in right

    if aac:
        code_matches = []
        for option in office_options:
            if option == ALL_CONTRACTING_OFFICES:
                continue
            code, _name = _option_parts(option)
            if code == aac:
                code_matches.append(option)
        attempts.append({"strategy": "office_aac_code", "candidate": aac, "matches": code_matches})
        if len(code_matches) == 1:
            return {
                "filter_option": code_matches[0],
                "mapping_status": "Exact match",
                "preselect": True,
                "attempts": attempts,
            }
        if len(code_matches) > 1 and office_name:
            refined_matches = []
            for option in code_matches:
                _code, name = _option_parts(option)
                if _names_equivalent(office_name, name):
                    refined_matches.append(option)
            attempts.append(
                {
                    "strategy": "office_aac_code_and_name",
                    "candidate": f"{aac} / {office_name}",
                    "matches": refined_matches,
                }
            )
            if len(refined_matches) == 1:
                return {
                    "filter_option": refined_matches[0],
                    "mapping_status": "Exact match",
                    "preselect": True,
                    "attempts": attempts,
                }
            if len(refined_matches) > 1:
                return {
                    "filter_option": refined_matches[0],
                    "mapping_status": "Suggested match",
                    "preselect": False,
                    "attempts": attempts,
                }
        if len(code_matches) > 1:
            return {
                "filter_option": None,
                "mapping_status": "Unmapped",
                "preselect": False,
                "attempts": attempts,
                "unmapped_extracted_display": f"{aac} / {office_name}".strip(" /"),
            }

    if office_name:
        exact_name_matches = []
        for option in office_options:
            if option == ALL_CONTRACTING_OFFICES:
                continue
            code, name = _option_parts(option)
            if aac and code != aac:
                continue
            if _names_equivalent(office_name, name):
                exact_name_matches.append(option)
        attempts.append(
            {
                "strategy": "issuing_office_name_exact",
                "candidate": office_name,
                "matches": exact_name_matches,
            }
        )
        if len(exact_name_matches) == 1:
            return {
                "filter_option": exact_name_matches[0],
                "mapping_status": "Exact match",
                "preselect": True,
                "attempts": attempts,
            }
        if len(exact_name_matches) > 1:
            return {
                "filter_option": None,
                "mapping_status": "Unmapped",
                "preselect": False,
                "attempts": attempts,
                "unmapped_extracted_display": f"{aac} / {office_name}".strip(" /"),
            }

    return {
        "filter_option": None,
        "mapping_status": "Unmapped",
        "preselect": False,
        "attempts": attempts,
        "unmapped_extracted_display": f"{aac} / {office_name}".strip(" /") if (aac or office_name) else "",
    }


def match_funding_office_to_options(
    office_code: str | None,
    office_name: str | None,
    office_options: list[str],
) -> dict:
    attempts: list[dict] = []
    code = clean_office_value(office_code).upper()
    name = clean_office_value(office_name)
    extracted_display = " / ".join(part for part in [code, name] if part) or name or code

    if code:
        code_matches = []
        for option in office_options:
            if option == ALL_FUNDING_OFFICES:
                continue
            option_code, _option_name = decode_option(option)
            if option_code and option_code.upper() == code:
                code_matches.append(option)
        attempts.append({"strategy": "funding_office_code", "candidate": code, "matches": code_matches})
        if len(code_matches) == 1:
            return {
                "filter_option": code_matches[0],
                "mapping_status": "Exact match",
                "preselect": True,
                "attempts": attempts,
            }

    if name:
        normalized_name = comparable_org_name(name)
        exact_name_matches = []
        for option in office_options:
            if option == ALL_FUNDING_OFFICES:
                continue
            _option_code, option_name = decode_option(option)
            if comparable_org_name(option_name) == normalized_name:
                exact_name_matches.append(option)
        attempts.append({"strategy": "funding_office_name_exact", "candidate": name, "matches": exact_name_matches})
        if len(exact_name_matches) == 1:
            return {
                "filter_option": exact_name_matches[0],
                "mapping_status": "Exact match",
                "preselect": True,
                "attempts": attempts,
            }

        fuzzy_matches = []
        for option in office_options:
            if option == ALL_FUNDING_OFFICES:
                continue
            _option_code, option_name = decode_option(option)
            option_key = comparable_org_name(option_name)
            if normalized_name and (normalized_name in option_key or option_key in normalized_name):
                fuzzy_matches.append(option)
        attempts.append({"strategy": "funding_office_name_fuzzy", "candidate": name, "matches": fuzzy_matches})
        if fuzzy_matches:
            return {
                "filter_option": fuzzy_matches[0],
                "mapping_status": "Suggested match",
                "preselect": False,
                "attempts": attempts,
            }

    return {
        "filter_option": None,
        "mapping_status": "Unmapped",
        "preselect": False,
        "attempts": attempts,
        "unmapped_extracted_display": extracted_display,
    }


def match_category_option_to_options(
    extracted_value: object | None,
    options: list[str],
    default_option: str,
) -> dict:
    code = extract_category_code(extracted_value)
    attempts = [{"strategy": "exact_code", "candidate": code, "matches": []}]
    if not code:
        return {
            "filter_option": None,
            "mapping_status": "Unmapped",
            "preselect": False,
            "attempts": attempts,
            "extracted_code": "",
        }
    matches = []
    for option in options:
        if option == default_option:
            continue
        option_code, _label = decode_option(option)
        if option_code.upper() == code.upper():
            matches.append(option)
    attempts[0]["matches"] = matches
    if len(matches) == 1:
        return {
            "filter_option": matches[0],
            "mapping_status": "Exact match",
            "preselect": True,
            "attempts": attempts,
            "extracted_code": code,
        }
    return {
        "filter_option": None,
        "mapping_status": "Unmapped",
        "preselect": False,
        "attempts": attempts,
        "extracted_code": code,
    }


def match_contract_type_to_options(extracted_value: object | None, contract_type_options: list[str]) -> dict:
    matches = map_contract_type_texts(extracted_value)
    valid_options = [
        encode_option(code, label or CONTRACT_TYPE_OPTIONS.get(code, code))
        for code, label in matches
        if encode_option(code, label or CONTRACT_TYPE_OPTIONS.get(code, code)) in contract_type_options
    ]
    attempts = [
        {
            "strategy": "contract_type_informational_only",
            "candidate": extracted_value,
            "contract_types": [label for _code, label in matches],
            "valid_options": valid_options,
            "contract_type_filter_source": None,
            "rejection_reason": "Contract Type is analyst-controlled and never AI-preselected.",
        }
    ]
    return {
        "filter_option": None,
        "mapping_status": "Manual selection required",
        "preselect": False,
        "attempts": attempts,
        "unmapped_extracted_display": "Select manually",
        "contract_types": [label for _code, label in matches],
        "contract_type_filter_source": None,
        "is_hybrid": len(matches) > 1,
    }


def match_set_aside_to_options(extracted_value: object | None, set_aside_options: list[str]) -> dict:
    code, label = map_set_aside_text(extracted_value)
    attempts = [{"strategy": "set_aside_alias", "candidate": extracted_value, "code": code, "label": label}]
    if not code:
        return {
            "filter_option": None,
            "mapping_status": "Unmapped",
            "preselect": False,
            "attempts": attempts,
        }
    target = encode_option(code, label or SET_ASIDE_TYPE_OPTIONS.get(code, code))
    if target in set_aside_options:
        return {
            "filter_option": target,
            "mapping_status": "Exact match",
            "preselect": True,
            "attempts": attempts,
        }
    return {
        "filter_option": None,
        "mapping_status": "Unmapped",
        "preselect": False,
        "attempts": attempts,
    }


def match_pop_location_to_options(extracted_value: object | None, pop_location_options: list[str]) -> dict:
    state_code, state_name = parse_pop_state_from_text(extracted_value)
    attempts = [{"strategy": "parse_us_state", "candidate": extracted_value, "state_code": state_code}]
    if not state_code:
        return {
            "filter_option": None,
            "mapping_status": "Context only",
            "preselect": False,
            "attempts": attempts,
        }
    target = encode_option(state_code, state_name or STATE_OPTIONS[state_code])
    if target in pop_location_options:
        return {
            "filter_option": target,
            "mapping_status": "Suggested match",
            "preselect": False,
            "attempts": attempts,
        }
    return {
        "filter_option": None,
        "mapping_status": "Context only",
        "preselect": False,
        "attempts": attempts,
    }


def apply_confidence_to_mapping(row: dict) -> dict:
    if row.get("deterministic_mapping"):
        return row
    confidence = str(row.get("confidence") or "").strip().lower()
    mapping_status = row.get("mapping_status")
    if mapping_status in ("Context only", "Unmapped"):
        row["preselect"] = False
        return row
    if confidence == "low":
        row["preselect"] = False
        if mapping_status == "Exact match":
            row["mapping_status"] = "Suggested match"
    elif confidence == "medium":
        row["preselect"] = False
        if mapping_status == "Exact match":
            row["mapping_status"] = "Suggested match"
    return row


def build_solicitation_available_filter_options(
    agency_records: list[dict],
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
) -> dict:
    mapping_started = time.monotonic()
    deadline = mapping_started + solicitation_mapping_timeout_sec()
    request_timeout = solicitation_option_request_timeout_sec()
    stage_events: list[dict] = []
    unavailable_fields: set[str] = set()

    def record(stage: str, status: str, started_at: float, function: str, exc: Exception | None = None) -> None:
        stage_events.append(_solicitation_mapping_stage(stage, started_at, status, function=function, exc=exc))

    def default_options() -> dict:
        return {
            "naics": [ALL_NAICS_CODES],
            "psc": [ALL_PRODUCT_SERVICE_CODES],
            "pop_locations": [ALL_POP_LOCATIONS]
            + [encode_option(code, name) for code, name in sorted(STATE_OPTIONS.items(), key=lambda item: item[1])],
            "contracting_offices": [ALL_CONTRACTING_OFFICES],
            "funding_offices": [ALL_FUNDING_OFFICES],
        }

    bureau_name = canonical_bureau_name(bureau_name)
    market_filters = default_market_filters()
    fallback = default_options()
    try:
        started_at = time.time()
        _check_solicitation_mapping_deadline(deadline, "NAICS option retrieval")
        naics_options = fetch_category_filter_options(
            agency_name,
            bureau_name,
            int(fiscal_year),
            "naics",
            ALL_NAICS_CODES,
            market_filters=market_filters,
            request_timeout_sec=request_timeout,
        )
        record("NAICS options retrieval", "completed", started_at, "fetch_category_filter_options")
    except Exception as exc:
        naics_options = fallback["naics"]
        unavailable_fields.add("NAICS")
        record("NAICS options retrieval", "timed_out" if isinstance(exc, TimeoutError) else "failed", started_at, "fetch_category_filter_options", exc)
    try:
        started_at = time.time()
        _check_solicitation_mapping_deadline(deadline, "PSC option retrieval")
        psc_options = fetch_category_filter_options(
            agency_name,
            bureau_name,
            int(fiscal_year),
            "psc",
            ALL_PRODUCT_SERVICE_CODES,
            market_filters=market_filters,
            request_timeout_sec=request_timeout,
        )
        record("PSC options retrieval", "completed", started_at, "fetch_category_filter_options")
    except Exception as exc:
        psc_options = fallback["psc"]
        unavailable_fields.add("PSC")
        record("PSC options retrieval", "timed_out" if isinstance(exc, TimeoutError) else "failed", started_at, "fetch_category_filter_options", exc)
    pop_scope_filters = normalize_market_filters({**market_filters, "pop_state": ALL_POP_LOCATIONS})
    us_state_options = [
        encode_option(code, name) for code, name in sorted(STATE_OPTIONS.items(), key=lambda item: item[1])
    ]
    try:
        started_at = time.time()
        _check_solicitation_mapping_deadline(deadline, "Place of Performance option retrieval")
        country_options = fetch_pop_country_filter_options(
            agency_name,
            bureau_name,
            int(fiscal_year),
            ALL_POP_LOCATIONS,
            market_filters=pop_scope_filters,
            request_timeout_sec=request_timeout,
        )
        record("Place of Performance options retrieval", "completed", started_at, "fetch_pop_country_filter_options")
    except Exception as exc:
        country_options = [ALL_POP_LOCATIONS]
        unavailable_fields.add("Place of Performance")
        record("Place of Performance options retrieval", "timed_out" if isinstance(exc, TimeoutError) else "failed", started_at, "fetch_pop_country_filter_options", exc)
    foreign_country_options = [option for option in country_options if option != ALL_POP_LOCATIONS]
    pop_location_options = [ALL_POP_LOCATIONS] + us_state_options + foreign_country_options
    contracting_options_key = office_options_scope_key(
        agency_name,
        bureau_name,
        int(fiscal_year),
        "contracting",
        ALL_CONTRACTING_OFFICES,
        market_filters,
    )
    office_options_cache = st.session_state.get("transaction_office_options_cache", {})
    cached_contracting_set = office_options_cache.get(
        contracting_options_key,
        {"options": [ALL_CONTRACTING_OFFICES], "stats": {}},
    )
    contracting_office_options = cached_contracting_set.get("options") or [ALL_CONTRACTING_OFFICES]
    if len(contracting_office_options) <= 1:
        try:
            started_at = time.time()
            _check_solicitation_mapping_deadline(deadline, "Contracting Office option retrieval")
            remaining = max(1.0, min(request_timeout, deadline - time.monotonic()))
            download_rows = fetch_transaction_download_office_rows(
                agency_name,
                bureau_name,
                int(fiscal_year),
                market_filters=market_filters,
                cache_version=APP_CACHE_VERSION,
                request_timeout_sec=request_timeout,
                max_elapsed_sec=remaining,
            )
            downloaded_contracting_set = scoped_office_option_set(
                download_rows,
                "awarding",
                ALL_CONTRACTING_OFFICES,
                ALL_CONTRACTING_OFFICES,
                market_filters,
            )
            contracting_office_options = merge_office_option_sets(
                ALL_CONTRACTING_OFFICES,
                "awarding",
                cached_contracting_set,
                downloaded_contracting_set,
            ).get("options", [ALL_CONTRACTING_OFFICES])
            record("Contracting Office options retrieval", "completed", started_at, "fetch_transaction_download_office_rows")
        except Exception as exc:
            contracting_office_options = [ALL_CONTRACTING_OFFICES]
            unavailable_fields.add("Contracting Office")
            record(
                "Contracting Office options retrieval",
                "timed_out" if isinstance(exc, TimeoutError) else "failed",
                started_at,
                "fetch_transaction_download_office_rows",
                exc,
            )

    funding_options_key = office_options_scope_key(
        agency_name,
        bureau_name,
        int(fiscal_year),
        "funding",
        ALL_CONTRACTING_OFFICES,
        market_filters,
    )
    cached_funding_set = office_options_cache.get(
        funding_options_key,
        {"options": [ALL_FUNDING_OFFICES], "stats": {}},
    )
    funding_office_options = cached_funding_set.get("options") or [ALL_FUNDING_OFFICES]

    return {
        "agencies": agency_names_from_records(agency_records),
        "bureaus": solicitation_bureau_options_for_agency(agency_records, agency_name, fiscal_year),
        "contracting_offices": contracting_office_options,
        "funding_offices": funding_office_options,
        "naics": naics_options,
        "psc": psc_options,
        "contract_types": default_filter_options()["contract_types"],
        "set_asides": default_filter_options()["set_asides"],
        "pop_locations": pop_location_options,
        "mapping_unavailable_fields": sorted(unavailable_fields),
        "mapping_diagnostics": {
            "timeoutSeconds": solicitation_mapping_timeout_sec(),
            "requestTimeoutSeconds": request_timeout,
            "elapsedSeconds": round(time.monotonic() - mapping_started, 3),
            "stageEvents": stage_events,
        },
        "scope": {
            "agency": agency_name,
            "bureau": bureau_name,
            "fiscal_year": int(fiscal_year),
        },
    }


def map_solicitation_signals_to_dashboard_filters(
    scope_preview: dict,
    agency_records: list[dict],
    fiscal_year: int,
    *,
    mapped_agency: str | None = None,
    mapped_bureau: str | None = None,
) -> dict:
    organization_mapping = map_solicitation_organization(scope_preview, agency_records, fiscal_year)
    effective_agency = mapped_agency or organization_mapping.get("mapped_agency")
    effective_bureau = canonical_bureau_name(mapped_bureau or organization_mapping.get("mapped_bureau"))
    if not effective_agency:
        effective_agency = agency_names_from_records(agency_records)[0]

    available_filter_options = build_solicitation_available_filter_options(
        agency_records,
        effective_agency,
        effective_bureau,
        int(fiscal_year),
    )
    market_mapping = map_solicitation_market_filters(scope_preview, available_filter_options)

    rows = [
        *organization_mapping["rows"],
        *market_mapping["rows"],
        *market_mapping["context_rows"],
    ]
    mapping_attempts = {
        **organization_mapping.get("mapping_attempts", {}),
        **market_mapping.get("mapping_attempts", {}),
    }
    pending_filters = {
        "agency": None,
        "bureau": None,
        "contracting_office": None,
        "market_filters": {},
    }
    unmapped_fields = [
        row["field"]
        for row in rows
        if row.get("filter_key") and row.get("mapping_status") in ("Unmapped", "Suggested match")
        and not row.get("filter_option")
    ]

    loaded_signal_ids = [field_data.get("signal_id") for field_data in scope_preview.values() if field_data.get("signal_id")]
    canonical_values = {
        field_name: field_data.get("value") for field_name, field_data in scope_preview.items()
    }
    confidences = {
        field_name: field_data.get("confidence") for field_name, field_data in scope_preview.items()
    }

    return {
        "rows": rows,
        "pending_filters": pending_filters,
        "available_filter_options": available_filter_options,
        "organization_mapping": organization_mapping,
        "debug": {
            "loaded_signal_ids": loaded_signal_ids,
            "canonical_values": canonical_values,
            "confidences": confidences,
            "mapping_attempts": mapping_attempts,
            "mapping_option_scope": available_filter_options.get("scope"),
            "final_selected_pending_filters": pending_filters,
            "unmapped_fields": unmapped_fields,
        },
    }


def degraded_solicitation_mapping_result(
    scope_preview: dict,
    agency_records: list[dict],
    fiscal_year: int,
    *,
    mapped_agency: str | None = None,
    mapped_bureau: str | None = None,
    reason: str = "Dashboard option matching has not completed.",
) -> dict:
    started = time.time()
    organization_mapping = map_solicitation_organization(scope_preview, agency_records, fiscal_year)
    effective_agency = mapped_agency or organization_mapping.get("mapped_agency") or agency_names_from_records(agency_records)[0]
    effective_bureau = canonical_bureau_name(mapped_bureau or organization_mapping.get("mapped_bureau"))
    defaults = default_filter_options()
    available_filter_options = {
        "agencies": agency_names_from_records(agency_records),
        "bureaus": solicitation_bureau_options_for_agency(agency_records, effective_agency, fiscal_year),
        "contracting_offices": [ALL_CONTRACTING_OFFICES],
        "funding_offices": [ALL_FUNDING_OFFICES],
        "naics": defaults["naics"],
        "psc": defaults["psc"],
        "contract_types": defaults["contract_types"],
        "set_asides": defaults["set_asides"],
        "pop_locations": [ALL_POP_LOCATIONS]
        + [encode_option(code, name) for code, name in sorted(STATE_OPTIONS.items())],
        "mapping_unavailable_fields": ["Contracting Office", "Funding Office", "NAICS", "PSC"],
        "mapping_diagnostics": {
            "status": "partial",
            "reason": reason,
            "elapsedSeconds": round(time.time() - started, 3),
            "stageEvents": [
                _solicitation_mapping_stage(
                    "Build degraded solicitation review rows",
                    started,
                    "completed",
                    function="degraded_solicitation_mapping_result",
                )
            ],
        },
        "scope": {
            "agency": effective_agency,
            "bureau": effective_bureau,
            "fiscal_year": int(fiscal_year),
        },
    }
    market_mapping = map_solicitation_market_filters(scope_preview, available_filter_options)
    rows = [
        *organization_mapping["rows"],
        *market_mapping["rows"],
        *market_mapping["context_rows"],
    ]
    return {
        "rows": rows,
        "pending_filters": {
            "agency": None,
            "bureau": None,
            "contracting_office": None,
            "market_filters": {},
        },
        "available_filter_options": available_filter_options,
        "organization_mapping": organization_mapping,
        "debug": {
            "mapping_degraded": True,
            "mapping_failure_reason": reason,
            "mapping_diagnostics": available_filter_options["mapping_diagnostics"],
        },
    }


def _auto_pending_filters_from_mapping_rows(rows: list[dict]) -> dict:
    """Build auto-suggested pending sidebar filters from high-confidence mapping rows only."""
    pending_sidebar_filters: dict = {
        "agency": None,
        "bureau": None,
        "contracting_office": None,
        "market_filters": {},
    }
    for row in rows:
        if not row.get("preselect") or not row.get("filter_option"):
            continue
        filter_key = row.get("filter_key")
        if filter_key == "agency":
            pending_sidebar_filters["agency"] = row["filter_option"]
        elif filter_key == "bureau":
            pending_sidebar_filters["bureau"] = row["filter_option"]
        elif filter_key == "contracting_office":
            pending_sidebar_filters["contracting_office"] = row["filter_option"]
        elif filter_key:
            pending_sidebar_filters["market_filters"][filter_key] = row["filter_option"]
    return pending_sidebar_filters


def apply_solicitation_pending_filters(
    pending_sidebar_filters: dict,
    available_filter_options: dict | None = None,
) -> None:
    """Apply solicitation mappings to pending sidebar state only.

    Uses active_* session keys, which mirror the sidebar selectors and remain
    distinct from analyzed_* keys used by the last completed dashboard run.
    analyzed_* is updated only by mark_analysis_started().
    """
    available_filter_options = available_filter_options or {}

    def option_allowed(filter_key: str | None, selected: str | None, field_name: str | None = None) -> bool:
        if not selected or selected == KEEP_CURRENT_SOLICITATION_FILTER:
            return False
        if filter_key == "agency":
            return selected in available_filter_options.get("agencies", [])
        if filter_key == "bureau":
            return selected in available_filter_options.get("bureaus", [])
        if filter_key == "contracting_office":
            return selected in available_filter_options.get("contracting_offices", [])
        if filter_key == "naics_code":
            return selected in available_filter_options.get("naics", [])
        if filter_key == "psc_code":
            return selected in available_filter_options.get("psc", [])
        if filter_key == "contract_type":
            return selected in available_filter_options.get("contract_types", [])
        if filter_key == "set_aside_type":
            return selected in available_filter_options.get("set_asides", [])
        if filter_key == "pop_state":
            return selected in available_filter_options.get("pop_locations", [])
        if filter_key == "funding_office":
            return selected in available_filter_options.get("funding_offices", [])
        return False

    agency = pending_sidebar_filters.get("agency")
    if option_allowed("agency", agency):
        st.session_state.active_agency = agency

    bureau = canonical_bureau_name(pending_sidebar_filters.get("bureau"))
    if option_allowed("bureau", bureau):
        st.session_state.active_bureau = bureau

    contracting_office = pending_sidebar_filters.get("contracting_office")
    if option_allowed("contracting_office", contracting_office):
        st.session_state.active_contracting_office = contracting_office

    market_filters = normalize_market_filters(st.session_state.active_market_filters)
    for key, value in (pending_sidebar_filters.get("market_filters") or {}).items():
        if option_allowed(key, value):
            market_filters[key] = value
    st.session_state.active_market_filters = market_filters
    st.session_state.solicitation_last_applied_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    st.session_state.solicitation_last_applied_filters = {
        "agency": st.session_state.get("active_agency"),
        "bureau": st.session_state.get("active_bureau"),
        "fiscal_year": st.session_state.get("active_fiscal_year"),
        "contracting_office": st.session_state.get("active_contracting_office"),
        "market_filters": normalize_market_filters(st.session_state.get("active_market_filters")),
    }


def utc_analysis_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def classify_analysis_data_source(*source_labels: str | None) -> str:
    labels = [str(label or "").strip().lower() for label in source_labels if str(label or "").strip()]
    if any("fixture" in label or "upload" in label or "file" in label for label in labels):
        return "uploaded_fixture_or_file"
    if any("live" in label for label in labels):
        return "live_usaspending_api"
    if any("error" in label for label in labels):
        return "api_error"
    return "unknown"


def build_usaspending_filter_summary(payload: dict | None) -> dict:
    if not isinstance(payload, dict):
        return {}
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    return {
        "agencies": filters.get("agencies"),
        "time_period": filters.get("time_period"),
        "naics_codes": filters.get("naics_codes"),
        "psc_codes": filters.get("psc_codes"),
        "contract_pricing_type_codes": filters.get("contract_pricing_type_codes"),
        "set_aside_type_codes": filters.get("set_aside_type_codes"),
        "place_of_performance_locations": filters.get("place_of_performance_locations"),
        "award_type_codes": filters.get("award_type_codes"),
        "award_or_idv_flag": filters.get("award_or_idv_flag"),
    }


def solicitation_validation_context() -> dict:
    context: dict = {}
    if st.session_state.get("solicitation_resolved_signals"):
        context["signals_artifact_source"] = st.session_state.get(
            "solicitation_signals_source",
            "uploaded_resolved_signals_json",
        )
        context["signals_loaded_at"] = st.session_state.get("solicitation_signals_loaded_at")
        context["signals_run_id"] = st.session_state.get("solicitation_loaded_run_id")
        context["signals_dev_path"] = st.session_state.get("solicitation_dev_path")
    if st.session_state.get("solicitation_last_applied_at"):
        context["solicitation_filters_applied_at"] = st.session_state.get("solicitation_last_applied_at")
        context["solicitation_filters_applied"] = st.session_state.get("solicitation_last_applied_filters")
    mapping_result = st.session_state.get("solicitation_mapping_result")
    if isinstance(mapping_result, dict):
        context["mapping_option_scope"] = mapping_result.get("debug", {}).get("mapping_option_scope")
        context["final_selected_pending_filters"] = mapping_result.get("debug", {}).get(
            "final_selected_pending_filters"
        )
    return context


def build_analysis_validation_metadata(
    *,
    agency: str,
    bureau: str | None,
    fiscal_year: int,
    contracting_office: str | None = None,
    market_filters: dict | None = None,
    active_scope_payload: dict | None = None,
    usaspending_payload_summary: dict | None = None,
    transaction_row_count: int | None = None,
    unique_award_count: int | None = None,
    total_obligations: float | None = None,
    data_source_labels: list[str] | None = None,
    analysis_context: str = "dashboard",
) -> dict:
    market_filters = normalize_market_filters(market_filters)
    solicitation_context = solicitation_validation_context()
    return {
        "as_of": utc_analysis_timestamp(),
        "analysis_context": analysis_context,
        "analysis_run_started_at": st.session_state.get("analysis_run_started_at"),
        "analysis_run_completed_at": st.session_state.get("analysis_run_completed_at"),
        "selected_agency": agency,
        "selected_subagency_bureau": canonical_bureau_name(bureau),
        "selected_fiscal_year": int(fiscal_year),
        "selected_fiscal_year_label": fiscal_year_label(int(fiscal_year)),
        "active_filter_scope": active_scope_payload or active_filter_payload(
            agency,
            bureau,
            int(fiscal_year),
            market_filters,
            contracting_office,
        ),
        "active_mapped_solicitation_filters": solicitation_context.get("solicitation_filters_applied"),
        "solicitation_signals_context": {
            key: value
            for key, value in solicitation_context.items()
            if key not in ("solicitation_filters_applied", "final_selected_pending_filters")
        },
        "usaspending_filter_summary": usaspending_payload_summary or {},
        "transaction_row_count": transaction_row_count,
        "unique_award_count": unique_award_count,
        "total_obligations_returned": round(float(total_obligations), 2) if total_obligations is not None else None,
        "data_source": classify_analysis_data_source(*(data_source_labels or [])),
        "app_cache_version": APP_CACHE_VERSION,
    }


def default_filter_options() -> dict:
    return {
        "naics": [ALL_NAICS_CODES],
        "contract_types": [ALL_CONTRACT_TYPES]
        + [
            encode_option(code, label)
            for code, label in sorted(CONTRACT_TYPE_OPTIONS.items(), key=lambda item: item[1])
        ],
        "psc": [ALL_PRODUCT_SERVICE_CODES],
        "set_asides": [ALL_SET_ASIDE_TYPES]
        + [
            encode_option(code, label)
            for code, label in sorted(SET_ASIDE_TYPE_OPTIONS.items(), key=lambda item: item[1])
        ],
        "pop_locations": [ALL_POP_LOCATIONS],
        "offices": [ALL_CONTRACTING_OFFICES],
        "funding_offices": [ALL_FUNDING_OFFICES],
    }


def normalize_market_filters(market_filters: dict | None) -> dict:
    normalized = default_market_filters()
    if isinstance(market_filters, dict):
        normalized.update({key: value for key, value in market_filters.items() if value})
    if normalized.get("pop_state") == LEGACY_ALL_POP_STATES:
        normalized["pop_state"] = ALL_POP_LOCATIONS
    return normalized


def market_filters_without_contract_type(market_filters: dict | None) -> dict:
    normalized = normalize_market_filters(market_filters)
    normalized["contract_type"] = ALL_CONTRACT_TYPES
    return normalized


def market_filters_without_offices(market_filters: dict | None) -> dict:
    normalized = normalize_market_filters(market_filters)
    normalized["funding_office"] = ALL_FUNDING_OFFICES
    return normalized


def office_download_scope_key(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    market_filters: dict | None,
) -> str:
    market_filters = market_filters_without_offices(market_filters)
    return json.dumps(
        {
            "agency": agency_name,
            "bureau": bureau_name or ALL_BUREAUS,
            "fiscal_year": int(fiscal_year),
            "naics": market_filters["naics_code"],
            "contract_type": market_filters["contract_type"],
            "psc": market_filters["psc_code"],
            "set_aside": market_filters["set_aside_type"],
            "state": market_filters["pop_state"],
        },
        sort_keys=True,
    )


def office_options_scope_key(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    office_kind: str,
    contracting_office: str | None,
    market_filters: dict | None,
) -> str:
    market_filters = normalize_market_filters(market_filters)
    contracting_code, contracting_name = decode_contracting_office(contracting_office)
    funding_code, funding_name = decode_option(market_filters["funding_office"])
    return json.dumps(
        {
            "agency": agency_name,
            "bureau": bureau_name or ALL_BUREAUS,
            "fiscal_year": int(fiscal_year),
            "office_kind": office_kind,
            "naics": market_filters["naics_code"],
            "contract_type": market_filters["contract_type"],
            "psc": market_filters["psc_code"],
            "set_aside": market_filters["set_aside_type"],
            "contracting_office": ""
            if office_kind == "contracting"
            else {"code": contracting_code, "name": contracting_name},
            "funding_office": ""
            if office_kind == "funding"
            else {"code": funding_code, "name": funding_name},
            "state": market_filters["pop_state"],
        },
        sort_keys=True,
    )


def option_label(option: str, default_option: str, default_label: str, prefix: str | None = None) -> str:
    if not option or option == default_option:
        return default_label
    code, description = decode_option(option)
    label = description or code or option
    if prefix and code:
        return f"{prefix} {code} — {label}" if description else f"{prefix} {label}"
    return label


def market_filter_labels(market_filters: dict | None, contracting_office: str | None) -> dict:
    market_filters = normalize_market_filters(market_filters)
    return {
        "naics": option_label(market_filters["naics_code"], ALL_NAICS_CODES, "All NAICS", "NAICS"),
        "contract_type": option_label(
            market_filters["contract_type"],
            ALL_CONTRACT_TYPES,
            "All Contract Types",
            "Contract Type:",
        ),
        "psc": option_label(market_filters["psc_code"], ALL_PRODUCT_SERVICE_CODES, "All PSCs", "PSC:"),
        "set_aside": option_label(
            market_filters["set_aside_type"],
            ALL_SET_ASIDE_TYPES,
            "All Set-Asides",
            "Set-Aside:",
        ),
        "contracting_office": (
            "All Contracting Offices"
            if not contracting_office or contracting_office == ALL_CONTRACTING_OFFICES
            else f"Contracting Office: {format_contracting_office_option(contracting_office)}"
        ),
        "funding_office": (
            "All Funding Offices"
            if market_filters["funding_office"] == ALL_FUNDING_OFFICES
            else f"Funding Office: {format_funding_office_option(market_filters['funding_office'])}"
        ),
        "place_of_performance": option_label(
            market_filters["pop_state"],
            ALL_POP_LOCATIONS,
            "All Locations",
            "Place of Performance:",
        ),
    }


def has_active_refinements(market_filters: dict | None, contracting_office: str | None) -> bool:
    market_filters = normalize_market_filters(market_filters)
    return any(
        [
            market_filters["naics_code"] != ALL_NAICS_CODES,
            market_filters["contract_type"] != ALL_CONTRACT_TYPES,
            market_filters["psc_code"] != ALL_PRODUCT_SERVICE_CODES,
            market_filters["set_aside_type"] != ALL_SET_ASIDE_TYPES,
            market_filters["funding_office"] != ALL_FUNDING_OFFICES,
            market_filters["pop_state"] not in (ALL_POP_LOCATIONS, LEGACY_ALL_POP_STATES),
            bool(contracting_office and contracting_office != ALL_CONTRACTING_OFFICES),
        ]
    )


def active_filter_payload(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    market_filters: dict | None,
    contracting_office: str | None,
) -> dict:
    bureau_name = canonical_bureau_name(bureau_name)
    market_filters = normalize_market_filters(market_filters)
    start_date, end_date = fiscal_year_date_range(fiscal_year)
    naics_code, naics_label = decode_option(market_filters["naics_code"])
    contract_type_code, contract_type_label = decode_option(market_filters["contract_type"])
    psc_code, psc_label = decode_option(market_filters["psc_code"])
    set_aside_code, set_aside_label = decode_option(market_filters["set_aside_type"])
    contracting_office_code, contracting_office_label = decode_contracting_office(contracting_office)
    funding_office_code, funding_office_label = decode_option(market_filters["funding_office"])
    pop_code, pop_label = decode_option(market_filters["pop_state"])
    pop_filter = (
        {}
        if pop_code in (ALL_POP_LOCATIONS, LEGACY_ALL_POP_STATES)
        else build_place_of_performance_location_filter(pop_code)
    )
    return {
        "agency": agency_name,
        "subagency_bureau": bureau_name or ALL_BUREAUS,
        "fiscal_year": fiscal_year_label(int(fiscal_year)),
        "fiscal_year_date_range": {"start_date": start_date, "end_date": end_date},
        "naics": {"code": "" if naics_code == ALL_NAICS_CODES else naics_code, "label": naics_label or ALL_NAICS_CODES},
        "contract_type": {
            "type_of_contract_pricing_code": "" if contract_type_code == ALL_CONTRACT_TYPES else contract_type_code,
            "type_of_contract_pricing": contract_type_label or ALL_CONTRACT_TYPES,
        },
        "psc": {"code": "" if psc_code == ALL_PRODUCT_SERVICE_CODES else psc_code, "label": psc_label or ALL_PRODUCT_SERVICE_CODES},
        "set_aside": {
            "code": "" if set_aside_code == ALL_SET_ASIDE_TYPES else set_aside_code,
            "label": set_aside_label or ALL_SET_ASIDE_TYPES,
        },
        "contracting_office": {
            "awarding_office_code": ""
            if not contracting_office_code or contracting_office == ALL_CONTRACTING_OFFICES
            else contracting_office_code,
            "awarding_office_name": contracting_office_label or ALL_CONTRACTING_OFFICES,
        },
        "funding_office": {
            "funding_office_code": "" if funding_office_code == ALL_FUNDING_OFFICES else funding_office_code,
            "funding_office_name": funding_office_label or ALL_FUNDING_OFFICES,
        },
        "place_of_performance": {
            "code": "" if pop_code in (ALL_POP_LOCATIONS, LEGACY_ALL_POP_STATES) else pop_code,
            "label": pop_label or ALL_POP_LOCATIONS,
            "country": pop_filter.get("country", ""),
            "state": pop_filter.get("state", ""),
        },
    }


def clear_active_refinements() -> None:
    st.session_state.active_market_filters = default_market_filters()
    st.session_state.active_contracting_office = ALL_CONTRACTING_OFFICES


def reset_active_refinement(refinement_key: str) -> None:
    filters = normalize_market_filters(st.session_state.active_market_filters)
    if refinement_key == "contracting_office":
        st.session_state.active_contracting_office = ALL_CONTRACTING_OFFICES
    elif refinement_key in filters:
        filters[refinement_key] = default_market_filters()[refinement_key]
        st.session_state.active_market_filters = filters


def _applied_filter_chips(
    selected_bureau: str | None,
    selected_market_filters: dict,
    selected_contracting_office: str,
) -> list[dict]:
    labels = market_filter_labels(selected_market_filters, selected_contracting_office)
    market_filters = normalize_market_filters(selected_market_filters)
    chips = []
    if canonical_bureau_name(selected_bureau) != ALL_BUREAUS:
        chips.append({"key": "bureau", "label": f"Subagency/Bureau: {canonical_bureau_name(selected_bureau)}"})
    chip_config = [
        ("contracting_office", selected_contracting_office, ALL_CONTRACTING_OFFICES, labels["contracting_office"]),
        ("naics_code", market_filters["naics_code"], ALL_NAICS_CODES, labels["naics"]),
        ("contract_type", market_filters["contract_type"], ALL_CONTRACT_TYPES, labels["contract_type"]),
        ("psc_code", market_filters["psc_code"], ALL_PRODUCT_SERVICE_CODES, labels["psc"]),
        ("set_aside_type", market_filters["set_aside_type"], ALL_SET_ASIDE_TYPES, labels["set_aside"]),
        ("funding_office", market_filters["funding_office"], ALL_FUNDING_OFFICES, labels["funding_office"]),
        ("pop_state", market_filters["pop_state"], ALL_POP_LOCATIONS, labels["place_of_performance"]),
    ]
    for key, value, default_value, label in chip_config:
        active = value not in (ALL_POP_LOCATIONS, LEGACY_ALL_POP_STATES) if key == "pop_state" else value != default_value
        if active:
            chips.append({"key": key, "label": label})
    return chips


def remove_applied_filter_and_rerun(
    filter_key: str,
    active_agency: str,
    selected_bureau: str | None,
    selected_year: int,
    selected_market_filters: dict,
    selected_contracting_office: str,
) -> None:
    next_bureau = canonical_bureau_name(selected_bureau)
    next_contracting_office = selected_contracting_office
    next_market_filters = normalize_market_filters(selected_market_filters)
    if filter_key == "bureau":
        next_bureau = ALL_BUREAUS
        st.session_state.active_bureau = ALL_BUREAUS
    elif filter_key == "contracting_office":
        next_contracting_office = ALL_CONTRACTING_OFFICES
        st.session_state.active_contracting_office = ALL_CONTRACTING_OFFICES
        st.session_state.solicitation_comparable_market = False
    elif filter_key in next_market_filters:
        next_market_filters[filter_key] = default_market_filters()[filter_key]
        st.session_state.active_market_filters = next_market_filters
    else:
        return

    st.session_state.active_agency = active_agency
    st.session_state.active_fiscal_year = int(selected_year)
    st.session_state.active_contracting_office = next_contracting_office
    st.session_state.active_market_filters = next_market_filters
    mark_analysis_started(
        active_agency,
        next_bureau,
        int(selected_year),
        next_contracting_office,
        next_market_filters,
    )
    st.rerun()


def format_money(value) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "$0.00"
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1_000_000_000_000:
        return f"{sign}${amount / 1_000_000_000_000:.3f}T"
    if amount >= 1_000_000_000:
        return f"{sign}${amount / 1_000_000_000:.3f}B"
    if amount >= 10_000_000:
        return f"{sign}${amount / 1_000_000:.1f}M"
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:.2f}M"
    if amount >= 1_000:
        formatted = f"{amount / 1_000:.1f}".removesuffix(".0")
        return f"{sign}${formatted}K"
    return f"{sign}${amount:,.2f}"


def format_full_money(value) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        amount = 0.0
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def format_money_with_full(value) -> str:
    return f"{format_money(value)} ({format_full_money(value)})"


def format_count(value) -> str:
    try:
        return f"{int(value):,}"
    except (TypeError, ValueError):
        return "0"


def json_safe_payload(value, seen: set[int] | None = None):
    if seen is None:
        seen = set()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    value_id = id(value)
    if value_id in seen:
        return "[Circular reference omitted]"
    if isinstance(value, dict):
        seen.add(value_id)
        safe_dict = {str(key): json_safe_payload(item, seen) for key, item in value.items()}
        seen.remove(value_id)
        return safe_dict
    if isinstance(value, (list, tuple, set)):
        seen.add(value_id)
        safe_list = [json_safe_payload(item, seen) for item in value]
        seen.remove(value_id)
        return safe_list
    if isinstance(value, (date, pd.Timestamp)):
        return value.isoformat()
    return str(value)


def format_delta(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:+.1f}%"


def money_ticks(max_value: float) -> tuple[list[float], list[str]]:
    if not max_value or max_value <= 0:
        return [0], ["$0.00"]
    ticks = [max_value * i / 4 for i in range(5)]
    return ticks, [format_money(tick) for tick in ticks]


def get_openai_api_key() -> str | None:
    try:
        secret_key = st.secrets.get("OPENAI_API_KEY")
        if secret_key:
            return str(secret_key)
    except Exception:
        pass

    env_key = os.getenv("OPENAI_API_KEY")
    if env_key:
        return env_key

    env_path = Path.cwd() / ".env"
    if env_path.exists():
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key.strip() == "OPENAI_API_KEY" and value.strip():
                    return value.strip().strip('"').strip("'")
        except OSError:
            return None
    return None


def agency_filter(agency_name: str, bureau_name: str | None = None) -> list[dict]:
    toptier_name = normalize_agency_name(agency_name)
    bureau_filter_name = resolve_bureau_filter_name(bureau_name)
    if bureau_filter_name:
        return [
            {
                "type": "awarding",
                "tier": "subtier",
                "name": bureau_filter_name,
                "toptier_name": toptier_name,
            }
        ]
    return [{"type": "awarding", "tier": "toptier", "name": toptier_name}]


def contract_award_filters(
    agency_name: str,
    bureau_name: str | None = None,
    market_filters: dict | None = None,
    time_period: list[dict] | None = None,
) -> dict:
    filters = {
        "agencies": agency_filter(agency_name, bureau_name),
        "award_type_codes": AWARD_TYPE_CODES,
        "award_or_idv_flag": AWARD_OR_IDV_FLAG,
    }
    market_filters = normalize_market_filters(market_filters)

    naics_code, _naics_description = decode_option(market_filters["naics_code"])
    if naics_code and naics_code != ALL_NAICS_CODES:
        filters["naics_codes"] = {"require": [naics_code]}

    contract_type_code, _contract_type_description = decode_option(market_filters["contract_type"])
    if contract_type_code and contract_type_code != ALL_CONTRACT_TYPES:
        filters["contract_pricing_type_codes"] = [contract_type_code]

    psc_code, _psc_description = decode_option(market_filters["psc_code"])
    if psc_code and psc_code != ALL_PRODUCT_SERVICE_CODES:
        filters["psc_codes"] = [psc_code]

    set_aside_code, _set_aside_description = decode_option(market_filters["set_aside_type"])
    if set_aside_code and set_aside_code != ALL_SET_ASIDE_TYPES:
        filters["set_aside_type_codes"] = [set_aside_code]

    pop_code, _pop_description = decode_option(market_filters["pop_state"])
    if pop_code and pop_code not in (ALL_POP_LOCATIONS, LEGACY_ALL_POP_STATES):
        filters["place_of_performance_locations"] = [build_place_of_performance_location_filter(pop_code)]

    if time_period:
        filters["time_period"] = time_period

    return filters


def dashboard_base_filters(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    market_filters: dict | None = None,
) -> dict:
    if fiscal_year is not None and (start_date is None or end_date is None):
        start_date, end_date = fiscal_year_date_range(fiscal_year)
    time_period = [{"start_date": start_date, "end_date": end_date}] if start_date and end_date else None
    return contract_award_filters(
        agency_name,
        canonical_bureau_name(bureau_name),
        market_filters=market_filters,
        time_period=time_period,
    )


def build_trends_payload(
    agency_name: str,
    bureau_name: str | None = None,
    time_period: list[dict] | None = None,
    market_filters: dict | None = None,
) -> dict:
    filters = dashboard_base_filters(
        agency_name,
        bureau_name,
        market_filters=market_filters,
    )
    if time_period:
        filters = dashboard_base_filters(
            agency_name,
            bureau_name,
            start_date=time_period[0].get("start_date"),
            end_date=time_period[0].get("end_date"),
            market_filters=market_filters,
        )
    return {
        "group": "fiscal_year",
        "spending_level": "transactions",
        "filters": filters,
    }


def time_grain_group(time_grain: str) -> str:
    if time_grain == TIME_GRAIN_MONTH:
        return "month"
    if time_grain == TIME_GRAIN_FISCAL_QUARTER:
        return "quarter"
    return "fiscal_year"


def build_obligations_time_series_payload(
    agency_name: str,
    bureau_name: str | None,
    selected_year: int,
    time_grain: str,
    market_filters: dict | None = None,
) -> dict:
    filters = dashboard_base_filters(
        agency_name,
        bureau_name,
        fiscal_year=selected_year,
        market_filters=market_filters,
    )
    return {
        "group": time_grain_group(time_grain),
        "spending_level": "transactions",
        "filters": filters,
    }


def build_vendor_payload(
    agency_name: str,
    bureau_name: str | None = None,
    market_filters: dict | None = None,
) -> dict:
    return {
        "category": "recipient",
        "spending_level": "awards",
        "limit": 10,
        "page": 1,
        "filters": dashboard_base_filters(agency_name, bureau_name, market_filters=market_filters),
    }


def build_category_options_payload(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    category: str,
    market_filters: dict | None = None,
    limit: int = 50,
) -> dict:
    filters = dashboard_base_filters(
        agency_name,
        bureau_name,
        fiscal_year=fiscal_year,
        market_filters=market_filters,
    )
    return {
        "category": category,
        "spending_level": "transactions",
        "limit": limit,
        "page": 1,
        "filters": filters,
    }


def build_transaction_download_office_payload(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    market_filters: dict | None = None,
    limit: int = 10_000,
) -> dict:
    filters = dashboard_base_filters(
        agency_name,
        bureau_name,
        fiscal_year=fiscal_year,
        market_filters=market_filters_without_offices(market_filters),
    )
    return {
        "filters": filters,
        "columns": [
            "contract_award_unique_key",
            "award_id_piid",
            "modification_number",
            "transaction_description",
            "federal_action_obligation",
            "total_dollars_obligated",
            "total_outlayed_amount_for_overall_award",
            "current_total_value_of_award",
            "potential_total_value_of_award",
            "action_date",
            "action_type",
            "recipient_name",
            "naics_code",
            "naics_description",
            "product_or_service_code",
            "product_or_service_code_description",
            "awarding_office_code",
            "awarding_office_name",
            "funding_office_code",
            "funding_office_name",
        ],
        "file_format": "csv",
        "limit": limit,
    }


def build_award_scope_download_payload(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    market_filters: dict | None = None,
    limit: int = 10_000,
) -> dict:
    filters = dashboard_base_filters(
        agency_name,
        bureau_name,
        fiscal_year=fiscal_year,
        market_filters=market_filters_without_offices(market_filters),
    )
    return {
        "filters": filters,
        "columns": AWARD_SCOPE_DOWNLOAD_COLUMNS,
        "file_format": "csv",
        "limit": limit,
    }


def fiscal_year_date_range(fiscal_year: int) -> tuple[str, str]:
    return f"{int(fiscal_year) - 1}-10-01", f"{int(fiscal_year)}-09-30"


def prior_ytd_date_range(fiscal_year: int, as_of: date | None = None) -> tuple[str, str]:
    as_of = as_of or date.today()
    current_start = date(int(fiscal_year) - 1, 10, 1)
    prior_start = date(int(fiscal_year) - 2, 10, 1)
    elapsed_days = max((as_of - current_start).days, 0)
    prior_end = prior_start + timedelta(days=elapsed_days)
    prior_fy_end = date(int(fiscal_year) - 1, 9, 30)
    if prior_end > prior_fy_end:
        prior_end = prior_fy_end
    return prior_start.isoformat(), prior_end.isoformat()


def build_transaction_payload(
    agency_name: str,
    bureau_name: str | None,
    start_date: str,
    end_date: str,
    page: int = 1,
    include_office_fields: bool = True,
    market_filters: dict | None = None,
) -> dict:
    return {
        "filters": dashboard_base_filters(
            agency_name,
            bureau_name,
            start_date=start_date,
            end_date=end_date,
            market_filters=market_filters,
        ),
        "fields": TRANSACTION_FIELDS if include_office_fields else BASE_TRANSACTION_FIELDS,
        "limit": 100,
        "page": page,
        "sort": "Action Date",
        "order": "desc",
    }


def post_usaspending(endpoint: str, payload: dict, *, timeout_sec: float = 18) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            json=payload,
            headers=request_headers(),
            timeout=timeout_sec,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as e:
        if endpoint == "/api/v2/search/spending_by_transaction/":
            print("--- USASPENDING API DIAGNOSTIC LOG ---")
            if e.response:
                print(e.response.text)
        try:
            error_msg = e.response.json()
        except Exception:
            error_msg = e.response.text if e.response else str(e)
        return None, f"API Validation Breakdown: {error_msg}"
    except requests.RequestException as exc:
        return None, f"{type(exc).__name__}: live USAspending request unavailable"
    except ValueError:
        return None, "Invalid JSON response from USAspending"


def parse_autocomplete_names(data: dict) -> list[str]:
    results = data.get("results") or []
    names: list[str] = []
    for item in results:
        if isinstance(item, str):
            names.append(item)
            continue
        if not isinstance(item, dict):
            continue
        for key in ("name", "agency_name", "toptier_name", "subtier_name"):
            value = item.get(key)
            if isinstance(value, str):
                names.append(value)
        for nested_key in ("toptier_agency", "subtier_agency", "agency"):
            nested = item.get(nested_key)
            if isinstance(nested, dict) and isinstance(nested.get("name"), str):
                names.append(nested["name"])
    seen = set()
    unique = []
    for name in names:
        normalized = normalize_agency_name(name)
        key = normalized.lower()
        if key not in seen:
            unique.append(normalized)
            seen.add(key)
    return unique[:10]


@st.cache_data(ttl=3600, show_spinner=False)
def agency_autocomplete(
    search_text: str,
    cache_version: str = APP_CACHE_VERSION,
) -> list[str]:
    payload = {"search_text": search_text or "", "limit": 10}
    data, _ = post_usaspending("/api/v2/autocomplete/awarding_agency/", payload)
    if data:
        names = parse_autocomplete_names(data)
        if names:
            return names

    needle = (search_text or "").strip().lower()
    live_names = agency_names_from_records(fetch_toptier_agencies())
    if not needle:
        return live_names[:10]
    filtered = [agency for agency in live_names if needle in agency.lower()]
    alias_match = AGENCY_ALIASES.get(needle)
    if alias_match and alias_match not in filtered:
        filtered.insert(0, alias_match)
    return (filtered or live_names)[:10]


def normalize_trend_response(data: dict) -> pd.DataFrame:
    rows = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        period = item.get("time_period") if isinstance(item.get("time_period"), dict) else {}
        fiscal_year = (
            first_present(item, ["fiscal_year", "label", "period", "x_axis"])
            or period.get("fiscal_year")
            or period.get("start_date", "")[:4]
        )
        amount = first_present(
            item,
            ["aggregated_amount", "amount", "total", "obligation", "award_amount"],
        )
        try:
            rows.append(
                {
                    "fiscal_year": int(str(fiscal_year).replace("FY", "").strip()),
                    "amount": float(amount or 0),
                }
            )
        except (TypeError, ValueError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return (
        df.groupby("fiscal_year", as_index=False)["amount"]
        .sum()
        .sort_values("fiscal_year")
    )


def parse_action_date(value) -> date | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def fiscal_year_from_date(action_date: date) -> int:
    return action_date.year + 1 if action_date.month >= 10 else action_date.year


def fiscal_quarter_from_date(action_date: date) -> int:
    return ((action_date.month - 10) % 12) // 3 + 1


def fiscal_period_sort_key(action_date: date, time_grain: str) -> tuple[int, int]:
    fiscal_year = fiscal_year_from_date(action_date)
    if time_grain == TIME_GRAIN_MONTH:
        return fiscal_year, (action_date.month - 10) % 12 + 1
    if time_grain == TIME_GRAIN_FISCAL_QUARTER:
        return fiscal_year, fiscal_quarter_from_date(action_date)
    return fiscal_year, 0


def fiscal_month_start_date(fiscal_year: int, fiscal_month: int) -> date:
    calendar_month = ((int(fiscal_month) + 8) % 12) + 1
    calendar_year = int(fiscal_year) - 1 if calendar_month >= 10 else int(fiscal_year)
    return date(calendar_year, calendar_month, 1)


def spending_period_start_date(period: dict, time_grain: str) -> date | None:
    fiscal_year = period.get("fiscal_year")
    try:
        fiscal_year = int(str(fiscal_year).replace("FY", "").strip())
    except (TypeError, ValueError):
        return None

    if time_grain == TIME_GRAIN_MONTH and period.get("month") is not None:
        try:
            return fiscal_month_start_date(fiscal_year, int(period["month"]))
        except (TypeError, ValueError):
            return None

    if time_grain == TIME_GRAIN_FISCAL_QUARTER and period.get("quarter") is not None:
        try:
            fiscal_month = (int(period["quarter"]) - 1) * 3 + 1
            return fiscal_month_start_date(fiscal_year, fiscal_month)
        except (TypeError, ValueError):
            return None

    if time_grain == TIME_GRAIN_FISCAL_YEAR:
        return date(fiscal_year - 1, 10, 1)

    return None


def is_future_current_fy_bucket(action_date: date, time_grain: str) -> bool:
    if time_grain == TIME_GRAIN_FISCAL_YEAR:
        return False
    return fiscal_year_from_date(action_date) == current_fiscal_year() and action_date > date.today()


def is_current_incomplete_bucket(action_date: date, time_grain: str) -> bool:
    today = date.today()
    if fiscal_year_from_date(action_date) != current_fiscal_year():
        return False
    if time_grain == TIME_GRAIN_MONTH:
        return action_date.year == today.year and action_date.month == today.month
    if time_grain == TIME_GRAIN_FISCAL_QUARTER:
        return fiscal_quarter_from_date(action_date) == fiscal_quarter_from_date(today)
    return True


def time_bucket_label(action_date: date, time_grain: str) -> str:
    fiscal_year = fiscal_year_from_date(action_date)
    if time_grain == TIME_GRAIN_MONTH:
        label = action_date.strftime("%b %Y")
    elif time_grain == TIME_GRAIN_FISCAL_QUARTER:
        label = f"FY{fiscal_year} Q{fiscal_quarter_from_date(action_date)}"
    else:
        label = f"FY{fiscal_year}"
    if is_current_incomplete_bucket(action_date, time_grain):
        return f"{label} YTD"
    return label


def normalize_obligation_time_series_response(data: dict, time_grain: str) -> pd.DataFrame:
    rows = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        period = item.get("time_period") if isinstance(item.get("time_period"), dict) else {}
        start_date = parse_action_date(
            first_present(item, ["start_date", "period_start", "date"])
            or period.get("start_date")
            or period.get("period_start")
        )
        if start_date is None:
            fiscal_period = {
                "fiscal_year": (
                    first_present(item, ["fiscal_year", "label", "period", "x_axis"])
                    or period.get("fiscal_year")
                    or period.get("start_date", "")[:4]
                ),
                "month": period.get("month") or item.get("month"),
                "quarter": period.get("quarter") or item.get("quarter"),
            }
            start_date = spending_period_start_date(fiscal_period, time_grain)
        if start_date is None:
            continue
        if is_future_current_fy_bucket(start_date, time_grain):
            continue
        amount = first_present(
            item,
            ["aggregated_amount", "amount", "total", "obligation", "award_amount"],
        )
        transaction_count = first_present(
            item,
            ["transaction_count", "transaction_count_sum", "count", "total_transactions"],
        )
        try:
            rows.append(
                {
                    "bucket_label": time_bucket_label(start_date, time_grain),
                    "sort_fiscal_year": fiscal_period_sort_key(start_date, time_grain)[0],
                    "sort_period": fiscal_period_sort_key(start_date, time_grain)[1],
                    "amount": float(amount or 0),
                    "transaction_count": int(transaction_count) if transaction_count is not None else None,
                }
            )
        except (TypeError, ValueError):
            continue

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["bucket_label", "amount", "transaction_count"])
    grouped = (
        df.groupby(["bucket_label", "sort_fiscal_year", "sort_period"], as_index=False)
        .agg(
            amount=("amount", "sum"),
            transaction_count=(
                "transaction_count",
                lambda values: values.sum() if values.notna().any() else None,
            ),
        )
        .sort_values(["sort_fiscal_year", "sort_period"])
    )
    return grouped[["bucket_label", "amount", "transaction_count"]]


def normalize_vendor_response(data: dict) -> tuple[pd.DataFrame, int | None]:
    rows = []
    for item in data.get("results") or []:
        if not isinstance(item, dict):
            continue
        name = first_present(
            item,
            [
                "recipient_name",
                "name",
                "label",
                "category",
                "recipient",
                "description",
            ],
        )
        amount = first_present(item, ["amount", "aggregated_amount", "total", "obligation"])
        if not name:
            continue
        try:
            rows.append({"recipient": str(name), "amount": float(amount or 0)})
        except (TypeError, ValueError):
            continue

    metadata = data.get("page_metadata") or data.get("pagination") or {}
    contractor_count = first_present(
        metadata,
        ["total", "total_results", "total_records", "count", "total_entries"],
    )
    try:
        contractor_count = int(contractor_count) if contractor_count is not None else None
    except (TypeError, ValueError):
        contractor_count = None

    df = pd.DataFrame(rows)
    if df.empty:
        return df, contractor_count
    df = df.sort_values("amount", ascending=False).head(10)
    return df, contractor_count


def parse_currency_amount(value) -> float:
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    cleaned = str(value).strip().replace("$", "").replace(",", "")
    if cleaned.startswith("(") and cleaned.endswith(")"):
        cleaned = f"-{cleaned[1:-1]}"
    try:
        return float(cleaned)
    except ValueError:
        return 0.0


def parse_optional_currency_amount(value) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    return parse_currency_amount(text)


def transaction_amount(item: dict) -> float:
    return parse_currency_amount(
        first_present(
            item,
            [
                "Transaction Amount",
                "transaction_amount",
                "transaction_obligated_amount",
                "Federal Action Obligation",
                "federal_action_obligation",
                "Award Amount",
                "award_amount",
                "obligation",
                "amount",
            ],
        )
    )


def clean_office_value(value) -> str:
    return " ".join(str(value or "").strip().split())


def awarding_office_parts(item: dict) -> tuple[str, str]:
    office = item.get("Awarding Office") or item.get("awarding_office")
    office_code = ""
    office_name = ""

    if isinstance(office, dict):
        office_code = clean_office_value(
            first_present(office, ["code", "id", "office_code", "awarding_office_code"])
        )
        office_name = clean_office_value(
            first_present(office, ["name", "office_name", "awarding_office_name"])
        )
    elif isinstance(office, str):
        office_name = clean_office_value(office)

    office_code = office_code or clean_office_value(
        first_present(item, ["Awarding Office Code", "awarding_office_code"])
    )
    office_name = office_name or clean_office_value(
        first_present(item, ["Awarding Office Name", "awarding_office_name"])
    )
    return office_code, office_name


def encode_contracting_office(office_code: str, office_name: str) -> str:
    return f"{office_code}{CONTRACTING_OFFICE_SEPARATOR}{office_name}"


def decode_contracting_office(contracting_office: str | None) -> tuple[str, str]:
    if not contracting_office or contracting_office == ALL_CONTRACTING_OFFICES:
        return "", ""
    if CONTRACTING_OFFICE_SEPARATOR not in contracting_office:
        return "", clean_office_value(contracting_office)
    office_code, office_name = contracting_office.split(CONTRACTING_OFFICE_SEPARATOR, 1)
    return clean_office_value(office_code), clean_office_value(office_name)


def format_contracting_office_option(contracting_office: str) -> str:
    if contracting_office == ALL_CONTRACTING_OFFICES:
        return contracting_office
    office_code, office_name = decode_contracting_office(contracting_office)
    if office_code and office_name:
        return f"{office_code} — {office_name}"
    return office_name or office_code or contracting_office


def format_funding_office_option(funding_office: str) -> str:
    if funding_office == ALL_FUNDING_OFFICES:
        return funding_office
    office_code, office_name = decode_option(funding_office)
    if office_code and office_name:
        return f"{office_code} — {office_name}"
    return office_name or office_code or funding_office


def transaction_contracting_office(item: dict) -> str:
    office_code, office_name = awarding_office_parts(item)
    if not office_name:
        return ""
    return encode_contracting_office(office_code, office_name)


def funding_office_parts(item: dict) -> tuple[str, str]:
    office = item.get("Funding Office") or item.get("funding_office")
    office_code = ""
    office_name = ""
    if isinstance(office, dict):
        office_code = clean_office_value(
            first_present(office, ["code", "id", "office_code", "funding_office_code"])
        )
        office_name = clean_office_value(
            first_present(office, ["name", "office_name", "funding_office_name"])
        )
    elif isinstance(office, str):
        office_name = clean_office_value(office)
    office_code = office_code or clean_office_value(
        first_present(item, ["Funding Office Code", "funding_office_code"])
    )
    office_name = office_name or clean_office_value(
        first_present(item, ["Funding Office Name", "funding_office_name"])
    )
    return office_code, office_name


def transaction_funding_office(item: dict) -> str:
    office_code, office_name = funding_office_parts(item)
    if not office_name:
        return ""
    return encode_option(office_code, office_name)


def contracting_office_matches(item: dict, contracting_office: str | None) -> bool:
    if not contracting_office or contracting_office == ALL_CONTRACTING_OFFICES:
        return True
    selected_code, selected_name = decode_contracting_office(contracting_office)
    item_code, item_name = awarding_office_parts(item)
    if selected_code:
        return item_code.lower() == selected_code.lower()
    if not selected_name:
        return True
    return selected_name.lower() in item_name.lower()


def funding_office_matches(item: dict, funding_office: str | None) -> bool:
    if not funding_office or funding_office == ALL_FUNDING_OFFICES:
        return True
    selected_code, selected_name = decode_option(funding_office)
    item_code, item_name = funding_office_parts(item)
    if selected_code:
        return item_code.lower() == selected_code.lower()
    return bool(selected_name) and item_name.lower() == selected_name.lower()


def extract_code_description(value) -> tuple[str, str]:
    if isinstance(value, dict):
        code = clean_office_value(
            first_present(value, ["code", "id", "naics", "psc", "type", "award_type_code"])
        )
        description = clean_office_value(
            first_present(value, ["description", "name", "label", "text"])
        )
        return code, description
    if isinstance(value, list) and value:
        return extract_code_description(value[0])
    text = clean_office_value(value)
    if " - " in text:
        code, description = text.split(" - ", 1)
        return code.strip(), description.strip()
    if " — " in text:
        code, description = text.split(" — ", 1)
        return code.strip(), description.strip()
    return text, ""


def transaction_naics_parts(item: dict) -> tuple[str, str]:
    code, description = extract_code_description(
        first_present(item, ["NAICS", "naics"])
    )
    code = code or clean_office_value(
        first_present(item, ["NAICS Code", "naics_code", "naicsCode", "naics_code_current"])
    )
    description = description or clean_office_value(
        first_present(
            item,
            [
                "NAICS Description",
                "naics_description",
                "naics_description_current",
                "NAICS Description Current",
                "naics_desc",
            ],
        )
    )
    return code, description


def transaction_psc_parts(item: dict) -> tuple[str, str]:
    code, description = extract_code_description(
        first_present(
            item,
            [
                "PSC",
                "psc",
            ],
        )
    )
    code = code or clean_office_value(
        first_present(
            item,
            [
                "PSC Code",
                "psc_code",
                "product_or_service_code",
                "product_or_service_code_current",
                "Product or Service Code",
            ],
        )
    )
    description = description or clean_office_value(
        first_present(
            item,
            [
                "PSC Description",
                "psc_description",
                "product_or_service_code_description",
                "product_or_service_code_description_current",
                "Product or Service Code Description",
            ],
        )
    )
    return code, description


def transaction_contract_type_parts(item: dict) -> tuple[str, str]:
    code = clean_office_value(
        first_present(
            item,
            [
                "Type of Contract Pricing Code",
                "type_of_contract_pricing_code",
                "contract_pricing_type_code",
            ],
        )
    )
    description = clean_office_value(
        first_present(
            item,
            [
                "Type of Contract Pricing",
                "type_of_contract_pricing",
                "contract_pricing_type",
            ],
        )
    )
    if not code and description in CONTRACT_TYPE_OPTIONS:
        code = description
        description = CONTRACT_TYPE_OPTIONS[code]
    if code and not description:
        description = CONTRACT_TYPE_OPTIONS.get(code, code)
    return code, description


def transaction_pop_location_parts(item: dict) -> tuple[str, str]:
    place = first_present(
        item,
        [
            "Primary Place of Performance",
            "primary_place_of_performance",
            "Place of Performance",
            "place_of_performance",
        ],
    )
    if isinstance(place, dict):
        country = clean_office_value(
            first_present(place, ["country_code", "country", "country_code_alpha3"])
        ).upper()
        state = clean_office_value(first_present(place, ["state_code", "state"])).upper()
        return country, state

    country = clean_office_value(
        first_present(
            item,
            [
                "Place of Performance Country Code",
                "place_of_performance_country_code",
                "pop_country_code",
                "country_code",
            ],
        )
    ).upper()
    state = clean_office_value(
        first_present(
            item,
            [
                "Place of Performance State Code",
                "place_of_performance_state_code",
                "pop_state",
            ],
        )
    ).upper()
    return country, state


def transaction_matches_pop_location(item: dict, location_code: str) -> bool:
    item_country, item_state = transaction_pop_location_parts(item)
    normalized = str(location_code or "").strip().upper()
    if is_us_pop_state_code(normalized):
        if item_state != normalized:
            return False
        return not item_country or item_country in ("USA", "US")
    return item_country == normalized


def transaction_matches_market_filters(item: dict, market_filters: dict | None) -> bool:
    market_filters = normalize_market_filters(market_filters)

    naics_code, _naics_description = decode_option(market_filters["naics_code"])
    if naics_code and naics_code != ALL_NAICS_CODES:
        item_naics, _item_naics_description = transaction_naics_parts(item)
        if not item_naics.startswith(naics_code):
            return False

    contract_type_code, contract_type_description = decode_option(market_filters["contract_type"])
    if contract_type_code and contract_type_code != ALL_CONTRACT_TYPES:
        item_contract_type, item_contract_description = transaction_contract_type_parts(item)
        if (item_contract_type or item_contract_description) and (
            item_contract_type.lower() != contract_type_code.lower()
            and item_contract_description.lower() != contract_type_description.lower()
        ):
            return False

    psc_code, _psc_description = decode_option(market_filters["psc_code"])
    if psc_code and psc_code != ALL_PRODUCT_SERVICE_CODES:
        item_psc, _item_psc_description = transaction_psc_parts(item)
        if not item_psc.startswith(psc_code):
            return False

    funding_office = market_filters["funding_office"]
    if funding_office != ALL_FUNDING_OFFICES and not funding_office_matches(item, funding_office):
        return False

    pop_code, _pop_description = decode_option(market_filters["pop_state"])
    if pop_code and pop_code not in (ALL_POP_LOCATIONS, LEGACY_ALL_POP_STATES):
        if not transaction_matches_pop_location(item, pop_code):
            return False

    return True


def parse_action_type_code(value) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        value = value.get("code") or value.get("id") or value.get("name")
    text = str(value).strip().upper()
    return text[:1] if text else ""


def classify_cancellation_description(description: str) -> str:
    text = (description or "").upper()
    if "DEFAULT" in text:
        return "Default"
    if "CONVENIENCE" in text:
        return "Convenience"
    if "CANCEL" in text or "TERMINATION" in text:
        return "Cancellation"
    return ""


def classify_negative_obligation_signal(description: str) -> str:
    text = " ".join((description or "").upper().split())
    if not text or text == "NO OFFICIAL TRANSACTION DESCRIPTION PROVIDED":
        return "Review Required"
    if any(term in text for term in EXPLICIT_TERMINATION_TERMS):
        return "Explicit Termination"
    if any(term in text for term in ADMINISTRATIVE_TERMS):
        return "Administrative / Correction"
    if any(term in text for term in DEOBLIGATION_TERMS):
        return "Possible De-obligation"
    return "Possible De-obligation"


def normalize_transaction_response(rows: list[dict]) -> pd.DataFrame:
    columns = [
        "Raw Row Order",
        "Contract Award Unique Key",
        "Award ID",
        "Contractor Name",
        "Subagency / Bureau",
        "Mod",
        "Transaction Number",
        "Obligation Amount",
        "Current Award Value",
        "Award Ceiling",
        "NAICS Code",
        "NAICS Description",
        "PSC Code",
        "PSC Description",
        "Action Date",
        "Action Code",
        "Action Type",
        "Contracting Office",
        "Contracting Office Name",
        "Funding Office",
        "Funding Office Name",
        "Description",
    ]
    normalized_rows = []
    for row_order, item in enumerate(rows):
        if not isinstance(item, dict):
            continue
        amount = transaction_amount(item)
        description = (
            first_present(
                item,
                [
                    "Transaction Description",
                    "transaction_description",
                    "Description",
                    "description",
                    "Award Description",
                    "award_description",
                ],
            )
            or "No official transaction description provided"
        )
        action_date = parse_action_date(
            first_present(
                item,
                ["Action Date", "action_date", "date", "period_start"],
            )
        )
        action_code = parse_action_type_code(
            first_present(
                item,
                ["Action Type", "ActionType", "action_type", "action_type_code", "action_type_code_desc"],
            )
        )
        naics_code, naics_description = transaction_naics_parts(item)
        psc_code, psc_description = transaction_psc_parts(item)
        normalized_rows.append(
            {
                "Raw Row Order": row_order,
                "Contract Award Unique Key": award_unique_key(item),
                "Award ID": clean_office_value(first_present(item, ["award_id_piid", "Award ID", "PIID"])),
                "Contractor Name": first_present(
                    item,
                    [
                        "Recipient Name",
                        "recipient_name",
                        "Recipient",
                        "recipient",
                        "Awardee Name",
                    ],
                )
                or "Unknown Contractor",
                "Subagency / Bureau": first_present(
                    item,
                    [
                        "Awarding Sub Agency",
                        "Awarding Subagency",
                        "Awarding Sub Agency Name",
                        "awarding_sub_agency",
                        "awarding_subagency_name",
                        "subagency_name",
                    ],
                )
                or "Unspecified Bureau",
                "Mod": first_present(
                    item,
                    [
                        "Mod",
                        "mod",
                        "Modification Number",
                        "modification_number",
                    ],
                )
                or "",
                "Transaction Number": clean_office_value(item.get("transaction_number")),
                "Obligation Amount": amount,
                "Current Award Value": parse_currency_amount(item.get("current_total_value_of_award")),
                "Award Ceiling": parse_currency_amount(item.get("potential_total_value_of_award")),
                "NAICS Code": clean_office_value(naics_code) or "Unspecified",
                "NAICS Description": clean_office_value(naics_description) or "Unspecified",
                "PSC Code": clean_office_value(psc_code) or "Unspecified",
                "PSC Description": clean_office_value(psc_description) or "Unspecified",
                "Action Date": action_date,
                "Action Code": action_code,
                "Action Type": classify_cancellation_description(description)
                or TERMINATION_ACTION_MAP.get(action_code, action_code or "Unspecified"),
                "Contracting Office": format_contracting_office_option(transaction_contracting_office(item))
                if transaction_contracting_office(item)
                else "Unspecified Office",
                "Contracting Office Name": clean_office_value(
                    first_present(item, ["awarding_office_name", "Awarding Office Name"])
                )
                or "Unspecified Office",
                "Funding Office": format_funding_office_option(transaction_funding_office(item))
                if transaction_funding_office(item)
                else "Unspecified Office",
                "Funding Office Name": clean_office_value(
                    first_present(item, ["funding_office_name", "Funding Office Name"])
                )
                or "Unspecified Office",
                "Description": description,
            }
        )

    return pd.DataFrame(normalized_rows, columns=columns)


def current_fiscal_year() -> int:
    today = date.today()
    return today.year + 1 if today.month >= 10 else today.year


def fiscal_year_label(fiscal_year: int) -> str:
    label = f"FY{int(fiscal_year)}"
    if int(fiscal_year) == current_fiscal_year():
        return f"{label} YTD"
    return label


def fiscal_year_compact_label(fiscal_year: int) -> str:
    return fiscal_year_label(fiscal_year)


def response_has_next(page_metadata: dict) -> bool:
    has_next = page_metadata.get("hasNext")
    if isinstance(has_next, str):
        return has_next.strip().lower() == "true"
    return bool(has_next)


def transaction_page_signature(rows: list[dict]) -> str:
    serialized = json.dumps(rows, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def bureau_filter_active(bureau_name: str | None) -> bool:
    return resolve_bureau_filter_name(bureau_name) is not None


def subagency_scope_kind(bureau_name: str | None) -> str:
    if bureau_is_ui_only_not_applicable(bureau_name):
        return "not_applicable"
    return "explicit" if bureau_filter_active(bureau_name) else "all_bureaus"


def payload_agency_filters(payload: dict | None) -> list[dict]:
    if not isinstance(payload, dict):
        return []
    filters = payload.get("filters") if isinstance(payload.get("filters"), dict) else {}
    return filters.get("agencies") or []


def explicit_subagency_used_top_tier(payload: dict | None, selected_bureau: str | None) -> bool:
    if subagency_scope_kind(selected_bureau) != "explicit":
        return False
    return any(
        isinstance(agency, dict)
        and agency.get("type") == "awarding"
        and agency.get("tier") == "toptier"
        for agency in payload_agency_filters(payload)
    )


def component_scope_debug(
    component_name: str,
    payload: dict | None,
    selected_bureau: str | None,
    top_tier_fallback_used: bool = False,
) -> dict:
    selected_label = canonical_bureau_name(selected_bureau)
    fallback_violation = explicit_subagency_used_top_tier(payload, selected_bureau) or (
        subagency_scope_kind(selected_bureau) == "explicit" and top_tier_fallback_used
    )
    return {
        "component": component_name,
        "selected_subagency_label": selected_label,
        "subagency_scope": subagency_scope_kind(selected_bureau),
        "top_tier_fallback_used": bool(top_tier_fallback_used),
        "final_agency_filter_sent": payload_agency_filters(payload),
        "fallback_violation": fallback_violation,
        "log_message": "ERROR: explicit subagency selected but component used top-tier fallback"
        if fallback_violation
        else "",
    }


def attach_component_scope_debug(
    payload: dict,
    component_name: str,
    selected_bureau: str | None,
    top_tier_fallback_used: bool = False,
) -> dict:
    if not isinstance(payload, dict):
        return payload
    payload.setdefault("component_scope_debug", {})
    payload["component_scope_debug"][component_name] = component_scope_debug(
        component_name,
        payload,
        selected_bureau,
        top_tier_fallback_used=top_tier_fallback_used,
    )
    if payload["component_scope_debug"][component_name]["fallback_violation"]:
        print("ERROR: explicit subagency selected but component used top-tier fallback")
    return payload


def dataframe_has_spend(df: pd.DataFrame, amount_column: str) -> bool:
    if df.empty or amount_column not in df.columns:
        return False
    numeric_amounts = pd.to_numeric(df[amount_column], errors="coerce").fillna(0)
    return bool(numeric_amounts.abs().sum() > 0)


@st.cache_data(ttl=2592000, show_spinner=False)
def fetch_transaction_page(
    agency_name: str,
    bureau_name: str | None,
    start_date: str,
    end_date: str,
    page: int,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> tuple[dict | None, str | None, dict]:
    payload = build_transaction_payload(
        agency_name,
        bureau_name,
        start_date,
        end_date,
        page=page,
        market_filters=market_filters,
    )
    data, error = post_usaspending("/api/v2/search/spending_by_transaction/", payload)
    if error:
        fallback_payload = build_transaction_payload(
            agency_name,
            bureau_name,
            start_date,
            end_date,
            page=page,
            include_office_fields=False,
            market_filters=market_filters,
        )
        fallback_data, fallback_error = post_usaspending(
            "/api/v2/search/spending_by_transaction/",
            fallback_payload,
        )
        if fallback_data:
            fallback_payload["office_fields_unavailable"] = True
            return fallback_data, None, fallback_payload
        return data, error or fallback_error, payload
    return data, error, payload


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_trends(
    agency_name: str,
    bureau_name: str | None,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> tuple[pd.DataFrame, dict, str, str | None]:
    payload = build_trends_payload(agency_name, bureau_name, market_filters=market_filters)
    data, error = post_usaspending("/api/v2/search/spending_over_time/", payload)
    if data:
        df = normalize_trend_response(data)
        if not df.empty:
            return df, payload, "Live USAspending.gov", None

    return pd.DataFrame(columns=["fiscal_year", "amount"]), payload, "USAspending.gov error", error or "No trend rows returned"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_trend_period_total(
    agency_name: str,
    bureau_name: str | None,
    start_date: str,
    end_date: str,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> tuple[float | None, dict, str | None]:
    payload = build_trends_payload(
        agency_name,
        bureau_name,
        time_period=[{"start_date": start_date, "end_date": end_date}],
        market_filters=market_filters,
    )
    data, error = post_usaspending("/api/v2/search/spending_over_time/", payload)
    if data:
        df = normalize_trend_response(data)
        if not df.empty:
            return float(df["amount"].sum()), payload, None
    return None, payload, error or "No trend rows returned"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_obligations_time_series(
    agency_name: str,
    bureau_name: str | None,
    selected_year: int,
    time_grain: str,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> tuple[pd.DataFrame, dict, str, str | None]:
    payload = build_obligations_time_series_payload(
        agency_name,
        bureau_name,
        selected_year,
        time_grain,
        market_filters=market_filters,
    )
    data, error = post_usaspending("/api/v2/search/spending_over_time/", payload)
    if data:
        df = normalize_obligation_time_series_response(data, time_grain)
        if not df.empty:
            return df, payload, "Live USAspending.gov", None

    return (
        pd.DataFrame(columns=["bucket_label", "amount", "transaction_count"]),
        payload,
        "USAspending.gov error",
        error or "No obligation time-series rows returned",
    )


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vendors(
    agency_name: str,
    bureau_name: str | None,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> tuple[pd.DataFrame, int, dict, str, str | None]:
    payload = build_vendor_payload(agency_name, bureau_name, market_filters)
    data, error = post_usaspending("/api/v2/search/spending_by_category/recipient/", payload)
    if data:
        df, contractor_count = normalize_vendor_response(data)
        return (
            df,
            contractor_count or (len(df["recipient"].unique()) if "recipient" in df.columns else 0),
            payload,
            "Live USAspending.gov",
            None,
        )

    return pd.DataFrame(columns=["recipient", "amount"]), 0, payload, "USAspending.gov error", error or "No vendor rows returned"


def fetch_transaction_pages(
    agency_name: str,
    bureau_name: str | None,
    start_date: str,
    end_date: str,
    progress_text=None,
    contracting_office: str | None = None,
    include_positive: bool = False,
    market_filters: dict | None = None,
) -> tuple[list[dict], dict, str | None, int, float]:
    master_rows = []
    payload_log = build_transaction_payload(
        agency_name,
        bureau_name,
        start_date,
        end_date,
        page=1,
        market_filters=market_filters,
    )
    if contracting_office and contracting_office != ALL_CONTRACTING_OFFICES:
        office_code, office_name = decode_contracting_office(contracting_office)
        payload_log["client_side_filter"] = {
            "contracting_office_code": office_code,
            "contracting_office_name": office_name,
        }
    market_filters = normalize_market_filters(market_filters)
    if market_filters["funding_office"] != ALL_FUNDING_OFFICES:
        funding_office_code, funding_office_name = decode_option(market_filters["funding_office"])
        payload_log.setdefault("client_side_filter", {})
        payload_log["client_side_filter"].update(
            {
                "funding_office_code": funding_office_code,
                "funding_office_name": funding_office_name,
            }
        )
    first_error = None
    total_records = 0
    total_obligation_magnitude = 0.0
    seen_page_signatures = set()

    page = 1
    has_next = True
    while has_next:
        if progress_text is not None:
            progress_text.info(
                f"{SYNC_ICON} Synchronizing Live Federal Registry... Compiled {total_records:,} transactions so far."
            )
        data, error, payload = fetch_transaction_page(
            agency_name,
            bureau_name,
            start_date,
            end_date,
            page=page,
            market_filters=market_filters,
        )
        if not data:
            first_error = first_error or error or f"Transaction page {page} returned no data"
            break

        page_rows = data.get("results") or []
        if not page_rows:
            break

        signature = transaction_page_signature(page_rows)
        if signature in seen_page_signatures:
            first_error = first_error or f"Duplicate transaction page detected at page {page}; synchronization stopped safely"
            break
        seen_page_signatures.add(signature)

        scoped_rows = [
            row
            for row in page_rows
            if contracting_office_matches(row, contracting_office)
            and transaction_matches_market_filters(row, market_filters)
        ]
        total_obligation_magnitude += sum(abs(transaction_amount(row)) for row in scoped_rows)
        master_rows.extend(
            row for row in scoped_rows if include_positive or transaction_amount(row) < 0
        )
        total_records += len(scoped_rows)
        if progress_text is not None:
            progress_text.info(
                f"{SYNC_ICON} Synchronizing Live Federal Registry... Compiled {total_records:,} transactions so far."
            )
        has_next = response_has_next(data.get("page_metadata") or {})
        page += 1
        if page > 100:
            break

    return master_rows, payload_log, first_error, total_records, total_obligation_magnitude


def fetch_transactions(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    contracting_office: str | None = None,
    include_positive: bool = False,
    market_filters: dict | None = None,
    progress_text=None,
) -> tuple[pd.DataFrame, dict, str, str | None]:
    start_date, end_date = fiscal_year_date_range(fiscal_year)
    if progress_text is None:
        progress_text = st.empty()

    try:
        master_rows, payload_log, first_error, _records_seen, _obligation_magnitude = fetch_transaction_pages(
            agency_name,
            bureau_name,
            start_date,
            end_date,
            progress_text,
            contracting_office=contracting_office,
            include_positive=include_positive,
            market_filters=market_filters,
        )
        transaction_df = normalize_transaction_response(master_rows)
        payload_log["derived_office_options"] = transaction_office_filter_options(master_rows)
    finally:
        progress_text.empty()

    source = "Live USAspending.gov" if first_error is None else "Partial USAspending.gov"
    return transaction_df, payload_log, source, first_error


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_transactions_cached(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    contracting_office: str | None = None,
    include_positive: bool = False,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> tuple[pd.DataFrame, dict, str, str | None]:
    start_date, end_date = fiscal_year_date_range(fiscal_year)
    master_rows, payload_log, first_error, _records_seen, _obligation_magnitude = fetch_transaction_pages(
        agency_name,
        bureau_name,
        start_date,
        end_date,
        progress_text=None,
        contracting_office=contracting_office,
        include_positive=include_positive,
        market_filters=market_filters,
    )
    transaction_df = normalize_transaction_response(master_rows)
    payload_log["derived_office_options"] = transaction_office_filter_options(master_rows)

    source = "Live USAspending.gov" if first_error is None else "Partial USAspending.gov"
    return transaction_df, payload_log, source, first_error


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_transaction_period_total(
    agency_name: str,
    bureau_name: str | None,
    start_date: str,
    end_date: str,
    contracting_office: str | None = None,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> tuple[float | None, dict, str | None]:
    rows, payload_log, first_error, records_seen, _obligation_magnitude = fetch_transaction_pages(
        agency_name,
        bureau_name,
        start_date,
        end_date,
        contracting_office=contracting_office,
        include_positive=True,
        market_filters=market_filters,
    )
    if records_seen == 0:
        return None, payload_log, first_error or "No transaction rows returned"
    transaction_df = normalize_transaction_response(rows)
    total = float(transaction_df["Obligation Amount"].sum()) if not transaction_df.empty else 0.0
    return total, payload_log, first_error


def read_uploaded_document(uploaded_file) -> str:
    data = uploaded_file.getvalue()
    suffix = Path(uploaded_file.name).suffix.lower()
    try:
        if suffix == ".pdf":
            reader = PdfReader(io.BytesIO(data))
            pieces = []
            for page in reader.pages:
                pieces.append(page.extract_text() or "")
                if len(" ".join(pieces)) >= 4_000:
                    break
            return " ".join(pieces)[:4_000]
        return data.decode("utf-8", errors="ignore")[:4_000]
    except Exception:
        return ""


def heuristic_agency_match(text: str) -> str | None:
    haystack = text.lower()
    for token, agency in AGENCY_ALIASES.items():
        if token in haystack:
            return agency
    for agency in agency_names_from_records(fetch_toptier_agencies()):
        if agency.lower() in haystack:
            return agency
    return None


def extract_agency_from_document(text: str) -> tuple[str | None, str]:
    api_key = get_openai_api_key()
    if not api_key:
        return heuristic_agency_match(text), "OpenAI key unavailable; local agency match used"

    try:
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": AGENCY_EXTRACTION_PROMPT},
                {"role": "user", "content": text[:4_000]},
            ],
        )
        content = response.choices[0].message.content or "null"
        parsed = json.loads(content)
        if parsed is None:
            return heuristic_agency_match(text), "OpenAI returned null; local agency match used"
        agency = parsed.get("agency") if isinstance(parsed, dict) else None
        if agency:
            return normalize_agency_name(agency), "OpenAI extraction"
        return heuristic_agency_match(text), "OpenAI returned no agency; local agency match used"
    except Exception:
        return heuristic_agency_match(text), "OpenAI extraction unavailable; local agency match used"


def get_bureau_options(toptier_code: str, fiscal_year: int) -> list[str]:
    return fetch_subagencies(toptier_code, fiscal_year)


def normalize_category_options(data: dict, default_option: str) -> list[str]:
    options = {}
    for item in data.get("results") or []:
        code = clean_office_value(first_present(item, ["code", "id"]))
        description = clean_office_value(first_present(item, ["name", "description"]))
        if code:
            options[code] = encode_option(code, description)
    return [default_option] + sorted(
        options.values(),
        key=lambda option: format_code_description_option(option).lower(),
    )


def normalize_pop_country_options(data: dict, default_option: str) -> list[str]:
    options = {}
    for item in data.get("results") or []:
        code = clean_office_value(first_present(item, ["code", "id"])).upper()
        description = clean_office_value(first_present(item, ["name", "description"]))
        if not code or code in POP_COUNTRY_OPTION_EXCLUSIONS or is_us_pop_state_code(code):
            continue
        options[code] = encode_option(code, description)
    return [default_option] + sorted(
        options.values(),
        key=lambda option: format_code_description_option(option).lower(),
    )


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_pop_country_filter_options(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    default_option: str,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
    request_timeout_sec: float = 18,
) -> list[str]:
    payload = build_category_options_payload(
        agency_name,
        bureau_name,
        fiscal_year,
        "country",
        market_filters=market_filters,
        limit=100,
    )
    data, error = post_usaspending(
        "/api/v2/search/spending_by_category/country/",
        payload,
        timeout_sec=request_timeout_sec,
    )
    if error or not data:
        return [default_option]
    return normalize_pop_country_options(data, default_option)


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_category_filter_options(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    category: str,
    default_option: str,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
    request_timeout_sec: float = 18,
) -> list[str]:
    payload = build_category_options_payload(
        agency_name,
        bureau_name,
        fiscal_year,
        category,
        market_filters=market_filters,
    )
    data, error = post_usaspending(
        f"/api/v2/search/spending_by_category/{category}/",
        payload,
        timeout_sec=request_timeout_sec,
    )
    if error or not data:
        return [default_option]
    return normalize_category_options(data, default_option)


def office_option_parts(option: str, office_kind: str) -> tuple[str, str]:
    if office_kind == "funding":
        return decode_option(option)
    return decode_contracting_office(option)


def office_option_dedupe_key(option: str, office_kind: str) -> str:
    code, name = office_option_parts(option, office_kind)
    return (code or name).lower()


def office_filter_option_set(rows: list[dict], office_kind: str, default_option: str) -> dict:
    options = {}
    stats = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        if office_kind == "funding":
            office_code, office_name = funding_office_parts(row)
            encoded = encode_option(office_code, office_name)
        else:
            office_code, office_name = awarding_office_parts(row)
            encoded = encode_contracting_office(office_code, office_name)
        if not office_name:
            continue
        dedupe_key = office_code or office_name.lower()
        options[dedupe_key] = encoded
        option_stats = stats.setdefault(
            dedupe_key,
            {
                "code": office_code,
                "name": office_name,
                "transaction_count": 0,
                "obligation_sum": 0.0,
            },
        )
        option_stats["transaction_count"] += 1
        option_stats["obligation_sum"] += transaction_amount(row)
    formatter = format_funding_office_option if office_kind == "funding" else format_contracting_office_option
    sorted_options = [default_option] + sorted(options.values(), key=lambda option: formatter(option).lower())
    encoded_stats = {
        options[dedupe_key]: {
            **option_stats,
            "obligation_sum": round(float(option_stats["obligation_sum"]), 2),
        }
        for dedupe_key, option_stats in stats.items()
        if option_stats["transaction_count"] > 0 and dedupe_key in options
    }
    return {"options": sorted_options, "stats": encoded_stats}


def normalize_office_filter_options(rows: list[dict], office_kind: str, default_option: str) -> list[str]:
    return office_filter_option_set(rows, office_kind, default_option)["options"]


def transaction_office_filter_options(rows: list[dict]) -> dict:
    contracting_set = office_filter_option_set(
        rows,
        "awarding",
        ALL_CONTRACTING_OFFICES,
    )
    funding_set = office_filter_option_set(
        rows,
        "funding",
        ALL_FUNDING_OFFICES,
    )
    return {
        "contracting_offices": contracting_set["options"],
        "contracting_office_stats": contracting_set["stats"],
        "funding_offices": funding_set["options"],
        "funding_office_stats": funding_set["stats"],
    }


def merge_office_option_sets(default_option: str, office_kind: str, *option_sets: dict) -> dict:
    merged = {}
    merged_stats = {}
    for option_set in option_sets:
        option_list = option_set.get("options", []) if isinstance(option_set, dict) else option_set
        option_stats = option_set.get("stats", {}) if isinstance(option_set, dict) else {}
        for option in option_list or []:
            if not option or option == default_option:
                continue
            dedupe_key = office_option_dedupe_key(option, office_kind)
            merged[dedupe_key] = option
            if option in option_stats:
                stats = merged_stats.setdefault(
                    dedupe_key,
                    {
                        "code": option_stats[option].get("code", ""),
                        "name": option_stats[option].get("name", ""),
                        "transaction_count": 0,
                        "obligation_sum": 0.0,
                    },
                )
                stats["transaction_count"] += int(option_stats[option].get("transaction_count") or 0)
                stats["obligation_sum"] += float(option_stats[option].get("obligation_sum") or 0)
    formatter = format_funding_office_option if office_kind == "funding" else format_contracting_office_option
    sorted_options = [default_option] + sorted(
        merged.values(),
        key=lambda option: formatter(option).lower(),
    )
    encoded_stats = {
        option: {
            **merged_stats.get(dedupe_key, {}),
            "obligation_sum": round(float(merged_stats.get(dedupe_key, {}).get("obligation_sum", 0)), 2),
        }
        for dedupe_key, option in merged.items()
        if int(merged_stats.get(dedupe_key, {}).get("transaction_count") or 0) > 0
    }
    return {"options": sorted_options, "stats": encoded_stats}


def format_office_option_with_stats(option: str, default_option: str, office_kind: str, stats: dict) -> str:
    if option == default_option:
        return default_option
    formatter = format_funding_office_option if office_kind == "funding" else format_contracting_office_option
    label = formatter(option)
    option_stats = stats.get(option, {})
    transaction_count = int(option_stats.get("transaction_count") or 0)
    obligation_sum = float(option_stats.get("obligation_sum") or 0)
    if abs(obligation_sum) >= 0.005:
        return f"{label} ({format_money(obligation_sum)})"
    if transaction_count > 0:
        suffix = "txn" if transaction_count == 1 else "txns"
        return f"{label} · {transaction_count:,} {suffix}"
    return label


def scoped_office_option_set(
    rows: list[dict],
    office_kind: str,
    default_option: str,
    contracting_office: str | None,
    market_filters: dict | None,
) -> dict:
    market_filters = normalize_market_filters(market_filters)
    scoped_rows = []
    for row in rows:
        if office_kind == "funding" and not contracting_office_matches(row, contracting_office):
            continue
        if office_kind != "funding" and not funding_office_matches(row, market_filters["funding_office"]):
            continue
        scoped_rows.append(row)
    return office_filter_option_set(scoped_rows, office_kind, default_option)


def download_status_finished(status_payload: dict) -> bool:
    return str(status_payload.get("status") or "").lower() == "finished"


def fetch_transaction_download_rows_with_diagnostics(
    payload: dict,
    *,
    request_timeout_sec: float = 18,
    max_elapsed_sec: float | None = None,
) -> tuple[list[dict], dict, str | None]:
    start_time = time.monotonic()
    diagnostic = {
        "endpoint": "/api/v2/download/transactions/",
        "request_body": payload,
        "requested_columns": payload.get("columns", []),
        "filters": payload.get("filters", {}),
        "date_range": (payload.get("filters", {}).get("time_period") or [{}])[0],
        "agency_filters": payload.get("filters", {}).get("agencies", []),
        "award_or_idv_flag": payload.get("filters", {}).get("award_or_idv_flag"),
        "award_type_codes": payload.get("filters", {}).get("award_type_codes"),
        "refinement_filters": {
            key: value
            for key, value in payload.get("filters", {}).items()
            if key not in {"agencies", "award_type_codes", "award_or_idv_flag", "time_period"}
        },
        "download_request_submitted": False,
        "status_poll_responses": [],
        "final_status": "",
        "final_download_url": "",
        "download_http_status": None,
        "download_file_size_bytes": 0,
        "csv_rows_parsed": 0,
        "first_3_parsed_rows": [],
        "exact_csv_headers_returned": [],
        "elapsed_seconds": 0.0,
        "row_limit_requested": payload.get("limit"),
        "only_100_rows_processed": payload.get("limit") == 100,
        "full_download_processed": payload.get("limit") in (None, 0),
        "total_expected_result_count": None,
        "failure_mode": None,
    }

    try:
        response = requests.post(
            f"{BASE_URL}/api/v2/download/transactions/",
            json=payload,
            headers=request_headers(),
            timeout=request_timeout_sec,
        )
        diagnostic["download_request_submitted"] = True
        diagnostic["submission_http_status"] = response.status_code
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError) as exc:
        diagnostic["failure_mode"] = "download_job_failed"
        diagnostic["failure_detail"] = str(exc)
        diagnostic["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
        return [], diagnostic, "download_job_failed"

    status_url = data.get("status_url")
    file_url = data.get("file_url")
    if status_url:
        for _attempt in range(30):
            if max_elapsed_sec is not None and time.monotonic() - start_time > max_elapsed_sec:
                diagnostic["failure_mode"] = "download_job_timed_out"
                diagnostic["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
                return [], diagnostic, "download_job_timed_out"
            try:
                status_response = requests.get(status_url, headers=request_headers(), timeout=request_timeout_sec)
                status_response.raise_for_status()
                status_payload = status_response.json()
            except (requests.RequestException, ValueError):
                diagnostic["failure_mode"] = "download_job_failed"
                diagnostic["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
                break
            status = str(status_payload.get("status") or "").lower()
            diagnostic["status_poll_responses"].append(status_payload)
            diagnostic["final_status"] = status
            diagnostic["total_expected_result_count"] = (
                status_payload.get("total_rows")
                or status_payload.get("total_records")
                or status_payload.get("record_count")
                or diagnostic["total_expected_result_count"]
            )
            if status == "failed":
                diagnostic["failure_mode"] = "download_job_failed"
                diagnostic["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
                return [], diagnostic, "download_job_failed"
            if download_status_finished(status_payload):
                file_url = status_payload.get("file_url") or file_url
                break
            time.sleep(0.75)

    if not file_url:
        diagnostic["failure_mode"] = diagnostic["failure_mode"] or "download_job_timed_out"
        diagnostic["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
        return [], diagnostic, diagnostic["failure_mode"]

    try:
        if max_elapsed_sec is not None and time.monotonic() - start_time > max_elapsed_sec:
            diagnostic["failure_mode"] = "download_job_timed_out"
            diagnostic["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
            return [], diagnostic, "download_job_timed_out"
        zip_response = requests.get(file_url, headers=request_headers(), timeout=request_timeout_sec)
        diagnostic["final_download_url"] = file_url
        diagnostic["download_http_status"] = zip_response.status_code
        diagnostic["download_file_size_bytes"] = len(zip_response.content or b"")
        zip_response.raise_for_status()
        if not zip_response.content:
            diagnostic["failure_mode"] = "download_file_empty"
            diagnostic["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
            return [], diagnostic, "download_file_empty"
        with zipfile.ZipFile(io.BytesIO(zip_response.content)) as archive:
            prime_files = [name for name in archive.namelist() if "PrimeTransactions" in name]
            rows = []
            for filename in prime_files:
                with archive.open(filename) as csv_file:
                    reader = csv.DictReader(io.TextIOWrapper(csv_file, encoding="utf-8-sig"))
                    if reader.fieldnames and not diagnostic["exact_csv_headers_returned"]:
                        diagnostic["exact_csv_headers_returned"] = list(reader.fieldnames)
                    rows.extend(dict(row) for row in reader)
    except (requests.RequestException, zipfile.BadZipFile, OSError, UnicodeDecodeError, csv.Error) as exc:
        diagnostic["failure_mode"] = "csv_parse_error"
        diagnostic["failure_detail"] = str(exc)
        diagnostic["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
        return [], diagnostic, "csv_parse_error"

    diagnostic["csv_rows_parsed"] = len(rows)
    diagnostic["first_3_parsed_rows"] = rows[:3]
    diagnostic["elapsed_seconds"] = round(time.monotonic() - start_time, 2)
    if not rows:
        if diagnostic["exact_csv_headers_returned"]:
            diagnostic["zero_result_state"] = True
            return [], diagnostic, None
        diagnostic["failure_mode"] = "download_file_empty"
        return [], diagnostic, "download_file_empty"
    return rows, diagnostic, None


def fetch_transaction_download_rows_from_payload(
    payload: dict,
    *,
    request_timeout_sec: float = 18,
    max_elapsed_sec: float | None = None,
) -> list[dict]:
    rows, _diagnostic, _failure_mode = fetch_transaction_download_rows_with_diagnostics(
        payload,
        request_timeout_sec=request_timeout_sec,
        max_elapsed_sec=max_elapsed_sec,
    )
    return rows


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_transaction_download_office_rows(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
    request_timeout_sec: float = 18,
    max_elapsed_sec: float | None = None,
) -> list[dict]:
    payload = build_transaction_download_office_payload(
        agency_name,
        bureau_name,
        fiscal_year,
        market_filters=market_filters,
    )
    return fetch_transaction_download_rows_from_payload(
        payload,
        request_timeout_sec=request_timeout_sec,
        max_elapsed_sec=max_elapsed_sec,
    )


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_award_scope_download_rows(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> list[dict]:
    payload = build_award_scope_download_payload(
        agency_name,
        bureau_name,
        fiscal_year,
        market_filters=market_filters,
    )
    return fetch_transaction_download_rows_from_payload(payload)


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_office_filtered_download_transactions(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    contracting_office: str | None = None,
    include_positive: bool = True,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> tuple[pd.DataFrame, dict, str, str | None]:
    payload = build_transaction_download_office_payload(
        agency_name,
        bureau_name,
        fiscal_year,
        market_filters=market_filters,
    )
    rows = fetch_transaction_download_office_rows(
        agency_name,
        bureau_name,
        fiscal_year,
        market_filters=market_filters,
        cache_version=cache_version,
    )
    market_filters = normalize_market_filters(market_filters)
    scoped_rows = [
        row
        for row in rows
        if contracting_office_matches(row, contracting_office)
        and funding_office_matches(row, market_filters["funding_office"])
    ]
    if not include_positive:
        scoped_rows = [row for row in scoped_rows if transaction_amount(row) < 0]
    payload["client_side_filter"] = {}
    if contracting_office and contracting_office != ALL_CONTRACTING_OFFICES:
        office_code, office_name = decode_contracting_office(contracting_office)
        payload["client_side_filter"].update(
            {
                "awarding_office_code": office_code,
                "awarding_office_name": office_name,
            }
        )
    if market_filters["funding_office"] != ALL_FUNDING_OFFICES:
        funding_code, funding_name = decode_option(market_filters["funding_office"])
        payload["client_side_filter"].update(
            {
                "funding_office_code": funding_code,
                "funding_office_name": funding_name,
            }
        )
    if not payload["client_side_filter"]:
        payload.pop("client_side_filter")
    payload["transaction_download_rows_returned"] = len(rows)
    payload["transaction_count_returned"] = len(scoped_rows)
    payload["obligation_sum_returned"] = round(sum(transaction_amount(row) for row in scoped_rows), 2)
    payload["transaction_lane_field_names_found"] = {
        "naics": transaction_row_field_names_found(
            scoped_rows,
            NAICS_CODE_ALIASES + NAICS_DESCRIPTION_ALIASES,
        ),
        "psc": transaction_row_field_names_found(
            scoped_rows,
            PSC_CODE_ALIASES + PSC_DESCRIPTION_ALIASES,
        ),
    }
    payload["derived_office_options"] = transaction_office_filter_options(rows)
    df = normalize_transaction_response(scoped_rows)
    return df, payload, "Live USAspending.gov transaction download", None


def fetch_award_scope_from_download(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    contracting_office: str | None = None,
    market_filters: dict | None = None,
    bypass_cache: bool = False,
) -> tuple[pd.DataFrame, dict, str | None]:
    payload = build_award_scope_download_payload(
        agency_name,
        bureau_name,
        fiscal_year,
        market_filters=market_filters,
    )
    if bypass_cache:
        raw_rows, download_diagnostic, download_failure_mode = fetch_transaction_download_rows_with_diagnostics(payload)
        cache_diagnostic = {"cache_bypassed": True, "cache_returned_empty_result": False}
    else:
        cached_rows = fetch_award_scope_download_rows(
            agency_name,
            bureau_name,
            fiscal_year,
            market_filters=market_filters,
        )
        raw_rows, download_diagnostic, download_failure_mode = fetch_transaction_download_rows_with_diagnostics(payload)
        cache_diagnostic = {
            "cache_bypassed": False,
            "cached_row_count": len(cached_rows),
            "fresh_row_count": len(raw_rows),
            "cache_returned_empty_result": len(cached_rows) == 0 and len(raw_rows) > 0,
        }
        if cached_rows and not raw_rows:
            raw_rows = cached_rows
    rows = normalize_award_scope_download_rows(raw_rows)
    market_filters = normalize_market_filters(market_filters)
    scoped_rows = [
        row
        for row in rows
        if contracting_office_matches(row, contracting_office)
        and funding_office_matches(row, market_filters["funding_office"])
    ]
    if contracting_office and contracting_office != ALL_CONTRACTING_OFFICES:
        office_code, office_name = decode_contracting_office(contracting_office)
        payload.setdefault("client_side_filter", {})
        payload["client_side_filter"].update(
            {
                "awarding_office_code": office_code,
                "awarding_office_name": office_name,
            }
        )
    if market_filters["funding_office"] != ALL_FUNDING_OFFICES:
        funding_code, funding_name = decode_option(market_filters["funding_office"])
        payload.setdefault("client_side_filter", {})
        payload["client_side_filter"].update(
            {
                "funding_office_code": funding_code,
                "funding_office_name": funding_name,
            }
        )
    award_df, award_debug = award_scope_dataframe(scoped_rows)
    award_totals = award_scope_totals(award_df)
    returned_columns = list(raw_rows[0].keys()) if raw_rows else []
    missing_required_columns = []
    if rows:
        missing_required_columns = [
            canonical_key
            for canonical_key in [
                "contract_award_unique_key",
                "potential_total_value_of_award",
                "current_total_value_of_award",
                "total_dollars_obligated",
                "total_outlayed_amount_for_overall_award",
                "action_date",
                "recipient_name",
            ]
            if non_empty_field_count(rows, canonical_key) == 0
        ]
    if cache_diagnostic["cache_returned_empty_result"]:
        failure_mode = "cache_returned_empty_result"
    elif download_failure_mode:
        failure_mode = download_failure_mode
    elif missing_required_columns:
        failure_mode = "missing_required_columns"
    elif scoped_rows and award_df.empty:
        failure_mode = "zero_unique_award_keys_after_dedupe"
    else:
        failure_mode = None
    award_scope_request_body = {
        "filters": payload.get("filters"),
        "columns": payload.get("columns"),
        "file_format": payload.get("file_format"),
        "limit": payload.get("limit"),
    }
    payload["award_scope_debug"] = {
        **award_debug,
        "failure_mode": failure_mode,
        "download_lifecycle": download_diagnostic,
        "cache_diagnostic": cache_diagnostic,
        "award_scope_transaction_download_payload": {
            "endpoint": "/api/v2/download/transactions/",
            "request_body": award_scope_request_body,
            "filters": payload.get("filters"),
            "columns": payload.get("columns"),
            "file_format": payload.get("file_format"),
            "limit": payload.get("limit"),
        },
        "main_kpi_equivalent_filters": dashboard_base_filters(
            agency_name,
            bureau_name,
            fiscal_year=fiscal_year,
            market_filters=market_filters,
        ),
        "payload_matches_main_kpi_filters": payload.get("filters")
        == dashboard_base_filters(
            agency_name,
            bureau_name,
            fiscal_year=fiscal_year,
            market_filters=market_filters_without_offices(market_filters),
        ),
        "not_applicable_present_in_payload": "Not applicable" in json.dumps(award_scope_request_body, default=str),
        "raw_transaction_rows_returned": len(raw_rows),
        "first_3_raw_rows": raw_rows[:3],
        "exact_column_names_returned": returned_columns,
        "missing_required_columns": missing_required_columns,
        "count_non_null_contract_award_unique_key": non_empty_field_count(rows, "contract_award_unique_key"),
        "count_non_null_potential_total_value_of_award": non_empty_field_count(rows, "potential_total_value_of_award"),
        "count_non_null_current_total_value_of_award": non_empty_field_count(rows, "current_total_value_of_award"),
        "unique_awards_deduped": int(len(award_df)),
        "total_active_award_ceiling": round(award_totals["active_award_ceiling"], 2),
        "total_current_award_value": round(award_totals["current_award_value"], 2),
        "total_remaining_ceiling": round(award_totals["remaining_ceiling"], 2),
    }
    payload["award_scope_rows_returned"] = len(scoped_rows)
    payload["unique_awards_returned"] = int(len(award_df))
    if failure_mode:
        return award_df, payload, failure_mode
    return award_df, payload, None


def metric_card(
    label: str,
    value: str,
    subtext: str,
    accent: str,
    value_color: str | None = None,
    helper_text: str | None = None,
) -> None:
    metric_value_color = value_color or "var(--text)"
    helper_markup = ""
    title_attr = ""
    if helper_text:
        safe_helper = html.escape(helper_text)
        helper_markup = f'<div class="metric-helper">{safe_helper}</div>'
        title_attr = f' title="{safe_helper}"'
    st.markdown(
        f"""
        <div class="metric-card"{title_attr} style="--accent: {accent}; --metric-value-color: {metric_value_color};">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-sub">{html.escape(subtext)}</div>
            {helper_markup}
        </div>
        """,
        unsafe_allow_html=True,
    )


def source_chip(label: str) -> None:
    st.markdown(
        f"""
        <div class="source-chip">
            <span class="source-dot"></span>
            <span>{html.escape(label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def loading_detail_text(fiscal_year: int, transaction_count: int | None = None, unique_award_count: int | None = None) -> str:
    fy_label = fiscal_year_label(int(fiscal_year))
    if transaction_count is not None and unique_award_count is not None:
        return (
            f"Processing {format_count(transaction_count)} {fy_label} contract transactions "
            f"and {format_count(unique_award_count)} unique awards..."
        )
    if transaction_count is not None:
        return f"Processing {format_count(transaction_count)} {fy_label} contract transactions..."
    if unique_award_count is not None:
        return f"Processing {fy_label} contract transactions and {format_count(unique_award_count)} unique awards..."
    return f"Processing {fy_label} contract transactions..."


def render_analysis_loading_card(
    target,
    fiscal_year: int,
    transaction_count: int | None = None,
    unique_award_count: int | None = None,
) -> None:
    fy_label = fiscal_year_label(int(fiscal_year))
    body = (
        f"We’re processing USAspending contract data for {fy_label}: obligations, award ceilings, "
        "contractor rankings, and trends. This can take a minute for large scopes."
    )
    detail = loading_detail_text(int(fiscal_year), transaction_count, unique_award_count)
    stage_markup = "\n".join(
        f'<span style="animation-delay: {index * 2}s;">{html.escape(message)}</span>'
        for index, message in enumerate(ANALYSIS_LOADING_MESSAGES)
    )
    count_markup = ""
    if transaction_count is not None or unique_award_count is not None:
        count_tiles = []
        if transaction_count is not None:
            count_tiles.append(
                f"""
                <div class="analysis-loading-count">
                    <div class="analysis-loading-count-value">{html.escape(format_count(transaction_count))}</div>
                    <div class="analysis-loading-count-label">Transactions processed</div>
                </div>
                """
            )
        if unique_award_count is not None:
            count_tiles.append(
                f"""
                <div class="analysis-loading-count">
                    <div class="analysis-loading-count-value">{html.escape(format_count(unique_award_count))}</div>
                    <div class="analysis-loading-count-label">Unique awards deduped</div>
                </div>
                """
            )
        count_markup = f'<div class="analysis-loading-counts">{"".join(count_tiles)}</div>'
    target.markdown(
        (
            '<section class="analysis-loading-card" role="status" aria-live="polite">'
            f'<p class="analysis-loading-body">{html.escape(body)}</p>'
            '<div class="analysis-loading-detail">'
            '<span class="analysis-loading-spinner" aria-hidden="true"></span>'
            f"<span>{html.escape(detail)}</span>"
            "</div>"
            f'<div class="analysis-loading-stage">{stage_markup}</div>'
            f"{count_markup}"
            '<div class="analysis-loading-footer">Accuracy first. No estimates - only reconciled USAspending data.</div>'
            "</section>"
        ),
        unsafe_allow_html=True,
    )


def render_extraction_loading_card(target, progress_detail: dict) -> None:
    label = str(progress_detail.get("label") or "Extracting solicitation package…")
    pct = float(progress_detail.get("pct") or 0.0)
    pct_display = max(0, min(100, int(round(pct * 100))))
    doc_index = progress_detail.get("docIndex")
    doc_total = progress_detail.get("docTotal")
    filename = str(progress_detail.get("filename") or "").strip()
    phase = str(progress_detail.get("phase") or "").strip()

    if doc_index and doc_total:
        doc_counter = f"{doc_index} / {doc_total}"
    else:
        doc_counter = "—"
    current_doc = filename or "—"
    phase_label = {
        "reading": "Analyzing document",
        "ocr": "OCR extraction",
        "corpus": "Package assembly",
        "gpt": "GPT extraction",
        "validate": "Evidence validation",
        "resolve": "Filter mapping",
        "cache": "Cache reuse",
    }.get(phase, "Current step")

    target.markdown(
        f"""
        <section class="analysis-loading-card extraction-loading-card" role="status" aria-live="polite">
            <p class="analysis-loading-body">Extracting market scope from the uploaded solicitation package. Large packages may take 30–60 seconds on the first uncached run.</p>
            <div class="analysis-loading-detail">
                <span class="analysis-loading-spinner" aria-hidden="true"></span>
                <span>{html.escape(label)}</span>
            </div>
            <div class="analysis-loading-counts">
                <div class="analysis-loading-count">
                    <div class="analysis-loading-count-value">{pct_display}%</div>
                    <div class="analysis-loading-count-label">Progress</div>
                </div>
                <div class="analysis-loading-count">
                    <div class="analysis-loading-count-value">{html.escape(doc_counter)}</div>
                    <div class="analysis-loading-count-label">Documents processed</div>
                </div>
                <div class="analysis-loading-count">
                    <div class="analysis-loading-count-value extraction-doc-name">{html.escape(_solicitation_truncate(current_doc, 28) or "—")}</div>
                    <div class="analysis-loading-count-label">{html.escape(phase_label)}</div>
                </div>
            </div>
            <div class="analysis-loading-footer">Working through each file before the single GPT market-scope request.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def show_analysis_success(transaction_count: int | None) -> None:
    if not hasattr(st, "toast"):
        return
    if transaction_count is None:
        st.toast("Analysis complete.")
    else:
        st.toast(f"Analysis complete. {format_count(transaction_count)} transactions processed.")


def render_market_scope_summary(
    active_agency: str,
    selected_bureau: str | None,
    selected_year: int,
    selected_market_filters: dict,
    selected_contracting_office: str,
) -> None:
    labels = market_filter_labels(selected_market_filters, selected_contracting_office)
    bureau_label = canonical_bureau_name(selected_bureau)
    scope_line = f"{active_agency} / {bureau_label} / {fiscal_year_label(int(selected_year))}"
    refinements_line = " · ".join(
        [
            labels["naics"],
            labels["contract_type"],
            labels["psc"],
            labels["set_aside"],
            labels["contracting_office"],
            labels["funding_office"],
            labels["place_of_performance"],
        ]
    )
    base_chips = [
        f"Fiscal Year: {fiscal_year_label(int(selected_year))}",
        f"Agency: {active_agency}",
        f"Subagency/Bureau: {bureau_label}",
    ]
    chip_html = "".join(
        f'<span class="filter-chip">{html.escape(chip)}</span>' for chip in base_chips
    )
    st.markdown(
        f"""
        <section class="market-scope">
            <div class="market-scope-title">Market Scope</div>
            <div class="market-scope-line">{html.escape(scope_line)}</div>
            <div class="market-scope-line">Refinements: {html.escape(refinements_line)}</div>
            <div class="filter-chip-row">{chip_html}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    applied_chips = _applied_filter_chips(selected_bureau, selected_market_filters, selected_contracting_office)
    if applied_chips:
        st.markdown('<div class="applied-filter-heading">Applied filters</div>', unsafe_allow_html=True)
        st.markdown('<div class="applied-filter-chip-row">', unsafe_allow_html=True)
        for row_start in range(0, len(applied_chips), 3):
            row = applied_chips[row_start : row_start + 3]
            columns = st.columns(len(row))
            for column, chip in zip(columns, row):
                with column:
                    if st.button(
                        f"{chip['label']} ×",
                        key=f"remove-applied-filter-{chip['key']}",
                        help=f"Remove {chip['label']}",
                    ):
                        remove_applied_filter_and_rerun(
                            chip["key"],
                            active_agency,
                            selected_bureau,
                            int(selected_year),
                            selected_market_filters,
                            selected_contracting_office,
                        )
        st.markdown("</div>", unsafe_allow_html=True)
    active_refinements = []
    defaults = []

    if active_refinements:
        st.caption("Active refinements")
        active_chip_html = "".join(
            f'<span class="filter-chip active">{html.escape(label)} ×</span>'
            for _key, label in active_refinements
        )
        st.markdown(
            f'<div class="filter-chip-row">{active_chip_html}</div>',
            unsafe_allow_html=True,
        )
        if st.button("Clear Refinements", key="clear-refinements-summary"):
            clear_active_refinements()
            st.rerun()
        if defaults:
            st.markdown(
                f'<div class="other-filters">Other filters: {html.escape(" · ".join(defaults))}</div>',
                unsafe_allow_html=True,
            )
    elif defaults:
        st.markdown(
            f'<div class="other-filters">Other filters: {html.escape(" · ".join(defaults))}</div>',
            unsafe_allow_html=True,
        )


def sentence_list(items: list[str]) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if len(cleaned) == 2:
        return f"{cleaned[0]} and {cleaned[1]}"
    return f"{', '.join(cleaned[:-1])}, and {cleaned[-1]}"


SUMMARY_TITLE_CASE_ACRONYMS = {
    "AI": "AI",
    "API": "API",
    "CIO-SP": "CIO-SP",
    "IDIQ": "IDIQ",
    "IT": "IT",
    "NAICS": "NAICS",
    "O&M": "O&M",
    "PSC": "PSC",
    "R&D": "R&D",
    "SAAS": "SaaS",
    "USA": "USA",
}
SUMMARY_TITLE_CASE_SMALL_WORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "but",
    "by",
    "for",
    "from",
    "in",
    "nor",
    "of",
    "on",
    "or",
    "per",
    "the",
    "to",
    "with",
}


def summary_title_case_token(token: str, word_index: int, word_count: int) -> str:
    if not token:
        return token
    if "/" in token:
        return "/".join(
            summary_title_case_token(part, word_index, word_count)
            for part in token.split("/")
        )
    if "-" in token and token != "-":
        return "-".join(
            summary_title_case_token(part, word_index, word_count)
            for part in token.split("-")
        )
    prefix_match = re.match(r"^[^A-Za-z0-9&]+", token)
    suffix_match = re.search(r"[^A-Za-z0-9&]+$", token)
    prefix = prefix_match.group(0) if prefix_match else ""
    suffix = suffix_match.group(0) if suffix_match else ""
    core_start = len(prefix)
    core_end = len(token) - len(suffix) if suffix else len(token)
    core = token[core_start:core_end]
    if not core:
        return token
    normalized = core.upper()
    if normalized in SUMMARY_TITLE_CASE_ACRONYMS:
        cased = SUMMARY_TITLE_CASE_ACRONYMS[normalized]
    elif 0 < word_index < word_count - 1 and core in SUMMARY_TITLE_CASE_SMALL_WORDS:
        cased = core
    else:
        cased = core.capitalize()
    return f"{prefix}{cased}{suffix}"


def summary_title_case_description(description: str) -> str:
    text = clean_office_value(description)
    if not text:
        return ""
    words = re.split(r"(\s+)", text.lower())
    cased_words = []
    word_index = 0
    word_count = sum(1 for token in words if token.strip())
    for token in words:
        if not token.strip():
            cased_words.append(token)
            continue
        cased_words.append(summary_title_case_token(token, word_index, word_count))
        word_index += 1
    return "".join(cased_words)


def market_summary_active_refinements(
    selected_market_filters: dict,
    selected_contracting_office: str,
) -> list[str]:
    labels = market_filter_labels(selected_market_filters, selected_contracting_office)
    market_filters = normalize_market_filters(selected_market_filters)
    refinements = []
    refinement_config = [
        ("naics_code", ALL_NAICS_CODES, labels["naics"]),
        ("contract_type", ALL_CONTRACT_TYPES, labels["contract_type"]),
        ("psc_code", ALL_PRODUCT_SERVICE_CODES, labels["psc"]),
        ("set_aside_type", ALL_SET_ASIDE_TYPES, labels["set_aside"]),
        ("contracting_office", ALL_CONTRACTING_OFFICES, labels["contracting_office"]),
        ("funding_office", ALL_FUNDING_OFFICES, labels["funding_office"]),
        ("pop_state", ALL_POP_LOCATIONS, labels["place_of_performance"]),
    ]
    for key, default_value, label in refinement_config:
        value = selected_contracting_office if key == "contracting_office" else market_filters[key]
        if value != default_value:
            refinements.append(label)
    return refinements


def lane_summary_record(df: pd.DataFrame, code_column: str) -> dict:
    if df.empty:
        return {}
    top_row = df.iloc[0]
    code = clean_office_value(top_row.get(code_column)) or "Unspecified"
    description = clean_office_value(top_row.get("Full Description")) or "Unspecified"
    amount = float(pd.to_numeric(pd.Series([top_row.get("Obligated")]), errors="coerce").fillna(0).iloc[0])
    percentage = float(top_row.get("% of Scope") or 0.0)
    if code == "Unspecified" and description == "Unspecified":
        return {}
    return {
        "code": code,
        "description": description,
        "summary_description": summary_title_case_description(description),
        "amount": amount,
        "percentage": percentage,
    }


def market_summary_top_awards(award_df: pd.DataFrame, limit: int = 3) -> list[dict]:
    if award_df.empty:
        return []
    sorted_df = sorted_award_drilldown_dataframe(award_df, "Obligations This Period")
    records = []
    for row in sorted_df.head(limit).to_dict("records"):
        records.append(
            {
                "award_id": clean_office_value(row.get("Award ID")) or "Unavailable",
                "contractor": clean_office_value(row.get("Contractor")) or "Unknown Contractor",
                "obligations": float(row.get("Obligations This Period") or 0.0),
                "current_award_value": float(row.get("Current Award Value") or 0.0),
                "award_ceiling": float(row.get("Award Ceiling") or 0.0),
            }
        )
    return records


def market_summary_negative_obligations(negative_transaction_df: pd.DataFrame, limit: int = 3) -> list[dict]:
    if negative_transaction_df.empty:
        return []
    grouped = (
        negative_transaction_df.groupby("Contractor Name", as_index=False)["Obligation Amount"]
        .sum()
        .sort_values("Obligation Amount", ascending=True)
        .head(limit)
    )
    return [
        {
            "contractor": clean_office_value(row.get("Contractor Name")) or "Unknown Contractor",
            "amount": float(row.get("Obligation Amount") or 0.0),
        }
        for row in grouped.to_dict("records")
    ]


def build_market_summary(
    active_agency: str,
    selected_bureau: str | None,
    selected_year: int,
    selected_contracting_office: str,
    selected_market_filters: dict,
    current_total: float,
    vendor_df: pd.DataFrame,
    award_totals: dict,
    award_scope_error: bool,
    market_concentration: dict,
    market_concentration_debug: dict,
    transaction_df: pd.DataFrame,
    award_drilldown_df: pd.DataFrame,
    negative_transaction_df: pd.DataFrame,
    negative_obligation_total: float,
) -> tuple[str, dict]:
    selected_bureau = canonical_bureau_name(selected_bureau)
    selected_market_filters = normalize_market_filters(selected_market_filters)
    fiscal_label = fiscal_year_label(int(selected_year))
    scope_line = f"{active_agency} / {selected_bureau} / {fiscal_label}"
    active_refinements = market_summary_active_refinements(selected_market_filters, selected_contracting_office)
    zero_result = abs(float(current_total or 0.0)) < 0.005
    grouped_contractor_count = int(market_concentration_debug.get("grouped_contractor_count") or 0)
    contractors_shown = int(len(vendor_df)) if not vendor_df.empty else 0
    unique_awards = int(award_totals.get("award_count") or 0)
    top_share = market_concentration_debug.get("concentration_percentage")
    top_share_value = float(top_share) if top_share is not None else None
    top_contractors = [
        {
            "recipient": clean_office_value(row.get("recipient")) or "Unknown Contractor",
            "amount": float(row.get("amount") or 0.0),
        }
        for row in vendor_df.head(5).to_dict("records")
    ] if not vendor_df.empty else []

    naics_df, naics_debug = market_lane_mix_dataframe(transaction_df, "naics", current_total, selected_market_filters)
    psc_df, psc_debug = market_lane_mix_dataframe(transaction_df, "psc", current_total, selected_market_filters)
    top_naics = lane_summary_record(naics_df, "NAICS")
    top_psc = lane_summary_record(psc_df, "PSC")
    selected_naics_code, selected_naics_label = decode_option(selected_market_filters["naics_code"])
    selected_psc_code, selected_psc_label = decode_option(selected_market_filters["psc_code"])
    selected_naics_active = bool(selected_naics_code and selected_naics_code != ALL_NAICS_CODES)
    selected_psc_active = bool(selected_psc_code and selected_psc_code != ALL_PRODUCT_SERVICE_CODES)
    top_awards = market_summary_top_awards(award_drilldown_df)
    negative_records = market_summary_negative_obligations(negative_transaction_df)

    source_note = "Source note: Summary reflects the selected contract-focused USAspending scope."
    paragraphs = [f"Market Summary: {scope_line}"]
    if active_refinements:
        paragraphs.append(f"Filters: {'; '.join(active_refinements)}")

    if zero_result:
        paragraphs.append(
            f"Contract Market Snapshot: No {fiscal_label} prime contract obligations were found for this selected scope. "
            "USAspending may still show grants or assistance awards outside this contract-focused view."
        )
        paragraphs.append(
            "Competitive Concentration: No contractor concentration was calculated because no prime contract obligations were found."
        )
        paragraphs.append("Market Lane: NAICS/PSC mix was not available from returned transaction rows for this scope.")
        paragraphs.append("Award Activity: No top contract awards were identified in this selected FY scope.")
        paragraphs.append("Negative Obligation / De-obligation Signal: No negative obligations were identified in this selected FY scope.")
        paragraphs.append(
            "Capture Takeaway: Broaden the scope to All Bureaus or remove refinements to identify adjacent contract opportunities."
        )
    else:
        contractor_phrase = f"The dashboard displays the top {format_count(contractors_shown)} contractors"
        if grouped_contractor_count:
            contractor_phrase += (
                f", with {format_count(grouped_contractor_count)} total grouped contractors identified in this scope"
            )
        contractor_phrase += "."
        if award_scope_error:
            award_scope_sentence = "Award ceiling, current award value, and remaining ceiling were unavailable for this scope."
            source_note = (
                "Source note: Summary is generated from USAspending prime contract transaction data for the selected scope. "
                "Award value fields were unavailable for this scope."
            )
        else:
            award_scope_sentence = (
                f"Active award ceiling totals {format_money(award_totals.get('active_award_ceiling'))}, "
                f"with {format_money(award_totals.get('current_award_value'))} in current award value and "
                f"{format_money(award_totals.get('remaining_ceiling'))} in remaining ceiling."
            )
            source_note = (
                "Source note: Summary is generated from USAspending prime contract transaction and award value data "
                "for the selected scope."
            )
        paragraphs.append(
            f"Contract Market Snapshot: {active_agency} has obligated {format_money(current_total)} in {fiscal_label} prime contract awards "
            f"across {format_count(unique_awards)} unique awards. {contractor_phrase} {award_scope_sentence}"
        )

        if grouped_contractor_count == 1 and top_contractors:
            paragraphs.append(
                f"Competitive Concentration: This is a single-vendor filtered scope: {top_contractors[0]['recipient']} accounts for 100.0% of positive obligations."
            )
        elif top_share_value is not None and top_share_value >= 70:
            paragraphs.append(
                f"Competitive Concentration: This is a highly concentrated market: the top 5 contractors captured {top_share_value:.1f}% of positive obligations in this scope."
            )
        elif top_share_value is not None and top_share_value >= 40:
            paragraphs.append(
                f"Competitive Concentration: This market has moderate concentration: the top 5 contractors captured {top_share_value:.1f}% of positive obligations in this scope."
            )
        elif top_share_value is not None:
            paragraphs.append(
                f"Competitive Concentration: This market appears fragmented: the top 5 contractors captured only {top_share_value:.1f}% of positive obligations in this scope."
            )

        if top_contractors:
            contractor_items = [
                f"{record['recipient']} ({format_money(record['amount'])})"
                for record in top_contractors
            ]
            paragraphs.append(
                f"Top Contractors: Top contractors by {fiscal_label} obligations include {sentence_list(contractor_items)}."
            )

        lane_sentences = []
        if selected_naics_active:
            selected_label = summary_title_case_description(selected_naics_label) or selected_naics_code
            lane_sentences.append(f"This view is narrowed to NAICS {selected_naics_code} — {selected_label}.")
        elif top_naics:
            lane_sentences.append(
                f"The leading NAICS lane is {top_naics['code']} — {top_naics['summary_description']}, "
                f"representing {format_money(top_naics['amount'])} and {top_naics['percentage']:.1f}% of the selected scope."
            )
        if selected_psc_active:
            selected_label = summary_title_case_description(selected_psc_label) or selected_psc_code
            lane_sentences.append(f"This view is narrowed to PSC {selected_psc_code} — {selected_label}.")
        elif top_psc:
            lane_sentences.append(
                f"The leading PSC is {top_psc['code']} — {top_psc['summary_description']}, "
                f"representing {format_money(top_psc['amount'])} and {top_psc['percentage']:.1f}% of the selected scope."
            )
        if lane_sentences:
            paragraphs.append(f"Market Lane: {' '.join(lane_sentences)}")
        else:
            paragraphs.append("Market Lane: NAICS/PSC mix was not available from returned transaction rows for this scope.")

        if top_awards:
            award_items = [
                (
                    f"{record['award_id']} to {record['contractor']} "
                    f"({format_money(record['obligations'])} obligated this period; "
                    f"{format_money(record['current_award_value'])} current value; "
                    f"{format_money(record['award_ceiling'])} ceiling)"
                )
                for record in top_awards
            ]
            paragraphs.append(
                f"Award Activity: The largest awards driving this scope include {sentence_list(award_items)}. "
                "These awards should be reviewed first when assessing incumbent positioning, contract vehicles, and recompete context."
            )
        else:
            paragraphs.append("Award Activity: No top contract awards were identified in this selected FY scope.")

        if negative_obligation_total < -0.005:
            negative_items = [
                f"{record['contractor']} ({format_money(record['amount'])})"
                for record in negative_records
            ]
            paragraphs.append(
                f"Negative Obligation / De-obligation Signal: The scope also includes {format_money(negative_obligation_total)} "
                "in negative obligations/de-obligations"
                + (f", led by {sentence_list(negative_items)}. " if negative_items else ". ")
                + "Negative obligations may reflect de-obligations, funding adjustments, closeouts, or scope changes and should be reviewed at the award level before drawing conclusions."
            )
        else:
            paragraphs.append(
                "Negative Obligation / De-obligation Signal: No negative obligations were identified in this selected FY scope."
            )

        takeaway_sentences = []
        if grouped_contractor_count == 1:
            takeaway_sentences.append(
                "Capture Takeaway: This is a narrow single-vendor filtered scope. Review the award details before drawing broader market conclusions."
            )
        elif top_share_value is not None and top_share_value >= 70:
            takeaway_sentences.append(
                "Capture Takeaway: This appears to be a concentrated market with strong incumbent presence. New entrants should study the top contractors, award vehicles, and major active awards before pursuing this lane."
            )
        elif top_share_value is not None and top_share_value >= 40:
            takeaway_sentences.append(
                "Capture Takeaway: This market has moderate concentration, suggesting meaningful incumbent presence but room for competition."
            )
        else:
            takeaway_sentences.append(
                "Capture Takeaway: This market appears fragmented, which may create more room for challengers or niche vendors."
            )
        if float(award_totals.get("remaining_ceiling") or 0.0) > 0 and not award_scope_error:
            takeaway_sentences.append(
                "Remaining ceiling should be interpreted as active award capacity, not reported future obligations."
            )
        paragraphs.append(" ".join(takeaway_sentences))

    paragraphs.append(source_note)

    concentration_classification = "zero-result"
    if not zero_result:
        if grouped_contractor_count == 1:
            concentration_classification = "single-vendor"
        elif top_share_value is not None and top_share_value >= 70:
            concentration_classification = "highly concentrated"
        elif top_share_value is not None and top_share_value >= 40:
            concentration_classification = "moderately concentrated"
        else:
            concentration_classification = "fragmented"
    summary_debug = {
        "active_scope_used": {
            "agency": active_agency,
            "bureau": selected_bureau,
            "fiscal_year": fiscal_label,
            "contracting_office": selected_contracting_office,
            "market_filters": selected_market_filters,
            "active_refinements": active_refinements,
        },
        "kpi_values_used": {
            "fy_obligations": round(float(current_total or 0.0), 2),
            "contractors_shown": contractors_shown,
            "grouped_contractors": grouped_contractor_count,
            "unique_awards": unique_awards,
            "active_award_ceiling": round(float(award_totals.get("active_award_ceiling") or 0.0), 2),
            "current_award_value": round(float(award_totals.get("current_award_value") or 0.0), 2),
            "remaining_ceiling": round(float(award_totals.get("remaining_ceiling") or 0.0), 2),
            "award_scope_error": bool(award_scope_error),
        },
        "contractor_list_used": top_contractors,
        "naics_psc_values_used": {
            "top_naics": top_naics,
            "top_psc": top_psc,
            "naics_debug": naics_debug,
            "psc_debug": psc_debug,
        },
        "top_awards_used": top_awards,
        "negative_obligation_values_used": {
            "total": round(float(negative_obligation_total or 0.0), 2),
            "top_negative_contractors": negative_records,
        },
        "concentration_classification": concentration_classification,
        "market_concentration": market_concentration,
        "market_concentration_debug": market_concentration_debug,
        "zero_result_flag": zero_result,
    }
    return "\n\n".join(paragraphs), summary_debug


def filename_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", str(value or "").lower()).strip("_")
    return slug or "market"


def render_market_summary_controls(
    summary_text: str,
    scope_key: str,
    active_agency: str,
    selected_year: int,
) -> None:
    generated = st.session_state.get("market_summary_visible_scope_key") == scope_key
    fiscal_slug = filename_slug(fiscal_year_label(int(selected_year)).replace(" ", "_"))
    agency_slug = filename_slug(active_agency)
    file_name = f"{agency_slug}_{fiscal_slug}_market_summary.txt"
    st.markdown(
        """
        <section class="market-summary-panel">
            <div class="market-summary-title">Copy Market Summary</div>
            <div class="market-summary-helper">Summary reflects the last applied analysis.</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    button_cols = st.columns([1, 1, 2.2])
    with button_cols[0]:
        if st.button("Generate Market Summary", key=f"generate-market-summary-{scope_key}", use_container_width=True):
            st.session_state.market_summary_visible_scope_key = scope_key
            generated = True
    with button_cols[1]:
        st.download_button(
            "Download Summary TXT",
            data=(summary_text if generated else "").encode("utf-8"),
            file_name=file_name,
            mime="text/plain",
            key=f"download-market-summary-{scope_key}",
            use_container_width=True,
            disabled=not generated,
        )
    if generated:
        with st.expander("Market Summary", expanded=False):
            st.caption("Ready to copy.")
            st.text_area(
                "Market Summary — ready to copy",
                value=summary_text,
                height=260,
                key=f"market-summary-text-{scope_key}",
            )
    else:
        st.caption("Generate the summary to preview or download the copy-ready brief.")


def render_award_scope_diagnostic_probe() -> None:
    st.markdown("Award Scope Diagnostic Probe")
    st.caption("Runs the same award-scope function as the dashboard for NASA FY2026, bypassing cached rows.")
    if st.button("Run NASA FY2026 Award Scope Probe", key="run-nasa-award-scope-probe"):
        probe_start = time.monotonic()
        probe_df, probe_payload, probe_error = fetch_award_scope_from_download(
            DEFAULT_AGENCY_NAME,
            ALL_BUREAUS,
            current_fiscal_year(),
            contracting_office=ALL_CONTRACTING_OFFICES,
            market_filters=default_market_filters(),
            bypass_cache=True,
        )
        probe_totals = award_scope_totals(probe_df)
        probe_debug = probe_payload.get("award_scope_debug", {})
        st.json(
            {
                "failure_mode": probe_error,
                "raw_rows_returned": probe_debug.get("raw_transaction_rows_returned"),
                "unique_award_keys": probe_debug.get("unique_awards_deduped"),
                "non_null_potential_value_count": probe_debug.get(
                    "count_non_null_potential_total_value_of_award"
                ),
                "non_null_current_value_count": probe_debug.get(
                    "count_non_null_current_total_value_of_award"
                ),
                "active_award_ceiling_total": round(probe_totals["active_award_ceiling"], 2),
                "current_award_value_total": round(probe_totals["current_award_value"], 2),
                "remaining_ceiling_total": round(probe_totals["remaining_ceiling"], 2),
                "elapsed_seconds": round(time.monotonic() - probe_start, 2),
                "diagnostic": probe_debug,
            }
        )


def make_trend_chart(df: pd.DataFrame, selected_year: int) -> go.Figure:
    chart_df = df.copy().sort_values("fiscal_year")
    chart_df["display_amount"] = chart_df["amount"].apply(format_money_with_full)
    chart_df["fiscal_year_label"] = chart_df["fiscal_year"].apply(fiscal_year_label)
    max_value = float(chart_df["amount"].max() or 0) if not chart_df.empty else 0
    tickvals, ticktext = money_ticks(max_value)
    current_fy = current_fiscal_year()
    completed_df = chart_df[chart_df["fiscal_year"] < current_fy].copy()
    current_ytd_df = chart_df[chart_df["fiscal_year"] == current_fy].copy()

    fig = go.Figure()
    if chart_df.empty:
        fig.add_annotation(
            text="Spend trend unavailable from USAspending.gov",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#dce5ef", size=14),
        )
    else:
        if not completed_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=completed_df["fiscal_year_label"],
                    y=completed_df["amount"],
                    customdata=completed_df["display_amount"],
                    mode="lines+markers",
                    line=dict(color="#2dd4bf", width=4, shape="spline"),
                    marker=dict(size=9, color="#f4f7fb", line=dict(color="#2dd4bf", width=2)),
                    hovertemplate="<b>%{x} Contract Obligations</b><br>%{customdata}<extra></extra>",
                    name="Completed Fiscal Years",
                )
            )
        if not current_ytd_df.empty:
            fig.add_trace(
                go.Scatter(
                    x=current_ytd_df["fiscal_year_label"],
                    y=current_ytd_df["amount"],
                    customdata=current_ytd_df["display_amount"],
                    mode="markers",
                    marker=dict(size=14, color="#f59e0b", line=dict(color="#fef3c7", width=2)),
                    hovertemplate=(
                        "<b>%{x} Contract Obligations</b><br>"
                        "%{customdata}<br>Year-to-date only<extra></extra>"
                    ),
                    name="Current FY YTD",
                )
            )
            fig.add_annotation(
                x=current_ytd_df["fiscal_year_label"].iloc[0],
                y=float(current_ytd_df["amount"].iloc[0]),
                text="YTD only",
                showarrow=True,
                arrowcolor="#f59e0b",
                ax=0,
                ay=-35,
                font=dict(color="#fef3c7", size=12),
                bgcolor="rgba(15, 23, 42, 0.85)",
                bordercolor="rgba(245, 158, 11, 0.7)",
                borderwidth=1,
            )
    fig.update_layout(
        title=dict(text="Contract Obligation Trend", font=dict(size=18, color="#f4f7fb")),
        height=430,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5ef", family="Inter, Segoe UI, Arial, sans-serif"),
        xaxis=dict(
            title="Fiscal Year",
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
        ),
        yaxis=dict(
            title="Contract Obligations",
            gridcolor="rgba(255,255,255,0.08)",
            tickvals=tickvals,
            ticktext=ticktext,
            zeroline=False,
        ),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_obligations_over_time_chart(df: pd.DataFrame, time_grain: str) -> go.Figure:
    chart_df = df.copy()
    if not chart_df.empty:
        chart_df["abbreviated_amount"] = chart_df["amount"].apply(format_money)
        chart_df["full_amount"] = chart_df["amount"].apply(format_full_money)
        chart_df["transaction_count_display"] = chart_df["transaction_count"].apply(
            lambda value: format_count(value) if pd.notna(value) and int(value or 0) > 0 else "Unavailable"
        )
    max_value = float(chart_df["amount"].max() or 0) if not chart_df.empty else 0
    tickvals, ticktext = money_ticks(max_value)
    fig = go.Figure()
    if chart_df.empty:
        fig.add_annotation(
            text="Obligations over time unavailable from USAspending.gov",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#dce5ef", size=14),
        )
    else:
        fig.add_trace(
            go.Bar(
                x=chart_df["bucket_label"],
                y=chart_df["amount"],
                customdata=chart_df[
                    ["abbreviated_amount", "full_amount", "transaction_count_display"]
                ],
                marker=dict(
                    color=chart_df["amount"],
                    colorscale=[
                        [0, "#38bdf8"],
                        [0.55, "#2dd4bf"],
                        [1, "#f59e0b"],
                    ],
                    line=dict(color="rgba(255,255,255,0.14)", width=1),
                ),
                hovertemplate=(
                    "<b>%{x}</b><br>"
                    "Obligations: %{customdata[0]}<br>"
                    "Exact amount: %{customdata[1]}<br>"
                    "Transactions: %{customdata[2]}<extra></extra>"
                ),
            )
        )

    x_title = {
        TIME_GRAIN_MONTH: "Month",
        TIME_GRAIN_FISCAL_QUARTER: "Fiscal Quarter",
        TIME_GRAIN_FISCAL_YEAR: "Fiscal Year",
    }.get(time_grain, "Fiscal Year")
    fig.update_layout(
        title=dict(text="Contract Obligations Over Time", font=dict(size=18, color="#f4f7fb")),
        height=430,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5ef", family="Inter, Segoe UI, Arial, sans-serif"),
        xaxis=dict(
            title=x_title,
            gridcolor="rgba(255,255,255,0.08)",
            zeroline=False,
            tickangle=-20 if time_grain == TIME_GRAIN_MONTH else 0,
        ),
        yaxis=dict(
            title="Contract Obligations",
            gridcolor="rgba(255,255,255,0.08)",
            tickvals=tickvals,
            ticktext=ticktext,
            zeroline=False,
        ),
        showlegend=False,
    )
    return fig


def make_vendor_chart(
    df: pd.DataFrame,
    title: str = "Top Contractor Leaderboard",
    amount_label: str = "Obligated Amount",
) -> go.Figure:
    chart_df = df.sort_values("amount", ascending=True).copy()
    chart_df["display_amount"] = chart_df["amount"].apply(format_money_with_full)
    max_value = float(chart_df["amount"].max() or 0) if not chart_df.empty else 0
    tickvals, ticktext = money_ticks(max_value)
    fig = go.Figure()
    if chart_df.empty:
        fig.add_annotation(
            text="Contractor leaderboard awaiting analysis",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#dce5ef", size=14),
        )
    else:
        fig.add_trace(
            go.Bar(
                x=chart_df["amount"],
                y=chart_df["recipient"],
                customdata=chart_df["display_amount"],
                orientation="h",
                marker=dict(
                    color=chart_df["amount"],
                    colorscale=[
                        [0, "#38bdf8"],
                        [0.55, "#2dd4bf"],
                        [1, "#f59e0b"],
                    ],
                    line=dict(color="rgba(255,255,255,0.14)", width=1),
                ),
                hovertemplate=f"<b>%{{y}}</b><br>{html.escape(amount_label)}: %{{customdata}}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color="#f4f7fb")),
        height=430,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5ef", family="Inter, Segoe UI, Arial, sans-serif"),
        xaxis=dict(
            title=amount_label,
            gridcolor="rgba(255,255,255,0.08)",
            tickvals=tickvals,
            ticktext=ticktext,
            zeroline=False,
        ),
        yaxis=dict(title="", gridcolor="rgba(255,255,255,0.04)", automargin=True),
        showlegend=False,
    )
    return fig


def make_budget_reduction_chart(negative_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if negative_df.empty:
        fig.add_annotation(
            text="No negative obligations found for this cycle",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font=dict(color="#dce5ef", size=14),
        )
        min_amount = -1
    else:
        chart_df = (
            negative_df.groupby("Contractor Name", as_index=False)["Obligation Amount"]
            .sum()
            .sort_values("Obligation Amount", ascending=True)
            .head(5)
        )
        chart_df["display_amount"] = chart_df["Obligation Amount"].apply(format_money_with_full)
        min_amount = float(chart_df["Obligation Amount"].min())
        fig.add_trace(
            go.Bar(
                x=chart_df["Obligation Amount"],
                y=chart_df["Contractor Name"],
                customdata=chart_df["display_amount"],
                orientation="h",
                marker=dict(color=CRIMSON, line=dict(color="rgba(255,255,255,0.20)", width=1)),
                hovertemplate="<b>%{y}</b><br>Negative obligation: %{customdata}<extra></extra>",
            )
        )

    tickvals = [min_amount * i / 4 for i in range(4, -1, -1)]
    fig.update_layout(
        title=dict(text="Top Vendors by Negative Obligations", font=dict(size=18, color="#f4f7fb")),
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5ef", family="Inter, Segoe UI, Arial, sans-serif"),
        xaxis=dict(
            title="Negative Obligation Amount",
            gridcolor="rgba(255,255,255,0.08)",
            tickvals=tickvals,
            ticktext=[format_money(value) for value in tickvals],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.35)",
        ),
        yaxis=dict(title="", gridcolor="rgba(255,255,255,0.04)", automargin=True, autorange="reversed"),
        showlegend=False,
    )
    return fig


def audit_log_dataframe(transaction_df: pd.DataFrame) -> pd.DataFrame:
    audit_columns = [
        "Contractor Name",
        "Subagency / Bureau",
        "Mod",
        "Obligation Amount",
        "Signal Classification",
        "Description",
    ]
    if transaction_df.empty:
        return pd.DataFrame(columns=audit_columns)
    audit_mask = transaction_df["Obligation Amount"] < 0
    audit_df = transaction_df[audit_mask].copy()
    if audit_df.empty:
        return pd.DataFrame(columns=audit_columns)
    audit_df["Signal Classification"] = audit_df["Description"].apply(classify_negative_obligation_signal)
    audit_df["Obligation Amount"] = audit_df["Obligation Amount"].apply(format_money)
    return audit_df[audit_columns]


def current_and_previous(df: pd.DataFrame, selected_year: int) -> tuple[float, float | None]:
    selected = df[df["fiscal_year"] == selected_year]
    current_total = float(selected["amount"].iloc[0]) if not selected.empty else 0
    previous = df[df["fiscal_year"] == selected_year - 1]
    previous_total = float(previous["amount"].iloc[0]) if not previous.empty else None
    return current_total, previous_total


def yoy_comparison_context(
    active_agency: str,
    selected_bureau: str | None,
    selected_year: int,
    trend_df: pd.DataFrame,
    current_total: float,
    office_filter_active: bool,
    selected_contracting_office: str,
    selected_market_filters: dict,
) -> tuple[float | None, str, str]:
    if office_filter_active:
        if selected_year == current_fiscal_year():
            return None, "YoY YTD Change in Contract Obligations", "Not calculated for office-level refinements"
        return None, "YoY Change in Contract Obligations", "Not calculated for office-level refinements"

    if selected_year == current_fiscal_year():
        prior_start, prior_end = prior_ytd_date_range(selected_year)
        previous_total, _payload, _error = fetch_trend_period_total(
            active_agency,
            selected_bureau,
            prior_start,
            prior_end,
            market_filters=selected_market_filters,
        )
        return (
            previous_total,
            "YoY YTD Change in Contract Obligations",
            "Compared with same period in prior fiscal year",
        )

    if office_filter_active:
        prior_start, prior_end = fiscal_year_date_range(selected_year - 1)
        previous_total, _payload, _error = fetch_transaction_period_total(
            active_agency,
            selected_bureau,
            prior_start,
            prior_end,
            contracting_office=selected_contracting_office,
            market_filters=selected_market_filters,
        )
    else:
        _current_total, previous_total = current_and_previous(trend_df, selected_year)
    return previous_total, "YoY Change in Contract Obligations", "Compared with prior fiscal year"


def transaction_trend_dataframe(transaction_df: pd.DataFrame, selected_year: int) -> pd.DataFrame:
    total = 0.0
    if not transaction_df.empty and "Obligation Amount" in transaction_df.columns:
        total = float(transaction_df["Obligation Amount"].sum())
    return pd.DataFrame([{"fiscal_year": int(selected_year), "amount": total}])


def transaction_vendor_dataframe(transaction_df: pd.DataFrame) -> pd.DataFrame:
    if transaction_df.empty:
        return pd.DataFrame(columns=["recipient", "amount"])
    vendor_df = (
        transaction_df.groupby("Contractor Name", as_index=False)["Obligation Amount"]
        .sum()
        .rename(columns={"Contractor Name": "recipient", "Obligation Amount": "amount"})
    )
    vendor_df = vendor_df[pd.to_numeric(vendor_df["amount"], errors="coerce").fillna(0).abs() >= 0.005]
    return vendor_df.sort_values("amount", ascending=False).head(10)


MARKET_CONCENTRATION_SEGMENT_COLORS = (
    "#2dd4bf",
    "#38bdf8",
    "#a78bfa",
    "#f59e0b",
    "#fb7185",
    "#64748b",
)


def calculate_market_concentration(
    filtered_transactions: pd.DataFrame,
    *,
    contractor_key_col: str,
    obligation_col: str = "Obligation Amount",
    top_n: int = 5,
) -> dict:
    if (
        filtered_transactions is None
        or filtered_transactions.empty
        or contractor_key_col not in filtered_transactions.columns
        or obligation_col not in filtered_transactions.columns
    ):
        return {
            "metric": f"top_{top_n}_positive_obligation_share",
            "net_obligations": 0.0,
            "positive_obligations": 0.0,
            "negative_obligations": 0.0,
            "top_n_positive_obligations": 0.0,
            "top_n_share": None,
            "contractor_breakdown": [],
            "contractor_count_positive": 0,
            "calculation_basis": "positive_transaction_obligations",
            "legacy_top_n_net_obligations": 0.0,
            "legacy_net_denominator": 0.0,
            "legacy_net_denominator_share": None,
        }

    working_df = filtered_transactions.copy()
    working_df[obligation_col] = pd.to_numeric(working_df[obligation_col], errors="coerce").fillna(0.0)
    net_obligations = float(working_df[obligation_col].sum())
    positive_rows = working_df[working_df[obligation_col] > 0].copy()
    negative_obligations = float(working_df.loc[working_df[obligation_col] < 0, obligation_col].sum())
    positive_obligations = float(positive_rows[obligation_col].sum())

    legacy_grouped = (
        working_df.groupby(contractor_key_col, dropna=False, as_index=False)[obligation_col]
        .sum()
        .rename(columns={contractor_key_col: "contractor", obligation_col: "amount"})
    )
    legacy_grouped = legacy_grouped[
        pd.to_numeric(legacy_grouped["amount"], errors="coerce").fillna(0.0).abs() >= 0.005
    ]
    legacy_grouped = legacy_grouped.sort_values("amount", ascending=False).reset_index(drop=True)
    legacy_top_n_net_obligations = float(
        pd.to_numeric(legacy_grouped.head(top_n)["amount"], errors="coerce").fillna(0.0).sum()
    )
    legacy_net_denominator_share = (
        legacy_top_n_net_obligations / net_obligations if abs(net_obligations) >= 0.005 else None
    )

    if positive_obligations <= 0:
        return {
            "metric": f"top_{top_n}_positive_obligation_share",
            "net_obligations": net_obligations,
            "positive_obligations": positive_obligations,
            "negative_obligations": negative_obligations,
            "top_n_positive_obligations": 0.0,
            "top_n_share": None,
            "contractor_breakdown": [],
            "contractor_count_positive": 0,
            "calculation_basis": "positive_transaction_obligations",
            "legacy_top_n_net_obligations": legacy_top_n_net_obligations,
            "legacy_net_denominator": net_obligations,
            "legacy_net_denominator_share": legacy_net_denominator_share,
        }

    positive_contractor_totals = (
        positive_rows.groupby(contractor_key_col, dropna=False, as_index=False)[obligation_col]
        .sum()
        .rename(columns={contractor_key_col: "contractor", obligation_col: "amount"})
    )
    positive_contractor_totals = positive_contractor_totals[
        pd.to_numeric(positive_contractor_totals["amount"], errors="coerce").fillna(0.0) > 0.005
    ]
    positive_contractor_totals = positive_contractor_totals.sort_values("amount", ascending=False).reset_index(drop=True)
    top_positive_contractors = positive_contractor_totals.head(top_n)
    top_n_positive_obligations = float(
        pd.to_numeric(top_positive_contractors["amount"], errors="coerce").fillna(0.0).sum()
    )
    top_n_share = top_n_positive_obligations / positive_obligations
    contractor_breakdown = [
        {
            "contractor": clean_office_value(row.get("contractor")) or "Unknown Contractor",
            "amount": float(row.get("amount") or 0.0),
            "share": float(row.get("amount") or 0.0) / positive_obligations,
        }
        for row in top_positive_contractors.to_dict("records")
    ]
    return {
        "metric": f"top_{top_n}_positive_obligation_share",
        "net_obligations": net_obligations,
        "positive_obligations": positive_obligations,
        "negative_obligations": negative_obligations,
        "top_n_positive_obligations": top_n_positive_obligations,
        "top_n_share": top_n_share,
        "contractor_breakdown": contractor_breakdown,
        "contractor_count_positive": int(len(positive_contractor_totals)),
        "calculation_basis": "positive_transaction_obligations",
        "legacy_top_n_net_obligations": legacy_top_n_net_obligations,
        "legacy_net_denominator": net_obligations,
        "legacy_net_denominator_share": legacy_net_denominator_share,
    }


def hex_color_with_alpha(hex_color: str, alpha: float) -> str:
    normalized = str(hex_color or "").strip().lstrip("#")
    if len(normalized) != 6:
        return f"rgba(100, 116, 139, {alpha})"
    red = int(normalized[0:2], 16)
    green = int(normalized[2:4], 16)
    blue = int(normalized[4:6], 16)
    return f"rgba({red}, {green}, {blue}, {alpha})"


def market_concentration_summary(transaction_df: pd.DataFrame, total_obligations: float) -> tuple[dict, dict]:
    empty_result = {
        "value": "N/A",
        "subtitle": "Top 5 share of positive obligations",
        "supporting_text": "No positive obligation transactions in this scope.",
        "helper_text": (
            "Uses positive obligation transactions. Negative obligations remain included in the net Contract "
            "Obligations KPI and are reported separately."
        ),
        "classification": "",
        "donut_slice_data": [],
        "concentration_segments": [],
        "grouped_contractor_count": 0,
        "other_share_percentage": None,
    }
    if transaction_df.empty or not {"Contractor Name", "Obligation Amount"}.issubset(transaction_df.columns):
        return empty_result, {
            "metric": "top_5_positive_obligation_share",
            "net_market_obligations": round(float(total_obligations or 0.0), 2),
            "gross_positive_obligations": 0.0,
            "gross_negative_obligations": 0.0,
            "top_5_positive_obligations": 0.0,
            "top_5_positive_share": None,
            "calculation_basis": "positive_transaction_obligations",
            "total_obligations": round(float(total_obligations or 0.0), 2),
            "grouped_contractor_count": 0,
            "contractor_count_positive": 0,
            "top_5_contractor_names": [],
            "top_5_contractor_sums": [],
            "top_5_sum": 0.0,
            "other_contractor_sum": 0.0,
            "concentration_percentage": None,
            "donut_slice_data": [],
            "concentration_segments": [],
            "top_5_contractor_percentages": [],
            "other_contractor_percentage": 0.0,
            "total_segment_percentage": 0.0,
            "reconciliation_status": False,
        }

    concentration = calculate_market_concentration(
        transaction_df,
        contractor_key_col="Contractor Name",
        obligation_col="Obligation Amount",
        top_n=5,
    )
    total = float(concentration["net_obligations"])
    positive_total = float(concentration["positive_obligations"])
    negative_total = float(concentration["negative_obligations"])
    top_5_sum = float(concentration["top_n_positive_obligations"])
    top_5_share = concentration["top_n_share"]
    top_5 = concentration["contractor_breakdown"]
    other_sum = positive_total - top_5_sum
    grouped_total = total
    reconciles = abs(grouped_total - float(total_obligations or 0.0)) <= 0.01
    if not reconciles:
        print("ERROR: Market concentration total does not reconcile to KPI.")
    if top_5_share is None:
        summary = empty_result
        concentration_pct = None
    else:
        concentration_pct = top_5_share * 100
        if concentration_pct >= 70:
            classification = "Highly concentrated market"
        elif concentration_pct >= 40:
            classification = "Moderately concentrated market"
        else:
            classification = "Fragmented market"
        shown_count = int(len(top_5))
        summary = {
            "value": f"{concentration_pct:.1f}%",
            "subtitle": "Top 5 share of positive obligations",
            "supporting_text": (
                f"Top {shown_count} contractors captured {concentration_pct:.1f}% of positive obligations "
                "in this scope."
            ),
            "helper_text": (
                "Uses positive obligation transactions. Negative obligations remain included in the net Contract "
                "Obligations KPI and are reported separately."
            ),
            "classification": classification,
        }
        if shown_count == 1 and int(concentration["contractor_count_positive"]) == 1:
            contractor_name = str(top_5[0].get("contractor") or "The only contractor")
            summary["subtitle"] = "Single-vendor scope"
            summary["supporting_text"] = (
                f"{contractor_name} accounts for all positive obligations in this filtered market."
            )
    donut_slice_data = [
        {
            "contractor": clean_office_value(row.get("contractor")) or "Unknown Contractor",
            "amount": round(float(row.get("amount") or 0.0), 2),
            "display_amount": format_money(row.get("amount")),
        }
        for row in top_5
        if float(row.get("amount") or 0.0) > 0.005
    ]
    if other_sum > 0.005:
        donut_slice_data.append(
            {
                "contractor": "All Other Contractors",
                "amount": round(other_sum, 2),
                "display_amount": format_money(other_sum),
            }
        )
    concentration_segments = [
        {
            "contractor": row["contractor"],
            "amount": row["amount"],
            "display_amount": row["display_amount"],
            "percentage": round((float(row["amount"] or 0.0) / positive_total) * 100, 1)
            if positive_total > 0
            else 0.0,
        }
        for row in donut_slice_data
        if row["contractor"] != "All Other Contractors"
    ]
    other_pct = round((max(other_sum, 0.0) / positive_total) * 100, 1) if positive_total > 0 else 0.0
    if other_sum > 0.005:
        concentration_segments.append(
            {
                "contractor": "All Other Contractors",
                "amount": round(other_sum, 2),
                "display_amount": format_money(other_sum),
                "percentage": other_pct,
            }
        )
    total_segment_pct = round(sum(float(row["percentage"] or 0.0) for row in concentration_segments), 1)
    summary["donut_slice_data"] = donut_slice_data
    summary["concentration_segments"] = concentration_segments
    summary["grouped_contractor_count"] = int(concentration["contractor_count_positive"])
    summary["other_share_percentage"] = other_pct if concentration_pct is not None else None
    debug = {
        "metric": "top_5_positive_obligation_share",
        "net_market_obligations": round(total, 2),
        "gross_positive_obligations": round(positive_total, 2),
        "gross_negative_obligations": round(negative_total, 2),
        "top_5_positive_obligations": round(top_5_sum, 2),
        "top_5_positive_share": round(float(top_5_share), 6) if top_5_share is not None else None,
        "contractor_count_positive": int(concentration["contractor_count_positive"]),
        "calculation_basis": "positive_transaction_obligations",
        "legacy_top_5_net_contractor_obligations": round(
            float(concentration["legacy_top_n_net_obligations"]), 2
        ),
        "legacy_net_denominator": round(float(concentration["legacy_net_denominator"]), 2),
        "legacy_net_denominator_percentage": round(
            float(concentration["legacy_net_denominator_share"]) * 100, 6
        )
        if concentration["legacy_net_denominator_share"] is not None
        else None,
        "total_obligations": round(total, 2),
        "positive_obligations": round(positive_total, 2),
        "negative_obligations": round(negative_total, 2),
        "grouped_contractor_count": int(concentration["contractor_count_positive"]),
        "top_5_contractor_names": [str(row.get("contractor") or "") for row in top_5],
        "top_5_contractor_sums": [
            round(float(row.get("amount") or 0.0), 2) for row in top_5
        ],
        "top_5_contractor_percentages": [
            round((float(row.get("amount") or 0.0) / positive_total) * 100, 1)
            if positive_total > 0
            else 0.0
            for row in top_5
        ],
        "top_5_sum": round(top_5_sum, 2),
        "other_contractor_sum": round(other_sum, 2),
        "other_contractor_percentage": other_pct,
        "concentration_percentage": round(concentration_pct, 1) if concentration_pct is not None else None,
        "donut_slice_data": donut_slice_data,
        "concentration_segments": concentration_segments,
        "total_segment_percentage": total_segment_pct,
        "reconciliation_status": reconciles,
    }
    return summary, debug


RAW_HTML_VISIBLE_TOKENS = ["<div", "</div>", "<span", "</span>", "class=", "style="]


def visible_text_contains_raw_html(text: str) -> bool:
    lowered = str(text or "").lower()
    return any(token in lowered for token in RAW_HTML_VISIBLE_TOKENS)


def clean_visible_market_text(text: str) -> str:
    cleaned = str(text or "")
    for token in RAW_HTML_VISIBLE_TOKENS:
        cleaned = re.sub(re.escape(token), "", cleaned, flags=re.IGNORECASE)
    return cleaned.replace("<", "").replace(">", "")


def render_market_concentration_legend(legend_rows: list[dict]) -> None:
    if not legend_rows:
        return
    row_markup = []
    for row in legend_rows:
        contractor = clean_visible_market_text(row.get("contractor") or "Unknown Contractor")
        amount = clean_visible_market_text(row.get("display_amount") or format_money(row.get("amount")))
        try:
            percentage = float(row.get("percentage") or 0.0)
        except (TypeError, ValueError):
            percentage = 0.0
        color = str(row.get("color") or MARKET_CONCENTRATION_SEGMENT_COLORS[-1])
        visible_values = [contractor, amount, f"{percentage:.1f}%"]
        if any(visible_text_contains_raw_html(value) for value in visible_values):
            print("ERROR: raw HTML detected in Market Concentration legend text.")
        row_bg = hex_color_with_alpha(color, 0.15)
        badge_border = hex_color_with_alpha(color, 0.38)
        row_markup.append(
            f'<div class="market-concentration-legend-row" style="border-left-color: {html.escape(color)}; '
            f'background: {row_bg};">'
            f'<div class="market-concentration-legend-name">{html.escape(contractor)}</div>'
            f'<div class="market-concentration-legend-metrics">'
            f'<span class="market-concentration-legend-badge" style="color: {html.escape(color)}; '
            f'border-color: {badge_border};">{percentage:.1f}%</span>'
            f'<span class="market-concentration-legend-badge" style="color: {html.escape(color)}; '
            f'border-color: {badge_border};">{html.escape(amount)}</span>'
            f"</div></div>"
        )
    st.markdown(
        '<div class="market-concentration-legend">'
        '<div class="market-concentration-legend-heading">Contractor breakdown - positive obligations</div>'
        + "".join(row_markup)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_market_concentration_card(market_concentration: dict) -> None:
    visible_text_values = [
        str(market_concentration.get("value") or "N/A"),
        str(market_concentration.get("subtitle") or "Top 5 share of positive obligations"),
        str(
            market_concentration.get("supporting_text")
            or market_concentration.get("subtitle")
            or "No positive obligation transactions in this scope."
        ),
        str(market_concentration.get("helper_text") or ""),
        str(market_concentration.get("classification") or ""),
    ]
    if any(visible_text_contains_raw_html(text) for text in visible_text_values):
        print("ERROR: raw HTML detected in Market Concentration visible text.")
    value = clean_visible_market_text(visible_text_values[0])
    subtitle = clean_visible_market_text(visible_text_values[1])
    supporting_text = clean_visible_market_text(
        market_concentration.get("supporting_text")
        or market_concentration.get("subtitle")
        or "No positive obligation transactions in this scope."
    )
    helper_text = clean_visible_market_text(market_concentration.get("helper_text") or "")
    classification = clean_visible_market_text(visible_text_values[4])
    try:
        top_share = float(value.replace("%", "")) if value != "N/A" else 0.0
    except ValueError:
        top_share = 0.0
    if top_share > 100.0:
        print("ERROR: Market concentration percentage exceeded 100%.")
    segments = market_concentration.get("concentration_segments") or []
    segment_markup = ""
    legend_rows = []
    if value != "N/A" and segments:
        segment_parts = []
        for index, segment in enumerate(segments[:6]):
            color = MARKET_CONCENTRATION_SEGMENT_COLORS[index % len(MARKET_CONCENTRATION_SEGMENT_COLORS)]
            contractor = str(segment.get("contractor") or "Unknown Contractor")
            amount = str(segment.get("display_amount") or format_money(segment.get("amount")))
            percentage = float(segment.get("percentage") or 0.0)
            title = f"{contractor}: {amount} ({percentage:.1f}% of positive obligations)"
            segment_parts.append(
                f'<div class="market-concentration-segment" style="width: {max(percentage, 0.0):.1f}%; '
                f'background: {color};" title="{html.escape(title)}"></div>'
            )
            legend_rows.append(
                {
                    "contractor": contractor,
                    "percentage": percentage,
                    "display_amount": amount,
                    "color": color,
                }
            )
        segment_markup = (
            '<div class="market-concentration-bar" aria-label="Market concentration by contractor">'
            + "".join(segment_parts)
            + "</div>"
        )
    st.markdown(
        f"""
        <section class="market-intel-card" style="--accent: #a78bfa;">
            <div class="market-intel-label">Market Concentration</div>
            <div class="market-intel-value">{html.escape(value)}</div>
            <div class="market-intel-subtitle">{html.escape(subtitle)}</div>
            {segment_markup}
            <div class="market-intel-helper">{html.escape(supporting_text)}</div>
            <div class="market-intel-helper" title="{html.escape(helper_text)}">{html.escape(helper_text)}</div>
            <div class="market-intel-helper">{html.escape(classification)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_market_concentration_legend(legend_rows)


NAICS_CODE_ALIASES = [
    "NAICS Code",
    "naics_code",
    "naics",
    "NAICS",
    "naicsCode",
    "naics_code_current",
]
NAICS_DESCRIPTION_ALIASES = [
    "NAICS Description",
    "naics_description",
    "naics_description_current",
    "NAICS Description Current",
    "naics_desc",
]
PSC_CODE_ALIASES = [
    "PSC Code",
    "product_or_service_code",
    "product_or_service_code_current",
    "psc_code",
    "psc",
    "PSC",
    "Product or Service Code",
]
PSC_DESCRIPTION_ALIASES = [
    "PSC Description",
    "product_or_service_code_description",
    "product_or_service_code_description_current",
    "psc_description",
    "Product or Service Code Description",
]


def dataframe_alias_columns_found(df: pd.DataFrame, aliases: list[str]) -> list[str]:
    return [
        alias
        for alias in aliases
        if alias in df.columns
        and df[alias].apply(lambda value: clean_office_value(value) not in {"", "Unspecified"}).any()
    ]


def dataframe_first_non_null_lane_samples(
    df: pd.DataFrame,
    code_aliases: list[str],
    description_aliases: list[str],
    limit: int = 5,
) -> list[dict]:
    samples = []
    for row in df.to_dict("records"):
        code = clean_office_value(first_present(row, code_aliases))
        description = clean_office_value(first_present(row, description_aliases))
        if code and code != "Unspecified" or description and description != "Unspecified":
            samples.append({"code": code, "description": description})
        if len(samples) >= limit:
            break
    return samples


def dataframe_lane_value(row: dict, aliases: list[str]) -> str:
    return clean_office_value(first_present(row, aliases))


def market_lane_mix_dataframe(
    transaction_df: pd.DataFrame,
    lane_kind: str,
    total_obligations: float,
    market_filters: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    market_filters = normalize_market_filters(market_filters)
    if lane_kind == "naics":
        output_code_column = "NAICS"
        code_aliases = NAICS_CODE_ALIASES
        description_aliases = NAICS_DESCRIPTION_ALIASES
        selected_code, selected_description = decode_option(market_filters["naics_code"])
        selected_filter_active = bool(selected_code and selected_code != ALL_NAICS_CODES)
        field_names_found_key = "transaction_row_naics_field_names_found"
        candidates_found_key = "naics_column_candidates_found"
        samples_key = "first_5_non_null_naics_values"
    else:
        output_code_column = "PSC"
        code_aliases = PSC_CODE_ALIASES
        description_aliases = PSC_DESCRIPTION_ALIASES
        selected_code, selected_description = decode_option(market_filters["psc_code"])
        selected_filter_active = bool(selected_code and selected_code != ALL_PRODUCT_SERVICE_CODES)
        field_names_found_key = "transaction_row_psc_field_names_found"
        candidates_found_key = "psc_column_candidates_found"
        samples_key = "first_5_non_null_psc_values"
    grouped_vs_kpi_key = f"grouped_{lane_kind}_total_vs_kpi"

    columns = [
        output_code_column,
        "Description",
        "Full Description",
        "Obligated",
        "% of Scope",
        "Awards",
        "Contractors",
    ]
    if transaction_df.empty:
        return pd.DataFrame(columns=columns), {
            "raw_transaction_rows_used": 0,
            "grouped_row_count": 0,
            "grouped_obligation_sum": 0.0,
            "main_kpi_total": round(float(total_obligations or 0.0), 2),
            "reconciliation_status": abs(float(total_obligations or 0.0)) <= 0.01,
            field_names_found_key: [],
            "transaction_df_columns": list(transaction_df.columns),
            candidates_found_key: [],
            samples_key: [],
            "selected_filter_fallback_used": False,
            grouped_vs_kpi_key: {
                "grouped_obligation_sum": 0.0,
                "main_kpi_total": round(float(total_obligations or 0.0), 2),
                "difference": round(0.0 - float(total_obligations or 0.0), 2),
            },
        }

    scoped_df = transaction_df.copy()
    code_columns_found = dataframe_alias_columns_found(scoped_df, code_aliases)
    description_columns_found = dataframe_alias_columns_found(scoped_df, description_aliases)
    field_names_found = sorted(set(code_columns_found + description_columns_found))
    sample_values = dataframe_first_non_null_lane_samples(scoped_df, code_aliases, description_aliases)
    selected_filter_fallback_used = False

    def lane_code_value(row: dict) -> str:
        nonlocal selected_filter_fallback_used
        code = dataframe_lane_value(row, code_aliases)
        if code and code != "Unspecified":
            return code
        if selected_filter_active:
            selected_filter_fallback_used = True
            return clean_office_value(selected_code) or "Unspecified"
        return "Unspecified"

    def lane_description_value(row: dict) -> str:
        nonlocal selected_filter_fallback_used
        description = dataframe_lane_value(row, description_aliases)
        if description and description != "Unspecified":
            return description
        if selected_filter_active:
            selected_filter_fallback_used = True
            return clean_office_value(selected_description) or clean_office_value(selected_code) or "Unspecified"
        return "Unspecified"

    lane_code_column = f"_{lane_kind}_lane_code"
    lane_description_column = f"_{lane_kind}_lane_description"
    scoped_df[lane_code_column] = [lane_code_value(row) for row in scoped_df.to_dict("records")]
    scoped_df[lane_description_column] = [lane_description_value(row) for row in scoped_df.to_dict("records")]
    scoped_df["_lane_award_key"] = scoped_df["Contract Award Unique Key"].apply(clean_office_value)
    grouped = (
        scoped_df.groupby([lane_code_column, lane_description_column], as_index=False)
        .agg(
            Obligated=("Obligation Amount", "sum"),
            Awards=("_lane_award_key", lambda values: values.replace("", pd.NA).nunique(dropna=True)),
            Contractors=("Contractor Name", "nunique"),
        )
        .rename(columns={lane_code_column: output_code_column, lane_description_column: "Full Description"})
    )
    total = float(total_obligations or 0.0)
    grouped["% of Scope"] = grouped["Obligated"].apply(lambda amount: (float(amount or 0.0) / total) * 100 if total else 0.0)
    grouped["Description"] = grouped["Full Description"].apply(truncate_award_description)
    grouped = grouped.sort_values("Obligated", ascending=False).reset_index(drop=True)
    grouped_total = float(pd.to_numeric(grouped["Obligated"], errors="coerce").fillna(0).sum())
    debug = {
        "raw_transaction_rows_used": int(len(scoped_df)),
        "grouped_row_count": int(len(grouped)),
        "grouped_obligation_sum": round(grouped_total, 2),
        "main_kpi_total": round(total, 2),
        "reconciliation_status": abs(grouped_total - total) <= 0.01,
        field_names_found_key: field_names_found,
        "transaction_df_columns": list(transaction_df.columns),
        candidates_found_key: field_names_found,
        samples_key: sample_values,
        "selected_filter_fallback_used": selected_filter_fallback_used,
        grouped_vs_kpi_key: {
            "grouped_obligation_sum": round(grouped_total, 2),
            "main_kpi_total": round(total, 2),
            "difference": round(grouped_total - total, 2),
        },
    }
    return grouped[columns], debug


def render_market_lane_mix(
    transaction_df: pd.DataFrame,
    total_obligations: float,
    market_filters: dict | None = None,
    market_concentration: dict | None = None,
) -> tuple[dict, dict]:
    st.markdown('<div class="audit-heading">Market Intelligence</div>', unsafe_allow_html=True)
    market_filters = normalize_market_filters(market_filters)
    naics_df, naics_debug = market_lane_mix_dataframe(transaction_df, "naics", total_obligations, market_filters)
    psc_df, psc_debug = market_lane_mix_dataframe(transaction_df, "psc", total_obligations, market_filters)
    market_concentration = market_concentration or {
        "value": "N/A",
        "subtitle": "No contract obligations found for this scope.",
        "classification": "",
    }

    def top_lane_summary(
        df: pd.DataFrame,
        code_column: str,
        empty_title: str,
        selected_filter_active: bool,
        unavailable_message: str,
    ) -> dict:
        if df.empty:
            return {
                "label": empty_title,
                "value": "N/A",
                "subtitle": "No prime contract obligations found.",
                "helper": "",
                "meaningful": False,
                "note": unavailable_message,
            }
        top_row = df.iloc[0]
        code = clean_office_value(top_row.get(code_column)) or "Unspecified"
        description = clean_office_value(top_row.get("Full Description")) or "Unspecified"
        obligated = format_money(top_row.get("Obligated"))
        pct = f"{float(top_row.get('% of Scope') or 0.0):.1f}%"
        is_unfiltered_unspecified = (
            not selected_filter_active
            and code == "Unspecified"
            and description == "Unspecified"
            and float(top_row.get("% of Scope") or 0.0) >= 99.95
        )
        return {
            "label": empty_title,
            "value": code,
            "subtitle": f"{obligated} · {pct} of scope",
            "helper": description,
            "meaningful": not is_unfiltered_unspecified,
            "note": unavailable_message if is_unfiltered_unspecified else "",
        }

    selected_naics_code, _selected_naics_description = decode_option(market_filters["naics_code"])
    selected_psc_code, _selected_psc_description = decode_option(market_filters["psc_code"])
    naics_summary = top_lane_summary(
        naics_df,
        "NAICS",
        "Top NAICS by Obligations",
        bool(selected_naics_code and selected_naics_code != ALL_NAICS_CODES),
        "NAICS mix unavailable from returned transaction rows for this scope.",
    )
    psc_summary = top_lane_summary(
        psc_df,
        "PSC",
        "Top PSC by Obligations",
        bool(selected_psc_code and selected_psc_code != ALL_PRODUCT_SERVICE_CODES),
        "PSC mix unavailable from returned transaction rows for this scope.",
    )

    def render_lane_summary(summary: dict, accent: str) -> None:
        if not summary.get("meaningful"):
            st.markdown(
                f"""
                <section class="market-intel-note">
                    {html.escape(str(summary.get("note") or "Lane mix unavailable from returned transaction rows for this scope."))}
                </section>
                """,
                unsafe_allow_html=True,
            )
            return
        metric_card(
            summary["label"],
            summary["value"],
            summary["subtitle"],
            accent,
            helper_text=summary["helper"],
        )

    intelligence_cols = st.columns(3)
    with intelligence_cols[0]:
        render_market_concentration_card(market_concentration)
    with intelligence_cols[1]:
        render_lane_summary(naics_summary, "#38bdf8")
    with intelligence_cols[2]:
        render_lane_summary(psc_summary, "#2dd4bf")

    if transaction_df.empty:
        st.info("No NAICS or PSC mix available because this scope has no prime contract obligations.")
        return naics_debug, psc_debug

    def display_lane_table(df: pd.DataFrame, title: str) -> None:
        st.markdown(f"**{title}**")
        display_df = df.head(10).copy()
        display_df["Obligated"] = display_df["Obligated"].apply(format_money)
        display_df["% of Scope"] = display_df["% of Scope"].apply(lambda value: f"{float(value or 0.0):.1f}%")
        display_df = display_df.drop(columns=["Full Description"])
        render_dark_html_table(
            display_df,
            columns=list(display_df.columns),
        )

    with st.expander("View Market Lane Mix", expanded=False):
        mix_cols = st.columns(2)
        with mix_cols[0]:
            display_lane_table(naics_df, "Top NAICS by Obligations")
        with mix_cols[1]:
            display_lane_table(psc_df, "Top PSC by Obligations")
    return naics_debug, psc_debug


def usaspending_award_url(contract_award_unique_key: str) -> str:
    award_key = clean_office_value(contract_award_unique_key)
    if not award_key:
        return ""
    return f"https://www.usaspending.gov/award/{quote(award_key, safe='_')}"


def award_drilldown_dataframe(transaction_df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = [
        "Contractor",
        "Award ID",
        "Full Award Description",
        "Obligations This Period",
        "Current Award Value",
        "Award Ceiling",
        "Remaining Ceiling",
        "Contracting Office",
        "Funding Office",
        "Action Date",
        "contract_award_unique_key",
        "USAspending",
    ]
    if transaction_df.empty:
        return pd.DataFrame(columns=columns), {
            "raw_transaction_rows_used": 0,
            "grouped_award_count": 0,
            "total_grouped_award_obligation_sum": 0.0,
            "sample_generated_usaspending_links": [],
        }

    scoped_df = transaction_df.copy()
    grouped_awards: dict[tuple[str, str], list[dict]] = {}
    for row_order, row in enumerate(scoped_df.to_dict("records")):
        award_key = clean_office_value(row.get("Contract Award Unique Key"))
        award_id = clean_office_value(row.get("Award ID"))
        group_key = ("contract_award_unique_key", award_key) if award_key else ("missing_contract_award_unique_key", award_id or str(row_order))
        row["_award_drilldown_row_order"] = int(row.get("Raw Row Order") or row_order)
        grouped_awards.setdefault(group_key, []).append(row)

    award_rows = []
    for (_key_type, award_key), award_rows_for_key in grouped_awards.items():
        latest_row = sorted(
            award_rows_for_key,
            key=lambda row: (
                parse_action_date(row.get("Action Date")) or date.min,
                award_sequence_value(
                    {
                        "modification_number": row.get("Mod"),
                        "transaction_number": row.get("Transaction Number"),
                    }
                ),
                int(row.get("_award_drilldown_row_order") or 0),
            ),
        )[-1]
        obligations_this_period = sum(
            parse_currency_amount(row.get("Obligation Amount")) for row in award_rows_for_key
        )
        current_award_value = parse_currency_amount(latest_row.get("Current Award Value"))
        award_ceiling = parse_currency_amount(latest_row.get("Award Ceiling"))
        remaining_ceiling = max(award_ceiling - current_award_value, 0.0)
        award_url = usaspending_award_url(latest_row.get("Contract Award Unique Key"))
        award_rows.append(
            {
                "Contractor": clean_office_value(latest_row.get("Contractor Name")) or "Unknown Contractor",
                "Award ID": clean_office_value(latest_row.get("Award ID")) or "Unavailable",
                "Full Award Description": clean_office_value(latest_row.get("Description")),
                "Obligations This Period": obligations_this_period,
                "Current Award Value": current_award_value,
                "Award Ceiling": award_ceiling,
                "Remaining Ceiling": remaining_ceiling,
                "Contracting Office": clean_office_value(latest_row.get("Contracting Office Name"))
                or clean_office_value(latest_row.get("Contracting Office")),
                "Funding Office": clean_office_value(latest_row.get("Funding Office Name"))
                or clean_office_value(latest_row.get("Funding Office")),
                "Action Date": latest_row.get("Action Date"),
                "contract_award_unique_key": clean_office_value(latest_row.get("Contract Award Unique Key")),
                "USAspending": award_url or None,
            }
        )

    award_df = pd.DataFrame(award_rows, columns=columns)
    if not award_df.empty:
        award_df = award_df.sort_values("Obligations This Period", ascending=False).reset_index(drop=True)
    grouped_total = (
        float(pd.to_numeric(award_df["Obligations This Period"], errors="coerce").fillna(0).sum())
        if not award_df.empty
        else 0.0
    )
    sample_links = [
        {
            "award_id": str(row.get("Award ID") or ""),
            "url": str(row.get("USAspending") or ""),
        }
        for row in award_df[award_df["USAspending"].notna()].head(5).to_dict("records")
    ]
    return award_df, {
        "raw_transaction_rows_used": int(len(scoped_df)),
        "grouped_award_count": int(len(award_df)),
        "total_grouped_award_obligation_sum": round(grouped_total, 2),
        "sample_generated_usaspending_links": sample_links,
    }


def truncate_award_description(description: str, limit: int = 125) -> str:
    text = clean_office_value(description)
    if len(text) <= limit:
        return text
    return f"{text[: limit - 1].rstrip()}..."


def sorted_award_drilldown_dataframe(award_df: pd.DataFrame, rank_by: str) -> pd.DataFrame:
    if award_df.empty:
        return award_df.copy()
    sort_config = {
        "Obligations This Period": ("Obligations This Period", date.min),
        "Current Award Value": ("Current Award Value", 0.0),
        "Award Ceiling": ("Award Ceiling", 0.0),
        "Remaining Ceiling": ("Remaining Ceiling", 0.0),
        "Most Recent Action": ("Action Date", date.min),
    }
    sort_column, fill_value = sort_config.get(rank_by, sort_config["Obligations This Period"])
    sortable_df = award_df.copy()
    if sort_column == "Action Date":
        sortable_df["_award_drilldown_sort"] = sortable_df[sort_column].apply(
            lambda value: parse_action_date(value) or date.min
        )
    else:
        sortable_df["_award_drilldown_sort"] = pd.to_numeric(
            sortable_df[sort_column], errors="coerce"
        ).fillna(fill_value)
    return (
        sortable_df.sort_values("_award_drilldown_sort", ascending=False)
        .drop(columns=["_award_drilldown_sort"])
        .reset_index(drop=True)
    )


def award_drilldown_csv_bytes(award_df: pd.DataFrame) -> bytes:
    export_columns = [
        "Contractor",
        "Award ID",
        "Full Award Description",
        "Obligations This Period",
        "Current Award Value",
        "Award Ceiling",
        "Remaining Ceiling",
        "Contracting Office",
        "Funding Office",
        "Action Date",
        "contract_award_unique_key",
        "USAspending URL",
    ]
    export_df = award_df.copy()
    if "Action Date" in export_df.columns:
        export_df["Action Date"] = export_df["Action Date"].apply(
            lambda value: value.isoformat() if isinstance(value, date) else clean_office_value(value)
        )
    export_df = export_df.rename(columns={"USAspending": "USAspending URL"})
    return export_df[export_columns].to_csv(index=False).encode("utf-8")


def render_dark_html_table(df: pd.DataFrame, columns: list[str] | None = None) -> None:
    if df.empty:
        return
    display_columns = columns or list(df.columns)
    header_markup = "".join(f"<th>{html.escape(str(column))}</th>" for column in display_columns)
    rows_markup = []
    for row in df[display_columns].to_dict("records"):
        cells = []
        for column in display_columns:
            value = row.get(column)
            if isinstance(value, date):
                display_value = value.isoformat()
            else:
                display_value = clean_office_value(value)
            cells.append(f"<td>{html.escape(display_value)}</td>")
        rows_markup.append(f"<tr>{''.join(cells)}</tr>")
    st.markdown(
        f"""
        <div class="award-drilldown-table-wrap">
            <table class="award-drilldown-table">
                <thead><tr>{header_markup}</tr></thead>
                <tbody>{''.join(rows_markup)}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_award_drilldown_html_table(visible_df: pd.DataFrame) -> None:
    headers = [
        "Contractor",
        "Award ID",
        "Description",
        "Obligated",
        "Current Value",
        "Ceiling",
        "Remaining",
        "Contracting Office",
        "Funding Office",
        "Date",
    ]
    rows_markup = []
    for row in visible_df.to_dict("records"):
        award_id = clean_office_value(row.get("Award ID")) or "Unavailable"
        award_url = clean_office_value(row.get("USAspending"))
        if award_url:
            award_id_markup = (
                f'<a href="{html.escape(award_url)}" target="_blank" rel="noopener noreferrer">'
                f"{html.escape(award_id)} &#8599;</a>"
            )
        else:
            award_id_markup = html.escape(award_id)
        action_date = row.get("Action Date")
        action_date_text = action_date.isoformat() if isinstance(action_date, date) else clean_office_value(action_date)
        description = clean_office_value(row.get("Full Award Description"))
        row_values = [
            html.escape(clean_office_value(row.get("Contractor"))),
            award_id_markup,
            (
                f'<span title="{html.escape(description)}">'
                f"{html.escape(truncate_award_description(description))}</span>"
            ),
            html.escape(format_full_money(row.get("Obligations This Period"))),
            html.escape(format_full_money(row.get("Current Award Value"))),
            html.escape(format_full_money(row.get("Award Ceiling"))),
            html.escape(format_full_money(row.get("Remaining Ceiling"))),
            f'<span title="{html.escape(clean_office_value(row.get("Contracting Office")))}">{html.escape(clean_office_value(row.get("Contracting Office")))}</span>',
            f'<span title="{html.escape(clean_office_value(row.get("Funding Office")))}">{html.escape(clean_office_value(row.get("Funding Office")))}</span>',
            html.escape(action_date_text),
        ]
        cells = "".join(f"<td>{value}</td>" for value in row_values)
        rows_markup.append(f"<tr>{cells}</tr>")

    header_markup = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body_markup = "".join(rows_markup)
    st.markdown(
        f"""
        <div class="award-drilldown-table-wrap award-drilldown-table-wrap--scroll">
            <table class="award-drilldown-table">
                <thead><tr>{header_markup}</tr></thead>
                <tbody>{body_markup}</tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_awards_drilldown(award_df: pd.DataFrame, scope_key: str) -> dict:
    st.markdown('<div class="audit-heading">Top Awards in This Market</div>', unsafe_allow_html=True)
    if award_df.empty:
        st.info("No contract awards found for this selected market scope.")
        return {"displayed_award_count": 0, "row_count_selection": "Top 25", "rank_by": "Obligations This Period"}

    rank_options = [
        "Obligations This Period",
        "Current Award Value",
        "Award Ceiling",
        "Remaining Ceiling",
        "Most Recent Action",
    ]
    control_cols = st.columns([1, 1.5, 1.2])
    with control_cols[0]:
        row_count_label = st.selectbox(
            "Rows",
            ["Top 25", "Top 100", "All"],
            index=0,
            key=f"award-drilldown-row-count-{scope_key}",
        )
    with control_cols[1]:
        rank_by = st.selectbox(
            "Rank awards by",
            rank_options,
            index=0,
            key=f"award-drilldown-rank-by-{scope_key}",
        )
    sorted_df = sorted_award_drilldown_dataframe(award_df, rank_by)
    total_unique_awards = int(len(sorted_df))
    if row_count_label == "Top 25":
        visible_df = sorted_df.head(25)
    elif row_count_label == "Top 100":
        visible_df = sorted_df.head(100)
    else:
        visible_df = sorted_df
        if total_unique_awards > 1_000:
            st.warning(
                f"This market has {total_unique_awards:,} awards. Showing all rows may be slow. "
                "Consider downloading the CSV instead."
            )
    visible_count = int(len(visible_df))
    total_obligations = float(
        pd.to_numeric(award_df["Obligations This Period"], errors="coerce").fillna(0).sum()
    )
    st.caption(
        f"Showing top {visible_count:,} of {total_unique_awards:,} unique awards "
        f"\u00b7 {format_money(total_obligations)} obligated in this scope"
    )
    with control_cols[2]:
        st.markdown('<div class="control-button-spacer"></div>', unsafe_allow_html=True)
        st.download_button(
            "Download All Awards CSV",
            data=award_drilldown_csv_bytes(sorted_df),
            file_name="top_awards_in_this_market.csv",
            mime="text/csv",
            use_container_width=True,
        )
    render_award_drilldown_html_table(visible_df)
    return {
        "displayed_award_count": visible_count,
        "row_count_selection": row_count_label,
        "rank_by": rank_by,
    }


def active_refine_filter_application_debug(
    payload: dict,
    market_filters: dict | None,
    contracting_office: str | None,
) -> dict:
    market_filters = normalize_market_filters(market_filters)
    filters = payload.get("filters", {}) if isinstance(payload, dict) else {}
    client_side_filter = payload.get("client_side_filter", {}) if isinstance(payload, dict) else {}
    naics_code, _naics_label = decode_option(market_filters["naics_code"])
    contract_type_code, _contract_type_label = decode_option(market_filters["contract_type"])
    psc_code, _psc_label = decode_option(market_filters["psc_code"])
    set_aside_code, _set_aside_label = decode_option(market_filters["set_aside_type"])
    funding_office_code, funding_office_name = decode_option(market_filters["funding_office"])
    pop_code, _pop_label = decode_option(market_filters["pop_state"])
    contracting_office_code, contracting_office_name = decode_contracting_office(contracting_office)
    checks = {
        "naics_code": (
            naics_code == ALL_NAICS_CODES
            or naics_code in (filters.get("naics_codes", {}).get("require") or [])
        ),
        "contract_type": (
            contract_type_code == ALL_CONTRACT_TYPES
            or contract_type_code in (filters.get("contract_pricing_type_codes") or [])
        ),
        "product_service_code": (
            psc_code == ALL_PRODUCT_SERVICE_CODES
            or psc_code in (filters.get("psc_codes") or [])
        ),
        "set_aside_type": (
            set_aside_code == ALL_SET_ASIDE_TYPES
            or set_aside_code in (filters.get("set_aside_type_codes") or [])
        ),
        "contracting_office": (
            not contracting_office
            or contracting_office == ALL_CONTRACTING_OFFICES
            or client_side_filter.get("awarding_office_code") == contracting_office_code
            or client_side_filter.get("awarding_office_name") == contracting_office_name
        ),
        "funding_office": (
            market_filters["funding_office"] == ALL_FUNDING_OFFICES
            or client_side_filter.get("funding_office_code") == funding_office_code
            or client_side_filter.get("funding_office_name") == funding_office_name
        ),
        "place_of_performance": place_of_performance_filter_matches(
            pop_code,
            filters.get("place_of_performance_locations", []),
        ),
    }
    return {
        "checks": checks,
        "all_active_refine_filters_applied": all(checks.values()),
    }


AWARD_SCOPE_FIELD_ALIASES = {
    "contract_award_unique_key": [
        "contract_award_unique_key",
        "Contract Award Unique Key",
    ],
    "generated_unique_award_id": [
        "generated_unique_award_id",
        "Generated Unique Award ID",
    ],
    "generated_internal_id": [
        "generated_internal_id",
        "Generated Internal ID",
    ],
    "potential_total_value_of_award": [
        "potential_total_value_of_award",
        "Potential Total Value of Award",
        "Potential Award Amount",
    ],
    "current_total_value_of_award": [
        "current_total_value_of_award",
        "Current Total Value of Award",
        "Current Award Amount",
    ],
    "total_dollars_obligated": [
        "total_dollars_obligated",
        "total_obligated_amount",
        "Total Dollars Obligated",
        "Total Obligated Amount",
    ],
    "total_outlayed_amount_for_overall_award": [
        "total_outlayed_amount_for_overall_award",
        "outlayed_amount",
        "Total Outlayed Amount for Overall Award",
        "Outlayed Amount",
    ],
    "action_date": ["action_date", "Action Date"],
    "recipient_name": ["recipient_name", "Recipient Name"],
    "modification_number": ["modification_number", "Mod", "Modification Number"],
    "transaction_number": ["transaction_number", "Transaction Number"],
    "awarding_office_code": ["awarding_office_code", "Awarding Office Code"],
    "awarding_office_name": ["awarding_office_name", "Awarding Office Name"],
    "funding_office_code": ["funding_office_code", "Funding Office Code"],
    "funding_office_name": ["funding_office_name", "Funding Office Name"],
}


def normalize_award_scope_download_rows(rows: list[dict]) -> list[dict]:
    normalized_rows = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        normalized_row = dict(row)
        for canonical_key, aliases in AWARD_SCOPE_FIELD_ALIASES.items():
            if clean_office_value(normalized_row.get(canonical_key)):
                continue
            value = first_present(row, aliases)
            if value is not None:
                normalized_row[canonical_key] = value
        normalized_rows.append(normalized_row)
    return normalized_rows


def non_empty_field_count(rows: list[dict], key: str) -> int:
    return sum(1 for row in rows if clean_office_value(row.get(key)))


def transaction_row_field_names_found(rows: list[dict], aliases: list[str]) -> list[str]:
    found = set()
    for row in rows:
        if not isinstance(row, dict):
            continue
        for alias in aliases:
            if alias in row and clean_office_value(row.get(alias)):
                found.add(alias)
    return sorted(found)


def award_unique_key(item: dict) -> str:
    return clean_office_value(
        first_present(
            item,
            [
                "contract_award_unique_key",
                "generated_unique_award_id",
                "generated_internal_id",
            ],
        )
    )


def award_unique_key_field(item: dict) -> str:
    for key in ["contract_award_unique_key", "generated_unique_award_id", "generated_internal_id"]:
        if clean_office_value(item.get(key)):
            return key
    return ""


def award_recipient_name(item: dict) -> str:
    return clean_office_value(
        first_present(item, ["recipient_name", "Recipient Name", "recipient_name_raw"])
    ) or "Unknown Contractor"


def award_metric_value(item: dict, keys: list[str]) -> float | None:
    for key in keys:
        amount = parse_optional_currency_amount(item.get(key))
        if amount is not None:
            return amount
    return None


def award_sequence_value(item: dict) -> int:
    sequence_text = clean_office_value(
        first_present(
            item,
            [
                "modification_number",
                "Mod",
                "transaction_number",
                "parent_award_modification_number",
            ],
        )
    )
    if not sequence_text:
        return 0
    digit_groups = re.findall(r"\d+", sequence_text)
    if not digit_groups:
        return 0
    try:
        return int(digit_groups[-1])
    except ValueError:
        return 0


def latest_award_row(rows: list[dict]) -> dict:
    if not rows:
        return {}
    return sorted(
        rows,
        key=lambda row: (
            parse_action_date(first_present(row, ["action_date", "Action Date"])) or date.min,
            award_sequence_value(row),
            int(row.get("_award_scope_row_order") or 0),
        ),
    )[-1]


def award_debug_row(item: dict) -> dict:
    return {
        "row_order": int(item.get("_award_scope_row_order") or 0),
        "award_key_field": award_unique_key_field(item),
        "award_key": award_unique_key(item),
        "recipient_name": award_recipient_name(item),
        "action_date": clean_office_value(first_present(item, ["action_date", "Action Date"])),
        "modification_number": clean_office_value(first_present(item, ["modification_number", "Mod"])),
        "transaction_number": clean_office_value(item.get("transaction_number")),
        "potential_total_value_of_award": parse_currency_amount(item.get("potential_total_value_of_award")),
        "current_total_value_of_award": parse_currency_amount(item.get("current_total_value_of_award")),
        "total_dollars_obligated": parse_currency_amount(
            first_present(item, ["total_dollars_obligated", "total_obligated_amount"])
        ),
        "total_outlayed_amount_for_overall_award": parse_currency_amount(
            first_present(item, ["total_outlayed_amount_for_overall_award", "outlayed_amount"])
        ),
    }


def award_scope_dataframe(rows: list[dict]) -> tuple[pd.DataFrame, dict]:
    awards: dict[str, list[dict]] = {}
    key_field_counts = {"contract_award_unique_key": 0, "generated_unique_award_id": 0, "generated_internal_id": 0}
    for row_order, raw_row in enumerate(rows):
        if not isinstance(raw_row, dict):
            continue
        row = dict(raw_row)
        row["_award_scope_row_order"] = row_order
        key = award_unique_key(row)
        if not key:
            continue
        key_field = award_unique_key_field(row)
        if key_field in key_field_counts:
            key_field_counts[key_field] += 1
        awards.setdefault(key, []).append(row)

    award_rows = []
    debug_sample_before = []
    debug_sample_after = {}
    for key, award_rows_for_key in awards.items():
        latest_row = latest_award_row(award_rows_for_key)
        award_ceiling = parse_currency_amount(latest_row.get("potential_total_value_of_award"))
        current_award_value = parse_currency_amount(latest_row.get("current_total_value_of_award"))
        lifetime_obligated = parse_currency_amount(
            first_present(latest_row, ["total_dollars_obligated", "total_obligated_amount"])
        )
        reported_outlays = parse_currency_amount(
            first_present(latest_row, ["total_outlayed_amount_for_overall_award", "outlayed_amount"])
        )
        remaining_ceiling = max(award_ceiling - current_award_value, 0.0)
        if not debug_sample_before or len(award_rows_for_key) > len(debug_sample_before):
            debug_sample_before = [award_debug_row(row) for row in award_rows_for_key[:5]]
            debug_sample_after = {
                "award_key": key,
                "raw_rows_for_award": len(award_rows_for_key),
                "selected_latest_row": award_debug_row(latest_row),
                "deduped_award_scope_record": {
                    "recipient": award_recipient_name(latest_row),
                    "award_ceiling": award_ceiling,
                    "current_award_value": current_award_value,
                    "lifetime_obligated": lifetime_obligated,
                    "reported_outlays": reported_outlays,
                    "remaining_ceiling": remaining_ceiling,
                },
            }
        award_rows.append(
            {
                "award_key": key,
                "recipient": award_recipient_name(latest_row),
                "award_ceiling": award_ceiling,
                "current_award_value": current_award_value,
                "lifetime_obligated": lifetime_obligated,
                "reported_outlays": reported_outlays,
                "remaining_ceiling": remaining_ceiling,
            }
        )

    selected_key_field = next(
        (key for key in ["contract_award_unique_key", "generated_unique_award_id", "generated_internal_id"] if key_field_counts[key] > 0),
        "",
    )
    debug_payload = {
        "raw_transaction_rows": len(rows),
        "unique_awards": len(awards),
        "selected_unique_key_field": selected_key_field,
        "unique_key_field_counts": key_field_counts,
        "sample_award_before_dedupe": debug_sample_before,
        "sample_award_after_dedupe": debug_sample_after,
    }
    return pd.DataFrame(
        award_rows,
        columns=[
            "award_key",
            "recipient",
            "award_ceiling",
            "current_award_value",
            "lifetime_obligated",
            "reported_outlays",
            "remaining_ceiling",
        ],
    ), debug_payload


def award_scope_totals(award_df: pd.DataFrame) -> dict:
    if award_df.empty:
        return {
            "active_award_ceiling": 0.0,
            "current_award_value": 0.0,
            "lifetime_obligated": 0.0,
            "remaining_ceiling": 0.0,
            "reported_outlays": 0.0,
            "award_count": 0,
        }
    return {
        "active_award_ceiling": float(pd.to_numeric(award_df["award_ceiling"], errors="coerce").fillna(0).sum()),
        "current_award_value": float(pd.to_numeric(award_df["current_award_value"], errors="coerce").fillna(0).sum()),
        "lifetime_obligated": float(pd.to_numeric(award_df["lifetime_obligated"], errors="coerce").fillna(0).sum()),
        "remaining_ceiling": float(pd.to_numeric(award_df["remaining_ceiling"], errors="coerce").fillna(0).sum()),
        "reported_outlays": float(pd.to_numeric(award_df["reported_outlays"], errors="coerce").fillna(0).sum()),
        "award_count": int(len(award_df)),
    }


def award_scope_vendor_dataframe(award_df: pd.DataFrame, metric_column: str) -> pd.DataFrame:
    if award_df.empty or metric_column not in award_df.columns:
        return pd.DataFrame(columns=["recipient", "amount"])
    return (
        award_df.groupby("recipient", as_index=False)[metric_column]
        .sum()
        .rename(columns={metric_column: "amount"})
        .sort_values("amount", ascending=False)
        .head(10)
    )


def trend_time_series_dataframe(trend_df: pd.DataFrame) -> pd.DataFrame:
    if trend_df.empty:
        return pd.DataFrame(columns=["bucket_label", "amount", "transaction_count"])
    chart_df = trend_df.copy().sort_values("fiscal_year")
    chart_df["bucket_label"] = chart_df["fiscal_year"].apply(fiscal_year_label)
    chart_df["transaction_count"] = None
    return chart_df.rename(columns={"amount": "amount"})[
        ["bucket_label", "amount", "transaction_count"]
    ]


def transaction_obligations_time_series(transaction_df: pd.DataFrame, time_grain: str) -> pd.DataFrame:
    if transaction_df.empty or "Action Date" not in transaction_df.columns:
        return pd.DataFrame(columns=["bucket_label", "amount", "transaction_count"])
    scoped_df = transaction_df.copy()
    scoped_df["Action Date"] = scoped_df["Action Date"].apply(parse_action_date)
    scoped_df = scoped_df.dropna(subset=["Action Date"])
    if scoped_df.empty:
        return pd.DataFrame(columns=["bucket_label", "amount", "transaction_count"])
    scoped_df["bucket_label"] = scoped_df["Action Date"].apply(lambda value: time_bucket_label(value, time_grain))
    scoped_df["sort_fiscal_year"] = scoped_df["Action Date"].apply(
        lambda value: fiscal_period_sort_key(value, time_grain)[0]
    )
    scoped_df["sort_period"] = scoped_df["Action Date"].apply(
        lambda value: fiscal_period_sort_key(value, time_grain)[1]
    )
    grouped = (
        scoped_df.groupby(["bucket_label", "sort_fiscal_year", "sort_period"], as_index=False)
        .agg(amount=("Obligation Amount", "sum"), transaction_count=("Obligation Amount", "size"))
        .sort_values(["sort_fiscal_year", "sort_period"])
    )
    return grouped[["bucket_label", "amount", "transaction_count"]]


def hide_sidebar_for_landing() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def mark_analysis_started(
    active_agency: str,
    selected_bureau: str,
    selected_year: int,
    selected_contracting_office: str,
    selected_market_filters: dict,
) -> None:
    """Copy pending sidebar selections (active_*) into analyzed_* dashboard state."""
    st.session_state.dashboard_started = True
    st.session_state.analysis_loading_requested = True
    st.session_state.analyzed_agency = active_agency
    st.session_state.analyzed_bureau = canonical_bureau_name(selected_bureau)
    st.session_state.analyzed_year = int(selected_year)
    st.session_state.analyzed_contracting_office = selected_contracting_office
    st.session_state.analyzed_market_filters = normalize_market_filters(selected_market_filters)
    st.session_state.analyzed_office_option_stats = st.session_state.get(
        "pending_office_option_stats",
        {},
    )
    st.session_state.analysis_run_started_at = utc_analysis_timestamp()
    st.session_state.pop("analysis_run_completed_at", None)


def clear_all_cached_dashboard_data() -> None:
    st.cache_data.clear()
    st.session_state.transaction_office_options_cache = {}
    st.session_state.pending_office_option_stats = {}
    st.session_state.analyzed_agency = None
    st.session_state.analyzed_bureau = None
    st.session_state.analyzed_year = None
    st.session_state.analyzed_contracting_office = ALL_CONTRACTING_OFFICES
    st.session_state.analyzed_market_filters = default_market_filters()
    st.session_state.analyzed_office_option_stats = {}
    st.session_state.analysis_loading_requested = False
    st.session_state.cache_cleared_notice = True


def default_time_grain(selected_year: int) -> str:
    if int(selected_year) == current_fiscal_year():
        return TIME_GRAIN_MONTH
    return TIME_GRAIN_FISCAL_YEAR


def time_grain_scope_key(
    active_agency: str,
    selected_bureau: str | None,
    selected_year: int,
    selected_contracting_office: str,
    selected_market_filters: dict,
) -> str:
    return hashlib.sha256(
        json.dumps(
            {
                "agency": active_agency,
                "bureau": selected_bureau or ALL_BUREAUS,
                "fiscal_year": int(selected_year),
                "contracting_office": selected_contracting_office,
                "market_filters": normalize_market_filters(selected_market_filters),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()[:12]


def render_workflow_or_divider() -> None:
    st.markdown('<div class="workflow-or-divider">OR</div>', unsafe_allow_html=True)


def render_solicitation_workflow_cta() -> None:
    st.markdown(
        """
        <div class="workflow-cta-title">Have a solicitation in hand?</div>
        <div class="workflow-cta-subtitle">Upload the solicitation package and GovCon Agency Trends will extract the market scope, identify amendments, and prepare confirmed filters for your review.</div>
        <div class="workflow-cta-helper">Upload the base solicitation, amendments, PWS/SOW, Section L/M, pricing files, Q&amp;A, and related attachments. Multiple files are supported.</div>
        """,
        unsafe_allow_html=True,
    )


def _valid_completed_resolved_payload(payload: object) -> bool:
    if not isinstance(payload, dict):
        return False
    if payload.get("version") != RESOLVED_SIGNALS_VERSION:
        return False
    signals = payload.get("signals")
    if not isinstance(signals, list) or not signals:
        return False
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    requested = int(summary.get("requestedSignalCount") or 0)
    if requested:
        countable = [item for item in signals if isinstance(item, dict) and item.get("id") != "process_sources_v1"]
        return len(countable) >= requested
    return True


def _latest_completed_resolved_artifact() -> tuple[str, Path, dict] | None:
    runs_dir = get_data_dir() / "runs"
    if not runs_dir.exists():
        return None
    candidates: list[tuple[float, Path]] = []
    for path in runs_dir.glob("run_*/signals/resolved_signals.json"):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue
    for _modified, path in sorted(candidates, reverse=True):
        try:
            payload = read_json(path)
        except Exception:
            continue
        if _valid_completed_resolved_payload(payload):
            return path.parents[1].name, path, payload
    return None


def _apply_loaded_resolved_signals(payload: dict, *, source_label: str) -> None:
    st.session_state.solicitation_resolved_signals = payload
    st.session_state.solicitation_scope_preview = build_solicitation_scope_preview(payload)
    st.session_state.solicitation_user_filter_overrides = {}
    st.session_state.solicitation_removed_filter_fields = []
    st.session_state.solicitation_loaded_run_id = payload.get("runId")
    st.session_state.solicitation_signals_source = source_label
    st.session_state.solicitation_signals_loaded_at = utc_analysis_timestamp()
    st.session_state.pop("solicitation_mapping_result", None)
    st.session_state.solicitation_extraction_complete = True
    st.session_state.solicitation_extraction_status = "completed"
    st.session_state.solicitation_mapping_status = "not_started"
    st.session_state.solicitation_review_status = "ready"
    st.session_state.solicitation_scope_applied = False
    st.session_state.solicitation_scope_review_open = False
    st.session_state.solicitation_comparable_market = False
    st.session_state.pop("solicitation_saved_contracting_office", None)


def render_upload_solicitation_package_panel() -> None:
    uploaded_files = st.file_uploader(
        "Upload Solicitation Package",
        type=["pdf", "docx", "xlsx", "xls", "csv", "txt"],
        accept_multiple_files=True,
        key="solicitation_package_uploader",
        help="Upload the base solicitation, amendments, PWS/SOW, Section L/M, pricing files, Q&A, and related attachments.",
    )

    staged = validate_and_stage_uploads(uploaded_files or [])
    load_project_env()
    extraction_mode = get_extraction_mode()
    from extraction.config import get_extraction_profile

    extraction_profile = get_extraction_profile()
    st.session_state.solicitation_engine_diagnostics = {
        "mode": extraction_mode,
        "profile": extraction_profile,
    }

    if not uploaded_files and not st.session_state.get("solicitation_resolved_signals"):
        recovered_artifact = _latest_completed_resolved_artifact()
        if recovered_artifact:
            run_id, resolved_path, _payload = recovered_artifact
            payload = read_json(resolved_path)
            _apply_loaded_resolved_signals(payload, source_label="completed_run_resolved_signals_json")
            st.session_state.solicitation_extraction_result = {
                "summary": {
                    "runId": run_id,
                    "recoveredExistingArtifact": True,
                    "resolvedSignalsJson": str(resolved_path),
                },
                "success": True,
                "errorMessage": None,
            }
            st.session_state.solicitation_extraction_complete = True
            st.session_state.solicitation_extraction_status = "completed"

    if uploaded_files:
        st.caption(f"{len(uploaded_files)} files selected")

    current_fingerprint = st.session_state.get("solicitation_upload_fingerprint")
    files_changed = bool(staged.fingerprint and staged.fingerprint != current_fingerprint)

    extract_clicked = st.button(
        "Extract Solicitation",
        type="primary",
        use_container_width=True,
        key="extract_solicitation_btn",
    )
    should_extract = bool(extract_clicked and staged.paths)

    if should_extract and files_changed:
        if any(item.level == "error" for item in staged.findings):
            for finding in staged.findings:
                if finding.level == "error":
                    details = finding.details or {}
                    filename = details.get("filename") or details.get("path") or ""
                    st.error(f"{finding.message} ({filename})" if filename else finding.message)
        else:
            st.session_state.solicitation_extraction_status = "running"
            st.session_state.solicitation_mapping_status = "not_started"
            st.session_state.solicitation_review_status = "not_ready"
            st.session_state.solicitation_extraction_transition_rerun_done = False
            progress = st.progress(0.0, text="Saving solicitation files")
            status = st.empty()
            progress_detail: dict = {"label": "Saving solicitation files", "pct": 0.0}

            def _progress(label: str, pct: float) -> None:
                progress_detail["label"] = label
                progress_detail["pct"] = pct
                progress.progress(min(max(pct, 0.0), 1.0), text=label)
                render_extraction_loading_card(status, progress_detail)

            os.environ.pop("GOVCON_FORCE_PACKAGE_REFRESH", None)
            render_extraction_loading_card(status, progress_detail)
            result = run_extraction_from_paths(
                staged.paths,
                progress=_progress,
                progress_detail=progress_detail,
            )

            st.session_state.solicitation_extraction_result = {
                "summary": result.summary,
                "success": result.success,
                "errorMessage": result.error_message,
            }
            st.session_state.solicitation_extraction_diagnostics = [
                item.to_dict() for item in result.findings if item.level in {"warn", "error", "warning"}
            ]
            st.session_state.solicitation_upload_fingerprint = staged.fingerprint
            st.session_state.solicitation_upload_staged_paths = [str(path) for path in staged.paths]

            if result.resolved_signals:
                _apply_loaded_resolved_signals(result.resolved_signals, source_label="pipeline_full_extraction")
                st.session_state.solicitation_extraction_complete = result.success
                st.session_state.solicitation_extraction_status = "completed" if result.success else "failed"
            else:
                st.session_state.solicitation_extraction_complete = False
                st.session_state.solicitation_extraction_status = "failed"

            cleanup_staged_paths(staged.paths)
            progress.progress(1.0, text="Market scope extracted successfully")
            status.empty()
            progress.empty()
            if not st.session_state.get("solicitation_extraction_transition_rerun_done"):
                st.session_state.solicitation_extraction_transition_rerun_done = True
                st.rerun()

    extraction_result = st.session_state.get("solicitation_extraction_result")
    if extraction_result and not extraction_result.get("success"):
        summary = extraction_result.get("summary") or {}
        st.warning(extraction_result.get("errorMessage") or "Extraction completed with issues. Review warnings below.")
        excluded_files = summary.get("excludedFiles") or []
        for item in excluded_files[:6]:
            message = str(item.get("message") or "File excluded")
            details = item.get("details") or {}
            filename = details.get("filename") or details.get("path") or ""
            st.warning(f"{message} - {filename}" if filename else message)
        if should_show_blocking_ocr_error(summary, st.session_state.get("solicitation_extraction_diagnostics") or []):
            st.error("A controlling document could not be read. Required market-scope values may be incomplete.")


def _render_resolved_signals_upload_content(*, show_test_mode_note: bool = True) -> None:
    if show_test_mode_note:
        st.caption("For testing or re-loading a previously generated resolved_signals.json artifact.")
    uploaded = st.file_uploader(
        "resolved_signals.json",
        type=["json"],
        key="solicitation_signals_uploader",
    )
    dev_path = st.text_input(
        "Local debug path (optional)",
        value=st.session_state.get("solicitation_dev_path", ""),
        placeholder="C:\\path\\to\\resolved_signals.json",
        key="solicitation_dev_path_input",
    )
    load_clicked = st.button("Load Signals", key="load_solicitation_signals", use_container_width=True)
    clear_clicked = st.button("Clear Loaded Signals", key="clear_solicitation_signals", use_container_width=True)

    if clear_clicked:
        for key in (
            "solicitation_resolved_signals",
            "solicitation_scope_preview",
            "solicitation_mapping_result",
            "solicitation_user_filter_overrides",
            "solicitation_loaded_run_id",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    if load_clicked:
        try:
            source = uploaded
            if source is None and dev_path.strip():
                source = dev_path.strip()
                st.session_state.solicitation_dev_path = dev_path.strip()
            payload = load_resolved_signals_json(source)
            _apply_loaded_resolved_signals(
                payload,
                source_label="uploaded_resolved_signals_json" if uploaded is not None else "local_resolved_signals_json_path",
            )
            st.success(f"Loaded {len(payload.get('signals') or [])} signals.")
            st.rerun()
        except Exception as exc:
            st.error(str(exc))


def render_load_solicitation_signals_panel(*, show_test_mode_note: bool = True) -> None:
    with st.expander("Advanced: Upload Existing resolved_signals.json", expanded=False):
        _render_resolved_signals_upload_content(show_test_mode_note=show_test_mode_note)


def _solicitation_user_override(user_overrides: dict[str, str], field_name: str) -> str | None:
    legacy_names = {
        "Agency": "Mapped Agency",
        "Subagency / Bureau": "Mapped Subagency / Bureau",
        "Contracting Office": "Mapped Contracting Office",
    }
    selected = user_overrides.get(field_name)
    if selected:
        return selected
    legacy = legacy_names.get(field_name)
    return user_overrides.get(legacy) if legacy else None


def _solicitation_truncate(value: object | None, max_length: int = 180) -> str:
    text = str(value or "").strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


SOLICITATION_MANUAL_FILTER_SPECS = [
    ("Agency", "agency"),
    ("Subagency / Bureau", "bureau"),
    ("Contracting Office", "contracting_office"),
    ("Funding Office", "funding_office"),
    ("NAICS", "naics_code"),
    ("PSC", "psc_code"),
    ("Contract Type", "contract_type"),
    ("Set-Aside", "set_aside_type"),
    ("Place of Performance", "pop_state"),
]

SOLICITATION_OPTIONAL_FILTER_SPECS = [
    ("Contracting Office", "contracting_office"),
    ("Funding Office", "funding_office"),
    ("NAICS", "naics_code"),
    ("PSC", "psc_code"),
    ("Contract Type", "contract_type"),
    ("Set-Aside", "set_aside_type"),
    ("Place of Performance", "pop_state"),
]

SOLICITATION_FILTER_SIGNAL_IDS = {
    "rfp_solicitation_number_v1",
    "rfp_solicitation_id_v1",
    "rfp_issuing_agency_v1",
    "rfp_issuing_office_v1",
    "rfp_office_aac_v1",
    "rfp_funding_office_v1",
    "rfp_funding_office_name_v1",
    "rfp_primary_naics_v1",
    "rfp_primary_psc_v1",
    "rfp_contract_type_v1",
    "rfp_set_aside_v1",
    "rfp_competition_type_v1",
    "rfp_place_of_performance_v1",
}

SOLICITATION_MAPPING_BASIS_LABELS = {
    "deterministic_hierarchy": "hierarchy match",
    "exact_normalized": "exact normalized name",
    "known_alias": "known alias",
    "fuzzy_org_name": "fuzzy organization name",
    "exact_subagency": "exact subagency",
    "fuzzy_subagency": "fuzzy subagency",
    "all_bureaus": "all bureaus default",
    "office_aac_code": "exact office code",
    "issuing_office_name": "fuzzy office name",
    "funding_office_code": "exact funding office code",
    "funding_office_name_exact": "exact funding office name",
    "funding_office_name_fuzzy": "fuzzy funding office name",
    "exact_code": "exact code",
    "contract_type_alias": "contract type alias",
    "set_aside_alias": "set-aside alias",
    "parse_us_state": "parsed US state",
    "default_all_bureaus": "default all bureaus",
}


def _solicitation_mapping_basis(field_name: str, mapping_attempts: dict) -> str:
    attempts = mapping_attempts.get(field_name) or []
    if not attempts:
        return ""
    strategy = str(attempts[0].get("strategy") or "").strip()
    return SOLICITATION_MAPPING_BASIS_LABELS.get(strategy, strategy.replace("_", " "))


def _solicitation_removed_fields() -> set[str]:
    raw = st.session_state.get("solicitation_removed_filter_fields") or []
    return {str(item) for item in raw}


def _solicitation_set_removed_fields(fields: set[str]) -> None:
    st.session_state.solicitation_removed_filter_fields = sorted(fields)


def _solicitation_requires_user_confirmation(row: dict) -> bool:
    return row.get("filter_key") in SOLICITATION_USER_CONFIRMATION_FILTER_KEYS


def _solicitation_pop_parsed_display(extracted_value: object | None) -> str:
    state_code, state_name = parse_pop_state_from_text(extracted_value)
    if not state_code:
        return ""
    label = state_name or STATE_OPTIONS.get(state_code, state_code)
    return f"{state_code} — {label}"


def is_solicitation_confirmed_filter(row: dict, available_filter_options: dict) -> bool:
    filter_key = row.get("filter_key")
    if not filter_key or not row.get("filter_option"):
        return False
    if filter_key in SOLICITATION_USER_CONFIRMATION_FILTER_KEYS:
        return False
    if row.get("mapping_status") != "Exact match" or not row.get("preselect"):
        return False
    if not _solicitation_option_is_valid(filter_key, row["filter_option"], available_filter_options):
        return False
    validation_status = str(row.get("validation_status") or "").strip().lower()
    evidence_validated = bool(str(row.get("evidence_snippet") or "").strip())
    if validation_status not in SOLICITATION_CONFIRMED_VALIDATION_STATUSES:
        return False
    if not evidence_validated:
        return False
    confidence_high = str(row.get("confidence") or "").strip().lower() == "high"
    if row.get("deterministic_mapping"):
        return confidence_high
    return confidence_high


def _solicitation_workflow_status(
    row: dict,
    user_overrides: dict[str, str],
    available_filter_options: dict,
) -> str:
    if row.get("mapping_unavailable") or row.get("mapping_status") == SOLICITATION_MAPPING_UNAVAILABLE_STATUS:
        return SOLICITATION_MAPPING_UNAVAILABLE_STATUS
    if row.get("mapping_status") == "Context only" or not row.get("filter_key"):
        return "Context only"
    field_name = row["field"]
    selected = _solicitation_user_override(user_overrides, field_name)
    if _solicitation_requires_user_confirmation(row):
        if (
            selected
            and selected != KEEP_CURRENT_SOLICITATION_FILTER
            and _solicitation_option_is_valid(row.get("filter_key"), selected, available_filter_options)
        ):
            return "Manually selected"
        if row.get("filter_option"):
            return "Suggested / needs confirmation"
        return "Unmapped"
    confirmed = is_solicitation_confirmed_filter(row, available_filter_options)
    if (
        selected
        and selected != KEEP_CURRENT_SOLICITATION_FILTER
        and _solicitation_option_is_valid(row.get("filter_key"), selected, available_filter_options)
    ):
        if confirmed and selected == row.get("filter_option"):
            return "Suggested from solicitation"
        return "Manually selected"
    if confirmed:
        return "Suggested from solicitation"
    if not row.get("filter_option"):
        return "Unmapped"
    if row.get("mapping_status") == "Suggested match" or str(row.get("confidence") or "").lower() in (
        "medium",
        "low",
    ):
        return "Suggested / needs confirmation"
    return "Unmapped"


def _solicitation_status_badge_html(status: str) -> str:
    css_class = {
        "Suggested": "solicitation-status-suggested",
        "Confirmed by analyst": "solicitation-status-auto",
        "Edited by analyst": "solicitation-status-manual",
        "Needs review": "solicitation-status-confirm",
        "Not found in uploaded package": "solicitation-status-unmapped",
        "Not available in dashboard scope": "solicitation-status-unmapped",
        "Excluded by analyst": "solicitation-status-removed",
        "Suggested from solicitation": "solicitation-status-suggested",
        "Suggested / needs confirmation": "solicitation-status-confirm",
        "Manually selected": "solicitation-status-manual",
        "Removed by user": "solicitation-status-removed",
        "Unmapped": "solicitation-status-unmapped",
        "Context only": "solicitation-status-context",
        "Exact match": "solicitation-status-exact",
        "Suggested match": "solicitation-status-suggested",
        "Requires confirmation": "solicitation-status-confirm",
        SOLICITATION_MAPPING_UNAVAILABLE_STATUS: "solicitation-status-unmapped",
    }.get(status, "solicitation-status-context")
    safe_status = html.escape(status)
    return f'<span class="solicitation-status-badge {css_class}">{safe_status}</span>'


def _solicitation_mapped_display_value(
    row: dict,
    user_overrides: dict[str, str],
    available_filter_options: dict,
) -> str:
    field_name = row["field"]
    filter_key = row.get("filter_key")
    selected = _solicitation_user_override(user_overrides, field_name)
    if selected and selected != KEEP_CURRENT_SOLICITATION_FILTER:
        if _solicitation_option_is_valid(filter_key, selected, available_filter_options):
            return _format_solicitation_filter_option(row, selected)
    if row.get("filter_option"):
        return _format_solicitation_filter_option(row, row["filter_option"])
    return str(row.get("unmapped_extracted_display") or row.get("mapped_filter_display") or "—")


SOLICITATION_OPPORTUNITY_CONTEXT_SIGNAL_SPECS = [
    ("Solicitation Number", ["rfp_solicitation_number_v1", "rfp_solicitation_id_v1"]),
    ("Description", ["rfp_description_v1"]),
    ("Incumbent", ["rfp_incumbent_data_v1"]),
    ("Period of Performance", ["rfp_period_of_performance_v1"]),
    ("POP Start", ["rfp_pop_start_v1"]),
    ("POP End", ["rfp_pop_end_v1"]),
    ("POP Years", ["rfp_pop_years_v1"]),
    ("Primary POC", ["rfp_primary_poc_v1"]),
    ("Contracting Officer (KO)", ["rfp_ko_name_v1"]),
    ("Contract Specialist", ["rfp_contract_specialist_name_v1"]),
    ("Evaluation method", ["rfp_eval_method_v1"]),
    ("Evaluation weights", ["rfp_eval_weights_v1"]),
    ("Past performance requirements", ["rfp_past_perf_reqs_v1"]),
    ("Pricing constraints", ["rfp_pricing_constraints_v1"]),
    ("Fee constraints", ["rfp_fee_constraints_v1"]),
    ("Key personnel", ["rfp_key_personnel_v1"]),
    ("Clearances", ["rfp_clearances_v1"]),
    ("Competition type", ["rfp_competition_type_v1"]),
    ("Requirement name", ["rfp_requirement_name_v1"]),
    ("Title", ["rfp_title_v1"]),
    ("Strategic intent", ["rfp_strategic_intent_v1"]),
    ("Submission destination", ["rfp_submission_destination_v1"]),
    ("Submission format", ["rfp_submission_format_v1"]),
    ("Submission instructions", ["rfp_submission_instructions_v1"]),
    ("Submission method", ["rfp_submission_method_v1"]),
    ("Questions due", ["rfp_questions_due_v1"]),
    ("Prior contract PIID", ["rfp_prior_contract_piid_v1"]),
    ("Technical factors", ["rfp_tech_factors_v1"]),
]


def build_solicitation_opportunity_context_rows(
    resolved_signals: dict,
    scope_preview: dict,
) -> list[dict]:
    rows: list[dict] = []
    seen_fields: set[str] = set()
    for field_name, signal_ids in SOLICITATION_OPPORTUNITY_CONTEXT_SIGNAL_SPECS:
        value, confidence, signal_id, signal = first_signal_value(resolved_signals, signal_ids)
        if value is None:
            field_data = scope_preview.get(field_name, {})
            if field_data.get("value") is None:
                continue
            value = field_data.get("value")
            confidence = field_data.get("confidence")
            evidence_snippet = field_data.get("evidence_snippet", "")
            evidence_source = field_data.get("evidence_source", "")
            evidence_locator = field_data.get("evidence_locator", "")
        else:
            evidence_details = signal_evidence_details(signal)
            evidence_snippet = signal_evidence_snippet(signal)
            evidence_source = evidence_details.get("source", "")
            evidence_locator = evidence_details.get("locator", "")
        if field_name in seen_fields:
            continue
        seen_fields.add(field_name)
        rows.append(
            {
                "field": field_name,
                "extracted_value": value,
                "confidence": confidence,
                "signal_id": signal_id,
                "evidence_snippet": evidence_snippet,
                "evidence_source": evidence_source,
                "evidence_locator": evidence_locator,
            }
        )
    return rows


def _solicitation_rows_by_field(rows: list[dict]) -> dict[str, dict]:
    by_field: dict[str, dict] = {}
    for row in rows:
        field_name = row.get("field")
        if not field_name or field_name in by_field:
            continue
        if row.get("filter_key") or row.get("mapping_status") == "Context only":
            by_field[field_name] = row
    return by_field


def _solicitation_is_suggested_filter(row: dict, available_filter_options: dict) -> bool:
    if not row.get("filter_key"):
        return False
    if is_solicitation_confirmed_filter(row, available_filter_options):
        return False
    if row.get("filter_option"):
        return True
    if row.get("mapping_status") == "Suggested match":
        return True
    return str(row.get("confidence") or "").strip().lower() in ("medium", "low")


def _solicitation_manual_review_rows(
    rows: list[dict],
    available_filter_options: dict,
    funding_office_extracted: bool,
) -> list[dict]:
    rows_by_field = _solicitation_rows_by_field(rows)
    manual_rows: list[dict] = []
    for field_name, filter_key in SOLICITATION_MANUAL_FILTER_SPECS:
        row = rows_by_field.get(field_name)
        if row and is_solicitation_confirmed_filter(row, available_filter_options):
            continue
        if row and _solicitation_is_suggested_filter(row, available_filter_options):
            continue
        if field_name == "Funding Office" and not funding_office_extracted and not row:
            manual_rows.append(
                {
                    "field": field_name,
                    "filter_key": filter_key,
                    "extracted_value": None,
                    "filter_option": None,
                    "mapping_status": "Unmapped",
                    "confidence": None,
                    "evidence_snippet": "",
                }
            )
            continue
        if row:
            manual_rows.append(row)
        else:
            manual_rows.append(
                {
                    "field": field_name,
                    "filter_key": filter_key,
                    "extracted_value": None,
                    "filter_option": None,
                    "mapping_status": "Unmapped",
                    "confidence": None,
                    "evidence_snippet": "",
                }
            )
    return manual_rows


def _solicitation_summary_title(resolved_signals: dict) -> str:
    for signal_ids in (["rfp_title_v1"], ["rfp_requirement_name_v1"]):
        value, _, _, _ = first_signal_value(resolved_signals, signal_ids)
        if value is not None:
            return _solicitation_truncate(value, 120)
    return ""


def _solicitation_loaded_context_line(resolved_signals: dict, scope_preview: dict) -> str:
    solicitation_data = scope_preview.get("Solicitation Number", {})
    solicitation_number = str(solicitation_data.get("value") or "").strip()
    if not solicitation_number:
        value, _, _, _ = first_signal_value(
            resolved_signals,
            ["rfp_solicitation_number_v1", "rfp_solicitation_id_v1"],
        )
        solicitation_number = str(value or "").strip()
    title = _solicitation_summary_title(resolved_signals)
    if solicitation_number and title:
        return f"Solicitation loaded: {solicitation_number} — {title}"
    if solicitation_number:
        return f"Solicitation loaded: {solicitation_number}"
    if title:
        return f"Solicitation loaded: {title}"
    return "Solicitation loaded"


def _solicitation_summary_specs(
    scope_preview: dict,
    rows_by_field: dict[str, dict],
) -> tuple[list[tuple[str, str, bool]], object | None]:
    org_row = rows_by_field.get("Extracted Organization")
    extracted_org = org_row.get("extracted_value") if org_row else scope_preview.get("Issuing Agency", {}).get("value")
    specs: list[tuple[str, str, bool]] = [
        ("Solicitation number", "Solicitation Number", False),
        ("Extracted organization", "", False),
        ("Mapped Agency", "Agency", True),
        ("Mapped Subagency / Bureau", "Subagency / Bureau", True),
        ("Contracting Office", "Contracting Office", True),
        ("Funding Office", "Funding Office", True),
        ("NAICS", "NAICS", True),
        ("PSC", "PSC", True),
        ("Contract Type", "Contract Type", True),
        ("Set-Aside", "Set-Aside", True),
        ("Place of Performance", "Place of Performance", True),
        ("Incumbent", "Incumbent Data", False),
    ]
    return specs, extracted_org


def _solicitation_confirmed_display_rows(
    rows: list[dict],
    available_filter_options: dict,
    user_overrides: dict[str, str],
    removed_fields: set[str],
) -> list[tuple[dict, str]]:
    rows_by_field = _solicitation_rows_by_field(rows)
    display_rows: list[tuple[dict, str]] = []
    for row in rows:
        field_name = row.get("field") or ""
        if is_solicitation_filter_removed(field_name, removed_fields):
            continue
        if row.get("filter_key") and is_solicitation_confirmed_filter(row, available_filter_options):
            display_rows.append((row, "Suggested from solicitation"))
    pop_row = rows_by_field.get("Place of Performance")
    if pop_row:
        selected = _solicitation_user_override(user_overrides, "Place of Performance")
        if (
            selected
            and selected != KEEP_CURRENT_SOLICITATION_FILTER
            and _solicitation_option_is_valid("pop_state", selected, available_filter_options)
        ):
            display_rows.append(
                (
                    {
                        **pop_row,
                        "filter_option": selected,
                        "mapped_filter_display": _format_solicitation_filter_option(pop_row, selected),
                    },
                    "Manually selected",
                )
            )
    return display_rows


def _solicitation_optional_filter_rows(
    rows: list[dict],
    available_filter_options: dict,
    *,
    removed_fields: set[str],
    funding_office_extracted: bool,
) -> list[dict]:
    return build_solicitation_additional_filter_rows(
        rows,
        removed_fields=removed_fields,
        funding_office_extracted=funding_office_extracted,
        is_confirmed_filter=is_solicitation_confirmed_filter,
        requires_user_confirmation=_solicitation_requires_user_confirmation,
        available_filter_options=available_filter_options,
    )


def _solicitation_optional_filter_hint(
    row: dict,
    available_filter_options: dict,
    *,
    funding_office_extracted: bool,
) -> str:
    field_name = row["field"]
    if row.get("mapping_unavailable"):
        extracted = _solicitation_truncate(row.get("extracted_value"), 160)
        if extracted:
            return f"Extracted: {extracted}. Dashboard option matching is temporarily unavailable."
        return "Dashboard option matching is temporarily unavailable."
    if is_solicitation_confirmed_filter(row, available_filter_options):
        return ""
    if field_name == "Funding Office":
        if not funding_office_extracted and row.get("extracted_value") is None:
            return "No funding office was extracted. Select one manually only if you know it applies."
        if row.get("filter_option") and _solicitation_is_suggested_filter(row, available_filter_options):
            return (
                f"Suggested: {_format_solicitation_filter_option(row, row['filter_option'])} "
                "— select to apply."
            )
    if field_name == "PSC":
        extracted_code = extract_category_code(row.get("extracted_value")) or str(
            row.get("unmapped_extracted_display") or ""
        ).strip()
        if extracted_code and not row.get("filter_option"):
            return (
                f"Extracted PSC: {extracted_code}. Not applied because it was not found "
                "in the available scoped PSC options."
            )
    if field_name == "Place of Performance":
        parsed_pop = _solicitation_pop_parsed_display(row.get("extracted_value"))
        if parsed_pop:
            return (
                f"Place of Performance parsed as {parsed_pop}. Analyst confirmation is required because "
                "solicitations may include multiple or remote performance locations."
            )
    if row.get("filter_option") and _solicitation_is_suggested_filter(row, available_filter_options):
        return (
            f"Suggested: {_format_solicitation_filter_option(row, row['filter_option'])} "
            "— select to apply."
        )
    if row.get("extracted_value") is not None and not row.get("filter_option"):
        return f"Extracted: {_solicitation_truncate(row.get('extracted_value'), 160)}. Analyst confirmation is required."
    return ""


def _solicitation_validation_display(row: dict) -> str:
    status = str(row.get("validation_status") or "").strip().lower()
    if status in SOLICITATION_CONFIRMED_VALIDATION_STATUSES:
        return "confirmed"
    if status:
        return "needs review"
    return "not found"


def _solicitation_row_has_evidence(row: dict) -> bool:
    return bool(str(row.get("evidence_snippet") or "").strip())


def _solicitation_review_session_key(prefix: str, field_name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", field_name.lower()).strip("_")
    return f"solicitation_scope_{prefix}_{slug}"


def _solicitation_selected_review_value(row: dict, available_filter_options: dict) -> str | None:
    field_name = row["field"]
    filter_key = row.get("filter_key")
    if not filter_key:
        return None
    suggested = row.get("filter_option")
    options = _solicitation_filter_options_for_row(row, available_filter_options)
    placeholder = row.get("unmapped_extracted_display") if row.get("is_hybrid") else None
    select_key = _solicitation_review_session_key("value", field_name)
    stored = st.session_state.get(select_key)
    if stored == placeholder or stored in {KEEP_CURRENT_SOLICITATION_FILTER, "Select...", "Select manually"}:
        return None
    if stored in options:
        return stored
    if filter_key == "contract_type":
        return None
    if suggested in options:
        return suggested
    return None


def _solicitation_initialize_widget_state(key: str, value: object) -> None:
    if key not in st.session_state:
        st.session_state[key] = value


def _solicitation_confidence_label(row: dict) -> str:
    confidence = str(row.get("confidence") or "").strip().lower()
    if confidence == "high":
        return "High confidence"
    if confidence == "medium":
        return "Medium confidence"
    if confidence == "low":
        return "Low confidence"
    return "Confidence unavailable"


def _solicitation_status_line(item: dict) -> str:
    row = item["row"]
    status = _solicitation_decision_status_label(item.get("decision_status") or "needs_review")
    if item.get("decision_status") == "suggested":
        status = "Suggested"
    if item.get("decision_status") == "not_available_in_dashboard_scope":
        return "Extracted - mapping temporarily unavailable"
    if item.get("decision_status") == "not_found_in_uploaded_package":
        return "Not found in uploaded package"
    return f"{_solicitation_confidence_label(row)} · {status}"


def _solicitation_decision_status_label(status: str) -> str:
    return {
        "suggested": "Suggested",
        "confirmed_by_analyst": "Confirmed by analyst",
        "edited_by_analyst": "Edited by analyst",
        "needs_review": "Needs review",
        "not_found_in_uploaded_package": "Not found in uploaded package",
        "not_available_in_dashboard_scope": "Not available in dashboard scope",
        "excluded_by_analyst": "Excluded by analyst",
    }.get(status, "Needs review")


def _solicitation_audit_field_key(row: dict) -> str:
    filter_key = row.get("filter_key")
    if filter_key:
        return filter_key
    return re.sub(r"[^a-z0-9]+", "_", str(row.get("field") or "").lower()).strip("_")


def _solicitation_review_group(row: dict, preselected: bool, missing: bool, mapping_unavailable: bool) -> str:
    if missing or row.get("field") == "Funding Office":
        return "manual"
    if preselected and not mapping_unavailable:
        return "suggested"
    return "review"


def _contract_type_filter_source(
    field_name: str,
    analyst_selected: bool,
    selected_value: str | None,
) -> str | None:
    if field_name != "Contract Type":
        return None
    assert None in {"analyst_selection", "existing_pending_state", None}
    if analyst_selected and selected_value:
        return "analyst_selection"
    active_contract_type = normalize_market_filters(st.session_state.get("active_market_filters") or {}).get("contract_type")
    if active_contract_type and active_contract_type != ALL_CONTRACT_TYPES:
        return "existing_pending_state"
    return None


def _preselection_decision_for_row(row: dict, preselected: bool) -> tuple[bool, str]:
    field_name = row.get("field")
    if field_name == "Contract Type":
        return False, "Contract Type is analyst-controlled and never AI-preselected."
    if field_name in {"Place of Performance", "Funding Office"}:
        return False, f"{field_name} requires analyst confirmation."
    if preselected:
        return True, "Exact valid dashboard option with confirmed evidence."
    if row.get("rejection_reason"):
        return False, row["rejection_reason"]
    if row.get("filter_option"):
        return False, "Valid option available but analyst confirmation required."
    return False, "No exact valid dashboard option was preselected."


def _solicitation_review_rows(
    rows: list[dict],
    scope_preview: dict,
    available_filter_options: dict,
) -> list[dict]:
    rows_by_field = _solicitation_rows_by_field(rows)
    review_rows: list[dict] = []
    seen: set[str] = set()
    for field_name, filter_key in SOLICITATION_MARKET_REVIEW_FIELDS:
        if field_name in seen:
            continue
        seen.add(field_name)
        if field_name == "AAC":
            aac_data = scope_preview.get("Office AAC", {})
            value = aac_data.get("value")
            row = {
                "field": "AAC",
                "filter_key": None,
                "extracted_value": value,
                "filter_option": None,
                "mapped_filter_display": str(value or ""),
                "mapping_status": "Context only" if value is not None else "Unmapped",
                "preselect": False,
                "confidence": aac_data.get("confidence"),
                "evidence_snippet": aac_data.get("evidence_snippet", ""),
                "evidence_source": aac_data.get("evidence_source", ""),
                "evidence_locator": aac_data.get("evidence_locator", ""),
                "validation_status": aac_data.get("validation_status", ""),
            }
        else:
            row = rows_by_field.get(field_name)
            if not row:
                row = {
                    "field": field_name,
                    "filter_key": filter_key,
                    "extracted_value": None,
                    "filter_option": None,
                    "mapped_filter_display": "",
                    "mapping_status": "Unmapped",
                    "preselect": False,
                    "confidence": None,
                    "evidence_snippet": "",
                    "evidence_source": "",
                    "evidence_locator": "",
                    "validation_status": "",
                }
        preselected = is_solicitation_confirmed_filter(row, available_filter_options)
        if field_name in {"Place of Performance", "Funding Office", "Contract Type"}:
            preselected = False
        check_key = _solicitation_review_session_key("use", field_name)
        if check_key not in st.session_state:
            st.session_state[check_key] = bool(preselected)
        select_key = _solicitation_review_session_key("value", field_name)
        if select_key not in st.session_state and row.get("filter_option") and field_name != "Contract Type":
            options = _solicitation_filter_options_for_row(row, available_filter_options) if row.get("filter_key") else []
            if row.get("filter_option") in options:
                st.session_state[select_key] = row.get("filter_option")
        selected_value = _solicitation_selected_review_value(row, available_filter_options)
        analyst_selected = bool(st.session_state.get(check_key)) and bool(selected_value)
        replacement = None
        analyst_edited_fields = st.session_state.get("solicitation_analyst_edited_fields") or set()
        analyst_edited = field_name in analyst_edited_fields
        if analyst_selected and selected_value and row.get("filter_option") and selected_value != row.get("filter_option"):
            replacement = selected_value
        missing = row.get("extracted_value") is None and not row.get("filter_option")
        mapping_unavailable = bool(row.get("mapping_unavailable")) or row.get("mapping_status") == SOLICITATION_MAPPING_UNAVAILABLE_STATUS
        if missing:
            status = "not_found_in_uploaded_package"
        elif mapping_unavailable:
            status = "not_available_in_dashboard_scope"
        elif not row.get("filter_key"):
            status = "needs_review"
        elif not analyst_selected:
            status = "excluded_by_analyst"
        elif replacement or (analyst_selected and analyst_edited):
            status = "edited_by_analyst"
        elif preselected:
            status = "suggested"
        else:
            status = "confirmed_by_analyst"
        contract_type_source = _contract_type_filter_source(field_name, analyst_selected, selected_value)
        preselection_allowed, preselection_reason = _preselection_decision_for_row(row, bool(preselected))
        review_rows.append(
            {
                "row": row,
                "field": field_name,
                "field_key": _solicitation_audit_field_key(row),
                "filter_key": row.get("filter_key"),
                "preselected": bool(preselected),
                "analyst_selected": bool(analyst_selected),
                "analyst_replacement": replacement,
                "selected_value": selected_value if analyst_selected else None,
                "decision_status": status,
                "group": _solicitation_review_group(row, preselected, missing, mapping_unavailable),
                "contract_type_filter_source": contract_type_source,
                "preselection_allowed": preselection_allowed,
                "preselection_reason": preselection_reason,
            }
        )
    return review_rows


def _solicitation_review_counts(review_rows: list[dict]) -> dict[str, int]:
    return {
        "selected": sum(1 for item in review_rows if item.get("analyst_selected")),
        "review": sum(1 for item in review_rows if item.get("group") == "review"),
        "manual": sum(1 for item in review_rows if item.get("group") == "manual"),
    }


def _solicitation_source_completeness(review_rows: list[dict]) -> tuple[str, list[str]]:
    by_field = {item["field"]: item for item in review_rows}
    missing_critical = [
        field
        for field in SOLICITATION_SOURCE_COMPLETENESS_CRITICAL_FIELDS
        if by_field.get(field, {}).get("group") == "manual"
    ]
    agency_ok = bool(by_field.get("Agency", {}).get("row", {}).get("extracted_value")) or bool(
        by_field.get("Subagency / Bureau", {}).get("row", {}).get("extracted_value")
    )
    available_fields = sum(1 for item in review_rows if item.get("row", {}).get("extracted_value"))
    if not agency_ok and available_fields < 3:
        return "Insufficient market scope found", missing_critical
    if missing_critical:
        return "Partial market scope found", missing_critical
    return "Complete market scope found", missing_critical


def _solicitation_evidence_expander(row: dict) -> None:
    if not _solicitation_row_has_evidence(row):
        st.caption("—")
        return
    with st.expander("Evidence diagnostics", expanded=False):
        st.caption(f"Filename: {row.get('evidence_source') or '—'}")
        st.caption(f"Page or sheet: {row.get('evidence_locator') or '—'}")
        st.caption(f"Quote: {_solicitation_truncate(row.get('evidence_snippet'), 300)}")
        st.caption(f"Confidence: {str(row.get('confidence') or '—').title()}")
        st.caption(f"Validation status: {_solicitation_validation_display(row)}")


def _solicitation_display_value_for_item(item: dict) -> str:
    row = item["row"]
    selected = item.get("selected_value")
    if selected:
        return _format_solicitation_filter_option(row, selected)
    if row.get("filter_option"):
        return _format_solicitation_filter_option(row, row["filter_option"])
    value = row.get("mapped_filter_display") or row.get("extracted_value")
    return str(value or "Not found")


def _render_scope_review_text(text: str, *, tone: str = "muted", target=None) -> None:
    css_class = "scope-review-summary" if tone == "secondary" else "scope-review-helper"
    renderer = target or st
    renderer.markdown(f'<div class="{css_class}">{html.escape(text)}</div>', unsafe_allow_html=True)


def _solicitation_manual_options_for_item(item: dict, available_filter_options: dict) -> list[str]:
    row = item["row"]
    options = _solicitation_filter_options_for_row(row, available_filter_options) if row.get("filter_key") else []
    return [option for option in options if not str(option).lower().startswith("all ")]


def _solicitation_select_placeholder_for_item(item: dict) -> str:
    row = item["row"]
    if row.get("filter_key") == "contract_type":
        active_contract_type = normalize_market_filters(st.session_state.get("active_market_filters") or {}).get("contract_type")
        if active_contract_type and active_contract_type != ALL_CONTRACT_TYPES:
            return KEEP_CURRENT_SOLICITATION_FILTER
        return "Select manually"
    if row.get("filter_option") or row.get("extracted_value"):
        return "Select..."
    return KEEP_CURRENT_SOLICITATION_FILTER


def _solicitation_dropdown_options_for_item(item: dict, available_filter_options: dict) -> list[str]:
    row = item["row"]
    options = _solicitation_manual_options_for_item(item, available_filter_options)
    placeholder = _solicitation_select_placeholder_for_item(item)
    selected = st.session_state.get(_solicitation_review_session_key("value", item["field"]))
    if selected and selected not in options and selected != placeholder:
        selected = None
    if row.get("filter_option") and row.get("filter_option") in options and not selected:
        selected = row.get("filter_option")
    if selected in options:
        return [placeholder, *options]
    return [placeholder, *options]


def _render_compact_solicitation_review_row(item: dict, available_filter_options: dict) -> None:
    row = item["row"]
    field_name = item["field"]
    if field_name == "AAC":
        return
    check_key = _solicitation_review_session_key("use", field_name)
    select_key = _solicitation_review_session_key("value", field_name)
    is_manual = item.get("group") == "manual"
    has_value = bool(row.get("filter_option") or row.get("extracted_value"))
    use_col, label_col, value_col = st.columns([0.08, 0.30, 0.62])
    with label_col:
        st.markdown(f"**{field_name}**")
    if is_manual:
        options = _solicitation_dropdown_options_for_item(item, available_filter_options)
        if len(options) > 1:
            if select_key not in st.session_state:
                st.session_state[select_key] = options[0]
            with value_col:
                selected = st.selectbox(
                    f"{field_name} manual value",
                    options,
                    format_func=lambda option, mapping_row=row: _format_solicitation_filter_option(mapping_row, option),
                    key=select_key,
                    label_visibility="collapsed",
                )
                if selected == options[0]:
                    _render_scope_review_text("Not found in uploaded package", target=value_col)
                else:
                    st.session_state[check_key] = True
            return
        with value_col:
            st.markdown("Not found in uploaded package")
            _render_scope_review_text("Needs manual selection", target=value_col)
        return
    _solicitation_initialize_widget_state(check_key, bool(st.session_state.get(check_key, item.get("preselected"))))
    with use_col:
        st.checkbox(
            f"Use {field_name}",
            key=check_key,
            disabled=not has_value,
            label_visibility="collapsed",
        )
    with value_col:
        options = _solicitation_dropdown_options_for_item(item, available_filter_options)
        if len(options) > 1 and row.get("filter_key"):
            if select_key not in st.session_state:
                suggested = None if row.get("filter_key") == "contract_type" else row.get("filter_option")
                st.session_state[select_key] = suggested if suggested in options else options[0]
            selected = st.selectbox(
                f"{field_name} value",
                options,
                format_func=lambda option, mapping_row=row: _format_solicitation_filter_option(mapping_row, option),
                key=select_key,
                label_visibility="collapsed",
            )
            if selected != options[0] and selected != row.get("filter_option"):
                st.session_state.setdefault("solicitation_analyst_edited_fields", set()).add(field_name)
        else:
            st.markdown(_solicitation_display_value_for_item(item))
        if item.get("group") == "review":
            _render_scope_review_text("Review before applying", target=value_col)
        elif item.get("decision_status") == "not_available_in_dashboard_scope":
            _render_scope_review_text("Mapping unavailable", target=value_col)


def _render_compact_solicitation_review(
    review_rows: list[dict],
    available_filter_options: dict,
) -> None:
    suggested_rows = [
        item
        for item in review_rows
        if item.get("group") in {"suggested", "review"} and item.get("field") != "AAC"
    ]
    manual_rows = [
        item
        for item in review_rows
        if item.get("group") == "manual" and item.get("field") not in {"AAC"}
    ]
    with st.container(border=True):
        if suggested_rows:
            st.markdown("#### Suggested filters")
            for item in suggested_rows:
                _render_compact_solicitation_review_row(item, available_filter_options)
        if manual_rows:
            missing_labels = [item["field"] for item in manual_rows if not _solicitation_manual_options_for_item(item, available_filter_options)]
            if missing_labels:
                _render_scope_review_text("Missing: " + ", ".join(missing_labels))
            st.markdown("#### Additional filters")
            for item in manual_rows:
                _render_compact_solicitation_review_row(item, available_filter_options)


def _render_solicitation_review_group(
    title: str,
    items: list[dict],
    available_filter_options: dict,
) -> None:
    st.markdown(f"#### {title} ({len(items)})")
    for item in items:
        _render_compact_solicitation_review_row(item, available_filter_options)


def _render_solicitation_confirmation_summary(review_rows: list[dict]) -> None:
    counts = _solicitation_review_counts(review_rows)
    _render_scope_review_text(f"{counts['selected']} filters selected", tone="secondary")


def _build_solicitation_audit_state(review_rows: list[dict]) -> list[dict]:
    audit_rows = []
    for item in review_rows:
        row = item["row"]
        final_value = item.get("selected_value") if item.get("analyst_selected") else None
        audit_rows.append(
            {
                "field_key": item.get("field_key"),
                "field": item.get("field"),
                "raw_extracted_value": row.get("extracted_value"),
                "validated_value": row.get("extracted_value"),
                "resolved_value": row.get("mapped_filter_display") or row.get("filter_option"),
                "hierarchy_match_level": row.get("hierarchy_match_level"),
                "mapped_dashboard_value": row.get("filter_option"),
                "preselection_allowed": bool(item.get("preselection_allowed")),
                "preselection_reason": item.get("preselection_reason"),
                "analyst_value": item.get("selected_value") if item.get("analyst_selected") else None,
                "extracted_value": row.get("extracted_value"),
                "mapped_value": row.get("filter_option"),
                "validation_status": _solicitation_validation_display(row),
                "confidence": str(row.get("confidence") or "").lower() or None,
                "evidence": {
                    "filename": row.get("evidence_source") or "",
                    "page_or_sheet": row.get("evidence_locator") or "",
                    "quote": row.get("evidence_snippet") or "",
                },
                "preselected": bool(item.get("preselected")),
                "analyst_selected": bool(item.get("analyst_selected")),
                "analyst_replacement": item.get("analyst_replacement")
                or (final_value if item.get("decision_status") == "edited_by_analyst" else None),
                "final_pending_value": final_value,
                "decision_status": item.get("decision_status"),
                "contract_type_filter_source": item.get("contract_type_filter_source"),
            }
        )
    return audit_rows


def _build_pending_filters_from_review_rows(review_rows: list[dict], available_filter_options: dict) -> dict:
    pending_sidebar_filters: dict = {
        "agency": None,
        "bureau": None,
        "contracting_office": None,
        "market_filters": {},
    }
    for item in review_rows:
        if not item.get("analyst_selected"):
            continue
        row = item["row"]
        filter_key = row.get("filter_key")
        selected = item.get("selected_value")
        if filter_key == "contract_type":
            source = item.get("contract_type_filter_source")
            assert source in {"analyst_selection", "existing_pending_state", None}
            if source != "analyst_selection":
                st.session_state.setdefault("solicitation_invariant_diagnostics", []).append(
                    {
                        "invariant": "contract_type_final_source_not_ai",
                        "status": "blocked",
                        "contract_type_filter_source": source,
                    }
                )
                continue
        if not filter_key or not _solicitation_option_is_valid(filter_key, selected, available_filter_options):
            continue
        _apply_pending_filter_selection(pending_sidebar_filters, filter_key, selected)
    sanitized, diagnostics = _sanitize_review_pending_filters(pending_sidebar_filters, available_filter_options)
    if diagnostics:
        st.session_state.setdefault("solicitation_invariant_diagnostics", []).extend(diagnostics)
    return sanitized


def _render_solicitation_detail_summary(
    resolved_signals: dict,
    scope_preview: dict,
    rows: list[dict],
    user_overrides: dict[str, str],
    available_filter_options: dict,
    extracted_org: object | None,
) -> None:
    rows_by_field = _solicitation_rows_by_field(rows)
    st.markdown("**Solicitation overview**")
    overview_rows = []
    solicitation_data = scope_preview.get("Solicitation Number", {})
    solicitation_number = str(solicitation_data.get("value") or "—")
    title_value = _solicitation_summary_title(resolved_signals) or "—"
    overview_rows.append(("Solicitation number", solicitation_number))
    overview_rows.append(("Title", title_value))
    overview_rows.append(("Extracted organization", str(extracted_org or "—")))
    for label, value in overview_rows:
        st.caption(f"{label}: {value}")

    st.markdown("**Mapped summary**")
    body_rows = []
    summary_items, _ = _solicitation_summary_specs(scope_preview, rows_by_field)

    def append_summary_row(label: str, display_value: str, status: str) -> None:
        body_rows.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f"<td>{html.escape(_solicitation_truncate(display_value, 120))}</td>"
            f"<td>{_solicitation_status_badge_html(status)}</td>"
            "</tr>"
        )

    for label, field_key, is_filter in summary_items:
        if label == "Extracted organization":
            display_value = str(extracted_org or "—")
            append_summary_row(label, display_value, "Context only")
            continue
        if label == "Solicitation number":
            field_data = scope_preview.get("Solicitation Number", {})
            display_value = str(field_data.get("value") or "—")
            append_summary_row(label, display_value, "Context only")
            title_value = _solicitation_summary_title(resolved_signals)
            if title_value:
                append_summary_row("Title", title_value, "Context only")
            continue
        if not is_filter:
            field_data = scope_preview.get(field_key, {})
            display_value = _solicitation_truncate(field_data.get("value"), 80) or "—"
            append_summary_row(label, display_value, "Context only")
            continue
        row = rows_by_field.get(field_key)
        if not row:
            if field_key == "Funding Office":
                append_summary_row(label, "No funding office extracted", "Unmapped")
            else:
                append_summary_row(label, "No value extracted", "Unmapped")
            continue
        status = _solicitation_workflow_status(row, user_overrides, available_filter_options)
        if field_key == "Place of Performance":
            parsed_text = _solicitation_pop_parsed_display(row.get("extracted_value"))
            display_value = parsed_text or _solicitation_truncate(row.get("extracted_value"), 80) or "—"
        else:
            display_value = _solicitation_mapped_display_value(row, user_overrides, available_filter_options)
        if field_key == "Funding Office" and row.get("extracted_value") is None:
            display_value = "No funding office extracted"
        append_summary_row(label, display_value, status)

    st.markdown(
        '<table class="solicitation-preview-table">'
        "<thead><tr><th>Field</th><th>Value</th><th>Status</th></tr></thead><tbody>"
        + "".join(body_rows)
        + "</tbody></table>",
        unsafe_allow_html=True,
    )


def _render_solicitation_filters_found(
    rows: list[dict],
    available_filter_options: dict,
    user_overrides: dict[str, str],
    removed_fields: set[str],
) -> None:
    confirmed_rows = _solicitation_confirmed_display_rows(
        rows,
        available_filter_options,
        user_overrides,
        removed_fields,
    )
    st.markdown("#### Filters Found in Solicitation")
    st.caption(
        "These exact matches were found in the solicitation and matched to valid dashboard filters."
    )
    if not confirmed_rows:
        st.caption("No exact matches were available as solicitation suggestions.")
        return
    header = st.columns([2.0, 3.0, 2.2, 0.35])
    header[0].markdown("**Filter**")
    header[1].markdown("**Matched dashboard value**")
    header[2].markdown("**Status**")
    header[3].markdown("")
    for row, status_label in confirmed_rows:
        field_name = row["field"]
        cols = st.columns([2.0, 3.0, 2.2, 0.35])
        cols[0].markdown(field_name)
        cols[1].markdown(_format_solicitation_filter_option(row, row["filter_option"]))
        cols[2].markdown(_solicitation_status_badge_html(status_label), unsafe_allow_html=True)
        if cols[3].button(
            "×",
            key=f"remove_solicitation_filter_{row.get('filter_key') or field_name}",
            help="Remove this filter from the pending solicitation scope",
        ):
            updated_removed = set(removed_fields)
            updated_removed.add(field_name)
            _solicitation_set_removed_fields(updated_removed)
            user_overrides[field_name] = KEEP_REMOVED_SOLICITATION_FILTER
            st.rerun()
    office_row = next((row for row, _status in confirmed_rows if row.get("field") == "Contracting Office"), None)
    if office_row:
        office_code, _office_name = decode_contracting_office(office_row.get("filter_option") or "")
        if office_code:
            st.caption(
                f"Contracting Office filters transactions by awarding office code {office_code}."
            )


def _render_solicitation_filter_dropdown(
    row: dict,
    user_overrides: dict[str, str],
    available_filter_options: dict,
    *,
    key_prefix: str,
    default_selection: str | None = None,
    include_suggested_default: bool = False,
    removed_fields: set[str] | None = None,
) -> None:
    field_name = row["field"]
    removed_fields = removed_fields or set()
    options = [KEEP_CURRENT_SOLICITATION_FILTER]
    if is_solicitation_filter_removed(field_name, removed_fields):
        options = [KEEP_REMOVED_SOLICITATION_FILTER, KEEP_CURRENT_SOLICITATION_FILTER]
    options.extend(_solicitation_filter_options_for_row(row, available_filter_options))
    selected_override = _solicitation_user_override(user_overrides, field_name)
    if selected_override and selected_override in options:
        default_selection = selected_override
    elif is_solicitation_filter_removed(field_name, removed_fields):
        default_selection = KEEP_REMOVED_SOLICITATION_FILTER
    elif include_suggested_default and row.get("filter_option") and row["filter_option"] in options:
        default_selection = row["filter_option"]
    elif not default_selection or default_selection not in options:
        default_selection = KEEP_CURRENT_SOLICITATION_FILTER
    selected = st.selectbox(
        field_name,
        options,
        index=options.index(default_selection),
        format_func=lambda option, mapping_row=row: _format_solicitation_filter_option(mapping_row, option),
        key=f"{key_prefix}_{row.get('filter_key') or field_name}",
        label_visibility="collapsed",
    )
    user_overrides[field_name] = selected


def _render_solicitation_optional_filters(
    rows: list[dict],
    user_overrides: dict[str, str],
    available_filter_options: dict,
    *,
    funding_office_extracted: bool,
    removed_fields: set[str],
) -> None:
    optional_rows = _solicitation_optional_filter_rows(
        rows,
        available_filter_options,
        removed_fields=removed_fields,
        funding_office_extracted=funding_office_extracted,
    )
    if not optional_rows:
        st.caption("All extracted market filters are represented in the review groups.")
        return
    st.markdown("#### Want to add any additional filters?")
    st.caption(
        "Only unresolved, review-only, removed, or manual fields appear here."
    )
    for row in optional_rows:
        field_name = row["field"]
        st.markdown(f"**{field_name}**")
        hint = _solicitation_optional_filter_hint(
            row,
            available_filter_options,
            funding_office_extracted=funding_office_extracted,
        )
        if hint:
            st.caption(hint)
        if is_solicitation_filter_removed(field_name, removed_fields) and row.get("filter_option"):
            if st.button(
                "Restore extracted value",
                key=f"restore_solicitation_filter_{row.get('filter_key') or field_name}",
            ):
                updated_removed = set(removed_fields)
                updated_removed.discard(field_name)
                _solicitation_set_removed_fields(updated_removed)
                user_overrides[field_name] = row["filter_option"]
                st.rerun()
        _render_solicitation_filter_dropdown(
            row,
            user_overrides,
            available_filter_options,
            key_prefix="solicitation_optional",
            default_selection=(
                KEEP_REMOVED_SOLICITATION_FILTER
                if is_solicitation_filter_removed(field_name, removed_fields)
                else KEEP_CURRENT_SOLICITATION_FILTER
            ),
            removed_fields=removed_fields,
        )


def _render_solicitation_full_details_content(
    resolved_signals: dict,
    scope_preview: dict,
    rows: list[dict],
    context_rows: list[dict],
    user_overrides: dict[str, str],
    available_filter_options: dict,
    extracted_org: object | None,
    *,
    run_id: str | None = None,
) -> None:
    if True:
        if run_id:
            st.caption(f"Run ID: {run_id}")
        summary = (resolved_signals.get("summary") or {}) if isinstance(resolved_signals, dict) else {}
        if summary.get("profile") == "market_scope_fast" or summary.get("extractionProfile") == "market_scope_fast":
            detail_rows = build_fast_scope_detail_rows(resolved_signals)
            display_rows = []
            for item in detail_rows:
                value = item.get("value")
                if isinstance(value, dict):
                    value = None
                display_rows.append(
                    {
                        "Signal": item.get("label"),
                        "Value": _solicitation_truncate(value, 240) if value is not None else "—",
                        "Confidence": item.get("confidence") or "",
                        "Validation": item.get("validation_status") or "",
                        "Source": item.get("source") or "",
                        "Page": item.get("page") or "",
                        "Quote": _solicitation_truncate(item.get("quote"), 220),
                    }
                )
            st.caption(f"Fast market-scope profile ({fast_scope_requested_count(resolved_signals)} signals)")
            if display_rows:
                st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
            else:
                st.caption("No fast-scope signals available.")
            return
        _render_solicitation_detail_summary(
            resolved_signals,
            scope_preview,
            rows,
            user_overrides,
            available_filter_options,
            extracted_org,
        )
        st.markdown("**Extracted signals**")
        if not context_rows:
            st.caption("No extracted solicitation signals were available.")
        else:
            signal_rows = [
                {
                    "Field": row["field"],
                    "Extracted value": _solicitation_truncate(row.get("extracted_value"), 400),
                    "Confidence": row.get("confidence") or "",
                    "Evidence": _solicitation_truncate(row.get("evidence_snippet"), 240),
                }
                for row in context_rows
            ]
            st.dataframe(pd.DataFrame(signal_rows), use_container_width=True, hide_index=True)

        evidence_rows = []
        seen_fields: set[str] = set()
        for row in [*rows, *context_rows]:
            field_name = row.get("field")
            if not field_name or field_name in seen_fields:
                continue
            snippet = str(row.get("evidence_snippet") or "").strip()
            if not snippet:
                field_data = scope_preview.get(field_name, {})
                snippet = str(field_data.get("evidence_snippet") or "").strip()
            if not snippet:
                continue
            seen_fields.add(field_name)
            evidence_rows.append(
                {
                    "Field": field_name,
                    "Extracted value": _solicitation_truncate(row.get("extracted_value"), 160),
                    "Confidence": row.get("confidence") or "",
                    "Source document": row.get("evidence_source")
                    or scope_preview.get(field_name, {}).get("evidence_source", ""),
                    "Locator / page": row.get("evidence_locator")
                    or scope_preview.get(field_name, {}).get("evidence_locator", ""),
                    "Snippet": _solicitation_truncate(snippet, 320),
                }
            )
        if evidence_rows:
            st.markdown("**Signal evidence**")
            st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)


def _solicitation_filter_options_for_row(row: dict, available_filter_options: dict) -> list[str]:
    filter_key = row.get("filter_key")
    if filter_key == "agency":
        return available_filter_options["agencies"]
    if filter_key == "bureau":
        return available_filter_options.get("bureaus", [ALL_BUREAUS])
    if filter_key == "contracting_office":
        return available_filter_options["contracting_offices"]
    if filter_key == "funding_office":
        return available_filter_options.get("funding_offices", [ALL_FUNDING_OFFICES])
    if filter_key == "naics_code":
        return available_filter_options["naics"]
    if filter_key == "psc_code":
        return available_filter_options["psc"]
    if filter_key == "contract_type":
        return available_filter_options["contract_types"]
    if filter_key == "set_aside_type":
        return available_filter_options["set_asides"]
    if filter_key == "pop_state":
        return available_filter_options["pop_locations"]
    return []


def _format_solicitation_filter_option(row: dict, option: str) -> str:
    if option in {KEEP_CURRENT_SOLICITATION_FILTER, KEEP_REMOVED_SOLICITATION_FILTER}:
        return option
    filter_key = row.get("filter_key")
    if filter_key == "agency":
        return option
    if filter_key == "bureau":
        return option
    if filter_key == "contracting_office":
        return format_contracting_office_option(option)
    if filter_key == "funding_office":
        return format_funding_office_option(option)
    if filter_key == "pop_state":
        return option if option in (ALL_POP_LOCATIONS, LEGACY_ALL_POP_STATES) else format_code_description_option(option)
    if filter_key in ("naics_code", "psc_code", "contract_type", "set_aside_type"):
        defaults = {
            "naics_code": ALL_NAICS_CODES,
            "psc_code": ALL_PRODUCT_SERVICE_CODES,
            "contract_type": ALL_CONTRACT_TYPES,
            "set_aside_type": ALL_SET_ASIDE_TYPES,
        }
        default_option = defaults[filter_key]
        return option if option == default_option else format_code_description_option(option)
    return option


def _solicitation_option_is_valid(
    filter_key: str | None,
    selected: str | None,
    available_filter_options: dict,
) -> bool:
    if not selected or selected in {KEEP_CURRENT_SOLICITATION_FILTER, KEEP_REMOVED_SOLICITATION_FILTER}:
        return False
    option_map = {
        "agency": "agencies",
        "bureau": "bureaus",
        "contracting_office": "contracting_offices",
        "funding_office": "funding_offices",
        "naics_code": "naics",
        "psc_code": "psc",
        "contract_type": "contract_types",
        "set_aside_type": "set_asides",
        "pop_state": "pop_locations",
    }
    options_key = option_map.get(filter_key or "")
    if not options_key:
        return False
    return selected in available_filter_options.get(options_key, [])


def _apply_pending_filter_selection(
    pending_sidebar_filters: dict,
    filter_key: str,
    selected: str,
) -> None:
    if filter_key == "agency":
        pending_sidebar_filters["agency"] = selected
    elif filter_key == "bureau":
        pending_sidebar_filters["bureau"] = selected
    elif filter_key == "contracting_office":
        pending_sidebar_filters["contracting_office"] = selected
    elif filter_key == "funding_office":
        pending_sidebar_filters["market_filters"]["funding_office"] = selected
    else:
        pending_sidebar_filters["market_filters"][filter_key] = selected


def _selected_is_hidden_broad_default(filter_key: str | None, selected: str | None) -> bool:
    if not selected:
        return False
    broad_defaults = {
        "bureau": ALL_BUREAUS,
        "contracting_office": ALL_CONTRACTING_OFFICES,
        "funding_office": ALL_FUNDING_OFFICES,
        "naics_code": ALL_NAICS_CODES,
        "psc_code": ALL_PRODUCT_SERVICE_CODES,
        "contract_type": ALL_CONTRACT_TYPES,
        "set_aside_type": ALL_SET_ASIDE_TYPES,
        "pop_state": ALL_POP_LOCATIONS,
    }
    return selected == broad_defaults.get(filter_key or "")


def _sanitize_review_pending_filters(
    pending_sidebar_filters: dict,
    available_filter_options: dict,
) -> tuple[dict, list[dict]]:
    sanitized = {
        "agency": pending_sidebar_filters.get("agency"),
        "bureau": pending_sidebar_filters.get("bureau"),
        "contracting_office": pending_sidebar_filters.get("contracting_office"),
        "market_filters": dict(pending_sidebar_filters.get("market_filters") or {}),
    }
    diagnostics: list[dict] = []
    agency = sanitized.get("agency")
    bureau = sanitized.get("bureau")
    if agency and bureau and bureau != ALL_BUREAUS and bureau not in available_filter_options.get("bureaus", []):
        diagnostics.append(
            {
                "invariant": "subagency_parent_child_validation",
                "status": "blocked",
                "agency": agency,
                "subagency": bureau,
                "reason": "subagency_value is not in valid_subagencies_for(agency_value)",
            }
        )
        sanitized["bureau"] = None
    if sanitized.get("agency") and sanitized.get("bureau") == sanitized.get("agency"):
        diagnostics.append(
            {
                "invariant": "known_subagency_cannot_populate_agency",
                "status": "blocked",
                "agency": sanitized.get("agency"),
                "subagency": sanitized.get("bureau"),
            }
        )
        sanitized["agency"] = None
        sanitized["bureau"] = None
    for filter_key, selected in [
        ("bureau", sanitized.get("bureau")),
        ("contracting_office", sanitized.get("contracting_office")),
        *list((sanitized.get("market_filters") or {}).items()),
    ]:
        if _selected_is_hidden_broad_default(filter_key, selected):
            diagnostics.append(
                {
                    "invariant": "no_failed_match_uses_broad_fallback",
                    "status": "blocked",
                    "filter_key": filter_key,
                    "selected": selected,
                    "reason": "Broad defaults cannot be introduced by solicitation mapping.",
                }
            )
            if filter_key == "bureau":
                sanitized["bureau"] = None
            elif filter_key == "contracting_office":
                sanitized["contracting_office"] = None
            else:
                sanitized["market_filters"].pop(filter_key, None)
    return sanitized, diagnostics


def _build_confirmed_pending_filters(
    rows: list[dict],
    available_filter_options: dict,
    removed_fields: set[str] | None = None,
) -> dict:
    """Apply only high-confidence exact/validated filters; never include suggested or manual picks."""
    removed_fields = removed_fields or set()
    pending_sidebar_filters: dict = {
        "agency": None,
        "bureau": None,
        "contracting_office": None,
        "market_filters": {},
    }
    for row in rows:
        filter_key = row.get("filter_key")
        if not filter_key or not is_solicitation_confirmed_filter(row, available_filter_options):
            continue
        if filter_key == "contract_type":
            continue
        if is_solicitation_filter_removed(row.get("field") or "", removed_fields):
            continue
        selected = row["filter_option"]
        if not _solicitation_option_is_valid(filter_key, selected, available_filter_options):
            continue
        _apply_pending_filter_selection(pending_sidebar_filters, filter_key, selected)
    sanitized, diagnostics = _sanitize_review_pending_filters(pending_sidebar_filters, available_filter_options)
    if diagnostics:
        st.session_state.setdefault("solicitation_invariant_diagnostics", []).extend(diagnostics)
    return sanitized


def _build_reviewed_pending_filters(
    rows: list[dict],
    user_overrides: dict[str, str],
    available_filter_options: dict,
    removed_fields: set[str] | None = None,
) -> dict:
    """Apply confirmed filters plus explicit user-reviewed optional filter selections."""
    removed_fields = removed_fields or set()
    pending_sidebar_filters: dict = {
        "agency": None,
        "bureau": None,
        "contracting_office": None,
        "market_filters": {},
    }
    for row in rows:
        filter_key = row.get("filter_key")
        field_name = row.get("field") or ""
        if not filter_key:
            continue
        user_override = _solicitation_user_override(user_overrides, field_name)
        selected = resolve_solicitation_filter_pending_value(
            row,
            removed_fields=removed_fields,
            user_override=user_override,
            keep_current_token=KEEP_CURRENT_SOLICITATION_FILTER,
            keep_removed_token=KEEP_REMOVED_SOLICITATION_FILTER,
            is_confirmed_filter=is_solicitation_confirmed_filter,
            option_is_valid=_solicitation_option_is_valid,
            available_filter_options=available_filter_options,
        )
        if filter_key == "contract_type" and not (
            user_override
            and user_override not in {KEEP_CURRENT_SOLICITATION_FILTER, KEEP_REMOVED_SOLICITATION_FILTER}
            and _solicitation_option_is_valid(filter_key, user_override, available_filter_options)
        ):
            continue
        if not selected:
            continue
        _apply_pending_filter_selection(pending_sidebar_filters, filter_key, selected)
    sanitized, diagnostics = _sanitize_review_pending_filters(pending_sidebar_filters, available_filter_options)
    if diagnostics:
        st.session_state.setdefault("solicitation_invariant_diagnostics", []).extend(diagnostics)
    return sanitized


def _should_show_solicitation_scope_review() -> bool:
    return solicitation_scope_review_visible(
        has_resolved_signals=bool(st.session_state.get("solicitation_resolved_signals")),
        scope_applied=bool(st.session_state.get("solicitation_scope_applied")),
        review_open=bool(st.session_state.get("solicitation_scope_review_open")),
    )


def _solicitation_extraction_elapsed_sec() -> float | None:
    summary = (st.session_state.get("solicitation_extraction_result") or {}).get("summary") or {}
    elapsed = summary.get("elapsedSec")
    return float(elapsed) if elapsed is not None else None


def _mark_solicitation_scope_applied(
    reviewed_pending_filters: dict,
    *,
    mapping_result: dict,
    resolved_signals: dict,
    scope_preview: dict,
) -> None:
    rows = mapping_result.get("rows") or []
    available_filter_options = mapping_result.get("available_filter_options") or {}
    audit_state = st.session_state.get("solicitation_scope_audit_state") or []
    mapped_count = sum(1 for item in audit_state if isinstance(item, dict) and item.get("final_pending_value"))
    sol_data = scope_preview.get("Solicitation Number", {})
    st.session_state.solicitation_scope_applied = True
    st.session_state.solicitation_scope_review_open = False
    st.session_state.solicitation_comparable_market = False
    if reviewed_pending_filters.get("contracting_office"):
        st.session_state.solicitation_saved_contracting_office = reviewed_pending_filters["contracting_office"]
    st.session_state.solicitation_applied_metadata = {
        "solicitation_number": str(sol_data.get("value") or "").strip(),
        "title": _solicitation_summary_title(resolved_signals),
        "status_alert": solicitation_status_alert_text(resolved_signals),
        "mapped_filter_count": mapped_count,
        "source_completeness": (mapping_result.get("debug") or {}).get("source_completeness"),
        "elapsed_sec": _solicitation_extraction_elapsed_sec(),
        "requested_signal_count": fast_scope_requested_count(resolved_signals),
    }


def clear_solicitation_scope() -> None:
    last_applied = st.session_state.get("solicitation_last_applied_filters") or {}
    if last_applied.get("contracting_office"):
        st.session_state.active_contracting_office = ALL_CONTRACTING_OFFICES
    market_filters = normalize_market_filters(st.session_state.active_market_filters)
    defaults = default_market_filters()
    for key in (last_applied.get("market_filters") or {}):
        if key in defaults:
            market_filters[key] = defaults[key]
    st.session_state.active_market_filters = market_filters
    st.session_state.solicitation_scope_applied = False
    st.session_state.solicitation_scope_review_open = False
    st.session_state.solicitation_comparable_market = False
    st.session_state.pop("solicitation_saved_contracting_office", None)
    st.session_state.pop("solicitation_applied_metadata", None)


def render_solicitation_sidebar_scope() -> None:
    if not st.session_state.get("solicitation_scope_applied"):
        return
    if st.session_state.get("solicitation_scope_review_open"):
        return
    metadata = st.session_state.get("solicitation_applied_metadata") or {}
    if not metadata and not st.session_state.get("solicitation_resolved_signals"):
        return
    st.markdown("## Solicitation Scope")
    sol_number = metadata.get("solicitation_number") or "—"
    title = metadata.get("title") or "—"
    mapped_count = metadata.get("mapped_filter_count")
    st.caption(f"**{sol_number}**")
    st.caption(_solicitation_truncate(title, 72) or "—")
    if mapped_count is not None:
        st.caption(f"{mapped_count} filters confirmed by analyst")
    status_alert = metadata.get("status_alert")
    if status_alert:
        st.caption(f"Status: {_solicitation_truncate(status_alert, 90)}")
    if st.session_state.get("solicitation_comparable_market"):
        st.caption("Comparable market mode (office filter removed)")
    if st.button("Review / Edit Scope", key="sidebar_review_solicitation_scope", use_container_width=True):
        st.session_state.solicitation_scope_review_open = True
        st.rerun()
    if st.button("Clear Solicitation Scope", key="sidebar_clear_solicitation_scope", use_container_width=True):
        clear_solicitation_scope()
        st.rerun()


def _render_solicitation_scope_heading(resolved_signals: dict, scope_preview: dict) -> None:
    sol_data = scope_preview.get("Solicitation Number", {})
    sol_number = str(sol_data.get("value") or "").strip()
    if not sol_number:
        value, _, _, _ = first_signal_value(resolved_signals, ["rfp_solicitation_number_v1", "rfp_solicitation_id_v1"])
        sol_number = str(value or "").strip()
    title = _solicitation_summary_title(resolved_signals)
    if sol_number and title:
        st.markdown(f"### {html.escape(sol_number)} — {html.escape(title)}")
    elif sol_number:
        st.markdown(f"### {html.escape(sol_number)}")
    elif title:
        st.markdown(f"### {html.escape(title)}")
    elapsed = _solicitation_extraction_elapsed_sec()
    if elapsed is not None:
        st.markdown(
            f'<div class="solicitation-scope-muted">Market scope extracted in {int(round(elapsed))} seconds</div>',
            unsafe_allow_html=True,
        )


def _render_solicitation_status_alert(resolved_signals: dict) -> None:
    alert = solicitation_status_alert_text(resolved_signals)
    if not alert:
        return
    st.markdown(
        f'<div class="solicitation-status-alert-card">Opportunity status: {html.escape(alert)}</div>',
        unsafe_allow_html=True,
    )


def _render_extraction_diagnostics_content(resolved_signals: dict | None = None) -> None:
    extraction_result = st.session_state.get("solicitation_extraction_result") or {}
    summary = extraction_result.get("summary") or {}
    package_diagnostics = summary.get("packageDiagnostics") or {}
    build_info = package_diagnostics.get("buildInfo") or {}
    findings = dedupe_extraction_findings(st.session_state.get("solicitation_extraction_diagnostics") or [])
    ocr_status = ocr_environment_status()
    if True:
        st.caption("Run metadata, cache, tokens, OCR environment, and build/version details.")
        diag_payload = {
            "runId": summary.get("runId") or st.session_state.get("solicitation_loaded_run_id"),
            "model": package_diagnostics.get("model") or build_info.get("model"),
            "reasoningEffort": package_diagnostics.get("reasoningEffort") or build_info.get("reasoningEffort"),
            "elapsedSec": summary.get("elapsedSec"),
            "cacheHit": package_diagnostics.get("cacheHit"),
            "inputTokens": package_diagnostics.get("inputTokens"),
            "outputTokens": package_diagnostics.get("outputTokens"),
            "reasoningTokens": package_diagnostics.get("reasoningTokens"),
            "openaiRequestCount": package_diagnostics.get("openaiRequestCount"),
            "requestedSignalCount": fast_scope_requested_count(resolved_signals or {}),
            "ocrEnvironment": ocr_status,
            "buildInfo": build_info,
        }
        st.json(diag_payload)
        blocking = [item for item in findings if not is_ocr_environment_notice(str(item.get("message") or ""))]
        env_notices = [item for item in findings if is_ocr_environment_notice(str(item.get("message") or ""))]
        if env_notices:
            st.markdown("**OCR environment**")
            st.json(env_notices)
        if blocking:
            st.markdown("**Validation / extraction findings**")
            st.json(blocking)
        if st.session_state.get("solicitation_scope_applied"):
            st.markdown("**Analyzed USAspending scope (audit)**")
            market_filters = normalize_market_filters(st.session_state.get("active_market_filters") or {})
            office_code, office_name = decode_contracting_office(st.session_state.get("active_contracting_office") or "")
            naics_code, naics_label = decode_option(market_filters.get("naics_code", ALL_NAICS_CODES))
            psc_code, psc_label = decode_option(market_filters.get("psc_code", ALL_PRODUCT_SERVICE_CODES))
            contract_code, contract_label = decode_option(market_filters.get("contract_type", ALL_CONTRACT_TYPES))
            set_aside_code, set_aside_label = decode_option(market_filters.get("set_aside_type", ALL_SET_ASIDE_TYPES))
            pop_code, pop_label = decode_option(market_filters.get("pop_state", ALL_POP_LOCATIONS))
            funding_code, funding_label = decode_option(market_filters.get("funding_office", ALL_FUNDING_OFFICES))
            scope_audit = {
                "fiscalYear": st.session_state.get("active_fiscal_year"),
                "agency": st.session_state.get("active_agency"),
                "subagency": st.session_state.get("active_bureau"),
                "contractingOffice": st.session_state.get("active_contracting_office"),
                "awardingOfficeCode": office_code,
                "awardingOfficeName": office_name,
                "fundingOffice": funding_label if funding_code != ALL_FUNDING_OFFICES else ALL_FUNDING_OFFICES,
                "naics": naics_label,
                "psc": psc_label,
                "contractType": contract_label,
                "setAside": set_aside_label,
                "placeOfPerformance": pop_label,
                "comparableMarketMode": bool(st.session_state.get("solicitation_comparable_market")),
            }
            st.json(scope_audit)
            validation = st.session_state.get("last_analysis_validation_metadata") or {}
            if validation:
                st.markdown("**Last analyzed metrics (audit)**")
                st.json(
                    {
                        "transactionRowCount": validation.get("transaction_row_count"),
                        "uniqueAwardCount": validation.get("unique_award_count"),
                        "totalObligations": validation.get("total_obligations_returned"),
                        "dataSource": validation.get("data_source"),
                        "usaspendingFilterSummary": validation.get("usaspending_filter_summary"),
                    }
                )
        st.markdown("**Full extraction summary**")
        st.json(summary)


def render_solicitation_extraction_diagnostics(resolved_signals: dict | None = None) -> None:
    with st.expander("Extraction diagnostics", expanded=False):
        _render_extraction_diagnostics_content(resolved_signals)


def render_solicitation_baseline_notice() -> None:
    metadata = st.session_state.get("solicitation_applied_metadata") or {}
    if not metadata or st.session_state.get("solicitation_scope_review_open"):
        return
    sol_number = metadata.get("solicitation_number")
    mapped_count = metadata.get("mapped_filter_count")
    if not sol_number:
        return
    suffix = f" · {mapped_count} filters confirmed by analyst" if mapped_count is not None else ""
    st.markdown(
        f'<div class="solicitation-baseline-line">Analysis based on solicitation {html.escape(sol_number)}{html.escape(suffix)}</div>',
        unsafe_allow_html=True,
    )


def render_solicitation_scope_preview(
    agency_records: list[dict],
    pending_agency: str,
    pending_bureau: str | None,
    pending_year: int,
    *,
    show_advanced_upload: bool = True,
) -> bool:
    if not _should_show_solicitation_scope_review():
        return False
    resolved_signals = st.session_state.get("solicitation_resolved_signals")
    scope_preview = st.session_state.get("solicitation_scope_preview")
    if not resolved_signals or not scope_preview:
        return False

    user_overrides: dict[str, str] = st.session_state.get("solicitation_user_filter_overrides") or {}
    removed_fields = _solicitation_removed_fields()
    organization_seed = map_solicitation_organization(scope_preview, agency_records, int(pending_year))
    agency_widget_value = st.session_state.get(_solicitation_review_session_key("value", "Agency"))
    scoped_agency = agency_widget_value if agency_widget_value in agency_names_from_records(agency_records) else _solicitation_user_override(user_overrides, "Agency")
    if not scoped_agency or scoped_agency == KEEP_CURRENT_SOLICITATION_FILTER:
        scoped_agency = organization_seed.get("mapped_agency") or pending_agency
    scoped_bureau_options = solicitation_bureau_options_for_agency(agency_records, scoped_agency, int(pending_year))
    bureau_widget_value = st.session_state.get(_solicitation_review_session_key("value", "Subagency / Bureau"))
    scoped_bureau = bureau_widget_value if bureau_widget_value in scoped_bureau_options else _solicitation_user_override(user_overrides, "Subagency / Bureau")
    if not scoped_bureau or scoped_bureau == KEEP_CURRENT_SOLICITATION_FILTER:
        scoped_bureau = organization_seed.get("mapped_bureau") or pending_bureau
    if canonical_bureau_name(scoped_bureau) not in scoped_bureau_options:
        scoped_bureau = ALL_BUREAUS
        st.session_state.pop(_solicitation_review_session_key("value", "Subagency / Bureau"), None)
        st.session_state.pop(_solicitation_review_session_key("value", "Contracting Office"), None)
        st.session_state.pop(_solicitation_review_session_key("value", "Funding Office"), None)
    review_hierarchy_scope = (scoped_agency, canonical_bureau_name(scoped_bureau))
    if st.session_state.get("solicitation_review_hierarchy_scope") != review_hierarchy_scope:
        st.session_state.solicitation_review_hierarchy_scope = review_hierarchy_scope
        st.session_state.pop(_solicitation_review_session_key("value", "Contracting Office"), None)
        st.session_state.pop(_solicitation_review_session_key("value", "Funding Office"), None)

    mapping_cache_key = {
        "run_id": st.session_state.get("solicitation_loaded_run_id"),
        "agency": scoped_agency,
        "bureau": scoped_bureau,
        "year": int(pending_year),
    }
    cached_mapping = st.session_state.get("solicitation_mapping_result")
    cached_key = st.session_state.get("solicitation_mapping_cache_key")
    if isinstance(cached_mapping, dict) and cached_key == mapping_cache_key:
        mapping_result = cached_mapping
    else:
        st.session_state.solicitation_mapping_status = "running"
        try:
            mapping_result = map_solicitation_signals_to_dashboard_filters(
                scope_preview,
                agency_records,
                int(pending_year),
                mapped_agency=scoped_agency,
                mapped_bureau=scoped_bureau,
            )
            st.session_state.solicitation_mapping_status = "completed"
        except TimeoutError as exc:
            mapping_result = degraded_solicitation_mapping_result(
                scope_preview,
                agency_records,
                int(pending_year),
                mapped_agency=scoped_agency,
                mapped_bureau=scoped_bureau,
                reason=f"Dashboard option matching timed out: {exc}",
            )
            st.session_state.solicitation_mapping_status = "timed_out"
        except Exception as exc:
            mapping_result = degraded_solicitation_mapping_result(
                scope_preview,
                agency_records,
                int(pending_year),
                mapped_agency=scoped_agency,
                mapped_bureau=scoped_bureau,
                reason=f"Dashboard option matching failed: {exc}",
            )
            st.session_state.solicitation_mapping_status = "failed"
        st.session_state.solicitation_mapping_result = mapping_result
        st.session_state.solicitation_mapping_cache_key = mapping_cache_key
    rows = mapping_result["rows"]
    available_filter_options = mapping_result["available_filter_options"]
    funding_office_extracted = scope_preview.get("Funding Office", {}).get("value") is not None
    rows_by_field = _solicitation_rows_by_field(rows)
    org_row = rows_by_field.get("Extracted Organization")
    extracted_org = (
        org_row.get("extracted_value")
        if org_row
        else scope_preview.get("Issuing Agency", {}).get("value")
    )
    context_rows = build_solicitation_opportunity_context_rows(resolved_signals, scope_preview)
    optional_rows = _solicitation_optional_filter_rows(
        rows,
        available_filter_options,
        removed_fields=removed_fields,
        funding_office_extracted=funding_office_extracted,
    )

    st.markdown('<div class="market-summary-panel">', unsafe_allow_html=True)
    st.markdown('<div class="market-summary-title">Solicitation Scope Review</div>', unsafe_allow_html=True)
    _render_solicitation_scope_heading(resolved_signals, scope_preview)
    _render_solicitation_status_alert(resolved_signals)
    mapping_diagnostics = available_filter_options.get("mapping_diagnostics") or {}
    unavailable_fields = available_filter_options.get("mapping_unavailable_fields") or []
    mapping_status = st.session_state.get("solicitation_mapping_status", "not_started")

    review_started = time.time()
    review_rows = _solicitation_review_rows(rows, scope_preview, available_filter_options)
    review_elapsed = round(time.time() - review_started, 3)
    completeness_started = time.time()
    source_completeness, missing_market_fields = _solicitation_source_completeness(review_rows)
    completeness_elapsed = round(time.time() - completeness_started, 3)
    extracted_count = sum(
        1
        for item in review_rows
        if item.get("field") != "AAC" and item.get("row", {}).get("extracted_value") is not None
    )
    selected_count = sum(
        1
        for item in review_rows
        if item.get("field") != "AAC" and item.get("analyst_selected")
    )
    review_count = sum(
        1
        for item in review_rows
        if item.get("field") != "AAC" and not item.get("analyst_selected")
    )
    _render_scope_review_text(
        f"{extracted_count} fields extracted · {selected_count} filters selected · {review_count} require review",
        tone="secondary",
    )
    if unavailable_fields:
        _render_scope_review_text("Some optional filters were not available and can be entered manually.")

    _render_compact_solicitation_review(review_rows, available_filter_options)
    review_rebuild_started = time.time()
    review_rows = _solicitation_review_rows(rows, scope_preview, available_filter_options)
    review_rebuild_elapsed = round(time.time() - review_rebuild_started, 3)
    reviewed_pending_filters = _build_pending_filters_from_review_rows(review_rows, available_filter_options)
    audit_state = _build_solicitation_audit_state(review_rows)
    st.session_state.solicitation_scope_audit_state = audit_state

    st.session_state.solicitation_user_filter_overrides = user_overrides

    mapping_result["debug"]["review_audit_state"] = audit_state
    mapping_result["debug"]["source_completeness"] = source_completeness
    mapping_result["debug"]["missing_market_fields"] = missing_market_fields
    mapping_result["debug"]["review_stage_events"] = [
        {
            "stage": "build solicitation audit rows",
            "status": "completed",
            "elapsedSeconds": review_elapsed,
        },
        {
            "stage": "calculate source completeness",
            "status": "completed",
            "elapsedSeconds": completeness_elapsed,
        },
        {
            "stage": "rebuild review rows after controls",
            "status": "completed",
            "elapsedSeconds": review_rebuild_elapsed,
        },
    ]
    mapping_result["debug"]["final_pending_filters_to_apply"] = reviewed_pending_filters
    mapping_result["debug"]["mapping_diagnostics"] = mapping_diagnostics
    mapping_result["debug"]["mapping_unavailable_fields"] = unavailable_fields
    mapping_result["debug"]["scoped_option_counts"] = {
        key: len(available_filter_options.get(key, []))
        for key in (
            "agencies",
            "bureaus",
            "contracting_offices",
            "funding_offices",
            "naics",
            "psc",
            "contract_types",
            "set_asides",
            "pop_locations",
        )
    }
    mapping_result["debug"]["warnings"] = []
    subagency_row = rows_by_field.get("Subagency / Bureau")
    if subagency_row and subagency_row.get("unmapped_extracted_display") and not subagency_row.get("filter_option"):
        mapping_result["debug"]["warnings"].append(
            "Extracted subagency is not available in scoped bureau options and requires user confirmation."
        )
    st.session_state.solicitation_mapping_result = mapping_result
    st.session_state.solicitation_mapping_cache_key = mapping_cache_key

    _render_solicitation_confirmation_summary(review_rows)
    st.session_state.solicitation_review_status = "displayed"

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        apply_filters = st.button(
            "Apply Selected Filters",
            key="apply_solicitation_filters",
            use_container_width=True,
        )
    with action_col2:
        apply_filters_and_run = st.button(
            "Apply Selected Filters & Run Analysis",
            key="apply_solicitation_filters_and_run",
            type="primary",
            use_container_width=True,
        )

    if apply_filters or apply_filters_and_run:
        apply_solicitation_pending_filters(reviewed_pending_filters, available_filter_options)
        _mark_solicitation_scope_applied(
            reviewed_pending_filters,
            mapping_result=mapping_result,
            resolved_signals=resolved_signals,
            scope_preview=scope_preview,
        )
        if apply_filters_and_run:
            mark_analysis_started(
                st.session_state.active_agency,
                st.session_state.active_bureau,
                st.session_state.active_fiscal_year,
                st.session_state.active_contracting_office,
                st.session_state.active_market_filters,
            )
        st.rerun()

    with st.expander("Developer diagnostics", expanded=False):
        details_tab, diagnostics_tab, tools_tab = st.tabs(
            ["Extracted details", "Diagnostics", "Advanced tools"]
        )
        engine_diagnostics = st.session_state.get("solicitation_engine_diagnostics") or {}
        if engine_diagnostics:
            st.caption(
                "Extraction engine: "
                f"{engine_diagnostics.get('mode') or '-'} / {engine_diagnostics.get('profile') or '-'}"
            )
        with details_tab:
            _render_solicitation_full_details_content(
                resolved_signals,
                scope_preview,
                rows,
                context_rows,
                user_overrides,
                available_filter_options,
                extracted_org,
                run_id=st.session_state.get("solicitation_loaded_run_id"),
            )
        with diagnostics_tab:
            _render_extraction_diagnostics_content(resolved_signals)
            preview_validation = build_analysis_validation_metadata(
                agency=scoped_agency,
                bureau=scoped_bureau,
                fiscal_year=int(pending_year),
                contracting_office=st.session_state.get("active_contracting_office"),
                market_filters=st.session_state.get("active_market_filters"),
                data_source_labels=[
                    st.session_state.get("solicitation_signals_source", "uploaded_resolved_signals_json")
                ],
                analysis_context="solicitation_mapping_preview",
            )
            preview_validation["mapping_option_scope"] = available_filter_options.get("scope")
            mapping_result["debug"]["as_of_validation"] = preview_validation
            st.json(mapping_result["debug"])
        with tools_tab:
            if show_advanced_upload:
                _render_resolved_signals_upload_content(show_test_mode_note=False)
            else:
                st.caption("Advanced upload tools are unavailable in this context.")

    st.markdown("</div>", unsafe_allow_html=True)
    return True


def render_landing_page(agency_records: list[dict]) -> None:
    hide_sidebar_for_landing()

    st.markdown('<div class="landing-control-spacer"></div>', unsafe_allow_html=True)
    _left, center, _right = st.columns([1, 1.25, 1])

    with center:
        st.markdown(
            """
            <div class="landing-title">Government Award Data, Simplified.</div>
            <div class="landing-subtitle">Monitor shifting agency spending trajectories, isolate hidden budget reductions, and view the dominant vendors at every agency or subagency.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="workflow-section-label">Start with a market</div>
            <div class="workflow-section-helper">Select an agency, bureau, and fiscal year to explore prime contract obligations.</div>
            """,
            unsafe_allow_html=True,
        )
        (
            active_agency,
            selected_bureau,
            selected_year,
            _active_toptier_code,
            selected_contracting_office,
            selected_market_filters,
        ) = render_market_selectors(
            agency_records
        )
        st.write("")
        if st.button("Run Data Analysis", type="primary", use_container_width=True):
            mark_analysis_started(
                active_agency,
                selected_bureau,
                selected_year,
                selected_contracting_office,
                selected_market_filters,
            )
            st.rerun()

        render_workflow_or_divider()
        render_solicitation_workflow_cta()
        render_upload_solicitation_package_panel()
        render_solicitation_scope_preview(
            agency_records,
            active_agency,
            selected_bureau,
            selected_year,
            show_advanced_upload=True,
        )


def render_market_selectors(agency_records: list[dict]) -> tuple[str, str, int, str, str, dict]:
    agency_options = agency_names_from_records(agency_records)
    if not agency_options:
        st.error("USAspending agency registry is unavailable. Please refresh and try again.")
        st.stop()

    active_record = agency_record_by_name(agency_records, st.session_state.active_agency)
    active_agency = active_record["agency_name"]
    active_toptier_code = active_record["toptier_code"]

    selected_agency = st.selectbox(
        "Select Federal Agency",
        agency_options,
        index=agency_options.index(active_agency),
    )
    selected_record = agency_record_by_name(agency_records, selected_agency)
    active_agency = selected_record["agency_name"]
    active_toptier_code = selected_record["toptier_code"]

    bureau_slot = st.empty()
    fiscal_year_slot = st.empty()

    fiscal_year_options = list(range(current_fiscal_year(), current_fiscal_year() - 8, -1))
    selected_year = int(st.session_state.active_fiscal_year)
    if selected_year not in fiscal_year_options:
        selected_year = fiscal_year_options[0]
    with fiscal_year_slot:
        selected_year = st.selectbox(
            "Fiscal Year",
            fiscal_year_options,
            index=fiscal_year_options.index(selected_year),
            format_func=fiscal_year_label,
        )

    bureau_options = meaningful_bureau_options(
        active_agency,
        get_bureau_options(active_toptier_code, selected_year),
    )
    active_bureau = st.session_state.active_bureau
    active_bureau = canonical_bureau_name(active_bureau)
    if active_bureau not in bureau_options:
        active_bureau = ALL_BUREAUS

    if len(bureau_options) > 1:
        with bureau_slot:
            selected_bureau = st.selectbox(
                "Subagency / Bureau",
                bureau_options,
                index=bureau_options.index(active_bureau),
            )
    else:
        with bureau_slot:
            st.selectbox(
                "Subagency / Bureau",
                [NOT_APPLICABLE_BUREAU],
                index=0,
                disabled=True,
                help="USAspending does not provide a meaningful bureau breakdown for this agency.",
            )
        selected_bureau = ALL_BUREAUS

    current_selector_scope = (active_agency, selected_bureau, int(selected_year))
    if st.session_state.get("active_selector_scope") != current_selector_scope:
        st.session_state.active_selector_scope = current_selector_scope
        st.session_state.active_contracting_office = ALL_CONTRACTING_OFFICES
        st.session_state.active_market_filters = default_market_filters()

    st.session_state.active_agency = active_agency
    st.session_state.active_toptier_code = active_toptier_code
    st.session_state.active_bureau = selected_bureau
    st.session_state.active_fiscal_year = int(selected_year)

    return (
        active_agency,
        selected_bureau,
        int(selected_year),
        active_toptier_code,
        st.session_state.active_contracting_office,
        normalize_market_filters(st.session_state.active_market_filters),
    )


def render_refine_market_selectors(
    active_agency: str,
    selected_bureau: str | None,
    selected_year: int,
) -> tuple[str, dict]:
    active_market_filters = normalize_market_filters(st.session_state.active_market_filters)
    selected_contracting_office = st.session_state.active_contracting_office

    naics_scope_filters = normalize_market_filters({**active_market_filters, "naics_code": ALL_NAICS_CODES})
    naics_options = fetch_category_filter_options(
        active_agency,
        selected_bureau,
        int(selected_year),
        "naics",
        ALL_NAICS_CODES,
        market_filters=naics_scope_filters,
    )
    psc_scope_filters = normalize_market_filters({**active_market_filters, "psc_code": ALL_PRODUCT_SERVICE_CODES})
    psc_options = fetch_category_filter_options(
        active_agency,
        selected_bureau,
        int(selected_year),
        "psc",
        ALL_PRODUCT_SERVICE_CODES,
        market_filters=psc_scope_filters,
    )
    contract_type_options = default_filter_options()["contract_types"]
    set_aside_options = default_filter_options()["set_asides"]
    pop_scope_filters = normalize_market_filters({**active_market_filters, "pop_state": ALL_POP_LOCATIONS})
    country_options = fetch_pop_country_filter_options(
        active_agency,
        selected_bureau,
        int(selected_year),
        ALL_POP_LOCATIONS,
        market_filters=pop_scope_filters,
    )
    us_state_options = [
        encode_option(code, name) for code, name in sorted(STATE_OPTIONS.items(), key=lambda item: item[1])
    ]
    foreign_country_options = [
        option for option in country_options if option != ALL_POP_LOCATIONS
    ]
    pop_location_options = [ALL_POP_LOCATIONS] + us_state_options + foreign_country_options
    contracting_options_key = office_options_scope_key(
        active_agency,
        selected_bureau,
        int(selected_year),
        "contracting",
        selected_contracting_office,
        active_market_filters,
    )
    funding_options_key = office_options_scope_key(
        active_agency,
        selected_bureau,
        int(selected_year),
        "funding",
        selected_contracting_office,
        active_market_filters,
    )
    cached_contracting_set = st.session_state.transaction_office_options_cache.get(
        contracting_options_key,
        {"options": [ALL_CONTRACTING_OFFICES], "stats": {}},
    )
    cached_funding_set = st.session_state.transaction_office_options_cache.get(
        funding_options_key,
        {"options": [ALL_FUNDING_OFFICES], "stats": {}},
    )

    download_rows = []
    if len(cached_contracting_set.get("options", [])) <= 1 or len(cached_funding_set.get("options", [])) <= 1:
        download_rows = fetch_transaction_download_office_rows(
            active_agency,
            selected_bureau,
            int(selected_year),
            market_filters=active_market_filters,
            cache_version=APP_CACHE_VERSION,
        )

    downloaded_contracting_set = scoped_office_option_set(
        download_rows,
        "awarding",
        ALL_CONTRACTING_OFFICES,
        selected_contracting_office,
        active_market_filters,
    )
    downloaded_funding_set = scoped_office_option_set(
        download_rows,
        "funding",
        ALL_FUNDING_OFFICES,
        selected_contracting_office,
        active_market_filters,
    )
    contracting_option_set = merge_office_option_sets(
        ALL_CONTRACTING_OFFICES,
        "awarding",
        cached_contracting_set,
        downloaded_contracting_set,
    )
    funding_option_set = merge_office_option_sets(
        ALL_FUNDING_OFFICES,
        "funding",
        cached_funding_set,
        downloaded_funding_set,
    )
    if len(contracting_option_set["options"]) > 1:
        st.session_state.transaction_office_options_cache[contracting_options_key] = contracting_option_set
    if len(funding_option_set["options"]) > 1:
        st.session_state.transaction_office_options_cache[funding_options_key] = funding_option_set

    contracting_office_options = contracting_option_set["options"]
    contracting_office_stats = contracting_option_set["stats"]
    funding_office_options = funding_option_set["options"]
    funding_office_stats = funding_option_set["stats"]

    active_naics = active_market_filters["naics_code"]
    if active_naics not in naics_options:
        active_naics = ALL_NAICS_CODES
    active_contract_type = active_market_filters["contract_type"]
    if active_contract_type not in contract_type_options:
        active_contract_type = ALL_CONTRACT_TYPES
    active_psc = active_market_filters["psc_code"]
    if active_psc not in psc_options:
        active_psc = ALL_PRODUCT_SERVICE_CODES
    active_set_aside = active_market_filters["set_aside_type"]
    if active_set_aside not in set_aside_options:
        active_set_aside = ALL_SET_ASIDE_TYPES
    active_pop_location = active_market_filters["pop_state"]
    if active_pop_location not in pop_location_options:
        active_pop_location = ALL_POP_LOCATIONS
    active_funding_office = active_market_filters["funding_office"]
    if active_funding_office not in funding_office_options:
        active_funding_office = ALL_FUNDING_OFFICES
    if selected_contracting_office not in contracting_office_options:
        selected_contracting_office = ALL_CONTRACTING_OFFICES

    advanced_filters_active = (
        active_naics != ALL_NAICS_CODES
        or active_contract_type != ALL_CONTRACT_TYPES
        or active_psc != ALL_PRODUCT_SERVICE_CODES
        or active_set_aside != ALL_SET_ASIDE_TYPES
        or selected_contracting_office != ALL_CONTRACTING_OFFICES
        or active_funding_office != ALL_FUNDING_OFFICES
        or active_pop_location not in (ALL_POP_LOCATIONS, LEGACY_ALL_POP_STATES)
    )

    with st.expander("Refine Market", expanded=advanced_filters_active):
        st.caption("Optional filters stay on All until you explicitly narrow the market.")
        if advanced_filters_active and st.button("Clear Refinements", key="clear-refinements-sidebar"):
            clear_active_refinements()
            st.rerun()
        selected_naics = st.selectbox(
            "NAICS Code",
            naics_options,
            index=naics_options.index(active_naics),
            format_func=lambda option: option if option == ALL_NAICS_CODES else format_code_description_option(option),
        )
        selected_contract_type = st.selectbox(
            "Contract Type",
            contract_type_options,
            index=contract_type_options.index(active_contract_type),
            format_func=lambda option: option if option == ALL_CONTRACT_TYPES else format_code_description_option(option),
        )
        selected_psc = st.selectbox(
            "Product / Service Code",
            psc_options,
            index=psc_options.index(active_psc),
            format_func=lambda option: option
            if option == ALL_PRODUCT_SERVICE_CODES
            else format_code_description_option(option),
        )
        selected_set_aside = st.selectbox(
            "Set-Aside Type",
            set_aside_options,
            index=set_aside_options.index(active_set_aside),
            format_func=lambda option: option if option == ALL_SET_ASIDE_TYPES else format_code_description_option(option),
        )
        selected_contracting_office = st.selectbox(
            "Contracting Office",
            contracting_office_options,
            index=contracting_office_options.index(selected_contracting_office),
            format_func=lambda option: format_office_option_with_stats(
                option,
                ALL_CONTRACTING_OFFICES,
                "awarding",
                contracting_office_stats,
            ),
            help="Office that issued the contract.",
        )
        selected_funding_office = st.selectbox(
            "Funding Office",
            funding_office_options,
            index=funding_office_options.index(active_funding_office),
            format_func=lambda option: format_office_option_with_stats(
                option,
                ALL_FUNDING_OFFICES,
                "funding",
                funding_office_stats,
            ),
            help="Office or program that funded the award.",
        )
        selected_pop_location = st.selectbox(
            "Place of Performance",
            pop_location_options,
            index=pop_location_options.index(active_pop_location),
            format_func=lambda option: option
            if option in (ALL_POP_LOCATIONS, LEGACY_ALL_POP_STATES)
            else format_code_description_option(option),
            help="Filter by US state or foreign country where work is performed.",
        )

    selected_market_filters = normalize_market_filters(
        {
            "naics_code": selected_naics,
            "contract_type": selected_contract_type,
            "psc_code": selected_psc,
            "set_aside_type": selected_set_aside,
            "funding_office": selected_funding_office,
            "pop_state": selected_pop_location,
        }
    )
    st.session_state.active_contracting_office = selected_contracting_office
    st.session_state.active_market_filters = selected_market_filters
    st.session_state.pending_office_option_stats = {
        "contracting_office": contracting_office_stats.get(selected_contracting_office),
        "funding_office": funding_office_stats.get(selected_funding_office),
    }
    return selected_contracting_office, selected_market_filters


def render_dashboard_header(active_agency: str, selected_bureau: str | None) -> None:
    safe_agency = html.escape(active_agency)
    selected_bureau = canonical_bureau_name(selected_bureau)
    safe_bureau = "" if selected_bureau == ALL_BUREAUS else f" / {html.escape(selected_bureau)}"
    agency_header = f"{safe_agency}{safe_bureau}"
    st.markdown(
        f"""
        <section class="hero">
            <h1>{agency_header}</h1>
            <p class="brand-subtitle">GovCon Pulse: Federal Spending Intelligence Hub</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_dashboard(
    active_agency: str,
    selected_bureau: str | None,
    selected_year: int,
    selected_contracting_office: str,
    selected_market_filters: dict,
    loading_slot=None,
) -> dict:
    selected_bureau = canonical_bureau_name(selected_bureau)
    selected_market_filters = normalize_market_filters(selected_market_filters)
    processed_transaction_count = None
    processed_unique_award_count = None

    def update_loading_card() -> None:
        if loading_slot is not None:
            render_analysis_loading_card(
                loading_slot,
                int(selected_year),
                processed_transaction_count,
                processed_unique_award_count,
            )

    office_filter_active = (
        selected_contracting_office != ALL_CONTRACTING_OFFICES
        or selected_market_filters["funding_office"] != ALL_FUNDING_OFFICES
    )
    active_refinements = has_active_refinements(selected_market_filters, selected_contracting_office)
    transaction_df = pd.DataFrame()
    transaction_payload = {}
    transaction_source = ""
    transaction_error = None

    if office_filter_active:
        transaction_df, transaction_payload, transaction_source, transaction_error = fetch_office_filtered_download_transactions(
            active_agency,
            selected_bureau,
            int(selected_year),
            contracting_office=selected_contracting_office,
            include_positive=True,
            market_filters=selected_market_filters,
        )
        trend_df = transaction_trend_dataframe(transaction_df, int(selected_year))
        vendor_df = transaction_vendor_dataframe(transaction_df)
        trend_payload = transaction_payload
        vendor_payload = transaction_payload
        trend_source = "Client-filtered transaction records"
        vendor_source = "Client-filtered transaction records"
        trend_error = transaction_error
        vendor_error = transaction_error
        processed_transaction_count = int(len(transaction_df))
        update_loading_card()
    else:
        trend_df, trend_payload, trend_source, trend_error = fetch_trends(
            active_agency,
            selected_bureau,
            market_filters=selected_market_filters,
        )
        vendor_df, contractor_count, vendor_payload, vendor_source, vendor_error = fetch_vendors(
            active_agency,
            selected_bureau,
            market_filters=selected_market_filters,
        )

    macro_live = trend_source.startswith("Live") and vendor_source.startswith("Live")
    if office_filter_active:
        source_label = "Client-filtered transaction records"
    elif macro_live:
        source_label = "Live USAspending.gov macro endpoints"
    else:
        source_label = "USAspending.gov macro endpoint issue"

    source_chip(source_label)
    st.caption(
        "Default view shows prime contract transaction obligations from USAspending. "
        "These represent obligated contract spending, not necessarily cash payments/outlays."
    )
    if selected_contracting_office != ALL_CONTRACTING_OFFICES:
        st.caption(f"Contracting office: {format_contracting_office_option(selected_contracting_office)}")
    if selected_market_filters["funding_office"] != ALL_FUNDING_OFFICES:
        st.caption(f"Funding office: {format_funding_office_option(selected_market_filters['funding_office'])}")
    if trend_error:
        st.warning("Historical trend data is unavailable for this scope. Reconciliation details are available in API Payloads.")
    if vendor_error and office_filter_active:
        st.warning("Vendor rankings are unavailable for this office-filtered scope. Reconciliation details are available in API Payloads.")

    active_scope_payload = active_filter_payload(
        active_agency,
        selected_bureau,
        int(selected_year),
        selected_market_filters,
        selected_contracting_office,
    )
    dashboard_scope_key = time_grain_scope_key(
        active_agency,
        selected_bureau,
        int(selected_year),
        selected_contracting_office,
        selected_market_filters,
    )
    render_market_scope_summary(
        active_agency,
        selected_bureau,
        int(selected_year),
        selected_market_filters,
        selected_contracting_office,
    )

    selected_year_label = fiscal_year_compact_label(int(selected_year))
    current_total, previous_total = current_and_previous(trend_df, int(selected_year))
    previous_total, yoy_metric_label, yoy_metric_subtitle = yoy_comparison_context(
        active_agency,
        selected_bureau,
        int(selected_year),
        trend_df,
        current_total,
        office_filter_active,
        selected_contracting_office,
        selected_market_filters,
    )
    yoy_delta = None
    if previous_total and previous_total > 0:
        yoy_delta = ((current_total - previous_total) / previous_total) * 100
    total_spend_value = "Unavailable" if trend_error else format_money(current_total)
    yoy_delta_value = "Unavailable" if trend_error else format_delta(yoy_delta)
    contractor_count_value = "Unavailable" if vendor_error else format_count(len(vendor_df))
    leaderboard_total = (
        float(pd.to_numeric(vendor_df["amount"], errors="coerce").fillna(0).sum())
        if not vendor_df.empty and "amount" in vendor_df.columns
        else 0.0
    )
    award_scope_df, award_scope_payload, award_scope_error = fetch_award_scope_from_download(
        active_agency,
        selected_bureau,
        int(selected_year),
        contracting_office=selected_contracting_office,
        market_filters=selected_market_filters,
    )
    award_scope_payload.setdefault("award_scope_debug", {})
    award_scope_payload["award_scope_debug"]["working_main_kpi_payload"] = {
        "endpoint": "/api/v2/search/spending_over_time/",
        "request_body": trend_payload,
    }
    award_scope_request_body = (
        award_scope_payload.get("award_scope_debug", {})
        .get("award_scope_transaction_download_payload", {})
        .get("request_body", {})
    )
    award_scope_payload["award_scope_debug"]["main_vs_award_scope_filter_comparison"] = {
        "agency_filters_match": (trend_payload.get("filters", {}).get("agencies") == award_scope_payload.get("filters", {}).get("agencies")),
        "award_type_filters_match": (
            trend_payload.get("filters", {}).get("award_type_codes")
            == award_scope_payload.get("filters", {}).get("award_type_codes")
        ),
        "award_or_idv_flag_match": (
            trend_payload.get("filters", {}).get("award_or_idv_flag")
            == award_scope_payload.get("filters", {}).get("award_or_idv_flag")
        ),
        "main_kpi_time_period": trend_payload.get("filters", {}).get("time_period"),
        "award_scope_time_period": award_scope_payload.get("filters", {}).get("time_period"),
        "subagency_filter_sent": resolve_bureau_filter_name(selected_bureau),
        "not_applicable_in_award_scope_payload": "Not applicable" in json.dumps(award_scope_request_body, default=str),
    }
    award_totals = award_scope_totals(award_scope_df)
    processed_unique_award_count = int(award_totals["award_count"])
    update_loading_card()
    if award_scope_error:
        if current_total > 0:
            st.info(
                "Award scope metrics are unavailable for this scope. Contract obligation metrics remain available."
            )
        else:
            st.info("Award scope metrics are unavailable for this scope.")
    if not office_filter_active:
        transaction_df, transaction_payload, transaction_source, transaction_error = fetch_office_filtered_download_transactions(
            active_agency,
            selected_bureau,
            int(selected_year),
            contracting_office=None,
            include_positive=True,
            market_filters=selected_market_filters,
        )
        processed_transaction_count = int(len(transaction_df))
        update_loading_card()
        if transaction_error is None:
            vendor_df = transaction_vendor_dataframe(transaction_df)
            vendor_payload = transaction_payload
            vendor_source = "Client-filtered transaction records"
            vendor_error = None
            contractor_count_value = format_count(len(vendor_df))
            leaderboard_total = (
                float(pd.to_numeric(vendor_df["amount"], errors="coerce").fillna(0).sum())
                if not vendor_df.empty and "amount" in vendor_df.columns
                else 0.0
            )
            if active_refinements:
                trend_df = transaction_trend_dataframe(transaction_df, int(selected_year))
                current_total, previous_total = current_and_previous(trend_df, int(selected_year))
                yoy_delta = None
                yoy_delta_value = format_delta(yoy_delta)
                total_spend_value = format_money(current_total)
    if office_filter_active:
        returned_transaction_count = int(len(transaction_df))
        returned_obligation_sum = (
            float(pd.to_numeric(transaction_df["Obligation Amount"], errors="coerce").fillna(0).sum())
            if not transaction_df.empty and "Obligation Amount" in transaction_df.columns
            else 0.0
        )
        recipient_amounts = {}
        if not transaction_df.empty and {"Contractor Name", "Obligation Amount"}.issubset(transaction_df.columns):
            recipient_amounts = {
                str(row["Contractor Name"]): round(float(row["Obligation Amount"]), 2)
                for row in transaction_df.groupby("Contractor Name", as_index=False)["Obligation Amount"]
                .sum()
                .sort_values("Obligation Amount", ascending=False)
                .to_dict("records")
            }
        option_stats = st.session_state.get("analyzed_office_option_stats", {})
        transaction_payload["office_selection_debug"] = {
            "selected_contracting_office": active_scope_payload["contracting_office"],
            "selected_funding_office": active_scope_payload["funding_office"],
            "option_list_stats": option_stats,
            "returned_transaction_count": returned_transaction_count,
            "returned_obligation_sum": round(returned_obligation_sum, 2),
            "recipients_returned": recipient_amounts,
            "pending_filters_equal_active_filters": (
                st.session_state.get("active_agency") == active_agency
                and st.session_state.get("active_bureau") == selected_bureau
                and int(st.session_state.get("active_fiscal_year") or 0) == int(selected_year)
                and st.session_state.get("active_contracting_office") == selected_contracting_office
                and normalize_market_filters(st.session_state.get("active_market_filters"))
                == selected_market_filters
            ),
        }
        selected_option_had_rows = any(
            int((stats or {}).get("transaction_count") or 0) > 0
            for stats in option_stats.values()
        )
        if selected_option_had_rows and returned_transaction_count == 0:
            st.warning(
                "Office filter mismatch: the selected office option had scoped transactions in the option list, "
                "but the dashboard returned zero rows. Check the office_selection_debug payload below."
            )
    if current_total == 0 and leaderboard_total != 0:
        st.warning(
            "Consistency check: KPI obligations are $0.00, but the contractor leaderboard has non-zero obligations. "
            "Inspect the active filter payload and USAspending responses for a filter mismatch."
        )
    if current_total == 0 and not trend_error:
        message = (
            f"No {fiscal_year_label(int(selected_year))} prime contract obligations found for this selected scope. "
            "USAspending may still show grants or assistance awards outside this contract-focused view."
        )
        if active_refinements:
            message += " Try clearing one or more refinements."
        st.info(message)
        if active_refinements:
            if st.button("Clear Refinements", key="clear-refinements-empty-state"):
                clear_active_refinements()
                st.rerun()

    market_concentration, market_concentration_debug = market_concentration_summary(transaction_df, current_total)
    award_drilldown_df, award_drilldown_debug = award_drilldown_dataframe(transaction_df)
    award_drilldown_total = float(award_drilldown_debug.get("total_grouped_award_obligation_sum") or 0.0)
    award_drilldown_reconciles = abs(award_drilldown_total - float(current_total or 0.0)) <= 0.01
    award_drilldown_debug.update(
        {
            "main_kpi_obligation_sum": round(float(current_total or 0.0), 2),
            "grouped_sum_reconciles_to_kpi": award_drilldown_reconciles,
        }
    )
    if not award_drilldown_reconciles:
        print("ERROR: award drilldown obligation total does not reconcile to KPI.")
    negative_transaction_df = transaction_df[transaction_df["Obligation Amount"] < 0].copy()
    negative_obligation_total = float(negative_transaction_df["Obligation Amount"].sum())
    market_summary_text, summary_debug = build_market_summary(
        active_agency,
        selected_bureau,
        int(selected_year),
        selected_contracting_office,
        selected_market_filters,
        current_total,
        vendor_df,
        award_totals,
        bool(award_scope_error),
        market_concentration,
        market_concentration_debug,
        transaction_df,
        award_drilldown_df,
        negative_transaction_df,
        negative_obligation_total,
    )
    metric_cols = st.columns(3)
    with metric_cols[0]:
        metric_card(
            f"{selected_year_label} Contract Obligations",
            total_spend_value,
            "Prime contract obligations reported for the selected scope.",
            "#2dd4bf",
        )
    with metric_cols[1]:
        metric_card(
            yoy_metric_label,
            yoy_delta_value,
            "Compared with the same period in the prior fiscal year."
            if yoy_metric_subtitle == "Compared with same period in prior fiscal year"
            else yoy_metric_subtitle,
            "#f59e0b" if (yoy_delta or 0) >= 0 else "#fb7185",
        )
    with metric_cols[2]:
        metric_card(
            "Contractors Shown",
            contractor_count_value,
            "Top recipient records shown.",
            "#38bdf8",
        )

    award_metric_cols = st.columns(3)
    with award_metric_cols[0]:
        metric_card(
            "Active Award Ceiling",
            "Unavailable" if award_scope_error else format_money(award_totals["active_award_ceiling"]),
            f"{format_count(award_totals['award_count'])} unique awards",
            "#a78bfa",
            helper_text="Total potential value of active awards in this scope.",
        )
    with award_metric_cols[1]:
        metric_card(
            "Current Award Value",
            "Unavailable" if award_scope_error else format_money(award_totals["current_award_value"]),
            "Current total value of active awards in this scope.",
            "#60a5fa",
            helper_text="This is award value, not period obligations.",
        )
    with award_metric_cols[2]:
        metric_card(
            "Remaining Ceiling",
            "Unavailable" if award_scope_error else format_money(award_totals["remaining_ceiling"]),
            "Award ceiling minus current award value.",
            "#f59e0b",
            helper_text="Capacity indicator only; not future obligations.",
        )

    render_market_summary_controls(
        market_summary_text,
        dashboard_scope_key,
        active_agency,
        int(selected_year),
    )

    market_lane_naics_debug, market_lane_psc_debug = render_market_lane_mix(
        transaction_df,
        current_total,
        selected_market_filters,
        market_concentration,
    )
    lane_field_names_found = transaction_payload.get("transaction_lane_field_names_found", {})
    if lane_field_names_found:
        market_lane_naics_debug["transaction_row_naics_field_names_found"] = lane_field_names_found.get("naics", [])
        market_lane_psc_debug["transaction_row_psc_field_names_found"] = lane_field_names_found.get("psc", [])

    st.write("")
    time_grain_key = f"obligation-time-grain-{dashboard_scope_key}"
    time_grain = st.radio(
        "Time Grain",
        TIME_GRAIN_OPTIONS,
        index=TIME_GRAIN_OPTIONS.index(default_time_grain(int(selected_year))),
        horizontal=True,
        key=time_grain_key,
    )
    if office_filter_active:
        obligation_time_df = transaction_obligations_time_series(transaction_df, time_grain)
        obligation_time_payload = transaction_payload
        obligation_time_error = transaction_error
    elif time_grain == TIME_GRAIN_FISCAL_YEAR:
        obligation_time_df = trend_time_series_dataframe(trend_df)
        obligation_time_payload = trend_payload
        obligation_time_error = trend_error
    else:
        obligation_time_df, obligation_time_payload, _obligation_time_source, obligation_time_error = fetch_obligations_time_series(
            active_agency,
            selected_bureau,
            int(selected_year),
            time_grain,
            market_filters=selected_market_filters,
        )
    if obligation_time_error and current_total != 0:
        fallback_time_df = transaction_obligations_time_series(transaction_df, time_grain)
        if not fallback_time_df.empty:
            obligation_time_df = fallback_time_df
            obligation_time_payload = transaction_payload
            obligation_time_error = None
        else:
            st.info("Time-series unavailable for this scope. KPI totals remain available.")
    elif obligation_time_error:
        st.info("Time-series unavailable for this scope. KPI totals remain available.")

    if obligation_time_error:
        obligation_time_df = pd.DataFrame(columns=["bucket_label", "amount", "transaction_count"])
    elif time_grain != TIME_GRAIN_FISCAL_YEAR and not obligation_time_df.empty:
        visible_bucket_total = float(pd.to_numeric(obligation_time_df["amount"], errors="coerce").fillna(0).sum())
        if abs(visible_bucket_total - current_total) > max(1.0, abs(current_total) * 0.001):
            st.warning(
                "Consistency check: visible time buckets do not reconcile to the KPI total. "
                "Inspect the active filter payload and obligations-over-time response below."
            )

    leaderboard_mode = st.radio(
        "Leaderboard Metric",
        LEADERBOARD_OPTIONS,
        index=0,
        horizontal=True,
        key=f"leaderboard-mode-{dashboard_scope_key}",
    )
    if leaderboard_mode == LEADERBOARD_CURRENT_VALUE:
        leaderboard_df = award_scope_vendor_dataframe(award_scope_df, "current_award_value")
        leaderboard_title = "Top Contractors by Current Award Value"
        leaderboard_amount_label = "Current Award Value"
    elif leaderboard_mode == LEADERBOARD_AWARD_CEILING:
        leaderboard_df = award_scope_vendor_dataframe(award_scope_df, "award_ceiling")
        leaderboard_title = "Top Contractors by Award Ceiling"
        leaderboard_amount_label = "Award Ceiling"
    else:
        leaderboard_df = vendor_df
        leaderboard_title = "Top Contractors by Obligations This Period"
        leaderboard_amount_label = "Obligations This Period"

    chart_cols = st.columns([1.15, 1])
    with chart_cols[0]:
        st.plotly_chart(
            make_obligations_over_time_chart(obligation_time_df, time_grain),
            use_container_width=True,
            config={"responsive": True, "displayModeBar": False},
        )
        if time_grain == TIME_GRAIN_FISCAL_YEAR and not trend_df.empty and int(current_fiscal_year()) in set(trend_df["fiscal_year"].astype(int)):
            st.caption(
                f"{fiscal_year_label(current_fiscal_year())} is year-to-date. Obligations are reported contract activity, not profit or loss, and are not directly comparable to completed fiscal years."
            )
    with chart_cols[1]:
        if leaderboard_df.empty:
            st.info("Top Contractor Leaderboard has no prime contract obligations for this scope.")
        else:
            st.plotly_chart(
                make_vendor_chart(
                    leaderboard_df,
                    title=leaderboard_title,
                    amount_label=leaderboard_amount_label,
                ),
                use_container_width=True,
                config={"responsive": True, "displayModeBar": False},
            )

    award_drilldown_debug.update(
        render_top_awards_drilldown(
            award_drilldown_df,
            dashboard_scope_key,
        )
    )

    if transaction_error:
        st.error("Analysis could not complete for this scope. Try clearing refinements or running a broader agency-level query.")
    derived_office_options = transaction_payload.get("derived_office_options")
    if derived_office_options:
        contracting_options_key = office_options_scope_key(
            active_agency,
            selected_bureau,
            int(selected_year),
            "contracting",
            selected_contracting_office,
            selected_market_filters,
        )
        funding_options_key = office_options_scope_key(
            active_agency,
            selected_bureau,
            int(selected_year),
            "funding",
            selected_contracting_office,
            selected_market_filters,
        )
        st.session_state.transaction_office_options_cache[contracting_options_key] = {
            "options": derived_office_options.get("contracting_offices", [ALL_CONTRACTING_OFFICES]),
            "stats": derived_office_options.get("contracting_office_stats", {}),
        }
        st.session_state.transaction_office_options_cache[funding_options_key] = {
            "options": derived_office_options.get("funding_offices", [ALL_FUNDING_OFFICES]),
            "stats": derived_office_options.get("funding_office_stats", {}),
        }
    audit_df = audit_log_dataframe(transaction_df)

    risk_metric_col, _risk_space = st.columns([1, 2])
    with risk_metric_col:
        metric_card(
            "FY Negative Obligations",
            format_money(negative_obligation_total),
            "De-obligated transaction amount",
            CRIMSON,
            value_color=CRIMSON if negative_obligation_total < 0 else None,
        )

    st.plotly_chart(
        make_budget_reduction_chart(negative_transaction_df),
        use_container_width=True,
        config={"responsive": True, "displayModeBar": False},
    )

    time_series_total = (
        float(pd.to_numeric(obligation_time_df["amount"], errors="coerce").fillna(0).sum())
        if not obligation_time_df.empty and "amount" in obligation_time_df.columns
        else 0.0
    )
    transaction_kpi_sum = (
        float(pd.to_numeric(transaction_df["Obligation Amount"], errors="coerce").fillna(0).sum())
        if not transaction_df.empty and "Obligation Amount" in transaction_df.columns
        else 0.0
    )
    leaderboard_transaction_row_count = int(len(transaction_df))
    grouped_vendor_count = int(len(vendor_df)) if not vendor_df.empty else 0
    top_vendor_records = (
        [
            {
                "recipient": str(row.get("recipient") or ""),
                "amount": round(float(row.get("amount") or 0), 2),
            }
            for row in vendor_df.to_dict("records")
        ]
        if not vendor_df.empty
        else []
    )
    leaderboard_filter_application = active_refine_filter_application_debug(
        transaction_payload,
        selected_market_filters,
        selected_contracting_office,
    )
    leaderboard_mismatch = abs(float(current_total or 0)) >= 0.005 and grouped_vendor_count == 0
    if leaderboard_mismatch:
        print("Leaderboard mismatch: KPI has obligations but no grouped vendors.")
        st.warning("Leaderboard mismatch: KPI has obligations but no grouped vendors.")
    component_debug_payloads = {
        "main_kpi": attach_component_scope_debug(
            trend_payload,
            "main_kpi",
            selected_bureau,
            top_tier_fallback_used="top-tier fallback" in trend_source.lower(),
        ),
        "time_series": attach_component_scope_debug(
            obligation_time_payload,
            "time_series",
            selected_bureau,
            top_tier_fallback_used=False,
        ),
        "obligations_leaderboard": attach_component_scope_debug(
            vendor_payload,
            "obligations_leaderboard",
            selected_bureau,
            top_tier_fallback_used="top-tier fallback" in vendor_source.lower(),
        ),
        "award_drilldown": attach_component_scope_debug(
            transaction_payload,
            "award_drilldown",
            selected_bureau,
            top_tier_fallback_used="top-tier fallback" in transaction_source.lower(),
        ),
        "award_scope": attach_component_scope_debug(
            award_scope_payload,
            "award_scope",
            selected_bureau,
            top_tier_fallback_used=False,
        ),
        "negative_obligations": attach_component_scope_debug(
            transaction_payload,
            "negative_obligations",
            selected_bureau,
            top_tier_fallback_used="top-tier fallback" in transaction_source.lower(),
        ),
        "signal_log": attach_component_scope_debug(
            transaction_payload,
            "signal_log",
            selected_bureau,
            top_tier_fallback_used="top-tier fallback" in transaction_source.lower(),
        ),
        "office_filter_options": attach_component_scope_debug(
            transaction_payload,
            "office_filter_options",
            selected_bureau,
            top_tier_fallback_used="top-tier fallback" in transaction_source.lower(),
        ),
    }
    component_scope_debug = {
        component_name: payload.get("component_scope_debug", {}).get(component_name, {})
        for component_name, payload in component_debug_payloads.items()
        if isinstance(payload, dict)
    }
    fallback_violations = [
        debug["log_message"]
        for debug in component_scope_debug.values()
        if debug.get("fallback_violation")
    ]
    if fallback_violations:
        st.warning("ERROR: explicit subagency selected but component used top-tier fallback")
    award_scope_debug = award_scope_payload.get("award_scope_debug", {})
    validation_metadata = build_analysis_validation_metadata(
        agency=active_agency,
        bureau=selected_bureau,
        fiscal_year=int(selected_year),
        contracting_office=selected_contracting_office,
        market_filters=selected_market_filters,
        active_scope_payload=active_scope_payload,
        usaspending_payload_summary=build_usaspending_filter_summary(trend_payload),
        transaction_row_count=int(processed_transaction_count or len(transaction_df)),
        unique_award_count=int(processed_unique_award_count or 0) if processed_unique_award_count is not None else None,
        total_obligations=float(current_total),
        data_source_labels=[trend_source, transaction_source, vendor_source],
        analysis_context="dashboard_analysis",
    )
    st.session_state.analysis_run_completed_at = validation_metadata["as_of"]
    st.session_state.last_analysis_validation_metadata = validation_metadata
    dashboard_debug_payload = {
        "as_of_validation": validation_metadata,
        "main_kpi_obligation_total": round(float(current_total), 2),
        "award_scope_raw_transaction_row_count": int(
            award_scope_debug.get("raw_transaction_rows")
            or award_scope_payload.get("award_scope_rows_returned")
            or 0
        ),
        "award_scope_unique_award_count": int(
            award_scope_debug.get("unique_awards")
            or award_scope_payload.get("unique_awards_returned")
            or 0
        ),
        "time_series_row_count": int(len(obligation_time_df)),
        "time_series_total_obligation_sum": round(time_series_total, 2),
        "active_subagency_filter_sent_to_usaspending": resolve_bureau_filter_name(selected_bureau),
        "subagency_was_ui_only_not_applicable": bureau_is_ui_only_not_applicable(selected_bureau),
        "component_scope_debug": component_scope_debug,
        "fallback_violations": fallback_violations,
        "market_concentration": market_concentration_debug,
        "market_lane_mix": {
            "naics": market_lane_naics_debug,
            "psc": market_lane_psc_debug,
        },
        "summary_debug": summary_debug,
        "award_drilldown": award_drilldown_debug,
        "leaderboard_reconciliation": {
            "active_filter_payload": active_scope_payload,
            "transaction_row_count_used_by_kpi": int(len(transaction_df)),
            "kpi_obligation_sum": round(float(current_total), 2),
            "transaction_dataframe_obligation_sum": round(transaction_kpi_sum, 2),
            "leaderboard_source": "active filtered transaction dataframe",
            "leaderboard_transaction_row_count": leaderboard_transaction_row_count,
            "grouped_vendor_count": grouped_vendor_count,
            "top_vendors": top_vendor_records,
            "all_active_refine_filters_applied_to_leaderboard_rows": leaderboard_filter_application[
                "all_active_refine_filters_applied"
            ],
            "active_refine_filter_application": leaderboard_filter_application,
            "mismatch_error": "Leaderboard mismatch: KPI has obligations but no grouped vendors."
            if leaderboard_mismatch
            else "",
        },
    }

    st.markdown(
        '<div class="audit-heading">&#9888;&#65039; Negative Obligation / De-obligation Signal Log</div>',
        unsafe_allow_html=True,
    )
    if audit_df.empty:
        st.success(
            "No negative obligation or de-obligation signal rows found for this cycle."
        )
    else:
        render_dark_html_table(audit_df, columns=list(audit_df.columns))

    st.markdown('<div class="debug-section-note">Diagnostics and source payloads</div>', unsafe_allow_html=True)
    with st.expander("API Payloads / Debug", expanded=False):
        if st.button("Clear All Cached Data", key="clear-all-cached-data"):
            clear_all_cached_dashboard_data()
            st.rerun()
        st.caption(f"App cache version: {APP_CACHE_VERSION}")
        st.markdown("As-Of / Validation Metadata")
        st.json(json_safe_payload(validation_metadata))
        st.markdown("Active Filter Scope")
        st.json(json_safe_payload(active_scope_payload))
        st.markdown("Dashboard Reconciliation Debug")
        st.json(json_safe_payload(dashboard_debug_payload))
        st.markdown("Historical Trends")
        st.json(json_safe_payload(trend_payload))
        st.markdown("Obligations Over Time")
        st.json(json_safe_payload(obligation_time_payload))
        st.markdown("Vendor Rankings")
        st.json(json_safe_payload(vendor_payload))
        st.markdown("Award Scope Metrics")
        st.json(json_safe_payload(award_scope_payload))
        st.markdown("Transaction Negative Obligations")
        st.json(json_safe_payload(transaction_payload))
        st.divider()
        render_award_scope_diagnostic_probe()
        if trend_error or vendor_error or transaction_error or award_scope_error:
            st.caption("Live API issues are surfaced above; no synthetic data is used.")

    return {
        "transaction_count": processed_transaction_count,
        "unique_award_count": processed_unique_award_count,
        "failed": bool(transaction_error),
    }


def main() -> None:
    inject_styles()

    # Pending sidebar selector state (active_*). These mirror the Control Panel and
    # can diverge from analyzed_* until the user runs or updates analysis.
    if "active_agency" not in st.session_state:
        st.session_state.active_agency = DEFAULT_AGENCY_NAME
    if "active_toptier_code" not in st.session_state:
        st.session_state.active_toptier_code = DEFAULT_TOPTIER_CODE
    if "active_bureau" not in st.session_state:
        st.session_state.active_bureau = ALL_BUREAUS
    if "active_fiscal_year" not in st.session_state:
        st.session_state.active_fiscal_year = current_fiscal_year()
    if "active_contracting_office" not in st.session_state:
        st.session_state.active_contracting_office = ALL_CONTRACTING_OFFICES
    if "active_market_filters" not in st.session_state:
        st.session_state.active_market_filters = default_market_filters()
    if "active_selector_scope" not in st.session_state:
        st.session_state.active_selector_scope = None
    if "transaction_office_options_cache" not in st.session_state:
        st.session_state.transaction_office_options_cache = {}
    if "pending_office_option_stats" not in st.session_state:
        st.session_state.pending_office_option_stats = {}

    # Last analyzed dashboard state (analyzed_*). Rendered KPIs/charts use these values.
    if "analyzed_agency" not in st.session_state:
        st.session_state.analyzed_agency = None
    if "analyzed_bureau" not in st.session_state:
        st.session_state.analyzed_bureau = None
    if "analyzed_year" not in st.session_state:
        st.session_state.analyzed_year = None
    if "analyzed_contracting_office" not in st.session_state:
        st.session_state.analyzed_contracting_office = ALL_CONTRACTING_OFFICES
    if "analyzed_market_filters" not in st.session_state:
        st.session_state.analyzed_market_filters = default_market_filters()
    if "analyzed_office_option_stats" not in st.session_state:
        st.session_state.analyzed_office_option_stats = {}
    if "dashboard_started" not in st.session_state:
        st.session_state.dashboard_started = False
    if "solicitation_user_filter_overrides" not in st.session_state:
        st.session_state.solicitation_user_filter_overrides = {}
    if "solicitation_removed_filter_fields" not in st.session_state:
        st.session_state.solicitation_removed_filter_fields = []
    if "solicitation_scope_applied" not in st.session_state:
        st.session_state.solicitation_scope_applied = False
    if "solicitation_scope_review_open" not in st.session_state:
        st.session_state.solicitation_scope_review_open = False
    if "solicitation_comparable_market" not in st.session_state:
        st.session_state.solicitation_comparable_market = False
    if "solicitation_extraction_status" not in st.session_state:
        st.session_state.solicitation_extraction_status = "idle"
    if "solicitation_mapping_status" not in st.session_state:
        st.session_state.solicitation_mapping_status = "not_started"
    if "solicitation_review_status" not in st.session_state:
        st.session_state.solicitation_review_status = "not_ready"

    agency_records = fetch_toptier_agencies()
    if not agency_records:
        st.error("USAspending agency registry is unavailable. Please refresh and try again.")
        st.stop()
    active_record = agency_record_by_name(agency_records, st.session_state.active_agency)
    st.session_state.active_agency = active_record["agency_name"]
    st.session_state.active_toptier_code = active_record["toptier_code"]

    if not st.session_state.dashboard_started:
        render_landing_page(agency_records)
        return

    # Render Sidebar Control Panel
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-title">Government Award Data, Simplified.</div>
            <div class="sidebar-subtitle">Choose an agency, then narrow the dashboard with its linked bureau and fiscal-year filters.</div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown('<div class="sidebar-section">Agency</div>', unsafe_allow_html=True)
        (
            active_agency,
            selected_bureau,
            selected_year,
            _active_toptier_code,
            selected_contracting_office,
            selected_market_filters,
        ) = render_market_selectors(
            agency_records
        )
        basic_scope_matches_analysis = (
            st.session_state.analyzed_agency == active_agency
            and st.session_state.analyzed_bureau == selected_bureau
            and st.session_state.analyzed_year == selected_year
        )
        if basic_scope_matches_analysis:
            st.markdown('<div class="sidebar-section">Refine Market</div>', unsafe_allow_html=True)
            selected_contracting_office, selected_market_filters = render_refine_market_selectors(
                active_agency,
                selected_bureau,
                selected_year,
            )
        has_completed_analysis = st.session_state.analyzed_agency is not None
        pending_filters_changed = has_completed_analysis and not (
            st.session_state.analyzed_agency == active_agency
            and st.session_state.analyzed_bureau == selected_bureau
            and st.session_state.analyzed_year == selected_year
            and st.session_state.analyzed_contracting_office == selected_contracting_office
            and normalize_market_filters(st.session_state.analyzed_market_filters) == selected_market_filters
        )
        analysis_button_label = "Update Analysis" if pending_filters_changed else "Run Data Analysis"
        analysis_triggered = st.button(analysis_button_label, type="primary", use_container_width=True)
        if pending_filters_changed and not analysis_triggered:
            st.info("Filters changed. Click Update Analysis to apply.")
        st.write("")
        st.divider()
        st.caption(f"Active agency: {active_agency}")
        if selected_bureau != ALL_BUREAUS:
            st.caption(f"Active bureau: {selected_bureau}")
        if selected_contracting_office != ALL_CONTRACTING_OFFICES:
            st.caption(f"Contracting office: {format_contracting_office_option(selected_contracting_office)}")
        st.divider()
        render_upload_solicitation_package_panel()
        render_solicitation_sidebar_scope()
        if not _should_show_solicitation_scope_review():
            render_load_solicitation_signals_panel(show_test_mode_note=False)

    # Lock in parameters when the button is clicked
    if analysis_triggered:
        mark_analysis_started(
            active_agency,
            selected_bureau,
            selected_year,
            selected_contracting_office,
            selected_market_filters,
        )
        pending_filters_changed = False

    loading_requested = bool(st.session_state.pop("analysis_loading_requested", False) or analysis_triggered)
    has_completed_analysis = st.session_state.analyzed_agency is not None
    if has_completed_analysis:
        displayed_agency = st.session_state.analyzed_agency
        displayed_bureau = st.session_state.analyzed_bureau
        displayed_year = int(st.session_state.analyzed_year)
        displayed_contracting_office = st.session_state.analyzed_contracting_office
        displayed_market_filters = normalize_market_filters(st.session_state.analyzed_market_filters)
        render_dashboard_header(displayed_agency, displayed_bureau)
        if _should_show_solicitation_scope_review():
            render_solicitation_scope_preview(
                agency_records,
                active_agency,
                selected_bureau,
                selected_year,
                show_advanced_upload=False,
            )
        else:
            render_solicitation_baseline_notice()
        loading_slot = st.empty() if loading_requested else None
        if loading_slot is not None:
            render_analysis_loading_card(loading_slot, displayed_year)
        if pending_filters_changed:
            st.info(
                "Current results are based on the last run. Click Update Analysis to apply sidebar changes."
            )
        try:
            analysis_summary = render_analysis_dashboard(
                displayed_agency,
                displayed_bureau,
                displayed_year,
                displayed_contracting_office,
                displayed_market_filters,
                loading_slot=loading_slot,
            )
        except Exception as exc:
            if loading_slot is not None:
                loading_slot.empty()
            st.error("Analysis could not complete for this scope. Try clearing refinements or running a broader agency-level query.")
            with st.expander("API Payloads / Debug"):
                st.json({"unexpected_error": str(exc)})
            return
        if loading_slot is not None:
            loading_slot.empty()
            if not analysis_summary.get("failed"):
                show_analysis_success(analysis_summary.get("transaction_count"))
    else:
        render_dashboard_header(active_agency, selected_bureau)
        if _should_show_solicitation_scope_review():
            render_solicitation_scope_preview(
                agency_records,
                active_agency,
                selected_bureau,
                selected_year,
                show_advanced_upload=False,
            )
        else:
            render_solicitation_baseline_notice()
        if st.session_state.pop("cache_cleared_notice", False):
            st.success("All cached data cleared. Run analysis again to fetch fresh USAspending data.")
        st.info("👈 Select your parameters in the Control Panel and click 'Run Data Analysis' to begin.")


if __name__ == "__main__":
    main()
