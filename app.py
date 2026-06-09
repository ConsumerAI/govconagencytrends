import csv
import hashlib
import html
import io
import json
import os
import re
import time
import zipfile
from datetime import date, timedelta
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from openai import OpenAI
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
ALL_POP_STATES = "All States"
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
AGENCY_ALIASES = {
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
            --text: #f4f7fb;
            --muted: #aab4c2;
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
            color: var(--muted) !important;
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
            color: #f4f7fb !important;
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
            color: #1A1A1A !important;
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
            color: #f4f7fb;
            font-size: 22px;
            font-weight: 850;
            line-height: 1.12;
            letter-spacing: 0;
            margin: 0 0 6px;
        }
        [data-testid="stSidebar"] .sidebar-subtitle {
            color: #aab4c2;
            font-size: 13px;
            line-height: 1.35;
            margin: 0 0 18px;
        }
        [data-testid="stSidebar"] .sidebar-section {
            color: #2dd4bf;
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
            color: rgba(170, 180, 194, 0.86);
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
            color: rgba(170, 180, 194, 0.88);
            font-size: 12px;
            font-weight: 700;
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
        .award-drilldown-table-wrap {
            width: 100%;
            overflow-x: auto;
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 10px;
            background: #ffffff;
            margin-top: 8px;
        }
        .award-drilldown-table {
            width: 100%;
            border-collapse: collapse;
            color: #1A1A1A;
            font-size: 13px;
            line-height: 1.35;
        }
        .award-drilldown-table th {
            position: sticky;
            top: 0;
            z-index: 1;
            background: #f3f6fb;
            color: #1A1A1A;
            font-weight: 850;
            text-align: left;
            white-space: nowrap;
            padding: 10px 12px;
            border-bottom: 1px solid rgba(15, 23, 42, 0.16);
        }
        .award-drilldown-table td {
            max-width: 260px;
            padding: 9px 12px;
            border-bottom: 1px solid rgba(15, 23, 42, 0.10);
            vertical-align: top;
        }
        .award-drilldown-table td:nth-child(3),
        .award-drilldown-table td:nth-child(8),
        .award-drilldown-table td:nth-child(9) {
            min-width: 220px;
        }
        .award-drilldown-table a {
            color: #0369a1;
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
            color: rgba(170, 180, 194, 0.90);
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
        .market-intel-note {
            display: flex;
            align-items: center;
            min-height: 84px;
            color: rgba(170, 180, 194, 0.92);
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
        [data-testid="stMetricValue"] {
            color: var(--text);
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
        "pop_state": ALL_POP_STATES,
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
        "states": [ALL_POP_STATES],
        "offices": [ALL_CONTRACTING_OFFICES],
        "funding_offices": [ALL_FUNDING_OFFICES],
    }


def normalize_market_filters(market_filters: dict | None) -> dict:
    normalized = default_market_filters()
    if isinstance(market_filters, dict):
        normalized.update({key: value for key, value in market_filters.items() if value})
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
        "state": option_label(market_filters["pop_state"], ALL_POP_STATES, "All States", "State:"),
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
            market_filters["pop_state"] != ALL_POP_STATES,
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
    state_code, state_label = decode_option(market_filters["pop_state"])
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
        "place_of_performance_state": {
            "code": "" if state_code == ALL_POP_STATES else state_code,
            "label": state_label or ALL_POP_STATES,
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

    pop_state, _state_description = decode_option(market_filters["pop_state"])
    if pop_state and pop_state != ALL_POP_STATES:
        filters["place_of_performance_locations"] = [{"country": "USA", "state": pop_state}]

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


def post_usaspending(endpoint: str, payload: dict) -> tuple[dict | None, str | None]:
    try:
        response = requests.post(
            f"{BASE_URL}{endpoint}",
            json=payload,
            headers=request_headers(),
            timeout=18,
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


def transaction_pop_state_parts(item: dict) -> tuple[str, str]:
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
        state = clean_office_value(first_present(place, ["state", "state_code"]))
        state_name = clean_office_value(first_present(place, ["state_name", "name"]))
        return state, state_name
    return clean_office_value(first_present(item, ["Place of Performance State Code", "pop_state"])), ""


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

    pop_state, _state_description = decode_option(market_filters["pop_state"])
    if pop_state and pop_state != ALL_POP_STATES:
        item_state, _item_state_description = transaction_pop_state_parts(item)
        if item_state.lower() != pop_state.lower():
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


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_category_filter_options(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    category: str,
    default_option: str,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> list[str]:
    payload = build_category_options_payload(
        agency_name,
        bureau_name,
        fiscal_year,
        category,
        market_filters=market_filters,
    )
    data, error = post_usaspending(f"/api/v2/search/spending_by_category/{category}/", payload)
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


def fetch_transaction_download_rows_with_diagnostics(payload: dict) -> tuple[list[dict], dict, str | None]:
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
            timeout=18,
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
            try:
                status_response = requests.get(status_url, headers=request_headers(), timeout=15)
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
        zip_response = requests.get(file_url, headers=request_headers(), timeout=30)
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


def fetch_transaction_download_rows_from_payload(payload: dict) -> list[dict]:
    rows, _diagnostic, _failure_mode = fetch_transaction_download_rows_with_diagnostics(payload)
    return rows


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_transaction_download_office_rows(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
    market_filters: dict | None = None,
    cache_version: str = APP_CACHE_VERSION,
) -> list[dict]:
    payload = build_transaction_download_office_payload(
        agency_name,
        bureau_name,
        fiscal_year,
        market_filters=market_filters,
    )
    return fetch_transaction_download_rows_from_payload(payload)


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
        f"""
        <section class="analysis-loading-card" role="status" aria-live="polite">
            <p class="analysis-loading-body">{html.escape(body)}</p>
            <div class="analysis-loading-detail">
                <span class="analysis-loading-spinner" aria-hidden="true"></span>
                <span>{html.escape(detail)}</span>
            </div>
            <div class="analysis-loading-stage">{stage_markup}</div>
            {count_markup}
            <div class="analysis-loading-footer">Accuracy first. No estimates — only reconciled USAspending data.</div>
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
            labels["state"],
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

    active_refinements = []
    defaults = []
    market_filters = normalize_market_filters(selected_market_filters)
    optional_config = [
        ("naics_code", ALL_NAICS_CODES, labels["naics"]),
        ("contract_type", ALL_CONTRACT_TYPES, labels["contract_type"]),
        ("psc_code", ALL_PRODUCT_SERVICE_CODES, labels["psc"]),
        ("set_aside_type", ALL_SET_ASIDE_TYPES, labels["set_aside"]),
        ("contracting_office", ALL_CONTRACTING_OFFICES, labels["contracting_office"]),
        ("funding_office", ALL_FUNDING_OFFICES, labels["funding_office"]),
        ("pop_state", ALL_POP_STATES, labels["state"]),
    ]
    for key, default_value, label in optional_config:
        current_value = selected_contracting_office if key == "contracting_office" else market_filters[key]
        if current_value != default_value:
            active_refinements.append((key, label))
        else:
            defaults.append(label)

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


def market_concentration_summary(transaction_df: pd.DataFrame, total_obligations: float) -> tuple[dict, dict]:
    empty_result = {
        "value": "N/A",
        "subtitle": "No contract obligations found for this scope.",
        "supporting_text": "No contract obligations found for this scope.",
        "classification": "",
        "donut_slice_data": [],
        "concentration_segments": [],
        "grouped_contractor_count": 0,
        "other_share_percentage": None,
    }
    if transaction_df.empty or "Obligation Amount" not in transaction_df.columns:
        return empty_result, {
            "total_obligations": round(float(total_obligations or 0.0), 2),
            "grouped_contractor_count": 0,
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

    grouped = (
        transaction_df.groupby("Contractor Name", as_index=False)["Obligation Amount"]
        .sum()
        .rename(columns={"Contractor Name": "contractor", "Obligation Amount": "amount"})
    )
    grouped = grouped[pd.to_numeric(grouped["amount"], errors="coerce").fillna(0).abs() >= 0.005]
    grouped = grouped.sort_values("amount", ascending=False).reset_index(drop=True)
    grouped_total = float(pd.to_numeric(grouped["amount"], errors="coerce").fillna(0).sum())
    top_5 = grouped.head(5)
    top_5_sum = float(pd.to_numeric(top_5["amount"], errors="coerce").fillna(0).sum())
    total = float(total_obligations or 0.0)
    other_sum = total - top_5_sum
    reconciles = abs(grouped_total - total) <= 0.01
    if not reconciles:
        print("ERROR: Market concentration total does not reconcile to KPI.")
    if total <= 0:
        summary = empty_result
        concentration_pct = None
    else:
        concentration_pct = (top_5_sum / total) * 100
        if concentration_pct >= 70:
            classification = "Highly concentrated market"
        elif concentration_pct >= 40:
            classification = "Moderately concentrated market"
        else:
            classification = "Fragmented market"
        summary = {
            "value": f"{concentration_pct:.1f}%",
            "subtitle": "Top 5 contractor share",
            "supporting_text": f"Top 5 contractors captured {concentration_pct:.1f}% of obligations in this scope.",
            "classification": classification,
        }
        if len(grouped) == 1:
            contractor_name = str(top_5.iloc[0].get("contractor") or "The only contractor")
            summary["subtitle"] = "Single-vendor scope"
            summary["supporting_text"] = f"{contractor_name} accounts for all obligations in this filtered market."
    donut_slice_data = [
        {
            "contractor": str(row.get("contractor") or "Unknown Contractor"),
            "amount": round(float(row.get("amount") or 0.0), 2),
            "display_amount": format_money(row.get("amount")),
        }
        for row in top_5.to_dict("records")
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
            "percentage": round((float(row["amount"] or 0.0) / total) * 100, 1) if total > 0 else 0.0,
        }
        for row in donut_slice_data
        if row["contractor"] != "All Other Contractors"
    ]
    other_pct = round((max(other_sum, 0.0) / total) * 100, 1) if total > 0 else 0.0
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
    summary["grouped_contractor_count"] = int(len(grouped))
    summary["other_share_percentage"] = other_pct if concentration_pct is not None else None
    debug = {
        "total_obligations": round(total, 2),
        "grouped_contractor_count": int(len(grouped)),
        "top_5_contractor_names": [str(row.get("contractor") or "") for row in top_5.to_dict("records")],
        "top_5_contractor_sums": [
            round(float(row.get("amount") or 0.0), 2) for row in top_5.to_dict("records")
        ],
        "top_5_contractor_percentages": [
            round((float(row.get("amount") or 0.0) / total) * 100, 1) if total > 0 else 0.0
            for row in top_5.to_dict("records")
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


def render_market_concentration_legend(legend_lines: list[str]) -> None:
    if not legend_lines:
        return
    safe_lines = []
    for line in legend_lines:
        if visible_text_contains_raw_html(line):
            print("ERROR: raw HTML detected in Market Concentration legend text.")
        safe_lines.append(clean_visible_market_text(line))
    st.markdown("\n".join(f"- {line}" for line in safe_lines))


def render_market_concentration_card(market_concentration: dict) -> None:
    visible_text_values = [
        str(market_concentration.get("value") or "N/A"),
        str(market_concentration.get("subtitle") or "Top 5 contractor share"),
        str(
            market_concentration.get("supporting_text")
            or market_concentration.get("subtitle")
            or "No contract obligations found for this scope."
        ),
        str(market_concentration.get("classification") or ""),
    ]
    if any(visible_text_contains_raw_html(text) for text in visible_text_values):
        print("ERROR: raw HTML detected in Market Concentration visible text.")
    value = clean_visible_market_text(visible_text_values[0])
    subtitle = clean_visible_market_text(visible_text_values[1])
    supporting_text = clean_visible_market_text(
        market_concentration.get("supporting_text")
        or market_concentration.get("subtitle")
        or "No contract obligations found for this scope."
    )
    classification = clean_visible_market_text(visible_text_values[3])
    try:
        top_share = float(value.replace("%", "")) if value != "N/A" else 0.0
    except ValueError:
        top_share = 0.0
    top_share = min(max(top_share, 0.0), 100.0)
    segments = market_concentration.get("concentration_segments") or []
    colors = ["#2dd4bf", "#38bdf8", "#a78bfa", "#f59e0b", "#fb7185", "#64748b"]
    segment_markup = ""
    legend_lines = []
    if value != "N/A" and segments:
        segment_parts = []
        for index, segment in enumerate(segments[:6]):
            color = colors[index % len(colors)]
            contractor = str(segment.get("contractor") or "Unknown Contractor")
            amount = str(segment.get("display_amount") or format_money(segment.get("amount")))
            percentage = float(segment.get("percentage") or 0.0)
            title = f"{contractor}: {amount} ({percentage:.1f}% of scope)"
            short_name = contractor if len(contractor) <= 42 else f"{contractor[:39].rstrip()}..."
            segment_parts.append(
                f'<div class="market-concentration-segment" style="width: {max(percentage, 0.0):.1f}%; '
                f'background: {color};" title="{html.escape(title)}"></div>'
            )
            legend_lines.append(f"{short_name} — {percentage:.1f}% · {amount}")
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
            <div class="market-intel-helper">{html.escape(classification)}</div>
        </section>
        """,
        unsafe_allow_html=True,
    )
    render_market_concentration_legend(legend_lines)


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
        "Top NAICS",
        bool(selected_naics_code and selected_naics_code != ALL_NAICS_CODES),
        "NAICS mix unavailable from returned transaction rows for this scope.",
    )
    psc_summary = top_lane_summary(
        psc_df,
        "PSC",
        "Top PSC",
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
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "Description": st.column_config.TextColumn(width="large"),
                "Obligated": st.column_config.TextColumn(width="small"),
                "% of Scope": st.column_config.TextColumn(width="small"),
                "Awards": st.column_config.NumberColumn(width="small"),
                "Contractors": st.column_config.NumberColumn(width="small"),
            },
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
        <div class="award-drilldown-table-wrap">
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
    control_cols = st.columns([1, 1, 1.4])
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
    pop_state, _state_label = decode_option(market_filters["pop_state"])
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
        "place_of_performance_state": (
            pop_state == ALL_POP_STATES
            or any(
                isinstance(location, dict) and location.get("state") == pop_state
                for location in filters.get("place_of_performance_locations", [])
            )
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

        st.divider()
        st.caption(f"Active agency: {active_agency}")
        if selected_bureau != ALL_BUREAUS:
            st.caption(f"Active bureau: {selected_bureau}")
        if selected_contracting_office != ALL_CONTRACTING_OFFICES:
            st.caption(f"Contracting office: {format_contracting_office_option(selected_contracting_office)}")


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
    state_options = [ALL_POP_STATES] + [
        encode_option(code, name) for code, name in sorted(STATE_OPTIONS.items(), key=lambda item: item[1])
    ]
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
    active_pop_state = active_market_filters["pop_state"]
    if active_pop_state not in state_options:
        active_pop_state = ALL_POP_STATES
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
        or active_pop_state != ALL_POP_STATES
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
        selected_pop_state = st.selectbox(
            "Place of Performance State",
            state_options,
            index=state_options.index(active_pop_state),
            format_func=lambda option: option if option == ALL_POP_STATES else format_code_description_option(option),
        )

    selected_market_filters = normalize_market_filters(
        {
            "naics_code": selected_naics,
            "contract_type": selected_contract_type,
            "psc_code": selected_psc,
            "set_aside_type": selected_set_aside,
            "funding_office": selected_funding_office,
            "pop_state": selected_pop_state,
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
    metric_cols = st.columns(3)
    with metric_cols[0]:
        metric_card(
            f"{selected_year_label} Contract Obligations",
            total_spend_value,
            "Prime contract award transactions",
            "#2dd4bf",
            helper_text=(
                "Shows contract obligations reported to USAspending for the selected fiscal year. "
                "Includes prime contract award transactions only. Excludes IDVs, grants, loans, "
                "direct payments, and other assistance awards."
            ),
        )
    with metric_cols[1]:
        metric_card(
            yoy_metric_label,
            yoy_delta_value,
            yoy_metric_subtitle,
            "#f59e0b" if (yoy_delta or 0) >= 0 else "#fb7185",
        )
    with metric_cols[2]:
        metric_card(
            "Contractors Shown",
            contractor_count_value,
            "Top recipient records shown",
            "#38bdf8",
        )

    award_metric_cols = st.columns(3)
    with award_metric_cols[0]:
        metric_card(
            "Active Award Ceiling",
            "Unavailable" if award_scope_error else format_money(award_totals["active_award_ceiling"]),
            f"{format_count(award_totals['award_count'])} unique awards deduped",
            "#a78bfa",
            helper_text=(
                "Award Scope View: one record per unique award, then summed. "
                "This uses potential_total_value_of_award as Award Ceiling, not obligated spend."
            ),
        )
    with award_metric_cols[1]:
        metric_card(
            "Current Award Value",
            "Unavailable" if award_scope_error else format_money(award_totals["current_award_value"]),
            "Deduped current_total_value_of_award",
            "#60a5fa",
            helper_text="Current Award Value is not obligated spend.",
        )
    with award_metric_cols[2]:
        metric_card(
            "Remaining Ceiling",
            "Unavailable" if award_scope_error else format_money(award_totals["remaining_ceiling"]),
            "Award Ceiling minus Current Award Value",
            "#f59e0b",
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

    with st.expander("Award Scope Diagnostic Probe"):
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

    st.write("")
    time_grain_key = f"obligation-time-grain-{time_grain_scope_key(active_agency, selected_bureau, int(selected_year), selected_contracting_office, selected_market_filters)}"
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
        key=f"leaderboard-mode-{time_grain_scope_key(active_agency, selected_bureau, int(selected_year), selected_contracting_office, selected_market_filters)}",
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
                "Current fiscal year is year-to-date and is not directly comparable to completed fiscal years."
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
    award_drilldown_debug.update(
        render_top_awards_drilldown(
            award_drilldown_df,
            time_grain_scope_key(
                active_agency,
                selected_bureau,
                int(selected_year),
                selected_contracting_office,
                selected_market_filters,
            ),
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
    negative_transaction_df = transaction_df[transaction_df["Obligation Amount"] < 0].copy()
    negative_obligation_total = float(negative_transaction_df["Obligation Amount"].sum())
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
    dashboard_debug_payload = {
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

    with st.expander("API Payloads"):
        if st.button("Clear All Cached Data", key="clear-all-cached-data"):
            clear_all_cached_dashboard_data()
            st.rerun()
        st.caption(f"App cache version: {APP_CACHE_VERSION}")
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
        if trend_error or vendor_error or transaction_error or award_scope_error:
            st.caption("Live API issues are surfaced above; no synthetic data is used.")

    st.markdown(
        '<div class="audit-heading">&#9888;&#65039; Negative Obligation / Termination Signal Log</div>',
        unsafe_allow_html=True,
    )
    if audit_df.empty:
        st.success(
            "No negative obligation or termination signal rows found for this cycle."
        )
    else:
        styled_audit_df = audit_df.style.set_properties(
            **{"color": "#1A1A1A", "background-color": "#FFFFFF"}
        )
        st.dataframe(styled_audit_df, use_container_width=True, hide_index=True)

    return {
        "transaction_count": processed_transaction_count,
        "unique_award_count": processed_unique_award_count,
        "failed": bool(transaction_error),
    }


def main() -> None:
    inject_styles()

    # Initialize baseline selector states
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

    # Persistent state tracking to prevent the dashboard from vanishing on chart interaction
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
        if st.session_state.pop("cache_cleared_notice", False):
            st.success("All cached data cleared. Run analysis again to fetch fresh USAspending data.")
        st.info("👈 Select your parameters in the Control Panel and click 'Run Data Analysis' to begin.")


if __name__ == "__main__":
    main()