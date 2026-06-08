import hashlib
import html
import io
import json
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


BASE_URL = "https://api.usaspending.gov"
AWARD_TYPE_CODES = ["A", "B", "C", "D"]
AWARD_OR_IDV_FLAG = "AWARD"
ALL_BUREAUS = "All Bureaus"
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
CONTRACTING_OFFICE_SEPARATOR = "||"
TERMINATION_ACTION_MAP = {
    "E": "Default",
    "F": "Convenience",
    "N": "Cancellation",
}
TRANSACTION_FIELDS = [
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
]
BASE_TRANSACTION_FIELDS = [
    field for field in TRANSACTION_FIELDS if "Office" not in field
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
def fetch_toptier_agencies() -> list[dict]:
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
def fetch_subagencies(toptier_code: str, fiscal_year: int) -> list[str]:
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
        .audit-heading {
            color: var(--text);
            font-size: 22px;
            font-weight: 850;
            letter-spacing: 0;
            margin: 26px 0 12px;
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


def resolve_bureau_filter_name(bureau_name: str | None) -> str | None:
    if not bureau_name or bureau_name == ALL_BUREAUS:
        return None
    return bureau_name


def first_present(mapping: dict, keys: list[str]):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


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


def contract_award_filters(agency_name: str, bureau_name: str | None = None) -> dict:
    return {
        "agencies": agency_filter(agency_name, bureau_name),
        "award_type_codes": AWARD_TYPE_CODES,
        "award_or_idv_flag": AWARD_OR_IDV_FLAG,
    }


def build_trends_payload(
    agency_name: str,
    bureau_name: str | None = None,
    time_period: list[dict] | None = None,
) -> dict:
    filters = contract_award_filters(agency_name, bureau_name)
    if time_period:
        filters["time_period"] = time_period
    return {
        "group": "fiscal_year",
        "spending_level": "transactions",
        "filters": filters,
    }


def build_vendor_payload(agency_name: str, bureau_name: str | None = None) -> dict:
    return {
        "category": "recipient",
        "spending_level": "awards",
        "limit": 10,
        "page": 1,
        "filters": contract_award_filters(agency_name, bureau_name),
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
) -> dict:
    return {
        "filters": {
            **contract_award_filters(agency_name, bureau_name),
            "time_period": [{"start_date": start_date, "end_date": end_date}],
        },
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
def agency_autocomplete(search_text: str) -> list[str]:
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
        return f"{office_name} ({office_code})"
    return office_name or office_code or contracting_office


def transaction_contracting_office(item: dict) -> str:
    office_code, office_name = awarding_office_parts(item)
    if not office_name:
        return ""
    return encode_contracting_office(office_code, office_name)


def contracting_office_matches(item: dict, contracting_office: str | None) -> bool:
    if not contracting_office or contracting_office == ALL_CONTRACTING_OFFICES:
        return True
    selected_code, selected_name = decode_contracting_office(contracting_office)
    item_code, item_name = awarding_office_parts(item)
    if selected_code:
        return item_code.lower() == selected_code.lower()
    return bool(selected_name) and item_name.lower() == selected_name.lower()


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
        "Contractor Name",
        "Subagency / Bureau",
        "Mod",
        "Obligation Amount",
        "Action Code",
        "Action Type",
        "Contracting Office",
        "Description",
    ]
    normalized_rows = []
    for item in rows:
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
        action_code = parse_action_type_code(
            first_present(
                item,
                ["Action Type", "ActionType", "action_type", "action_type_code", "action_type_code_desc"],
            )
        )
        normalized_rows.append(
            {
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
                "Obligation Amount": amount,
                "Action Code": action_code,
                "Action Type": classify_cancellation_description(description)
                or TERMINATION_ACTION_MAP.get(action_code, action_code or "Unspecified"),
                "Contracting Office": format_contracting_office_option(transaction_contracting_office(item))
                if transaction_contracting_office(item)
                else "Unspecified Office",
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


def dataframe_has_spend(df: pd.DataFrame, amount_column: str) -> bool:
    if df.empty or amount_column not in df.columns:
        return False
    numeric_amounts = pd.to_numeric(df[amount_column], errors="coerce").fillna(0)
    return bool(numeric_amounts.abs().sum() > 0)


@st.cache_data(ttl=2592000)
def fetch_transaction_page(
    agency_name: str,
    bureau_name: str | None,
    start_date: str,
    end_date: str,
    page: int,
) -> tuple[dict | None, str | None, dict]:
    payload = build_transaction_payload(
        agency_name,
        bureau_name,
        start_date,
        end_date,
        page=page,
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
) -> tuple[pd.DataFrame, dict, str, str | None]:
    payload = build_trends_payload(agency_name, bureau_name)
    data, error = post_usaspending("/api/v2/search/spending_over_time/", payload)
    if data:
        df = normalize_trend_response(data)
        if dataframe_has_spend(df, "amount"):
            return df, payload, "Live USAspending.gov", None

    if bureau_filter_active(bureau_name):
        fallback_payload = build_trends_payload(agency_name, None)
        fallback_data, fallback_error = post_usaspending("/api/v2/search/spending_over_time/", fallback_payload)
        if fallback_data:
            fallback_df = normalize_trend_response(fallback_data)
            if dataframe_has_spend(fallback_df, "amount"):
                return fallback_df, fallback_payload, "Live USAspending.gov (top-tier fallback)", None
        return (
            pd.DataFrame(columns=["fiscal_year", "amount"]),
            fallback_payload,
            "USAspending.gov error",
            fallback_error or error or "No trend rows returned",
        )

    return pd.DataFrame(columns=["fiscal_year", "amount"]), payload, "USAspending.gov error", error or "No trend rows returned"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_trend_period_total(
    agency_name: str,
    bureau_name: str | None,
    start_date: str,
    end_date: str,
) -> tuple[float | None, dict, str | None]:
    payload = build_trends_payload(
        agency_name,
        bureau_name,
        time_period=[{"start_date": start_date, "end_date": end_date}],
    )
    data, error = post_usaspending("/api/v2/search/spending_over_time/", payload)
    if data:
        df = normalize_trend_response(data)
        if not df.empty:
            return float(df["amount"].sum()), payload, None
    return None, payload, error or "No trend rows returned"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vendors(
    agency_name: str,
    bureau_name: str | None,
) -> tuple[pd.DataFrame, int, dict, str, str | None]:
    payload = build_vendor_payload(agency_name, bureau_name)
    data, error = post_usaspending("/api/v2/search/spending_by_category/recipient/", payload)
    if data:
        df, contractor_count = normalize_vendor_response(data)
        if dataframe_has_spend(df, "amount"):
            return (
                df,
                contractor_count or len(df["recipient"].unique()),
                payload,
                "Live USAspending.gov",
                None,
            )

    if bureau_filter_active(bureau_name):
        fallback_payload = build_vendor_payload(agency_name, None)
        fallback_data, fallback_error = post_usaspending("/api/v2/search/spending_by_category/recipient/", fallback_payload)
        if fallback_data:
            fallback_df, fallback_contractor_count = normalize_vendor_response(fallback_data)
            if dataframe_has_spend(fallback_df, "amount"):
                return (
                    fallback_df,
                    fallback_contractor_count or len(fallback_df["recipient"].unique()),
                    fallback_payload,
                    "Live USAspending.gov (top-tier fallback)",
                    None,
                )
        return (
            pd.DataFrame(columns=["recipient", "amount"]),
            0,
            fallback_payload,
            "USAspending.gov error",
            fallback_error or error or "No vendor rows returned",
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
) -> tuple[list[dict], dict, str | None, int, float]:
    master_rows = []
    payload_log = build_transaction_payload(
        agency_name,
        bureau_name,
        start_date,
        end_date,
        page=1,
    )
    if contracting_office and contracting_office != ALL_CONTRACTING_OFFICES:
        office_code, office_name = decode_contracting_office(contracting_office)
        payload_log["client_side_filter"] = {
            "contracting_office_code": office_code,
            "contracting_office_name": office_name,
        }
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
            row for row in page_rows if contracting_office_matches(row, contracting_office)
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
    progress_text=None,
) -> tuple[pd.DataFrame, dict, str, str | None]:
    start_date, end_date = fiscal_year_date_range(fiscal_year)
    if progress_text is None:
        progress_text = st.empty()

    try:
        master_rows, payload_log, first_error, records_seen, obligation_magnitude = fetch_transaction_pages(
            agency_name,
            bureau_name,
            start_date,
            end_date,
            progress_text,
            contracting_office=contracting_office,
            include_positive=include_positive,
        )
        transaction_df = normalize_transaction_response(master_rows)

        if (
            bureau_filter_active(bureau_name)
            and (not contracting_office or contracting_office == ALL_CONTRACTING_OFFICES)
            and (records_seen == 0 or obligation_magnitude == 0)
        ):
            (
                fallback_rows,
                fallback_payload,
                fallback_error,
                fallback_records_seen,
                fallback_obligation_magnitude,
            ) = fetch_transaction_pages(
                agency_name,
                None,
                start_date,
                end_date,
                progress_text,
                contracting_office=contracting_office,
                include_positive=include_positive,
            )
            fallback_df = normalize_transaction_response(fallback_rows)
            if fallback_records_seen > 0 and fallback_obligation_magnitude > 0:
                fallback_source = "Live USAspending.gov (top-tier fallback)" if fallback_error is None else "Partial USAspending.gov (top-tier fallback)"
                return fallback_df, fallback_payload, fallback_source, fallback_error
            return (
                fallback_df,
                fallback_payload,
                "USAspending.gov error",
                fallback_error or first_error or "No transaction rows returned",
            )
    finally:
        progress_text.empty()

    source = "Live USAspending.gov" if first_error is None else "Partial USAspending.gov"
    return transaction_df, payload_log, source, first_error


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_transaction_period_total(
    agency_name: str,
    bureau_name: str | None,
    start_date: str,
    end_date: str,
    contracting_office: str | None = None,
) -> tuple[float | None, dict, str | None]:
    rows, payload_log, first_error, records_seen, _obligation_magnitude = fetch_transaction_pages(
        agency_name,
        bureau_name,
        start_date,
        end_date,
        contracting_office=contracting_office,
        include_positive=True,
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


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def fetch_contracting_offices(
    agency_name: str,
    bureau_name: str | None,
    fiscal_year: int,
) -> list[str]:
    if not bureau_name or bureau_name == ALL_BUREAUS:
        return []

    start_date, end_date = fiscal_year_date_range(fiscal_year)
    offices_by_key = {}
    seen_page_signatures = set()
    page = 1
    has_next = True

    while has_next:
        data, error, _payload = fetch_transaction_page(
            agency_name,
            bureau_name,
            start_date,
            end_date,
            page=page,
        )
        if error or not data:
            break

        page_rows = data.get("results") or []
        if not page_rows:
            break

        signature = transaction_page_signature(page_rows)
        if signature in seen_page_signatures:
            break
        seen_page_signatures.add(signature)

        for row in page_rows:
            office_code, office_name = awarding_office_parts(row)
            if not office_name:
                continue
            dedupe_key = office_code or office_name.lower()
            offices_by_key[dedupe_key] = (office_code, office_name)

        has_next = response_has_next(data.get("page_metadata") or {})
        page += 1
        if page > 100:
            break

    office_options = [
        encode_contracting_office(office_code, office_name)
        for office_code, office_name in sorted(
            offices_by_key.values(),
            key=lambda office: office[1].lower(),
        )
    ]
    return [ALL_CONTRACTING_OFFICES] + office_options if office_options else []


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


def make_trend_chart(df: pd.DataFrame, selected_year: int) -> go.Figure:
    chart_df = df.copy()
    chart_df["display_amount"] = chart_df["amount"].apply(format_money_with_full)
    chart_df["fiscal_year_label"] = chart_df["fiscal_year"].apply(fiscal_year_label)
    max_value = float(chart_df["amount"].max() or 0) if not chart_df.empty else 0
    tickvals, ticktext = money_ticks(max_value)

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
        fig.add_trace(
            go.Scatter(
                x=chart_df["fiscal_year_label"],
                y=chart_df["amount"],
                customdata=chart_df["display_amount"],
                mode="lines+markers",
                line=dict(color="#2dd4bf", width=4, shape="spline"),
                marker=dict(size=9, color="#f4f7fb", line=dict(color="#2dd4bf", width=2)),
                hovertemplate="<b>%{x}</b><br>Contract Obligations: %{customdata}<extra></extra>",
                name="Contract Obligations",
            )
        )
    fig.update_layout(
        title=dict(text="Spend Trend", font=dict(size=18, color="#f4f7fb")),
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
        showlegend=False,
    )
    return fig


def make_vendor_chart(df: pd.DataFrame) -> go.Figure:
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
                hovertemplate="<b>%{y}</b><br>Obligated Amount: %{customdata}<extra></extra>",
            )
        )
    fig.update_layout(
        title=dict(text="Top Contractor Leaderboard", font=dict(size=18, color="#f4f7fb")),
        height=430,
        margin=dict(l=20, r=20, t=30, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dce5ef", family="Inter, Segoe UI, Arial, sans-serif"),
        xaxis=dict(
            title="Obligated Amount",
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
) -> tuple[float | None, str, str]:
    if selected_year == current_fiscal_year():
        prior_start, prior_end = prior_ytd_date_range(selected_year)
        if office_filter_active:
            previous_total, _payload, _error = fetch_transaction_period_total(
                active_agency,
                selected_bureau,
                prior_start,
                prior_end,
                contracting_office=selected_contracting_office,
            )
        else:
            previous_total, _payload, _error = fetch_trend_period_total(
                active_agency,
                selected_bureau,
                prior_start,
                prior_end,
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
        .sort_values("amount", ascending=False)
        .head(10)
    )
    return vendor_df


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
) -> None:
    st.session_state.dashboard_started = True
    st.session_state.analyzed_agency = active_agency
    st.session_state.analyzed_bureau = selected_bureau
    st.session_state.analyzed_year = int(selected_year)
    st.session_state.analyzed_contracting_office = selected_contracting_office


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
        ) = render_market_selectors(
            agency_records
        )
        st.write("")
        if st.button("Run Data Analysis", type="primary", use_container_width=True):
            mark_analysis_started(active_agency, selected_bureau, selected_year, selected_contracting_office)
            st.rerun()

        st.divider()
        st.caption(f"Active agency: {active_agency}")
        if selected_bureau != ALL_BUREAUS:
            st.caption(f"Active bureau: {selected_bureau}")
        if selected_contracting_office != ALL_CONTRACTING_OFFICES:
            st.caption(f"Contracting office: {format_contracting_office_option(selected_contracting_office)}")


def render_market_selectors(agency_records: list[dict]) -> tuple[str, str, int, str, str]:
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

    bureau_options = get_bureau_options(active_toptier_code, selected_year)
    active_bureau = st.session_state.active_bureau
    if active_bureau not in bureau_options:
        active_bureau = ALL_BUREAUS

    with bureau_slot:
        selected_bureau = st.selectbox(
            "Subagency / Bureau",
            bureau_options,
            index=bureau_options.index(active_bureau),
        )

    selected_contracting_office = ALL_CONTRACTING_OFFICES
    if selected_bureau != ALL_BUREAUS:
        office_options = fetch_contracting_offices(active_agency, selected_bureau, int(selected_year))
        if office_options:
            selected_contracting_office = st.selectbox(
                "Contracting Office",
                office_options,
                index=0,
                format_func=format_contracting_office_option,
                help="Optional. Narrow results to a specific office that issued the awards.",
            )
            st.caption("Optional. Narrow results to a specific office that issued the awards.")
        else:
            st.caption("No contracting office breakdown available for this selection.")

    st.session_state.active_agency = active_agency
    st.session_state.active_toptier_code = active_toptier_code
    st.session_state.active_bureau = selected_bureau
    st.session_state.active_fiscal_year = int(selected_year)
    st.session_state.active_contracting_office = selected_contracting_office

    return active_agency, selected_bureau, int(selected_year), active_toptier_code, selected_contracting_office


def render_dashboard_header(active_agency: str, selected_bureau: str | None) -> None:
    safe_agency = html.escape(active_agency)
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
) -> None:
    office_filter_active = selected_contracting_office != ALL_CONTRACTING_OFFICES
    transaction_df = pd.DataFrame()
    transaction_payload = {}
    transaction_error = None

    if office_filter_active:
        progress_text = st.empty()
        transaction_df, transaction_payload, _transaction_source, transaction_error = fetch_transactions(
            active_agency,
            selected_bureau,
            int(selected_year),
            contracting_office=selected_contracting_office,
            include_positive=True,
            progress_text=progress_text,
        )
        trend_df = transaction_trend_dataframe(transaction_df, int(selected_year))
        vendor_df = transaction_vendor_dataframe(transaction_df)
        trend_payload = transaction_payload
        vendor_payload = transaction_payload
        trend_source = "Client-filtered transaction records"
        vendor_source = "Client-filtered transaction records"
        trend_error = transaction_error
        vendor_error = transaction_error
    else:
        trend_df, trend_payload, trend_source, trend_error = fetch_trends(active_agency, selected_bureau)
        vendor_df, contractor_count, vendor_payload, vendor_source, vendor_error = fetch_vendors(
            active_agency,
            selected_bureau,
        )

    macro_live = trend_source.startswith("Live") and vendor_source.startswith("Live")
    source_label = (
        "Live USAspending.gov macro endpoints"
        if macro_live
        else "USAspending.gov macro endpoint issue"
    )

    source_chip(source_label)
    st.caption(
        "Default view shows prime contract transaction obligations from USAspending. "
        "These represent obligated contract spending, not necessarily cash payments/outlays."
    )
    if office_filter_active:
        st.caption(f"Contracting office: {format_contracting_office_option(selected_contracting_office)}")
    if trend_error:
        st.error(f"Historical trends API issue: {trend_error}")
    if vendor_error:
        st.error(f"Vendor rankings API issue: {vendor_error}")

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
    )
    yoy_delta = None
    if previous_total and previous_total > 0:
        yoy_delta = ((current_total - previous_total) / previous_total) * 100
    total_spend_value = "Unavailable" if trend_error else format_money(current_total)
    yoy_delta_value = "Unavailable" if trend_error else format_delta(yoy_delta)
    contractor_count_value = "Unavailable" if vendor_error else format_count(len(vendor_df))

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

    st.write("")
    chart_cols = st.columns([1.15, 1])
    with chart_cols[0]:
        st.plotly_chart(
            make_trend_chart(trend_df, int(selected_year)),
            use_container_width=True,
            config={"responsive": True, "displayModeBar": False},
        )
    with chart_cols[1]:
        if vendor_df.empty:
            st.error("Top Contractor Leaderboard unavailable from USAspending.gov.")
        else:
            st.plotly_chart(
                make_vendor_chart(vendor_df),
                use_container_width=True,
                config={"responsive": True, "displayModeBar": False},
            )

    if not office_filter_active:
        st.write("")
        progress_text = st.empty()
        transaction_df, transaction_payload, _transaction_source, transaction_error = fetch_transactions(
            active_agency,
            selected_bureau,
            int(selected_year),
            contracting_office=None,
            progress_text=progress_text,
        )
    if transaction_error:
        st.error(f"Transaction registry API issue: {transaction_error}")
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

    with st.expander("API Payloads"):
        st.markdown("Historical Trends")
        st.json(trend_payload)
        st.markdown("Vendor Rankings")
        st.json(vendor_payload)
        st.markdown("Transaction Negative Obligations")
        st.json(transaction_payload)
        if trend_error or vendor_error or transaction_error:
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

    # Persistent state tracking to prevent the dashboard from vanishing on chart interaction
    if "analyzed_agency" not in st.session_state:
        st.session_state.analyzed_agency = None
    if "analyzed_bureau" not in st.session_state:
        st.session_state.analyzed_bureau = None
    if "analyzed_year" not in st.session_state:
        st.session_state.analyzed_year = None
    if "analyzed_contracting_office" not in st.session_state:
        st.session_state.analyzed_contracting_office = ALL_CONTRACTING_OFFICES
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
        ) = render_market_selectors(
            agency_records
        )
        st.write("")
        analysis_triggered = st.button("Run Data Analysis", type="primary", use_container_width=True)
        st.divider()
        st.caption(f"Active agency: {active_agency}")
        if selected_bureau != ALL_BUREAUS:
            st.caption(f"Active bureau: {selected_bureau}")
        if selected_contracting_office != ALL_CONTRACTING_OFFICES:
            st.caption(f"Contracting office: {format_contracting_office_option(selected_contracting_office)}")

    # Main Screen: Headers stay permanently fixed right here
    render_dashboard_header(active_agency, selected_bureau)

    # Lock in parameters when the button is clicked
    if analysis_triggered:
        mark_analysis_started(active_agency, selected_bureau, selected_year, selected_contracting_office)

    # Only show results if current sidebar selections match what was explicitly analyzed
    if (st.session_state.analyzed_agency == active_agency and 
        st.session_state.analyzed_bureau == selected_bureau and 
        st.session_state.analyzed_year == selected_year and
        st.session_state.analyzed_contracting_office == selected_contracting_office):
        
        render_analysis_dashboard(active_agency, selected_bureau, selected_year, selected_contracting_office)
    else:
        st.info("👈 Select your parameters in the Control Panel and click 'Run Data Analysis' to begin.")


if __name__ == "__main__":
    main()