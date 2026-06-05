import hashlib
import html
import io
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from openai import OpenAI
from pypdf import PdfReader


BASE_URL = "https://api.usaspending.gov"
AWARD_TYPE_CODES = ["A", "B", "C", "D"]
ALL_BUREAUS = "All Bureaus"
OPENAI_MODEL = "gpt-4o-mini"
AGENCY_EXTRACTION_PROMPT = (
    "Extract the exact formal federal Awarding Agency from this text. "
    "Return strictly as valid JSON: {'agency': '...'}. If unknown, return null."
)
DEFAULT_AGENCY_NAME = "Department of Defense"
DEFAULT_TOPTIER_CODE = "097"
FALLBACK_AGENCY_RECORDS = [
    {"agency_name": DEFAULT_AGENCY_NAME, "toptier_code": DEFAULT_TOPTIER_CODE, "abbreviation": "DOD"},
    {"agency_name": "Department of the Interior", "toptier_code": "014", "abbreviation": "DOI"},
    {"agency_name": "Environmental Protection Agency", "toptier_code": "020", "abbreviation": "EPA"},
    {"agency_name": "National Aeronautics and Space Administration", "toptier_code": "080", "abbreviation": "NASA"},
    {"agency_name": "National Science Foundation", "toptier_code": "490", "abbreviation": "NSF"},
]

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

AGENCY_SPEND_BASE = {
    "Department of Defense": 465_000_000_000,
    "Department of Veterans Affairs": 58_000_000_000,
    "Department of Health and Human Services": 44_000_000_000,
    "Department of Homeland Security": 28_000_000_000,
    "Department of Energy": 36_000_000_000,
    "National Science Foundation": 1_700_000_000,
    "Environmental Protection Agency": 3_500_000_000,
    "Nuclear Regulatory Commission": 950_000_000,
    "National Aeronautics and Space Administration": 18_000_000_000,
    "Department of Transportation": 16_000_000_000,
    "Department of Agriculture": 12_000_000_000,
    "Department of Justice": 11_000_000_000,
    "Department of State": 9_000_000_000,
    "Department of the Interior": 7_000_000_000,
    "Department of Commerce": 8_000_000_000,
    "Department of the Treasury": 10_000_000_000,
    "Department of Labor": 6_000_000_000,
    "Department of Education": 5_000_000_000,
    "General Services Administration": 22_000_000_000,
    "Agency for International Development": 7_500_000_000,
    "Social Security Administration": 4_800_000_000,
    "Office of Personnel Management": 2_200_000_000,
}

MOCK_CONTRACTORS = [
    "Lockheed Martin Corporation",
    "RTX Corporation",
    "Leidos Holdings, Inc.",
    "General Dynamics Corporation",
    "Booz Allen Hamilton Holding Corporation",
    "Northrop Grumman Corporation",
    "Science Applications International Corporation",
    "The Boeing Company",
    "CACI International Inc.",
    "HII Mission Technologies",
    "Accenture Federal Services LLC",
    "Deloitte Government & Public Services",
    "Peraton Inc.",
    "KBR, Inc.",
    "L3Harris Technologies, Inc.",
]


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
        return FALLBACK_AGENCY_RECORDS

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

    return sorted(records, key=lambda record: record["agency_name"]) or FALLBACK_AGENCY_RECORDS


@st.cache_data(ttl=60 * 60, show_spinner=False)
def fetch_subagencies(toptier_code: str, fiscal_year: int) -> list[str]:
    if not toptier_code:
        return [ALL_BUREAUS]

    all_results = []
    page = 1
    try:
        while page <= 100:
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
    return agency_records[0] if agency_records else FALLBACK_AGENCY_RECORDS[0]


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
            margin: 0 0 12px;
        }
        .landing-subtitle {
            color: var(--muted);
            font-size: 16px;
            line-height: 1.5;
            margin: 0 auto 22px;
            max-width: 620px;
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
            color: var(--text);
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
        .source-dot.mock {
            background: var(--amber);
            box-shadow: 0 0 18px var(--amber);
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


def stable_int(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def first_present(mapping: dict, keys: list[str]):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def format_money(value) -> str:
    try:
        amount = float(value)
    except (TypeError, ValueError):
        return "$0"
    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1_000_000_000_000:
        return f"{sign}${amount / 1_000_000_000_000:.1f}T"
    if amount >= 1_000_000_000:
        return f"{sign}${amount / 1_000_000_000:.1f}B"
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:.0f}M"
    if amount >= 1_000:
        return f"{sign}${amount / 1_000:.0f}K"
    return f"{sign}${amount:,.0f}"


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
        return [0], ["$0"]
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
    bureau_filter_name = resolve_bureau_filter_name(bureau_name)
    if bureau_filter_name:
        return [{"type": "awarding", "tier": "subtier", "name": bureau_filter_name}]
    return [{"type": "awarding", "tier": "toptier", "name": normalize_agency_name(agency_name)}]


def build_trends_payload(agency_name: str, bureau_name: str | None = None) -> dict:
    return {
        "group": "fiscal_year",
        "spending_level": "awards",
        "filters": {
            "agencies": agency_filter(agency_name, bureau_name),
            "award_type_codes": AWARD_TYPE_CODES,
        },
    }


def build_vendor_payload(agency_name: str, bureau_name: str | None = None) -> dict:
    return {
        "category": "recipient",
        "spending_level": "awards",
        "limit": 10,
        "page": 1,
        "filters": {
            "agencies": agency_filter(agency_name, bureau_name),
            "award_type_codes": AWARD_TYPE_CODES,
        },
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
    return (filtered or live_names or [DEFAULT_AGENCY_NAME])[:10]


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


def complete_fiscal_year() -> int:
    today = date.today()
    return today.year - 1 if today.month < 10 else today.year


def mock_trend_data(agency_name: str) -> pd.DataFrame:
    normalized = normalize_agency_name(agency_name)
    base = AGENCY_SPEND_BASE.get(normalized)
    if base is None:
        base = 4_000_000_000 + (stable_int(normalized) % 26_000_000_000)
    fiscal_years = list(range(complete_fiscal_year() - 7, complete_fiscal_year() + 1))
    seed = stable_int(normalized)
    rows = []
    for index, fiscal_year in enumerate(fiscal_years):
        growth = 0.965 + index * 0.018
        pulse = 0.94 + ((seed >> (index % 8)) & 15) / 120
        rows.append({"fiscal_year": fiscal_year, "amount": base * growth * pulse})
    return pd.DataFrame(rows)


def mock_vendor_data(agency_name: str, latest_total: float) -> tuple[pd.DataFrame, int]:
    seed = stable_int(normalize_agency_name(agency_name))
    contractors = MOCK_CONTRACTORS[:]
    offset = seed % len(contractors)
    contractors = contractors[offset:] + contractors[:offset]
    share_total = latest_total * (0.28 + (seed % 9) / 100)
    weights = [1 / (rank + 1.4) for rank in range(10)]
    weight_total = sum(weights)
    rows = [
        {
            "recipient": contractors[rank],
            "amount": share_total * weights[rank] / weight_total,
        }
        for rank in range(10)
    ]
    contractor_count = 750 + (seed % 4_800)
    return pd.DataFrame(rows).sort_values("amount", ascending=False), contractor_count


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_trends(agency_name: str, bureau_name: str | None) -> tuple[pd.DataFrame, dict, str, str | None]:
    payload = build_trends_payload(agency_name, bureau_name)
    data, error = post_usaspending("/api/v2/search/spending_over_time/", payload)
    if data:
        df = normalize_trend_response(data)
        if not df.empty:
            return df, payload, "Live USAspending.gov", None
    return mock_trend_data(agency_name), payload, "Realistic fallback", error or "No trend rows returned"


@st.cache_data(ttl=1800, show_spinner=False)
def fetch_vendors(
    agency_name: str,
    bureau_name: str | None,
    latest_total: float,
) -> tuple[pd.DataFrame, int, dict, str, str | None]:
    payload = build_vendor_payload(agency_name, bureau_name)
    data, error = post_usaspending("/api/v2/search/spending_by_category/recipient/", payload)
    if data:
        df, contractor_count = normalize_vendor_response(data)
        if not df.empty:
            return (
                df,
                contractor_count or len(df["recipient"].unique()),
                payload,
                "Live USAspending.gov",
                None,
            )
    df, contractor_count = mock_vendor_data(agency_name, latest_total)
    return df, contractor_count, payload, "Realistic fallback", error or "No vendor rows returned"


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


def metric_card(label: str, value: str, subtext: str, accent: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card" style="--accent: {accent};">
            <div class="metric-label">{html.escape(label)}</div>
            <div class="metric-value">{html.escape(value)}</div>
            <div class="metric-sub">{html.escape(subtext)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def source_chip(label: str) -> None:
    is_mock = "fallback" in label.lower()
    dot_class = "source-dot mock" if is_mock else "source-dot"
    st.markdown(
        f"""
        <div class="source-chip">
            <span class="{dot_class}"></span>
            <span>{html.escape(label)}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def make_trend_chart(df: pd.DataFrame, selected_year: int) -> go.Figure:
    chart_df = df.copy()
    chart_df["display_amount"] = chart_df["amount"].apply(format_money)
    tickvals, ticktext = money_ticks(chart_df["amount"].max())
    selected = chart_df[chart_df["fiscal_year"] == selected_year]
    selected_amount = float(selected["amount"].iloc[0]) if not selected.empty else 0

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=chart_df["fiscal_year"],
            y=chart_df["amount"],
            customdata=chart_df["display_amount"],
            mode="lines+markers",
            line=dict(color="#2dd4bf", width=4, shape="spline"),
            marker=dict(size=9, color="#f4f7fb", line=dict(color="#2dd4bf", width=2)),
            fill="tozeroy",
            fillcolor="rgba(45, 212, 191, 0.13)",
            hovertemplate="<b>FY %{x}</b><br>Obligations: %{customdata}<extra></extra>",
            name="Obligations",
        )
    )
    if selected_amount:
        fig.add_trace(
            go.Scatter(
                x=[selected_year],
                y=[selected_amount],
                mode="markers",
                marker=dict(size=16, color="#f59e0b", line=dict(color="#fff7ed", width=2)),
                hovertemplate=(
                    f"<b>Selected FY {selected_year}</b><br>"
                    f"Obligations: {format_money(selected_amount)}<extra></extra>"
                ),
                name="Selected FY",
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
            tickmode="linear",
            dtick=1,
            zeroline=False,
        ),
        yaxis=dict(
            title="Obligated Spend",
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
    chart_df["display_amount"] = chart_df["amount"].apply(format_money)
    tickvals, ticktext = money_ticks(chart_df["amount"].max())
    fig = go.Figure()
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
            hovertemplate="<b>%{y}</b><br>Funding: %{customdata}<extra></extra>",
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
            title="Funding Amount",
            gridcolor="rgba(255,255,255,0.08)",
            tickvals=tickvals,
            ticktext=ticktext,
            zeroline=False,
        ),
        yaxis=dict(title="", gridcolor="rgba(255,255,255,0.04)", automargin=True),
        showlegend=False,
    )
    return fig


def current_and_previous(df: pd.DataFrame, selected_year: int) -> tuple[float, float | None]:
    selected = df[df["fiscal_year"] == selected_year]
    current_total = float(selected["amount"].iloc[0]) if not selected.empty else 0
    previous = df[df["fiscal_year"] == selected_year - 1]
    previous_total = float(previous["amount"].iloc[0]) if not previous.empty else None
    return current_total, previous_total


def hide_sidebar_for_landing() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stSidebar"],
        [data-testid="collapsedControl"] {
            display: none !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_market_selectors(agency_records: list[dict]) -> tuple[str, str, int, str]:
    agency_options = agency_names_from_records(agency_records)
    if not agency_options:
        agency_records = FALLBACK_AGENCY_RECORDS
        agency_options = agency_names_from_records(agency_records)

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

    fiscal_year_options = list(range(complete_fiscal_year(), complete_fiscal_year() - 8, -1))
    selected_year = int(st.session_state.active_fiscal_year)
    if selected_year not in fiscal_year_options:
        selected_year = fiscal_year_options[0]
    with fiscal_year_slot:
        selected_year = st.selectbox(
            "Fiscal Year",
            fiscal_year_options,
            index=fiscal_year_options.index(selected_year),
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

    st.session_state.active_agency = active_agency
    st.session_state.active_toptier_code = active_toptier_code
    st.session_state.active_bureau = selected_bureau
    st.session_state.active_fiscal_year = int(selected_year)

    return active_agency, selected_bureau, int(selected_year), active_toptier_code


def main() -> None:
    inject_styles()

    if "active_agency" not in st.session_state:
        st.session_state.active_agency = DEFAULT_AGENCY_NAME
    if "active_toptier_code" not in st.session_state:
        st.session_state.active_toptier_code = DEFAULT_TOPTIER_CODE
    if "active_bureau" not in st.session_state:
        st.session_state.active_bureau = ALL_BUREAUS
    if "active_fiscal_year" not in st.session_state:
        st.session_state.active_fiscal_year = complete_fiscal_year()
    if "searched" not in st.session_state:
        st.session_state.searched = False

    agency_records = fetch_toptier_agencies()
    active_record = agency_record_by_name(agency_records, st.session_state.active_agency)
    st.session_state.active_agency = active_record["agency_name"]
    st.session_state.active_toptier_code = active_record["toptier_code"]

    if not st.session_state.searched:
        hide_sidebar_for_landing()
        st.markdown('<div class="landing-control-spacer"></div>', unsafe_allow_html=True)
        _left_col, center_col, _right_col = st.columns([1, 1.15, 1])
        with center_col:
            st.markdown(
                """
                <div class="landing-panel">
                    <h1 class="landing-title">Federal Spending Analysis Portal</h1>
                    <p class="landing-subtitle">Configure an agency, bureau, and fiscal year to generate live market intelligence from USAspending.gov.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.write("")
            render_market_selectors(agency_records)
            st.write("")
            if st.button("Generate Market Intelligence", use_container_width=True):
                st.session_state.searched = True
                st.rerun()
        return

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-title">Control Panel</div>
            <div class="sidebar-subtitle">Choose an agency, then narrow the dashboard with its linked bureau and fiscal-year filters.</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="sidebar-section">Agency</div>', unsafe_allow_html=True)
        active_agency, selected_bureau, selected_year, _active_toptier_code = render_market_selectors(
            agency_records
        )
        selected_agency = active_agency

    trend_df, trend_payload, trend_source, trend_error = fetch_trends(active_agency, selected_bureau)
    latest_total_for_vendor = float(trend_df["amount"].iloc[-1]) if not trend_df.empty else 0
    vendor_df, contractor_count, vendor_payload, vendor_source, vendor_error = fetch_vendors(
        active_agency,
        selected_bureau,
        latest_total_for_vendor,
    )

    with st.sidebar:
        st.divider()
        st.caption(f"Active agency: {selected_agency}")
        if selected_bureau != ALL_BUREAUS:
            st.caption(f"Active bureau: {selected_bureau}")

    source_label = (
        "Live USAspending.gov"
        if trend_source.startswith("Live") and vendor_source.startswith("Live")
        else "Realistic fallback data active"
    )

    safe_agency = html.escape(selected_agency)
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
    source_chip(source_label)

    current_total, previous_total = current_and_previous(trend_df, int(selected_year))
    yoy_delta = None
    if previous_total and previous_total > 0:
        yoy_delta = ((current_total - previous_total) / previous_total) * 100

    metric_cols = st.columns(3)
    with metric_cols[0]:
        metric_card(
            "Selected FY Total Spend",
            format_money(current_total),
            f"FY {selected_year} obligations",
            "#2dd4bf",
        )
    with metric_cols[1]:
        metric_card(
            "Year-over-Year Delta",
            format_delta(yoy_delta),
            "Compared with prior fiscal year",
            "#f59e0b" if (yoy_delta or 0) >= 0 else "#fb7185",
        )
    with metric_cols[2]:
        metric_card(
            "Unique Contractor Count",
            format_count(contractor_count),
            "Recipient records from ranking endpoint",
            "#38bdf8",
        )

    st.write("")
    chart_cols = st.columns([1.15, 1])
    with chart_cols[0]:
        st.plotly_chart(
            make_trend_chart(trend_df, int(selected_year)),
            use_container_width=True,
            config={"responsive": True},
        )
    with chart_cols[1]:
        st.plotly_chart(
            make_vendor_chart(vendor_df),
            use_container_width=True,
            config={"responsive": True},
        )

    with st.expander("API Payloads"):
        st.markdown("Historical Trends")
        st.json(trend_payload)
        st.markdown("Vendor Rankings")
        st.json(vendor_payload)
        if trend_error or vendor_error:
            st.caption("Fallback reason captured without surfacing raw service errors.")


if __name__ == "__main__":
    main()
