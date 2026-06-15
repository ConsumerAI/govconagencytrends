from __future__ import annotations

import os
from pathlib import Path

DEFAULT_DATA_DIR = "data"
RUN_ID_PREFIX = "run_"


def get_project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_project_env() -> None:
    """Load project-root `.env` without overriding existing OS environment variables."""
    if os.getenv("GOVCON_DISABLE_DOTENV", "").strip().lower() in {"1", "true", "yes"}:
        return
    custom_path = os.getenv("GOVCON_DOTENV_PATH", "").strip()
    env_path = Path(custom_path) if custom_path else get_project_root() / ".env"
    if not env_path.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path, override=False)
    except ImportError:
        return


def get_openai_api_key() -> str | None:
    key = (os.getenv("OPENAI_API_KEY") or "").strip()
    return key or None


def has_openai_api_key() -> bool:
    return bool(get_openai_api_key())


def live_openai_tests_enabled() -> bool:
    return os.getenv("GOVCON_LIVE_OPENAI_TESTS", "").strip().lower() in {"1", "true", "yes"}


def get_data_dir() -> Path:
    raw = (os.getenv("GOVCON_DATA_DIR") or os.getenv("DATA_DIR") or DEFAULT_DATA_DIR).strip()
    return Path(raw).resolve()


def get_run_dir(run_id: str) -> Path:
    return get_data_dir() / "runs" / run_id


def sources_dir(run_id: str) -> Path:
    return get_run_dir(run_id) / "sources"


def manifest_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "manifest.json"


def diagnostics_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "diagnostics.json"


def process_sources_signal_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "signals" / "process_sources_v1.json"


def corpus_v1_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "corpus" / "corpus.v1.json"


def full_corpus_text_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "corpus" / "full_corpus_text.txt"


def source_storage_key(run_id: str, sha256: str, ext: str) -> str:
    return f"runs/{run_id}/sources/{sha256}.{ext}"


def signals_json_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "signals" / "signals.json"


def resolved_signals_json_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "signals" / "resolved_signals.json"


def parity_report_json_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "parity_report.json"


def parity_report_md_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "parity_report.md"


def documents_dir(run_id: str) -> Path:
    return get_run_dir(run_id) / "documents"


def document_dir(run_id: str, source_id: str) -> Path:
    return documents_dir(run_id) / source_id


def docset_manifest_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "docset" / "docset.manifest.v1.json"


def docset_diagnostics_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "docset" / "diagnostics.json"


def merged_docset_signals_path(run_id: str) -> Path:
    return get_run_dir(run_id) / "docset" / "merged_signals.json"


def package_extraction_dir(run_id: str) -> Path:
    return get_run_dir(run_id) / "package_extraction"


def package_corpus_text_path(run_id: str) -> Path:
    return package_extraction_dir(run_id) / "package_corpus.txt"


def package_extraction_raw_path(run_id: str) -> Path:
    return package_extraction_dir(run_id) / "package_extraction.raw.json"


def package_extraction_validated_path(run_id: str) -> Path:
    return package_extraction_dir(run_id) / "package_extraction.validated.json"


def validation_findings_path(run_id: str) -> Path:
    return package_extraction_dir(run_id) / "validation_findings.json"


def package_cache_dir() -> Path:
    return get_data_dir() / "package_cache"


def get_extraction_mode() -> str:
    mode = (os.getenv("GOVCON_EXTRACTION_MODE") or "package_llm").strip().lower()
    return mode if mode in {"package_llm", "legacy"} else "package_llm"


def get_extraction_model() -> str:
    return (os.getenv("GOVCON_EXTRACTION_MODEL") or os.getenv("AEGIS_LLM_MODEL") or "gpt-5.5").strip()


def get_extraction_profile() -> str:
    profile = (os.getenv("GOVCON_EXTRACTION_PROFILE") or "market_scope_fast").strip()
    if profile in {"market_scope_fast", "package_llm_full"}:
        return profile
    return "market_scope_fast"


def get_extraction_reasoning_effort() -> str:
    profile = get_extraction_profile()
    if profile == "market_scope_fast":
        return (os.getenv("GOVCON_FAST_REASONING_EFFORT") or "low").strip()
    return (os.getenv("GOVCON_EXTRACTION_REASONING_EFFORT") or os.getenv("AEGIS_LLM_REASONING_EFFORT") or "medium").strip()


def get_merge_run_id() -> str | None:
    raw = (os.getenv("GOVCON_MERGE_RUN_ID") or "").strip()
    return raw or None


def package_force_refresh() -> bool:
    return os.getenv("GOVCON_FORCE_PACKAGE_REFRESH", "").strip().lower() in {"1", "true", "yes"}


def get_extraction_timeout_sec() -> float:
    raw = (os.getenv("GOVCON_GPT_TIMEOUT_SEC") or os.getenv("GOVCON_EXTRACTION_TIMEOUT_SEC") or "120").strip()
    try:
        return max(30.0, float(raw))
    except ValueError:
        return 120.0


def get_extraction_subprocess_timeout_sec() -> float:
    raw = (os.getenv("GOVCON_EXTRACTION_SUBPROCESS_TIMEOUT_SEC") or "180").strip()
    try:
        return max(60.0, float(raw))
    except ValueError:
        return 180.0


load_project_env()
