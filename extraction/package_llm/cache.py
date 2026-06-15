from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

from extraction.config import (
    get_extraction_mode,
    get_extraction_model,
    get_extraction_profile,
    get_extraction_reasoning_effort,
    package_cache_dir,
    package_extraction_raw_path,
    package_extraction_validated_path,
    resolved_signals_json_path,
    validation_findings_path,
)
from extraction.ingest.hash import sha256_hex
from extraction.package_llm.build_info import cache_versions_match, collect_build_info
from extraction.package_llm.versions import PROFILE_VERSIONS
from extraction.persist import read_json, write_json
from extraction.types import SourceRecord


def package_fingerprint(sources: list[SourceRecord]) -> str:
    parts = sorted(f"{source.sha256}:{source.original_filename}" for source in sources if not source.did_dedupe)
    return sha256_hex("|".join(parts).encode("utf-8"))


def cache_key(sources: list[SourceRecord], *, profile: str | None = None) -> str:
    profile = profile or get_extraction_profile()
    versions = PROFILE_VERSIONS.get(profile, PROFILE_VERSIONS["package_llm_full"])
    payload = "|".join(
        [
            package_fingerprint(sources),
            get_extraction_mode(),
            profile,
            versions["schemaVersion"],
            versions["promptVersion"],
            get_extraction_model(),
            get_extraction_reasoning_effort(),
            versions["corpusBuilderVersion"],
            versions["formExtractorVersion"],
            versions["validatorVersion"],
            versions["canonicalizerVersion"],
            versions["resolverVersion"],
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def cache_dir_for_key(key: str) -> Path:
    return package_cache_dir() / key


def load_cached_artifacts(key: str, *, profile: str | None = None) -> dict[str, Any] | None:
    profile = profile or get_extraction_profile()
    directory = cache_dir_for_key(key)
    raw_path = directory / "package_extraction.raw.json"
    validated_path = directory / "package_extraction.validated.json"
    resolved_path = directory / "resolved_signals.json"
    if not raw_path.exists() or not validated_path.exists() or not resolved_path.exists():
        return None
    diagnostics = read_json(directory / "diagnostics.json") if (directory / "diagnostics.json").exists() else {}
    build_info = diagnostics.get("buildInfo") if isinstance(diagnostics, dict) else None
    if not cache_versions_match(build_info if isinstance(build_info, dict) else None, profile=profile):
        return None
    return {
        "raw": read_json(raw_path),
        "validated": read_json(validated_path),
        "resolved": read_json(resolved_path),
        "validation_findings": read_json(directory / "validation_findings.json")
        if (directory / "validation_findings.json").exists()
        else [],
        "diagnostics": diagnostics if isinstance(diagnostics, dict) else {},
        "cacheDir": str(directory),
    }


def write_cache_artifacts(
    key: str,
    *,
    raw: dict[str, Any],
    validated: dict[str, Any],
    resolved: dict[str, Any],
    validation_findings: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> Path:
    directory = cache_dir_for_key(key)
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory / "package_extraction.raw.json", raw)
    write_json(directory / "package_extraction.validated.json", validated)
    write_json(directory / "resolved_signals.json", resolved)
    write_json(directory / "validation_findings.json", validation_findings)
    write_json(directory / "diagnostics.json", diagnostics)
    build_info = diagnostics.get("buildInfo")
    if isinstance(build_info, dict):
        write_json(directory / "build_info.json", build_info)
    return directory


def copy_cache_to_run(key: str, run_id: str) -> None:
    src = cache_dir_for_key(key)
    if not src.exists():
        return
    from extraction.config import package_extraction_dir

    dest = package_extraction_dir(run_id)
    dest.mkdir(parents=True, exist_ok=True)
    for name in (
        "package_extraction.raw.json",
        "package_extraction.validated.json",
        "validation_findings.json",
        "diagnostics.json",
        "build_info.json",
    ):
        source_file = src / name
        if source_file.exists():
            shutil.copy2(source_file, dest / name)
    resolved_src = src / "resolved_signals.json"
    if resolved_src.exists():
        shutil.copy2(resolved_src, resolved_signals_json_path(run_id))


def write_run_artifacts(
    run_id: str,
    *,
    raw: dict[str, Any],
    validated: dict[str, Any],
    validation_findings: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> None:
    from extraction.config import package_extraction_dir

    dest = package_extraction_dir(run_id)
    dest.mkdir(parents=True, exist_ok=True)
    write_json(package_extraction_raw_path(run_id), raw)
    write_json(package_extraction_validated_path(run_id), validated)
    write_json(validation_findings_path(run_id), validation_findings)
    write_json(dest / "diagnostics.json", diagnostics)
    build_info = diagnostics.get("buildInfo")
    if isinstance(build_info, dict):
        write_json(dest / "build_info.json", build_info)
