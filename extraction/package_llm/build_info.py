from __future__ import annotations

import os
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from typing import Any

from extraction.config import get_extraction_mode, get_extraction_model, get_extraction_profile, get_extraction_reasoning_effort, get_project_root
from extraction.package_llm.versions import PROFILE_VERSIONS


def _git_commit(root: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
        if completed.returncode == 0:
            return completed.stdout.strip() or None
    except Exception:
        return None
    return None


def _module_path(module_name: str) -> str | None:
    try:
        module = import_module(module_name)
        file_path = getattr(module, "__file__", None)
        return str(Path(file_path).resolve()) if file_path else None
    except Exception:
        return None


def collect_build_info(*, profile: str | None = None) -> dict[str, Any]:
    profile = profile or get_extraction_profile()
    root = get_project_root()
    profile_versions = dict(PROFILE_VERSIONS.get(profile, PROFILE_VERSIONS["package_llm_full"]))
    profile_versions["extractionProfile"] = profile_versions.pop("profile", profile)
    return {
        "pythonExecutable": sys.executable,
        "workingDirectory": str(Path.cwd().resolve()),
        "projectRoot": str(root.resolve()),
        "extractionMode": get_extraction_mode(),
        "extractionProfile": profile,
        "model": get_extraction_model(),
        "reasoningEffort": get_extraction_reasoning_effort(),
        "gitCommit": _git_commit(root),
        "modulePaths": {
            "extraction": _module_path("extraction"),
            "validate": _module_path("extraction.package_llm.validate"),
            "canonicalize": _module_path("extraction.package_llm.canonicalize"),
            "corpus": _module_path("extraction.package_llm.corpus"),
            "evidence_match": _module_path("extraction.package_llm.validators.evidence_match"),
            "forms": _module_path("extraction.package_llm.forms.pdf_forms"),
        },
        **profile_versions,
    }


def build_info_finding(profile: str | None = None):
    from extraction.types import Finding

    info = collect_build_info(profile=profile)
    return Finding("info", "EXTRACTION_BUILD_INFO", "Extraction build metadata", info)


def cache_versions_match(cached_build: dict[str, Any] | None, *, profile: str) -> bool:
    if not isinstance(cached_build, dict):
        return False
    expected = collect_build_info(profile=profile)
    keys = (
        "extractionProfile",
        "schemaVersion",
        "promptVersion",
        "corpusBuilderVersion",
        "formExtractorVersion",
        "validatorVersion",
        "canonicalizerVersion",
        "resolverVersion",
    )
    return all(str(cached_build.get(key) or "") == str(expected.get(key) or "") for key in keys)
