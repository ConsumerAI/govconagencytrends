from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from extraction.types import Finding


def pipeline_branch_entered(mode: str) -> Finding:
    code = "PACKAGE_LLM_ENTERED" if mode == "package_llm" else "LEGACY_PIPELINE_ENTERED"
    return Finding("info", code, f"Extraction branch entered: {mode}", {"mode": mode})


def pipeline_branch_completed(mode: str, *, elapsed_sec: float) -> Finding:
    code = "PACKAGE_LLM_COMPLETED" if mode == "package_llm" else "LEGACY_PIPELINE_COMPLETED"
    return Finding("info", code, f"Extraction branch completed: {mode}", {"mode": mode, "elapsedSec": elapsed_sec})


def artifact_written(path: str, *, producer: str, mode: str) -> Finding:
    return Finding(
        "info",
        "ARTIFACT_WRITTEN",
        f"Wrote {path}",
        {
            "path": path,
            "producer": producer,
            "mode": mode,
            "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        },
    )
