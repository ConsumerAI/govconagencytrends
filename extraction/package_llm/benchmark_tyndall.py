from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _run(label: str, reasoning: str, sources: list[Path], *, force: bool) -> dict:
    env = os.environ.copy()
    root = Path(__file__).resolve().parents[2]
    env["PYTHONPATH"] = str(root) + (os.pathsep + env.get("PYTHONPATH", ""))
    env["GOVCON_EXTRACTION_PROFILE"] = "market_scope_fast"
    env["GOVCON_FAST_REASONING_EFFORT"] = reasoning
    if force:
        env["GOVCON_FORCE_PACKAGE_REFRESH"] = "1"
    else:
        env.pop("GOVCON_FORCE_PACKAGE_REFRESH", None)
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, "-m", "extraction.cli", "--full-extraction", *[str(path) for path in sources]],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(root),
        check=False,
    )
    elapsed = round(time.perf_counter() - started, 2)
    if completed.returncode not in {0, 1} or not completed.stdout.strip():
        return {"label": label, "error": completed.stderr or completed.stdout, "elapsedSec": elapsed}
    summary = json.loads(completed.stdout)
    run_id = summary.get("runId")
    run_dir = root / "data" / "runs" / str(run_id)
    resolved = json.loads((run_dir / "signals" / "resolved_signals.json").read_text(encoding="utf-8"))
    diagnostics = json.loads((run_dir / "package_extraction" / "diagnostics.json").read_text(encoding="utf-8"))
    signals = {item.get("id"): item for item in resolved.get("signals") or [] if isinstance(item, dict)}
    checks = {
        "solId": (signals.get("rfp_solicitation_id_v1") or {}).get("canonical_value"),
        "naics": (signals.get("rfp_primary_naics_v1") or {}).get("canonical_value"),
        "office": (signals.get("rfp_issuing_office_v1") or {}).get("canonical_value"),
        "aac": (signals.get("rfp_office_aac_v1") or {}).get("canonical_value"),
        "psc": (signals.get("rfp_primary_psc_v1") or {}).get("canonical_value"),
        "setAside": (signals.get("rfp_set_aside_v1") or {}).get("canonical_value"),
        "statusAlert": (signals.get("solicitation_status_alert_v1") or {}).get("canonical_value"),
        "requestedCount": (resolved.get("summary") or {}).get("requestedSignalCount"),
        "signalCount": len([s for s in resolved.get("signals") or [] if s.get("id") != "process_sources_v1"]),
    }
    return {
        "label": label,
        "reasoningEffort": reasoning,
        "elapsedSec": elapsed,
        "runId": run_id,
        "cacheHit": diagnostics.get("cacheHit"),
        "inputTokens": diagnostics.get("inputTokens"),
        "outputTokens": diagnostics.get("outputTokens"),
        "reasoningTokens": diagnostics.get("reasoningTokens"),
        "gptRequestSec": (diagnostics.get("timings") or {}).get("gptRequestSec"),
        "validationSec": (diagnostics.get("timings") or {}).get("validationSec"),
        "fastCorpusChars": diagnostics.get("fastCorpusCharCount"),
        "openaiRequestCount": diagnostics.get("openaiRequestCount"),
        "checks": checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark Tyndall 11-signal market-scope extraction.")
    parser.add_argument("--sources-dir", default=None)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.sources_dir:
        sources = sorted(Path(args.sources_dir).glob("*"))
    else:
        runs = sorted((root / "data" / "runs").glob("run_*"), key=lambda p: p.stat().st_mtime, reverse=True)
        sources = []
        for run in runs:
            candidate = sorted((run / "sources").glob("*"))
            if len(candidate) >= 8:
                sources = candidate
                break
    sources = [path for path in sources if path.is_file()]
    if not sources:
        print(json.dumps({"error": "No Tyndall source files found."}, indent=2))
        return 1

    rows = [
        _run("11-signal low", "low", sources, force=True),
        _run("11-signal medium", "medium", sources, force=True),
        _run("11-signal low cache hit", "low", sources, force=False),
    ]
    print(json.dumps({"sources": [path.name for path in sources], "results": rows}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
