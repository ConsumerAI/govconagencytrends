from __future__ import annotations

from typing import Any

from extraction.config import full_corpus_text_path, resolved_signals_json_path, signals_json_path
from extraction.persist import read_json, write_json
from extraction.postprocess.run import run_postprocess
from extraction.resolve.resolve_signals import resolve_signals_v1
from extraction.types import Finding


def resolve_run_signals(
    run_id: str,
    *,
    signals: list[dict[str, Any]] | None = None,
    corpus_text: str | None = None,
) -> tuple[dict[str, Any], list[Finding], str | None]:
    findings: list[Finding] = []
    raw_signals = signals
    if raw_signals is None:
        path = signals_json_path(run_id)
        if not path.exists():
            findings.append(Finding("error", "RESOLVE_NO_SIGNALS", f"Missing signals.json: {path}"))
            return {}, findings, None
        raw_signals = read_json(path)
    if not isinstance(raw_signals, list):
        findings.append(Finding("error", "RESOLVE_INVALID_SIGNALS", "signals.json must be an array"))
        return {}, findings, None

    if corpus_text is None:
        corpus_file = full_corpus_text_path(run_id)
        corpus_text = corpus_file.read_text(encoding="utf-8") if corpus_file.exists() else ""

    processed, post_findings = run_postprocess(raw_signals, corpus_text=corpus_text)
    findings.extend(post_findings)

    artifact = resolve_signals_v1(run_id, processed)
    resolved_path = resolved_signals_json_path(run_id)
    write_json(resolved_path, artifact)
    return artifact, findings, str(resolved_path)
