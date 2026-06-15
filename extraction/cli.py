from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import extraction.config  # noqa: F401  — load project .env before pipeline imports

from extraction.config import diagnostics_path, has_openai_api_key
from extraction.persist import read_json


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="govcon-extract",
        description="GovCon Agency Trends local solicitation extraction (Milestones 1–5).",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Solicitation package files (PDF, XLSX, DOCX, CSV, TXT, ...).",
    )
    parser.add_argument(
        "--run-id",
        dest="run_id",
        default=None,
        help="Optional run id (default: run_<uuid>).",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Milestone 1 only: store sources and manifest without text extraction.",
    )
    parser.add_argument(
        "--extract-signals",
        action="store_true",
        help="Milestone 3: run OpenAI signal extraction after corpus build (requires OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--with-llm",
        action="store_true",
        help="Alias for --extract-signals.",
    )
    parser.add_argument(
        "--resolve-signals",
        action="store_true",
        help="Milestone 4: resolve signals.json into resolved_signals.json.",
    )
    parser.add_argument(
        "--full-extraction",
        action="store_true",
        help="Run corpus build, LLM extraction, and resolver in one command.",
    )
    parser.add_argument(
        "--extract-signals-only",
        action="store_true",
        help="Run Milestone 3 against an existing run (requires --run-id, no new files).",
    )
    parser.add_argument(
        "--compare-golden",
        dest="compare_golden",
        default=None,
        help="After resolve, compare local resolved_signals.json against a golden artifact and write parity_report.json/md.",
    )
    parser.add_argument(
        "--resolve-signals-only",
        action="store_true",
        help="Run Milestone 4 against an existing run (requires --run-id, no new files).",
    )
    parser.add_argument(
        "--profile",
        dest="profile",
        default=None,
        choices=["market_scope_fast", "package_llm_full"],
        help="Package extraction profile (default: GOVCON_EXTRACTION_PROFILE or market_scope_fast).",
    )
    parser.add_argument(
        "--merge-run-id",
        dest="merge_run_id",
        default=None,
        help="When set, merge full-detail extraction into an existing fast-scope run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.profile:
        os.environ["GOVCON_EXTRACTION_PROFILE"] = args.profile
    if args.merge_run_id:
        os.environ["GOVCON_MERGE_RUN_ID"] = args.merge_run_id

    full_extraction = args.full_extraction
    extract_signals = args.extract_signals or args.with_llm or full_extraction
    resolve_signals = args.resolve_signals or full_extraction

    if args.resolve_signals_only:
        if not args.run_id:
            parser.error("--resolve-signals-only requires --run-id")
        from extraction.pipeline import run_resolve_for_existing_run

        result = run_resolve_for_existing_run(args.run_id)
    elif args.extract_signals_only:
        if not args.run_id:
            parser.error("--extract-signals-only requires --run-id")
        from extraction.pipeline import run_signal_extraction_for_existing_run

        result = run_signal_extraction_for_existing_run(args.run_id)
    else:
        if not args.files:
            parser.error("At least one input file is required unless using a *-only mode")
        from extraction.pipeline import run_ingest_pipeline

        file_paths = [Path(item) for item in args.files]
        result = run_ingest_pipeline(
            file_paths,
            run_id=args.run_id,
            extract_text=not args.ingest_only,
            extract_signals=extract_signals and not args.ingest_only,
            resolve_signals=resolve_signals and not args.ingest_only,
        )

    summary = {
        "runId": result.run_id,
        "manifest": result.manifest_path,
        "diagnostics": result.diagnostics_path,
        "processSourcesSignal": result.process_sources_path,
        "corpusV1": result.corpus_path,
        "fullCorpusText": result.full_text_path,
        "signalsJson": result.signals_path,
        "resolvedSignalsJson": result.resolved_signals_path,
        "rawSignalCount": result.signal_count,
        "resolvedSignalCount": result.resolved_signal_count,
        "alternatesCount": result.alternates_count,
        "derivedSignalCount": result.derived_signal_count,
        "modelUsed": result.model_used,
        "corpusCharCount": result.corpus_char_count,
        "elapsedSec": result.elapsed_sec,
        "openaiApiKeyPresent": has_openai_api_key(),
        "findings": [finding.to_dict() for finding in result.findings],
        "solicitationSetDetected": result.solicitation_set_detected,
        "baseSolicitationFilename": result.base_solicitation_filename,
        "amendmentsDetectedInOrder": result.amendments_detected,
        "documentCount": result.document_count,
        "supersededCandidateCount": result.superseded_candidate_count,
        "unresolvedAmendmentConflicts": result.unresolved_amendment_conflicts,
        "docsetManifest": result.docset_manifest_path,
        "buildInfo": (read_json(diagnostics_path(result.run_id)).get("packageDiagnostics") or {}).get("buildInfo")
        if result.run_id
        else None,
    }
    result_path = os.getenv("GOVCON_EXTRACTION_RESULT_FILE", "").strip()
    summary_text = json.dumps(summary, indent=2)
    if result_path:
        target = Path(result_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(summary_text + "\n", encoding="utf-8")
    else:
        print(summary_text)

    if args.compare_golden:
        from extraction.parity.report import run_parity_for_run

        if not result.resolved_signals_path:
            print(json.dumps({"parityError": "No resolved_signals.json produced; run with --full-extraction or --resolve-signals"}, indent=2))
            return 1
        parity = run_parity_for_run(
            golden_path=Path(args.compare_golden),
            run_id=result.run_id,
            local_path=Path(result.resolved_signals_path),
        )
        print(json.dumps({"parityPassed": parity.get("passed"), "parityReportJson": parity.get("parityReportJson"), "parityReportMd": parity.get("parityReportMd")}, indent=2))
        if not parity.get("passed"):
            return 1

    has_error = any(finding.level == "error" for finding in result.findings)
    return 1 if has_error else 0


if __name__ == "__main__":
    sys.exit(main())
