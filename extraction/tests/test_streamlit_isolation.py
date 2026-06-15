from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from extraction.config import get_extraction_mode, load_project_env
from extraction.streamlit_upload import run_extraction_from_paths


class StreamlitIsolationTests(unittest.TestCase):
    def test_run_extraction_from_paths_uses_subprocess_for_package_mode(self) -> None:
        load_project_env()
        if get_extraction_mode() != "package_llm":
            self.skipTest("Requires default package_llm mode")

        fixture = Path(__file__).resolve().parent / "fixtures" / "tyndall_sample.txt"
        with tempfile.TemporaryDirectory() as temp_dir:
            staged = Path(temp_dir) / fixture.name
            staged.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
            with patch("extraction.streamlit_upload.run_extraction_subprocess") as mock_subprocess:
                from extraction.types import PipelineResult

                mock_subprocess.return_value = (
                    PipelineResult(
                        run_id="run_mock",
                        manifest_path="manifest.json",
                        diagnostics_path="diagnostics.json",
                        resolved_signals_path=str(Path(temp_dir) / "resolved_signals.json"),
                        resolved_signal_count=1,
                        document_count=1,
                    ),
                    {"runId": "run_mock"},
                )
                with patch("extraction.streamlit_upload.read_json", return_value={"signals": [], "summary": {"producer": "package_llm"}}):
                    result = run_extraction_from_paths([staged])
                mock_subprocess.assert_called_once()
                self.assertTrue(result.summary.get("isolatedSubprocess"))

    def test_subprocess_result_file_handoff_does_not_wait_for_child_exit(self) -> None:
        from extraction.persist import write_json
        from extraction.streamlit_isolated import run_extraction_subprocess

        class NeverExitsProcess:
            returncode = None
            pid = 999999

            def __init__(self, result_payload: dict):
                self.result_payload = result_payload
                self.wrote = False

            def poll(self):
                if not self.wrote:
                    result_path = Path(self.env["GOVCON_EXTRACTION_RESULT_FILE"])
                    result_path.write_text(__import__("json").dumps(self.result_payload), encoding="utf-8")
                    self.stdout.write("x" * 200000)
                    self.stderr.write("diagnostic\n" * 1000)
                    self.stdout.flush()
                    self.stderr.flush()
                    self.wrote = True
                return None

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            run_id = "run_handoff"
            write_json(data_dir / "runs" / run_id / "diagnostics.json", {"findings": [], "packageDiagnostics": {}})
            summary = {
                "runId": run_id,
                "manifest": "manifest.json",
                "diagnostics": str(data_dir / "runs" / run_id / "diagnostics.json"),
                "resolvedSignalsJson": str(data_dir / "runs" / run_id / "signals" / "resolved_signals.json"),
                "resolvedSignalCount": 1,
                "findings": [],
                "documentCount": 1,
            }
            proc = NeverExitsProcess(summary)

            def fake_popen(_cmd, stdout, stderr, text, env, cwd):
                proc.stdout = stdout
                proc.stderr = stderr
                proc.env = env
                return proc

            with patch.dict("os.environ", {"GOVCON_DATA_DIR": str(data_dir)}, clear=False):
                with patch("subprocess.Popen", side_effect=fake_popen):
                    with patch("extraction.streamlit_isolated._terminate_process_tree") as mock_kill:
                        pipeline, cli_summary = run_extraction_subprocess([Path("sample.pdf")])

            self.assertEqual(cli_summary["runId"], run_id)
            self.assertEqual(pipeline.run_id, run_id)
            mock_kill.assert_called_once()

    def test_resolved_artifact_handoff_does_not_wait_for_cli_result_file(self) -> None:
        from extraction.persist import write_json
        from extraction.streamlit_isolated import run_extraction_subprocess

        class ResolvedArtifactProcess:
            returncode = None
            pid = 999998

            def __init__(self, data_dir: Path, run_id: str):
                self.data_dir = data_dir
                self.run_id = run_id
                self.wrote = False

            def poll(self):
                if not self.wrote:
                    write_json(
                        self.data_dir / "runs" / self.run_id / "signals" / "resolved_signals.json",
                        {
                            "version": "resolved_signals.v1",
                            "runId": self.run_id,
                            "signals": [{"id": "rfp_primary_naics_v1", "canonical_value": "722310"}],
                            "summary": {"requestedSignalCount": 1, "resolved_signal_count": 1},
                        },
                    )
                    self.stdout.write("artifact written\n")
                    self.stdout.flush()
                    self.wrote = True
                return None

        progress_events = []
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            run_id = "run_artifact_boundary"
            proc = ResolvedArtifactProcess(data_dir, run_id)

            def fake_popen(_cmd, stdout, stderr, text, env, cwd):
                proc.stdout = stdout
                proc.stderr = stderr
                proc.env = env
                return proc

            with patch.dict("os.environ", {"GOVCON_DATA_DIR": str(data_dir)}, clear=False):
                with patch("subprocess.Popen", side_effect=fake_popen):
                    with patch("extraction.streamlit_isolated._terminate_process_tree") as mock_kill:
                        pipeline, cli_summary = run_extraction_subprocess(
                            [Path("sample.pdf")],
                            progress=lambda label, pct: progress_events.append((label, pct)),
                        )

            self.assertEqual(pipeline.run_id, run_id)
            self.assertTrue(cli_summary["artifactBoundaryAccepted"])
            self.assertIn(("Market scope extracted successfully", 1.0), progress_events)
            mock_kill.assert_called_once()

    def test_timeout_after_resolved_artifact_loads_saved_scope(self) -> None:
        from extraction.persist import write_json

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            run_id = "run_recovered"
            resolved_path = data_dir / "runs" / run_id / "signals" / "resolved_signals.json"
            write_json(
                resolved_path,
                {
                    "version": "resolved_signals.v1",
                    "runId": run_id,
                    "signals": [{"id": "rfp_primary_naics_v1", "canonical_value": "722310"}],
                    "summary": {"extractionProfile": "market_scope_fast"},
                },
            )
            now = time.time()
            resolved_path.touch()
            with patch.dict("os.environ", {"GOVCON_DATA_DIR": str(data_dir)}, clear=False):
                with patch("extraction.streamlit_upload.run_extraction_subprocess", side_effect=TimeoutError("stuck")):
                    with patch("extraction.streamlit_upload.time.time", return_value=now):
                        staged = Path(temp_dir) / "sample.txt"
                        staged.write_text("sample", encoding="utf-8")
                        result = run_extraction_from_paths([staged])

            self.assertTrue(result.success)
            self.assertEqual(result.resolved_signals["runId"], run_id)
            self.assertTrue(result.summary["recoveredFromPostExtractionFailure"])
            self.assertEqual(result.findings[0].code, "POST_EXTRACTION_HANDOFF_RECOVERED")


if __name__ == "__main__":
    unittest.main()
