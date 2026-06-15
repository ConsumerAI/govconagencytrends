from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

from extraction.config import live_openai_tests_enabled
from extraction.llm.map_signals import map_extraction_to_signals

FIXTURES = Path(__file__).resolve().parent / "fixtures"
TYNDALL_FIXTURE = FIXTURES / "tyndall_sample.txt"
REPO_ROOT = Path(__file__).resolve().parents[2]


def _offline_subprocess_env() -> dict[str, str]:
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env["GOVCON_DISABLE_DOTENV"] = "1"
    env["GOVCON_EXTRACTION_MODE"] = "legacy"
    env.pop("GOVCON_LIVE_OPENAI_TESTS", None)
    return env


class MapSignalsTests(unittest.TestCase):
    def test_tyndall_fixture_mapping(self) -> None:
        extraction = {
            "solicitation_id": {
                "value": "FA481926R0001",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "FA481926R0001"}],
                "notes": "",
            },
            "issuing_agency": {
                "value": "Department of the Air Force",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "Department of the Air Force"}],
                "notes": "",
            },
            "issuing_office": {
                "value": "325 CONS PKP",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "325 CONS PKP"}],
                "notes": "",
            },
            "issuing_office_aac": {
                "value": "FA4819",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "FA4819"}],
                "notes": "",
            },
            "primary_naics_code": {
                "value": "722310",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "722310"}],
                "notes": "",
            },
            "psc_code": {
                "value": "S203",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "S203"}],
                "notes": "",
            },
            "contract_type": {
                "value": "Firm Fixed Price (FFP)",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "Firm Fixed Price (FFP)"}],
                "notes": "",
            },
            "set_aside_type": {
                "value": "8(a) Set-Aside",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "8(a) small business set-aside"}],
                "notes": "",
            },
            "places_of_performance": {
                "value": "Tyndall AFB, FL",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "Tyndall AFB, FL"}],
                "notes": "",
            },
            "period_of_performance_start": {
                "value": "01 Oct 2026",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "01 Oct 2026"}],
                "notes": "",
            },
            "period_of_performance_end": {
                "value": "30 Sep 2031",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "30 Sep 2031"}],
                "notes": "",
            },
            "incumbent_data": {
                "value": "Trogon Group Services, LLC; FA481924C0006",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "Trogon Group Services, LLC"}],
                "notes": "",
            },
            "prior_contract_piid": {
                "value": "FA481924C0006",
                "status": "explicit",
                "confidence": "high",
                "page": 1,
                "evidence": [{"source_file": "tyndall_sample.txt", "quote": "FA481924C0006"}],
                "notes": "",
            },
        }
        signals, _ = map_extraction_to_signals("run_test", extraction)
        by_id = {item["id"]: item for item in signals}
        self.assertEqual(by_id["rfp_solicitation_id_v1"]["value"], "FA481926R0001")
        self.assertEqual(by_id["rfp_issuing_agency_v1"]["value"], "Department of the Air Force")
        self.assertEqual(by_id["rfp_primary_naics_v1"]["value"], "722310")
        self.assertTrue(by_id["rfp_solicitation_id_v1"]["evidence"])


class PipelineTests(unittest.TestCase):
    def test_missing_openai_key_skips_llm_but_keeps_deterministic(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "extraction.cli",
                str(TYNDALL_FIXTURE),
                "--extract-signals",
            ],
            cwd=str(REPO_ROOT),
            env=_offline_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        payload = json.loads(completed.stdout)
        self.assertFalse(payload["openaiApiKeyPresent"])
        self.assertIsNone(payload["modelUsed"])
        codes = {item["code"] for item in payload["findings"]}
        self.assertIn("OPENAI_API_KEY_MISSING", codes)
        if payload["signalsJson"]:
            self.assertGreater(payload["rawSignalCount"], 0)

    def test_milestone2_without_llm(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "extraction.cli", str(TYNDALL_FIXTURE)],
            cwd=str(REPO_ROOT),
            env=_offline_subprocess_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)
        payload = json.loads(completed.stdout)
        self.assertIsNotNone(payload["fullCorpusText"])
        self.assertIsNone(payload["signalsJson"])


class DotenvTests(unittest.TestCase):
    def test_project_env_loads_openai_key_from_dotenv(self) -> None:
        env_path = REPO_ROOT / ".env"
        if not env_path.is_file():
            self.skipTest("Project .env not present")
        env = os.environ.copy()
        env.pop("OPENAI_API_KEY", None)
        env.pop("GOVCON_DISABLE_DOTENV", None)
        env["GOVCON_DOTENV_PATH"] = str(env_path)
        completed = subprocess.run(
            [sys.executable, "-c", "from extraction.config import has_openai_api_key; print(has_openai_api_key())"],
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "True")


@unittest.skipUnless(live_openai_tests_enabled(), "Set GOVCON_LIVE_OPENAI_TESTS=1 to run live OpenAI integration tests")
class LiveOpenAIIntegrationTests(unittest.TestCase):
    def test_live_openai_extraction_smoke(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "extraction.cli", str(TYNDALL_FIXTURE), "--extract-signals"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr or completed.stdout)


if __name__ == "__main__":
    unittest.main()
