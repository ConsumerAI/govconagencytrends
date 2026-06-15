from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extraction.config import get_project_root, load_project_env
from extraction.package_llm.build_info import cache_versions_match, collect_build_info
from extraction.package_llm.cache import cache_key
from extraction.package_llm.fast_schema import MARKET_SCOPE_FAST_SIGNAL_IDS
from extraction.package_llm.merge import merge_resolved_signals
from extraction.package_llm.versions import PROFILE_VERSIONS
from extraction.types import SourceRecord


def _source(sha: str, name: str) -> SourceRecord:
    return SourceRecord(
        key=f"runs/test/sources/{sha}.pdf",
        ext="pdf",
        bytes=10,
        sha256=sha,
        original_filename=name,
        abs_path=str(Path(name)),
    )


class BuildInfoTests(unittest.TestCase):
    def test_collect_build_info_includes_module_paths(self) -> None:
        info = collect_build_info(profile="market_scope_fast")
        self.assertEqual(info["extractionProfile"], "market_scope_fast")
        self.assertTrue(info["modulePaths"]["extraction"])
        self.assertTrue(info["modulePaths"]["validate"])
        self.assertTrue(info["modulePaths"]["canonicalize"])
        self.assertTrue(info["modulePaths"]["corpus"])

    def test_subprocess_imports_local_repository(self) -> None:
        root = get_project_root()
        script = (
            "import json; "
            "from extraction.package_llm.build_info import collect_build_info; "
            "print(json.dumps(collect_build_info()))"
        )
        env = os.environ.copy()
        env["PYTHONPATH"] = str(root) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            cwd=str(root),
            env=env,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        payload = __import__("json").loads(completed.stdout)
        self.assertIn(str(root.resolve()), payload["projectRoot"])


class CacheVersionTests(unittest.TestCase):
    def test_cache_key_changes_on_component_version(self) -> None:
        sources = [_source("abc", "Solicitation.pdf")]
        with patch.dict(os.environ, {"GOVCON_EXTRACTION_PROFILE": "market_scope_fast"}, clear=False):
            key_a = cache_key(sources)
        original = PROFILE_VERSIONS["market_scope_fast"]["validatorVersion"]
        PROFILE_VERSIONS["market_scope_fast"]["validatorVersion"] = "evidence_match.test_bump"
        try:
            with patch.dict(os.environ, {"GOVCON_EXTRACTION_PROFILE": "market_scope_fast"}, clear=False):
                key_b = cache_key(sources)
        finally:
            PROFILE_VERSIONS["market_scope_fast"]["validatorVersion"] = original
        self.assertNotEqual(key_a, key_b)

    def test_cache_versions_match_rejects_stale_build(self) -> None:
        expected = collect_build_info(profile="market_scope_fast")
        stale = dict(expected)
        stale["validatorVersion"] = "evidence_match.v0"
        self.assertFalse(cache_versions_match(stale, profile="market_scope_fast"))
        self.assertTrue(cache_versions_match(expected, profile="market_scope_fast"))


class FastSchemaTests(unittest.TestCase):
    def test_fast_schema_has_required_signals_once(self) -> None:
        self.assertEqual(len(MARKET_SCOPE_FAST_SIGNAL_IDS), 11)
        self.assertEqual(len(MARKET_SCOPE_FAST_SIGNAL_IDS), len(set(MARKET_SCOPE_FAST_SIGNAL_IDS)))
        self.assertIn("rfp_primary_naics_v1", MARKET_SCOPE_FAST_SIGNAL_IDS)
        self.assertIn("rfp_solicitation_id_v1", MARKET_SCOPE_FAST_SIGNAL_IDS)


class MergeTests(unittest.TestCase):
    def test_merge_does_not_overwrite_validated_fast_scope(self) -> None:
        base = {
            "runId": "run_fast",
            "signals": [
                {
                    "id": "rfp_primary_naics_v1",
                    "canonical_value": "722310",
                    "canonical_confidence": "high",
                    "resolution_status": "validated_model_extraction",
                },
                {
                    "id": "rfp_description_v1",
                    "canonical_value": None,
                    "resolution_status": "not_found",
                },
            ],
            "summary": {"extractionProfile": "market_scope_fast"},
        }
        incoming = {
            "runId": "run_full",
            "signals": [
                {
                    "id": "rfp_primary_naics_v1",
                    "canonical_value": "562910",
                    "canonical_confidence": "medium",
                    "resolution_status": "validated_model_extraction",
                },
                {
                    "id": "rfp_description_v1",
                    "canonical_value": "Catering services",
                    "resolution_status": "validated_model_extraction",
                },
            ],
            "summary": {"extractionProfile": "package_llm_full"},
        }
        merged = merge_resolved_signals(base, incoming)
        by_id = {item["id"]: item for item in merged["signals"]}
        self.assertEqual(by_id["rfp_primary_naics_v1"]["canonical_value"], "722310")
        self.assertEqual(by_id["rfp_description_v1"]["canonical_value"], "Catering services")


class DisplayScalarTests(unittest.TestCase):
    def test_canonical_scalar_for_office_dict(self) -> None:
        from extraction.package_llm.canonicalize import canonical_scalar_for_signal

        scalar, _ = canonical_scalar_for_signal("rfp_issuing_office_v1", {"office": "325 CONS PKP"})
        self.assertEqual(scalar, "325 CONS PKP")


if __name__ == "__main__":
    unittest.main()
