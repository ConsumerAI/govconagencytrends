from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from extraction.progress import (
    PROGRESS_ENV,
    emit_progress,
    read_extraction_progress,
    write_extraction_progress,
)


class ExtractionProgressTests(unittest.TestCase):
    def test_write_and_read_progress_payload(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.json"
            write_extraction_progress(
                "Analyzing document 2 of 8 — sample.pdf",
                0.42,
                doc_index=2,
                doc_total=8,
                filename="sample.pdf",
                phase="reading",
                path=path,
            )
            payload = read_extraction_progress(path)
            assert payload is not None
            self.assertEqual(payload["label"], "Analyzing document 2 of 8 — sample.pdf")
            self.assertEqual(payload["docIndex"], 2)
            self.assertEqual(payload["docTotal"], 8)
            self.assertEqual(payload["filename"], "sample.pdf")
            self.assertEqual(payload["phase"], "reading")

    def test_emit_progress_uses_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.json"
            os.environ[PROGRESS_ENV] = str(path)
            try:
                emit_progress(None, "Extracting market scope", 0.5, phase="gpt")
            finally:
                os.environ.pop(PROGRESS_ENV, None)
            payload = read_extraction_progress(path)
            assert payload is not None
            self.assertEqual(payload["label"], "Extracting market scope")
            self.assertEqual(payload["phase"], "gpt")

    def test_write_progress_retries_transient_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.json"
            original_replace = Path.replace
            attempts = {"count": 0}

            def flaky_replace(self, target):
                attempts["count"] += 1
                if attempts["count"] == 1:
                    raise PermissionError("locked")
                return original_replace(self, target)

            with patch.object(Path, "replace", flaky_replace):
                write_extraction_progress("Reading documents", 0.1, path=path)

            payload = read_extraction_progress(path)
            assert payload is not None
            self.assertEqual(payload["label"], "Reading documents")
            self.assertEqual(attempts["count"], 2)

    def test_write_progress_ignores_persistent_permission_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "progress.json"
            with patch.object(Path, "replace", side_effect=PermissionError("locked")):
                write_extraction_progress("Reading documents", 0.1, path=path)
            self.assertFalse(path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
