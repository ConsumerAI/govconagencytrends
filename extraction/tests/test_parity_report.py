from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from extraction.parity.compare import classify_comparison, compare_resolved_signals, ui_disposition
from extraction.parity.report import run_parity_comparison


def _record(
    signal_id: str,
    value: object,
    *,
    confidence: str = "high",
    status: str = "passthrough",
    excerpt: str = "evidence text",
    amendment: str | None = None,
    withheld: bool = False,
) -> dict:
    ev1 = {"excerpt": excerpt, "source": "solicitation.pdf"}
    if amendment:
        ev1["amendmentNumber"] = amendment
    item = {
        "id": signal_id,
        "canonical_value": value,
        "canonical_confidence": confidence,
        "resolution_status": status,
        "source_summary": {"authority_tiers": [2], "evidence_v1_sources": ["solicitation.pdf"]},
        "evidence": {"legacy": [{"sourceId": "solicitation.pdf", "artifact": "text", "locator": "p:0", "snippet": excerpt}], "evidence_v1": ev1},
        "alternates": [],
    }
    if withheld:
        item["_withheld"] = True
    return item


def _payload(*signals: dict) -> dict:
    return {"version": "resolved_signals.v1", "runId": "run_test", "signals": list(signals)}


class ParityCompareTests(unittest.TestCase):
    def test_exact_match(self) -> None:
        golden = _payload(_record("rfp_primary_naics_v1", "541511"))
        local = _payload(_record("rfp_primary_naics_v1", "541511"))
        row = compare_resolved_signals(golden_payload=golden, local_payload=local, signal_ids=("rfp_primary_naics_v1",))
        self.assertEqual(row["signals"][0]["comparison"], "exact match")
        self.assertTrue(row["passed"])

    def test_normalized_equivalent(self) -> None:
        golden = _payload(_record("rfp_set_aside_v1", "Total Small Business Set-Aside"))
        local = _payload(_record("rfp_set_aside_v1", "total small business set-aside"))
        row = compare_resolved_signals(golden_payload=golden, local_payload=local, signal_ids=("rfp_set_aside_v1",))
        self.assertEqual(row["signals"][0]["comparison"], "normalized-equivalent match")

    def test_material_mismatch(self) -> None:
        golden = _payload(_record("rfp_primary_naics_v1", "541511"))
        local = _payload(_record("rfp_primary_naics_v1", "722310"))
        row = compare_resolved_signals(golden_payload=golden, local_payload=local, signal_ids=("rfp_primary_naics_v1",))
        self.assertEqual(row["signals"][0]["comparison"], "material mismatch")
        self.assertFalse(row["passed"])

    def test_missing_signal(self) -> None:
        golden = _payload(_record("rfp_primary_naics_v1", "541511"))
        local = _payload()
        row = compare_resolved_signals(golden_payload=golden, local_payload=local, signal_ids=("rfp_primary_naics_v1",))
        self.assertEqual(row["signals"][0]["comparison"], "missing locally")

    def test_withheld_conflict(self) -> None:
        local = _record("rfp_primary_naics_v1", "541511", status="unresolved_conflict", withheld=True)
        comparison = classify_comparison(golden=_record("rfp_primary_naics_v1", "541511"), local=local)
        self.assertEqual(comparison, "withheld due to conflict")
        self.assertEqual(ui_disposition(local), "withheld")

    def test_ui_disposition_confirmed_vs_unmapped(self) -> None:
        confirmed = _record("rfp_primary_naics_v1", "541511", confidence="high")
        self.assertEqual(ui_disposition(confirmed), "confirmed")
        self.assertEqual(ui_disposition(None), "unmapped")

    def test_report_written(self) -> None:
        golden = _payload(_record("rfp_primary_naics_v1", "541511"))
        local = _payload(_record("rfp_primary_naics_v1", "541511"))
        with tempfile.TemporaryDirectory() as tmp:
            golden_path = Path(tmp) / "golden.json"
            local_path = Path(tmp) / "local.json"
            golden_path.write_text(json.dumps(golden), encoding="utf-8")
            local_path.write_text(json.dumps(local), encoding="utf-8")
            report = run_parity_comparison(golden_path=golden_path, local_path=local_path, run_id="run_parity_test")
            self.assertTrue(Path(report["parityReportJson"]).is_file())
            self.assertTrue(Path(report["parityReportMd"]).is_file())
            self.assertIn("scopeReview", report)
            self.assertTrue(report["scopeReview"]["loaderAccepted"])
            self.assertTrue(report["scopeReview"]["previewConstructed"])


if __name__ == "__main__":
    unittest.main()
