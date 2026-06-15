from __future__ import annotations

import re
from typing import Any

from extraction.resolve.evidence_text import FEDERAL_PIID_EXTRACT_RE, extract_piid_from_incumbent_blob
from extraction.types import Finding

FEDERAL_PIID_BACKSTOP_RE = FEDERAL_PIID_EXTRACT_RE


def _signal_value(signals: list[dict[str, Any]], signal_id: str) -> str | None:
    for signal in signals:
        if str(signal.get("id") or "") == signal_id:
            value = signal.get("value")
            if value is None:
                return None
            text = str(value).strip()
            return text or None
    return None


def read_prior_contract_seed_piid(signals: list[dict[str, Any]]) -> str | None:
    incumbent = _signal_value(signals, "rfp_incumbent_data_v1")
    if not incumbent:
        return None
    return extract_piid_from_incumbent_blob(incumbent)


def inject_prior_contract_piid_backstop(
    signals: list[dict[str, Any]],
    corpus_text: str = "",
) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    if _signal_value(signals, "rfp_prior_contract_piid_v1"):
        return signals, findings

    seed = read_prior_contract_seed_piid(signals)
    if seed:
        injected = {
            "id": "rfp_prior_contract_piid_v1",
            "value": seed,
            "confidence": "high",
            "evidence": [],
            "findings": [
                Finding(
                    "info",
                    "PRIOR_PIID_FROM_INCUMBENT_DATA",
                    "rfp_prior_contract_piid_v1 derived from rfp_incumbent_data_v1",
                ).to_dict()
            ],
        }
        donor = next((signal for signal in signals if signal.get("id") == "rfp_incumbent_data_v1"), None)
        if donor:
            injected["evidence"] = donor.get("evidence") or []
            injected["evidence_v1"] = donor.get("evidence_v1")
            donor_conf = str(donor.get("confidence") or "medium")
            if donor_conf in {"high", "medium", "low"}:
                injected["confidence"] = donor_conf
        merged = [signal for signal in signals if signal.get("id") != "rfp_prior_contract_piid_v1"]
        merged.append(injected)
        merged.sort(key=lambda item: str(item.get("id") or ""))
        findings.append(
            Finding("info", "PRIOR_PIID_FROM_INCUMBENT_DATA", f"rfp_prior_contract_piid_v1 set to {seed}")
        )
        return merged, findings

    solicitation_id = (_signal_value(signals, "rfp_solicitation_id_v1") or "").upper()
    text = corpus_text or ""
    if not text.strip():
        findings.append(Finding("info", "PIID_BACKSTOP_SKIPPED_EMPTY_CORPUS", "PIID backstop skipped: empty corpus text"))
        return signals, findings

    matches: list[tuple[str, int]] = []
    for match in FEDERAL_PIID_BACKSTOP_RE.finditer(text):
        piid = match.group(0).upper()
        if solicitation_id and piid == solicitation_id:
            continue
        matches.append((piid, match.start()))

    if not matches:
        findings.append(Finding("info", "PIID_BACKSTOP_NO_CANDIDATE", "PIID backstop found no predecessor-style identifiers"))
        return signals, findings

    preferred = next((item for item in matches if re.search(r"-\d{2,3}-R-", item[0], flags=re.IGNORECASE)), matches[0])
    piid, index = preferred
    excerpt = text[max(0, index - 120) : min(len(text), index + 120)].replace("\n", " ").strip()
    injected = {
        "id": "rfp_prior_contract_piid_v1",
        "value": piid,
        "confidence": "high",
        "evidence": [
            {
                "sourceId": "corpus",
                "artifact": "text",
                "locator": "piid_backstop",
                "snippet": excerpt[:200],
            }
        ],
        "evidence_v1": {"spanHashes": [], "excerpt": excerpt[:280], "source": "corpus"},
        "findings": [
            Finding("info", "PIID_BACKSTOP_INJECTED", f"Injected prior contract PIID from corpus regex: {piid}").to_dict()
        ],
    }
    merged = [signal for signal in signals if signal.get("id") != "rfp_prior_contract_piid_v1"]
    merged.append(injected)
    merged.sort(key=lambda item: str(item.get("id") or ""))
    findings.append(Finding("info", "PIID_BACKSTOP_APPLIED", f"rfp_prior_contract_piid_v1 set to {piid}"))
    return merged, findings
