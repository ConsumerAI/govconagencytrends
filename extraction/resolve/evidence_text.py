from __future__ import annotations

import re
from typing import Any

# Federal award / contract id (DoD concatenated PIID + dashed PIIN).
FEDERAL_PIID_EXTRACT_RE = re.compile(
    r"\b(?:[A-Z]{2}\d{4}[A-Z0-9]{4,8}|[A-Z0-9]{4,6}-\d{2,3}-[A-Z]-\d{4})\b",
    re.IGNORECASE,
)


def _normalize_seed(value: str) -> str | None:
    token = value.strip().upper()
    return token or None


def _is_federal_piid_token(token: str) -> bool:
    return bool(
        re.fullmatch(r"[A-Z]{2}\d{4}[A-Z0-9]{4,8}", token)
        or re.fullmatch(r"[A-Z0-9]{4,6}-\d{2,3}-[A-Z]-\d{4}", token)
    )


def _extract_federal_piid_candidates(text: str) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for match in FEDERAL_PIID_EXTRACT_RE.finditer(text or ""):
        token = _normalize_seed(match.group(0))
        if not token or not _is_federal_piid_token(token) or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


def _prefer_incumbent_contract_piid(candidates: list[str]) -> str | None:
    if not candidates:
        return None
    for candidate in candidates:
        if re.search(r"^[A-Z]{2}\d{4}[CD][A-Z0-9]*$", candidate, flags=re.IGNORECASE) or re.search(
            r"-\d{2,3}-[CD]-", candidate, flags=re.IGNORECASE
        ):
            return candidate
    return candidates[0]


def extract_piid_from_incumbent_blob(text: str) -> str | None:
    normalized = (text or "").strip()
    if not normalized:
        return None

    whole = _extract_federal_piid_candidates(normalized)
    if whole:
        return _prefer_incumbent_contract_piid(whole)

    segments = [
        segment.strip()
        for segment in re.split(r"\s*[;,]\s*|\s*[—–]\s*|\s+-\s+", normalized)
        if segment.strip()
    ]
    for segment in segments:
        from_segment = _extract_federal_piid_candidates(segment)
        if from_segment:
            return _prefer_incumbent_contract_piid(from_segment)
        bare = _normalize_seed(segment)
        if bare and _is_federal_piid_token(bare):
            return bare
    return None


def collect_record_evidence_text_hits(record: dict[str, Any]) -> list[dict[str, str]]:
    hits: list[dict[str, str]] = []
    evidence = record.get("evidence") or {}
    legacy = evidence.get("legacy") if isinstance(evidence, dict) else record.get("evidence")
    if isinstance(legacy, list):
        for item in legacy:
            if isinstance(item, dict) and item.get("snippet"):
                hits.append(
                    {
                        "text": str(item["snippet"]),
                        "source_artifact": str(item.get("sourceId") or record.get("id") or "unknown"),
                        "locator": str(item.get("locator") or "legacy"),
                    }
                )
    ev1 = evidence.get("evidence_v1") if isinstance(evidence, dict) else record.get("evidence_v1")
    if isinstance(ev1, dict) and ev1.get("excerpt"):
        hits.append(
            {
                "text": str(ev1["excerpt"]),
                "source_artifact": str(ev1.get("source") or record.get("id") or "unknown"),
                "locator": "evidence_v1",
            }
        )
    canonical = record.get("canonical_value")
    if isinstance(canonical, str) and canonical.strip():
        hits.append(
            {
                "text": canonical.strip(),
                "source_artifact": str(record.get("id") or "canonical"),
                "locator": "canonical_value",
            }
        )
    return hits
