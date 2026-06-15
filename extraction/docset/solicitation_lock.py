from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any


SOLICITATION_ID_RE = re.compile(
    r"\b(?:solicitation\s*(?:no\.?|number)?|rf[qp]\s*(?:no\.?|number)?)\s*[:\-]?\s*([A-Z0-9][A-Z0-9\-./]{4,})\b",
    re.I,
)
PIID_RE = re.compile(r"\b(?:prior\s+contract\s+piid|contract\s+number|piid)\s*[:\-]?\s*([A-Z0-9-]{8,20})\b", re.I)


@dataclass
class PackageSolicitationLock:
    locked_solicitation_id: str | None
    confidence: str
    locking_evidence: list[dict[str, Any]] = field(default_factory=list)
    associated_documents: list[str] = field(default_factory=list)
    excluded_documents: list[dict[str, Any]] = field(default_factory=list)
    conflicts: list[dict[str, Any]] = field(default_factory=list)


def _normalize_id(raw: str) -> str:
    return re.sub(r"\s+", "", raw).upper()


def _authority_weight(doc: dict[str, Any]) -> int:
    doc_type = str(doc.get("documentType") or "")
    if doc_type == "base_solicitation":
        return 100
    if doc.get("isAmendment") and str(doc.get("amendmentOrderSource") or "") == "sf30":
        return 90
    if doc.get("isAmendment"):
        return 70
    if doc_type in {"attachment_exhibit", "qa"}:
        return 10
    return 40


def build_package_solicitation_lock(documents: list[dict[str, Any]]) -> PackageSolicitationLock:
    id_votes: dict[str, list[dict[str, Any]]] = {}
    for doc in documents:
        doc_key = str(doc.get("sourceId") or doc.get("docKey") or "")
        filename = str(doc.get("fileName") or "")
        weight = _authority_weight(doc)

        sol = doc.get("solicitationNumber")
        if sol:
            norm = _normalize_id(str(sol))
            id_votes.setdefault(norm, []).append(
                {"sourceId": doc_key, "filename": filename, "weight": weight, "source": "metadata"}
            )

        for signal in doc.get("signals") or []:
            if str(signal.get("id") or "") not in {"rfp_solicitation_id_v1", "rfp_solicitation_number_v1"}:
                continue
            value = signal.get("value")
            if not value:
                continue
            norm = _normalize_id(str(value))
            id_votes.setdefault(norm, []).append(
                {"sourceId": doc_key, "filename": filename, "weight": weight, "source": "signal"}
            )

    if not id_votes:
        return PackageSolicitationLock(
            locked_solicitation_id=None,
            confidence="unknown",
            conflicts=[{"reason": "no_solicitation_identifiers_found"}],
        )

    scored: list[tuple[str, int, list[dict[str, Any]]]] = []
    for sol_id, votes in id_votes.items():
        total = sum(int(v.get("weight") or 0) for v in votes)
        scored.append((sol_id, total, votes))
    scored.sort(key=lambda item: (-item[1], item[0]))

    if len(scored) > 1 and scored[0][1] == scored[1][1]:
        return PackageSolicitationLock(
            locked_solicitation_id=None,
            confidence="ambiguous",
            locking_evidence=scored[0][2],
            conflicts=[
                {
                    "reason": "tie_break_unresolved",
                    "candidates": [{"id": s, "score": score, "evidence": ev} for s, score, ev in scored[:3]],
                }
            ],
        )

    locked_id, score, evidence = scored[0]
    associated: list[str] = []
    excluded: list[dict[str, Any]] = []
    for doc in documents:
        doc_key = str(doc.get("sourceId") or doc.get("docKey") or "")
        doc_ids: set[str] = set()
        if doc.get("solicitationNumber"):
            doc_ids.add(_normalize_id(str(doc["solicitationNumber"])))
        for signal in doc.get("signals") or []:
            if str(signal.get("id") or "") in {"rfp_solicitation_id_v1", "rfp_solicitation_number_v1"} and signal.get("value"):
                doc_ids.add(_normalize_id(str(signal["value"])))
        piid_match = None
        for signal in doc.get("signals") or []:
            if str(signal.get("id") or "") == "rfp_prior_contract_piid_v1" and signal.get("value"):
                piid_match = _normalize_id(str(signal["value"]))
        if piid_match and piid_match == locked_id:
            excluded.append({"sourceId": doc_key, "reason": "prior_piid_not_solicitation"})
            continue
        if doc_ids and locked_id not in doc_ids:
            excluded.append({"sourceId": doc_key, "reason": "solicitation_mismatch", "found": sorted(doc_ids)})
            continue
        associated.append(doc_key)

    confidence = "authoritative" if score >= 100 else "provisional"
    return PackageSolicitationLock(
        locked_solicitation_id=locked_id,
        confidence=confidence,
        locking_evidence=evidence,
        associated_documents=associated,
        excluded_documents=excluded,
    )


def filter_documents_for_merge(
    documents: list[dict[str, Any]],
    lock: PackageSolicitationLock,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not lock.locked_solicitation_id or lock.confidence == "ambiguous":
        return documents, lock.excluded_documents
    excluded_ids = {item.get("sourceId") for item in lock.excluded_documents}
    active = [doc for doc in documents if str(doc.get("sourceId") or doc.get("docKey") or "") not in excluded_ids]
    return active, lock.excluded_documents
