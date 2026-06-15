from __future__ import annotations

import re
from typing import Any

from extraction.resolve.evidence_text import collect_record_evidence_text_hits, extract_piid_from_incumbent_blob
from extraction.resolve.helpers import normalize_confidence, rank_confidence
from extraction.resolve.materialize import create_materialized_promoted_signal
from extraction.resolve.phrase_decomposition import decompose_competition_set_aside_phrase
from extraction.resolve.resolve_signals import build_summary


def _is_mislabeled_competition_type(record: dict[str, Any]) -> bool:
    if record.get("id") != "rfp_competition_type_v1":
        return False
    value = str(record.get("canonical_value") or "").strip()
    if not value or re.fullmatch(r"competitive", value, flags=re.IGNORECASE):
        return False
    return bool(re.search(r"(?:set-aside|set aside|8\s*\(\s*a\s*\))", value, flags=re.IGNORECASE))


def _eligible_for_promotion(record: dict[str, Any] | None) -> bool:
    if not record:
        return True
    if _is_mislabeled_competition_type(record):
        return True
    return record.get("canonical_value") is None


def _apply_promotion(
    by_id: dict[str, dict[str, Any]],
    target_id: str,
    value: str,
    confidence: str,
    donor: dict[str, Any],
    *,
    rule_id: str,
    excerpt: str,
    source_artifact: str,
    locator: str,
) -> None:
    existing = by_id.get(target_id)
    if existing and not _eligible_for_promotion(existing):
        return
    promoted = create_materialized_promoted_signal(
        target_id,
        value,
        confidence,
        donor,
        rule_id=rule_id,
        excerpt=excerpt,
        source_artifact=source_artifact,
        locator=locator,
    )
    if existing and existing.get("alternates") is not None:
        promoted["alternates"] = [
            *existing.get("alternates", []),
            {
                "value": existing.get("canonical_value"),
                "confidence": existing.get("canonical_confidence"),
                "evidence": existing.get("evidence"),
                "source_summary": existing.get("source_summary"),
                "suppression_reason": "superseded_by_promotion",
            },
        ]
    by_id[target_id] = promoted


def apply_authoritative_evidence_promotions(artifact: dict[str, Any]) -> dict[str, Any]:
    by_id = {str(record.get("id") or ""): record for record in artifact.get("signals") or []}

    for record in list(by_id.values()):
        if not _is_mislabeled_competition_type(record):
            continue
        for hit in collect_record_evidence_text_hits(record):
            decomposed = decompose_competition_set_aside_phrase(hit["text"])
            if not decomposed:
                continue
            _apply_promotion(
                by_id,
                "rfp_competition_type_v1",
                decomposed["competition_type"],
                "high",
                record,
                rule_id="competition_set_aside_from_eval_phrase_v1",
                excerpt=hit["text"],
                source_artifact=hit["source_artifact"],
                locator=hit["locator"],
            )
            if decomposed.get("set_aside"):
                _apply_promotion(
                    by_id,
                    "rfp_set_aside_v1",
                    decomposed["set_aside"],
                    "high",
                    record,
                    rule_id="set_aside_from_eval_phrase_v1",
                    excerpt=hit["text"],
                    source_artifact=hit["source_artifact"],
                    locator=hit["locator"],
                )
            break

    for donor_id in ("rfp_set_aside_v1", "rfp_competition_type_v1", "rfp_eval_method_v1"):
        donor = by_id.get(donor_id)
        if not donor:
            continue
        for hit in collect_record_evidence_text_hits(donor):
            decomposed = decompose_competition_set_aside_phrase(hit["text"])
            if not decomposed:
                continue
            if decomposed.get("competition_type"):
                _apply_promotion(
                    by_id,
                    "rfp_competition_type_v1",
                    decomposed["competition_type"],
                    "high",
                    donor,
                    rule_id="competition_set_aside_from_eval_phrase_v1",
                    excerpt=hit["text"],
                    source_artifact=hit["source_artifact"],
                    locator=hit["locator"],
                )
            if decomposed.get("set_aside"):
                _apply_promotion(
                    by_id,
                    "rfp_set_aside_v1",
                    decomposed["set_aside"],
                    "high",
                    donor,
                    rule_id="set_aside_from_eval_phrase_v1",
                    excerpt=hit["text"],
                    source_artifact=hit["source_artifact"],
                    locator=hit["locator"],
                )

    incumbent = by_id.get("rfp_incumbent_data_v1")
    if incumbent and rank_confidence(normalize_confidence(incumbent.get("canonical_confidence"))) >= rank_confidence("high"):
        for hit in collect_record_evidence_text_hits(incumbent):
            piid = extract_piid_from_incumbent_blob(hit["text"])
            if not piid:
                continue
            donor_conf = str(incumbent.get("canonical_confidence") or "medium")
            _apply_promotion(
                by_id,
                "rfp_prior_contract_piid_v1",
                piid,
                donor_conf if donor_conf in {"high", "medium", "low"} else "medium",
                incumbent,
                rule_id="prior_piid_from_incumbent_blob_v1",
                excerpt=hit["text"],
                source_artifact=hit["source_artifact"],
                locator=hit["locator"],
            )
            break

    signals = sorted(by_id.values(), key=lambda item: str(item.get("id") or ""))
    return {**artifact, "signals": signals, "summary": build_summary(signals)}
