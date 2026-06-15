from __future__ import annotations

from typing import Any

from extraction.resolve.helpers import build_evidence, build_source_summary
from extraction.resolve.phrase_decomposition import decompose_competition_set_aside_phrase


def create_materialized_promoted_signal(
    signal_id: str,
    value: str,
    confidence: str,
    donor: dict[str, Any],
    *,
    rule_id: str,
    excerpt: str,
    source_artifact: str,
    locator: str,
) -> dict[str, Any]:
    return {
        "id": signal_id,
        "canonical_value": value,
        "canonical_confidence": confidence,
        "resolution_status": "resolved_with_promotion",
        "resolution_basis": "promoted_from_authoritative_evidence",
        "source_summary": build_source_summary([donor]),
        "evidence": build_evidence(donor),
        "alternates": [],
        "notes": f"Promoted from authoritative excerpt via {rule_id}.",
        "promotion": {
            "promotion_rule_id": rule_id,
            "promotion_source_field": str(donor.get("id") or ""),
            "promotion_source_artifact": source_artifact,
            "promotion_locator": locator,
            "promotion_excerpt": excerpt[:280],
            "prior_resolution_status": "missing_signal_id",
            "prior_resolution_basis": "absent_before_promotion",
        },
        "preservation": {
            "immutable": True,
            "contract": "passthrough_single_final_or_promoted",
        },
    }
