from __future__ import annotations

from typing import Any

from extraction.resolve.promote_from_evidence import apply_authoritative_evidence_promotions
from extraction.resolve.resolve_signals import build_summary


def enhance_resolved_signals_v1(artifact: dict[str, Any]) -> dict[str, Any]:
    promoted = apply_authoritative_evidence_promotions(artifact)
    signals: list[dict[str, Any]] = []
    for record in promoted.get("signals") or []:
        updated = dict(record)
        if updated.get("canonical_value") is not None and not updated.get("preservation"):
            updated["preservation"] = {
                "immutable": True,
                "contract": "passthrough_single_final_or_promoted",
            }
        signals.append(updated)
    return {**promoted, "signals": signals, "summary": build_summary(signals)}
