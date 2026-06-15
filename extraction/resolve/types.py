from __future__ import annotations

RESOLVED_SIGNALS_VERSION_V1 = "resolved_signals.v1"

RESOLUTION_STATUSES = (
    "passthrough",
    "resolved_equivalent_candidates",
    "resolved_with_conflict",
    "resolved_with_promotion",
    "canonical_derived",
    "canonical_explicit",
    "unresolved_absent",
    "unresolved_conflict",
)

SignalConfidence = str
