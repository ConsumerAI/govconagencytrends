from __future__ import annotations

import json
import re
from typing import Any

from extraction.types import Finding


def serialize_odc_plugs_value(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    return str(value).strip() or None


def apply_odc_serialization(signals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    for signal in signals:
        if str(signal.get("id") or "") != "rfp_odc_plugs_v1":
            updated.append(signal)
            continue
        serialized = serialize_odc_plugs_value(signal.get("value"))
        if serialized:
            updated.append({**signal, "value": serialized})
        else:
            updated.append(signal)
    return updated
