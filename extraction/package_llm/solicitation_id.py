from __future__ import annotations

import re
from typing import Any

PIID_TOKEN_RE = re.compile(r"[A-Z]{2}\d{4}[A-Z0-9]{2}[A-Z]\d{4}")
AMENDMENT_TAIL_RE = re.compile(r"000\d$")


def normalize_piid_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", (value or "").upper())


def strip_amendment_suffix(token: str) -> str | None:
    cleaned = normalize_piid_token(token)
    if not cleaned:
        return None
    if len(cleaned) > 13 and AMENDMENT_TAIL_RE.search(cleaned):
        return cleaned[:13]
    match = PIID_TOKEN_RE.search(cleaned)
    if match:
        return match.group(0)
    if len(cleaned) == 13:
        return cleaned
    return cleaned or None


def canonical_base_solicitation_id(value: Any) -> tuple[str | None, dict[str, Any] | None]:
    if value is None:
        return None, None
    if isinstance(value, (str, int, float, bool)):
        base = strip_amendment_suffix(str(value))
        structured = {"rawValue": str(value)}
        if base and normalize_piid_token(str(value)) != base:
            structured["variants"] = [str(value)]
        return base, structured if structured.get("variants") else None

    if not isinstance(value, dict):
        base = strip_amendment_suffix(str(value))
        return base, None

    structured = dict(value)
    variants: list[str] = []
    for key in ("display_id", "display_variant", "formatted_id", "hyphenated_id", "variant"):
        item = structured.get(key)
        if item:
            variants.append(str(item))
    for key in ("amendment_ids", "amendments", "amendment_identifiers"):
        items = structured.get(key)
        if isinstance(items, list):
            variants.extend(str(item) for item in items if item)

    for key in ("base_solicitation_id", "base_id", "solicitation_id", "piid", "id", "value"):
        candidate = structured.get(key)
        if candidate:
            base = strip_amendment_suffix(str(candidate))
            if base:
                if variants:
                    structured["variants"] = variants
                return base, structured

    for key in ("solicitation_number", "number", "label"):
        candidate = structured.get(key)
        if candidate:
            base = strip_amendment_suffix(str(candidate))
            if base:
                if variants:
                    structured["variants"] = variants
                return base, structured

    return None, structured if variants else None


def is_prior_piid_reference(value: str, *, candidate: str) -> bool:
    upper = (value or "").upper()
    token = normalize_piid_token(candidate)
    if not token:
        return False
    return "PRIOR" in upper and token in normalize_piid_token(value)
