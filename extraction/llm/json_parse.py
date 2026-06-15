from __future__ import annotations

import json
import re
from typing import TypeVar

T = TypeVar("T", bound=dict)


def strip_json_fences(raw: str) -> str:
    trimmed = raw.strip()
    match = re.match(r"^```(?:json)?\s*([\s\S]*?)```$", trimmed)
    if match:
        return match.group(1).strip()
    match = re.match(r"^```\s*([\s\S]*?)```$", trimmed)
    if match:
        return match.group(1).strip()
    return trimmed


def parse_json_object(text: str) -> dict:
    cleaned = strip_json_fences(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON: {exc}\nRaw Text was:\n{cleaned}") from exc
    if parsed is None or not isinstance(parsed, dict):
        raise ValueError("Result is not a JSON object")
    return parsed


def extract_output_text_from_response(body: object) -> str | None:
    if body is None or not isinstance(body, dict):
        return None
    output = body.get("output")
    if not isinstance(output, list):
        return None
    for entry in output:
        if not isinstance(entry, dict):
            continue
        content = entry.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if isinstance(item, dict) and item.get("type") == "output_text" and isinstance(item.get("text"), str):
                return item["text"]
    return None
