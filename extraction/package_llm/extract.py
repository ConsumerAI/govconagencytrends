from __future__ import annotations

import json
from typing import Any

import requests

from extraction.config import get_extraction_model, get_extraction_profile, get_extraction_reasoning_effort, get_extraction_timeout_sec, get_openai_api_key
from extraction.llm.json_parse import extract_output_text_from_response, parse_json_object
from extraction.package_llm.fast_normalize import normalize_fast_extraction_payload
from extraction.package_llm.fast_prompt import SYSTEM_INSTRUCTION as FAST_SYSTEM_INSTRUCTION
from extraction.package_llm.fast_prompt import USER_MESSAGE as FAST_USER_MESSAGE
from extraction.package_llm.fast_schema import MARKET_SCOPE_FAST_SIGNAL_IDS, build_fast_openai_json_schema
from extraction.package_llm.prompt import SYSTEM_INSTRUCTION, USER_MESSAGE
from extraction.package_llm.schema import PACKAGE_EXTRACTION_SIGNAL_IDS, SCHEMA_VERSION, build_openai_json_schema
from extraction.package_llm.versions import FAST_SCHEMA_VERSION, PROFILE_VERSIONS
from extraction.types import Finding

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"


def call_package_extraction(
    corpus_text: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
    timeout_sec: float | None = None,
    profile: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[Finding]]:
    api_key = get_openai_api_key()
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    profile = profile or get_extraction_profile()
    is_fast = profile == "market_scope_fast"
    model = model or get_extraction_model()
    reasoning_effort = reasoning_effort or get_extraction_reasoning_effort()
    timeout_sec = timeout_sec or get_extraction_timeout_sec()
    findings: list[Finding] = []

    system_instruction = FAST_SYSTEM_INSTRUCTION if is_fast else SYSTEM_INSTRUCTION
    user_message = FAST_USER_MESSAGE if is_fast else USER_MESSAGE
    schema = build_fast_openai_json_schema() if is_fast else build_openai_json_schema()
    schema_name = "market_scope_fast_v1" if is_fast else "package_extraction_v1"
    expected_ids = MARKET_SCOPE_FAST_SIGNAL_IDS if is_fast else PACKAGE_EXTRACTION_SIGNAL_IDS
    schema_version = FAST_SCHEMA_VERSION if is_fast else SCHEMA_VERSION

    user_content = f"{user_message}\n\n--- SOLICITATION PACKAGE CORPUS ---\n\n{corpus_text}"
    body = {
        "model": model,
        "reasoning": {"effort": reasoning_effort},
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": system_instruction}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_content}]},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "strict": False,
                "name": schema_name,
                "schema": schema,
            },
        },
    }
    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=timeout_sec,
    )
    raw_response_text = response.text
    if not response.ok:
        raise RuntimeError(f"OpenAI Responses API error {response.status_code}: {raw_response_text or response.reason}")

    parsed_body = json.loads(raw_response_text)
    output_text = extract_output_text_from_response(parsed_body)
    if not output_text:
        raise RuntimeError("OpenAI Responses API returned no output_text")

    payload = parse_json_object(output_text)
    usage = parsed_body.get("usage") if isinstance(parsed_body, dict) else {}
    diagnostics = {
        "model": model,
        "reasoningEffort": reasoning_effort,
        "schemaVersion": schema_version,
        "extractionProfile": profile,
        **{k: v for k, v in PROFILE_VERSIONS.get(profile, {}).items() if k != "profile"},
        "inputTokens": usage.get("input_tokens") if isinstance(usage, dict) else None,
        "outputTokens": usage.get("output_tokens") if isinstance(usage, dict) else None,
        "totalTokens": usage.get("total_tokens") if isinstance(usage, dict) else None,
        "cachedInputTokens": usage.get("input_tokens_details", {}).get("cached_tokens")
        if isinstance(usage, dict) and isinstance(usage.get("input_tokens_details"), dict)
        else None,
        "reasoningTokens": usage.get("output_tokens_details", {}).get("reasoning_tokens")
        if isinstance(usage, dict) and isinstance(usage.get("output_tokens_details"), dict)
        else None,
        "openaiRequestCount": 1,
    }

    signals = payload.get("signals") if isinstance(payload, dict) else None
    if not isinstance(signals, list):
        raise RuntimeError("Package extraction response missing signals[]")

    by_id = {str(item.get("id") or ""): item for item in signals if isinstance(item, dict)}
    missing = [sid for sid in expected_ids if sid not in by_id]
    if missing:
        findings.append(
            Finding(
                "warn",
                "PACKAGE_EXTRACTION_MISSING_SIGNALS",
                f"Model omitted {len(missing)} signal ids; filling not_found placeholders.",
                {"missing": missing},
            )
        )
        for sid in missing:
            if is_fast:
                by_id[sid] = {
                    "id": sid,
                    "value": None,
                    "confidence": "low",
                    "status": "not_found",
                    "source_id": None,
                    "filename": None,
                    "page": None,
                    "sheet": None,
                    "quote": "",
                    "review_note": "",
                    "alternates": [],
                }
            else:
                by_id[sid] = {
                    "id": sid,
                    "value": None,
                    "confidence": "low",
                    "status": "not_found",
                    "controlling_source": None,
                    "evidence": [],
                    "reasoning_summary": "Model omitted this signal id.",
                    "alternates": [],
                }
    if is_fast:
        payload = normalize_fast_extraction_payload(
            {"signals": [by_id[sid] for sid in expected_ids], "package_summary": payload.get("package_summary")}
        )
    else:
        payload["signals"] = [by_id[sid] for sid in expected_ids]
    findings.append(Finding("info", "PACKAGE_EXTRACTION_COMPLETE", "Package-level GPT extraction completed"))
    return payload, diagnostics, findings
