from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import requests

from extraction.config import (
    get_extraction_model,
    get_extraction_timeout_sec,
    get_openai_api_key,
    has_openai_api_key,
)
from extraction.llm.json_parse import extract_output_text_from_response, parse_json_object
from extraction.llm.map_signals import map_extraction_to_signals
from extraction.llm.signal_schema import (
    SYSTEM_INSTRUCTION_TEXT,
    USER_MESSAGE_TEXT,
    build_schema_from_field_list,
    schema_chunks,
)
from extraction.types import Finding

OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


def get_reasoning_effort() -> str:
    return (os.getenv("AEGIS_LLM_REASONING_EFFORT") or "medium").strip()


def _call_responses_api(
    api_key: str,
    model: str,
    corpus_text: str,
    schema_chunk: list[tuple[str, str]],
    chunk_index: int,
    timeout_sec: float,
) -> dict[str, Any]:
    user_content = [
        {"type": "input_text", "text": USER_MESSAGE_TEXT},
        {"type": "input_text", "text": corpus_text},
    ]
    body = {
        "model": model,
        "reasoning": {"effort": get_reasoning_effort()},
        "input": [
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_INSTRUCTION_TEXT}]},
            {"role": "user", "content": user_content},
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "strict": False,
                "name": "aegis_signals_extraction_v1",
                "schema": build_schema_from_field_list(schema_chunk),
            },
        },
    }
    response = requests.post(
        OPENAI_RESPONSES_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=timeout_sec,
    )
    raw_text = response.text
    if not response.ok:
        raise RuntimeError(f"OpenAI Responses API error {response.status_code} (chunk {chunk_index + 1}): {raw_text or response.reason}")

    parsed_body = json.loads(raw_text)
    output_text = extract_output_text_from_response(parsed_body)
    if not output_text:
        raise RuntimeError(f"OpenAI Responses API chunk {chunk_index + 1} had no output_text")

    obj = parse_json_object(output_text)
    obj.pop("_reasoning_scratchpad", None)
    return obj


def _call_chat_completions_api(
    api_key: str,
    model: str,
    corpus_text: str,
    schema_chunk: list[tuple[str, str]],
    chunk_index: int,
    timeout_sec: float,
) -> dict[str, Any]:
    schema = build_schema_from_field_list(schema_chunk)
    body = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": SYSTEM_INSTRUCTION_TEXT},
            {
                "role": "user",
                "content": f"{USER_MESSAGE_TEXT}\n\n--- SOLICITATION CORPUS ---\n\n{corpus_text}",
            },
        ],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "aegis_signals_extraction_v1",
                "strict": False,
                "schema": schema,
            },
        },
    }
    response = requests.post(
        OPENAI_CHAT_URL,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        json=body,
        timeout=timeout_sec,
    )
    raw_text = response.text
    if not response.ok:
        raise RuntimeError(f"OpenAI Chat API error {response.status_code} (chunk {chunk_index + 1}): {raw_text or response.reason}")

    parsed_body = json.loads(raw_text)
    choices = parsed_body.get("choices") if isinstance(parsed_body, dict) else None
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(f"OpenAI Chat API chunk {chunk_index + 1} returned no choices")
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(f"OpenAI Chat API chunk {chunk_index + 1} returned empty content")

    obj = parse_json_object(content)
    obj.pop("_reasoning_scratchpad", None)
    return obj


def _call_schema_chunk(
    api_key: str,
    model: str,
    corpus_text: str,
    schema_chunk: list[tuple[str, str]],
    chunk_index: int,
    timeout_sec: float,
    *,
    prefer_responses_api: bool,
) -> dict[str, Any]:
    if prefer_responses_api:
        try:
            return _call_responses_api(api_key, model, corpus_text, schema_chunk, chunk_index, timeout_sec)
        except Exception as responses_error:
            try:
                return _call_chat_completions_api(api_key, model, corpus_text, schema_chunk, chunk_index, timeout_sec)
            except Exception as chat_error:
                raise RuntimeError(
                    f"Responses API failed ({responses_error}); chat fallback failed ({chat_error})"
                ) from chat_error
    return _call_chat_completions_api(api_key, model, corpus_text, schema_chunk, chunk_index, timeout_sec)


def call_openai_signals_extraction(
    api_key: str,
    corpus_text: str,
    model: str | None = None,
    *,
    timeout_sec: float | None = None,
    prefer_responses_api: bool = True,
) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    if not corpus_text.strip():
        findings.append(Finding("warn", "OPENAI_EMPTY_TEXT", "No text provided for OpenAI extraction."))
        return {}, findings

    resolved_model = model or get_extraction_model()
    resolved_timeout = timeout_sec if timeout_sec is not None else get_extraction_timeout_sec()
    chunks = schema_chunks()
    merged: dict[str, Any] = {}

    with ThreadPoolExecutor(max_workers=len(chunks)) as executor:
        futures = {
            executor.submit(
                _call_schema_chunk,
                api_key,
                resolved_model,
                corpus_text,
                chunk,
                index,
                resolved_timeout,
                prefer_responses_api=prefer_responses_api,
            ): index
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            chunk_index = futures[future]
            try:
                result = future.result()
                merged.update(result)
            except Exception as exc:
                findings.append(
                    Finding(
                        "error",
                        "OPENAI_API_FAILED",
                        f"OpenAI extraction failed for schema chunk {chunk_index + 1}: {exc}",
                    )
                )

    if not merged and not any(item.code == "OPENAI_EMPTY_TEXT" for item in findings):
        findings.append(Finding("error", "OPENAI_EXTRACTION_EMPTY", "OpenAI extraction returned no fields."))
    return merged, findings


def extract_signals_from_corpus(
    run_id: str,
    corpus_text: str,
    *,
    fallback_source_file: str | None = None,
    base_signals: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[Finding], str | None]:
    """Return merged raw signals array, findings, and model used (or None if skipped)."""
    api_key = get_openai_api_key()
    if not api_key:
        return (
            list(base_signals or []),
            [Finding("warn", "OPENAI_API_KEY_MISSING", "OPENAI_API_KEY is not set; signal extraction skipped.")],
            None,
        )

    model = get_extraction_model()
    extraction_result, api_findings = call_openai_signals_extraction(api_key, corpus_text, model)
    findings = list(api_findings)
    if any(item.level == "error" for item in api_findings):
        return list(base_signals or []), findings, model

    mapped_signals, map_findings = map_extraction_to_signals(
        run_id,
        extraction_result,
        fallback_source_file=fallback_source_file,
    )
    findings.extend(map_findings)

    merged_by_id: dict[str, dict[str, Any]] = {}
    for signal in base_signals or []:
        signal_id = str(signal.get("id") or "").strip()
        if signal_id:
            merged_by_id[signal_id] = signal
    for signal in mapped_signals:
        merged_by_id[str(signal["id"])] = signal

    merged = sorted(merged_by_id.values(), key=lambda item: str(item.get("id", "")))
    return merged, findings, model
