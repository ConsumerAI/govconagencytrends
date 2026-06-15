from __future__ import annotations

import re
from typing import Any

from extraction.package_llm.corpus import PackageCorpus, PackageSourceText, source_lookup
from extraction.package_llm.solicitation_id import canonical_base_solicitation_id, is_prior_piid_reference, normalize_piid_token
from extraction.package_llm.validators.evidence_match import match_evidence_quote
from extraction.types import Finding

CHECKBOX_LABEL_RE = re.compile(
    r"\b(unrestricted|full\s+and\s+open|partial\s+small\s+business|total\s+small\s+business)\b",
    re.I,
)
CLAUSE_NAICS_RE = re.compile(
    r"\b(?:FAR|DFARS|52\.212-1|52\.219|environmental|remediation|562910|prior\s+contract|historical)\b",
    re.I,
)
AMENDMENT_EFFECTIVE_RE = re.compile(
    r"\b(?:effective\s+date|modification\s+effective|amendment\s+effective|date\s+of\s+modification)\b",
    re.I,
)
PROPOSAL_DUE_RE = re.compile(
    r"\b(?:proposal|offer|quotation|bid)s?\s+(?:due|closing|receipt|submit(?:ted)?|received|deadline)\b",
    re.I,
)
POC_EMAIL_CONTEXT_RE = re.compile(
    r"\b(?:poc|point\s+of\s+contact|contracting\s+officer|contract\s+specialist|questions?|q\s*&\s*a|ppq|protest|receipt\s+confirmation)\b",
    re.I,
)
FORM_NAICS_CONTEXT_RE = re.compile(
    r"\b(?:pdf form fields|10naics|north american industry classification standard\s*\(naics\))\b",
    re.I,
)
PIID_RE = re.compile(r"\b[A-Z]{2}\d{4}[A-Z0-9]{2}[A-Z]\d{4}\b")  # reserved for future PIID cross-checks


def normalize_quote(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def validate_source_exists(source_id: str | None, lookup: dict[str, PackageSourceText]) -> bool:
    return bool(source_id and source_id in lookup)


def validate_page_or_sheet_exists(source: PackageSourceText, *, page: int | None, sheet: str | None) -> bool:
    if sheet:
        return sheet in source.normalized_text_by_sheet or any(
            name.lower() == sheet.lower() for name in source.normalized_text_by_sheet
        )
    if page is None:
        return True
    index = page - 1 if page > 0 else page
    return index in source.normalized_text_by_page


def validate_evidence_quote(
    source: PackageSourceText,
    *,
    page: int | None,
    sheet: str | None,
    quote: str,
    source_id: str | None = None,
):
    return match_evidence_quote(source, page=page, sheet=sheet, quote=quote, source_id=source_id)


def _scalar_validation_value(signal_id: str, value: Any) -> Any:
    if isinstance(value, dict):
        if signal_id == "rfp_primary_naics_v1":
            return value.get("naics") or value.get("code") or value.get("value")
        if signal_id == "rfp_primary_psc_v1":
            return value.get("psc") or value.get("code") or value.get("value")
    return value


def validate_basic_format(signal_id: str, value: Any) -> list[Finding]:
    findings: list[Finding] = []
    if value is None:
        return findings
    check_value = _scalar_validation_value(signal_id, value)
    if signal_id == "rfp_primary_naics_v1":
        text = str(check_value or "").strip()
        if not re.fullmatch(r"\d{6}", text):
            findings.append(Finding("warn", "NAICS_FORMAT_INVALID", f"NAICS must be six digits: {value}", {"signalId": signal_id}))
    if signal_id == "rfp_primary_psc_v1":
        text = str(check_value or "").strip().upper()
        if not re.fullmatch(r"[A-Z0-9]{4}", text):
            findings.append(Finding("warn", "PSC_FORMAT_INVALID", f"PSC must be four characters: {value}", {"signalId": signal_id}))
    if signal_id in {"solicitation_is_stalled_v1"} and not isinstance(value, bool):
        lowered = str(value).strip().lower()
        if lowered not in {"true", "false", "yes", "no"}:
            findings.append(Finding("warn", "BOOLEAN_FORMAT_INVALID", f"Expected boolean: {value}", {"signalId": signal_id}))
    return findings


def validate_signal_semantics(signal: dict[str, Any], corpus: PackageCorpus) -> list[Finding]:
    findings: list[Finding] = []
    signal_id = str(signal.get("id") or "")
    value = signal.get("value")
    if value is None:
        return findings

    evidence_items = signal.get("evidence") or []
    primary_quote = ""
    if evidence_items and isinstance(evidence_items[0], dict):
        primary_quote = str(evidence_items[0].get("quote") or "")

    combined_evidence = " ".join(
        str(item.get("quote") or "") for item in evidence_items if isinstance(item, dict)
    )

    if signal_id == "rfp_primary_naics_v1":
        has_form_evidence = bool(FORM_NAICS_CONTEXT_RE.search(combined_evidence))
        if CLAUSE_NAICS_RE.search(combined_evidence) and not has_form_evidence and not re.search(
            r"\b(?:acquisition|solicitation|contract|primary|naics)\b", combined_evidence, re.I
        ):
            findings.append(
                Finding(
                    "warn",
                    "NAICS_CLAUSE_CONTEXT",
                    "NAICS evidence appears to come from clause or contextual language rather than acquisition NAICS.",
                    {"signalId": signal_id, "value": value},
                )
            )
        if (
            "562910" in str(value)
            and "722310" not in combined_evidence
            and CLAUSE_NAICS_RE.search(combined_evidence)
            and not has_form_evidence
        ):
            findings.append(
                Finding(
                    "warn",
                    "NAICS_CLAUSE_EXAMPLE_REJECTED",
                    "NAICS 562910 appears to be clause/environmental context, not solicitation NAICS.",
                    {"signalId": signal_id},
                )
            )

    if signal_id == "rfp_primary_psc_v1" and not re.search(
        r"\b(?:product\s+service\s+code|psc|service\s+code)\b", combined_evidence, re.I
    ):
        findings.append(
            Finding(
                "warn",
                "PSC_CONTEXT_WEAK",
                "PSC evidence does not clearly identify acquisition product/service code.",
                {"signalId": signal_id, "value": value},
            )
        )

    if signal_id == "rfp_set_aside_v1":
        if CHECKBOX_LABEL_RE.search(combined_evidence) and not re.search(
            r"\b(?:set[- ]?aside|selected|8\s*\(\s*a\s*\)|small business set[- ]aside)\b", combined_evidence, re.I
        ):
            findings.append(
                Finding(
                    "warn",
                    "SET_ASIDE_CHECKBOX_ONLY",
                    "Set-aside evidence appears to be unselected checkbox labels only.",
                    {"signalId": signal_id, "value": value},
                )
            )

    if signal_id == "rfp_competition_type_v1":
        if CHECKBOX_LABEL_RE.search(combined_evidence) and "set-aside" not in combined_evidence.lower():
            findings.append(
                Finding(
                    "warn",
                    "COMPETITION_CHECKBOX_ONLY",
                    "Competition type evidence appears to come from unselected form labels.",
                    {"signalId": signal_id, "value": value},
                )
            )

    if signal_id in {"rfp_questions_due_v1"}:
        pass
    if signal_id in {"rfp_submission_destination_v1", "rfp_submission_method_v1", "rfp_submission_instructions_v1"}:
        if "@" in str(value) and POC_EMAIL_CONTEXT_RE.search(combined_evidence) and not re.search(
            r"\b(?:submit|submission|proposal|offer|electronic)\b", combined_evidence, re.I
        ):
            findings.append(
                Finding(
                    "warn",
                    "SUBMISSION_POC_EMAIL",
                    "Submission destination appears to be a contact/Q&A email rather than proposal submission instructions.",
                    {"signalId": signal_id, "value": value},
                )
            )

    if signal_id == "rfp_solicitation_id_v1":
        base_id, _ = canonical_base_solicitation_id(value)
        text = str(value).strip().upper() if not isinstance(value, dict) else (base_id or "")
        normalized = normalize_piid_token(text or str(value))
        if base_id and re.fullmatch(r"[A-Z]{2}\d{4}[A-Z0-9]{2}[A-Z]\d{4}", base_id):
            pass
        elif len(normalized) > 13 and re.search(r"000\d+$", normalized):
            findings.append(
                Finding(
                    "warn",
                    "SOLICITATION_ID_AMENDMENT_SUFFIX",
                    "Solicitation ID appears to include amendment suffix rather than base solicitation id.",
                    {"signalId": signal_id, "value": value},
                )
            )
        if is_prior_piid_reference(combined_evidence, candidate=text or normalized):
            findings.append(
                Finding(
                    "warn",
                    "SOLICITATION_ID_PRIOR_PIID",
                    "Solicitation ID matches a prior-contract PIID reference in evidence.",
                    {"signalId": signal_id, "value": value},
                )
            )

    if signal_id in {"rfp_questions_due_v1"}:
        if PROPOSAL_DUE_RE.search(combined_evidence) and not re.search(
            r"\b(?:question|clarification|q\s*&\s*a)\b", combined_evidence, re.I
        ):
            findings.append(
                Finding(
                    "warn",
                    "QUESTIONS_DUE_PROPOSAL_CONTEXT",
                    "Questions due evidence appears to reference proposal deadline language.",
                    {"signalId": signal_id},
                )
            )

    # Proposal due date is represented via submission instructions cluster; check any date-like submission signal
    if signal_id == "rfp_submission_instructions_v1" or (
        signal_id == "rfp_questions_due_v1" and False
    ):
        pass

    if signal_id == "rfp_period_of_performance_v1" or signal_id == "rfp_pop_end_v1":
        if AMENDMENT_EFFECTIVE_RE.search(combined_evidence) and not PROPOSAL_DUE_RE.search(combined_evidence):
            if re.search(r"\b(?:202[0-9]|due|deadline)\b", str(value)):
                findings.append(
                    Finding(
                        "warn",
                        "DUE_DATE_AMENDMENT_EFFECTIVE",
                        "Date evidence appears to reference amendment effective date rather than proposal due date.",
                        {"signalId": signal_id, "value": value},
                    )
                )

    if "due" in signal_id or signal_id.endswith("_due_v1"):
        if AMENDMENT_EFFECTIVE_RE.search(combined_evidence) and not PROPOSAL_DUE_RE.search(combined_evidence):
            findings.append(
                Finding(
                    "warn",
                    "DUE_DATE_AMENDMENT_EFFECTIVE",
                    "Due date evidence appears to reference amendment effective date.",
                    {"signalId": signal_id, "value": value},
                )
            )

    _ = primary_quote
    return findings


def _controlling_source_id(signal: dict[str, Any]) -> str | None:
    controlling = signal.get("controlling_source")
    if isinstance(controlling, dict):
        source_id = str(controlling.get("source_id") or "").strip()
        return source_id or None
    return None


def _is_authoritative_evidence(signal: dict[str, Any], item: dict[str, Any], index: int) -> bool:
    if index == 0:
        return True
    controlling_id = _controlling_source_id(signal)
    source_id = str(item.get("source_id") or "").strip()
    if controlling_id and source_id == controlling_id:
        return True
    page = item.get("page")
    controlling = signal.get("controlling_source")
    if isinstance(controlling, dict) and page == controlling.get("page"):
        return True
    return False


def validate_signal_record(signal: dict[str, Any], corpus: PackageCorpus) -> tuple[dict[str, Any], list[Finding]]:
    findings: list[Finding] = []
    lookup = source_lookup(corpus)
    validated = dict(signal)
    status = str(validated.get("status") or "not_found")
    confidence = str(validated.get("confidence") or "low")
    value = validated.get("value")

    if value is None and status == "confirmed":
        status = "not_found"
        confidence = "low"

    evidence_validation: dict[str, Any] = {
        "validatedEvidenceCount": 0,
        "failedEvidenceCount": 0,
        "authoritativeEvidenceValidated": False,
        "matches": [],
    }

    if value is not None:
        findings.extend(validate_basic_format(str(validated.get("id") or ""), value))
        evidence_items = validated.get("evidence") or []
        if not evidence_items:
            findings.append(
                Finding(
                    "warn",
                    "EVIDENCE_MISSING",
                    "Non-null signal lacks evidence citations.",
                    {"signalId": validated.get("id")},
                )
            )
        for index, item in enumerate(evidence_items):
            if not isinstance(item, dict):
                continue
            source_id = str(item.get("source_id") or "")
            page = item.get("page")
            sheet = item.get("sheet")
            quote = str(item.get("quote") or "")
            authoritative = _is_authoritative_evidence(validated, item, index)
            if not validate_source_exists(source_id, lookup):
                evidence_validation["failedEvidenceCount"] += 1
                findings.append(
                    Finding(
                        "warn",
                        "EVIDENCE_SOURCE_MISSING",
                        f"Cited source_id not found: {source_id}",
                        {"signalId": validated.get("id"), "sourceId": source_id, "authoritative": authoritative},
                    )
                )
                continue
            source = lookup[source_id]
            page_int = page if isinstance(page, int) else None
            sheet_str = sheet if isinstance(sheet, str) else None
            if not validate_page_or_sheet_exists(source, page=page_int, sheet=sheet_str):
                findings.append(
                    Finding(
                        "warn",
                        "EVIDENCE_LOCATOR_MISSING",
                        "Cited page/sheet does not exist in source.",
                        {"signalId": validated.get("id"), "sourceId": source_id, "page": page, "sheet": sheet},
                    )
                )
            match_result = validate_evidence_quote(
                source,
                page=page_int,
                sheet=sheet_str,
                quote=quote,
                source_id=source_id,
            )
            match_record = {
                "index": index,
                "authoritative": authoritative,
                "matched": match_result.matched,
                "method": match_result.method,
                "score": round(match_result.score, 3),
                "citedPage": match_result.cited_page,
                "citedPageIndex": match_result.cited_page_index,
                "matchedPage": match_result.matched_page,
                "matchedPageIndex": match_result.matched_page_index,
                "sourceId": source_id,
                **match_result.diagnostics,
            }
            evidence_validation["matches"].append(match_record)
            if match_result.matched:
                evidence_validation["validatedEvidenceCount"] += 1
                if authoritative:
                    evidence_validation["authoritativeEvidenceValidated"] = True
            elif quote:
                evidence_validation["failedEvidenceCount"] += 1
                findings.append(
                    Finding(
                        "warn",
                        "EVIDENCE_QUOTE_NOT_FOUND",
                        "Evidence quote not found in cited source text.",
                        {
                            "signalId": validated.get("id"),
                            "sourceId": source_id,
                            "quote": quote[:120],
                            "authoritative": authoritative,
                            "matchScore": round(match_result.score, 3),
                            **match_result.diagnostics,
                        },
                    )
                )

        controlling = validated.get("controlling_source")
        if isinstance(controlling, dict):
            source_id = str(controlling.get("source_id") or "")
            if source_id and not validate_source_exists(source_id, lookup):
                findings.append(
                    Finding(
                        "warn",
                        "CONTROLLING_SOURCE_MISSING",
                        f"Controlling source_id not found: {source_id}",
                        {"signalId": validated.get("id")},
                    )
                )

        findings.extend(validate_signal_semantics(validated, corpus))

    quote_failures = [f for f in findings if f.code == "EVIDENCE_QUOTE_NOT_FOUND"]
    source_failures = [f for f in findings if f.code in {"EVIDENCE_SOURCE_MISSING", "EVIDENCE_LOCATOR_MISSING"}]
    semantic_fail_codes = {
        "NAICS_CLAUSE_EXAMPLE_REJECTED",
        "SET_ASIDE_CHECKBOX_ONLY",
        "COMPETITION_CHECKBOX_ONLY",
        "DUE_DATE_AMENDMENT_EFFECTIVE",
        "SUBMISSION_POC_EMAIL",
        "SOLICITATION_ID_AMENDMENT_SUFFIX",
        "SOLICITATION_ID_PRIOR_PIID",
    }
    hard_semantic_failures = [f for f in findings if f.code in semantic_fail_codes]

    if value is not None:
        if evidence_validation["authoritativeEvidenceValidated"] or (
            evidence_validation["validatedEvidenceCount"] > 0 and status in {"confirmed", "not_applicable"}
        ):
            if hard_semantic_failures:
                status = "review_required"
                if confidence == "high":
                    confidence = "medium"
            elif quote_failures and not evidence_validation["authoritativeEvidenceValidated"]:
                status = "review_required"
                if confidence == "high":
                    confidence = "medium"
            elif status == "confirmed":
                pass
        elif source_failures or (quote_failures and evidence_validation["validatedEvidenceCount"] == 0):
            status = "review_required"
            if confidence == "high":
                confidence = "medium"
        elif quote_failures:
            status = "review_required"
            if confidence == "high":
                confidence = "medium"
        elif any(f.level == "warn" for f in findings) and status == "confirmed":
            status = "review_required"
            confidence = "medium" if confidence == "high" else confidence

    validated["status"] = status
    validated["confidence"] = confidence
    validated["validation_findings"] = [item.to_dict() for item in findings]
    validated["evidence_validation"] = evidence_validation
    return validated, findings
