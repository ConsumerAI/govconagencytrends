from __future__ import annotations

import re
from typing import Any

from extraction.documents.amendment import index_of_highest_precedence, is_amendment_filename
from extraction.signals.authority import authority_rank
from extraction.types import Finding

# AEGIS target signals plus GovCon priority fields governed by amendment precedence.
AMENDMENT_TARGET_SIGNAL_IDS: tuple[str, ...] = (
    "rfp_due_date_v1",
    "rfp_due_datetime_local_v1",
    "rfp_questions_due_v1",
    "rfp_submission_instructions_v1",
    "rfp_submission_method_v1",
    "rfp_submission_destination_v1",
    "rfp_submission_format_v1",
    "rfp_evaluation_criteria_v1",
    "rfp_eval_method_v1",
    "rfp_eval_weights_v1",
    "rfp_tech_factors_v1",
    "rfp_primary_poc_v1",
    "rfp_issuing_office_v1",
    "rfp_primary_naics_v1",
    "rfp_primary_psc_v1",
    "rfp_set_aside_v1",
    "rfp_competition_type_v1",
    "rfp_contract_type_v1",
    "rfp_place_of_performance_v1",
    "rfp_pop_start_v1",
    "rfp_pop_end_v1",
    "rfp_pop_years_v1",
    "rfp_page_limits_v1",
    "rfp_pricing_constraints_v1",
    "rfp_incumbent_data_v1",
    "rfp_prior_contract_piid_v1",
)


def _cmp_lex(a: str, b: str) -> int:
    return (a > b) - (a < b)


def _has_value(signal: dict[str, Any] | None) -> bool:
    if not signal:
        return False
    if signal.get("_deleted"):
        return False
    value = signal.get("value")
    return value is not None and str(value).strip() != ""


def _value_str(signal: dict[str, Any]) -> str:
    return str(signal.get("value") or "").strip()


def _excerpt_of(signal: dict[str, Any]) -> str:
    ev1 = signal.get("evidence_v1")
    if isinstance(ev1, dict) and isinstance(ev1.get("excerpt"), str):
        return ev1["excerpt"]
    evidence = signal.get("evidence")
    if isinstance(evidence, list) and evidence:
        first = evidence[0]
        if isinstance(first, dict) and isinstance(first.get("snippet"), str):
            return first["snippet"]
    return ""


def _doc_meta(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_id": doc.get("sourceId") or doc.get("docKey"),
        "filename": doc.get("fileName") or doc.get("originalFilename"),
        "document_type": doc.get("documentType"),
        "amendment_order": doc.get("amendmentOrder"),
    }


def _provenance_for(signal: dict[str, Any], doc: dict[str, Any], *, supersedes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    docset = signal.get("_docset") if isinstance(signal.get("_docset"), dict) else {}
    meta = _doc_meta(doc)
    return {
        "controlling_source_id": docset.get("source_id") or meta["source_id"],
        "controlling_filename": docset.get("filename") or meta["filename"],
        "controlling_document_type": docset.get("document_type") or meta["document_type"],
        "controlling_amendment": docset.get("amendment_order") or meta["amendment_order"],
        "supersedes": supersedes or [],
    }


def _authority_score(signal: dict[str, Any]) -> int:
    authority = signal.get("authority")
    if isinstance(authority, dict):
        return authority_rank(authority)
    return 0


def _attachment_tokens(doc: dict[str, Any]) -> set[str]:
    filename = str(doc.get("fileName") or doc.get("originalFilename") or "").lower()
    tokens: set[str] = set()
    for match in re.finditer(r"attachment[\s_-]*(\d+|[a-z])", filename):
        tokens.add(match.group(1).lower())
    return tokens


def _is_superseded_attachment_doc(doc: dict[str, Any], documents: list[dict[str, Any]]) -> bool:
    tokens = _attachment_tokens(doc)
    if not tokens:
        return False
    for amd_doc in documents:
        if not (amd_doc.get("isAmendment") or is_amendment_filename(str(amd_doc.get("fileName") or ""))):
            continue
        if not _order_confident(amd_doc):
            continue
        for label in amd_doc.get("revisedAttachments") or []:
            token = str(label).replace("Attachment", "").strip().lower()
            if token in tokens:
                return True
    return False


def _filter_superseded_attachment_candidates(
    per_doc: list[tuple[int, dict[str, Any]]],
    documents: list[dict[str, Any]],
) -> list[tuple[int, dict[str, Any]]]:
    active = [
        (idx, sig)
        for idx, sig in per_doc
        if not _is_superseded_attachment_doc(documents[idx], documents)
    ]
    return active or per_doc


def _pick_highest_authority(populated: list[tuple[int, dict[str, Any]]]) -> tuple[int, dict[str, Any]]:
    return max(populated, key=lambda item: (_authority_score(item[1]), item[0]))


def _order_confident(doc: dict[str, Any]) -> bool:
    confidence = str(doc.get("amendmentOrderConfidence") or "").lower()
    source = str(doc.get("amendmentOrderSource") or "").lower()
    if confidence == "authoritative":
        return True
    return source in {"sf30", "body", "form"}


def _order_key(doc: dict[str, Any]) -> tuple[int, int]:
    order_raw = doc.get("amendmentOrder")
    try:
        order_num = int(str(order_raw)) if order_raw is not None else -1
    except ValueError:
        order_num = -1
    confidence_rank = 1 if _order_confident(doc) else 0
    return (confidence_rank, order_num)


def _index_of_controlling_amendment(documents: list[dict[str, Any]]) -> int:
    amendment_docs = [
        (index, doc)
        for index, doc in enumerate(documents)
        if doc.get("isAmendment") or is_amendment_filename(str(doc.get("fileName") or ""))
    ]
    if not amendment_docs:
        return index_of_highest_precedence(documents)
    confident = [(i, d) for i, d in amendment_docs if _order_confident(d) and d.get("amendmentOrder")]
    if confident:
        return max(confident, key=lambda item: _order_key(item[1]))[0]
    return amendment_docs[-1][0]


def _attach_provenance(signal: dict[str, Any], provenance: dict[str, Any]) -> dict[str, Any]:
    merged = dict(signal)
    merged["docset_provenance"] = provenance
    return merged


def merge_solicitation_set_signals(
    documents: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[Finding], int, int]:
    """Port of mergeSolicitationSetSignalsV1 with controlling provenance metadata."""
    findings: list[Finding] = []
    superseded_count = 0
    unresolved_conflicts = 0

    if not documents:
        return [], findings, 0, 0

    if len(documents) == 1:
        doc = documents[0]
        signals = list(doc.get("signals") or [])
        signals.sort(key=lambda item: str(item.get("id") or ""))
        doc_findings = doc.get("findings") or []
        if isinstance(doc_findings, list):
            for item in doc_findings:
                if isinstance(item, Finding):
                    findings.append(item)
                elif isinstance(item, dict):
                    findings.append(
                        Finding(
                            str(item.get("level") or "info"),
                            str(item.get("code") or ""),
                            str(item.get("message") or ""),
                            item.get("details") if isinstance(item.get("details"), dict) else None,
                        )
                    )
        provenance = _provenance_for(signal, doc, supersedes=[])
        tagged = [_attach_provenance(item, _provenance_for(item, doc, supersedes=[])) for item in signals]
        return tagged, findings, 0, 0

    highest_idx = _index_of_controlling_amendment(documents)
    amendment_indices = [
        index
        for index, doc in enumerate(documents)
        if doc.get("isAmendment") or is_amendment_filename(str(doc.get("fileName") or ""))
    ]

    all_ids: set[str] = set()
    for doc in documents:
        for signal in doc.get("signals") or []:
            if signal.get("id"):
                all_ids.add(str(signal["id"]))
    sorted_ids = sorted(all_ids, key=lambda item: item)

    merged_signals: list[dict[str, Any]] = []

    def get_signal(doc: dict[str, Any], signal_id: str) -> dict[str, Any] | None:
        for signal in doc.get("signals") or []:
            if str(signal.get("id") or "") == signal_id:
                return signal
        return None

    for signal_id in sorted_ids:
        is_target = signal_id in AMENDMENT_TARGET_SIGNAL_IDS
        per_doc: list[tuple[int, dict[str, Any]]] = []
        for index, doc in enumerate(documents):
            signal = get_signal(doc, signal_id)
            if signal is not None:
                per_doc.append((index, signal))

        if not per_doc:
            continue

        if not is_target:
            high_sig = get_signal(documents[highest_idx], signal_id)
            chosen: dict[str, Any] | None = None
            chosen_idx = highest_idx
            if _has_value(high_sig):
                chosen = high_sig
            else:
                populated = _filter_superseded_attachment_candidates(
                    [(idx, sig) for idx, sig in per_doc if _has_value(sig)],
                    documents,
                )
                if populated:
                    chosen_idx, chosen = _pick_highest_authority(populated)
            if chosen is None and per_doc:
                chosen_idx, chosen = per_doc[-1]
            if chosen is None:
                continue
            supersedes = [
                {
                    "value": _value_str(sig),
                    "source_id": _doc_meta(documents[idx])["source_id"],
                    "filename": _doc_meta(documents[idx])["filename"],
                    "amendment": _doc_meta(documents[idx])["amendment_order"],
                    "reason_not_selected": "lower precedence document",
                }
                for idx, sig in per_doc
                if idx != chosen_idx and _has_value(sig) and _value_str(sig) != _value_str(chosen)
            ]
            superseded_count += len(supersedes)
            merged_signals.append(
                _attach_provenance(chosen, _provenance_for(chosen, documents[chosen_idx], supersedes=supersedes))
            )
            continue

        # Explicit deletion: last amendment touching this field wins if it deleted and nothing later reinstates.
        last_touch_idx = -1
        last_touch_sig: dict[str, Any] | None = None
        for idx, sig in per_doc:
            if idx in amendment_indices or documents[idx].get("isAmendment"):
                last_touch_idx = idx
                last_touch_sig = sig
        if last_touch_sig and last_touch_sig.get("_deleted"):
            reinstated = any(
                idx > last_touch_idx and _has_value(sig) and not sig.get("_deleted")
                for idx, sig in per_doc
            )
            if not reinstated:
                supersedes = [
                    {
                        "value": _value_str(sig),
                        "source_id": _doc_meta(documents[idx])["source_id"],
                        "filename": _doc_meta(documents[idx])["filename"],
                        "amendment": _doc_meta(documents[idx])["amendment_order"],
                        "reason_not_selected": "explicitly deleted by amendment",
                    }
                    for idx, sig in per_doc
                    if _has_value(sig)
                ]
                superseded_count += len(supersedes)
                findings.append(
                    Finding(
                        "info",
                        "AMENDMENT_FIELD_DELETED",
                        f"Field {signal_id} explicitly removed by amendment",
                        {"signalId": signal_id, "supersedes": supersedes},
                    )
                )
                continue

        # Explicit deletion on highest doc (legacy path)
        high_sig = get_signal(documents[highest_idx], signal_id)
        if high_sig and high_sig.get("_deleted"):
            supersedes = [
                {
                    "value": _value_str(sig),
                    "source_id": _doc_meta(documents[idx])["source_id"],
                    "filename": _doc_meta(documents[idx])["filename"],
                    "amendment": _doc_meta(documents[idx])["amendment_order"],
                    "reason_not_selected": "explicitly deleted by controlling amendment",
                }
                for idx, sig in per_doc
                if _has_value(sig)
            ]
            superseded_count += len(supersedes)
            findings.append(
                Finding(
                    "info",
                    "AMENDMENT_FIELD_DELETED",
                    f"Field {signal_id} explicitly removed by controlling amendment",
                    {"signalId": signal_id, "supersedes": supersedes},
                )
            )
            continue

        amendment_values = [
            _value_str(get_signal(documents[i], signal_id) or {})
            for i in amendment_indices
            if _has_value(get_signal(documents[i], signal_id))
        ]
        distinct_amendment_values = sorted(set(amendment_values), key=lambda item: item)

        if len(distinct_amendment_values) > 1:
            amendment_populated = [
                (i, get_signal(documents[i], signal_id))
                for i in amendment_indices
                if _has_value(get_signal(documents[i], signal_id))
            ]
            confident_amendments = [
                (i, sig)
                for i, sig in amendment_populated
                if _order_confident(documents[i]) and documents[i].get("amendmentOrder")
            ]
            if len(confident_amendments) >= len(amendment_populated) and confident_amendments:
                chosen_idx, chosen = max(
                    confident_amendments,
                    key=lambda item: _order_key(documents[item[0]]),
                )
                high_val = _value_str(chosen)
                supersedes = [
                    {
                        "value": _value_str(sig),
                        "source_id": _doc_meta(documents[idx])["source_id"],
                        "filename": _doc_meta(documents[idx])["filename"],
                        "amendment": _doc_meta(documents[idx])["amendment_order"],
                        "reason_not_selected": "superseded by controlling amendment",
                    }
                    for idx, sig in amendment_populated
                    if idx != chosen_idx and sig and _value_str(sig) != high_val
                ]
                superseded_count += len(supersedes)
                findings.append(
                    Finding(
                        "info",
                        "AMENDMENT_CONTROLLING_VALUE",
                        f"Controlling amendment selected for {signal_id}",
                        {
                            "signalId": signal_id,
                            "selectedValue": high_val,
                            "controllingAmendment": _doc_meta(documents[chosen_idx])["amendment_order"],
                        },
                    )
                )
                merged_signals.append(
                    _attach_provenance(chosen, _provenance_for(chosen, documents[chosen_idx], supersedes=supersedes))
                )
                continue

            unresolved_conflicts += 1
            candidates = [
                {
                    "value": _value_str(sig),
                    "source_id": _doc_meta(documents[idx])["source_id"],
                    "filename": _doc_meta(documents[idx])["filename"],
                    "amendment": _doc_meta(documents[idx])["amendment_order"],
                    "amendment_order_source": documents[idx].get("amendmentOrderSource"),
                    "amendment_order_confidence": documents[idx].get("amendmentOrderConfidence"),
                    "excerpt": _excerpt_of(sig) if sig else "",
                }
                for idx, sig in amendment_populated
                if sig
            ]
            findings.append(
                Finding(
                    "warning",
                    "SIGNAL_CONFLICT",
                    f"Conflicting amendment values for {signal_id}; ordering not confidently established",
                    {
                        "signalId": signal_id,
                        "candidates": candidates,
                        "distinctValues": distinct_amendment_values,
                    },
                )
            )
            continue

        if _has_value(high_sig):
            high_val = _value_str(high_sig)
            earlier = [
                (idx, sig)
                for idx, sig in per_doc
                if idx < highest_idx and _has_value(sig) and _value_str(sig) != high_val
            ]
            supersedes = [
                {
                    "value": _value_str(sig),
                    "source_id": _doc_meta(documents[idx])["source_id"],
                    "filename": _doc_meta(documents[idx])["filename"],
                    "amendment": _doc_meta(documents[idx])["amendment_order"],
                    "reason_not_selected": "superseded by later amendment",
                }
                for idx, sig in earlier
            ]
            superseded_count += len(supersedes)
            if supersedes:
                findings.append(
                    Finding(
                        "info",
                        "AMENDMENT_SUPERSEDES_BASE",
                        f"Amendment supersedes base for {signal_id}",
                        {"signalId": signal_id, "selectedValue": high_val},
                    )
                )
            merged_signals.append(
                _attach_provenance(high_sig, _provenance_for(high_sig, documents[highest_idx], supersedes=supersedes))
            )
            continue

        populated = _filter_superseded_attachment_candidates(
            [(idx, sig) for idx, sig in per_doc if _has_value(sig)],
            documents,
        )
        if populated:
            chosen_idx, chosen = _pick_highest_authority(populated)
            supersedes = [
                {
                    "value": _value_str(sig),
                    "source_id": _doc_meta(documents[idx])["source_id"],
                    "filename": _doc_meta(documents[idx])["filename"],
                    "amendment": _doc_meta(documents[idx])["amendment_order"],
                    "reason_not_selected": "lower authority retained because higher amendment omitted field",
                }
                for idx, sig in populated[:-1]
            ]
            superseded_count += len(supersedes)
            findings.append(
                Finding(
                    "info",
                    "LOWER_AUTHORITY_USED_DUE_TO_MISSING_HIGHER",
                    f"Used lower-authority value for {signal_id}",
                    {"signalId": signal_id, "selectedValue": _value_str(chosen)},
                )
            )
            merged_signals.append(
                _attach_provenance(chosen, _provenance_for(chosen, documents[chosen_idx], supersedes=supersedes))
            )

    merged_signals.sort(key=lambda item: str(item.get("id") or ""))

    for doc in documents:
        for item in doc.get("findings") or []:
            if isinstance(item, Finding):
                findings.append(item)
            elif isinstance(item, dict) and item.get("code"):
                findings.append(
                    Finding(
                        str(item.get("level") or "info"),
                        str(item.get("code") or ""),
                        str(item.get("message") or ""),
                        item.get("details") if isinstance(item.get("details"), dict) else None,
                    )
                )

    return merged_signals, findings, superseded_count, unresolved_conflicts


def merge_llm_and_docset_signals(
    base_signals: list[dict[str, Any]],
    llm_signals: list[dict[str, Any]],
    docset_signals: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Docset populated values win over full-corpus LLM (AEGIS mergeGeminiAndDocsetSignalsV1)."""
    by_id: dict[str, dict[str, Any]] = {}
    for signal in base_signals:
        signal_id = str(signal.get("id") or "").strip()
        if signal_id:
            by_id[signal_id] = signal
    for signal in llm_signals:
        signal_id = str(signal.get("id") or "").strip()
        if signal_id:
            by_id[signal_id] = signal
    for signal in docset_signals:
        signal_id = str(signal.get("id") or "").strip()
        if not signal_id:
            continue
        value = signal.get("value")
        populated = value is not None and str(value).strip() != ""
        if populated:
            by_id[signal_id] = signal
        elif signal_id not in by_id:
            by_id[signal_id] = signal
    return sorted(by_id.values(), key=lambda item: str(item.get("id") or ""))
