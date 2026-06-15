from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from extraction.signals.authority import infer_authority_tier
from extraction.signals.contract_type import classify_contract_type
from extraction.signals.evaluation_extract import extract_evaluation_signals_v1
from extraction.signals.due_date_select import due_signals_from_candidates, extract_and_select_due_dates
from extraction.signals.incumbent import extract_incumbent_candidates, extract_prior_piid
from extraction.signals.naics_psc import extract_naics_candidates, extract_psc_candidates, pick_primary_code
from extraction.signals.pop_extract import extract_pop_from_text
from extraction.signals.section_l_chooser import choose_section_l_instruction_window, repair_wrapped_tlds
from extraction.signals.set_aside_competition import extract_procurement_from_text, signals_from_classification
from extraction.signals.source_lock import lock_windows, pick_solicitation_source_sha256
from extraction.signals.submission_extract import choose_submission_candidate, derive_submission_method_destination
from extraction.types import CorpusPage, Finding


@dataclass
class SpanLoc:
    source_sha256: str
    page_index: int
    page_ordinal: int
    start: int
    end: int


@dataclass
class Window:
    text: str
    span_hashes: list[str]
    source: str
    loc: SpanLoc
    provenance_prefix: str | None = None
    source_hint: str | None = None


@dataclass
class Candidate:
    value: str
    evidence: dict[str, Any]
    score: float
    page_ordinal: int
    source_sha256: str
    page_index: int
    start: int
    source_hint: str | None = None
    raw_chunk: str | None = None


def _cmp_lex(a: str, b: str) -> int:
    return (a > b) - (a < b)


def _normalize_ws(raw: str) -> str:
    s = raw.replace("\r\n", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def _truncate(text: str, max_chars: int = 280) -> str:
    return text if len(text) <= max_chars else f"{text[: max_chars - 1]}…"


def _build_evidence(
    source: str,
    span_hashes: list[str],
    window_text: str,
    match_index: int,
    provenance_prefix: str | None = None,
) -> dict[str, Any] | None:
    hashes = [h for h in span_hashes if h]
    if not hashes:
        return None
    start = max(0, match_index - 800)
    end = min(len(window_text), match_index + 800)
    raw = re.sub(r"\s+", " ", window_text[start:end]).strip()
    excerpt = _truncate(f"{provenance_prefix} {raw}".strip() if provenance_prefix else raw)
    return {"spanHashes": hashes, "excerpt": excerpt, "source": source}


def _pick_best(candidates: list[Candidate]) -> Candidate | None:
    if not candidates:
        return None
    candidates.sort(
        key=lambda c: (
            -c.score,
            c.page_ordinal,
            c.start,
            c.source_sha256,
            c.page_index,
        )
    )
    return candidates[0]


def _count_matches(text: str, pattern: re.Pattern[str]) -> int:
    return len(pattern.findall(text))


def _extract_from_windows(
    windows: list[Window],
    *,
    strong: list[re.Pattern[str]],
    weak: list[re.Pattern[str]],
    value_re: re.Pattern[str],
    normalize_value,
) -> Candidate | None:
    candidates: list[Candidate] = []
    for window in windows:
        strong_hits = sum(_count_matches(window.text, pat) for pat in strong)
        weak_hits = sum(_count_matches(window.text, pat) for pat in weak)
        score_base = strong_hits * 25 + weak_hits * 10
        for match in value_re.finditer(window.text):
            raw = (match.group(1) if match.lastindex else match.group(0) or "").strip()
            val = normalize_value(raw)
            ev = _build_evidence(window.source, window.span_hashes, window.text, match.start(), window.provenance_prefix)
            if val and ev:
                candidates.append(
                    Candidate(
                        value=str(val),
                        evidence=ev,
                        score=score_base,
                        page_ordinal=window.loc.page_ordinal,
                        source_sha256=window.loc.source_sha256,
                        page_index=window.loc.page_index,
                        start=window.loc.start + match.start(),
                        source_hint=window.source_hint,
                    )
                )
    return _pick_best(candidates)


def _legacy_evidence(run_id: str, ev1: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {
            "sourceId": f"runs/{run_id}/corpus/corpus.v1.json",
            "artifact": "text",
            "locator": f"spanHashes:{','.join(ev1.get('spanHashes') or [])[:3]}",
            "snippet": _truncate(str(ev1.get("excerpt") or ""), 200),
        }
    ]


def _push_signal(
    out: list[dict[str, Any]],
    *,
    signal_id: str,
    value: str,
    ev1: dict[str, Any],
    run_id: str,
    source_hint: str | None,
) -> None:
    out.append(
        {
            "id": signal_id,
            "value": value,
            "confidence": "high",
            "evidence": _legacy_evidence(run_id, ev1),
            "findings": [],
            "evidence_v1": ev1,
            "authority": infer_authority_tier(signal_id=signal_id, excerpt=str(ev1.get("excerpt") or ""), source_hint=source_hint),
        }
    )


def _push_missing(out: list[dict[str, Any]], findings: list[Finding], signal_id: str, code: str, message: str) -> None:
    findings.append(Finding("warn", code, message))
    out.append({"id": signal_id, "value": None, "confidence": "low", "evidence": [], "findings": [{"level": "warn", "code": code, "message": message}]})


def extract_signals_v1(
    *,
    run_id: str,
    pages: list[CorpusPage],
    structure: dict[str, Any] | None,
    sections: dict[str, Any] | None,
    clauses: dict[str, Any] | None,
    section_l_fulltext: dict[str, Any] | None,
    section_m_fulltext: dict[str, Any] | None,
    source_filename: str,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    out_signals: list[dict[str, Any]] = []
    span_loc = {}
    for page_ordinal, page in enumerate(pages):
        for span in page.spans:
            span_loc[span.sha256] = SpanLoc(page.source_sha256, page.page_index, page_ordinal, span.start, span.end)

    def structure_windows(letters: list[str]) -> list[Window]:
        windows: list[Window] = []
        letters_set = {x.upper() for x in letters}
        logical_map = {"SECTION_L": "L", "SECTION_M": "M"}
        for item in (structure or {}).get("items") or []:
            letter: str | None = None
            if item.get("kind") == "SECTION":
                letter = str(item.get("id") or "").upper()
            else:
                letter = logical_map.get(str(item.get("logicalType") or ""))
            if not letter or letter not in letters_set:
                continue
            sha = str((item.get("spanHashes") or [""])[0])
            loc = span_loc.get(sha)
            if not loc:
                continue
            hint = "structureLWindow" if letter == "L" else "structureMWindow" if letter == "M" else None
            windows.append(
                Window(
                    text=_normalize_ws(str(item.get("fullText") or item.get("excerpt") or "")),
                    span_hashes=list(item.get("spanHashes") or []),
                    source="structure",
                    loc=loc,
                    source_hint=hint,
                )
            )
        return windows

    def section_windows(names: list[str]) -> list[Window]:
        windows: list[Window] = []
        bucket = (sections or {}).get("sections") or {}
        for name in names:
            entry = bucket.get(name)
            if not entry:
                continue
            sha = str((entry.get("spanHashes") or [""])[0])
            loc = span_loc.get(sha)
            if not loc:
                continue
            windows.append(
                Window(
                    text=_normalize_ws(str(entry.get("excerpt") or "")),
                    span_hashes=list(entry.get("spanHashes") or []),
                    source="sections",
                    loc=loc,
                )
            )
        return windows

    def fulltext_windows(letters: list[str]) -> list[Window]:
        windows: list[Window] = []
        for letter in letters:
            ft = section_l_fulltext if letter == "L" else section_m_fulltext
            if not ft:
                continue
            text = _normalize_ws(str(ft.get("fullText") or ""))
            hashes = [str(x) for x in (ft.get("evidence_v1") or {}).get("spanHashes") or [] if x]
            loc = span_loc.get(hashes[0]) if hashes else None
            if not text or not hashes or not loc:
                continue
            windows.append(
                Window(
                    text=text,
                    span_hashes=hashes,
                    source=source_filename,
                    loc=loc,
                    provenance_prefix=f"[FULLTEXT:{letter}]",
                    source_hint="sectionLFulltext" if letter == "L" else "sectionMFulltext",
                )
            )
        return windows

    def normalize_id(raw: str) -> str | None:
        v = _normalize_ws(raw).upper().replace(" ", "")
        return v or None

    def normalize_line(raw: str) -> str | None:
        v = re.sub(r"\s+", " ", _normalize_ws(raw)).strip()
        return v or None

    def global_scan_windows() -> list[Window]:
        windows: list[Window] = []
        for page_ordinal, page in enumerate(pages):
            if not page.text.strip() or not page.spans:
                continue
            loc = span_loc.get(page.spans[0].sha256)
            if not loc:
                continue
            windows.append(
                Window(
                    text=_normalize_ws(page.text),
                    span_hashes=[span.sha256 for span in page.spans[:4]],
                    source=source_filename,
                    loc=loc,
                    source_hint="globalScan",
                )
            )
        return windows

    header_windows = structure_windows(["A", "B", "C", "L"]) + section_windows(["entities", "opportunity"])
    scan_windows = header_windows + global_scan_windows()

    source_lock = pick_solicitation_source_sha256(
        pages,
        section_l_fulltext=section_l_fulltext,
        section_m_fulltext=section_m_fulltext,
    )
    if not source_lock.source_sha256:
        findings.append(
            Finding("warn", "SOLICITATION_SOURCE_NOT_FOUND", "Solicitation sourceSha256 not found; submission/POC may be cross-document")
        )

    def lock_w(windows: list[Window]) -> list[Window]:
        return lock_windows(windows, source_lock.source_sha256)

    sol = _extract_from_windows(
        header_windows + structure_windows(["D", "F", "K", "J"]),
        strong=[re.compile(r"\bsolicitation\s*(number|no\.?)\b", re.I), re.compile(r"\b(rfq|rfp|ifb)\s*(number|no\.?)\b", re.I)],
        weak=[re.compile(r"\bsol\b", re.I), re.compile(r"\b(rfq|rfp)\b", re.I)],
        value_re=re.compile(r"\b(?:solicitation\s*(?:number|no\.?)\s*[:#]?\s*|(?:rfq|rfp|ifb)\s*(?:number|no\.?)\s*[:#]?\s*)([A-Z0-9][A-Z0-9\-./]{4,})\b", re.I),
        normalize_value=normalize_id,
    )
    if sol:
        _push_signal(out_signals, signal_id="rfp_solicitation_id_v1", value=sol.value, ev1=sol.evidence, run_id=run_id, source_hint=sol.source_hint)
    else:
        _push_missing(out_signals, findings, "rfp_solicitation_id_v1", "SOLICITATION_ID_NOT_FOUND", "Solicitation id not found")

    agency = _extract_from_windows(
        structure_windows(["A", "L", "B", "C", "D", "F"]) + section_windows(["entities", "opportunity"]),
        strong=[re.compile(r"\bissued\s+by\b", re.I), re.compile(r"\bissuing\s+agency\b", re.I), re.compile(r"\bcontracting\s+office\b", re.I)],
        weak=[re.compile(r"\bagency\b", re.I)],
        value_re=re.compile(r"\b(?:issued\s+by|issuing\s+agency|agency|contracting\s+office)\s*[:\-–—]\s*([^\n]{3,120})", re.I),
        normalize_value=normalize_line,
    )
    if agency:
        _push_signal(out_signals, signal_id="rfp_issuing_agency_v1", value=agency.value, ev1=agency.evidence, run_id=run_id, source_hint=agency.source_hint)
    else:
        _push_missing(out_signals, findings, "rfp_issuing_agency_v1", "AGENCY_NOT_FOUND", "Issuing agency/office not found")

    office = _extract_from_windows(
        structure_windows(["A", "L", "B", "C", "D", "F"]) + section_windows(["entities", "opportunity"]),
        strong=[re.compile(r"\bcontracting\s+office\b", re.I), re.compile(r"\bissuing\s+office\b", re.I)],
        weak=[re.compile(r"\bissued\s+by\b", re.I), re.compile(r"\boffice\b", re.I)],
        value_re=re.compile(r"\b(?:contracting\s+office|issuing\s+office)\s*[:\-–—]\s*([^\n(]{3,80})", re.I),
        normalize_value=lambda raw: normalize_line(raw) if normalize_line(raw) and len(normalize_line(raw) or "") > 1 else None,
    )
    if office:
        _push_signal(out_signals, signal_id="rfp_issuing_office_v1", value=office.value, ev1=office.evidence, run_id=run_id, source_hint=office.source_hint)
    else:
        _push_missing(out_signals, findings, "rfp_issuing_office_v1", "OFFICE_NOT_FOUND", "Issuing office not found")

    naics_candidates: list = []
    for window in lock_w(scan_windows):
        naics_candidates.extend(extract_naics_candidates(window.text))
    naics_winner, naics_conflicts = pick_primary_code(naics_candidates)
    if naics_winner:
        ev = _build_evidence(source_filename, [], naics_winner.excerpt, 0, None) or {"excerpt": naics_winner.excerpt, "source": source_filename}
        _push_signal(out_signals, signal_id="rfp_primary_naics_v1", value=naics_winner.value, ev1=ev, run_id=run_id, source_hint="form_field")
    elif naics_conflicts:
        findings.append(
            Finding(
                "warning",
                "NAICS_CONFLICT",
                "Multiple authoritative NAICS codes conflict",
                {"candidates": [item.__dict__ for item in naics_conflicts]},
            )
        )
        _push_missing(out_signals, findings, "rfp_primary_naics_v1", "NAICS_NOT_FOUND", "Primary NAICS code not found")
    else:
        _push_missing(out_signals, findings, "rfp_primary_naics_v1", "NAICS_NOT_FOUND", "Primary NAICS code not found")

    psc_candidates: list = []
    for window in lock_w(scan_windows):
        psc_candidates.extend(extract_psc_candidates(window.text))
    psc_winner, psc_conflicts = pick_primary_code(psc_candidates)
    if psc_winner:
        ev = _build_evidence(source_filename, [], psc_winner.excerpt, 0, None) or {"excerpt": psc_winner.excerpt, "source": source_filename}
        _push_signal(out_signals, signal_id="rfp_primary_psc_v1", value=psc_winner.value, ev1=ev, run_id=run_id, source_hint="form_field")
    elif psc_conflicts:
        findings.append(
            Finding(
                "warning",
                "PSC_CONFLICT",
                "Multiple authoritative PSC codes conflict",
                {"candidates": [item.__dict__ for item in psc_conflicts]},
            )
        )
        _push_missing(out_signals, findings, "rfp_primary_psc_v1", "PSC_NOT_FOUND", "Primary PSC not found")
    else:
        _push_missing(out_signals, findings, "rfp_primary_psc_v1", "PSC_NOT_FOUND", "Primary PSC not found")

    contract_type_found = False
    for window in lock_w(scan_windows):
        for match in re.finditer(r"\bcontract\s+type\s*[:\-–—]\s*([^\n;]{2,80})", window.text, re.I):
            classified = classify_contract_type(match.group(0))
            if not classified:
                continue
            ev = _build_evidence(window.source, window.span_hashes, window.text, match.start(), window.provenance_prefix)
            if not ev:
                continue
            _push_signal(
                out_signals,
                signal_id="rfp_contract_type_v1",
                value=classified.value,
                ev1=ev,
                run_id=run_id,
                source_hint=window.source_hint,
            )
            contract_type_found = True
            if classified.is_hybrid:
                findings.append(Finding("info", "CONTRACT_TYPE_HYBRID", "Hybrid contract type detected; review before filter application"))
            break
        if contract_type_found:
            break
    if not contract_type_found:
        _push_missing(out_signals, findings, "rfp_contract_type_v1", "CONTRACT_TYPE_NOT_FOUND", "Contract type not found")

    procurement_found = False
    for window in lock_w(scan_windows):
        for classified in extract_procurement_from_text(window.text):
            ev = _build_evidence(window.source, window.span_hashes, window.text, 0, window.provenance_prefix)
            if not ev:
                continue
            for signal in signals_from_classification(classified, evidence=ev, run_id=run_id, source_hint=window.source_hint):
                _push_signal(
                    out_signals,
                    signal_id=str(signal["id"]),
                    value=str(signal["value"]),
                    ev1=signal.get("evidence_v1") or ev,
                    run_id=run_id,
                    source_hint=window.source_hint,
                )
            procurement_found = True
            break
        if procurement_found:
            break
    if not procurement_found:
        _push_missing(out_signals, findings, "rfp_competition_type_v1", "COMPETITION_TYPE_NOT_FOUND", "Competition type not found")

    pop = _extract_from_windows(
        structure_windows(["B", "C", "D", "F", "L"]) + section_windows(["opportunity"]),
        strong=[re.compile(r"\bplace\s+of\s+performance\b", re.I), re.compile(r"\bwork\s+will\s+be\s+performed\b", re.I)],
        weak=[re.compile(r"\blocation\b", re.I)],
        value_re=re.compile(r"\b(?:place\s+of\s+performance|work\s+will\s+be\s+performed\s+at|location)\b[^:\n]{0,20}[:\-–—]?\s*([^\n]{4,160})", re.I),
        normalize_value=normalize_line,
    )
    if pop:
        _push_signal(out_signals, signal_id="rfp_place_of_performance_v1", value=pop.value, ev1=pop.evidence, run_id=run_id, source_hint=pop.source_hint)
    else:
        _push_missing(out_signals, findings, "rfp_place_of_performance_v1", "PLACE_OF_PERFORMANCE_NOT_FOUND", "Place of performance not found")

    pop_start = _extract_from_windows(
        header_windows,
        strong=[re.compile(r"\bperiod\s+of\s+performance\b", re.I), re.compile(r"\bpop\s+start\b", re.I)],
        weak=[],
        value_re=re.compile(r"\b(?:pop\s+start|period\s+of\s+performance\s+start|start\s+date)\s*[:\-–—]?\s*([^\n;]{4,40})", re.I),
        normalize_value=normalize_line,
    )
    if pop_start:
        _push_signal(out_signals, signal_id="rfp_pop_start_v1", value=pop_start.value, ev1=pop_start.evidence, run_id=run_id, source_hint=pop_start.source_hint)

    pop_end = _extract_from_windows(
        header_windows,
        strong=[re.compile(r"\bperiod\s+of\s+performance\b", re.I), re.compile(r"\bpop\s+end\b", re.I)],
        weak=[],
        value_re=re.compile(r"\b(?:pop\s+end|period\s+of\s+performance\s+end|end\s+date)\s*[:\-–—]?\s*([^\n;]{4,40})", re.I),
        normalize_value=normalize_line,
    )
    if pop_end:
        _push_signal(out_signals, signal_id="rfp_pop_end_v1", value=pop_end.value, ev1=pop_end.evidence, run_id=run_id, source_hint=pop_end.source_hint)

    pop_text = "\n".join(window.text for window in lock_w(scan_windows))
    pop_detail = extract_pop_from_text(pop_text)
    if pop_detail.composite:
        ev = {"excerpt": pop_detail.composite[:280], "source": source_filename}
        _push_signal(out_signals, signal_id="rfp_period_of_performance_v1", value=pop_detail.composite, ev1=ev, run_id=run_id, source_hint="form_field")
    if pop_detail.years and not (pop_detail.start and pop_detail.end):
        ev = {"excerpt": pop_detail.years, "source": source_filename}
        _push_signal(out_signals, signal_id="rfp_pop_years_v1", value=pop_detail.years, ev1=ev, run_id=run_id, source_hint="form_field")

    page_limits = _extract_from_windows(
        fulltext_windows(["L"]) + structure_windows(["L"]) + scan_windows,
        strong=[re.compile(r"\bpage\s+limit\b", re.I)],
        weak=[],
        value_re=re.compile(r"\bpage\s+limit\s*:\s*([^\n;]{2,80})", re.I),
        normalize_value=normalize_line,
    )
    if page_limits:
        _push_signal(out_signals, signal_id="rfp_page_limits_v1", value=page_limits.value, ev1=page_limits.evidence, run_id=run_id, source_hint=page_limits.source_hint)

    # Due date candidate model (proposal, questions, datetime/timezone)
    proposal_due, questions_due, due_candidates, due_findings = extract_and_select_due_dates(
        pages=pages,
        structure=structure,
        source_filename=source_filename,
        span_loc=span_loc,
        section_l_fulltext=section_l_fulltext,
        section_m_fulltext=section_m_fulltext,
    )
    findings.extend(due_findings)
    due_signal_payloads = due_signals_from_candidates(
        proposal_due,
        questions_due,
        due_candidates,
        run_id=run_id,
        source_filename=source_filename,
    )
    emitted_due_ids: set[str] = set()
    for payload in due_signal_payloads:
        signal_id = str(payload.get("id") or "")
        ev1 = payload.get("evidence_v1") or {}
        hint = None
        if signal_id.startswith("rfp_due") and proposal_due:
            hint = proposal_due.source_authority
        elif signal_id == "rfp_questions_due_v1" and questions_due:
            hint = questions_due.source_authority
        payload["authority"] = infer_authority_tier(
            signal_id=signal_id,
            excerpt=str(ev1.get("excerpt") or ""),
            source_hint=hint,
        )
        out_signals.append(payload)
        emitted_due_ids.add(signal_id)
    if "rfp_due_date_v1" not in emitted_due_ids:
        _push_missing(out_signals, findings, "rfp_due_date_v1", "DUE_DATE_NOT_FOUND", "Due date not found")

    # Section L submission via full chooser cascade
    l_full = str((section_l_fulltext or {}).get("fullText") or "")
    chosen = choose_section_l_instruction_window(l_full) if l_full else None
    submission = choose_submission_candidate(
        section_l_fulltext=l_full,
        chosen_l_window=chosen,
        l_locator_windows=[],
        fulltext_windows_l=fulltext_windows(["L"]),
        structure_windows_l=structure_windows(["L"]),
        fulltext_windows_m=fulltext_windows(["M"]),
        structure_windows_m=structure_windows(["M"]),
        section_windows=section_windows(["entities", "opportunity", "risks"]),
        locked_windows_fn=lock_w,
        pages=pages,
        source_sha256=source_lock.source_sha256,
        source_filename=source_filename,
        span_loc=span_loc,
    )
    if submission:
        _push_signal(
            out_signals,
            signal_id="rfp_submission_instructions_v1",
            value=submission.value[:1200],
            ev1=submission.evidence,
            run_id=run_id,
            source_hint=submission.source_hint,
        )
        method = submission.method
        destination = submission.destination
        if not method or not destination:
            derived_method, derived_dest = derive_submission_method_destination(submission.value)
            method = method or derived_method
            destination = destination or derived_dest
        if method:
            _push_signal(out_signals, signal_id="rfp_submission_method_v1", value=method, ev1=submission.evidence, run_id=run_id, source_hint=submission.source_hint)
        if destination:
            _push_signal(out_signals, signal_id="rfp_submission_destination_v1", value=destination, ev1=submission.evidence, run_id=run_id, source_hint=submission.source_hint)
    else:
        findings.append(Finding("warn", "SUBMISSION_INSTRUCTIONS_NOT_FOUND", "Submission instructions not found"))

    eval_signals, eval_findings = extract_evaluation_signals_v1(
        windows=lock_w(fulltext_windows(["M"]) + structure_windows(["M"])),
        structure=structure,
        source_filename=source_filename,
        run_id=run_id,
    )
    out_signals.extend(eval_signals)
    findings.extend(eval_findings)

    ko = _extract_from_windows(
        fulltext_windows(["L"]) + structure_windows(["L", "M"]),
        strong=[re.compile(r"\bcontracting\s+officer\b", re.I)],
        weak=[re.compile(r"\bofficer\b", re.I)],
        value_re=re.compile(r"\b(?:contracting\s+officer|ko)\s*[:\-–—]\s*([A-Za-z][A-Za-z .,'-]{2,80})", re.I),
        normalize_value=lambda raw: normalize_line(raw) if normalize_line(raw) and "tbd" not in (normalize_line(raw) or "").lower() else None,
    )
    if ko:
        _push_signal(out_signals, signal_id="rfp_ko_name_v1", value=ko.value, ev1=ko.evidence, run_id=run_id, source_hint=ko.source_hint)

    cs = _extract_from_windows(
        fulltext_windows(["L"]) + structure_windows(["L", "M"]),
        strong=[re.compile(r"\bcontract\s+specialist\b", re.I)],
        weak=[re.compile(r"\bspecialist\b", re.I)],
        value_re=re.compile(r"\bcontract\s+specialist\s*[:\-–—]\s*([A-Za-z][A-Za-z .,'-]{2,80})", re.I),
        normalize_value=lambda raw: normalize_line(raw) if normalize_line(raw) and "tbd" not in (normalize_line(raw) or "").lower() else None,
    )
    if cs:
        _push_signal(out_signals, signal_id="rfp_contract_specialist_name_v1", value=cs.value, ev1=cs.evidence, run_id=run_id, source_hint=cs.source_hint)

    poc = _extract_from_windows(
        fulltext_windows(["L"]) + structure_windows(["L"]),
        strong=[re.compile(r"\bpoint\s+of\s+contact\b", re.I), re.compile(r"\bpoc\b", re.I)],
        weak=[re.compile(r"\bemail\b", re.I)],
        value_re=re.compile(r"\b((?:[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})|(?:phone|telephone)[^\n]{0,20}[\d().\-+\s]{7,20})", re.I),
        normalize_value=normalize_line,
    )
    if poc:
        _push_signal(out_signals, signal_id="rfp_primary_poc_v1", value=poc.value, ev1=poc.evidence, run_id=run_id, source_hint=poc.source_hint)

    incumbent_candidates = extract_incumbent_candidates("\n".join(window.text for window in lock_w(scan_windows)))
    explicit = [item for item in incumbent_candidates if item.role == "explicit_incumbent"]
    if explicit:
        best = explicit[0]
        ev = {"excerpt": best.excerpt, "source": source_filename}
        _push_signal(out_signals, signal_id="rfp_incumbent_data_v1", value=best.value, ev1=ev, run_id=run_id, source_hint="form_field")
    elif incumbent_candidates:
        findings.append(
            Finding(
                "info",
                "INCUMBENT_AMBIGUOUS",
                "Vendor mentions found but no explicit incumbent designation",
                {"candidates": [item.__dict__ for item in incumbent_candidates[:3]]},
            )
        )

    prior_piid_value = extract_prior_piid("\n".join(window.text for window in lock_w(scan_windows)))
    if prior_piid_value:
        ev = {"excerpt": prior_piid_value, "source": source_filename}
        _push_signal(out_signals, signal_id="rfp_prior_contract_piid_v1", value=prior_piid_value, ev1=ev, run_id=run_id, source_hint="form_field")

    # Office AAC from solicitation id
    sol_signal = next((s for s in out_signals if s.get("id") == "rfp_solicitation_id_v1" and s.get("value")), None)
    if sol_signal and isinstance(sol_signal.get("value"), str) and len(sol_signal["value"]) >= 6:
        aac = sol_signal["value"][:6]
        _push_signal(out_signals, signal_id="rfp_office_aac_v1", value=aac, ev1=sol_signal["evidence_v1"], run_id=run_id, source_hint=sol.source_hint if sol else None)

    deduped: dict[str, dict[str, Any]] = {}
    for signal in out_signals:
        signal_id = str(signal.get("id") or "")
        if signal_id:
            deduped[signal_id] = signal
    out_signals = sorted(deduped.values(), key=lambda item: str(item.get("id") or ""))
    return out_signals, findings
