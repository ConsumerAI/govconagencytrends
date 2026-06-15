from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from extraction.artifacts.structure_relations import (
    is_superseded_structure_item,
    source_hint_for_structure_item,
    structure_items_for_extraction,
)
from extraction.signals.authority import infer_authority_tier
from extraction.signals.evaluation_authority import (
    evaluation_authority_score,
    is_section_l_submission_context,
)
from extraction.types import Finding


@dataclass
class EvaluationCandidate:
    signal_id: str
    value: str
    normalized_method: str | None = None
    excerpt: str = ""
    source_hint: str | None = None
    confidence: str = "medium"
    score: float = 0.0
    amendment_number: str | None = None
    logical_structure_type: str | None = None
    span_hashes: list[str] = field(default_factory=list)
    page_index: int = 0
    source_document: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_signal(self, *, run_id: str) -> dict[str, Any]:
        ev1 = {
            "spanHashes": self.span_hashes,
            "excerpt": self.excerpt[:400],
            "source": self.source_document,
            "evaluationMetadata": self.metadata,
            "normalizedMethod": self.normalized_method,
            "pageIndex": self.page_index,
        }
        if self.amendment_number:
            ev1["amendmentNumber"] = self.amendment_number
        return {
            "id": self.signal_id,
            "value": self.value,
            "confidence": self.confidence,
            "evidence": [
                {
                    "sourceId": self.source_document or f"runs/{run_id}/corpus/corpus.v1.json",
                    "artifact": "text",
                    "locator": f"spanHashes:{','.join(self.span_hashes[:3])}",
                    "snippet": self.excerpt[:200],
                }
            ],
            "evidence_v1": ev1,
            "findings": [],
            "authority": infer_authority_tier(
                signal_id=self.signal_id,
                excerpt=self.excerpt,
                source_hint=self.source_hint,
            ),
        }


LPTA_PATTERNS = [
    re.compile(r"\blowest\s+price\s+technically\s+acceptable\b", re.I),
    re.compile(r"\bLPTA\b", re.I),
    re.compile(r"\bprice\s+only\b", re.I),
]
TRADEOFF_PATTERNS = [
    re.compile(r"\bbest\s+value\s+trade[\s-]?off\b", re.I),
    re.compile(r"\btrade[\s-]?off\b", re.I),
    re.compile(r"\bcomparative\s+evaluation\b", re.I),
]
BEST_VALUE_PATTERNS = [
    re.compile(r"\bbest\s+value\b", re.I),
    re.compile(r"\bbasis\s+for\s+award\b", re.I),
]
PASS_FAIL_PATTERNS = [
    re.compile(r"\bpass[\s/]fail\b", re.I),
    re.compile(r"\bacceptable[\s/]unacceptable\b", re.I),
    re.compile(r"\bgo[\s/]no[\s-]?go\b", re.I),
]
RELATIVE_IMPORTANCE_RE = re.compile(
    r"(?:relative\s+importance|significantly\s+more\s+important|approximately\s+equal|"
    r"when\s+combined\s+are\s+approximately\s+equal|of\s+equal\s+importance)[^\n]{0,220}",
    re.I,
)
TECH_FACTORS_RE = re.compile(
    r"(?:technical\s+(?:factor|approach|capability)|factor\s*\d+\s+(?:[-–—:\s]+)?technical)[^\n]{0,220}",
    re.I,
)
MGMT_FACTORS_RE = re.compile(
    r"(?:management\s+(?:factor|approach)|factor\s*2\s*[-–—:]\s*management)[^\n]{0,220}",
    re.I,
)
PAST_PERF_RE = re.compile(
    r"(?:past\s+performance\s+(?:is\s+)?(?:(?:not\s+)?evaluated|advisory|informational|a\s+factor|considered)|"
    r"past\s+performance\s+is\s+advisory\s+and\s+not\s+evaluated)[^\n]{0,120}",
    re.I,
)
PRICE_ROLE_RE = re.compile(
    r"(?:price\s+(?:is\s+)?(?:not\s+)?(?:evaluated|a\s+factor|of\s+(?:equal|lesser|greater)\s+importance)|"
    r"cost\s+(?:is\s+)?(?:not\s+)?(?:evaluated|a\s+factor))[^\n]{0,180}",
    re.I,
)
RATING_STYLE_RE = re.compile(
    r"\b(?:adjectival|color\s+rating|numerical\s+rating|overall\s+rating)\b[^\n]{0,120}",
    re.I,
)
SUBFACTOR_RE = re.compile(r"\bsubfactor\b[^\n]{0,120}", re.I)
BOILERPLATE_RE = re.compile(r"\b(?:table\s+of\s+contents|see\s+section\s+l|instructions?\s+to\s+offerors)\b", re.I)


def _capture_two_lines(text: str, start: int) -> str:
    line_end = text.find("\n", start)
    if line_end == -1:
        return text[start : min(len(text), start + 220)].strip()
    line1 = text[start:line_end].strip()
    next_start = line_end + 1
    next_end = text.find("\n", next_start)
    line2 = text[next_start : next_end if next_end != -1 else next_start + 180].strip()
    combined = f"{line1} {line2}".strip()
    return combined[:420]


def classify_eval_method(excerpt: str) -> tuple[str | None, bool, bool]:
    has_lpta = any(pat.search(excerpt) for pat in LPTA_PATTERNS)
    has_tradeoff = any(pat.search(excerpt) for pat in TRADEOFF_PATTERNS)
    has_best_value = any(pat.search(excerpt) for pat in BEST_VALUE_PATTERNS)
    has_pass_fail = any(pat.search(excerpt) for pat in PASS_FAIL_PATTERNS)

    if has_lpta and has_tradeoff:
        return None, True, False
    if has_lpta:
        return "LPTA", False, False
    if has_tradeoff:
        return "Tradeoff", False, False
    if has_pass_fail and not has_best_value:
        return "Pass/Fail", False, False
    if has_best_value and not has_lpta:
        if has_pass_fail:
            return "Best Value (Pass/Fail factors)", False, True
        return "Best Value", False, True
    if re.search(r"\bevaluation\s+factors?\b", excerpt, re.I):
        return "Evaluation Factors", False, True
    return None, False, True


def _build_metadata(excerpt: str) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    rel = RELATIVE_IMPORTANCE_RE.search(excerpt)
    if rel:
        meta["relativeImportance"] = rel.group(0).strip()
    tech = TECH_FACTORS_RE.search(excerpt)
    if tech:
        meta["technicalFactors"] = tech.group(0).strip()
    mgmt = MGMT_FACTORS_RE.search(excerpt)
    if mgmt:
        meta["managementFactors"] = mgmt.group(0).strip()
    pp = PAST_PERF_RE.search(excerpt)
    if pp:
        meta["pastPerformanceRole"] = pp.group(0).strip()
    price = PRICE_ROLE_RE.search(excerpt)
    if price:
        meta["priceCostRole"] = price.group(0).strip()
    rating = RATING_STYLE_RE.search(excerpt)
    if rating:
        meta["ratingStyle"] = rating.group(0).strip()
    if SUBFACTOR_RE.search(excerpt):
        meta["hasSubfactors"] = True
    pf = PASS_FAIL_PATTERNS[0].search(excerpt) or PASS_FAIL_PATTERNS[1].search(excerpt)
    if pf:
        meta["passFailLanguage"] = pf.group(0).strip()
    return meta


def _score_excerpt(excerpt: str, source_hint: str | None, amendment_number: str | None, logical_type: str | None) -> float:
    score = evaluation_authority_score(
        source_hint=source_hint,
        excerpt=excerpt,
        amendment_number=amendment_number,
        logical_structure_type=logical_type,
    )
    method, ambiguous, vague = classify_eval_method(excerpt)
    if method == "LPTA":
        score += 100
    elif method == "Tradeoff":
        score += 90
    elif method == "Best Value":
        score += 70
    elif method == "Evaluation Factors":
        score += 50
    if ambiguous:
        score -= 40
    if vague:
        score -= 10
    if BOILERPLATE_RE.search(excerpt):
        score -= 30
    if re.search(r"\bevaluation\s+factors?\b", excerpt, re.I):
        score += 20
    return score


def _candidate_from_excerpt(
    *,
    excerpt: str,
    source_hint: str | None,
    source_document: str,
    span_hashes: list[str],
    page_index: int,
    amendment_number: str | None,
    logical_structure_type: str | None,
    metadata_source: str | None = None,
) -> EvaluationCandidate | None:
    if is_section_l_submission_context(source_hint, excerpt):
        return None
    if not excerpt.strip():
        return None

    method, ambiguous, vague = classify_eval_method(excerpt)
    metadata = _build_metadata(metadata_source or excerpt)
    score = _score_excerpt(excerpt, source_hint, amendment_number, logical_structure_type)
    if score < 0:
        return None

    if ambiguous:
        confidence = "low"
        value = "Review Required: conflicting LPTA and tradeoff language"
        method = None
    elif vague and not method:
        confidence = "low"
        value = excerpt[:280]
    elif method:
        confidence = "high" if score >= 400 and method in {"LPTA", "Tradeoff"} else "medium"
        value = method if method in {"LPTA", "Tradeoff", "Pass/Fail"} else excerpt[:280]
    else:
        confidence = "low"
        value = excerpt[:280]

    return EvaluationCandidate(
        signal_id="rfp_evaluation_criteria_v1",
        value=value,
        normalized_method=method,
        excerpt=excerpt,
        source_hint=source_hint,
        confidence=confidence,
        score=score,
        amendment_number=amendment_number,
        logical_structure_type=logical_structure_type,
        span_hashes=span_hashes,
        page_index=page_index,
        source_document=source_document,
        metadata=metadata,
    )


def extract_evaluation_candidates(
    *,
    windows: list[Any],
    structure: dict[str, Any] | None,
    source_filename: str,
    amendment_number: str | None = None,
) -> list[EvaluationCandidate]:
    candidates: list[EvaluationCandidate] = []

    phrase_res = [
        re.compile(
            r"\b(?:best\s+value|lowest\s+price\s+technically\s+acceptable|LPTA|evaluation\s+factors?|"
            r"basis\s+for\s+award|trade[\s-]?off|pass[\s/]fail|evaluation\s+factor)\b",
            re.I,
        ),
    ]
    for window in windows:
        text = str(getattr(window, "text", "") or "")
        source_hint = getattr(window, "source_hint", None)
        span_hashes = list(getattr(window, "span_hashes", []) or [])
        loc = getattr(window, "loc", None)
        page_index = int(getattr(loc, "page_index", 0) if loc else 0)
        for pattern in phrase_res:
            for match in pattern.finditer(text):
                excerpt = _capture_two_lines(text, max(0, match.start() - 40))
                cand = _candidate_from_excerpt(
                    excerpt=excerpt,
                    source_hint=source_hint,
                    source_document=source_filename,
                    span_hashes=span_hashes,
                    page_index=page_index,
                    amendment_number=amendment_number,
                    logical_structure_type="SECTION_M" if source_hint in {"sectionMFulltext", "structureMWindow"} else None,
                )
                if cand:
                    candidates.append(cand)

    for item in structure_items_for_extraction(structure):
        if is_superseded_structure_item(item, structure):
            continue
        logical = str(item.get("logicalType") or "")
        if logical not in {"SECTION_M", "QA", "ATTACHMENT", "EXHIBIT", "SF30", "SF30_CONTINUATION", "REVISED_PWS"}:
            continue
        text = str(item.get("fullText") or item.get("excerpt") or "")
        if not text.strip():
            continue
        source_hint = source_hint_for_structure_item(item)
        if logical == "SECTION_M":
            source_hint = "sectionMFulltext"
            if not any(pat.search(text) for pat in phrase_res):
                if re.search(r"\b(?:evaluation\s+factor|pass[\s/]fail|past\s+performance)\b", text, re.I):
                    excerpt = _capture_two_lines(text, 0)
                    cand = _candidate_from_excerpt(
                        excerpt=excerpt,
                        source_hint=source_hint,
                        source_document=source_filename,
                        span_hashes=list(item.get("spanHashes") or []),
                        page_index=int(item.get("pageIndex") or 0),
                        amendment_number=str(item.get("amendmentNumber") or amendment_number or "") or None,
                        logical_structure_type=logical,
                        metadata_source=text,
                    )
                    if cand:
                        candidates.append(cand)
        for match in phrase_res[0].finditer(text):
            excerpt = _capture_two_lines(text, max(0, match.start() - 40))
            if logical == "QA":
                excerpt = f"[QA clarification] {excerpt}"
            cand = _candidate_from_excerpt(
                excerpt=excerpt,
                source_hint=source_hint,
                source_document=source_filename,
                span_hashes=list(item.get("spanHashes") or []),
                page_index=int(item.get("pageIndex") or 0),
                amendment_number=str(item.get("amendmentNumber") or amendment_number or "") or None,
                logical_structure_type=logical,
                metadata_source=text if logical == "SECTION_M" else None,
            )
            if cand:
                candidates.append(cand)

        rel = RELATIVE_IMPORTANCE_RE.search(text)
        if rel and logical in {"SECTION_M", "SF30", "SF30_CONTINUATION"}:
            excerpt = rel.group(0).strip()
            candidates.append(
                EvaluationCandidate(
                    signal_id="rfp_eval_weights_v1",
                    value=excerpt[:280],
                    excerpt=excerpt,
                    source_hint=source_hint,
                    confidence="medium",
                    score=_score_excerpt(excerpt, source_hint, amendment_number, logical) + 15,
                    amendment_number=str(item.get("amendmentNumber") or amendment_number or "") or None,
                    logical_structure_type=logical,
                    span_hashes=list(item.get("spanHashes") or []),
                    page_index=int(item.get("pageIndex") or 0),
                    source_document=source_filename,
                    metadata=_build_metadata(excerpt),
                )
            )

        tech = TECH_FACTORS_RE.search(text)
        if tech and logical in {"SECTION_M", "SF30", "SF30_CONTINUATION", "ATTACHMENT"}:
            excerpt = tech.group(0).strip()
            candidates.append(
                EvaluationCandidate(
                    signal_id="rfp_tech_factors_v1",
                    value=excerpt[:280],
                    excerpt=excerpt,
                    source_hint=source_hint,
                    confidence="medium",
                    score=_score_excerpt(excerpt, source_hint, amendment_number, logical),
                    amendment_number=str(item.get("amendmentNumber") or amendment_number or "") or None,
                    logical_structure_type=logical,
                    span_hashes=list(item.get("spanHashes") or []),
                    page_index=int(item.get("pageIndex") or 0),
                    source_document=source_filename,
                    metadata=_build_metadata(excerpt),
                )
            )

    section_m_context = " ".join(
        str(getattr(window, "text", "") or "")
        for window in windows
        if getattr(window, "source_hint", None) in {"sectionMFulltext", "structureMWindow"}
    )
    for item in structure_items_for_extraction(structure):
        if str(item.get("logicalType") or "") == "SECTION_M":
            section_m_context = f"{section_m_context} {item.get('fullText') or item.get('excerpt') or ''}".strip()

    if section_m_context.strip():
        section_meta = _build_metadata(section_m_context)
        for cand in candidates:
            if cand.logical_structure_type == "SECTION_M" or cand.source_hint in {"sectionMFulltext", "structureMWindow"}:
                cand.metadata = {**section_meta, **cand.metadata}

    return candidates


def select_evaluation_winners(candidates: list[EvaluationCandidate]) -> tuple[dict[str, EvaluationCandidate], list[EvaluationCandidate]]:
    by_id: dict[str, list[EvaluationCandidate]] = {}
    for cand in candidates:
        by_id.setdefault(cand.signal_id, []).append(cand)

    winners: dict[str, EvaluationCandidate] = {}
    alternates: list[EvaluationCandidate] = []
    for signal_id, pool in by_id.items():
        pool.sort(key=lambda c: (-c.score, -(int(c.amendment_number) if c.amendment_number and c.amendment_number.isdigit() else -1), c.page_index))
        if not pool:
            continue
        winners[signal_id] = pool[0]
        alternates.extend(pool[1:])
    return winners, alternates


def extract_evaluation_signals_v1(
    *,
    windows: list[Any],
    structure: dict[str, Any] | None,
    source_filename: str,
    run_id: str,
    amendment_number: str | None = None,
) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    candidates = extract_evaluation_candidates(
        windows=windows,
        structure=structure,
        source_filename=source_filename,
        amendment_number=amendment_number,
    )
    if not candidates:
        findings.append(Finding("warn", "EVALUATION_CRITERIA_NOT_FOUND", "Evaluation criteria not found"))
        return [], findings

    winners, alternates = select_evaluation_winners(candidates)
    criteria = winners.get("rfp_evaluation_criteria_v1")
    if not criteria and not winners:
        findings.append(Finding("warn", "EVALUATION_CRITERIA_NOT_FOUND", "Evaluation criteria not found"))
        return [], findings

    if not criteria:
        findings.append(Finding("warn", "EVALUATION_CRITERIA_NOT_FOUND", "Evaluation criteria not found"))

    if criteria and criteria.normalized_method is None and "Review Required" in criteria.value:
        findings.append(
            Finding(
                "warn",
                "EVALUATION_REVIEW_REQUIRED",
                "Evaluation method ambiguous (LPTA vs tradeoff conflict)",
                {"excerpt": criteria.excerpt[:200]},
            )
        )

    signals: list[dict[str, Any]] = []
    if criteria:
        criteria_signal = criteria.to_signal(run_id=run_id)
        if alternates:
            criteria_signal["evidence_v1"]["evaluationAlternates"] = [
                {"signalId": alt.signal_id, "value": alt.value, "excerpt": alt.excerpt[:160], "sourceHint": alt.source_hint}
                for alt in alternates[:8]
            ]
        signals.append(criteria_signal)

        method_value = criteria.normalized_method or criteria.value
        if criteria.normalized_method and criteria.confidence != "low":
            method_signal = EvaluationCandidate(
                signal_id="rfp_eval_method_v1",
                value=method_value,
                normalized_method=criteria.normalized_method,
                excerpt=criteria.excerpt,
                source_hint=criteria.source_hint,
                confidence=criteria.confidence,
                score=criteria.score,
                amendment_number=criteria.amendment_number,
                logical_structure_type=criteria.logical_structure_type,
                span_hashes=criteria.span_hashes,
                page_index=criteria.page_index,
                source_document=criteria.source_document,
                metadata=criteria.metadata,
            ).to_signal(run_id=run_id)
            signals.append(method_signal)
        elif "Review Required" in criteria.value:
            method_signal = criteria.to_signal(run_id=run_id)
            method_signal["id"] = "rfp_eval_method_v1"
            method_signal["value"] = criteria.value
            method_signal["confidence"] = "low"
            method_signal["findings"] = [{"level": "warn", "code": "EVALUATION_REVIEW_REQUIRED", "message": criteria.value}]
            signals.append(method_signal)

    weights = winners.get("rfp_eval_weights_v1")
    if weights:
        signals.append(weights.to_signal(run_id=run_id))

    tech = winners.get("rfp_tech_factors_v1")
    if tech:
        signals.append(tech.to_signal(run_id=run_id))

    if signals or winners:
        findings.append(Finding("info", "EVALUATION_EXTRACTED", "Evaluation cluster extracted"))
    return signals, findings
