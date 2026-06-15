from __future__ import annotations

from typing import Any

from extraction.postprocess.amendment_status import apply_amendment_status_signals
from extraction.postprocess.derived_signals import apply_derived_signals
from extraction.postprocess.evaluation_canonicalize import canonicalize_evaluation_signals
from extraction.postprocess.merge import ensure_solicitation_number_alias, fix_mislabeled_competition_type, sort_signals_by_id
from extraction.postprocess.odc_serialize import apply_odc_serialization
from extraction.postprocess.piid_backstop import inject_prior_contract_piid_backstop
from extraction.postprocess.pop_compute import apply_computed_period_of_performance
from extraction.postprocess.submission_canonicalize import canonicalize_submission_signals
from extraction.types import Finding


def run_postprocess(
    signals: list[dict[str, Any]],
    *,
    corpus_text: str = "",
) -> tuple[list[dict[str, Any]], list[Finding]]:
    findings: list[Finding] = []
    current = sort_signals_by_id(list(signals))

    current = canonicalize_submission_signals(current)
    current, eval_findings = canonicalize_evaluation_signals(current)
    findings.extend(eval_findings)
    current = apply_derived_signals(current)
    current = apply_amendment_status_signals(current, corpus_text)

    current, competition_findings = fix_mislabeled_competition_type(current)
    findings.extend(competition_findings)

    current = apply_odc_serialization(current)
    current, piid_findings = inject_prior_contract_piid_backstop(current, corpus_text)
    findings.extend(piid_findings)
    current = ensure_solicitation_number_alias(current)
    current, pop_findings = apply_computed_period_of_performance(current)
    findings.extend(pop_findings)

    return current, findings
