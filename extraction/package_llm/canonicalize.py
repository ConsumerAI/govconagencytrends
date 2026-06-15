from __future__ import annotations

from typing import Any


def _join_parts(parts: list[Any], *, sep: str = " — ") -> str | None:
    cleaned = [str(part).strip() for part in parts if part not in (None, "")]
    return sep.join(cleaned) if cleaned else None


def canonical_scalar_for_signal(signal_id: str, value: Any) -> tuple[Any, dict[str, Any] | None]:
    """Map structured model values to scalar canonical values for Scope Review."""
    if signal_id == "rfp_solicitation_id_v1":
        from extraction.package_llm.solicitation_id import canonical_base_solicitation_id

        return canonical_base_solicitation_id(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value, None
    if not isinstance(value, dict):
        return value, None

    structured = dict(value)
    if signal_id == "rfp_title_v1":
        return structured.get("title") or structured.get("name") or structured.get("label"), structured
    if signal_id == "rfp_requirement_name_v1":
        return structured.get("requirement_name") or structured.get("name") or structured.get("title"), structured
    if signal_id == "rfp_issuing_office_v1":
        office = structured.get("office") or structured.get("name")
        address = structured.get("address")
        return _join_parts([office, address]), structured
    if signal_id == "rfp_issuing_agency_v1":
        return structured.get("agency") or structured.get("name") or structured.get("label"), structured
    if signal_id == "rfp_primary_naics_v1":
        return structured.get("naics") or structured.get("code") or structured.get("value"), structured
    if signal_id == "rfp_primary_psc_v1":
        return structured.get("psc") or structured.get("code") or structured.get("value"), structured
    if signal_id == "rfp_office_aac_v1":
        return structured.get("aac") or structured.get("code"), structured
    if signal_id == "rfp_competition_type_v1":
        return (
            structured.get("competition_type")
            or structured.get("type")
            or structured.get("label")
            or structured.get("value"),
            structured,
        )
    if signal_id == "rfp_set_aside_v1":
        return (
            structured.get("set_aside")
            or structured.get("type")
            or structured.get("label")
            or structured.get("value"),
            structured,
        )
    if signal_id == "rfp_contract_type_v1":
        return (
            structured.get("pricing_arrangement")
            or structured.get("overall")
            or structured.get("contract_type")
            or structured.get("label"),
            structured,
        )
    if signal_id == "rfp_eval_method_v1":
        return structured.get("method") or structured.get("evaluation_method") or structured.get("label"), structured
    if signal_id == "rfp_eval_weights_v1":
        weights = structured.get("weights") or structured.get("factors")
        if isinstance(weights, dict):
            parts = [f"{key}: {weights[key]}" for key in weights]
            return ("; ".join(parts) if parts else structured.get("summary")), structured
        if isinstance(weights, list):
            return ("; ".join(str(item) for item in weights if item) or structured.get("summary")), structured
        return structured.get("summary") or structured.get("label"), structured
    if signal_id == "rfp_description_v1":
        return structured.get("description") or structured.get("summary") or structured.get("text"), structured
    if signal_id == "rfp_primary_poc_v1":
        return _join_parts([structured.get("name"), structured.get("email"), structured.get("phone")]), structured
    if signal_id == "rfp_place_of_performance_v1":
        return _join_parts(
            [structured.get("city"), structured.get("state"), structured.get("country"), structured.get("address")],
            sep=", ",
        ), structured
    if signal_id == "rfp_incumbent_data_v1":
        return (
            structured.get("incumbent")
            or structured.get("contractor")
            or structured.get("name")
            or structured.get("label"),
            structured,
        )
    if signal_id == "rfp_period_of_performance_v1":
        parts = [structured.get("base_period")]
        options = structured.get("option_periods")
        if isinstance(options, list):
            for item in options:
                if isinstance(item, dict):
                    parts.append(item.get("period") or item.get("option"))
                elif item:
                    parts.append(str(item))
        elif structured.get("total_period"):
            parts.append(structured.get("total_period"))
        return _join_parts(parts, sep="; "), structured
    if signal_id == "rfp_submission_method_v1":
        return structured.get("method") or structured.get("submission_method") or structured.get("label"), structured
    if signal_id == "rfp_submission_destination_v1":
        return (
            structured.get("destination")
            or structured.get("email")
            or structured.get("url")
            or structured.get("label"),
            structured,
        )
    if signal_id == "solicitation_is_stalled_v1":
        return structured.get("stalled") if "stalled" in structured else structured.get("value"), structured
    if signal_id == "solicitation_status_alert_v1":
        return structured.get("alert") or structured.get("status") or structured.get("message"), structured
    return None, structured
