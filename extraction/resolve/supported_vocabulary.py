from __future__ import annotations

SUPPORTED_SIGNAL_IDS = {
    "process_sources_v1",
    "rfp_solicitation_id_v1",
    "rfp_solicitation_number_v1",
    "rfp_title_v1",
    "rfp_requirement_name_v1",
    "rfp_description_v1",
    "rfp_issuing_agency_v1",
    "rfp_issuing_office_v1",
    "rfp_office_aac_v1",
    "rfp_primary_naics_v1",
    "rfp_primary_psc_v1",
    "rfp_contract_type_v1",
    "rfp_set_aside_v1",
    "rfp_competition_type_v1",
    "rfp_place_of_performance_v1",
    "rfp_period_of_performance_v1",
    "rfp_pop_start_v1",
    "rfp_pop_end_v1",
    "rfp_pop_years_v1",
    "rfp_incumbent_data_v1",
    "rfp_prior_contract_piid_v1",
    "rfp_primary_poc_v1",
    "rfp_ko_name_v1",
    "rfp_contract_specialist_name_v1",
    "rfp_eval_method_v1",
    "rfp_eval_weights_v1",
    "rfp_tech_factors_v1",
    "rfp_submission_method_v1",
    "rfp_submission_destination_v1",
    "rfp_submission_format_v1",
    "rfp_submission_instructions_v1",
    "rfp_questions_due_v1",
    "rfp_odc_plugs_v1",
}


def is_materializable_signal_id(signal_id: str) -> bool:
    return signal_id in SUPPORTED_SIGNAL_IDS
