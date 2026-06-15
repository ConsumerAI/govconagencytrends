from __future__ import annotations

from typing import Any

SYSTEM_INSTRUCTION_TEXT = (
    "You are AEGIS_SIGNAL_EXTRACTOR, an audit-grade government solicitation extraction engine.\n\n"
    "PRIME DIRECTIVE:\n"
    "Extract only what is explicitly supported by the provided documents. Never infer, normalize, guess, or use external knowledge.\n\n"
    "MANDATORY OUTPUT STANDARD:\n"
    "For every field, return an object containing:\n"
    "- value: The extracted data, or null.\n"
    "- status: MUST be one of ['explicit', 'derived_from_explicit_inputs', 'null_due_to_absence', 'null_due_to_conflict'].\n"
    "- confidence: 'high', 'medium', or 'low'.\n"
    "- page: The integer page number of the primary evidence (use 0 for Excel).\n"
    "- evidence: An array of citation objects [{ source_file, quote }].\n"
    "- notes: Explain your decision, the rule used, or why a conflict occurred.\n\n"
    "MANDATORY EXTRACTION WORKFLOW:\n"
    "1. Search all likely source locations across all files.\n"
    "2. Collect all explicit candidate answers.\n"
    "3. Apply the field-specific search map and source precedence rule.\n"
    "4. If sources conflict and cannot be resolved, return null_due_to_conflict.\n"
    "5. If no value is found, return null_due_to_absence only after checking all likely sources.\n\n"
    "SOURCE PRECEDENCE:\n"
    "1. Solicitation face / SF33 / SF1449 / signed blocks\n"
    "2. Amendments explicitly revising the field\n"
    "3. Section L\n"
    "4. Section F / admin terms\n"
    "5. Section B / CLIN structure\n"
    "6. PWS/SOW/SOO\n"
    "7. Pricing workbook / GFPM\n"
    "8. Q&A / appendices\n\n"
    "NON-NEGOTIABLES:\n"
    "- Do not stop at the first plausible answer.\n"
    "- Do not confuse narrative work locations with priced staffing locations.\n"
    "- Do not compute derived values unless all mathematical inputs are explicit.\n\n"
    "FINAL COVERAGE AUDIT:\n"
    "Every field must end in one of the 4 approved statuses. Use the `_reasoning_scratchpad` to complete a hidden checklist confirming you checked all sources before returning null."
)

USER_MESSAGE_TEXT = (
    "Analyze the provided Solicitation, Performance Work Statement (PWS), and Government Format Pricing Model (GFPM). "
    "Cross-reference all documents to extract the required fields according to the strict JSON schema provided. "
    "Pay close attention to the descriptions of each field to ensure you do not confuse similar concepts "
    "(e.g., Issuing Agency vs. Issuing Office, or Set-Aside vs. Competition Type). "
    "For Section L and Section M (evaluation) content: extract the Rules of the Game. "
    'Every field MUST use array-based evidence: [{ source_file: "...", quote: "..." }]. '
    "When a value comes from both a PDF (e.g., Section L narrative) and Excel (e.g., labor rates), include one citation object per source."
)


def create_signal_field(description: str) -> dict[str, Any]:
    return {
        "type": "object",
        "description": description,
        "properties": {
            "value": {"type": ["string", "null"], "description": "The extracted value."},
            "status": {
                "type": "string",
                "description": "explicit, derived_from_explicit_inputs, null_due_to_absence, or null_due_to_conflict",
            },
            "confidence": {"type": "string", "description": "'high', 'medium', or 'low'"},
            "page": {
                "type": "integer",
                "description": "The integer page number where the primary evidence is found (0 for Excel/tabular).",
            },
            "notes": {
                "type": "string",
                "description": "Your internal reasoning, decision rules applied, or conflict explanations",
            },
            "evidence": {
                "type": "array",
                "description": (
                    "Array of citations. Each citation has source_file (exact file name) and quote (exact snippet). "
                    "If cross-referencing, include one object per source."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "source_file": {"type": "string", "description": "The exact file name this quote came from."},
                        "quote": {"type": "string", "description": "The exact text snippet or Excel row data."},
                    },
                    "required": ["source_file", "quote"],
                },
            },
        },
        "required": ["value", "status", "confidence", "page", "evidence", "notes"],
    }


def create_odc_plugs_field(description: str) -> dict[str, Any]:
    per_period = {
        "type": "array",
        "items": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "period": {"type": "string"},
                "amount": {"type": "number"},
            },
            "required": ["period", "amount"],
        },
    }
    return {
        "type": "object",
        "description": description,
        "properties": {
            "value": {
                "type": ["object", "null"],
                "additionalProperties": False,
                "properties": {
                    "travel": per_period,
                    "materials": per_period,
                    "total_travel": {"type": "number"},
                    "total_materials": {"type": "number"},
                    "total_odc_evaluated": {"type": "number"},
                },
                "required": ["travel", "materials", "total_travel", "total_materials", "total_odc_evaluated"],
            },
            "status": {
                "type": "string",
                "description": "explicit, derived_from_explicit_inputs, null_due_to_absence, or null_due_to_conflict",
            },
            "confidence": {"type": "string", "description": "'high', 'medium', or 'low'"},
            "page": {
                "type": "integer",
                "description": "The integer page number where the primary evidence is found (0 for Excel/tabular).",
            },
            "notes": {
                "type": "string",
                "description": "Your internal reasoning, decision rules applied, or conflict explanations",
            },
            "evidence": {
                "type": "array",
                "description": (
                    "Array of citations. Each citation has source_file (exact file name) and quote (exact snippet)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "source_file": {"type": "string"},
                        "quote": {"type": "string"},
                    },
                    "required": ["source_file", "quote"],
                },
            },
        },
        "required": ["value", "status", "confidence", "page", "evidence", "notes"],
    }


SIGNAL_FIELD_LIST: list[tuple[str, str]] = [
    ("issuing_office", "The specific base, command, or office issuing the solicitation (e.g., W6QK INSCOM BELVOIR, MICC - Fort Knox)."),
    ("issuing_agency", "The top-level cabinet department or federal agency (e.g., Department of the Army, Department of Defense, GSA). Do NOT use the specific base/command."),
    ("issuing_office_aac", "Extract the 6-character Activity Address Code (AAC) for the issuing office (the first 6 characters of the solicitation number, e.g., W50NH9)."),
    ("solicitation_title", "The official title or name of the project/program."),
    ("solicitation_id", "Extract the current solicitation/RFP number (e.g., W50NH926RA002)."),
    (
        "prior_contract_piid",
        "PIID or contract number of the predecessor contract being recompeted (e.g. W50NH9-025-R-0032). Search GFPM/pricing workbook headers, Section H, and transition/recompete text. Do NOT return the new solicitation number.",
    ),
    ("solicitation_description", "A brief description of the services or products being acquired."),
    ("requirement_name", "The exact title of the program/requirement (e.g., 'Intelligence and Automation Operations')."),
    ("contracting_officer_name", "The name of the primary Contracting Officer (KO/CO)."),
    ("contract_specialist_name", "The name of the Contract Specialist (CS) or point of contact."),
    ("primary_poc", "Primary point of contact email or phone for the solicitation."),
    ("primary_naics_code", "The 6-digit North American Industry Classification System (NAICS) code."),
    ("psc_code", "The 4-character Product Service Code (PSC)."),
    (
        "set_aside_type",
        "The socioeconomic or set-aside designation, if explicitly stated, such as 'Total Small Business Set-Aside', '8(a) Set-Aside', 'HUBZone Set-Aside', 'SDVOSB Set-Aside', or 'WOSB Set-Aside'. If full and open with no set-aside, return 'Unrestricted' or null based on existing convention. Do NOT return evaluation methodology.",
    ),
    (
        "competition_type",
        "The overall solicitation competition category, if explicitly stated. Return values such as 'Full and Open Competition', 'Total Small Business Set-Aside', '8(a) Set-Aside', 'HUBZone Set-Aside', 'SDVOSB Set-Aside', 'WOSB Set-Aside', or 'Sole Source'. Do NOT return evaluation methodology, proposal phases, down-select structure, tradeoff methodology, LPTA, pass/fail factors, or other source-selection procedures. If not explicitly stated, return null.",
    ),
    ("contract_type", "The pricing structure of the contract (e.g., Firm Fixed Price (FFP), Time & Materials (T&M), Cost Plus Fixed Fee (CPFF))."),
    ("proposal_due_date", "The deadline date for proposal submission."),
    ("period_of_performance_start", "The start date of the base period."),
    ("period_of_performance_end", "The end date of the final option period."),
    ("total_period_of_performance", "The total duration of the contract including all base and option periods (e.g., 5 years and 6 months)."),
    ("places_of_performance", "Extract a strictly exhaustive list of all Places of Performance (PoP) and work locations; reconcile narrative + pricing matrix sources."),
    ("total_ftes", "The total number of Full Time Equivalents (FTEs) required (usually pricing model/staffing plan)."),
    (
        "odc_and_travel_plugs",
        "Government-provided ODC and Travel plug numbers by period from any structured pricing attachment or CLIN/ODC table.",
    ),
    ("productive_hours_per_fte", "Extract baseline productive hours used for a single FTE for one year (e.g., 1920, 1880, 1912)."),
    ("fee_and_burden_constraints", "Extract explicit rules regarding profit/fee/indirect burdens (e.g., no fee on ODCs)."),
    ("contract_vehicle", "The specific contract vehicle/MAC this task order is under (if explicitly named)."),
    ("clearance_requirements", "Extract overarching facility or personnel security clearance requirements (e.g., Secret, TS/SCI)."),
    (
        "incumbent_data",
        "Search GFPM/pricing headers and Section H for PIIDs and incumbent contractor name. If a prior contract number is found, return it even without a company name. Do NOT return the current solicitation number.",
    ),
    ("evaluation_methodology", "Identify if award is 'LPTA' or 'Tradeoff' (Best Value) and relative importance of Price vs Technical."),
    ("technical_factors", "List the specific Technical Evaluation Factors and note pass/fail vs rated."),
    ("questions_due_date", "The deadline for industry questions/RFIs."),
    ("past_performance_requirements", "Number of references required and the recency/relevancy definitions."),
    ("pricing_instructions", "Extract Section L pricing constraints; cite both narrative + Excel where applicable."),
    ("key_personnel_specs", "Detailed requirements for Key Personnel (education, years of experience, clearances)."),
    ("submission_format_instructions", "Extract how the government wants the proposal delivered (email/upload/hard copy; Excel vs PDF pricing)."),
    ("submission_method", "How proposals must be submitted (portal, email, hard copy, DoD SAFE, etc.)."),
    ("submission_destination", "Where proposals must be sent (email address, portal URL, physical address)."),
    ("evaluation_factors_relative_importance", "In Section M, extract 'relative importance' statement."),
    ("uncompensated_overtime_policy", "Extract uncompensated overtime policy from Section L."),
    ("material_markup_ceiling", "Extract any caps/ceilings on ODC/Material handling or markup rates."),
    ("labor_mapping_mandate", "Does the RFP mandate specific Labor Categories or can we propose our own?"),
    ("security_clearance_depth", "Extract facility clearance level and any explicit percentage split (TS/SCI vs Secret)."),
    ("page_limit_constraints", "Extract page limits per volume (Technical/Management/Past Performance)."),
]


def build_schema_from_field_list(fields: list[tuple[str, str]]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "_reasoning_scratchpad": {"type": "string", "description": "Internal scratchpad for step-by-step reasoning."},
    }
    for key, description in fields:
        properties[key] = create_odc_plugs_field(description) if key == "odc_and_travel_plugs" else create_signal_field(description)
    return {"type": "object", "properties": properties, "required": ["_reasoning_scratchpad"]}


def schema_chunks() -> list[list[tuple[str, str]]]:
    midpoint = (len(SIGNAL_FIELD_LIST) + 1) // 2
    return [SIGNAL_FIELD_LIST[:midpoint], SIGNAL_FIELD_LIST[midpoint:]]
