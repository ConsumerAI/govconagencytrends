from __future__ import annotations

from typing import Any


def check_scope_review_compatibility(resolved_payload: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "loaderAccepted": False,
        "previewConstructed": False,
        "validationError": None,
        "filterDrivingFields": {},
        "confirmedMappings": [],
        "suggestedMappings": [],
        "unmappedValues": [],
        "withheldConflicts": [],
    }
    try:
        from app import build_solicitation_scope_preview, load_resolved_signals_json, map_solicitation_signals_to_dashboard_filters

        loaded = load_resolved_signals_json(resolved_payload)
        result["loaderAccepted"] = True
        preview = build_solicitation_scope_preview(loaded)
        result["previewConstructed"] = bool(preview)
        result["filterDrivingFields"] = {
            field: {
                "value": data.get("value"),
                "confidence": data.get("confidence"),
                "signalId": data.get("signal_id"),
            }
            for field, data in preview.items()
        }

        mapping = map_solicitation_signals_to_dashboard_filters(
            preview,
            agency_records=[],
            fiscal_year=2024,
        )
        for row in mapping.get("rows") or []:
            status = str(row.get("mapping_status") or "")
            entry = {
                "field": row.get("field"),
                "extractedValue": row.get("extracted_value"),
                "mappedFilter": row.get("mapped_filter"),
                "mappingStatus": status,
                "filterKey": row.get("filter_key"),
            }
            if status in {"Exact match", "Confirmed"} or row.get("preselect"):
                result["confirmedMappings"].append(entry)
            elif status in {"Suggested match", "Suggested"}:
                result["suggestedMappings"].append(entry)
            elif status in {"Unmapped", "Withheld", "Conflict"}:
                if status in {"Withheld", "Conflict"}:
                    result["withheldConflicts"].append(entry)
                else:
                    result["unmappedValues"].append(entry)

        for signal in loaded.get("signals") or []:
            if str(signal.get("resolution_status") or "") == "unresolved_conflict":
                result["withheldConflicts"].append(
                    {
                        "field": signal.get("id"),
                        "extractedValue": signal.get("canonical_value"),
                        "mappingStatus": "withheld",
                        "filterKey": signal.get("id"),
                    }
                )
    except Exception as exc:  # noqa: BLE001 - report validation failures in parity artifact
        result["validationError"] = str(exc)
    return result
