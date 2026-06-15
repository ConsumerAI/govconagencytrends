from __future__ import annotations

SF1449_FIELD_LABELS: dict[str, str] = {
    "05solicitationnumber": "SOLICITATION NUMBER",
    "06solissuedate": "SOLICITATION ISSUE DATE",
    "08offerduedate": "OFFER DUE DATE",
    "08offerduedatelocaltime": "OFFER DUE DATE LOCAL TIME",
    "09issuedby": "ISSUED BY",
    "09issuedbycode": "ISSUED BY CODE",
    "10naics": "NORTH AMERICAN INDUSTRY CLASSIFICATION STANDARD (NAICS)",
    "10sizestandard": "SIZE STANDARD",
    "10setasidecheckbox": "SET-ASIDE",
    "10setasidepercent": "SET-ASIDE PERCENT",
    "10_8acheckbox": "8(A) SET-ASIDE",
    "10smallbusinesscheckbox": "SMALL BUSINESS SET-ASIDE",
    "10hubzonecheckbox": "HUBZONE SET-ASIDE",
    "10sdvosbcheckbox": "SDVOSB SET-ASIDE",
    "10wosbcheckbox": "WOMEN-OWNED SMALL BUSINESS SET-ASIDE",
    "10unrestrictedcheckbox": "UNRESTRICTED",
    "10fullandopencheckbox": "FULL AND OPEN",
    "11psc": "PRODUCT OR SERVICE CODE (PSC)",
    "11psccode": "PRODUCT OR SERVICE CODE (PSC)",
    "15deliverto": "DELIVER TO",
    "16administeredby": "ADMINISTERED BY",
}


def humanize_field_name(field_name: str) -> str:
    lowered = field_name.lower()
    for key, label in SF1449_FIELD_LABELS.items():
        if key in lowered or lowered.endswith(key):
            return label
    tail = field_name.split(".")[-1]
    tail = tail.replace("[0]", "").strip()
    if tail.lower() in SF1449_FIELD_LABELS:
        return SF1449_FIELD_LABELS[tail.lower()]
    cleaned = tail
    if cleaned[:2].isdigit():
        cleaned = cleaned[2:]
    cleaned = cleaned.replace("checkbox", " checkbox").replace("_", " ").strip()
    return cleaned.upper() if cleaned else field_name
