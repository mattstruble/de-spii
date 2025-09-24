import re

from despii.core import DeSPII

REGEX_PATTERNS = {
    "EMAIL": re.compile(r"\b[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}\b"),
    "PHONE_US": re.compile(
        r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"
    ),
    "IPV4": re.compile(
        r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d{1,2})\.){3}(?:25[0-5]|2[0-4]\d|1?\d{1,2})\b"
    ),
    "CREDIT_CARD": re.compile(
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b"
    ),
    "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "MAC": re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b"),
    "UUID": re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-[89ABab][0-9a-fA-F]{3}-[0-9a-fA-F]{12}\b"
    ),
}


def regex_pass(despii: DeSPII) -> DeSPII:
    for label, pattern in REGEX_PATTERNS.items():
        for match in pattern.findall(despii.text):
            if match not in despii.pii_map.values():
                placeholder = despii.create_placeholder(label)
                despii.pii_map[placeholder] = match
                despii.text = despii.text.replace(match, placeholder)

    return despii
