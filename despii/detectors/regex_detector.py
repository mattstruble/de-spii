import logging
import re

from despii.core import RedactionContext

logger = logging.getLogger(__name__)

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


def regex_pass(ctx: RedactionContext) -> RedactionContext:
    logger.debug("Starting regex PII detection (text length: %d chars)", len(ctx.text))
    total_matches = 0

    for label, pattern in REGEX_PATTERNS.items():
        matches = pattern.findall(ctx.text)
        if matches:
            logger.debug("Found %d matches for pattern %s", len(matches), label)
            for match in matches:
                ctx.redact(match, label)
                total_matches += 1

    logger.info("Regex detector found %d PII matches", total_matches)
    return ctx
