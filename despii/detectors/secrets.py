import logging

from detect_secrets.core.scan import scan_line
from detect_secrets.settings import default_settings

from despii.core import RedactionContext

logger = logging.getLogger(__name__)


def _normalize_label(raw_label: str | None) -> str:
    if not raw_label:
        return "SECRET"
    return str(raw_label).upper().replace(" ", "_")


def secrets_pass(ctx: RedactionContext) -> RedactionContext:
    logger.debug("Starting secrets detection (text length: %d chars)", len(ctx.text))
    secret_count = 0

    with default_settings() as settings:
        settings.disable_plugins(
            "Base64HighEntropyString",
            "HexHighEntropyString",
        )
        for res in scan_line(ctx.text):
            secret = res.secret_value
            label = _normalize_label(res.type)
            logger.debug("Found secret: type=%s, length=%d", label, len(secret))
            ctx.redact(secret, label)
            secret_count += 1

    logger.info("Secrets detector found %d secrets", secret_count)
    return ctx
