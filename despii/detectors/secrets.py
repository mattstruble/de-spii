from detect_secrets.core.scan import scan_line
from detect_secrets.settings import default_settings

from despii.core import RedactionContext


def _normalize_label(raw_label: str | None) -> str:
    if not raw_label:
        return "SECRET"
    return str(raw_label).upper().replace(" ", "_")


def secrets_pass(ctx: RedactionContext) -> RedactionContext:
    with default_settings() as settings:
        settings.disable_plugins(
            "Base64HighEntropyString",
            "HexHighEntropyString",
        )
        for res in scan_line(ctx.text):
            secret = res.secret_value
            label = _normalize_label(res.type)
            ctx.redact(secret, label)

    return ctx
