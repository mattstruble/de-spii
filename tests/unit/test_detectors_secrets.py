from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from despii.detectors.secrets import secrets_pass


def test_secrets_pass_extracts_only_secret_substrings(ctx_factory, monkeypatch):
    pii = "AKIAIOSFODNN7EXAMPLE"  # pragma: allowlist secret
    text = f"Some preface {pii} and some suffix"
    ctx = ctx_factory(text)

    _ = secrets_pass(ctx)

    ctx.redact.assert_called_once_with(
        pii,
        "AWS_ACCESS_KEY",
    )


def test_secrets_pass_handles_multiple_lines(ctx_factory, monkeypatch):
    pii = "AKIAIOSFODNN7EXAMPLE"  # pragma: allowlist secret

    text = f"""line1
api = foo
key = {pii}
password = bar
"""  # pragma: allowlist secret
    ctx = ctx_factory(text)

    _ = secrets_pass(ctx)

    ctx.redact.assert_called_once_with(
        pii,
        "AWS_ACCESS_KEY",
    )


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.text(max_size=500))
def test_detector_does_not_crash_on_arbitrary_text(ctx_factory, text):
    """Detector should not raise on arbitrary input; redact calls optional."""
    ctx = ctx_factory(text)

    _ = secrets_pass(ctx)

    ctx.redact.assert_not_called()
