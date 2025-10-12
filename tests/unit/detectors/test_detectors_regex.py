import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from despii.detectors.regex_detector import REGEX_PATTERNS, regex_pass

PII_STRATEGIES = {}

for label, pattern in REGEX_PATTERNS.items():
    PII_STRATEGIES[label] = st.from_regex(pattern, fullmatch=True)


def test_no_pii_text(ctx_factory):
    """Detector should not call redact when no PII present."""
    text = "This is normal text with no sensitive information."
    ctx = ctx_factory(text)

    _ = regex_pass(ctx)
    ctx.redact.assert_not_called()


@pytest.mark.parametrize("pii_type", list(PII_STRATEGIES.keys()))
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.data())
def test_pii_detection_and_reconstruction(ctx_factory, pii_type, data):
    """Test that PII is detected, replaced, and reconstructable for each type."""
    pii_value = data.draw(PII_STRATEGIES[pii_type])
    text = f"Here is some PII: {pii_value} in the text."

    ctx = ctx_factory(text)

    _ = regex_pass(ctx)

    # At least one call matches the detected value and its label
    assert any(call.args == (pii_value, pii_type) for call in ctx.redact.call_args_list)


def test_duplicate_pii_handling_calls_redact_twice(ctx_factory):
    """Detector should call redact for each occurrence found by regex."""
    text = "Email me at test@example.com or test@example.com"
    ctx = ctx_factory(text)

    _ = regex_pass(ctx)

    redact_calls = [
        c.args for c in ctx.redact.call_args_list if c.args[0] == "test@example.com"
    ]
    assert len(redact_calls) == 2
    assert all(args[1] == "EMAIL" for args in redact_calls)


def test_multiple_pii_types_calls_with_expected_labels(ctx_factory):
    """Detector should call redact with correct labels for different PII."""
    text = "Contact john@example.com or call 555-123-4567 from server 192.168.1.1"
    ctx = ctx_factory(text)

    _ = regex_pass(ctx)

    calls = [(c.args[0], c.args[1]) for c in ctx.redact.call_args_list]
    assert ("john@example.com", "EMAIL") in calls
    assert ("555-123-4567", "PHONE_US") in calls
    assert ("192.168.1.1", "IPV4") in calls


@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(st.text(max_size=500))
def test_detector_does_not_crash_on_arbitrary_text(ctx_factory, text):
    """Detector should not raise on arbitrary input; redact calls optional."""
    ctx = ctx_factory(text)

    _ = regex_pass(ctx)

    ctx.redact.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
