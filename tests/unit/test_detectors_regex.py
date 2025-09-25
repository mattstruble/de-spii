import re

import pytest
from hypothesis import given
from hypothesis import strategies as st

from despii.core import RedactionContext
from despii.detectors.regex import REGEX_PATTERNS, regex_pass

PII_STRATEGIES = {}

for label, pattern in REGEX_PATTERNS.items():
    PII_STRATEGIES[label] = st.from_regex(pattern, fullmatch=True)


def test_no_pii_text():
    """Test with text containing no PII."""
    text = "This is normal text with no sensitive information."
    ctx = RedactionContext(text)
    result = regex_pass(ctx)

    assert result.text == text


@pytest.mark.parametrize("pii_type", list(PII_STRATEGIES.keys()))
@given(st.data())
def test_pii_detection_and_reconstruction(pii_type, data):
    """Test that PII is detected, replaced, and reconstructable for each type."""
    pii_value = data.draw(PII_STRATEGIES[pii_type])
    text = f"Here is some PII: {pii_value} in the text."

    ctx = RedactionContext(text)
    result = regex_pass(ctx)

    # PII should be replaced
    assert pii_value not in result.text

    # Text should be reconstructable
    assert result.unredact(result.text) == text


def test_duplicate_pii_handling():
    """Test that duplicate PII values share the same placeholder."""
    text = "Email me at test@example.com or test@example.com"
    ctx = RedactionContext(text)
    result = regex_pass(ctx)

    # Only one unique placeholder appears twice
    placeholders = re.findall(r"<PII_[A-Z_]+_\d+>", result.text)
    assert len(set(placeholders)) == 1
    assert result.text.count(list(set(placeholders))[0]) == 2
    assert "test@example.com" not in result.text

    # Should reconstruct correctly
    assert result.unredact(result.text) == text


def test_multiple_pii_types():
    """Test handling multiple different PII types."""
    text = "Contact john@example.com or call 555-123-4567 from server 192.168.1.1"
    ctx = RedactionContext(text)
    result = regex_pass(ctx)

    # Should detect all three PII items -> three unique placeholders
    print(result.text)
    placeholders = set(re.findall(r"<PII_[A-Z_\d]+_\d+>", result.text))
    assert len(placeholders) == 3

    # Should reconstruct correctly
    assert result.unredact(result.text) == text


@given(st.text(max_size=500))
def test_reconstruction_invariant(text):
    """Property test: any text should be perfectly reconstructable."""
    ctx = RedactionContext(text)
    result = regex_pass(ctx)

    # Core invariant: reconstruction always works
    assert result.unredact(result.text) == text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
