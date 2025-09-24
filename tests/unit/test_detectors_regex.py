import pytest
from hypothesis import given
from hypothesis import strategies as st

from despii.core import DeSPII, reconstruct_text
from despii.detectors.regex import REGEX_PATTERNS, regex_pass

PII_STRATEGIES = {}

for label, pattern in REGEX_PATTERNS.items():
    PII_STRATEGIES[label] = st.from_regex(pattern, fullmatch=True)


def test_no_pii_text():
    """Test with text containing no PII."""
    text = "This is normal text with no sensitive information."
    despii = DeSPII(text=text, pii_map={})
    result = regex_pass(despii)

    assert result.text == text
    assert result.pii_map == {}
    assert result.count == 0


@pytest.mark.parametrize("pii_type", list(PII_STRATEGIES.keys()))
@given(st.data())
def test_pii_detection_and_reconstruction(pii_type, data):
    """Test that PII is detected, replaced, and reconstructable for each type."""
    pii_value = data.draw(PII_STRATEGIES[pii_type])
    text = f"Here is some PII: {pii_value} in the text."

    despii = DeSPII(text=text, pii_map={})
    result = regex_pass(despii)

    # PII should be replaced
    assert pii_value not in result.text
    assert len(result.pii_map) == 1
    assert pii_value in result.pii_map.values()

    # Text should be reconstructable
    reconstructed = reconstruct_text(result.text, result)
    assert reconstructed == text


def test_duplicate_pii_handling():
    """Test that duplicate PII values share the same placeholder."""
    text = "Email me at test@example.com or test@example.com"
    despii = DeSPII(text=text, pii_map={})
    result = regex_pass(despii)

    # Only one mapping for the duplicate email
    assert len(result.pii_map) == 1
    assert "test@example.com" in result.pii_map.values()
    assert "test@example.com" not in result.text

    # Should reconstruct correctly
    reconstructed = reconstruct_text(result.text, result)
    assert reconstructed == text


def test_multiple_pii_types():
    """Test handling multiple different PII types."""
    text = "Contact john@example.com or call 555-123-4567 from server 192.168.1.1"
    despii = DeSPII(text=text, pii_map={})
    result = regex_pass(despii)

    # Should detect all three PII items
    assert len(result.pii_map) == 3

    # Should reconstruct correctly
    reconstructed = reconstruct_text(result.text, result)
    assert reconstructed == text


@given(st.text(max_size=500))
def test_reconstruction_invariant(text):
    """Property test: any text should be perfectly reconstructable."""
    despii = DeSPII(text=text, pii_map={})
    result = regex_pass(despii)

    # Core invariant: reconstruction always works
    reconstructed = reconstruct_text(result.text, result)
    assert reconstructed == text

    # Basic sanity checks
    assert isinstance(result.count, int)
    assert result.count >= 0
    assert len(result.pii_map) == result.count


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
