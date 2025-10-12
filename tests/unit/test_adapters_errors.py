import pytest

from despii.adapters.errors import UnsupportedModelInterfaceError
from despii.errors import DeSpiiError


class TestUnsupportedModelInterfaceError:
    """Test UnsupportedModelInterfaceError exception."""

    def test_error_inherits_from_despii_error(self):
        """Test that UnsupportedModelInterfaceError inherits from DeSpiiError."""
        assert issubclass(UnsupportedModelInterfaceError, DeSpiiError)

    def test_error_can_be_raised(self):
        """Test that the error can be raised with a message."""
        with pytest.raises(UnsupportedModelInterfaceError) as exc_info:
            raise UnsupportedModelInterfaceError("Test error message")
        assert str(exc_info.value) == "Test error message"

    def test_error_can_be_caught_as_despii_error(self):
        """Test that the error can be caught as DeSpiiError."""
        with pytest.raises(DeSpiiError):
            raise UnsupportedModelInterfaceError("Test error")

    def test_error_can_be_caught_as_exception(self):
        """Test that the error can be caught as generic Exception."""
        with pytest.raises(Exception):
            raise UnsupportedModelInterfaceError("Test error")

    def test_error_with_empty_message(self):
        """Test that the error works with an empty message."""
        with pytest.raises(UnsupportedModelInterfaceError):
            raise UnsupportedModelInterfaceError("")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

