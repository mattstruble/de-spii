from despii.errors import DeSpiiError


class UnsupportedModelInterfaceError(DeSpiiError):
    """Raised when a model doesn't implement a supported interface for its framework."""
