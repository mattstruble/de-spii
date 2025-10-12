"""Pytest configuration for integration tests."""

import subprocess

import pytest


def is_ollama_available() -> bool:
    """Check if Ollama is running and llama3 model is available.

    :return: True if Ollama is available with llama3 model
    """
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        return result.returncode == 0 and "llama3" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    """Check if Ollama is available for integration tests.

    :return: True if Ollama is available
    """
    return is_ollama_available()


@pytest.fixture(scope="session")
def skip_if_no_ollama(ollama_available: bool) -> None:  # noqa: PT004
    """Skip test if Ollama is not available.

    :param ollama_available: Whether Ollama is available
    """
    if not ollama_available:
        pytest.skip("Ollama with llama3 model not available")


@pytest.fixture(scope="session")
def ollama_model_name() -> str:
    """Get the Ollama model name to use for tests.

    :return: Model name (llama3.1:8b or llama3:8b)
    """
    result = subprocess.run(
        ["ollama", "list"],
        capture_output=True,
        text=True,
        timeout=5,
        check=False,
    )
    if "llama3.1:8b" in result.stdout:
        return "llama3.1:8b"
    return "llama3:8b"


@pytest.fixture
def test_prompt() -> str:
    """Provide a simple test prompt for integration tests.

    :return: Test prompt string
    """
    return "Say 'Hello, World!' and nothing else."


@pytest.fixture
def pii_test_text() -> str:
    """Provide test text containing PII for detector tests.

    :return: Text with PII for testing
    """
    return "My name is John Smith and my email is john.smith@example.com. I live in New York."
