"""Integration tests for DSPy adapter with real models."""

import pytest

pytest.importorskip("dspy")

import dspy  # noqa: E402

from despii.adapters.dspy import DSPyAdapter  # noqa: E402

pytestmark = pytest.mark.integration


class TestDSPyAdapterIntegration:
    """Integration tests for DSPy adapter with Ollama."""

    @pytest.fixture
    def ollama_model(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Create DSPy LM with Ollama.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        :return: DSPy LM instance
        """
        return dspy.LM(f"ollama_chat/{ollama_model_name}", api_base="http://localhost:11434", api_key="")

    @pytest.fixture
    def adapter(self, ollama_model):
        """Create DSPyAdapter with Ollama model.

        :param ollama_model: DSPy LM instance
        :return: DSPyAdapter instance
        """
        return DSPyAdapter(ollama_model)

    def test_dspy_adapter_can_generate_response(self, adapter, test_prompt):
        """Test that DSPy adapter can generate a response with real model.

        :param adapter: DSPyAdapter instance
        :param test_prompt: Test prompt fixture
        """
        response = adapter.generate(test_prompt)

        assert response is not None
        assert response.text
        assert isinstance(response.text, str)
        assert len(response.text) > 0
        assert response.framework == "dspy"

    def test_dspy_adapter_response_contains_expected_content(self, adapter):
        """Test that DSPy adapter returns relevant responses.

        :param adapter: DSPyAdapter instance
        """
        prompt = "What is 2+2? Answer with just the number."
        response = adapter.generate(prompt)

        assert response is not None
        # Should contain "4" somewhere in response
        assert "4" in response.text

    def test_dspy_adapter_handles_pii_detection_prompt(self, adapter):
        """Test DSPy adapter with a PII detection prompt.

        :param adapter: DSPyAdapter instance
        """
        prompt = """Identify any personally identifiable information (PII) in this text:
"My name is Alice Johnson and my email is alice@example.com"

Return a JSON array with objects containing 'pii_str' and 'label' fields."""

        response = adapter.generate(prompt)

        assert response is not None
        assert response.text
        # Response should mention some PII-related content
        assert any(
            word in response.text.lower() for word in ["alice", "email", "pii", "personal", "information", "name"]
        )

    def test_dspy_adapter_preserves_raw_response(self, adapter, test_prompt):
        """Test that raw response is preserved in LLMResponse.

        :param adapter: DSPyAdapter instance
        :param test_prompt: Test prompt fixture
        """
        response = adapter.generate(prompt=test_prompt)

        assert response.raw is not None
        # Raw response should be a list for DSPy
        assert isinstance(response.raw, list)
        assert len(response.raw) > 0
