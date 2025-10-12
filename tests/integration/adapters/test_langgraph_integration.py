"""Integration tests for LangGraph adapter with real models."""

import pytest

pytest.importorskip("langgraph")
pytest.importorskip("langchain")

from langchain_ollama import ChatOllama  # noqa: E402

from despii.adapters.langgraph import LangGraphAdapter  # noqa: E402

pytestmark = pytest.mark.integration


class TestLangGraphAdapterIntegration:
    """Integration tests for LangGraph adapter with Ollama."""

    @pytest.fixture
    def ollama_model(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Create ChatOllama model for LangGraph.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        :return: ChatOllama instance
        """
        return ChatOllama(model=ollama_model_name, base_url="http://localhost:11434")

    @pytest.fixture
    def adapter(self, ollama_model):
        """Create LangGraphAdapter with Ollama model.

        :param ollama_model: ChatOllama instance
        :return: LangGraphAdapter instance
        """
        return LangGraphAdapter(ollama_model)

    def test_langgraph_adapter_can_generate_response(self, adapter, test_prompt):
        """Test that LangGraph adapter can generate a response with real model.

        :param adapter: LangGraphAdapter instance
        :param test_prompt: Test prompt fixture
        """
        response = adapter.generate(test_prompt)

        assert response is not None
        assert response.text
        assert isinstance(response.text, str)
        assert len(response.text) > 0
        assert response.framework == "langgraph"

    def test_langgraph_adapter_response_contains_expected_content(self, adapter):
        """Test that LangGraph adapter returns relevant responses.

        :param adapter: LangGraphAdapter instance
        """
        prompt = "What is 2+2? Answer with just the number."
        response = adapter.generate(prompt)

        assert response is not None
        # Should contain "4" somewhere in response
        assert "4" in response.text

    def test_langgraph_adapter_handles_pii_detection_prompt(self, adapter):
        """Test LangGraph adapter with a PII detection prompt.

        :param adapter: LangGraphAdapter instance
        """
        prompt = """Identify any personally identifiable information (PII) in this text:
"My name is Charlie Brown and my SSN is 123-45-6789"

Return a JSON array with objects containing 'pii_str' and 'label' fields."""

        response = adapter.generate(prompt)

        assert response is not None
        assert response.text
        # Response should mention some PII-related content
        assert any(
            word in response.text.lower() for word in ["charlie", "ssn", "pii", "personal", "information", "name"]
        )

    def test_langgraph_adapter_preserves_raw_response(self, adapter, test_prompt):
        """Test that raw response is preserved in LLMResponse.

        :param adapter: LangGraphAdapter instance
        :param test_prompt: Test prompt fixture
        """
        response = adapter.generate(test_prompt)

        assert response.raw is not None
        # Raw should have content attribute for LangGraph (uses LangChain models)
        assert hasattr(response.raw, "content") or isinstance(response.raw, str)

    def test_langgraph_adapter_invoke_method(self, adapter):
        """Test LangGraph adapter uses invoke method correctly.

        :param adapter: LangGraphAdapter instance
        """
        prompt = "Say hello briefly"
        response = adapter.generate(prompt)

        assert response is not None
        assert response.text
        assert len(response.text) > 0
