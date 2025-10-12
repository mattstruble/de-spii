"""Integration tests to verify proper text extraction from LLM responses across all adapters."""

import pytest

pytest.importorskip("dspy")
pytest.importorskip("langchain")
pytest.importorskip("langgraph")

import dspy  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

from despii.adapters.dspy import DSPyAdapter  # noqa: E402
from despii.adapters.langchain import LangChainAdapter  # noqa: E402
from despii.adapters.langgraph import LangGraphAdapter  # noqa: E402

pytestmark = pytest.mark.integration


class TestTextExtractionDSPy:
    """Integration tests for DSPy adapter text extraction."""

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

    def test_text_field_is_string(self, adapter):
        """Test that DSPy adapter returns text field as string, not list.

        :param adapter: DSPyAdapter instance
        """
        response = adapter.generate("Say hello")

        # Text field must be a string, not a list
        assert isinstance(response.text, str)
        assert len(response.text) > 0
        # Should not have Python list representation characters
        assert not response.text.startswith("[")
        assert not response.text.endswith("]")

    def test_text_contains_actual_response(self, adapter):
        """Test that text field contains actual LLM response content.

        :param adapter: DSPyAdapter instance
        """
        response = adapter.generate("What is 2+2? Answer with just the number.")

        assert isinstance(response.text, str)
        # Should contain the answer
        assert "4" in response.text
        # Should not be a Python representation
        assert "['4']" not in response.text

    def test_json_response_properly_extracted(self, adapter):
        """Test that JSON responses are properly extracted as strings.

        :param adapter: DSPyAdapter instance
        """
        prompt = 'Return this JSON: [{"key": "value"}]'
        response = adapter.generate(prompt)

        assert isinstance(response.text, str)
        # Should be able to parse as JSON if LLM returns JSON
        # (might have markdown wrapping, but should be string)
        assert not response.text.startswith("['")


class TestTextExtractionLangChain:
    """Integration tests for LangChain adapter text extraction."""

    @pytest.fixture
    def ollama_model(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Create LangChain ChatOllama model.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        :return: ChatOllama instance
        """
        return ChatOllama(model=ollama_model_name, base_url="http://localhost:11434")

    @pytest.fixture
    def adapter(self, ollama_model):
        """Create LangChainAdapter with Ollama model.

        :param ollama_model: ChatOllama instance
        :return: LangChainAdapter instance
        """
        return LangChainAdapter(ollama_model)

    def test_text_field_is_string(self, adapter):
        """Test that LangChain adapter returns text field as string.

        :param adapter: LangChainAdapter instance
        """
        response = adapter.generate("Say hello")

        assert isinstance(response.text, str)
        assert len(response.text) > 0

    def test_text_contains_actual_response(self, adapter):
        """Test that text field contains actual LLM response content.

        :param adapter: LangChainAdapter instance
        """
        response = adapter.generate("What is 2+2? Answer with just the number.")

        assert isinstance(response.text, str)
        assert "4" in response.text

    def test_json_response_properly_extracted(self, adapter):
        """Test that JSON responses are properly extracted as strings.

        :param adapter: LangChainAdapter instance
        """
        prompt = 'Return this JSON: [{"key": "value"}]'
        response = adapter.generate(prompt)

        assert isinstance(response.text, str)


class TestTextExtractionLangGraph:
    """Integration tests for LangGraph adapter text extraction."""

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

    def test_text_field_is_string(self, adapter):
        """Test that LangGraph adapter returns text field as string.

        :param adapter: LangGraphAdapter instance
        """
        response = adapter.generate("Say hello")

        assert isinstance(response.text, str)
        assert len(response.text) > 0

    def test_text_contains_actual_response(self, adapter):
        """Test that text field contains actual LLM response content.

        :param adapter: LangGraphAdapter instance
        """
        response = adapter.generate("What is 2+2? Answer with just the number.")

        assert isinstance(response.text, str)
        assert "4" in response.text

    def test_json_response_properly_extracted(self, adapter):
        """Test that JSON responses are properly extracted as strings.

        :param adapter: LangGraphAdapter instance
        """
        prompt = 'Return this JSON: [{"key": "value"}]'
        response = adapter.generate(prompt)

        assert isinstance(response.text, str)
