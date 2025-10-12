from unittest.mock import Mock

import pytest

from despii.adapters.base import LLMAdapter, LLMRegistry, LLMResponse


class TestLLMResponse:
    """Test LLMResponse dataclass."""

    def test_llm_response_initialization_minimal(self):
        """Test creating LLMResponse with only required text field."""
        response = LLMResponse(text="Hello world")
        assert response.text == "Hello world"
        assert response.raw is None
        assert not response.framework
        assert response.model_name is None

    def test_llm_response_initialization_full(self):
        """Test creating LLMResponse with all fields."""
        raw_data = {"content": "test"}
        response = LLMResponse(
            text="Hello world",
            raw=raw_data,
            framework="test_framework",
            model_name="test-model",
        )
        assert response.text == "Hello world"
        assert response.raw == raw_data
        assert response.framework == "test_framework"
        assert response.model_name == "test-model"

    def test_llm_response_with_none_model_name(self):
        """Test that model_name can be None."""
        response = LLMResponse(text="test", model_name=None)
        assert response.model_name is None


class TestLLMAdapter:
    """Test LLMAdapter base class."""

    def test_llm_adapter_initialization(self):
        """Test that LLMAdapter can be initialized with a model."""
        model = Mock()
        adapter = LLMAdapter(model)
        assert adapter.model is model

    def test_llm_adapter_generate_not_implemented(self):
        """Test that generate() raises NotImplementedError in base class."""
        model = Mock()
        adapter = LLMAdapter(model)
        with pytest.raises(NotImplementedError):
            adapter.generate("test prompt")


class TestLLMRegistry:
    """Test LLMRegistry class."""

    def setup_method(self):
        """Reset registry before each test."""
        LLMRegistry._registry = {}

    def test_register_adapter(self):
        """Test registering an adapter."""
        mock_adapter = Mock(spec=LLMAdapter)
        LLMRegistry.register("test_framework", mock_adapter)
        assert "test_framework" in LLMRegistry._registry
        assert LLMRegistry._registry["test_framework"] is mock_adapter

    def test_register_adapter_case_insensitive(self):
        """Test that registration is case-insensitive."""
        mock_adapter = Mock(spec=LLMAdapter)
        LLMRegistry.register("TestFramework", mock_adapter)
        assert "testframework" in LLMRegistry._registry

    def test_get_adapter_exists(self):
        """Test retrieving an existing adapter."""
        mock_adapter = Mock(spec=LLMAdapter)
        LLMRegistry.register("test_framework", mock_adapter)
        retrieved = LLMRegistry.get_adapter("test_framework")
        assert retrieved is mock_adapter

    def test_get_adapter_case_insensitive(self):
        """Test that get_adapter is case-insensitive."""
        mock_adapter = Mock(spec=LLMAdapter)
        LLMRegistry.register("test_framework", mock_adapter)
        retrieved = LLMRegistry.get_adapter("TEST_FRAMEWORK")
        assert retrieved is mock_adapter

    def test_get_adapter_not_exists(self):
        """Test retrieving a non-existent adapter returns None."""
        result = LLMRegistry.get_adapter("nonexistent")
        assert result is None

    def test_detect_framework_from_module(self):
        """Test detecting framework from model's module name."""
        # Register a test adapter
        mock_adapter = Mock(spec=LLMAdapter)
        LLMRegistry.register("dspy", mock_adapter)

        # Create a mock model with dspy in its module
        mock_model = Mock()
        mock_model.__class__.__module__ = "dspy.predict.react"

        detected = LLMRegistry.detect(mock_model)
        assert detected == "dspy"

    def test_detect_framework_not_found(self):
        """Test that detect returns None when no framework matches."""
        mock_model = Mock()
        mock_model.__class__.__module__ = "unknown.module"

        detected = LLMRegistry.detect(mock_model)
        assert detected is None

    def test_detect_framework_case_insensitive(self):
        """Test that framework detection is case-insensitive."""
        mock_adapter = Mock(spec=LLMAdapter)
        LLMRegistry.register("langchain", mock_adapter)

        mock_model = Mock()
        mock_model.__class__.__module__ = "LangChain.llms.OpenAI"

        detected = LLMRegistry.detect(mock_model)
        assert detected == "langchain"

    def test_multiple_frameworks_registered(self):
        """Test that multiple frameworks can be registered and retrieved."""
        adapter1 = Mock(spec=LLMAdapter)
        adapter2 = Mock(spec=LLMAdapter)
        adapter3 = Mock(spec=LLMAdapter)

        LLMRegistry.register("dspy", adapter1)
        LLMRegistry.register("langchain", adapter2)
        LLMRegistry.register("langgraph", adapter3)

        assert LLMRegistry.get_adapter("dspy") is adapter1
        assert LLMRegistry.get_adapter("langchain") is adapter2
        assert LLMRegistry.get_adapter("langgraph") is adapter3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
