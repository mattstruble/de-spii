import pytest
from unittest.mock import Mock

from despii.adapters.base import LLMResponse
from despii.adapters.dspy import DSPyAdapter
from despii.adapters.errors import UnsupportedModelInterfaceError


class TestDSPyAdapter:
    """Test DSPyAdapter class."""

    def test_adapter_initialization(self):
        """Test that DSPyAdapter can be initialized with a model."""
        model = Mock()
        adapter = DSPyAdapter(model)
        assert adapter.model is model

    def test_generate_with_callable_model_string_response(self):
        """Test generate with a callable model that returns a string."""
        model = Mock()
        model.return_value = "Generated text response"

        adapter = DSPyAdapter(model)
        response = adapter.generate("test prompt")

        model.assert_called_once_with("test prompt")
        assert isinstance(response, LLMResponse)
        assert response.text == "Generated text response"
        assert response.raw == "Generated text response"
        assert response.framework == "dspy"

    def test_generate_with_callable_model_object_response(self):
        """Test generate with a callable model that returns an object with .text attribute."""
        model = Mock()
        response_obj = Mock()
        response_obj.text = "Object text response"
        model.return_value = response_obj

        adapter = DSPyAdapter(model)
        response = adapter.generate("test prompt")

        assert response.text == "Object text response"
        assert response.raw is response_obj
        assert response.framework == "dspy"

    def test_generate_with_generate_text_method(self):
        """Test generate with a model that has generate_text method."""
        # Create a non-callable object with generate_text method
        class ModelWithGenerateText:
            def generate_text(self, prompt):
                return "Generated via method"

        model = ModelWithGenerateText()
        # Mock the generate_text method to track calls
        model.generate_text = Mock(return_value="Generated via method")

        adapter = DSPyAdapter(model)
        response = adapter.generate("test prompt")

        model.generate_text.assert_called_once_with("test prompt")
        assert response.text == "Generated via method"
        assert response.raw == "Generated via method"
        assert response.framework == "dspy"

    def test_generate_prefers_callable_over_generate_text(self):
        """Test that callable interface is preferred when both are available."""
        model = Mock()
        model.return_value = "Via callable"
        model.generate_text = Mock(return_value="Via generate_text")

        adapter = DSPyAdapter(model)
        response = adapter.generate("test prompt")

        # Should call the model directly, not generate_text
        model.assert_called_once_with("test prompt")
        model.generate_text.assert_not_called()
        assert response.text == "Via callable"

    def test_generate_with_unsupported_model_interface(self):
        """Test that generate raises error for unsupported model interface."""
        # Create an object instance (not a Mock) that has no callable or generate_text
        class UnsupportedModel:
            pass

        model = UnsupportedModel()
        adapter = DSPyAdapter(model)

        with pytest.raises(UnsupportedModelInterfaceError) as exc_info:
            adapter.generate("test prompt")

        assert "does not implement '__call__' or 'generate_text'" in str(exc_info.value)

    def test_generate_extracts_model_name(self):
        """Test that generate extracts model name from model.model attribute."""
        model = Mock()
        model.return_value = "response"
        model.model = "gpt-4"

        adapter = DSPyAdapter(model)
        response = adapter.generate("test prompt")

        assert response.model_name == "gpt-4"

    def test_generate_model_name_none_when_not_present(self):
        """Test that model_name is None when model doesn't have model attribute."""
        model = Mock()
        model.return_value = "response"
        # Explicitly remove model attribute if it exists
        if hasattr(model, "model"):
            delattr(model, "model")

        adapter = DSPyAdapter(model)
        response = adapter.generate("test prompt")

        assert response.model_name is None

    def test_generate_with_object_without_text_attribute(self):
        """Test generate with object that doesn't have .text attribute."""
        # Create a custom object without .text attribute
        class ResponseObj:
            def __str__(self):
                return "String representation"

        model = Mock()
        response_obj = ResponseObj()
        model.return_value = response_obj

        adapter = DSPyAdapter(model)
        response = adapter.generate("test prompt")

        # Should fall back to str(raw)
        assert response.text == "String representation"
        assert response.raw is response_obj


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

