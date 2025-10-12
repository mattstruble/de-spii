import pytest
from unittest.mock import Mock

from despii.adapters.base import LLMResponse
from despii.adapters.langgraph import LangGraphAdapter
from despii.adapters.errors import UnsupportedModelInterfaceError


class TestLangGraphAdapter:
    """Test LangGraphAdapter class."""

    def test_adapter_initialization(self):
        """Test that LangGraphAdapter can be initialized with a model."""
        model = Mock()
        adapter = LangGraphAdapter(model)
        assert adapter.model is model

    def test_generate_with_invoke_method_string_response(self):
        """Test generate with model.invoke() returning a string."""
        model = Mock()
        model.invoke = Mock(return_value="Invoke response")

        adapter = LangGraphAdapter(model)
        response = adapter.generate("test prompt")

        model.invoke.assert_called_once_with("test prompt")
        assert isinstance(response, LLMResponse)
        assert response.text == "Invoke response"
        assert response.raw == "Invoke response"
        assert response.framework == "langgraph"

    def test_generate_with_invoke_method_object_response(self):
        """Test generate with model.invoke() returning an object with .content."""
        model = Mock()
        response_obj = Mock()
        response_obj.content = "Content from object"
        model.invoke = Mock(return_value=response_obj)

        adapter = LangGraphAdapter(model)
        response = adapter.generate("test prompt")

        assert response.text == "Content from object"
        assert response.raw is response_obj
        assert response.framework == "langgraph"

    def test_generate_with_callable_fallback(self):
        """Test generate with callable model as fallback."""
        model = Mock()
        model.return_value = "Callable response"
        # Remove invoke to test callable path
        if hasattr(model, "invoke"):
            delattr(model, "invoke")

        adapter = LangGraphAdapter(model)
        response = adapter.generate("test prompt")

        model.assert_called_once_with("test prompt")
        assert response.text == "Callable response"
        assert response.framework == "langgraph"

    def test_generate_prefers_invoke_over_callable(self):
        """Test that invoke is preferred when both are available."""
        model = Mock()
        model.invoke = Mock(return_value="Via invoke")
        model.return_value = "Via callable"

        adapter = LangGraphAdapter(model)
        response = adapter.generate("test prompt")

        model.invoke.assert_called_once()
        model.assert_not_called()
        assert response.text == "Via invoke"

    def test_generate_with_unsupported_model_interface(self):
        """Test that generate raises error for unsupported model interface."""
        # Create an object instance that has no supported methods
        class UnsupportedModel:
            pass

        model = UnsupportedModel()
        adapter = LangGraphAdapter(model)

        with pytest.raises(UnsupportedModelInterfaceError) as exc_info:
            adapter.generate("test prompt")

        assert "does not implement 'invoke' or '__call__'" in str(exc_info.value)

    def test_generate_extracts_model_name_from_name_attribute(self):
        """Test that generate extracts model_name from model.name attribute."""
        model = Mock()
        model.invoke = Mock(return_value="response")
        model.name = "claude-3-opus"

        adapter = LangGraphAdapter(model)
        response = adapter.generate("test prompt")

        assert response.model_name == "claude-3-opus"

    def test_generate_model_name_none_when_not_present(self):
        """Test that model_name is None when model doesn't have name attribute."""
        model = Mock()
        model.invoke = Mock(return_value="response")
        if hasattr(model, "name"):
            delattr(model, "name")

        adapter = LangGraphAdapter(model)
        response = adapter.generate("test prompt")

        assert response.model_name is None

    def test_generate_with_object_without_content_attribute(self):
        """Test generate with object that doesn't have .content attribute."""
        # Create a custom object without .content attribute
        class ResponseObj:
            def __str__(self):
                return "String representation"

        model = Mock()
        response_obj = ResponseObj()
        model.invoke = Mock(return_value=response_obj)

        adapter = LangGraphAdapter(model)
        response = adapter.generate("test prompt")

        # Should fall back to str(raw)
        assert response.text == "String representation"
        assert response.raw is response_obj

    def test_generate_does_not_pass_kwargs(self):
        """Test that LangGraphAdapter does not pass kwargs to invoke (per implementation)."""
        model = Mock()
        model.invoke = Mock(return_value="response")

        adapter = LangGraphAdapter(model)
        # Note: kwargs are accepted but not passed in the current implementation
        adapter.generate("test prompt", temperature=0.8)

        # Should only be called with prompt, no kwargs
        model.invoke.assert_called_once_with("test prompt")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

