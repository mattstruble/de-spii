from unittest.mock import Mock

import pytest

from despii.adapters.base import LLMResponse
from despii.adapters.errors import UnsupportedModelInterfaceError
from despii.adapters.langchain import LangChainAdapter


class TestLangChainAdapter:
    """Test LangChainAdapter class."""

    def test_adapter_initialization(self):
        """Test that LangChainAdapter can be initialized with a model."""
        model = Mock()
        adapter = LangChainAdapter(model)
        assert adapter.model is model

    def test_generate_with_invoke_method_string_response(self):
        """Test generate with model.invoke() returning a string."""
        model = Mock()
        model.invoke = Mock(return_value="Invoke response")

        adapter = LangChainAdapter(model)
        response = adapter.generate("test prompt", temperature=0.7)

        model.invoke.assert_called_once_with("test prompt", temperature=0.7)
        assert isinstance(response, LLMResponse)
        assert response.text == "Invoke response"
        assert response.raw == "Invoke response"
        assert response.framework == "langchain"

    def test_generate_with_invoke_method_object_response(self):
        """Test generate with model.invoke() returning an object with .content."""
        model = Mock()
        response_obj = Mock()
        response_obj.content = "Content from object"
        model.invoke = Mock(return_value=response_obj)

        adapter = LangChainAdapter(model)
        response = adapter.generate("test prompt")

        assert response.text == "Content from object"
        assert response.raw is response_obj
        assert response.framework == "langchain"

    def test_generate_with_predict_method(self):
        """Test generate with model.predict() method."""
        model = Mock()
        model.predict = Mock(return_value="Predict response")
        # Remove invoke to test predict path
        if hasattr(model, "invoke"):
            delattr(model, "invoke")

        adapter = LangChainAdapter(model)
        response = adapter.generate("test prompt", temperature=0.5)

        model.predict.assert_called_once_with("test prompt", temperature=0.5)
        assert response.text == "Predict response"
        assert response.framework == "langchain"

    def test_generate_with_callable_fallback(self):
        """Test generate with callable model as fallback."""
        model = Mock()
        model.return_value = "Callable response"
        # Remove invoke and predict
        if hasattr(model, "invoke"):
            delattr(model, "invoke")
        if hasattr(model, "predict"):
            delattr(model, "predict")

        adapter = LangChainAdapter(model)
        response = adapter.generate("test prompt")

        model.assert_called_once_with("test prompt")
        assert response.text == "Callable response"
        assert response.framework == "langchain"

    def test_generate_prefers_invoke_over_predict(self):
        """Test that invoke is preferred when both invoke and predict are available."""
        model = Mock()
        model.invoke = Mock(return_value="Via invoke")
        model.predict = Mock(return_value="Via predict")

        adapter = LangChainAdapter(model)
        response = adapter.generate("test prompt")

        model.invoke.assert_called_once()
        model.predict.assert_not_called()
        assert response.text == "Via invoke"

    def test_generate_prefers_predict_over_callable(self):
        """Test that predict is preferred over callable."""
        model = Mock()
        model.return_value = "Via callable"
        model.predict = Mock(return_value="Via predict")
        if hasattr(model, "invoke"):
            delattr(model, "invoke")

        adapter = LangChainAdapter(model)
        response = adapter.generate("test prompt")

        model.predict.assert_called_once()
        model.assert_not_called()
        assert response.text == "Via predict"

    def test_generate_with_unsupported_model_interface(self):
        """Test that generate raises error for unsupported model interface."""
        # Create an object instance that has no supported methods
        class UnsupportedModel:
            pass

        model = UnsupportedModel()
        adapter = LangChainAdapter(model)

        with pytest.raises(UnsupportedModelInterfaceError) as exc_info:
            adapter.generate("test prompt")

        assert "does not implement 'invoke', 'predict', or '__call__'" in str(exc_info.value)

    def test_generate_extracts_model_name(self):
        """Test that generate extracts model_name from model.model_name attribute."""
        model = Mock()
        model.invoke = Mock(return_value="response")
        model.model_name = "gpt-4"

        adapter = LangChainAdapter(model)
        response = adapter.generate("test prompt")

        assert response.model_name == "gpt-4"

    def test_generate_model_name_none_when_not_present(self):
        """Test that model_name is None when model doesn't have model_name attribute."""
        model = Mock()
        model.invoke = Mock(return_value="response")
        if hasattr(model, "model_name"):
            delattr(model, "model_name")

        adapter = LangChainAdapter(model)
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

        adapter = LangChainAdapter(model)
        response = adapter.generate("test prompt")

        # Should fall back to str(raw)
        assert response.text == "String representation"
        assert response.raw is response_obj

    def test_generate_passes_kwargs_to_invoke(self):
        """Test that kwargs are properly passed to invoke method."""
        model = Mock()
        model.invoke = Mock(return_value="response")

        adapter = LangChainAdapter(model)
        adapter.generate("test prompt", temperature=0.8, max_tokens=100, top_p=0.9)

        model.invoke.assert_called_once_with(
            "test prompt", temperature=0.8, max_tokens=100, top_p=0.9
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
