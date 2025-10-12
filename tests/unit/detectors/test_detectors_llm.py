from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from despii.adapters.base import LLMAdapter, LLMResponse
from despii.detectors.llm import PIIInfo, PiiLLM, _llm, llm_pass


class TestPIIInfo:
    """Test PIIInfo pydantic model."""

    def test_pii_info_creation(self):
        """Test creating PIIInfo with valid data."""
        pii = PIIInfo(pii_str="John Doe", label="Name")
        assert pii.pii_str == "John Doe"
        assert pii.label == "Name"

    def test_pii_info_validation(self):
        """Test that PIIInfo validates required fields."""
        with pytest.raises(ValidationError):
            PIIInfo(pii_str="test")  # Missing label


class TestPiiLLM:
    """Test PiiLLM class."""

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_init_with_model_and_adapter(self, mock_registry, mock_settings):
        """Test PiiLLM initialization with a valid model and adapter."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock()
        mock_adapter_instance = Mock(spec=LLMAdapter)
        mock_adapter_cls.return_value = mock_adapter_instance
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()

        assert pii_llm.model is mock_model
        assert pii_llm.framework == "dspy"
        assert pii_llm.adapter is mock_adapter_instance
        mock_registry.detect.assert_called_once_with(mock_model)
        mock_registry.get_adapter.assert_called_once_with("dspy")
        mock_adapter_cls.assert_called_once_with(mock_model)

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_init_with_no_model(self, mock_registry, mock_settings):
        """Test PiiLLM initialization when no model is configured."""
        mock_settings.local_lm = None

        pii_llm = PiiLLM()

        assert pii_llm.model is None
        assert pii_llm.framework is None
        assert pii_llm.adapter is None
        mock_registry.detect.assert_not_called()

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_init_with_undetected_framework(self, mock_registry, mock_settings):
        """Test PiiLLM initialization when framework cannot be detected."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model
        mock_registry.detect.return_value = None

        pii_llm = PiiLLM()

        assert pii_llm.model is mock_model
        assert pii_llm.framework is None
        assert pii_llm.adapter is None
        mock_registry.get_adapter.assert_not_called()

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_init_with_no_adapter_for_framework(self, mock_registry, mock_settings):
        """Test PiiLLM initialization when no adapter exists for detected framework."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model
        mock_registry.detect.return_value = "unknown_framework"
        mock_registry.get_adapter.return_value = None

        pii_llm = PiiLLM()

        assert pii_llm.model is mock_model
        assert pii_llm.framework == "unknown_framework"
        assert pii_llm.adapter is None

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_generate_with_adapter(self, mock_registry, mock_settings):
        """Test generate method with a valid adapter."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        # Setup adapter mock
        mock_adapter = Mock(spec=LLMAdapter)
        mock_response = Mock(spec=LLMResponse)
        mock_response.raw = ['[{"pii_str": "John Doe", "label": "Name"}]']
        mock_adapter.generate.return_value = mock_response

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock(return_value=mock_adapter)
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()
        result = pii_llm.generate("My name is John Doe")

        assert len(result) == 1
        assert isinstance(result[0], PIIInfo)
        assert result[0].pii_str == "John Doe"
        assert result[0].label == "Name"
        mock_adapter.generate.assert_called_once()

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_generate_without_adapter(self, mock_registry, mock_settings):
        """Test generate method returns empty list when no adapter."""
        mock_settings.local_lm = None

        pii_llm = PiiLLM()
        result = pii_llm.generate("My name is John Doe")

        assert result == []

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_generate_with_empty_response(self, mock_registry, mock_settings):
        """Test generate method with empty PII list response."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        mock_adapter = Mock(spec=LLMAdapter)
        mock_response = Mock(spec=LLMResponse)
        mock_response.raw = ["[]"]
        mock_adapter.generate.return_value = mock_response

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock(return_value=mock_adapter)
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()
        result = pii_llm.generate("No PII here")

        assert result == []

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_generate_with_multiple_pii(self, mock_registry, mock_settings):
        """Test generate method with multiple PII items."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        mock_adapter = Mock(spec=LLMAdapter)
        mock_response = Mock(spec=LLMResponse)
        mock_response.raw = [
            '[{"pii_str": "John Doe", "label": "Name"}, {"pii_str": "john@example.com", "label": "Email"}]'
        ]
        mock_adapter.generate.return_value = mock_response

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock(return_value=mock_adapter)
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()
        result = pii_llm.generate("Contact John Doe at john@example.com")

        assert len(result) == 2
        assert result[0].pii_str == "John Doe"
        assert result[0].label == "Name"
        assert result[1].pii_str == "john@example.com"
        assert result[1].label == "Email"

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_generate_passes_kwargs(self, mock_registry, mock_settings):
        """Test that generate passes kwargs to adapter."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        mock_adapter = Mock(spec=LLMAdapter)
        mock_response = Mock(spec=LLMResponse)
        mock_response.raw = ["[]"]
        mock_adapter.generate.return_value = mock_response

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock(return_value=mock_adapter)
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()
        pii_llm.generate("test", temperature=0.7, max_tokens=100)

        # Check that kwargs were passed
        call_args = mock_adapter.generate.call_args
        assert call_args.kwargs.get("temperature") == 0.7
        assert call_args.kwargs.get("max_tokens") == 100

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_generate_injects_text_into_prompt(self, mock_registry, mock_settings):
        """Test that generate correctly injects input text into the prompt."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        mock_adapter = Mock(spec=LLMAdapter)
        mock_response = Mock(spec=LLMResponse)
        mock_response.raw = ["[]"]
        mock_adapter.generate.return_value = mock_response

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock(return_value=mock_adapter)
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()
        test_text = "My name is Alice and my email is alice@example.com"
        pii_llm.generate(test_text)

        # Verify adapter.generate was called with a prompt containing the input text
        mock_adapter.generate.assert_called_once()
        call_args = mock_adapter.generate.call_args
        prompt = call_args.args[0]

        # Check that the input text appears in the prompt
        assert test_text in prompt
        # Check that it appears in the expected format (after "text: ")
        assert f"text: {test_text}" in prompt
        # Verify the placeholder is replaced (should not contain {{INPUT_TEXT}})
        assert "{{INPUT_TEXT}}" not in prompt

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    def test_generate_handles_multiple_calls_with_different_texts(self, mock_registry, mock_settings):
        """Test that generate correctly handles multiple calls with different input texts."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        mock_adapter = Mock(spec=LLMAdapter)
        mock_response = Mock(spec=LLMResponse)
        mock_response.raw = ["[]"]
        mock_adapter.generate.return_value = mock_response

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock(return_value=mock_adapter)
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()

        # First call with text1
        text1 = "First test with name John"
        pii_llm.generate(text1)

        first_call_args = mock_adapter.generate.call_args
        first_prompt = first_call_args.args[0]
        assert text1 in first_prompt
        assert f"text: {text1}" in first_prompt
        assert "{{INPUT_TEXT}}" not in first_prompt

        # Second call with text2
        text2 = "Second test with email bob@example.com"
        pii_llm.generate(text2)

        second_call_args = mock_adapter.generate.call_args
        second_prompt = second_call_args.args[0]
        assert text2 in second_prompt
        assert f"text: {text2}" in second_prompt
        assert "{{INPUT_TEXT}}" not in second_prompt

        # Verify the first text is NOT in the second prompt (no pollution)
        assert text1 not in second_prompt

        # Third call with text3
        text3 = "Third test with phone 555-1234"
        pii_llm.generate(text3)

        third_call_args = mock_adapter.generate.call_args
        third_prompt = third_call_args.args[0]
        assert text3 in third_prompt
        assert f"text: {text3}" in third_prompt
        assert "{{INPUT_TEXT}}" not in third_prompt

        # Verify previous texts are NOT in the third prompt
        assert text1 not in third_prompt
        assert text2 not in third_prompt

        # Verify all three calls happened
        assert mock_adapter.generate.call_count == 3

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    @patch("despii.detectors.llm.logger")
    def test_generate_handles_invalid_json(self, mock_logger, mock_registry, mock_settings):
        """Test that generate handles invalid JSON gracefully."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        # Mock adapter that returns invalid JSON
        mock_adapter = Mock(spec=LLMAdapter)
        mock_response = Mock(spec=LLMResponse)
        mock_response.raw = ["This is not valid JSON"]
        mock_adapter.generate.return_value = mock_response

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock(return_value=mock_adapter)
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()
        result = pii_llm.generate("Test text")

        # Should return empty list instead of raising
        assert result == []

        # Should log debug message
        mock_logger.debug.assert_called_once()
        log_call_args = mock_logger.debug.call_args
        assert "Failed to parse LLM response as JSON" in log_call_args.args[0]
        assert "This is not valid JSON" in str(log_call_args.args[1])

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    @patch("despii.detectors.llm.logger")
    def test_generate_handles_markdown_json(self, mock_logger, mock_registry, mock_settings):
        """Test that generate handles JSON wrapped in markdown code blocks."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        # Mock adapter that returns JSON in markdown
        mock_adapter = Mock(spec=LLMAdapter)
        mock_response = Mock(spec=LLMResponse)
        mock_response.raw = ['```json\n[{"pii_str": "John", "label": "Name"}]\n```']
        mock_adapter.generate.return_value = mock_response

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock(return_value=mock_adapter)
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()
        result = pii_llm.generate("Test text")

        # Should return empty list instead of raising
        assert result == []

        # Should log debug message
        mock_logger.debug.assert_called_once()

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    @patch("despii.detectors.llm.logger")
    def test_generate_handles_incomplete_json(self, mock_logger, mock_registry, mock_settings):
        """Test that generate handles incomplete/malformed JSON."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        # Mock adapter that returns incomplete JSON
        mock_adapter = Mock(spec=LLMAdapter)
        mock_response = Mock(spec=LLMResponse)
        mock_response.raw = ['[{"pii_str": "John", "label": "Name"']  # Missing closing brackets
        mock_adapter.generate.return_value = mock_response

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock(return_value=mock_adapter)
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()
        result = pii_llm.generate("Test text")

        # Should return empty list instead of raising
        assert result == []

        # Should log debug message with the response
        mock_logger.debug.assert_called_once()
        log_call_args = mock_logger.debug.call_args
        assert "Failed to parse LLM response as JSON" in log_call_args.args[0]

    @patch("despii.detectors.llm.settings")
    @patch("despii.detectors.llm.LLMRegistry")
    @patch("despii.detectors.llm.logger")
    def test_generate_handles_json_with_text_prefix(self, mock_logger, mock_registry, mock_settings):
        """Test that generate handles JSON with explanatory text prefix."""
        mock_model = Mock()
        mock_settings.local_lm = mock_model

        # Mock adapter that returns JSON with explanatory text
        mock_adapter = Mock(spec=LLMAdapter)
        mock_response = Mock(spec=LLMResponse)
        mock_response.raw = ['Here is the output: [{"pii_str": "John", "label": "Name"}]']
        mock_adapter.generate.return_value = mock_response

        mock_registry.detect.return_value = "dspy"
        mock_adapter_cls = Mock(return_value=mock_adapter)
        mock_registry.get_adapter.return_value = mock_adapter_cls

        pii_llm = PiiLLM()
        result = pii_llm.generate("Test text")

        # Should return empty list instead of raising
        assert result == []

        # Should log debug message
        mock_logger.debug.assert_called_once()


class TestLLMPass:
    """Test llm_pass function."""

    @patch("despii.detectors.llm._llm")
    def test_llm_pass_with_pii(self, mock_llm_factory):
        """Test llm_pass redacts detected PII."""
        # Mock the _llm factory function
        mock_llm_instance = Mock(spec=PiiLLM)
        mock_pii_1 = PIIInfo(pii_str="John Doe", label="Name")
        mock_pii_2 = PIIInfo(pii_str="john@example.com", label="Email")
        mock_llm_instance.generate.return_value = [mock_pii_1, mock_pii_2]
        mock_llm_factory.return_value = mock_llm_instance

        # Create a real context
        ctx = Mock()
        ctx.text = "Contact John Doe at john@example.com"
        ctx.redact = Mock()

        result = llm_pass(ctx)

        assert result is ctx
        mock_llm_instance.generate.assert_called_once_with("Contact John Doe at john@example.com")
        assert ctx.redact.call_count == 2
        ctx.redact.assert_any_call("John Doe", "Name")
        ctx.redact.assert_any_call("john@example.com", "Email")

    @patch("despii.detectors.llm._llm")
    def test_llm_pass_with_no_pii(self, mock_llm_factory):
        """Test llm_pass when no PII is detected."""
        mock_llm_instance = Mock(spec=PiiLLM)
        mock_llm_instance.generate.return_value = []
        mock_llm_factory.return_value = mock_llm_instance

        ctx = Mock()
        ctx.text = "No PII here"
        ctx.redact = Mock()

        result = llm_pass(ctx)

        assert result is ctx
        mock_llm_instance.generate.assert_called_once_with("No PII here")
        ctx.redact.assert_not_called()

    @patch("despii.detectors.llm._llm")
    def test_llm_pass_preserves_context(self, mock_llm_factory):
        """Test that llm_pass returns the same context object."""
        mock_llm_instance = Mock(spec=PiiLLM)
        mock_llm_instance.generate.return_value = []
        mock_llm_factory.return_value = mock_llm_instance

        ctx = Mock()
        ctx.text = "test"
        ctx.redact = Mock()

        result = llm_pass(ctx)

        assert result is ctx


class TestLLMFactory:
    """Test _llm factory function."""

    @patch("despii.detectors.llm.PiiLLM")
    def test_llm_factory_creates_instance(self, mock_pii_llm_cls):
        """Test that _llm factory creates PiiLLM instance."""
        mock_instance = Mock(spec=PiiLLM)
        mock_pii_llm_cls.return_value = mock_instance

        result = _llm()

        assert result is mock_instance
        mock_pii_llm_cls.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
