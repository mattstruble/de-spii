"""Integration tests for LLM detector with real models and full flow."""

import pytest

pytest.importorskip("dspy")
pytest.importorskip("langchain")
pytest.importorskip("langgraph")

import dspy  # noqa: E402
from langchain_ollama import ChatOllama  # noqa: E402

from despii.core import RedactionContext  # noqa: E402
from despii.detectors.llm import PiiLLM, llm_pass  # noqa: E402
from despii.settings import settings  # noqa: E402

pytestmark = pytest.mark.integration


class TestLLMDetectorIntegration:
    """Integration tests for LLM detector with real models."""

    def test_llm_detector_with_dspy_model(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Test LLM detector with DSPy model configured via settings.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        """
        # Configure DSPy model via settings
        model = dspy.LM(f"ollama_chat/{ollama_model_name}", api_base="http://localhost:11434", api_key="")
        settings.configure(local_lm=model)

        # Create PiiLLM instance
        detector = PiiLLM()

        # Verify framework detection
        assert detector.framework == "dspy"
        assert detector.adapter is not None

        # Test PII detection
        text = "My name is Alice and my email is alice@test.com"
        results = detector.generate(text)

        # Should return a list of PIIInfo objects
        assert isinstance(results, list)
        # May or may not detect PII depending on model, but should not crash
        for item in results:
            assert hasattr(item, "pii_str")
            assert hasattr(item, "label")

    def test_llm_detector_with_langchain_model(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Test LLM detector with LangChain model configured via settings.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        """
        # Configure LangChain model via settings
        model = ChatOllama(model=ollama_model_name, base_url="http://localhost:11434")
        settings.configure(local_lm=model)

        # Create PiiLLM instance
        detector = PiiLLM()

        # Verify framework detection
        assert detector.framework == "langchain"
        assert detector.adapter is not None

        # Test PII detection
        text = "My name is Bob and my phone is 555-0123"
        results = detector.generate(text)

        # Should return a list
        assert isinstance(results, list)
        for item in results:
            assert hasattr(item, "pii_str")
            assert hasattr(item, "label")

    def test_llm_detector_with_langgraph_model(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Test LLM detector with LangGraph model configured via settings.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        """
        # Configure LangGraph model (uses LangChain ChatOllama) via settings
        model = ChatOllama(model=ollama_model_name, base_url="http://localhost:11434")
        settings.configure(local_lm=model)

        # Create PiiLLM instance
        detector = PiiLLM()

        # Verify framework detection (should detect as langchain)
        assert detector.framework in {"langchain", "langgraph"}
        assert detector.adapter is not None

        # Test PII detection
        text = "Contact Charlie at charlie@example.org"
        results = detector.generate(text)

        # Should return a list
        assert isinstance(results, list)
        for item in results:
            assert hasattr(item, "pii_str")
            assert hasattr(item, "label")

    def test_llm_pass_full_flow_with_dspy(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Test full llm_pass flow with DSPy model.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        """
        # Configure model
        model = dspy.LM(f"ollama_chat/{ollama_model_name}", api_base="http://localhost:11434", api_key="")
        settings.configure(local_lm=model)

        # Create redaction context
        text = "My name is John Doe and my email is john.doe@company.com. I live at 123 Main St."
        ctx = RedactionContext(text)

        # Run LLM pass
        result_ctx = llm_pass(ctx)

        # Should return a RedactionContext
        assert isinstance(result_ctx, RedactionContext)
        # Redacted text may differ from original if PII was detected
        assert isinstance(result_ctx.text, str)
        # May have redactions depending on model performance
        # Just verify it doesn't crash and returns valid structure

    def test_llm_pass_full_flow_with_langchain(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Test full llm_pass flow with LangChain model.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        """
        # Configure model
        model = ChatOllama(model=ollama_model_name, base_url="http://localhost:11434")
        settings.configure(local_lm=model)

        # Create redaction context
        text = "Call me at 555-1234 or email jane@example.com"
        ctx = RedactionContext(text)

        # Run LLM pass
        result_ctx = llm_pass(ctx)

        # Should return a RedactionContext
        assert isinstance(result_ctx, RedactionContext)
        # Redacted text may differ from original if PII was detected
        assert isinstance(result_ctx.text, str)

    def test_llm_detector_handles_no_pii_text(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Test LLM detector with text containing no PII.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        """
        model = dspy.LM(f"ollama_chat/{ollama_model_name}", api_base="http://localhost:11434", api_key="")
        settings.configure(local_lm=model)

        detector = PiiLLM()
        text = "The weather is nice today. It is sunny and warm."
        results = detector.generate(text)

        # Should return a list (may be empty or have false positives)
        assert isinstance(results, list)

    def test_llm_detector_with_multiple_pii_types(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Test LLM detector with text containing multiple PII types.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        """
        model = ChatOllama(model=ollama_model_name, base_url="http://localhost:11434")
        settings.configure(local_lm=model)

        detector = PiiLLM()
        text = """
        Name: Alice Johnson
        Email: alice.johnson@company.com
        Phone: (555) 123-4567
        SSN: 123-45-6789
        Address: 456 Oak Avenue, Springfield
        """
        results = detector.generate(text)

        # Should return a list of detected PII
        assert isinstance(results, list)
        # With multiple clear PII examples, should detect at least something
        # (though we can't guarantee exact detection with LLMs)
        for item in results:
            assert hasattr(item, "pii_str")
            assert hasattr(item, "label")
            assert isinstance(item.pii_str, str)
            assert isinstance(item.label, str)

    def test_llm_detector_repeated_calls_different_inputs(self, skip_if_no_ollama, ollama_model_name):  # noqa: ARG002
        """Test that LLM detector handles multiple calls with different inputs correctly.

        :param skip_if_no_ollama: Fixture to skip if Ollama not available
        :param ollama_model_name: Name of Ollama model to use
        """
        model = dspy.LM(f"ollama_chat/{ollama_model_name}", api_base="http://localhost:11434", api_key="")
        settings.configure(local_lm=model)

        detector = PiiLLM()

        # First call
        results1 = detector.generate("My name is Alice")
        assert isinstance(results1, list)

        # Second call with different text
        results2 = detector.generate("Contact Bob at bob@test.com")
        assert isinstance(results2, list)

        # Third call
        results3 = detector.generate("The quick brown fox")
        assert isinstance(results3, list)

        # All should be independent
        # (we can't assert exact results, but they should all complete)
