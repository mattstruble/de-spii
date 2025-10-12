import logging
from typing import Any, override

from despii.adapters.base import LLMAdapter, LLMResponse
from despii.adapters.errors import UnsupportedModelInterfaceError

logger = logging.getLogger(__name__)


class DSPyAdapter(LLMAdapter):
    @override
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:  # noqa: ANN401
        logger.debug("DSPyAdapter generating response (prompt length: %d chars)", len(prompt))

        if callable(self.model):
            logger.debug("Using callable interface for DSPy model")
            raw = self.model(prompt)
        elif hasattr(self.model, "generate_text"):
            logger.debug("Using generate_text interface for DSPy model")
            raw = self.model.generate_text(prompt)
        else:
            logger.error("DSPy model %s does not implement required interface", type(self.model))
            raise UnsupportedModelInterfaceError(
                f"DSPy model {type(self.model)} does not implement '__call__' or 'generate_text'."
            )

        text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
        logger.debug("DSPyAdapter response generated (length: %d chars)", len(text))
        return LLMResponse(text=text, raw=raw, framework="dspy", model_name=getattr(self.model, "model", None))
