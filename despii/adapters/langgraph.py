import logging
from typing import Any

from despii.adapters.base import LLMAdapter, LLMResponse
from despii.adapters.errors import UnsupportedModelInterfaceError

logger = logging.getLogger(__name__)


class LangGraphAdapter(LLMAdapter):
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:  # noqa: ANN401
        """Generate a response using LangGraph model."""
        logger.debug("LangGraphAdapter generating response (prompt length: %d chars)", len(prompt))

        if hasattr(self.model, "invoke"):
            logger.debug("Using invoke interface for LangGraph model")
            raw = self.model.invoke(prompt)
        elif callable(self.model):
            logger.debug("Using callable interface for LangGraph model")
            raw = self.model(prompt)
        else:
            logger.error("LangGraph model %s does not implement required interface", type(self.model))
            raise UnsupportedModelInterfaceError(
                f"LangGraph model {type(self.model)} does not implement 'invoke' or '__call__'."
            )

        text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        logger.debug("LangGraphAdapter response generated (length: %d chars)", len(text))
        return LLMResponse(text=text, raw=raw, framework="langgraph", model_name=getattr(self.model, "name", None))
