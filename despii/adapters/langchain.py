import logging
from typing import Any

from despii.adapters.base import LLMAdapter, LLMResponse
from despii.adapters.errors import UnsupportedModelInterfaceError

logger = logging.getLogger(__name__)


class LangChainAdapter(LLMAdapter):
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:  # noqa: ANN401
        """Generate a response using LangChain model."""
        logger.debug("LangChainAdapter generating response (prompt length: %d chars)", len(prompt))

        if hasattr(self.model, "invoke"):
            logger.debug("Using invoke interface for LangChain model")
            raw = self.model.invoke(prompt, **kwargs)
        elif hasattr(self.model, "predict"):
            logger.debug("Using predict interface for LangChain model")
            raw = self.model.predict(prompt, **kwargs)
        elif callable(self.model):
            logger.debug("Using callable interface for LangChain model")
            raw = self.model(prompt)
        else:
            logger.error("LangChain model %s does not implement required interface", type(self.model))
            raise UnsupportedModelInterfaceError(
                f"LangChain model {type(self.model)} does not implement 'invoke', 'predict', or '__call__'."
            )

        text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        logger.debug("LangChainAdapter response generated (length: %d chars)", len(text))
        return LLMResponse(
            text=text, raw=raw, framework="langchain", model_name=getattr(self.model, "model_name", None)
        )
