from typing import Any, override

from despii.adapters.base import LLMAdapter, LLMResponse
from despii.adapters.errors import UnsupportedModelInterfaceError


class DSPyAdapter(LLMAdapter):
    @override
    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:  # noqa: ANN401
        if callable(self.model):
            raw = self.model(prompt)
        elif hasattr(self.model, "generate_text"):
            raw = self.model.generate_text(prompt)
        else:
            raise UnsupportedModelInterfaceError(
                f"DSPy model {type(self.model)} does not implement '__call__' or 'generate_text'."
            )

        text = raw if isinstance(raw, str) else getattr(raw, "text", str(raw))
        return LLMResponse(text=text, raw=raw, framework="dspy", model_name=getattr(self.model, "model", None))
