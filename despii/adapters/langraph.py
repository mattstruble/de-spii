from despii.adapters.base import LLMAdapter, LLMResponse
from despii.adapters.errors import UnsupportedModelInterfaceError


class LangGraphAdapter(LLMAdapter):
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if hasattr(self.model, "invoke"):
            raw = self.model.invoke(prompt)
        elif callable(self.model):
            raw = self.model(prompt)
        else:
            raise UnsupportedModelInterfaceError(
                f"LangGraph model {type(self.model)} does not implement 'invoke' or '__call__'."
            )

        text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        return LLMResponse(text=text, raw=raw, framework="langgraph", model_name=getattr(self.model, "name", None))
