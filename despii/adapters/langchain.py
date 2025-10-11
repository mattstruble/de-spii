from despii.adapters.base import LLMAdapter, LLMResponse
from despii.adapters.errors import UnsupportedModelInterfaceError


class LangChainAdapter(LLMAdapter):
    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        if hasattr(self.model, "invoke"):
            raw = self.model.invoke(prompt, **kwargs)
        elif hasattr(self.model, "predict"):
            raw = self.model.predict(prompt, **kwargs)
        elif callable(self.model):
            raw = self.model(prompt)
        else:
            raise UnsupportedModelInterfaceError(
                f"LangChain model {type(self.model)} does not implement 'invoke', 'predict', or '__call__'."
            )

        text = raw if isinstance(raw, str) else getattr(raw, "content", str(raw))
        return LLMResponse(
            text=text, raw=raw, framework="langchain", model_name=getattr(self.model, "model_name", None)
        )
