from dataclasses import dataclass
from typing import Any


@dataclass
class LLMResponse:
    text: str
    raw: Any = None
    framework: str = ""
    model_name: str | None = None


class LLMAdapter:
    """Base adapter."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def generate(self, prompt: str, **kwargs) -> LLMResponse:
        raise NotImplementedError


class LLMRegistry:
    """Registry for framework-specific adapters."""

    _registry: dict[str, LLMAdapter] = {}

    @classmethod
    def register(cls, name: str, adapter: LLMAdapter):
        cls._registry[name.lower()] = adapter

    @classmethod
    def get_adapter(cls, name: str) -> LLMAdapter | None:
        return cls._registry.get(name.lower())

    @classmethod
    def detect(cls, model: Any) -> str | None:
        mod = model.__class__.__module__.lower()
        for key in cls._registry.keys():
            if key in mod:
                return key
