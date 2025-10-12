import functools
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

    def __init__(self, model: Any) -> None:  # noqa: ANN401
        self.model = model

    def generate(self, prompt: str, **kwargs: Any) -> LLMResponse:  # noqa: ANN401
        """Generate a response from the model.

        :param prompt: The input prompt
        :param kwargs: Additional model-specific parameters
        :return: LLMResponse containing the generated text
        """
        raise NotImplementedError


class LLMRegistry:
    """Registry for framework-specific adapters."""

    _registry: dict[str, LLMAdapter] = {}

    @classmethod
    def register(cls, name: str, adapter: LLMAdapter) -> None:
        """Register an adapter for a framework.

        :param name: Framework name
        :param adapter: Adapter instance
        """
        cls._registry[name.lower()] = adapter

    @classmethod
    def get_adapter(cls, name: str) -> LLMAdapter | None:
        """Retrieve an adapter by framework name.

        :param name: Framework name
        :return: Adapter instance or None
        """
        return cls._registry.get(name.lower())

    @classmethod
    @functools.cache
    def detect(cls, model: Any) -> str | None:  # noqa: ANN401
        """Detect framework from model's module name.

        :param model: Model instance
        :return: Framework name or None
        """
        mod = model.__class__.__module__.lower()
        for key in cls._registry.keys():
            if key in mod:
                return key
