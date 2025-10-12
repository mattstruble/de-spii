from despii.adapters.base import LLMAdapter, LLMRegistry, LLMResponse
from despii.adapters.dspy import DSPyAdapter
from despii.adapters.errors import UnsupportedModelInterfaceError
from despii.adapters.langchain import LangChainAdapter
from despii.adapters.langgraph import LangGraphAdapter

__all__ = [
    "LLMAdapter",
    "LLMRegistry",
    "LLMResponse",
    "DSPyAdapter",
    "LangChainAdapter",
    "LangGraphAdapter",
    "UnsupportedModelInterfaceError",
]

LLMRegistry.register("langchain", LangChainAdapter)
LLMRegistry.register("langgraph", LangGraphAdapter)
LLMRegistry.register("dspy", DSPyAdapter)
