# Integration Tests

Integration tests for despii that verify functionality with real LLM models and services.

## Requirements

### Ollama Setup

Integration tests require [Ollama](https://ollama.ai/) to be installed and running locally with the `llama3:8b` model.

**Installation:**

```bash
# macOS/Linux
curl -fsSL https://ollama.ai/install.sh | sh

# Or via Homebrew on macOS
brew install ollama
```

**Setup llama3:8b model:**

```bash
ollama pull llama3:8b
```

**Start Ollama server:**

```bash
ollama serve
```

The server should be running at `http://localhost:11434` (default).

## Running Integration Tests

### Run Only Integration Tests

```bash
just test-integration
```

Or directly with pytest:

```bash
poetry run pytest tests/integration -m integration
```

### Run All Tests (Unit + Integration)

```bash
just test-all
```

### Run Specific Integration Test Files

```bash
# Test DSPy adapter
poetry run pytest tests/integration/adapters/test_dspy_integration.py -m integration

# Test LangChain adapter
poetry run pytest tests/integration/adapters/test_langchain_integration.py -m integration

# Test LangGraph adapter
poetry run pytest tests/integration/adapters/test_langgraph_integration.py -m integration

# Test full LLM detector flow
poetry run pytest tests/integration/detectors/test_llm_integration.py -m integration
```

## Test Coverage

### Adapter Tests

- **DSPy Integration** (`test_dspy_integration.py`):
  - Verify DSPy adapter works with real Ollama model
  - Test response generation and parsing
  - Test PII detection prompts
  - Verify raw response preservation

- **LangChain Integration** (`test_langchain_integration.py`):
  - Verify LangChain adapter works with ChatOllama
  - Test response generation
  - Test PII detection prompts
  - Test kwargs passing (temperature, max_tokens, etc.)

- **LangGraph Integration** (`test_langgraph_integration.py`):
  - Verify LangGraph adapter works with ChatOllama
  - Test invoke method functionality
  - Test PII detection prompts

### Detector Tests

- **LLM Detector Flow** (`test_llm_integration.py`):
  - Test PiiLLM with each adapter (DSPy, LangChain, LangGraph)
  - Verify framework detection via settings configuration
  - Test full `llm_pass()` flow with RedactionContext
  - Test with various PII types and edge cases
  - Verify multiple sequential calls work correctly

## Skipping Tests

Integration tests automatically skip if Ollama is not available:

```python
pytest.skip("Ollama with llama3:8b model not available")
```

No manual configuration needed - tests will gracefully skip if requirements aren't met.

## Notes

- Integration tests are slower than unit tests (require actual LLM calls)
- Tests use `timeout=300` seconds by default
- Ollama responses may vary, so tests focus on structure rather than exact content
- Tests verify the integration works without asserting specific PII detection accuracy
  (LLM behavior can be non-deterministic)

## Troubleshooting

**Tests are skipped:**

- Ensure Ollama is running: `ollama serve`
- Verify llama3:8b is installed: `ollama list`
- Check Ollama is accessible: `curl http://localhost:11434`

**Tests timeout:**

- First run may be slow (model loading)
- Check Ollama server logs for errors
- Increase timeout if needed for slower systems

**Import errors:**

- Run `poetry install` to install all dependencies
- Ensure dspy, langchain, and langgraph are installed
