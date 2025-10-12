# Logging in despii

The despii library follows Python logging best practices for libraries:

1. **No configuration by default** - The library never configures handlers, formatters, or log levels
2. **Standard hierarchy** - Uses `logging.getLogger(__name__)` throughout
3. **Application controls** - The importing application configures all logging

## Controlling despii Logging

Since despii doesn't configure logging, you control it using standard Python logging:

### Set Log Level for All despii Modules

```python
import logging
import despii

# Configure your application's logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set despii to DEBUG
logging.getLogger('despii').setLevel(logging.DEBUG)
```

### Granular Control by Module

```python
import logging

# Configure different levels for different modules
logging.getLogger('despii.adapters').setLevel(logging.DEBUG)  # Verbose adapter logs
logging.getLogger('despii.detectors').setLevel(logging.WARNING)  # Less detector logs
logging.getLogger('despii.detectors.llm').setLevel(logging.INFO)  # Specific detector
```

### Disable despii Logging

```python
import logging

# Disable all despii logging
logging.getLogger('despii').disabled = True

# Or set to CRITICAL to effectively silence it
logging.getLogger('despii').setLevel(logging.CRITICAL + 1)
```

### Filter Specific Loggers

```python
import logging

class NoSpacyFilter(logging.Filter):
    def filter(self, record):
        return 'spacy' not in record.name

logging.getLogger('despii').addFilter(NoSpacyFilter())
```

## Logger Hierarchy

The despii library uses a hierarchical logger structure:

```text
despii/
├── despii.adapters/
│   ├── despii.adapters.base
│   ├── despii.adapters.dspy
│   ├── despii.adapters.langchain
│   └── despii.adapters.langgraph
├── despii.detectors/
│   ├── despii.detectors.llm
│   ├── despii.detectors.spacy
│   ├── despii.detectors.secrets
│   └── despii.detectors.regex_detector
├── despii.core
└── despii.settings
```

## What Gets Logged

despii logs the following information (never logging actual PII text):

### DEBUG Level

- Detection events with labels only (not PII text)
- Model/adapter selection
- Framework detection
- Placeholder creation

### INFO Level

- Summary counts (e.g., "Found 3 PII entities")
- Model loading events
- Detection completion

### WARNING Level

- Missing adapters
- Framework detection failures
- Configuration issues

### ERROR Level

- Model interface errors
- Critical failures

## Privacy & Security

**Important**: despii never logs actual PII text. All logging operations log only:

- Labels (e.g., "EMAIL", "NAME", "SSN")
- Counts (e.g., "3 entities found")
- Lengths (e.g., "secret length: 32")
- Placeholders (e.g., "<PII_EMAIL_1>")

The actual PII values are never written to logs.

## Example: Full Logging Setup

```python
import logging
import sys
import despii

# Set up application logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('app.log')
    ]
)

# Configure despii logging levels
logging.getLogger('despii').setLevel(logging.INFO)
logging.getLogger('despii.adapters').setLevel(logging.DEBUG)

# Use despii
despii.configure(local_lm=my_model)
ctx = despii.core.RedactionContext("test text")
# ... your code ...
```

## Convenience Helper

despii exports a `get_logger()` helper for consistency:

```python
from despii import get_logger

logger = get_logger(__name__)
```

This is equivalent to `logging.getLogger(__name__)`.
