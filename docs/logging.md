# Logging

Structured JSON logging via [structlog](https://www.structlog.org/).

## CLI metadata

Every CLI invocation logs:

- `command` — subcommand name
- `args` — parsed arguments
- `version` — package version
- `git_sha` — short git SHA (or `unknown`)

## Usage

```python
from forge.logging import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger("forge.mystage")
logger.info("stage_complete", frames=42, fps=12.5)
```

## Implementation

`src/forge/logging.py`
