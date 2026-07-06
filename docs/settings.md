# Settings

Configuration via [pydantic-settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/) with `FORGE_` environment prefix.

## Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `FORGE_DATA_LAKE_ROOT` | `data/lake` | Parquet data lake root |
| `FORGE_MLFLOW_URI` | `file:./mlruns` | MLflow tracking URI (Phase 2+) |
| `FORGE_LOG_LEVEL` | `INFO` | structlog level |

Optional `.env` file is loaded automatically.

## Usage

```python
from forge.settings import get_settings

settings = get_settings()
print(settings.data_lake_root)
```

## Implementation

`src/forge/settings.py`
