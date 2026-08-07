# Visualization export

Phase 10: offline export of `pseudo_labels.parquet` for human review.

## Formats

| `--format` | Output | Notes |
|------------|--------|-------|
| `rerun` | `.rrd` | 3D boxes; skips `camera_only` sentinel geometry |
| `mcap` | `.mcap` | JSON messages on `/forge/pseudo_labels`; all rows |

## CLI

```bash
uv sync --extra visualize --dev
forge visualize --local --format rerun --output review.rrd
```

## Implementation

- `src/forge/visualize/rerun_export.py`
- `src/forge/visualize/mcap_export.py`

See [PHASE_10_COMPLETION.md](../PHASE_10_COMPLETION.md).
