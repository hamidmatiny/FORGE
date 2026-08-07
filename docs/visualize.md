# Visualization export

Phase 10: offline export of `pseudo_labels.parquet` for human review.

## Formats

| `--format` | Output | Notes |
|------------|--------|-------|
| `rerun` | `.rrd` | 3D boxes; skips `camera_only` sentinel geometry |
| `mcap` | `.mcap` | JSON messages on `/forge/pseudo_labels`; all rows |

Default output (no `--output`): `<lake>/visualize_export.rrd` or `visualize_export.mcap`.

## CLI

```bash
uv sync --extra visualize --dev
forge visualize --local --format rerun
```

## Implementation

- `src/forge/visualize/rerun_export.py`
- `src/forge/visualize/mcap_export.py`

See [PHASE_10_COMPLETION.md](../PHASE_10_COMPLETION.md).
