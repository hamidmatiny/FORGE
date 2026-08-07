# CLI

Entry point: `forge` (Typer).

## Commands

| Command | Phase | Description |
|---------|-------|-------------|
| `ingest` | 1 | Ingest sensor logs into Parquet lake |
| `detect2d` | 2 | 2D object detection |
| `detect3d` | 3 | 3D object detection |
| `track` | 4 | Multi-frame tracking |
| `fuse` | 5 | Multi-sensor fusion |
| `evaluate` | 6 | GT evaluation (evaluation-only) |
| `curate` | 7 | Dataset curation |
| `visualize` | 10 | rerun `.rrd` + MCAP JSON export (offline) |

All unimplemented commands exit with code `1` and reference [KNOWN_GAPS.md](../KNOWN_GAPS.md).

Every command accepts `--local` for single-process mode (Ray wiring arrives in later phases).

## Implementation

`src/forge/cli.py`
