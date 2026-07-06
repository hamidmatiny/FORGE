# Known Gaps

Tracked limitations and deferred work. Every unimplemented CLI command references this file.

## Phase 0 (Foundation)

| Item | Target Phase | Notes |
|------|--------------|-------|
| `forge ingest` | Phase 1 | nuScenes-mini ingestion into Parquet data lake |
| `forge detect2d` | Phase 2 | 2D detection; requires `[detect2d]` extras (torch, etc.) |
| `forge detect3d` | Phase 3 | 3D detection; requires `[detect3d]` extras |
| `forge track` | Phase 4 | Multi-object tracking across frames |
| `forge fuse` | Phase 5 | Multi-sensor fusion |
| `forge evaluate` | Phase 6 | GT comparison (nuScenes GT evaluation-only) |
| `forge curate` | Phase 7 | Dataset curation and export |
| `forge visualize` | Phase 8 | rerun.io / Foxglove MCAP visualization; requires `[viz]` extras |
| MLflow wiring | Phase 2+ | Settings stub exists; no tracking yet |
| Hydra pipeline configs | Phase 1+ | Using pydantic-settings only in Phase 0 |
| Ray distributed execution | Phase 2+ | `--local` flag reserved on all commands |
| DVC dataset versioning | Phase 1 | Not configured in Phase 0 |
| Real nuScenes fixture | Phase 1 | Synthetic `tests/fixtures/mini_lake/` until ingest lands |

## Schema Tables (not yet defined)

Additional Parquet tables will be added in their implementing phases. Phase 0 defines **frames v1** only.
