# Known Gaps

Tracked limitations and deferred work. Every unimplemented CLI command references this file.

## Phase 0 (Foundation)

| Item | Target Phase | Notes |
|------|--------------|-------|
| `forge ingest` | Phase 1 | nuScenes-mini ingestion into Parquet data lake |
| `forge detect2d` | Phase 2 | 2D detection; requires `[detect2d]` extras (torch, lightning) |
| `forge detect3d` | Phase 3 | 3D detection / BEV; requires `[detect3d]` extras |
| `forge track` | Phase 4 | Multi-object tracking across frames |
| `forge fuse` | Phase 5 | Multi-sensor fusion |
| `forge label` | Phase 6 | Active-learning selection + pseudo-labeling; confidence-gated review queue |
| `forge evaluate` | Phase 7 | GT comparison (nuScenes GT evaluation-only) + MLflow/W&B metric logging |
| `forge curate` | Phase 8 | LanceDB dedup/search index; dataset curation and export |
| Ray distributed execution | Phase 9 | `--local` flag reserved on all commands; Ray backend lands with cost-safety ADR (no real cluster in CI) |
| Terraform AWS lake (S3/Glue/Athena) | Phase 9 | Applied out-of-band only, same policy as Vulcan/PRISM/hydra-data-factory |
| `forge visualize` | Phase 10 | rerun.io / Foxglove MCAP / FiftyOne; requires `[viz]` extras |
| Productionization docs/runbook | Phase 11 | Not started |
| MLflow / W&B wiring | Phase 7 | Settings stub exists; no tracking yet |
| Hydra pipeline configs | Phase 1+ | Using pydantic-settings only in Phase 0 |
| DVC dataset versioning | Phase 1 | Not configured in Phase 0 |
| LanceDB vector index | Phase 8 | Not configured in Phase 0 |
| Real nuScenes fixture | Phase 1 | Synthetic `tests/fixtures/mini_lake/` until ingest lands |

## Schema Tables (not yet defined)

Additional Parquet tables will be added in their implementing phases. Phase 0 defines **frames v1** only.
