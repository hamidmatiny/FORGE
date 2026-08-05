# Known Gaps

Tracked limitations and deferred work. Every unimplemented CLI command references this file.

## Phase 0 (Foundation)

| Item | Target Phase | Notes |
|------|--------------|-------|
| `forge ingest` | Phase 1 | ✅ Done — nuScenes-devkit JSON layout, key-frames only by default |
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
| Hydra pipeline configs | Phase 1 | ✅ Done — Compose API loader in `forge/config.py`, `conf/ingest.yaml` |
| DVC dataset versioning | Phase 1 | ✅ Scaffolded — `dvc.yaml` ingest stage, local relative-path remote; no real data tracked yet (needs a real nuScenes-mini run, see `docs/runbooks/ingest-real-nuscenes.md`) |
| LanceDB vector index | Phase 8 | Not configured |
| Non-keyframe sweep ingestion | Phase 1+ | `--all-sweeps` flag exists and is tested, but downstream phases (2+) assume key-frames only until noted otherwise |
| Incremental/append ingest | Phase 1+ | Current `forge ingest` overwrites the lake tables on every run; append/merge semantics deferred |
| Real nuScenes-mini run | Manual | Never executed in CI (non-commercial license, ~4 GB); `tests/fixtures/nuscenes_mini_synthetic/` mirrors the real JSON layout for all automated tests |

## Schema Tables (not yet defined)

Additional Parquet tables will be added in their implementing phases. Phase 0 defines **frames v1** only.
