# Known Gaps

Tracked limitations and deferred work. Every unimplemented CLI command references this file.

## Phase 0 (Foundation)

| Item | Target Phase | Notes |
|------|--------------|-------|
| `forge ingest` | Phase 1 | ✅ Done — nuScenes-devkit JSON layout, key-frames only by default |
| `forge detect2d` | Phase 2 | ✅ Done — Faster R-CNN (random-init) + Lightning; see PHASE_2_COMPLETION.md |
| `forge detect3d` | Phase 3 | ✅ Done — PointNet-style encoder (random-init) + Lightning; see PHASE_3_COMPLETION.md |
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
| detect2d model accuracy | Phase 2+ | Faster R-CNN is randomly initialized, not trained on real labels; training loop is a verified-correct smoke test, not a tuned detector. See PHASE_2_COMPLETION.md |
| detect2d class taxonomy | Phase 2+ | `CLASS_NAMES` (background/vehicle/pedestrian/cyclist/traffic_sign) is a placeholder; not yet reconciled against any real labeled dataset |
| detect2d single-image inference | Phase 2+ | No batching in `run_inference` (one image at a time); fine at this scale, revisit if throughput matters later |
| Non-camera detect2d filtering | Phase 2 | `run_inference` silently skips non-`CAM_*` sensors by design; lidar/radar detection is Phase 3 |
| detect3d model accuracy | Phase 3+ | Randomly initialized, trained only on synthetic in-memory point clouds; not a tuned detector. See PHASE_3_COMPLETION.md |
| detect3d fixed query count | Phase 3+ | Predicts exactly `NUM_QUERIES` (4) box slots per point cloud, no variable object count or NMS/matching; a real detector needs proper set prediction (Hungarian matching) or an anchor/heatmap-based head. See DECISIONS.md ADR-014 |
| detect3d BEV/segmentation | Phase 3+ | Boxes only; no explicit BEV feature map or semantic segmentation output, despite the phase name — see ARCHITECTURE.md coverage table for how "BEV" is satisfied at a basic level (3D boxes in the ego/ground frame) |
| detect3d radar | Phase 3+ | Only `LIDAR_*` sensors are processed; radar point clouds are a different (sparser, Doppler-augmented) format and aren't handled |

## Schema Tables (not yet defined)

Additional Parquet tables will be added in their implementing phases. Phase 0 defines **frames v1** only.
