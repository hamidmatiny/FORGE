# Known Gaps

Tracked limitations and deferred work. Every unimplemented CLI command references this file.

## Phase 0 (Foundation)

| Item | Target Phase | Notes |
|------|--------------|-------|
| `forge ingest` | Phase 1 | ✅ Done — nuScenes-devkit JSON layout, key-frames only by default |
| `forge detect2d` | Phase 2 | ✅ Done — Faster R-CNN (random-init) + Lightning; see PHASE_2_COMPLETION.md |
| `forge detect3d` | Phase 3 | ✅ Done — PointNet-style encoder (random-init) + Lightning; see PHASE_3_COMPLETION.md |
| `forge track` | Phase 4 | ✅ Done — SORT-style Kalman filter + Hungarian IoU; see PHASE_4_COMPLETION.md |
| `forge fuse` | Phase 5 | ✅ Done — calibrated projection + IoU association; see PHASE_5_COMPLETION.md |
| `forge label` | Phase 6 | ✅ Done — trust scoring + entropy-based review priority; see PHASE_6_COMPLETION.md |
| `forge evaluate` | Phase 7 | ✅ Done — BEV distance matching, precision/recall/mAP, MLflow+W&B; see PHASE_7_COMPLETION.md |
| `forge curate` | Phase 8 | ✅ Done — LanceDB near-duplicate search over a geometric feature vector; see PHASE_8_COMPLETION.md |
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
| track 2D-only | Phase 4+ | Tracking runs over `detections_2d` only; 3D tracking (over `detections_3d`, needing 3D IoU) isn't implemented |
| track motion model | Phase 4+ | Constant-velocity Kalman filter; doesn't model turns, occluded re-identification via appearance, or camera-motion compensation (ego-motion isn't factored in despite `ego_pose` existing in the lake since Phase 1) |
| track re-identification | Phase 4+ | A track that's retired (exceeds `max_age`) and reappears gets a brand-new ID; no appearance-based re-ID to recover the old identity |
| fuse synchronization | Phase 5+ | Camera/lidar pairing assumes exact `(scene_id, timestamp_us)` equality; real nuScenes samples are nominally but not bit-exactly synchronized, so real-world timestamps would need a nearest-match window, not exact equality |
| fuse doesn't fuse tracks | Phase 5+ | Operates on `detections_2d`/`detections_3d` directly, not on `tracks` — a fused track-level output (carrying `track_id` through) isn't produced |
| fuse radar | Phase 5+ | Only camera+lidar; radar isn't part of the fusion (consistent with detect3d not handling radar either) |
| fuse calibration lookup | Phase 5+ | Looks up calibration by `sensor_id` alone (first match wins if duplicates exist for one channel), not by the specific `calibrated_sensor_token` a given frame actually used — fine for this fixture (one calibration per channel) but not the fully general nuScenes join |
| label static thresholds | Phase 6+ | `auto_accept_threshold`/`reject_threshold`/`single_modality_discount` are fixed CLI flags, not learned or calibrated against any labeled validation set — there isn't one yet (that's what this pipeline is for) |
| label no active queue consumption | Phase 6+ | Produces a prioritized `needs_review` list in `pseudo_labels.parquet`, but there's no reviewer UI/workflow to actually consume it and feed decisions back into retraining — that's the natural Phase 7+ extension once real labeled data exists |
| label doesn't use tracks | Phase 6+ | Scores each fused object independently frame-by-frame; doesn't use track continuity (e.g. "this object was auto-accepted in 9 of the last 10 frames" as an additional trust signal) despite `tracks` existing since Phase 4 |
| evaluate GT schema simplified | Phase 7+ | `ground_truth.category_name` is flattened directly onto each annotation row; real nuScenes requires an `instance.json` → `category.json` join. Also uses FORGE's own `[w,h,l]` dimension order, not nuScenes' native `[w,l,h]` — never an issue in practice since GT is eval-only |
| evaluate camera-only exclusion | Phase 7+ | Only `matched`/`lidar_only` pseudo-labels (real 3D centers) are scored against 3D GT; `camera_only` predictions have no 3D grounding and are excluded from evaluation entirely, not just penalized |
| evaluate no NDS | Phase 7+ | Reports precision/recall/F1/mAP; doesn't implement nuScenes' full NDS (nuScenes Detection Score) composite metric, which also weighs translation/scale/orientation/velocity error |
| evaluate fixed distance threshold | Phase 7+ | Uses one `--distance-threshold` per run (default 2m), not nuScenes' official multi-threshold sweep (0.5/1/2/4m averaged) |
| curate not a learned embedding | Phase 8+ | The LanceDB vector is a deterministic 8-dim geometric feature (center/dims/heading), not a learned visual embedding — no trained embedding model exists in this pipeline to produce one |
| curate no export format | Phase 8+ | Writes `curated.parquet` (kept + duplicate-flagged rows); doesn't yet export to a training-ready format (COCO JSON, WebDataset, etc.) |
| curate single distance threshold | Phase 8+ | One `--distance-threshold` for all classes; a pedestrian and a bus arguably need different near-duplicate distance tolerances given their different real-world sizes |
| curate camera_only never deduped | Phase 8+ | `camera_only` pseudo-labels have no real 3D center (sentinel `[0,0,0]`) and are excluded from geometric dedup entirely (passed straight through as always-kept) — a genuine 2D-only near-duplicate (e.g. two overlapping NMS-adjacent boxes on the same object) won't be caught without an appearance-based or 2D-IoU-based signal, which doesn't exist here |
| Partial-extras mypy/pytest runs | Operational | mypy type-checks all of `src/forge` regardless of what's installed, and uninstalled extras' tests get `importorskip`-skipped, dragging coverage below threshold — install every extra together (`uv sync --all-extras --dev`, matching CI) before running the full check suite. A narrower extra sync is fine for actually running just that phase's CLI command. See README.md. |
| Phase 9 incomplete | Phase 9 | Ray (local, wired into detect2d + detect3d) + Lambda + a one-table Glue/Athena catalog are built. Ray isn't wired into track/fuse/label/evaluate/curate yet, the Glue catalog only defines `pseudo_labels` (not the other ~9 lake tables), and no ECS/real-cluster worker consumes the Lambda's SQS queue. |
| Ray not verifiable in the development sandbox | Phase 9 | Ray's multi-process runtime (its C++ core's plasma object store) hangs/crashes in the sandbox that built this, reproduced with three different configurations (defaults, constrained object-store memory, explicit temp dir) — a sandbox limitation, not a code defect (see DECISIONS.md ADR-026). The `distributed=True` path is verified by mocking Ray's API boundary (`tests/test_distributed.py`), the same way AWS is mocked for the Lambda tests — not by actually running Ray's parallel execution here. Real multi-process execution *was* confirmed working on a user's machine, including surfacing a real `ray.put()` fix (see the next row) — but the fix itself still needs re-verification there since this sandbox can't confirm it. |
| ray.put() fix unverified end-to-end | Phase 9 | `run_distributed_map`'s `shared_args` (added to stop Ray re-serializing the whole detect2d model into the remote function definition, see DECISIONS.md) is verified by mocked tests confirming `ray.put()` is called once per shared arg — but the actual real-world effect (no more "72 MiB remote function" warning) hasn't been re-confirmed on a machine that can run real Ray, since this sandbox still can't. |
| Ray only wired into detect2d + detect3d | Phase 9+ | `forge.distributed.run_distributed_map` is a general-purpose utility (now used by both detector inference paths, proving it's not detect2d-specific), but track/fuse/label/evaluate/curate still run single-process — none of those commands have a `--distributed` flag yet. |
| CLI test output substring checks | Operational | `tests/test_cli.py`'s `_plain()` helper strips ANSI codes *and* collapses whitespace runs before substring-checking CLI output — found necessary after two separate real failures (ANSI codes fragmenting a word under one terminal-width/rendering condition; Rich word-wrapping a long line, inserting a literal newline mid-phrase, under another). If a future CLI test checks for exact multi-word phrasing without going through `_plain()`, it's at risk of the same class of failure. |
| Lambda never actually invoked | Phase 9 | Like every sibling repo's cost-safety policy, Terraform is never `apply`'d in CI or by Claude — the handler's logic is unit-tested with a mocked boto3 client (`tests/test_lambda_ingest_trigger.py`), but the deployed Lambda itself has never been exercised against real S3/SQS. |
| Terraform not `terraform validate`-checked | Phase 9 | No `terraform` binary was available in the development sandbox that built this; the `.tf` files were checked for HCL *syntax* validity with `python-hcl2`, not full semantic validation against the AWS provider schema (which `terraform validate`/`plan` would catch, e.g. a typo'd argument name). |
| SQS queue has no consumer | Phase 9 | The Lambda publishes to `forge-ingest-notifications-<env>`, but nothing consumes it yet — that's the Ray/ECS ingest worker's job, which doesn't exist. |
| Glue catalog covers one table | Phase 9 | Only `pseudo_labels` has a Glue table definition; the other ~9 lake tables would follow the identical mechanical pattern (map the Arrow schema to Glue/Hive types) but aren't built out. |

## Schema Tables (not yet defined)

Additional Parquet tables will be added in their implementing phases. Phase 0 defines **frames v1** only.
