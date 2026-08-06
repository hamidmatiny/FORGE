# FORGE Architecture

FORGE is an offline, batch auto-labeling pipeline for fleet sensor data. Every stage
reads its inputs from the Parquet data lake and writes its outputs back to it —
no stage calls another directly. This keeps stages independently testable, restartable,
and (from Phase 9) independently distributable across a Ray cluster.

```
nuScenes-mini (eval GT, never a pipeline input)
      |
      v
[1] ingest        --> frames / calibration / ego-pose tables (Parquet, versioned via DVC)
      |
      v
[2] detect2d ---+
[3] detect3d ---+--> per-frame detections (2D boxes + 3D boxes/BEV, class, score)
      |
      v
[4] track          --> detections associated into per-object tracks
      |
      v
[5] fuse           --> camera+lidar tracks merged into one fused-track table
      |
      v
[6] label           --> active-learning sample selection + pseudo-label generation,
      |                 confidence-gated: low-confidence frames route to a review queue
      v                 instead of being auto-accepted
[7] evaluate        --> auto-labels scored against nuScenes GT (eval split only);
      |                 metrics + run params logged to MLflow / W&B
      v
[8] curate          --> LanceDB embedding index for near-duplicate / hard-example
      |                 search; final dataset export with lineage
      v
[10] visualize       --> rerun.io + Foxglove MCAP scene playback, FiftyOne dataset
                          review app

Cross-cutting (Phase 9, no dedicated CLI verb):
  - Ray: local-multi-process distributed execution via `forge.distributed.run_distributed_map`,
    wired into detect2d and detect3d's per-frame inference via a `--distributed` flag (`--local` still works
    the same as before). Real Ray-cluster provisioning isn't built — this is local-CPU-core
    parallelism only, the same "no real cloud/GPU spend" policy as everywhere else.
  - Terraform: S3 raw-data + processed-lake buckets, Glue catalog database (one full table
    definition, `pseudo_labels`, as a representative example), an Athena workgroup, and the
    Lambda's own wiring — applied out-of-band only (never in CI)
  - Lambda: S3-upload-triggered validator that publishes valid nuScenes
    uploads to SQS for a downstream Ray/ECS ingest worker to consume
    (infra/lambda/ingest_trigger/) — Lambda handles the lightweight
    "notify something happened" layer; Ray/ECS handle the actual work
```

## Why this shape

- **Offline-first.** Every stage can see future frames and re-run with heavier models
  or multiple passes — this is what separates auto-labeling from real-time perception,
  and it's why quality gates (Phase 6, 7) exist as first-class stages instead of an
  afterthought.
- **Contract-first.** Every table is a `BaseTable`: a Pydantic record model paired with
  an explicit, versioned PyArrow schema (see `docs/schemas.md`). Stages never share
  in-memory objects — only the lake.
- **Cost-safety.** CI never downloads real fleet data, never calls a paid API, and never
  provisions real cloud infrastructure or GPUs — same policy as the sibling repos
  ([Vulcan ADR-002](https://github.com/hamidmatiny/Vulcan/blob/main/docs/adr/002-gpu-cost-safety-policy.md),
  [PRISM ADR-001](https://github.com/hamidmatiny/PRISM/blob/main/docs/adr/001-cost-safety-policy.md)).
  Synthetic fixtures and emulators (DuckDB/local Parquet, moto for AWS) stand in.

## Requirement coverage map

FORGE's phase plan is scoped so every stage maps to a real, load-bearing capability —
not just the ones that were easy to build. The table below traces each requirement to
the phase that actually implements it. Nothing here is aspirational; a row only
appears once its phase is built and tested.

| Requirement | FORGE phase | What it actually does |
|---|---|---|
| Active learning & pseudo-labeling, CV, model training | **6 — label** | Confidence-gated pseudo-label generation; low-confidence frames route to a review queue instead of auto-accept |
| 2D/3D object detection | **2, 3 — detect2d, detect3d** | Camera 2D boxes; lidar 3D boxes / BEV |
| Tracking | **4 — track** | Multi-object association across frames |
| Sensor fusion | **5 — fuse** | Camera + lidar track fusion (radar not handled, see KNOWN_GAPS.md) |
| Semantic segmentation / BEV | **3 — detect3d** | BEV representation as part of the 3D head |
| Scaled MLOps: ML frameworks, experiment tracking, model registry (MLflow, W&B) | **7 — evaluate** | Run params/metrics logged to MLflow (self-hosted) and W&B (offline mode) |
| ML metrics & evaluation quality | **7 — evaluate** | Auto-label vs. nuScenes-GT scoring and quality tracking |
| Distributed ML (PyTorch, Lightning, Ray) | **2–8 (training), 9 (Ray)** | PyTorch Lightning for detector training; Ray as the distributed execution backend (local multi-process, wired into detect2d's and detect3d's `--distributed` inference paths) |
| Model data curation — Parquet (PyArrow, Daft, Pandas) | **1, 8 — ingest, curate** | Parquet lake built directly on PyArrow (via the `BaseTable` schema pattern); Daft/Pandas aren't actually used anywhere in the repo, see KNOWN_GAPS.md |
| Python dev, CI (GitHub Actions), Docker | **0 — foundation** | Already in place: ruff, mypy strict, pytest ≥80% coverage, GH Actions matrix, Docker |
| Data ops: schema design, AWS storage, vector DB (LanceDB), MCAP | **1, 8, 10** | Versioned schemas (1); LanceDB dedup/search index (8); Foxglove MCAP export (10) |
| Data viz: OpenGL/three.js, foxglove, FiftyOne, Tableau | **10 — visualize** | rerun.io (OpenGL-backed) + Foxglove MCAP + FiftyOne review app |
| Cloud dev: Terraform, AWS (S3, Athena, Lambda, etc.) | **9 — infra** | Terraform-provisioned S3 lake + Glue/Athena catalog + a real Lambda (S3-upload validator → SQS), applied manually/out-of-band |
| Cloud orchestration, model inference orchestration | **9 — infra** | Ray distributed execution across stages |
| Guidelines/standards, technical leadership | **11 — productionization** | Runbook + engineering-bar docs, same pattern as the sibling repos |

## Build order

| Phase | Component | Status |
|---|---|---|
| 0 | Foundation — package, `BaseTable` schema pattern, CI, Docker | Done |
| 1 | `forge ingest` — nuScenes-mini → Parquet lake, DVC versioning, Hydra configs | Not started |
| 2 | `forge detect2d` — camera 2D detection, Lightning training loop | Not started |
| 3 | `forge detect3d` — lidar 3D detection / BEV | Not started |
| 4 | `forge track` — multi-object tracking | Not started |
| 5 | `forge fuse` — multi-sensor fusion | Not started |
| 6 | `forge label` — active learning + pseudo-labeling, review queue | Not started |
| 7 | `forge evaluate` — GT scoring, MLflow/W&B logging | Not started |
| 8 | `forge curate` — LanceDB dedup/search, dataset export | Not started |
| 9 | Distributed & cloud infra — Ray execution mode, Terraform S3/Athena/Lambda (no CLI verb) | Ray (local, wired into detect2d + detect3d) + Lambda done; Glue/Athena has one representative table; ECS worker, Ray for track/fuse/label/evaluate/curate still open |
| 10 | `forge visualize` — rerun.io, Foxglove MCAP, FiftyOne | Not started |
| 11 | Productionization — runbook, demo script, engineering-bar docs | Not started |

See `docs/schemas.md` for the current Parquet table contracts and `KNOWN_GAPS.md` for
what's explicitly deferred at each stage.
