# FORGE

**Fleet Offline Recognition & Ground-truth Engine** — a production-grade offline perception auto-labeling platform for autonomous-vehicle sensor data.

FORGE ingests logged camera, lidar, and radar data and produces high-quality annotations (2D/3D boxes, classes, tracks, segmentation) for model training and simulation. It runs **offline**, so it can use future frames, heavy models, and multi-pass refinement. Quality over latency, always.

> **Research / portfolio system.** FORGE is not a commercial product.

## Dataset Notice

Primary evaluation dataset: [nuScenes v1.0-mini](https://www.nuscenes.org/) (~4 GB, 10 scenes). nuScenes is licensed for **non-commercial** use only. Ground-truth annotations are used **only for evaluation** of auto-labels — never as pipeline input.

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (recommended) or pip

## Quickstart

```bash
# Clone and install
uv sync --dev

# Verify CLI
uv run forge --help

# Run tests
uv run pytest -q

# Generate / refresh synthetic fixture
uv run python scripts/make_fixture.py
```

> **Running the full check suite (mypy across all of `src/forge`, or pytest
> for full coverage) requires every extra installed together** — mypy
> type-checks the whole tree regardless of what's installed, and
> uninstalled extras' tests get skipped, pulling total coverage below the
> threshold. This matches what CI does (`.github/workflows/ci.yml`):
> ```bash
> uv sync --all-extras --dev
> uv run ruff check . && uv run ruff format --check . && uv run mypy src/forge && uv run pytest -q
> ```
> A narrower `uv sync --extra track --dev` (etc.) is for actually *running*
> just that phase's CLI command without the other phases' heavy deps —
> not for the full mypy/pytest pass, which will show partial-extras
> failures that aren't real bugs (see KNOWN_GAPS.md).

## Phase Checklist

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full pipeline diagram and a line-by-line
mapping of each phase to the requirement it's built to satisfy.

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Foundation (package, schemas, CI, Docker) | ✅ |
| 1 | `forge ingest` — nuScenes-mini → Parquet lake, DVC, Hydra configs | ✅ |
| 2 | `forge detect2d` — camera 2D detection (PyTorch Lightning) | ✅ |
| 3 | `forge detect3d` — lidar 3D detection / BEV | ✅ |
| 4 | `forge track` — multi-object tracking | ✅ |
| 5 | `forge fuse` — multi-sensor fusion | ✅ |
| 6 | `forge label` — active learning + pseudo-labeling, review queue | ✅ |
| 7 | `forge evaluate` — GT scoring, MLflow/W&B logging | ✅ |
| 8 | `forge curate` — LanceDB dedup/search, dataset export | ✅ |
| 9 | Distributed & cloud infra — Ray, Terraform S3/Athena/Lambda/EventBridge/StepFunctions/ECS | 🟡 Ray (local) + Lambda + EventBridge + Step Functions + ECS done |
| 10 | `forge visualize` — rerun.io, Foxglove MCAP, FiftyOne | ✅ |
| 11 | Productionization — runbook, demo script | ✅ |

## Demo (full synthetic pipeline)

```bash
uv sync --all-extras --dev
./scripts/demo.sh
```

See [RUNBOOK.md](RUNBOOK.md) for setup, per-stage commands, and troubleshooting.

## CLI

All stages are exposed through a single `forge` command:

```bash
forge ingest       # Phase 1
forge detect2d     # Phase 2
forge detect3d     # Phase 3
forge track        # Phase 4
forge fuse         # Phase 5
forge label        # Phase 6 — active learning / pseudo-labeling
forge evaluate     # Phase 7
forge curate       # Phase 8
forge visualize    # Phase 10
```

Unimplemented pipeline stages: none through Phase 10. Phase 11 adds operational docs only.

## Detection

> **Note:** `uv sync --extra X` resolves to *exactly* that extra set — running
> it again with a different extra uninstalls packages from the previous one.
> To have both `detect2d` and `detect3d` available at once:
> `uv sync --extra detect2d --extra detect3d --dev`.

```bash
# Install the heavy extras first (torch, torchvision, lightning)
uv sync --extra detect2d --dev

# Train a checkpoint (CPU, synthetic data — smoke-tests the training loop)
forge detect2d --mode train --max-steps 10 --output-checkpoint checkpoints/detect2d.pt --local

# Run inference over the camera frames already in the lake (needs 'forge ingest' first)
forge detect2d --mode infer --checkpoint checkpoints/detect2d.pt \
  --images-root /path/to/nuscenes-mini --local
```

Without `--checkpoint`, infer mode runs a freshly initialized (untrained)
model — useful to smoke-test the pipeline shape, not to get real detections.
See `PHASE_2_COMPLETION.md` for what this phase does and doesn't claim.

```bash
# 3D detection over lidar frames (torch/lightning again, no torchvision needed)
uv sync --extra detect3d --dev
forge detect3d --mode train --max-steps 10 --output-checkpoint checkpoints/detect3d.pt --local
forge detect3d --mode infer --checkpoint checkpoints/detect3d.pt \
  --pointcloud-root /path/to/nuscenes-mini --local
```

See `PHASE_3_COMPLETION.md` for the point-cloud model design and its
honestly-scoped limitations (fixed-slot prediction, no real 3D architecture
package available in this environment).

## Tracking

```bash
# Pure algorithmic — no torch/GPU needed, just numpy + scipy
uv sync --extra track --dev

# Requires 'forge ingest' and 'forge detect2d --mode infer' to have run first
forge track --iou-threshold 0.3 --max-age 3 --local
```

SORT-style: a Kalman filter (constant-velocity model) predicts each active
track forward one frame, then Hungarian assignment on IoU matches
predictions to this frame's detections. A fresh tracker runs per
`(scene, sensor)` sequence — tracks never span scenes. See
`PHASE_4_COMPLETION.md` for the design and a real bug found and fixed
during testing (track IDs colliding across different scenes).

## Fusion

```bash
# Also pure algorithmic — numpy + scipy, no torch
uv sync --extra fuse --dev

# Requires ingest + both detect2d and detect3d infer to have run first
forge fuse --iou-threshold 0.1 --local
```

Projects each 3D lidar detection's box into its synchronized camera frame
using the calibration recorded at ingest time (Phase 1), then matches
projected boxes to camera detections by IoU (reusing Phase 4's Hungarian
association code). Every row is tagged `matched`, `camera_only`, or
`lidar_only` — nothing is silently dropped. See `PHASE_5_COMPLETION.md`.

## Active Learning + Pseudo-Labeling

```bash
# No extra dependencies at all -- pure stdlib math
forge label --auto-accept-threshold 0.7 --reject-threshold 0.3 --local
```

Scores every fused object with a trust score that rewards cross-modal
agreement (a `matched` object gets the average of its camera + lidar
confidence; a single-modality object gets its raw confidence discounted,
since it lacks that cross-modal confirmation), then routes each one to
`auto_accept`, `needs_review`, or `rejected`. `needs_review` rows carry a
`review_priority` — binary entropy of the trust score, so a human reviewer
can work the queue in order of "most valuable to look at first" (classic
entropy/least-confidence active learning, applied to fused detections
instead of raw model logits). See `PHASE_6_COMPLETION.md`.

## Evaluation

```bash
uv sync --extra evaluate --dev

# Scores auto-accepted pseudo-labels against real nuScenes ground truth
# (eval-only -- see the Dataset Notice above)
forge evaluate --gt-input-dir /path/to/nuscenes-mini --local
```

Matches pseudo-labels to ground truth by BEV center distance — the same
convention nuScenes' own official detection metric uses, not 3D IoU — and
computes precision/recall/F1 plus mAP (mean average precision, VOC2012-
style interpolated PR curve) per class and overall. Logs every run to a
local MLflow SQLite store and an offline W&B run — no network calls, no
API keys. See `PHASE_7_COMPLETION.md`.

## Curation

```bash
uv sync --extra curate --dev
forge curate --distance-threshold 1.0 --local
```

Flags near-duplicate pseudo-labels using LanceDB vector search over an
8-dim geometric feature vector (center, dimensions, heading) — not a
learned visual embedding, since no trained embedding model exists in this
pipeline. Processes candidates highest-`trust_score`-first; every
near-duplicate is flagged with `duplicate_of_id` rather than dropped, so
the decision stays auditable. Never dedups across scenes or classes, even
at identical coordinates. See `PHASE_8_COMPLETION.md`.

## Infrastructure (Phase 9, partial — Ray + Lambda + EventBridge + Step Functions + ECS done, Ray-on-other-stages/rest-of-Glue-Athena open)

```bash
uv sync --extra detect2d --extra detect3d --extra aws --dev
uv run pytest tests/test_distributed.py tests/test_lambda_ingest_trigger.py -v
python3 scripts/validate_state_machine.py

# distributed inference (local Ray, no cluster) instead of --local
forge detect2d --mode infer --checkpoint <ckpt> --images-root <dir> --distributed
forge detect3d --mode infer --pointcloud-root <dir> --distributed
```

`forge.distributed.run_distributed_map`: a Ray-backed local-multi-process
map utility, wired into `detect2d`'s and `detect3d`'s per-frame inference.
`--distributed` runs each frame's inference across local CPU cores via
Ray instead of sequentially — same results either way, just execution
strategy. Large shared objects (the model) go through `ray.put()` once
via `shared_args`, not a closure, so Ray doesn't re-serialize them per
call. No real Ray cluster is provisioned (local CPU only, same
cost-safety policy as everywhere else).

`infra/lambda/ingest_trigger/handler.py`: an S3-upload-triggered Lambda
that validates uploads against the nuScenes-devkit layout and publishes
valid ones to both SQS and a custom EventBridge bus. The EventBridge
event triggers a Step Functions state machine
(`infra/terraform/step_functions.tf`) that chains all eight pipeline
stages — ingest through visualize — as `ecs:runTask.sync` calls against
one shared ECS Fargate task definition (`infra/terraform/ecs.tf`), each
overriding the container command to run the matching `forge` CLI
subcommand. Lambda still handles only the lightweight "notify something
happened" layer; EventBridge → Step Functions → ECS handle the actual
orchestration and work. `infra/terraform/`: the S3 buckets, SQS queue,
EventBridge bus/rule, Step Functions state machine, ECS cluster/task
definition, IAM roles, Lambda wiring, and a Glue/Athena catalog (one
representative table, `pseudo_labels`) — deployed out-of-band only, never
applied in CI, matching every sibling repo's cost-safety policy. See
`PHASE_9_COMPLETION.md`.

## Visualization

```bash
uv sync --extra visualize --dev
forge visualize --local --format rerun   # default: <lake>/visualize_export.rrd
forge visualize --local --format mcap --decision-filter auto_accept  # default: visualize_export.mcap
```

Exports `pseudo_labels.parquet` to an offline review file — rerun `.rrd` (3D
boxes per timestamp, OpenGL-backed viewer when opened locally) or Foxglove-compatible
MCAP with plain JSON messages. No live viewer is spawned from the CLI: this
environment is headless, and the intended workflow is write-then-open on a machine
with a display. Rerun export skips `camera_only` rows (sentinel `[0,0,0]` geometry
would stack meaningless boxes at the origin); MCAP export keeps them because
`bbox_xyxy` is still useful for 2D review. Boxes are batched per frame under one
entity path, not wired to `tracks.parquet` for persistent object identity across
frames. FiftyOne is not implemented in this pass. See `PHASE_10_COMPLETION.md`.

## Ingest

```bash
# Real nuScenes-mini (see docs/runbooks/ingest-real-nuscenes.md to get the data)
forge ingest --input-dir /path/to/nuscenes-mini --local

# Or exercise it right now against the committed synthetic fixture:
forge ingest --input-dir tests/fixtures/nuscenes_mini_synthetic --local
```

Only key-frame samples are ingested by default; pass `--all-sweeps` to include
non-keyframe sweeps. `--local` is required until Phase 9 adds Ray execution.

## Configuration

Environment variables use the `FORGE_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `FORGE_DATA_LAKE_ROOT` | `data/lake` | Parquet data lake root |
| `FORGE_MLFLOW_URI` | `file:./mlruns` | MLflow tracking URI (`forge evaluate`) |
| `FORGE_LOG_LEVEL` | `INFO` | Log level |

## Docker

```bash
docker compose -f docker/compose.yml run forge --help
```

## CI

![CI](https://github.com/OWNER/FORGE/actions/workflows/ci.yml/badge.svg)

GitHub Actions runs ruff, mypy (strict), pytest (≥80% coverage), and uv lock check on Python 3.11 and 3.12.

## Docs

- [Architecture + requirement coverage map](ARCHITECTURE.md)
- [Phase 1 completion](PHASE_1_COMPLETION.md)
- [Phase 2 completion](PHASE_2_COMPLETION.md)
- [Phase 3 completion](PHASE_3_COMPLETION.md)
- [Phase 4 completion](PHASE_4_COMPLETION.md)
- [Phase 5 completion](PHASE_5_COMPLETION.md)
- [Phase 6 completion](PHASE_6_COMPLETION.md)
- [Phase 7 completion](PHASE_7_COMPLETION.md)
- [Phase 8 completion](PHASE_8_COMPLETION.md)
- [Runbook](RUNBOOK.md)
- [Phase 10 completion](PHASE_10_COMPLETION.md)
- [Phase 11 completion](PHASE_11_COMPLETION.md)
- [Phase 9 completion](PHASE_9_COMPLETION.md)
- [Schema reference](docs/schemas.md)
- [Known gaps](KNOWN_GAPS.md)
- [Architecture decisions](DECISIONS.md)

## License

MIT (project code). nuScenes data remains under its own non-commercial license.
