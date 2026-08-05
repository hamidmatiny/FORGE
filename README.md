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
| 5 | `forge fuse` — multi-sensor fusion | ⬜ |
| 6 | `forge label` — active learning + pseudo-labeling, review queue | ⬜ |
| 7 | `forge evaluate` — GT scoring, MLflow/W&B logging | ⬜ |
| 8 | `forge curate` — LanceDB dedup/search, dataset export | ⬜ |
| 9 | Distributed & cloud infra — Ray, Terraform S3/Athena | ⬜ |
| 10 | `forge visualize` — rerun.io, Foxglove MCAP, FiftyOne | ⬜ |
| 11 | Productionization — runbook, demo script | ⬜ |

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

Unimplemented commands exit with a clear error and reference [KNOWN_GAPS.md](KNOWN_GAPS.md).

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
| `FORGE_MLFLOW_URI` | `file:./mlruns` | MLflow tracking URI (Phase 2+) |
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
- [Schema reference](docs/schemas.md)
- [Known gaps](KNOWN_GAPS.md)
- [Architecture decisions](DECISIONS.md)

## License

MIT (project code). nuScenes data remains under its own non-commercial license.
