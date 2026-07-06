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

| Phase | Scope | Status |
|-------|-------|--------|
| 0 | Foundation (package, schemas, CI, Docker) | ✅ |
| 1 | `forge ingest` — nuScenes-mini → Parquet lake | ⬜ |
| 2 | `forge detect2d` | ⬜ |
| 3 | `forge detect3d` | ⬜ |
| 4 | `forge track` | ⬜ |
| 5 | `forge fuse` | ⬜ |
| 6 | `forge evaluate` | ⬜ |
| 7 | `forge curate` | ⬜ |
| 8 | `forge visualize` | ⬜ |

## CLI

All stages are exposed through a single `forge` command:

```bash
forge ingest       # Phase 1
forge detect2d     # Phase 2
forge detect3d     # Phase 3
forge track        # Phase 4
forge fuse         # Phase 5
forge evaluate     # Phase 6
forge curate       # Phase 7
forge visualize    # Phase 8
```

Unimplemented commands exit with a clear error and reference [KNOWN_GAPS.md](KNOWN_GAPS.md).

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

- [Schema reference](docs/schemas.md)
- [Known gaps](KNOWN_GAPS.md)
- [Architecture decisions](DECISIONS.md)

## License

MIT (project code). nuScenes data remains under its own non-commercial license.
