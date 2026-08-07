# FORGE Runbook

Operational guide for running and debugging the offline auto-labeling pipeline.
For the one-page overview, see [README.md](README.md). For design history, see
[DECISIONS.md](DECISIONS.md) and the `PHASE_*_COMPLETION.md` files in the repo root.

## Setup from a clean clone

```bash
git clone https://github.com/hamidmatiny/FORGE.git
cd FORGE
uv sync --all-extras --dev   # matches CI; required for full mypy/pytest
uv run forge --help
```

**Extras (when you do not need everything):**

| Goal | Install |
|------|---------|
| Full CI-equivalent checks | `uv sync --all-extras --dev` |
| Single stage only | `uv sync --extra <name> --dev` (see `pyproject.toml`) |
| End-to-end demo | `uv sync --all-extras --dev` (demo touches every stage) |

`uv sync --extra detect2d` alone uninstalls other extras' packages. That is fine for
running one CLI command, but **not** for `mypy src/forge` or full `pytest` — see
[KNOWN_GAPS.md](KNOWN_GAPS.md) (partial-extras coverage gap).

**One command for the full check suite, in the right order, every time:**

```bash
./scripts/check.sh
```

Runs `uv sync --all-extras --dev`, ruff, both mypy invocations (`src/forge` and the
Lambda handler — they're separate on purpose, see `KNOWN_GAPS.md`), pytest, the
Terraform HCL syntax check, the Step Functions structural validation, and the Glue
catalog schema check (confirms every table's Glue columns still match its real
PyArrow schema — see DECISIONS.md ADR-038), in that order. Existed because piecing
these commands together by hand has caused confusing (but expected) partial-extras
failures multiple times across this project's history — this is the one command
that always does it right. The Terraform check needs `python-hcl2` (not `hcl2` or
`hcl` — both different, wrong packages on PyPI); the script installs it
automatically if missing.

Default lake root: `FORGE_DATA_LAKE_ROOT=data/lake` (override with env var).

## Run the full pipeline (demo)

```bash
chmod +x scripts/demo.sh   # once
./scripts/demo.sh
```

Uses `tests/fixtures/nuscenes_mini_synthetic/`, writes to `data/demo_lake/`, and
runs: ingest → detect2d (train + infer) → detect3d (infer) → fuse → label →
evaluate → curate → visualize (rerun + mcap). Skips `track` (fuse reads raw
detections). Log: `data/demo_lake/demo.log`.

Environment overrides: `FORGE_DEMO_LAKE`, `FORGE_DEMO_FIXTURE`, `FORGE_DEMO_CKPT_DIR`.

## Run stages individually

Same fixture and `--local` on every stage until Ray is wired beyond detect2d/detect3d:

```bash
export FORGE_DATA_LAKE_ROOT=data/lake
forge ingest --input-dir tests/fixtures/nuscenes_mini_synthetic --local
forge detect2d --mode train --max-steps 10 --output-checkpoint checkpoints/detect2d.pt --local
forge detect2d --mode infer --checkpoint checkpoints/detect2d.pt \
  --images-root tests/fixtures/nuscenes_mini_synthetic --local
forge detect3d --mode infer --pointcloud-root tests/fixtures/nuscenes_mini_synthetic --local
forge fuse --local
forge label --local
forge evaluate --gt-input-dir tests/fixtures/nuscenes_mini_synthetic --local
forge curate --local
forge visualize --local --format rerun   # default output: <lake>/visualize_export.rrd
forge visualize --local --format mcap    # default output: <lake>/visualize_export.mcap
```

Real nuScenes-mini: [docs/runbooks/ingest-real-nuscenes.md](docs/runbooks/ingest-real-nuscenes.md).

## Interpreting low counts on the synthetic fixture

**0 or very few detections / pseudo-labels is expected**, not a broken install.

- `detect2d` / `detect3d` use randomly initialized or lightly smoke-trained weights
  on tiny synthetic images/point clouds — not nuScenes-trained models.
- The demo runs only **5** detect2d training steps as a loop smoke test.
- Score thresholds and fusion IoU still apply; empty downstream tables can be valid.

See [PHASE_2_COMPLETION.md](docs/phases/PHASE_2_COMPLETION.md), [PHASE_3_COMPLETION.md](docs/phases/PHASE_3_COMPLETION.md),
and [KNOWN_GAPS.md](KNOWN_GAPS.md) (detect2d/detect3d model accuracy rows).

## Troubleshooting (issues actually hit in this repo)

### Partial extras → mypy/pytest/coverage failures

Symptom: `pytest` fails coverage threshold or imports skip many tests.  
Fix: `uv sync --all-extras --dev` before the full check suite (CI does this).

### `forge evaluate` / MLflow errors (sqlalchemy, alembic)

Symptom: Import or runtime errors mentioning SQLAlchemy when logging metrics.  
Fix: install the `[evaluate]` extra (`mlflow-skinny` pulls sqlalchemy/alembic).  
See DECISIONS.md ADR-021 and Phase 7 completion notes.

### CLI tests fail on substring match (`forge ingest` split across lines)

Symptom: pytest `test_*_requires_*_lake` can't find `forge ingest` in output.  
Cause: Rich ANSI codes or word-wrap inserting a literal newline (DECISIONS.md Fix
entries for `_plain()`).  
Fix: already in `tests/test_cli.py` — if you add new CLI substring tests, route
output through `_plain()`.

### Rich swallowed extra names in error messages

Symptom: `requires the  extra` with no extra name.  
Cause: `[track]` treated as markup (DECISIONS.md Fix).  
Fix: messages use quoted extras: `the 'track' extra`.

### Ray `--distributed` hangs or crashes in some environments

Symptom: plasma/object-store errors in CI-like sandboxes.  
Status: API path tested with mocks; real multi-process Ray verified on a user
machine, not in every sandbox (ADR-026, KNOWN_GAPS.md).

### GitHub Actions: job cancelled, "runner not acquired" / internal server error

Unrelated to code — re-run failed jobs on the workflow run. Do not debug application
logic for this flake (seen repeatedly during this project).

### `rerun rrd verify` fails after visualize (Phase 10)

Symptom: `Missing RRD footer / no RRD manifests`.  
Fix: export uses `RecordingStream.flush()` + `disconnect()` (rerun-sdk 0.35.0).
Update rerun-sdk if behavior regresses.

### visualize default output path

If `--output` is omitted, files go to `<FORGE_DATA_LAKE_ROOT>/visualize_export.rrd`
or `visualize_export.mcap` — not `visualize.<ext>`.

## When something breaks — where to look

1. [KNOWN_GAPS.md](KNOWN_GAPS.md) — deliberate limitations and deferred work  
2. [DECISIONS.md](DECISIONS.md) — ADRs and **Fix:** entries for past defects  
3. Relevant `PHASE_N_COMPLETION.md` — what was verified for that stage  
4. [docs/schemas.md](docs/schemas.md) — Parquet contracts  

**Note:** `PHASE_N_COMPLETION.md` files snapshot test counts at commit time; a fresh
`uv run pytest -q` may report a higher number as tests accrue — use pytest output as
ground truth.

## Docker (optional)

```bash
docker compose run --rm forge --help
```

Image includes base package only; full pipeline needs local `uv sync --all-extras`.
