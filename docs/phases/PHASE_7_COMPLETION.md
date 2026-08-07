# Phase 7 Completion — Evaluation

## Scope

`forge evaluate`: score pseudo-labels against real nuScenes ground truth
(strictly evaluation-only, per the dataset's license), compute detection
quality metrics, and log every run to MLflow and W&B. Writes a new
`eval_metrics` table.

## What was built

- **`GroundTruthTable`** — new versioned Parquet table for nuScenes human
  annotations. Deliberately simplified from the real nuScenes schema
  (flattened `category_name`, FORGE's own `[w,h,l]` dimension order) since
  GT never flows anywhere except into this phase's own scoring code — see
  DECISIONS.md ADR-020.
- **`forge.evaluate.ingest_gt.ingest_ground_truth`** — parses
  `sample_annotation.json`, joined against `sample.json`/`scene.json` for
  scene/timestamp, with quaternion-to-yaw conversion.
- **`forge.evaluate.metrics.evaluate_class`** — BEV center-distance
  matching (nuScenes' own official convention, not 3D IoU — see ADR-019)
  and standard VOC2012-style average precision via a score-ordered greedy
  match. **Verified against a known textbook example (AP=0.833 for a
  TP/FP/TP/FP pattern with 2 GT boxes)** before trusting it anywhere else.
- **`EvalMetricsTable`** — new table, one row per class plus one `overall`
  row per run.
- **`forge.evaluate.run.run_evaluation`** — excludes `camera_only`
  pseudo-labels (no real 3D center to compare), filters by `decision`
  (default `auto_accept` — "how good are the labels we'd actually use"),
  and aggregates `overall` correctly as mAP (mean of per-class AP) with
  micro-averaged precision/recall/F1 — not naive cross-class pooling,
  which would let a car prediction "match" a pedestrian GT box.
- **`forge.evaluate.tracking`** — MLflow (local SQLite tracking store) and
  W&B (offline mode) logging, both confirmed working end-to-end with
  **zero network calls**. Failures in either backend log a warning and
  continue rather than fail the whole evaluation run.
- **`forge evaluate --gt-input-dir --decision-filter --distance-threshold
  --local` CLI**.
- Synthetic `sample_annotation.json` added to the nuScenes fixture.
- **19 tests** in `tests/test_evaluate.py` covering quaternion-to-yaw
  math, GT ingestion, the metrics module (including the textbook AP check
  and a "two predictions can't double-claim one GT box" check), the full
  `run_evaluation` orchestration, and MLflow/W&B logging (including a
  simulated-missing-package test confirming logging failures don't raise).

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge            # strict, 46 source files, 0 errors
uv run pytest -q                  # 123 passed, 90.48% coverage (threshold 80%)
```

Plus manual end-to-end verification: hand-crafted synthetic pseudo-labels
with a known-good match, a known miss, and a `camera_only` case designed
to be excluded — confirmed the CLI's printed precision/recall/mAP matched
hand-calculated values exactly, confirmed the per-class breakdown was
correct, and confirmed both MLflow's SQLite database and W&B's offline run
directory were actually created on disk.

## What this phase does *not* claim

- No nuScenes Detection Score (NDS) — the dataset's own composite metric
  that also weighs translation/scale/orientation/velocity error; this
  phase reports precision/recall/F1/mAP only.
- Single fixed distance threshold per run, not nuScenes' official
  multi-threshold sweep (0.5/1/2/4m averaged).
- `camera_only` predictions are excluded from evaluation entirely, not
  scored with a 2D-only fallback metric.

## Verification note on this development environment

Installing `mlflow-skinny` here required chasing a deeper transitive
dependency tree than expected (opentelemetry, alembic, fastapi, etc.) due
to this sandbox's `--no-deps`-first installation approach (see DECISIONS.md
ADR-013 for why that approach exists at all). None of that is reflected in
`pyproject.toml` — a normal `uv sync --extra evaluate` resolves everything
in one step on a machine with normal disk/network access.

## Known gaps carried forward

See [KNOWN_GAPS.md](../../KNOWN_GAPS.md) for the full list and [ARCHITECTURE.md](../ARCHITECTURE.md) for how this
phase maps back to the platform's requirement-coverage table.
