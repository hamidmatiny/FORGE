# Phase 2 Completion — 2D Detection

## Scope

`forge detect2d`: train a 2D object detector (CPU, PyTorch Lightning) and
run it over camera frames from the lake, writing a `detections_2d` table.

## What was built

- **`Detections2DTable`** — new versioned Parquet table for predicted 2D
  boxes (`detection_id`, `frame_id`, `class_id`, `class_name`, `score`,
  `bbox_xyxy`, `model_version`).
- **`forge.detect2d.model.Detector2DModule`** — a `LightningModule`
  wrapping torchvision's `fasterrcnn_mobilenet_v3_large_320_fpn`
  (MobileNetV3+FPN backbone, real anchor-based RPN + NMS), randomly
  initialized — no pretrained weights downloaded. See DECISIONS.md
  ADR-011 for why.
- **`forge.detect2d.dataset.SyntheticDetectionDataset`** — random
  in-memory images + boxes for training-loop smoke tests (ADR-012);
  `load_image_tensor` for loading real images at inference time.
- **`forge.detect2d.train.train_detector`** — CPU training loop, saves a
  checkpoint (`model_state_dict` + `num_classes`).
- **`forge.detect2d.infer.load_detector` / `run_inference`** — loads a
  checkpoint (or falls back to untrained weights with a logged warning),
  runs inference over every `CAM_*` frame in the lake, resolves real image
  files via `--images-root`, skips missing files with a warning instead of
  crashing the whole run.
- **`forge detect2d --mode train|infer` CLI** — `--checkpoint`,
  `--output-checkpoint`, `--images-root`, `--max-steps`,
  `--score-threshold`, `--local`. Clean error if the `[detect2d]` extra
  (`torch`/`torchvision`/`lightning`) isn't installed — it's optional,
  never in the base install (ADR-004 pattern).
- **3 tiny synthetic JPEGs** added to the nuScenes fixture (matching the
  `CAM_FRONT` `data_path`s already in `sample_data.json`) so inference has
  real files to run against in tests, not just mocks.
- **10 tests** in `tests/test_detect2d.py`, skipped cleanly via
  `pytest.importorskip` if the `[detect2d]` extra isn't installed.

## What this phase does *not* claim

The detector is randomly initialized and trained only on synthetic
in-memory data — its predictions have no real accuracy. This phase proves
the *pipeline mechanics* (real multi-box architecture, Lightning training
loop, checkpoint save/load, Parquet-lake round trip) work correctly, not
that FORGE can detect real objects yet. See KNOWN_GAPS.md.

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge            # strict, 18 source files, 0 errors
uv run pytest -q                  # 35 passed, 91.65% coverage (threshold 80%)
uv run forge detect2d --mode train --max-steps 3 --local
uv run forge detect2d --mode infer --images-root tests/fixtures/nuscenes_mini_synthetic --local
```

A note on this specific development environment: installing `torch` here
required a workaround (installing without its normal CUDA dependency chain,
reusing CUDA libraries this sandbox happened to have preinstalled). That
workaround is **not** part of the repo or its dependency declarations — see
DECISIONS.md ADR-013. `uv sync --extra detect2d` resolves normally on a
machine with typical disk space and PyPI access.

## Known gaps carried forward

- Detector accuracy: untrained/randomly initialized until a real labeled
  training set exists (downstream of Phase 6).
- `CLASS_NAMES` is a placeholder taxonomy, not yet reconciled against any
  real labeled dataset.
- No batched inference (one image at a time) — fine at current scale.
- Non-camera sensors are silently skipped by design (lidar/radar is Phase 3).

See [KNOWN_GAPS.md](../../KNOWN_GAPS.md) for the full list and [ARCHITECTURE.md](../ARCHITECTURE.md) for how this
phase maps back to the platform's requirement-coverage table.
