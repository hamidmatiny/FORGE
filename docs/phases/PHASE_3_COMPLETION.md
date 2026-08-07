# Phase 3 Completion — 3D Detection

## Scope

`forge detect3d`: train a 3D object detector (CPU, PyTorch Lightning) over
lidar point clouds and run it over LIDAR frames from the lake, writing a
`detections_3d` table.

## What was built

- **`Detections3DTable`** — new versioned Parquet table for predicted 3D
  boxes (`detection_id`, `frame_id`, `class_id`, `class_name`, `score`,
  `center_xyz`, `dimensions_whl`, `yaw`, `model_version`).
- **`forge.detect3d.pointcloud.load_point_cloud`** — reads nuScenes-format
  `.pcd.bin` files (raw float32, 5 columns: x/y/z/intensity/ring).
- **`forge.detect3d.model.PointNetEncoder`** — a real per-point shared-MLP
  encoder with global max-pooling: the core permutation-invariant idea from
  PointNet. Verified permutation-invariant by test (shuffling point order
  doesn't change the output feature).
- **`forge.detect3d.model.Detector3DModule`** — a `LightningModule` wrapping
  the encoder + a fixed-slot detection head (`NUM_QUERIES = 4` box
  predictions per cloud: objectness, class, center/dimensions/yaw). See
  DECISIONS.md ADR-014 for why fixed slots instead of a full anchor-based
  or set-prediction (Hungarian matching) architecture — torchvision has no
  off-the-shelf 3D detector, unlike Phase 2.
- **`forge.detect3d.dataset.SyntheticPointCloudDataset`** — random in-memory
  point clouds with exactly `NUM_QUERIES` boxes each, for a CPU
  training-loop smoke test.
- **`forge.detect3d.train.train_detector`** — CPU training loop, saves a
  checkpoint. Verified the loss actually *decreases* with more steps (not
  just that it runs) before committing — confirms real gradient flow.
- **`forge.detect3d.infer.load_detector` / `run_inference`** — loads a
  checkpoint (or falls back to untrained weights with a logged warning),
  runs inference over every `LIDAR_*` frame in the lake, resolves real
  point-cloud files via `--pointcloud-root`, skips missing files with a
  warning instead of crashing.
- **`forge detect3d --mode train|infer` CLI** — same shape as `detect2d`:
  `--checkpoint`, `--output-checkpoint`, `--pointcloud-root`, `--max-steps`,
  `--score-threshold`, `--local`. Clean error if the `[detect3d]` extra
  isn't installed.
- **Synthetic lidar `.pcd.bin` fixtures** added to the nuScenes fixture
  (matching the `LIDAR_TOP` `data_path`s already in `sample_data.json`) so
  inference has real files to run against.
- **14 tests** in `tests/test_detect3d.py`, including a permutation-
  invariance check on the encoder and a loss-decreases-with-training check
  on the full loop — not just "doesn't crash."

## What this phase does *not* claim

- Fixed 4 box slots per cloud, not a real variable-object-count detector
  (no NMS, no set-prediction matching). See DECISIONS.md ADR-014.
- No BEV feature map or semantic segmentation output — "BEV" in the
  original phase plan is satisfied at the basic level of predicting boxes
  in the ground/ego frame, not a full bird's-eye-view representation.
- Randomly initialized, trained only on synthetic data — not tuned for
  accuracy.
- Lidar only; radar (a different, sparser, Doppler-augmented format) isn't
  handled.
- Different class taxonomy from `detect2d` (foreground-only, no background
  class) — see DECISIONS.md ADR-015.

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge            # strict, 25 source files, 0 errors
uv run pytest -q                  # 51 passed, 91.43% coverage (threshold 80%)
uv run forge detect3d --mode train --max-steps 5 --local
uv run forge detect3d --mode infer --pointcloud-root tests/fixtures/nuscenes_mini_synthetic --local
```

Same local-sandbox torch install note as Phase 2 applies (DECISIONS.md
ADR-013) — not part of the repo or its dependencies.

## Known gaps carried forward

See [KNOWN_GAPS.md](../../KNOWN_GAPS.md) for the full list and [ARCHITECTURE.md](../ARCHITECTURE.md) for how this
phase maps back to the platform's requirement-coverage table.
