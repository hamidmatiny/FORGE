# Phase 5 Completion — Fusion

## Scope

`forge fuse`: fuse camera (2D) and lidar (3D) detections via calibrated
geometric projection + IoU association, writing a new `fused_objects` table.

## What was built

- **`FusedObjectsTable`** — new versioned Parquet table. Every 2D and 3D
  detection ends up in exactly one output row, tagged `matched`,
  `camera_only`, or `lidar_only` — nothing is silently dropped.
- **`forge.fuse.projection`** — real pinhole-camera projection math:
  quaternion-to-rotation-matrix, ego-to-sensor transform, 3D box corner
  computation, and enclosing-2D-bbox projection. Verified with hand-crafted
  known-good cases (a point at a known depth on the principal axis projects
  to the principal point; a point behind the camera returns `None`) before
  writing anything downstream on top of it.
- **`forge.fuse.run.run_fusion`** — pairs each camera frame with its
  synchronized lidar frame (same `scene_id` + `timestamp_us`), projects
  every 3D detection into that camera's image plane, and reuses Phase 4's
  Hungarian/IoU association (`forge.track.association.associate`) to match
  projected boxes against real camera detections.
- **`forge fuse` CLI** — `--iou-threshold`, `--local`. Checks for all four
  required lake tables (`frames`, `calibration`, `detections_2d`,
  `detections_3d`) up front with a clear error naming which command to run
  first. Clean error if the `[fuse]` extra (`numpy`, `scipy` — no torch)
  isn't installed.
- **13 tests** in `tests/test_fuse.py` covering the projection math in
  isolation and `run_fusion`'s four outcomes (matched, camera-only,
  lidar-only from behind-camera projection, camera-only from missing
  calibration/no synchronized lidar frame).

## Two test bugs found and fixed (not code bugs)

1. `pytest.approx()` doesn't support nested lists — comparing a 3x3
   rotation matrix directly raised `TypeError`. Fixed with
   `numpy.testing.assert_allclose`.
2. An early fusion-matching test used a camera box that only loosely
   contained the projected lidar box, giving IoU ≈ 0.06 — below the
   default `iou_threshold=0.1`. The code was correct; the test's fixture
   data wasn't tuned to actually clear the threshold. Fixed by computing
   the real projected bbox first and sizing the test's camera box to
   closely bracket it.

See DECISIONS.md for the full writeup.

## What this phase does *not* claim

- Synchronization assumes exact `(scene_id, timestamp_us)` equality
  between camera and lidar frames — real nuScenes sensors are nominally
  but not bit-exactly synchronized; a real deployment would need a
  nearest-match time window.
- Operates on `detections_2d`/`detections_3d` directly, not on `tracks` —
  no track-level fused output carrying `track_id` through.
- Camera + lidar only; radar isn't part of the fusion (consistent with
  `detect3d` not handling radar either).
- Calibration lookup is by `sensor_id` alone (first match wins), not the
  full `calibrated_sensor_token` join nuScenes technically supports.

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge            # strict, 35 source files, 0 errors
uv run pytest -q                  # 85 passed, 91.07% coverage (threshold 80%)
```

Plus manual end-to-end verification: ingested the fixture, hand-crafted
synthetic 2D+3D detections using the fixture's *real* calibration values
(computing the actual projected bbox first, not guessing), ran `forge
fuse`, and confirmed the CLI output (1 matched, 1 camera-only, 1
lidar-only) and the Parquet round-trip matched exactly what the geometry
predicted.

## Known gaps carried forward

See [KNOWN_GAPS.md](../../KNOWN_GAPS.md) for the full list and [ARCHITECTURE.md](../ARCHITECTURE.md) for how this
phase maps back to the platform's requirement-coverage table.
