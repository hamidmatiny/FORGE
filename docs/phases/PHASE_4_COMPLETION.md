# Phase 4 Completion — Tracking

## Scope

`forge track`: associate 2D detections across frames into persistent
tracks, writing a new `tracks` table.

## What was built

- **`TracksTable`** — new versioned Parquet table: one row per detection,
  tagged with the track it was assigned to (`track_id`, `track_age`,
  `tracker_version`, plus the detection's own fields for convenience).
- **`forge.track.kalman.KalmanBoxTracker`** — a constant-velocity Kalman
  filter over `[center_x, center_y, area, aspect_ratio]` (the classic SORT
  state representation), implemented directly with NumPy linear algebra —
  no external Kalman-filter library.
- **`forge.track.association`** — pairwise IoU (`iou_batch`) and Hungarian
  assignment (`associate`, via `scipy.optimize.linear_sum_assignment`)
  between predicted track boxes and this frame's detections.
- **`forge.track.tracker.SortTracker`** — manages the full track lifecycle
  per `(scene, sensor)` sequence: predict all active tracks, associate,
  update matched tracks, spawn new tracks for unmatched detections, age and
  retire tracks that miss `max_age` consecutive frames.
- **`forge.track.run.run_tracking`** — orchestrates one independent
  `SortTracker` per `(scene_id, sensor_id)` group across the full lake,
  stepping every frame in timestamp order (including frames with zero
  detections, so stale tracks age out correctly rather than lingering).
- **`forge track` CLI** — `--iou-threshold`, `--max-age`, `--local`. No
  `--mode train/infer` split like detect2d/detect3d — this is a classical
  algorithm, not a trained model. Clean error if the `[track]` extra
  (`numpy`, `scipy` — no torch needed) isn't installed, and if
  `frames.parquet`/`detections_2d.parquet` don't exist yet.
- **20 tests** in `tests/test_track.py`, covering the Kalman filter
  round-trip, IoU correctness, association matching/threshold behavior,
  the full track lifecycle (birth, hit-streak increment, aging, retirement,
  new-ID-after-retirement), and `run_tracking` orchestration across scenes.

## A real bug found and fixed

`SortTracker` numbers tracks locally per instance (`track-000001`,
`track-000002`, ...). `run_tracking` creates one tracker per
`(scene_id, sensor_id)` group — so two different scenes' first tracks were
both literally the string `"track-000001"`, silently colliding when read
back from `tracks.parquet`. A dedicated test
(`test_run_tracking_separates_different_scenes`) caught this before it was
pushed. Fixed by scoping `track_id` globally:
`f"{scene_id}:{sensor_id}:{local_track_id}"`. See DECISIONS.md for the full
writeup.

## What this phase does *not* claim

- 2D only — operates on `detections_2d`; 3D tracking (over `detections_3d`)
  isn't implemented.
- Constant-velocity motion model only — no turn model, no ego-motion
  compensation (despite `ego_pose` being available in the lake since
  Phase 1).
- No appearance-based re-identification — a track that ages out and later
  reappears gets a brand-new ID, verified by test.

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge            # strict, 31 source files, 0 errors
uv run pytest -q                  # 70 passed, 91.13% coverage (threshold 80%)
```

Plus manual verification beyond the automated tests: a drifting-object
simulation (same track ID across 4 frames), an occlusion simulation (track
survives `max_age` missed frames then retires, reappearance gets a new ID),
and a full CLI run against synthetic multi-frame detections with the
Parquet output round-tripped and inspected.

## Known gaps carried forward

See [KNOWN_GAPS.md](../../KNOWN_GAPS.md) for the full list and [ARCHITECTURE.md](../ARCHITECTURE.md) for how this
phase maps back to the platform's requirement-coverage table.
