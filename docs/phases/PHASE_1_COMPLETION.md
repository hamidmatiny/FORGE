# Phase 1 Completion — Ingest

## Scope

`forge ingest`: nuScenes-devkit-format sensor logs → versioned Parquet lake.
Hydra Compose API for config. DVC scaffolding for dataset versioning.

## What was built

- **`forge.ingest.nuscenes.ingest_nuscenes`** — parses the standard nuScenes
  JSON tables (`scene`, `sample`, `sample_data`, `sensor`, `calibrated_sensor`,
  `ego_pose`) and writes three Parquet lake tables: `frames`, `calibration`,
  `ego_pose`. Deduplicates calibration and ego-pose rows across frames that
  share the same token.
- **`frames` schema bumped to v1.1** — added `data_path` (additive, see
  DECISIONS.md ADR-008).
- **Two new tables**: `calibration` (extrinsics + optional camera intrinsics)
  and `ego_pose` (global vehicle pose), both versioned `BaseTable`s.
- **`forge ingest` CLI command** — `--input-dir`, `--nuscenes-version`,
  `--split`, `--all-sweeps`, `--local`. Clear `FileNotFoundError` messaging
  when required nuScenes JSON tables are missing.
- **Hydra Compose API config** (`forge/config.py`, `conf/ingest.yaml`) — see
  DECISIONS.md ADR-009 for why the Compose API instead of `@hydra.main`.
- **DVC scaffolding** — `dvc init`, local relative-path remote, `params.yaml`,
  `dvc.yaml` `ingest` stage. See DECISIONS.md ADR-010 for why no real dataset
  is tracked yet.
- **Synthetic nuScenes-mini-shaped fixture** (`tests/fixtures/nuscenes_mini_synthetic/`)
  — 2 scenes, 2 sensors (camera + lidar), 6 `sample_data` rows including one
  non-keyframe sweep, used for all ingest tests so CI never touches the real
  (non-commercial-licensed, ~4 GB) dataset.
- **`docs/runbooks/ingest-real-nuscenes.md`** — how to run ingest against a
  real local nuScenes-mini checkout.

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge          # strict, 12 source files, 0 errors
uv run pytest -q                # 23 passed, 95.6% coverage (threshold 80%)
uv run forge ingest --input-dir tests/fixtures/nuscenes_mini_synthetic --local
dvc dag                         # validates dvc.yaml
```

## Known gaps carried forward

- Non-keyframe sweeps: supported (`--all-sweeps`) but downstream phases
  assume key-frames only until stated otherwise.
- Ingest overwrites the lake on every run; no incremental/append mode yet.
- No real nuScenes-mini run has been executed anywhere in this repo's
  history — by design, per the dataset's non-commercial license.

See [KNOWN_GAPS.md](../../KNOWN_GAPS.md) for the full list and [ARCHITECTURE.md](../ARCHITECTURE.md) for how this
phase maps back to the platform's requirement-coverage table.
