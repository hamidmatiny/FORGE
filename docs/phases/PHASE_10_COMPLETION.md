# Phase 10 Completion — Visualization

## Scope

`forge visualize`: export `pseudo_labels.parquet` to offline review artifacts —
rerun.io `.rrd` (3D boxes) or Foxglove-compatible MCAP (plain JSON messages).
Requires the `[visualize]` extra (`rerun-sdk`, `mcap`).

## What was built

- **`forge.visualize.rerun_export.build_rerun_recording`** — writes a headless
  rerun recording with 3D `Boxes3D` per `(scene_id, timestamp_us)`, batched under
  `scenes/{scene_id}/boxes`. Skips `camera_only` / sentinel `[0,0,0]` rows.
- **`forge.visualize.mcap_export.build_mcap_recording`** — writes MCAP with one
  JSON message per frame on `/forge/pseudo_labels` (all fusion types included).
- **`forge visualize --format rerun|mcap --output --decision-filter --local`**
  CLI wiring; reads `pseudo_labels.parquet` from the data lake.
- **9 tests** in `tests/test_visualize.py` (including `rerun rrd verify` via
  subprocess in-process) plus 2 CLI guard tests in `tests/test_cli.py`.

## Verified before commit

```
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge                              # strict, 55 source files, 0 errors
uv run mypy infra/lambda/ingest_trigger/handler.py  # strict, 0 errors
uv run pytest -q                                     # 178 passed, 90.25% coverage (threshold 80%)
```

Manual CLI on synthetic `pseudo_labels` (1 `matched` + 1 `camera_only` row):
`forge visualize --format rerun` printed **1 object**; `--format mcap` printed
**2 objects**; `.venv/bin/rerun rrd verify out.rrd` exited 0.

## A real bug found during implementation

Initial rerun export used global `rr.init()` + `rr.save()` after logging, then
returned without finalizing the file. `rerun rrd verify` failed in pytest with
`Missing RRD footer / no RRD manifests` — the footer only appeared on process
exit. Reproduced against rerun-sdk **0.35.0** (not assumed from docs): saving
after logging without `RecordingStream.flush()` + `disconnect()` leaves an
invalid RRD; global `rr.init(..., init_logging=False)` also leaves recording
disabled under pytest's logging capture, so the export path was switched to an
explicit `RecordingStream` with `save()` before logging, then `flush()` and
`disconnect()` before return. See DECISIONS.md ADR-028 and the Fix entry below.

MCAP export initially omitted `Writer.start()` — files failed `make_reader` with
`InvalidMagic` until `start()`/`finish()` were wired (mcap 1.4.0 API, verified
directly, not from memory).

## What this phase does *not* claim

- No live rerun viewer or Foxglove session is launched from the CLI.
- No FiftyOne dataset or UI integration.
- Rerun export does not join `tracks.parquet` for persistent per-object IDs.
- MCAP is JSON-on-a-topic, not Foxglove's native `SceneUpdate` schema.
- No camera images or lidar point clouds in either export — boxes/labels only.
- Ray/`--distributed` is not applicable; visualize is single-process file IO.

## Known gaps carried forward

See [KNOWN_GAPS.md](../../KNOWN_GAPS.md) (Phase 10 section) and [ARCHITECTURE.md](../ARCHITECTURE.md) for requirement
coverage. Phase 11 (productionization docs) remains open.
