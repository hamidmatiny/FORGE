# Phase 11 Completion — Productionization

## Scope

Final phase in README.md's checklist: operational runbook + end-to-end demo script
on the synthetic fixture. No new pipeline stages.

## What was built

- **`scripts/demo.sh`** — `set -euo pipefail` orchestration: ingest → detect2d
  (train + infer) → detect3d (infer) → fuse → label → evaluate → curate →
  visualize (rerun + mcap). Skips `track` (fuse consumes raw detections, same as
  the architecture diagram's happy path for this demo).
- **`RUNBOOK.md`** — setup, per-stage commands, synthetic-fixture expectations,
  troubleshooting drawn from this repo's real DECISIONS.md **Fix:** entries and
  KNOWN_GAPS.md (partial extras, MLflow deps, Rich/_plain(), Ray sandbox, RRD
  finalize, GitHub Actions runner flakes).
- **Consistency audit fixes** (confirmed before edit):
  - `ARCHITECTURE.md` build-order table still marked Phases 1–8 "Not started"
    while README showed ✅ — aligned to Done / Partial (Phase 9).
  - `KNOWN_GAPS.md` still said MLflow/W&B and LanceDB were unstubs after Phases
    7–8 — updated to reflect evaluate + curate wiring, with honest limits.
  - `forge visualize --help` `--output` help text clarified to
    `<lake>/visualize_export.<ext>` (matches `cli.py`).
  - README Visualization examples use default filenames, not `review.*`.

## Verified before commit

```
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge                              # strict, 0 errors
uv run mypy infra/lambda/ingest_trigger/handler.py  # strict, 0 errors
uv run pytest -q                                     # 178 passed, 90.25% coverage (threshold 80%)
./scripts/demo.sh                                    # full synthetic E2E
```

Fresh pytest/coverage numbers are recorded in the Phase 11 commit message —
prior `PHASE_N_COMPLETION.md` files may cite older counts (noted in RUNBOOK.md).

## What this phase does *not* claim

- Demo does not train a production-quality detector (5 detect2d steps, random-init
  detect3d infer) — low/zero detection counts remain expected on the fixture.
- Demo does not run `track`, Ray `--distributed`, Terraform apply, or real nuScenes
  download.
- RUNBOOK is not a substitute for nuScenes licensing or fleet-data handling policy.

## Known gaps carried forward

All pipeline limitations remain in `KNOWN_GAPS.md` (Phase 9 partial infra,
FiftyOne, radar, etc.). Phase 11 closes the README checklist only — not the
underlying gaps.
