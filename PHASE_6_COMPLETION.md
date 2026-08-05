# Phase 6 Completion — Active Learning + Pseudo-Labeling

## Scope

`forge label`: score every fused object with a trust score that rewards
cross-modal agreement, route it to `auto_accept`/`needs_review`/`rejected`,
and rank the review queue by entropy-based active-learning priority.
Writes a new `pseudo_labels` table.

This is the pipeline's core auto-labeling decision point — everything
before it (ingest → detect2d/3d → track → fuse) produces candidate object
detections; this phase decides which of them are trustworthy enough to use
as labels, which need a human, and which to discard.

## What was built

- **`PseudoLabelsTable`** — new versioned Parquet table. Every fused
  object gets a row with a `trust_score`, a `decision`, and a
  `review_priority` — `rejected` rows are kept, not dropped, so every
  decision is auditable.
- **`forge.label.scoring.compute_trust_score`** — cross-modal agreement as
  the trust signal: `matched` objects get the average of their camera +
  lidar confidence; single-modality objects get their raw confidence
  discounted (`DEFAULT_SINGLE_MODALITY_DISCOUNT = 0.7`) since they lack
  independent confirmation. Verified by test that a matched object beats a
  single-modality one at equal raw confidence — the whole point of the
  cross-modal bonus.
- **`forge.label.scoring.binary_entropy`** — Shannon entropy of the trust
  score, used as review priority (maximized at 0.5 — the most uncertain,
  most valuable-to-review point; verified symmetric around 0.5 and zero at
  the extremes by test).
- **`forge.label.run.run_labeling`** — looks the original per-modality
  detection scores back up via `fused_objects`' `detection_id_2d`/
  `detection_id_3d` fields (kept out of `fused_objects` itself to keep
  Phase 5's schema lean), scores every fused object, and routes it by the
  `auto_accept_threshold`/`reject_threshold` bands.
- **`forge label` CLI** — `--auto-accept-threshold`, `--reject-threshold`,
  `--single-modality-discount`, `--local`. No extra dependency required at
  all — pure stdlib math.
- **20 tests** in `tests/test_label.py` covering trust-score math for all
  three fusion types, error handling (missing scores, unknown fusion
  type, inverted thresholds), entropy properties, and the full
  `run_labeling` orchestration (correct bucket routing, geometry fields
  preserved, and — critically — that the borderline case actually gets a
  higher review priority than the confident one).

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge            # strict, 39 source files, 0 errors
uv run pytest -q                  # 104 passed, 90.57% coverage (threshold 80%)
```

Plus manual end-to-end verification: hand-crafted synthetic
`fused_objects`/`detections_2d`/`detections_3d` designed to hit all three
decision buckets (a high-confidence matched object, a borderline
camera-only object, a low-confidence lidar-only object), ran `forge
label`, and confirmed both the bucket counts and the exact trust-score
arithmetic (0.9, 0.385, 0.14) matched what the formulas predict — plus
confirmed the borderline case got the highest review priority, as entropy
sampling should produce.

## What this phase does *not* claim

- `auto_accept_threshold`/`reject_threshold`/`single_modality_discount`
  are fixed CLI flags, not learned or calibrated against a labeled
  validation set — there isn't one yet.
- Produces a prioritized review list but no reviewer UI/workflow to
  consume it or feed decisions back into retraining.
- Scores each fused object independently frame-by-frame; doesn't use
  track continuity (e.g. "accepted in 9 of the last 10 frames") as an
  additional signal, despite `tracks` existing since Phase 4.

## Known gaps carried forward

See `KNOWN_GAPS.md` for the full list and `ARCHITECTURE.md` for how this
phase maps back to the platform's requirement-coverage table.
