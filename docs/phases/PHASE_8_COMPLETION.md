# Phase 8 Completion — Curation

## Scope

`forge curate`: flag near-duplicate pseudo-labels using LanceDB vector
search, producing a curated dataset ready for export. Writes a new
`curated` table.

## What was built

- **`CuratedTable`** — new versioned Parquet table. Every candidate
  pseudo-label gets a row, tagged `is_duplicate` + `duplicate_of_id` —
  duplicates are flagged, never silently dropped, so the decision stays
  auditable (same principle as `fused_objects`' three-way tagging and
  `pseudo_labels`' three-way decision routing).
- **`forge.curate.features.build_feature_vector`** — an 8-dim
  deterministic geometric feature vector (center, dimensions, heading as
  `sin`/`cos`) built directly from each pseudo-label's own geometry. This
  is explicitly **not** a learned visual embedding — no trained embedding
  model exists anywhere in this pipeline. Verified by test that the
  `sin`/`cos` heading encoding correctly keeps `-π` and `+π` (the same
  heading) close in feature space rather than maximally far apart.
- **`forge.curate.run.run_curation`** — processes candidates
  highest-`trust_score`-first, incrementally inserting each kept object
  into a LanceDB table and querying for the nearest already-kept neighbor
  (filtered by `scene_id` + `class_name`, so dedup never crosses scene or
  class boundaries even at identical coordinates — verified by test)
  before deciding whether the new candidate is a duplicate.
- **`forge curate --distance-threshold --decision-filter --local` CLI**.
  Clean error if the `[curate]` extra (`lancedb`) isn't installed, and if
  `pseudo_labels.parquet` doesn't exist yet.
- **13 tests** in `tests/test_curate.py` covering the feature-vector math
  (including the heading-wraparound property) and `run_curation`'s
  behavior: near-duplicate flagging, distinct objects staying separate,
  never crossing class or scene boundaries, higher trust always winning
  regardless of input order, decision filtering, and geometry
  preservation.

## Verified before commit

```
uv run ruff check .
uv run ruff format --check .
uv run mypy src/forge            # strict, 50 source files, 0 errors
uv run pytest -q                  # 138 passed, 90.16% coverage (threshold 80%)
```

Plus manual end-to-end verification: hand-crafted synthetic pseudo-labels
with two near-identical detections of the same car (different trust
scores) and one genuinely distinct pedestrian, ran `forge curate`,
confirmed the CLI's printed kept/duplicate counts and — critically — that
`curated.parquet`'s `duplicate_of_id` pointed at exactly the
higher-trust car, not the lower one.

## A real bug found and fixed via user testing

Running `forge curate --decision-filter all` on a full pipeline output
initially collapsed 308 pseudo-labels down to just 11 kept. Root cause:
`camera_only` pseudo-labels all carry a sentinel `[0,0,0]` geometry (no
real 3D grounding), so every `camera_only` row's feature vector was
literally identical — the geometric dedup pass "correctly" matched them
by the letter of the distance threshold, but this was a false signal:
300 genuinely distinct real 2D detections got flagged as duplicates of
one arbitrary survivor. Fixed by applying the same reasoning
`forge.evaluate` already uses for the identical underlying problem
(ADR-019) — only `matched`/`lidar_only` rows go through geometric dedup;
`camera_only` rows pass straight through as always-kept. Verified with a
direct reproduction (5 distinct `camera_only` detections, previously
collapsed to 1, now correctly all stay separate) and two new tests. Full
writeup in `DECISIONS.md`.

## What this phase does *not* claim

- The feature vector is geometric, not appearance-based — two visually
  different objects that happen to share almost the same
  position/size/heading would incorrectly dedup, and the same object seen
  from very different angles wouldn't necessarily land close in this
  feature space.
- No export to a training-ready format (COCO JSON, WebDataset, etc.) —
  writes `curated.parquet` only.
- One `--distance-threshold` for every class; a pedestrian and a bus
  arguably need different near-duplicate tolerances given their different
  real-world sizes.

## Verification note on this development environment

`lancedb` (60.7 MB) resolved and installed cleanly here without the
`--no-deps`-chasing workaround Phases 2–3's `torch` install needed — worth
noting since it's a contrast, not every heavy-ish dependency in this repo
required that workaround.

## Known gaps carried forward

See [KNOWN_GAPS.md](../../KNOWN_GAPS.md) for the full list and [ARCHITECTURE.md](../ARCHITECTURE.md) for how this
phase maps back to the platform's requirement-coverage table.
