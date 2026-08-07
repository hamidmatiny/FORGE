#!/usr/bin/env bash
# End-to-end FORGE pipeline on the committed synthetic nuScenes fixture.
# Requires: uv sync --all-extras --dev (same as CI).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

FIXTURE="${FORGE_DEMO_FIXTURE:-$ROOT/tests/fixtures/nuscenes_mini_synthetic}"
LAKE="${FORGE_DEMO_LAKE:-$ROOT/data/demo_lake}"
CKPT_DIR="${FORGE_DEMO_CKPT_DIR:-$ROOT/checkpoints/demo}"
LOG="${FORGE_DEMO_LOG:-$LAKE/demo.log}"

export FORGE_DATA_LAKE_ROOT="$LAKE"
export WANDB_MODE=offline
export MLFLOW_TRACKING_URI="file:${LAKE}/mlruns"

rm -rf "$LAKE"
mkdir -p "$LAKE" "$CKPT_DIR"

step() {
  echo ""
  echo "==> $*"
  echo "==> $*" >>"$LOG"
}

run_forge() {
  # Strip Rich ANSI codes from the log file; stdout still shows color.
  uv run forge "$@" 2>&1 | tee -a >(sed -E 's/\x1b\[[0-9;]*[a-zA-Z]//g' >>"$LOG")
}

echo "FORGE demo pipeline (synthetic fixture)"
echo "  fixture: $FIXTURE"
echo "  lake:    $LAKE"
echo "  log:     $LOG"
: >"$LOG"

step "ingest"
run_forge ingest --input-dir "$FIXTURE" --local

step "detect2d train (smoke — random-init weights, not tuned)"
run_forge detect2d --mode train --max-steps 5 --output-checkpoint "$CKPT_DIR/detect2d.pt" --local

step "detect2d infer"
run_forge detect2d --mode infer \
  --checkpoint "$CKPT_DIR/detect2d.pt" \
  --images-root "$FIXTURE" \
  --local

step "detect3d infer (untrained checkpoint if none provided — expect low/zero useful detections)"
run_forge detect3d --mode infer --pointcloud-root "$FIXTURE" --local

step "fuse"
run_forge fuse --local

step "label"
run_forge label --local

step "evaluate (GT from fixture — eval-only, never a pipeline input)"
run_forge evaluate --gt-input-dir "$FIXTURE" --local --decision-filter all

step "curate"
run_forge curate --local --decision-filter all

step "visualize rerun"
run_forge visualize --local --format rerun --decision-filter all

step "visualize mcap"
run_forge visualize --local --format mcap --decision-filter all

echo ""
echo "=== Demo complete — stage OK lines from this run ==="
grep 'OK' "$LOG" | grep -v 'cli_invocation' || true
echo ""
echo "Full log: $LOG"
echo "Lake artifacts: $LAKE"
