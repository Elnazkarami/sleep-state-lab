#!/bin/bash
# The label-budget benchmark, sized and sequenced for an overnight run.
#
# Budgets run in priority order rather than numeric order: the primary
# comparison is declared as D3 minus D2 at 25%, so those six cells go first. If
# the machine is reclaimed at breakfast the headline is finished and only the
# context is missing.
#
# The runner is resumable -- it reads the predictions file and skips cells that
# already have rows -- so re-running this after an interruption continues rather
# than restarting.
set -u
cd /Volumes/maxone/sleep-state-lab || exit 1
export PYTHONPATH=src
PY=/Library/Frameworks/Python.framework/Versions/3.12/bin/python3
D=/Volumes/maxone/sleep-edf-data
S=outputs/split_cohort.json
E=runs/cohort_pretrain/encoder.pt
OUT=outputs/predictions_benchmark.csv
# The store is read in shuffled order every pass; on this USB exFAT volume that
# costs 10-15 ms an epoch, so it goes on internal storage instead.
STORE=/Users/Elnaz1/sleepstatelab-store

mkdir -p "$STORE"
echo "benchmark starting $(date)"
df -h / | tail -1

for budget in 0.25 0.1 1.0; do
  echo ""
  echo "############ BUDGET $budget  $(date) ############"
  $PY -u -m sleepstatelab.cli benchmark --config configs/default.yaml \
    --data-root "$D" --cache-dir "$D/cache" --store-dir "$STORE" \
    --split "$S" --pretrained-encoder "$E" --device mps \
    --budgets $budget --seeds 0 1 2 --models D2 D3 \
    --runs-dir runs/benchmark --output "$OUT" \
    || echo "BUDGET $budget did not finish; later budgets still attempted"
done

echo ""
echo "############ REPORT  $(date) ############"
$PY -u -m sleepstatelab.cli report "$OUT" --part test \
  --output docs/results_benchmark.md --json outputs/metrics_benchmark.json \
  --title "Label-budget benchmark: D3 against D2 at 10%, 25% and 100%" || true
$PY -u -m sleepstatelab.cli transitions "$OUT" --part test \
  --output docs/results_benchmark_transitions.md || true
echo "benchmark finished $(date)"
