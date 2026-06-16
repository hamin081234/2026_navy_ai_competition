#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

OUT="outputs/task2_repro_check/submission_reproduced_current_best.csv"

python3 scripts/task2_fixed_rank_blend.py \
  --out "$OUT" \
  --scheme flat \
  --path outputs/task2_panns_rank_blend_refine/submission_task2_fixed_rank_blend_best.csv \
  --weight 1.0 \
  --path outputs/task2_runs/projection_panns_cnn14_seed777_trainonly_ep1/submission_task2.csv \
  --weight 0.1

python3 scripts/task2_score_submissions.py \
  --out outputs/task2_repro_check/submission_reproduced_current_best_score.csv \
  "$OUT" \
  outputs/task2_runs/submission_task2_best.csv

cmp -s "$OUT" outputs/task2_runs/submission_task2_best.csv
echo "reproduced_matches_active"
