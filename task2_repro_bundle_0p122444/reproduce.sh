#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

python3 scripts/rank_blend.py \
  --out outputs/submission_reproduced_current_best.csv \
  --scheme flat \
  --path source_submissions/submission_task2_fixed_rank_blend_best.csv \
  --weight 1.0 \
  --path source_submissions/projection_panns_cnn14_seed777_trainonly_ep1_submission_task2.csv \
  --weight 0.1

python3 scripts/score_submission.py outputs/submission_reproduced_current_best.csv \
  | tee outputs/reproduced_score.csv

cmp -s outputs/submission_reproduced_current_best.csv expected/submission_task2_best.csv
echo "reproduced_matches_expected"
