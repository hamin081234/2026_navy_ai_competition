#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="${1:-../database}"
CACHE_DIR="${CACHE_DIR:-generated/cache/logmel96_f16000}"

if [[ ! -f "$CACHE_DIR/logmel.npy" || ! -f "$CACHE_DIR/metadata.csv" ]]; then
  python3 code/task2_cache_logmel.py \
    --data-dir "$DATA_DIR" \
    --out-dir "$CACHE_DIR"
fi

python3 infer_checkpoints.py \
  --data-dir "$DATA_DIR" \
  --cache-dir "$CACHE_DIR" \
  --checkpoint-dir checkpoints \
  --out-runs-dir generated/runs

python3 code/task2_multirun_score_ensemble.py \
  --data-dir "$DATA_DIR" \
  --runs-dir generated/runs \
  --out-dir generated/multirun_score_ensemble \
  --seed 42 \
  --holdout-frac 0.2 \
  --grid-units 20

mkdir -p generated/sources
cp generated/multirun_score_ensemble/validation_task2_multirun_score_ensemble_balanced.csv generated/sources/validation_multirun_balanced.csv
cp generated/multirun_score_ensemble/submission_task2_multirun_score_ensemble_balanced.csv generated/sources/submission_multirun_balanced.csv
cp generated/multirun_score_ensemble/validation_task2_multirun_score_ensemble_val086_robust.csv generated/sources/validation_multirun_val086_robust.csv
cp generated/multirun_score_ensemble/submission_task2_multirun_score_ensemble_val086_robust.csv generated/sources/submission_multirun_val086_robust.csv

# This source is a preserved rank-refinement artifact, not a single checkpoint model.
cp sources/validation_public_refined_validation_backed.csv generated/sources/validation_public_refined_validation_backed.csv
cp sources/submission_public_refined_validation_backed.csv generated/sources/submission_public_refined_validation_backed.csv

python3 rank_fusion_pipeline.py \
  --data-dir "$DATA_DIR" \
  --source-dir generated/sources \
  --out-dir generated/final
