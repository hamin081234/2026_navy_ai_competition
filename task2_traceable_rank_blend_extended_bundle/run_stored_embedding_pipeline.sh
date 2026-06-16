#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="${1:-../database}"

python3 code/task2_multirun_score_ensemble.py \
  --data-dir "$DATA_DIR" \
  --runs-dir checkpoints \
  --out-dir generated/multirun_score_ensemble_from_stored_embeddings \
  --seed 42 \
  --holdout-frac 0.2 \
  --grid-units 20

mkdir -p generated/sources_from_stored_embeddings
cp generated/multirun_score_ensemble_from_stored_embeddings/validation_task2_multirun_score_ensemble_balanced.csv generated/sources_from_stored_embeddings/validation_multirun_balanced.csv
cp generated/multirun_score_ensemble_from_stored_embeddings/submission_task2_multirun_score_ensemble_balanced.csv generated/sources_from_stored_embeddings/submission_multirun_balanced.csv
cp generated/multirun_score_ensemble_from_stored_embeddings/validation_task2_multirun_score_ensemble_val086_robust.csv generated/sources_from_stored_embeddings/validation_multirun_val086_robust.csv
cp generated/multirun_score_ensemble_from_stored_embeddings/submission_task2_multirun_score_ensemble_val086_robust.csv generated/sources_from_stored_embeddings/submission_multirun_val086_robust.csv
cp sources/validation_public_refined_validation_backed.csv generated/sources_from_stored_embeddings/validation_public_refined_validation_backed.csv
cp sources/submission_public_refined_validation_backed.csv generated/sources_from_stored_embeddings/submission_public_refined_validation_backed.csv

python3 rank_fusion_pipeline.py \
  --data-dir "$DATA_DIR" \
  --source-dir generated/sources_from_stored_embeddings \
  --out-dir generated/final_from_stored_embeddings
