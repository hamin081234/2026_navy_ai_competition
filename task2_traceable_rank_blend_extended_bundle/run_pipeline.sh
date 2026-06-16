#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

DATA_DIR="${1:-../database}"

python3 rank_fusion_pipeline.py \
  --data-dir "$DATA_DIR" \
  --source-dir sources \
  --out-dir outputs
