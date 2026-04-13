#!/usr/bin/env bash
set -euo pipefail

# Batch capture runner for stage-1 workload characterization.
# It calls export_sampled_batches.py with:
#   --save-csr --save-edge-index
#
# Default matrix:
#   datasets: reddit, ogbn-arxiv
#   models: gcn, graphsage
#   fanouts: 25,10 and 50,25
#   num_batches: 300

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

DATA_ROOT="${DATA_ROOT:-./datasets}"
OUT_ROOT="${OUT_ROOT:-./analysis/workload/step1_capture}"
SPLIT="${SPLIT:-train}"
NUM_BATCHES="${NUM_BATCHES:-300}"
SEED="${SEED:-0}"

# You can override these by exporting env vars before running:
#   DATASETS="reddit ogbn-arxiv"
#   MODELS="gcn graphsage"
#   FANOUTS="25,10 50,25"
DATASETS="${DATASETS:-reddit ogbn-arxiv}"
MODELS="${MODELS:-gcn graphsage}"
FANOUTS="${FANOUTS:-25,10 50,25}"

mkdir -p "$OUT_ROOT"

echo "ROOT_DIR=$ROOT_DIR"
echo "DATA_ROOT=$DATA_ROOT"
echo "OUT_ROOT=$OUT_ROOT"
echo "SPLIT=$SPLIT NUM_BATCHES=$NUM_BATCHES SEED=$SEED"
echo "DATASETS=$DATASETS"
echo "MODELS=$MODELS"
echo "FANOUTS=$FANOUTS"

for ds in $DATASETS; do
  for model in $MODELS; do
    for fanout in $FANOUTS; do
      # Conservative defaults to reduce OOM risk.
      batch_size=1024
      if [[ "$ds" == "ogbn-arxiv" ]]; then
        batch_size=512
      fi

      tag="${ds}_${model}_f${fanout//,/x}_b${NUM_BATCHES}"
      out_dir="${OUT_ROOT}/${tag}"

      echo
      echo ">>> Running: $tag"
      echo "    out_dir: $out_dir"

      env -u OMP_NUM_THREADS python tools/workload/export_sampled_batches.py \
        --dataset "$ds" \
        --model "$model" \
        --fanouts "$fanout" \
        --split "$SPLIT" \
        --batch-size "$batch_size" \
        --num-batches "$NUM_BATCHES" \
        --save-csr \
        --save-edge-index \
        --root "$DATA_ROOT" \
        --out-dir "$out_dir" \
        --seed "$SEED"
    done
  done
done

echo
echo "All captures finished."
