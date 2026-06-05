#!/bin/bash
##SBATCH -p gpu  # Optional: set your cluster partition
#SBATCH --nodes=1
#SBATCH --job-name=normal_mia_controlled
#SBATCH --qos=preemptive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

cd "$(dirname "$0")"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var is not set." >&2
  exit 1
fi

CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-"../checkpoints"}
DATASETS_DIR=${DATASETS_DIR:-"../data/training"}
OUT_DIR=${OUT_DIR:-"../outputs/normal_mia_controlled"}
mkdir -p "$OUT_DIR"

# Reuse pairs_controlled.csv from reference_mia/ (same target list, ref column unused here)
PAIRS="../reference_mia/pairs_controlled.csv"

tail -n +2 "$PAIRS" | while IFS=, read -r target ref; do
    [[ -z "$target" ]] && continue
    target_path="$CHECKPOINTS_DIR/$target"
    if [[ ! -d "$target_path" ]]; then
        echo "[SKIP] target dir missing: $target_path" >&2
        continue
    fi
    echo "=== target=$target  ref=$ref ==="
    python run_controlled.py \
        --target_model "$target_path" \
        --ref_model    "$ref" \
        --datasets_dir "$DATASETS_DIR" \
        --out_dir      "$OUT_DIR"
done
