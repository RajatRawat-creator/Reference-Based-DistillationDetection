#!/bin/bash
##SBATCH -p gpu  # Optional: set your cluster partition
#SBATCH --nodes=1
#SBATCH --job-name=ref_mia_controlled
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

# Roots for controlled checkpoints and MIA scoring inputs.
# Override via env vars if your data lives elsewhere.
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-"../checkpoints"}
DATASETS_DIR=${DATASETS_DIR:-"../data/MIADatasets"}
OUT_DIR=${OUT_DIR:-"../outputs/reference_mia_controlled"}
mkdir -p "$OUT_DIR"

# Skip header, loop over (target, reference) pairs
tail -n +2 pairs_controlled.csv | while IFS=, read -r target ref; do
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
