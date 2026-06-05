#!/bin/bash
##SBATCH -p gpu  # Optional: set your cluster partition
#SBATCH --nodes=1
#SBATCH --job-name=ref_mia_omi_cot
#SBATCH --qos=preemptive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Reference-normalized loss MIA for the OMI-CoT student models, using the
# per-student few-shot candidate datasets in data/omi_cot_fewshot/ ("+ In-context
# exemplars from S", the Reference_Fewshot condition). Same runner as
# run_controlled.py but with the few-shot FILES_MAP (see run_omi_cot.py); each
# student is scored against ITS OWN few-shot subfolder via --datasets_dir.

set -euo pipefail
mkdir -p logs

cd "$(dirname "$0")"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var is not set." >&2
  exit 1
fi

CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-"../checkpoints"}
FEWSHOT_BASE=${FEWSHOT_BASE:-"../data/omi_cot_fewshot"}
OUT_DIR=${OUT_DIR:-"../outputs/reference_mia_omi_cot"}
mkdir -p "$OUT_DIR"

# Skip header; columns: target,reference,fewshot_subdir
tail -n +2 pairs_omi_cot.csv | while IFS=, read -r target ref subdir; do
    [[ -z "$target" ]] && continue
    target_path="$CHECKPOINTS_DIR/$target"
    datasets_dir="$FEWSHOT_BASE/$subdir"
    if [[ ! -d "$target_path" ]]; then
        echo "[SKIP] target dir missing: $target_path" >&2
        continue
    fi
    if [[ ! -d "$datasets_dir" ]]; then
        echo "[SKIP] few-shot dir missing: $datasets_dir" >&2
        continue
    fi
    echo "=== target=$target  ref=$ref  subdir=$subdir ==="
    python run_omi_cot.py \
        --target_model "$target_path" \
        --ref_model    "$ref" \
        --datasets_dir "$datasets_dir" \
        --out_dir      "$OUT_DIR"
done
