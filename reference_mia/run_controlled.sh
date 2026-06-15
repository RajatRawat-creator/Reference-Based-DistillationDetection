#!/bin/bash
##SBATCH -p gpu                   # EDIT: your cluster partition
##SBATCH --nodelist=NODE          # EDIT (optional): a specific node
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

cd "$(dirname "$0")"   # EDIT for sbatch: cd /abs/path/to/DistillDetectRelease/reference_mia

export HF_TOKEN="${HF_TOKEN:-hf_PASTE_YOUR_TOKEN_HERE}"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"  # EDIT: HF cache dir
export PYTHONNOUSERSITE=1

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var is not set." >&2
  exit 1
fi

# Roots for controlled checkpoints and MIA scoring inputs.
# CHECKPOINTS_DIR points at the REAL post-trained tree on /data; pairs targets
# are paths RELATIVE to it (keep the Student=X/ or OMI_COT_Models/ prefix), so
# $CHECKPOINTS_DIR/$target resolves directly.
CHECKPOINTS_DIR=${CHECKPOINTS_DIR:-"../checkpoints"}   # EDIT: where train_*.sh wrote the students
DATASETS_DIR=${DATASETS_DIR:-"../data/MIADatasets"}
OUT_DIR=${OUT_DIR:-"../outputs/reference_mia_controlled"}
PAIRS_CSV=${PAIRS_CSV:-"pairs_controlled_data.csv"}
PYTHON=${PYTHON:-python}   # EDIT: set to your env's python if not on PATH
mkdir -p "$OUT_DIR"
echo "[INFO] CHECKPOINTS_DIR=$CHECKPOINTS_DIR  PAIRS_CSV=$PAIRS_CSV  OUT_DIR=$OUT_DIR"

# Skip header, loop over (target, reference[, dtype]) pairs.
# Optional 3rd column overrides dtype per model; blank -> float16 (Gemma needs bf16).
tail -n +2 "$PAIRS_CSV" | while IFS=, read -r target ref dtype; do
    [[ -z "$target" ]] && continue
    dtype="$(echo "${dtype:-}" | tr -d ' \r\t')"
    [[ -z "$dtype" ]] && dtype="float16"
    target_path="$CHECKPOINTS_DIR/$target"
    if [[ ! -d "$target_path" ]]; then
        echo "[SKIP] target dir missing: $target_path" >&2
        continue
    fi
    echo "=== target=$target  ref=$ref  dtype=$dtype ==="
    "$PYTHON" run_controlled.py \
        --target_model "$target_path" \
        --ref_model    "$ref" \
        --datasets_dir "$DATASETS_DIR" \
        --out_dir      "$OUT_DIR" \
        --dtype          "$dtype" \
        --device_map     auto \
        --ref_device_map auto \
        --max_memory_per_gpu 138GiB \
        || { echo "!!! FAILED pair: target=$target ref=$ref (continuing) !!!"; continue; }
done
