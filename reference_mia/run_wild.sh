#!/bin/bash
##SBATCH -p gpu  # Optional: set your cluster partition
#SBATCH --nodes=1
#SBATCH --job-name=ref_mia_wild
#SBATCH --qos=preemptive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

cd "$(dirname "$0")"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var is not set." >&2
  exit 1
fi

DATASETS_DIR=${DATASETS_DIR:-"../data/wild"}
OUT_DIR=${OUT_DIR:-"../outputs/reference_mia_wild"}
mkdir -p "$OUT_DIR"

# Skip header, loop over (target, reference) pairs
tail -n +2 pairs_wild.csv | while IFS=, read -r target ref; do
    [[ -z "$target" ]] && continue
    echo "=== target=$target  ref=$ref ==="
    # Output goes into a per-pair subdir matching ModelsinTheWildOutputs convention
    pair_tag=$(echo "${target}__REF__${ref}" | sed 's|/|__|g')
    pair_out="$OUT_DIR/$pair_tag"
    mkdir -p "$pair_out"
    python run_wild.py \
        --target_model "$target" \
        --ref_model    "$ref" \
        --datasets_dir "$DATASETS_DIR" \
        --out_dir      "$pair_out"
done
