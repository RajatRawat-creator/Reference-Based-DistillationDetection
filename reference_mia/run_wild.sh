#!/bin/bash
##SBATCH -p gpu                   # EDIT: your cluster partition
##SBATCH --nodelist=NODE          # EDIT (optional): a specific node
#SBATCH --nodes=1
#SBATCH --job-name=ref_mia_wild
#SBATCH --qos=preemptive
#SBATCH --gpus=2                  # 2 GPUs so 70B/32B targets+refs fit
#SBATCH --cpus-per-task=4
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# cd into this script's folder so ../data and ../outputs resolve.
# EDIT for sbatch: $0 is a spool copy under SLURM, so replace the line below with
#   cd /abs/path/to/DistillDetectRelease/reference_mia
cd "$(dirname "$0")"

# >>> Set HF_TOKEN in your environment, or paste it between the quotes below <<<
export HF_TOKEN="${HF_TOKEN:-hf_PASTE_YOUR_TOKEN_HERE}"

# HuggingFace cache dir (override via HF_HOME).
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"  # EDIT: HF cache dir

# Never let a stray ~/.local site-packages shadow the selected conda env
# (matters for the GPT-OSS env which pins transformers/kernels exactly).
export PYTHONNOUSERSITE=1

if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == "hf_PASTE_YOUR_TOKEN_HERE" ]]; then
  echo "ERROR: set your real HF_TOKEN at the top of run_wild.sh." >&2
  exit 1
fi

DATASETS_DIR=${DATASETS_DIR:-"../data/wild"}
OUT_DIR=${OUT_DIR:-"../outputs/reference_mia_wild"}
PAIRS_CSV=${PAIRS_CSV:-"pairs_wild.csv"}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-""}
mkdir -p "$OUT_DIR"
echo "[INFO] PAIRS_CSV=$PAIRS_CSV  OUT_DIR=$OUT_DIR"
if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
  echo "[INFO] ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
fi

# Candidate pool: the default 10-teacher FILES_MAP in run_wild.py is used for the
# R1-distill / s1.1 wild runs. XCoder uses an 11-teacher pool that ADDS
# Qwen-3-235B (one of its true teachers); that pool lives in data/wild as the
# drop-in _files_map_xcoder.json. run_wild.py auto-loads a file literally named
# _files_map.json, so for the XCoder sweep we briefly activate it, then remove it
# (data/wild stays clean = 10-way for every other target). Qwen-3-235B's two
# jsonls sit in data/wild but are unused unless this map is active.
if [[ "$(basename "$PAIRS_CSV")" == "pairs_xcoder.csv" && -f "$DATASETS_DIR/_files_map_xcoder.json" ]]; then
  cp "$DATASETS_DIR/_files_map_xcoder.json" "$DATASETS_DIR/_files_map.json"
  trap 'rm -f "$DATASETS_DIR/_files_map.json"' EXIT
  echo "[INFO] XCoder run: activated 11-candidate pool (Qwen-3-235B included)"
fi

# Python interpreter (override via PYTHON to point at your env).
PYTHON=${PYTHON:-python}   # EDIT: set to your env's python if not on PATH

# Skip header, loop over (target, reference[, dtype]) pairs.
# The optional 3rd column overrides dtype per model; blank -> float16 default.
tail -n +2 "$PAIRS_CSV" | while IFS=, read -r target ref dtype; do
    [[ -z "$target" ]] && continue
    dtype="$(echo "${dtype:-}" | tr -d ' \r\t')"   # strip spaces / CR / tabs
    [[ -z "$dtype" ]] && dtype="float16"           # default
    echo "=== target=$target  ref=$ref  dtype=$dtype ==="
    # Output goes into a per-pair subdir matching ModelsinTheWildOutputs convention
    pair_tag=$(echo "${target}__REF__${ref}" | sed 's|/|__|g')
    pair_out="$OUT_DIR/$pair_tag"
    mkdir -p "$pair_out"
    attn_args=()
    if [[ -n "$ATTN_IMPLEMENTATION" ]]; then
        attn_args=(--attn_implementation "$ATTN_IMPLEMENTATION")
    fi
    # Per-pair guard: a failure (e.g. OOM, unsupported arch) must NOT abort the
    # whole sweep via `set -e`. Log it and move on to the next pair.
    "$PYTHON" run_wild.py \
        --target_model "$target" \
        --ref_model    "$ref" \
        --datasets_dir "$DATASETS_DIR" \
        --out_dir      "$pair_out" \
        --dtype          "$dtype" \
        --device_map     auto \
        --ref_device_map auto \
        --max_memory_per_gpu 138GiB \
        "${attn_args[@]}" \
        || { echo "!!! FAILED pair: target=$target ref=$ref (continuing) !!!"; continue; }
done
