#!/bin/bash
#SBATCH --job-name=moe16b_loss
##SBATCH --partition=gpu          # EDIT: your cluster partition
##SBATCH --nodelist=NODE          # EDIT (optional): a specific node
#SBATCH --qos=preemptive
#SBATCH --gpus=1                 # deepseek-moe-16b-base ~33GB bf16, fits 1x H200
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# DeepSeek-MoE-16B-base single-model loss over BOTH pipelines:
#   1) wild reference-MIA datasets   (../data/wild,  run_moe16b_wild_loss.py)
#   2) o1 ASCII-vs-Unicode datasets  (../data/o1,    ../o1_detection/run_moe16b_o1_loss.py)
#
# Usage:
#   sbatch run_moe16b.sh            # full run (LIMIT_PER_DATASET=200)
#   LIMIT_PER_DATASET=2 sbatch run_moe16b.sh   # quick smoke test first
#   WHICH=wild sbatch run_moe16b.sh / WHICH=o1 sbatch run_moe16b.sh  # one half only

set -euo pipefail
mkdir -p logs

# EDIT: activate your environment, e.g.:
#   source ~/miniconda3/etc/profile.d/conda.sh && conda activate <your_env>

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"  # EDIT: HF cache dir
export HF_HUB_CACHE=$HF_HOME/hub
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HUB_CACHE
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export HF_TOKEN="${HF_TOKEN:-hf_PASTE_YOUR_TOKEN_HERE}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"

# DeepSeek remote-code load: prefer the portable Triton/PyTorch FP8 fallback.
export DISABLE_KERNEL_MAPPING=1
export HF_HUB_DISABLE_KERNELS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export PYTHONNOUSERSITE=1
export LIMIT_PER_DATASET="${LIMIT_PER_DATASET:-200}"

WHICH="${WHICH:-both}"
RM="$(cd "$(dirname "$0")" && pwd)"            # this script's dir (reference_mia); EDIT for sbatch
O1="$(cd "$RM/../o1_detection" && pwd)"

echo "[sbatch] host=$(hostname) WHICH=$WHICH LIMIT_PER_DATASET=$LIMIT_PER_DATASET at $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || true

if [[ "$WHICH" == "wild" || "$WHICH" == "both" ]]; then
  echo "=== [1/2] wild reference-MIA losses ==="
  cd "$RM"
  python -u run_moe16b_wild_loss.py
fi

if [[ "$WHICH" == "o1" || "$WHICH" == "both" ]]; then
  echo "=== [2/2] o1 ASCII-vs-Unicode losses ==="
  cd "$O1"
  python -u run_moe16b_o1_loss.py
fi

echo "[sbatch] finished at $(date)"
