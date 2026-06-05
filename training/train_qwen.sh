#!/bin/bash
# Slurm launcher for train_qwen.py.
# Submit with: sbatch training/train_qwen.sh
# Required env: HF_TOKEN (for gated models). Optional: SFT_DATASETS_DIR, SFT_OUTPUT_DIR.

##SBATCH -p gpu  # Optional: set your cluster partition
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --qos=preemptive
#SBATCH -J train_qwen_sft
#SBATCH -o logs/train_qwen_%j.log
#SBATCH -e logs/train_qwen_%j.err

set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"

mkdir -p "${REPO_ROOT}/logs"

# ---- env ----
export VLLM_USE_V1=0
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}}"

# Token must be exported in your shell before submission, e.g.
#   export HF_TOKEN=hf_...
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var is not set." >&2
  exit 1
fi

mkdir -p "$HF_HOME" "$VLLM_CACHE_ROOT"

if [[ -n "${CONDA_SH:-}" ]]; then
  source "$CONDA_SH"
fi
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi

python - <<'PY'
import torch
print("torch cuda available:", torch.cuda.is_available())
print("torch cuda count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("bf16 supported:", torch.cuda.is_bf16_supported())
PY

pip install --quiet trl transformers datasets accelerate sentencepiece protobuf

echo "=========================================="
echo "SFT training only: train_qwen"
echo "=========================================="

python "${SCRIPT_DIR}/train_qwen.py"

echo "Training complete: train_qwen"
