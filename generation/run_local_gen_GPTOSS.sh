#!/bin/bash
##SBATCH -p gpu  # Optional: set your cluster partition
#SBATCH --nodes=1
#SBATCH --job-name=gptoss_gen
#SBATCH --qos=preemptive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

echo "=== GPU Sanity Check ==="
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python -c "import torch; print(f'torch.cuda.device_count() = {torch.cuda.device_count()}')"
echo "========================"

source ~/.bashrc 2>/dev/null || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi
export PYTHONNOUSERSITE=1

cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NUM_WORKERS_MATERIALIZE=1
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var is not set." >&2
  exit 1
fi

# Cache locations. Override these env vars if your cluster needs a shared disk.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "$XDG_CACHE_HOME"

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
mkdir -p "$VLLM_CACHE_ROOT"

# Path to your question JSONL (rows with a "question" field). Override:
#   export QUESTIONS=/path/to/your_questions.jsonl
QUESTIONS="${QUESTIONS:-./questions.jsonl}"
OUT_DIR="${OUT_DIR:-outputs}"
mkdir -p "$OUT_DIR"

python - <<'PY'
import sys
mods = ["torch", "transformers", "numpy", "vllm"]
missing = []
for m in mods:
    try:
        __import__(m)
    except Exception as e:
        missing.append((m, str(e)))
if missing:
    print("Missing/broken packages:")
    for m, e in missing:
        print(f"  {m}: {e}")
    sys.exit(1)
print("Python package check passed.")
PY

echo "=== GPT-OSS 120B ==="
python generate_local_vllm.py \
    --model openai/gpt-oss-120b \
    --questions_path "$QUESTIONS" \
    --out_dir "$OUT_DIR" \
    --max_new_tokens 8192 \
    --max_model_len 12288 \
    --gpu_memory_utilization 0.90 \
    --dtype auto
echo "=== Done ==="
