#!/bin/bash
##SBATCH -p gpu  # Optional: set your cluster partition
#SBATCH --nodes=1
#SBATCH --job-name=local_gen_tbd
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
  echo "ERROR: HF_TOKEN is not set. export HF_TOKEN=hf_... before running."
  exit 1
fi

# Cache locations. Override these env vars if your cluster needs a shared disk.
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${HOME}/.cache}"
mkdir -p "$XDG_CACHE_HOME"

# Hugging Face cache.
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

# vLLM compile cache.
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-${HOME}/.cache/vllm}"
mkdir -p "$VLLM_CACHE_ROOT"

# Path to your question JSONL (rows with a "question" field). Override:
#   export QUESTIONS=/path/to/your_questions.jsonl
QUESTIONS="${QUESTIONS:-./questions.jsonl}"
OUT_DIR="${OUT_DIR:-outputs}"
mkdir -p "$OUT_DIR"

echo "=== Local vLLM Generation ==="
echo "Questions:  $QUESTIONS"
echo "Output dir: $OUT_DIR"
echo "=============================="

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

echo "=== Model 1: Gemma 3 27B IT ==="
python generate_local_vllm.py \
    --model google/gemma-3-27b-it \
    --questions_path "$QUESTIONS" \
    --out_dir "$OUT_DIR" \
    --max_new_tokens 4096 \
    --max_model_len 8192 \
    --gpu_memory_utilization 0.90 \
    --dtype bfloat16
echo "=== Gemma 3 done ==="

echo "=== Model 2: Llama 3.3 70B FP8 ==="
python generate_local_vllm.py \
    --model nvidia/Llama-3.3-70B-Instruct-FP8 \
    --questions_path "$QUESTIONS" \
    --out_dir "$OUT_DIR" \
    --max_new_tokens 4096 \
    --max_model_len 8192 \
    --gpu_memory_utilization 0.90 \
    --dtype auto
echo "=== Llama 3.3 done ==="

echo "=== All done. Outputs in: $OUT_DIR ==="
