#!/bin/bash
##SBATCH -p gpu  # Optional: set your cluster partition
#SBATCH --qos=preemptive
#SBATCH --job-name=fewshot_all_models
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Generate the OMI-CoT few-shot reference-MIA candidate datasets for ALL four
# candidate teachers, over every per-student few-shot prompt in FewShotPrompts/.
# Output lands in one subfolder per prompt (= per OMI-CoT student), matching
# data/omi_cot_fewshot/. Equivalent to the original
# scripts/sbatch_run_.sbatch (+ scripts/sbatch_generateData_fewshot.sbatch),
# with paths made relative and the HF token read from the environment.

set -euo pipefail
mkdir -p logs

# Gated teacher models (Llama, Gemma) require a Hugging Face token.
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set. export HF_TOKEN=hf_... before running."
  exit 1
fi

source ~/.bashrc 2>/dev/null || true
# conda activate <your_vllm_env>      # uncomment + set your env

cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export VLLM_USE_V1=0

SCRIPT="./generate_teacher_vllm_fewshot.py"
PROMPT_DIR="${PROMPT_DIR:-./FewShotPrompts}"
OUT_DIR="${OUT_DIR:-../data/omi_cot_fewshot}"

# Question source: "s1k" (loads simplescaling/s1K from HF; matches the shipped
# data/omi_cot_fewshot/) or "openmath_jsonl" (needs OPENMATH_PATH below).
INPUT_SOURCE="${INPUT_SOURCE:-s1k}"
OPENMATH_PATH="${OPENMATH_PATH:-./DATASETS/OpenMathInstruct1KQuestions.jsonl}"

LIMIT="${LIMIT:-200}"
MAX_TOKENS="${MAX_TOKENS:-2048}"

mkdir -p "$OUT_DIR"

# "HF model | tensor-parallel | gpu_util | max_model_len | batch_size"
MODELS=(
  "nvidia/Llama-3.3-70B-Instruct-FP8|2|0.90|8112|24"
  "Qwen/Qwen3-8B|1|0.90|8112|48"
  "google/gemma-3-27b-it|2|0.90|8112|32"
  "openai/gpt-oss-120b|2|0.85|8112|16"
)

if [[ ! -f "$SCRIPT" ]]; then
  echo "ERROR: generator not found: $SCRIPT"
  exit 1
fi

for MODEL_SPEC in "${MODELS[@]}"; do
  IFS='|' read -r MODEL_NAME TP GPU_UTIL MAX_LEN BATCH_SIZE <<< "$MODEL_SPEC"

  echo "============================================================"
  echo "MODEL: $MODEL_NAME"
  echo "TP: $TP | GPU_UTIL: $GPU_UTIL | MAX_LEN: $MAX_LEN | BATCH: $BATCH_SIZE"
  echo "Input source: $INPUT_SOURCE | out: $OUT_DIR"
  echo "============================================================"

  for PROMPT_FILE in "$PROMPT_DIR"/*.txt; do
    echo "------------------------------------------------------------"
    echo "PROMPT FILE: $PROMPT_FILE"
    echo "------------------------------------------------------------"

    python "$SCRIPT" \
      --teacher_model "$MODEL_NAME" \
      --input_source "$INPUT_SOURCE" \
      --fewshot_prompt_file "$PROMPT_FILE" \
      --openmath_path "$OPENMATH_PATH" \
      --out_dir "$OUT_DIR" \
      --tp "$TP" \
      --gpu_util "$GPU_UTIL" \
      --max_model_len "$MAX_LEN" \
      --batch_size "$BATCH_SIZE" \
      --max_tokens "$MAX_TOKENS" \
      --limit "$LIMIT"
  done
done

echo
echo "============================================================"
echo "Few-shot generation complete. Outputs in: $OUT_DIR"
echo "============================================================"
