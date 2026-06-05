#!/bin/bash
##SBATCH -p gpu  # Optional: set your cluster partition
#SBATCH --job-name=refMIA_O1_R1Distill_s11
#SBATCH --qos=preemptive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --time=96:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -uo pipefail
mkdir -p logs

source ~/.bashrc 2>/dev/null || true
if [[ -n "${CONDA_ENV:-}" ]]; then
  conda activate "$CONDA_ENV"
fi
export PYTHONNOUSERSITE=1

echo "=== GPU Sanity Check ==="
echo "HOSTNAME=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-not set}"
which python
nvidia-smi --query-gpu=index,name,memory.total --format=csv
python - <<'PY'
import sys, torch
print("python:", sys.executable)
print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("torch file:", torch.__file__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
x = torch.zeros(1, device="cuda:0")
print("cuda tensor ok:", x.device)
PY
echo "========================"

cd "$(dirname "$0")"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_NUM_WORKERS_MATERIALIZE=1
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN env var is not set." >&2
  exit 1
fi

export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

SCRIPT_PATH="./run_o1_ascii_unicode.py"
DATASETS_DIR="${DATASETS_DIR:-../data/o1}"
BASE_OUTDIR="${BASE_OUTDIR:-../outputs/o1_detection}"
mkdir -p "$BASE_OUTDIR"

DTYPE=bfloat16
DEVICE_MAP=auto
REF_DEVICE_MAP=cuda:0
MAX_MEMORY_PER_GPU=130GiB
MAX_MEMORY_CPU=80GiB

MAX_LENGTH=32768
MAX_ANSWER_TOKENS=4000
ANSWER_TRUNCATION_SIDE=right
STRIDE=512
PROMPT_PREFIX_TOKENS=3000
LIMIT_PER_DATASET=200

safe_tag () {
  echo "$1" | sed 's#[/:= ]#__#g' | sed 's#[()]##g'
}

MODELS=(
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B|Qwen/Qwen2.5-Math-1.5B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B|Qwen/Qwen2.5-Math-7B"
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B|meta-llama/Llama-3.1-8B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B|Qwen/Qwen2.5-14B"
  "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B|Qwen/Qwen2.5-32B"
  "simplescaling/s1.1-32B|Qwen/Qwen2.5-32B-Instruct"

  "meta-llama/Llama-3.1-8B-Instruct|meta-llama/Meta-Llama-3-8B-Instruct"
  "google/gemma-2-9b-it|google/gemma-1.1-7b-it"
  "google/gemma-3-12b-it|google/gemma-2-9b-it"
  "google/gemma-3-27b-it|google/gemma-2-27b-it"
  "google/gemma-2-9b|google/gemma-7b"
  "google/gemma-3-27b-pt|google/gemma-2-27b"

  "meta-llama/Llama-3.1-8B|meta-llama/Meta-Llama-3-8B"
  "openai/gpt-oss-20b|gpt2-xl"
  "meta-llama/Llama-3.1-70B|meta-llama/Meta-Llama-3-70B"
  "meta-llama/Llama-3.3-70B-Instruct|meta-llama/Llama-3.1-70B-Instruct"

  "deepseek-ai/DeepSeek-R1-Distill-Llama-70B|meta-llama/Llama-3.3-70B-Instruct"
  "openai/gpt-oss-120b|gpt2-xl"
  "openai/gpt-oss-120b|openai/gpt-oss-20b"
)

echo "=== Starting O1 + R1-Distill + s1.1-32B MIA sweep ==="
echo "Script:       $SCRIPT_PATH"
echo "Datasets:     $DATASETS_DIR"
echo "Base outdir:  $BASE_OUTDIR"
echo "dtype=$DTYPE device_map=$DEVICE_MAP ref_device_map=$REF_DEVICE_MAP"
echo "max_memory_per_gpu=$MAX_MEMORY_PER_GPU max_memory_cpu=$MAX_MEMORY_CPU"
echo "max_length=$MAX_LENGTH max_answer_tokens=$MAX_ANSWER_TOKENS"
echo "answer_truncation_side=$ANSWER_TRUNCATION_SIDE stride=$STRIDE prompt_prefix_tokens=$PROMPT_PREFIX_TOKENS"
echo "limit_per_dataset=$LIMIT_PER_DATASET"
echo "num pairs:    ${#MODELS[@]}"
echo

python - <<'PY'
import sys
mods = ["torch", "transformers", "numpy", "matplotlib", "huggingface_hub"]
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

echo "=== Dataset file check ==="
for fname in \
    "o1_openmath__responses_unicode.jsonl" \
    "o1__responses_default.jsonl"
do
    if [[ -f "$DATASETS_DIR/$fname" ]]; then
        nlines=$(wc -l < "$DATASETS_DIR/$fname")
        echo "  ✓ $fname ($nlines lines)"
    else
        echo "  ✗ MISSING: $fname"
    fi
done
echo

for pair in "${MODELS[@]}"; do
  IFS="|" read -r TARGET REF <<< "$pair"

  T_TAG="$(safe_tag "$TARGET")"
  R_TAG="$(safe_tag "$REF")"
  OUTDIR="$BASE_OUTDIR/${T_TAG}__REF__${R_TAG}"
  mkdir -p "$OUTDIR"

  echo "------------------------------------------------------------"
  echo "TARGET:    $TARGET"
  echo "REFERENCE: $REF"
  echo "OUTDIR:    $OUTDIR"
  echo "------------------------------------------------------------"

  python "$SCRIPT_PATH" \
    --target_model "$TARGET" \
    --ref_model "$REF" \
    --datasets_dir "$DATASETS_DIR" \
    --out_dir "$OUTDIR" \
    --dtype "$DTYPE" \
    --device_map "$DEVICE_MAP" \
    --ref_device_map "$REF_DEVICE_MAP" \
    --max_memory_per_gpu "$MAX_MEMORY_PER_GPU" \
    --max_memory_cpu "$MAX_MEMORY_CPU" \
    --max_length "$MAX_LENGTH" \
    --max_answer_tokens "$MAX_ANSWER_TOKENS" \
    --answer_truncation_side "$ANSWER_TRUNCATION_SIDE" \
    --stride "$STRIDE" \
    --prompt_prefix_tokens "$PROMPT_PREFIX_TOKENS" \
    --limit_per_dataset "$LIMIT_PER_DATASET" \
    --trust_remote_code_ref \
    --trust_remote_code_tgt \
    || { echo "  ❌ FAILED: $TARGET vs $REF — continuing sweep"; echo; continue; }

  echo "  ✅ Finished: $TARGET vs $REF"
  echo
done

echo "=== Done. Outputs under: $BASE_OUTDIR ==="
