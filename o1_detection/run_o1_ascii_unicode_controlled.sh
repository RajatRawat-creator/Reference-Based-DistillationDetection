#!/bin/bash
##SBATCH -p gpu                   # EDIT: your cluster partition
##SBATCH --nodelist=NODE          # EDIT (optional): a specific node
#SBATCH --job-name=refMIA_O1_controlled
#SBATCH --qos=preemptive
#SBATCH --gpus=1                   # targets+refs are all <=4B, fit one H200
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# o1 ASCII-vs-Unicode reference MIA for the 4 models we SFT'd on o1 data.
# Same script (run_o1_ascii_unicode.py), same datasets (../data/o1), same params
# as run_o1_ascii_unicode.sh — only the target/reference list differs.
# Each target = an o1-distilled student on /data; reference = its base model.

set -uo pipefail
mkdir -p logs

PYTHON=${PYTHON:-python}   # EDIT: set to your env's python if not on PATH
export PYTHONNOUSERSITE=1

echo "=== GPU Sanity Check ==="
echo "HOSTNAME=$(hostname)"
echo "PYTHON=$PYTHON"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

# Under sbatch, $0 is a copy in SLURM's spool dir, so `dirname $0` is wrong.
cd "$(dirname "$0")"   # EDIT for sbatch: cd /abs/path/to/DistillDetectRelease/o1_detection

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF=expandable_segments:True
export HF_TOKEN="${HF_TOKEN:-hf_PASTE_YOUR_TOKEN_HERE}"
if [[ -z "${HF_TOKEN:-}" || "${HF_TOKEN}" == "hf_PASTE_YOUR_TOKEN_HERE" ]]; then
  echo "ERROR: set your real HF_TOKEN (export it, or replace the placeholder above)." >&2
  exit 1
fi
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"  # EDIT: HF cache dir
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
mkdir -p "$HF_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE"

SCRIPT_PATH="./run_o1_ascii_unicode.py"
DATASETS_DIR="${DATASETS_DIR:-../data/o1}"
BASE_OUTDIR="${BASE_OUTDIR:-../outputs/o1_detection_controlled}"
mkdir -p "$BASE_OUTDIR"

DTYPE=bfloat16                    # default; Gemma needs bf16, others fine in bf16
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

CKPT_ROOT="${CKPT_ROOT:-../checkpoints}"   # EDIT: where train_*.sh wrote the students

# target (full /data path) | reference (base model)
MODELS=(
  "$CKPT_ROOT/Student=Qwen-2.5-1.5B/Student=Qwen-2.5-1.5B_o1__s1k__chat__openai_responses|Qwen/Qwen2.5-1.5B"
  "$CKPT_ROOT/Student=Qwen-2.5-3B/Student=Qwen-2.5-3B_o1__s1k__chat__openai_responses|Qwen/Qwen2.5-3B"
  "$CKPT_ROOT/Student=Gemma-3-4B-PT/Student=Gemma-3-4B-PT_o1__s1k__chat__openai_responses|google/gemma-3-4b-pt"
  "$CKPT_ROOT/Student=Llama-3.2-3B-Instruct/Student=Llama-3.2-3B-Instruct_o1__s1k__chat__openai_responses|meta-llama/Llama-3.2-3B-Instruct"
)

echo "=== Controlled o1-distillation ASCII/Unicode MIA ==="
echo "Script:      $SCRIPT_PATH   Datasets: $DATASETS_DIR   Outdir: $BASE_OUTDIR"
echo "num pairs:   ${#MODELS[@]}"
echo

echo "=== Dataset file check ==="
for fname in "o1_openmath__responses_unicode.jsonl" "o1__responses_ascii.jsonl"; do
    if [[ -f "$DATASETS_DIR/$fname" ]]; then
        echo "  ✓ $fname ($(wc -l < "$DATASETS_DIR/$fname") lines)"
    else
        echo "  ✗ MISSING: $fname"
    fi
done
echo

for pair in "${MODELS[@]}"; do
  IFS="|" read -r TARGET REF PAIR_DTYPE <<< "$pair"
  PAIR_DTYPE="${PAIR_DTYPE:-$DTYPE}"

  if [[ ! -d "$TARGET" ]]; then
    echo "[SKIP] target dir missing: $TARGET" >&2; continue
  fi
  # Tag the per-pair output dir by the student's leaf name (paths are long).
  T_TAG="$(basename "$TARGET")"
  R_TAG="$(echo "$REF" | sed 's#[/:= ]#__#g')"
  OUTDIR="$BASE_OUTDIR/${T_TAG}__REF__${R_TAG}"
  mkdir -p "$OUTDIR"

  echo "------------------------------------------------------------"
  echo "TARGET:    $TARGET"
  echo "REFERENCE: $REF"
  echo "DTYPE:     $PAIR_DTYPE"
  echo "OUTDIR:    $OUTDIR"
  echo "------------------------------------------------------------"

  # Our SFT checkpoints saved a malformed tokenizer_config (extra_special_tokens
  # as a list) that breaks AutoTokenizer; load the target tokenizer from the
  # base model instead (identical vocab, so tokenization is unchanged).
  "$PYTHON" "$SCRIPT_PATH" \
    --target_model "$TARGET" \
    --target_tokenizer "$REF" \
    --ref_model "$REF" \
    --datasets_dir "$DATASETS_DIR" \
    --out_dir "$OUTDIR" \
    --dtype "$PAIR_DTYPE" \
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

  echo "  ✅ Finished: $(basename "$TARGET") vs $REF"
  echo
done

echo "=== Done. Outputs under: $BASE_OUTDIR ==="
