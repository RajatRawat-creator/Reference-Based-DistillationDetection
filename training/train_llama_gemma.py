"""
SFT training: Llama-3.2-3B-Instruct + Gemma-3-4B-PT students.

Reads every *.jsonl file under data/training/ and runs SFT once per
(student, dataset) pair. Each student carries its own template:

    Llama-3.2-3B-Instruct : tokenizer.apply_chat_template (instruct-style)
    Gemma-3-4B-PT         : plain text "Problem:\\n{question}\\n\\nSolution:\\n"

Output directory naming:
    Student={student_prefix}_{jsonl_basename}

Configuration:
    Edit STUDENT_CONFIGS / DATASETS_DIR / OUTPUT_BASE_DIR below. Gated
    Hugging Face models (Llama, Gemma) require HF_TOKEN in env.

NOTE: Gemma has a 262K-token vocab; long sequences trigger OOM on
single-GPU H200, so block_size is held at 4096 for Gemma too.

Equivalent to the original FINALGITHUBALLSCRIPTS/training_script/run_b.py.
"""

import os, sys, gc, torch
from dataclasses import dataclass, field
from typing import Optional
import warnings; warnings.filterwarnings("ignore", category=FutureWarning)
import logging; logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
from datasets import load_dataset
import transformers
from transformers import DataCollatorForSeq2Seq
import trl
from huggingface_hub import login

# -----------------------------------------------------------------------------
# 0. Setup & Authentication
# -----------------------------------------------------------------------------
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(token=_hf_token)
else:
    logging.warning("HF_TOKEN not set in env; gated Llama/Gemma downloads will fail.")

# =====================================================================
# Config
# =====================================================================
# Llama runs first, then Gemma.
# Gemma's lower block_size avoids 262K-vocab OOM on long seqs.
STUDENT_CONFIGS = [
    #{
    #    "model_name": "meta-llama/Llama-3.2-3B-Instruct",
    #    "prefix": "Student=Llama-3.2-3B-Instruct",
    #    "use_chat_template": True,
    #    "needs_token_type_ids": False,
    #    "block_size": 4096,
    #},
    {
        "model_name": "google/gemma-3-4b-pt",
        "prefix": "Student=Gemma-3-4B-PT",
        "use_chat_template": False,
        "needs_token_type_ids": True,
        "block_size": 4096,
    },
]

# Repo-relative defaults. Override via env vars to point elsewhere.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)

DATASETS_DIR    = os.environ.get(
    "SFT_DATASETS_DIR",
    os.path.join(_REPO_ROOT, "data", "training"),
)
OUTPUT_BASE_DIR = os.environ.get(
    "SFT_OUTPUT_DIR_LLAMA_GEMMA",
    # Unified root: both train_*.py write here so reference_mia's single
    # CHECKPOINTS_DIR (default ../checkpoints) resolves Student=X/... directly.
    os.path.join(_REPO_ROOT, "checkpoints"),
)

LEARNING_RATE         = "1e-5"
NUM_EPOCHS            = "3"
PER_DEVICE_BATCH_SIZE = "4"
GRADIENT_ACCUM_STEPS  = "4"
WARMUP_RATIO          = "0.05"
LR_SCHEDULER          = "cosine"
# =====================================================================


@dataclass
class TrainingConfig:
    model_name: str = field(default="")
    block_size: int = field(default=65536)
    train_file_path: Optional[str] = field(default=None)


def train_on_dataset(model_name: str, file_path: str, output_dir: str,
                     use_chat_template: bool = False,
                     needs_token_type_ids: bool = False,
                     block_size: int = 65536):
    logging.info(f"{'='*60}")
    logging.info(f"  Student : {model_name}")
    logging.info(f"  Template: {'chat' if use_chat_template else 'plain-text'}")
    logging.info(f"  Block   : {block_size}")
    logging.info(f"  Dataset : {file_path}")
    logging.info(f"  Output  : {output_dir}")
    logging.info(f"{'='*60}")

    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))

    script_args = [
        "--output_dir",                  output_dir,
        "--model_name",                  model_name,
        "--train_file_path",             file_path,
        "--block_size",                  str(block_size),
        "--learning_rate",               LEARNING_RATE,
        "--num_train_epochs",            NUM_EPOCHS,
        "--per_device_train_batch_size", PER_DEVICE_BATCH_SIZE,
        "--gradient_accumulation_steps", GRADIENT_ACCUM_STEPS,
        "--bf16",                        "True",
        "--gradient_checkpointing",      "True",
        "--logging_steps",               "1",
        "--report_to",                   "none",
        "--lr_scheduler_type",           LR_SCHEDULER,
        "--warmup_ratio",                WARMUP_RATIO,
        "--save_strategy",               "no",
    ]

    # Gemma needs token_type_ids passed through to the model
    if needs_token_type_ids:
        script_args += ["--remove_unused_columns", "False"]

    config, args = parser.parse_args_into_dataclasses(args=script_args)

    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto",
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_name, use_fast=True, trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if use_chat_template:
        tokenizer.padding_side = "right"
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files=file_path, split="train")
    logging.info(f"Loaded {len(dataset)} examples")

    # -- Plain-text template (Gemma / base models) --
    def process_plain(example):
        prompt_text = f"Problem:\n{example['question']}\n\nSolution:\n"
        answer_text = str(example['response'])

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        eos_id = tokenizer.eos_token_id

        input_ids = prompt_ids + answer_ids + ([eos_id] if eos_id is not None else [])
        labels    = [-100]*len(prompt_ids) + answer_ids + ([eos_id] if eos_id is not None else [])

        if len(input_ids) > block_size:
            input_ids = input_ids[:block_size]
            labels    = labels[:block_size]

        result = {"input_ids": input_ids, "attention_mask": [1]*len(input_ids), "labels": labels}

        # Gemma 3 requires token_type_ids during training (all 0 for text-only)
        if needs_token_type_ids:
            result["token_type_ids"] = [0] * len(input_ids)

        return result

    # -- Chat template (Llama instruct) --
    def process_chat(example):
        q = example.get("question") or ""
        r = example.get("response") or ""

        full_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}, {"role": "assistant", "content": r}],
            tokenize=False,
        )
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True,
        )

        full_ids   = tokenizer(full_text,   add_special_tokens=False)["input_ids"]
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        labels = full_ids.copy()
        m = min(len(prompt_ids), len(labels))
        labels[:m] = [-100] * m

        if len(full_ids) > block_size:
            full_ids = full_ids[:block_size]
            labels   = labels[:block_size]

        return {"input_ids": full_ids, "attention_mask": [1]*len(full_ids), "labels": labels}

    process_func = process_chat if use_chat_template else process_plain

    logging.info("Tokenizing...")
    tokenized = dataset.map(process_func, remove_columns=dataset.column_names)

    seq_lens = [len(ex["input_ids"]) for ex in tokenized]
    logging.info(f"Seq lengths: min={min(seq_lens)}, median={sorted(seq_lens)[len(seq_lens)//2]}, max={max(seq_lens)}")
    logging.info(f"Truncated at {block_size}: {sum(1 for l in seq_lens if l == block_size)}/{len(seq_lens)}")

    args.max_seq_length = block_size
    trainer = trl.SFTTrainer(
        model, train_dataset=tokenized, args=args,
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8),
    )

    logging.info("Training...")
    trainer.train()
    trainer.save_model(output_dir=output_dir)
    tokenizer.save_pretrained(output_dir)
    logging.info(f"Saved to {output_dir}")

    del model, trainer, tokenizer, tokenized, dataset
    gc.collect(); torch.cuda.empty_cache()


def main():
    if not os.path.isdir(DATASETS_DIR):
        logging.error(f"DATASETS_DIR does not exist: {DATASETS_DIR}"); sys.exit(1)

    jsonl_files = sorted(f for f in os.listdir(DATASETS_DIR) if f.endswith(".jsonl"))
    if not jsonl_files:
        logging.error(f"No .jsonl files in {DATASETS_DIR}"); sys.exit(1)

    total = len(STUDENT_CONFIGS) * len(jsonl_files)
    logging.info(f"{len(STUDENT_CONFIGS)} students x {len(jsonl_files)} datasets = {total} runs")
    for jf in jsonl_files:
        logging.info(f"  - {jf}")

    for student in STUDENT_CONFIGS:
        student_dir = os.path.join(OUTPUT_BASE_DIR, student["prefix"])
        os.makedirs(student_dir, exist_ok=True)

        for jf in jsonl_files:
            dataset_id = jf[:-len(".jsonl")]
            run_name   = f"{student['prefix']}_{dataset_id}"
            output_dir = os.path.join(student_dir, run_name)

            if os.path.exists(output_dir) and os.listdir(output_dir):
                logging.info(f"Skipping {run_name} -- exists"); continue

            train_on_dataset(
                student["model_name"],
                os.path.join(DATASETS_DIR, jf),
                output_dir,
                use_chat_template=student["use_chat_template"],
                needs_token_type_ids=student["needs_token_type_ids"],
                block_size=student["block_size"],
            )

    logging.info("All Llama + Gemma runs complete!")


main()
