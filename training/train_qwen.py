"""
SFT training: Qwen-2.5-1.5B + Qwen-2.5-3B students.

Reads every *.jsonl file under data/training/ and runs SFT once per
(student, dataset) pair, producing 2 x N trained model directories.

Prompt template (plain text):
    Problem:
    {question}

    Solution:
    {response}

Output directory naming:
    Student={student_prefix}_{jsonl_basename}

Configuration:
    Edit the constants block below to change students, dataset dir,
    output dir, or hyperparameters. Authentication for gated Hugging
    Face models is read from the HF_TOKEN environment variable.

Equivalent to the original FINALGITHUBALLSCRIPTS/training_script/run_a.py.
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

# =====================================================================
# Config
# =====================================================================
STUDENT_CONFIGS = [
    {"model_name": "Qwen/Qwen2.5-1.5B",  "prefix": "Student=Qwen-2.5-1.5B"},
    {"model_name": "Qwen/Qwen2.5-3B",    "prefix": "Student=Qwen-2.5-3B"},
]

# Repo-relative defaults. Override via env vars if your data/checkpoints
# live elsewhere on disk.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)

DATASETS_DIR    = os.environ.get(
    "SFT_DATASETS_DIR",
    os.path.join(_REPO_ROOT, "data", "training"),
)
OUTPUT_BASE_DIR = os.environ.get(
    "SFT_OUTPUT_DIR",
    # Unified root: both train_*.py write here so reference_mia's single
    # CHECKPOINTS_DIR (default ../checkpoints) resolves Student=X/... directly.
    os.path.join(_REPO_ROOT, "checkpoints"),
)

BLOCK_SIZE            = 4096
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
    block_size: int = field(default=BLOCK_SIZE)
    train_file_path: Optional[str] = field(default=None)


def train_on_dataset(model_name: str, file_path: str, output_dir: str):
    logging.info(f"{'='*60}")
    logging.info(f"  Student : {model_name}")
    logging.info(f"  Dataset : {file_path}")
    logging.info(f"  Output  : {output_dir}")
    logging.info(f"{'='*60}")

    parser = transformers.HfArgumentParser((TrainingConfig, trl.SFTConfig))
    config, args = parser.parse_args_into_dataclasses(args=[
        "--output_dir",                  output_dir,
        "--model_name",                  model_name,
        "--train_file_path",             file_path,
        "--block_size",                  str(BLOCK_SIZE),
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
    ])

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
    model.config.pad_token_id = tokenizer.pad_token_id

    dataset = load_dataset("json", data_files=file_path, split="train")
    logging.info(f"Loaded {len(dataset)} examples")

    def process_func(example):
        prompt_text = f"Problem:\n{example['question']}\n\nSolution:\n"
        answer_text = str(example['response'])

        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
        eos_id = tokenizer.eos_token_id

        input_ids = prompt_ids + answer_ids + ([eos_id] if eos_id is not None else [])
        labels    = [-100]*len(prompt_ids) + answer_ids + ([eos_id] if eos_id is not None else [])

        if len(input_ids) > config.block_size:
            input_ids = input_ids[:config.block_size]
            labels    = labels[:config.block_size]

        return {"input_ids": input_ids, "attention_mask": [1]*len(input_ids), "labels": labels}

    logging.info("Tokenizing...")
    tokenized = dataset.map(process_func, remove_columns=dataset.column_names)

    seq_lens = [len(ex["input_ids"]) for ex in tokenized]
    logging.info(f"Seq lengths: min={min(seq_lens)}, median={sorted(seq_lens)[len(seq_lens)//2]}, max={max(seq_lens)}")
    logging.info(f"Truncated at {config.block_size}: {sum(1 for l in seq_lens if l == config.block_size)}/{len(seq_lens)}")

    args.max_seq_length = config.block_size
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
            dataset_id = jf[:-len(".jsonl")]                      # e.g. Teacher=GPT-OSS-120B_Data=OMI(1K)_Template=Chat
            run_name   = f"{student['prefix']}_{dataset_id}"      # e.g. Student=Qwen-2.5-1.5B_Teacher=...
            output_dir = os.path.join(student_dir, run_name)

            if os.path.exists(output_dir) and os.listdir(output_dir):
                logging.info(f"Skipping {run_name} -- exists"); continue

            train_on_dataset(student["model_name"], os.path.join(DATASETS_DIR, jf), output_dir)

    logging.info("All Qwen runs complete!")


main()
