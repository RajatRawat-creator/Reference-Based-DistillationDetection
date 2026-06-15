#!/usr/bin/env python3
"""
Single-model (loss-only) run of deepseek-ai/deepseek-moe-16b-base over the SAME
wild datasets and with the SAME loss machinery as run_wild.py (so the numbers are
directly comparable to yesterday's other-model wild losses).

We reuse run_wild.py's FILES_MAP, ModelWrapper (stride loss), load_jsonl and
resolve_dataset_file verbatim — only the model is different and there is no
reference subtraction (we just emit MoE's per-example loss per dataset).

IMPORTANT: run against ../data/wild AS-IS. Do not sync the o3/o1 row fixes into
data/wild first — the other models' yesterday-losses scored the current (stale)
text, so MoE must score the identical text to stay comparable.

Run:  python run_moe16b_wild_loss.py        (LIMIT_PER_DATASET=2 for a smoke test)
"""

import os
import sys
import gc
import json
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
# import the shared MoE loader FIRST (sets kernel env + patches, imports transformers)
sys.path.insert(0, str(SCRIPT_DIR.parent))   # DistillDetectRelease/
import moe16b_loader as moe                    # noqa: E402
sys.path.insert(0, str(SCRIPT_DIR))            # reference_mia/ (for run_wild)
import run_wild as rw                          # noqa: E402  reuse FILES_MAP + ModelWrapper

# ---- settings (match run_wild.sh defaults) -----------------------------------
DATASETS_DIR = SCRIPT_DIR.parent / "data" / "wild"
OUT_DIR = SCRIPT_DIR.parent / "outputs" / "reference_mia_wild_moe16b"
LIMIT_PER_DATASET = int(os.environ.get("LIMIT_PER_DATASET", "200"))
MAX_LENGTH = 4096            # MoE-16B ctx is 4096; do not exceed
MAX_ANSWER_TOKENS = 2048
ANSWER_TRUNCATION_SIDE = "right"
STRIDE = 512
PROMPT_PREFIX_TOKENS = 1000
DTYPE = os.environ.get("DTYPE", "bfloat16")
DEVICE_MAP = "auto"
MAX_MEMORY_PER_GPU = "130GiB"
MAX_MEMORY_CPU = "200GiB"


def main():
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_NUM_WORKERS_MATERIALIZE", "1")
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[info] model        = {moe.MODEL_NAME}", flush=True)
    print(f"[info] datasets_dir = {DATASETS_DIR}", flush=True)
    print(f"[info] out_dir      = {OUT_DIR}", flush=True)
    print(f"[info] limit        = {LIMIT_PER_DATASET}", flush=True)
    print(f"[info] dtype        = {DTYPE}  max_length={MAX_LENGTH} stride={STRIDE} "
          f"prompt_prefix={PROMPT_PREFIX_TOKENS} max_answer={MAX_ANSWER_TOKENS}", flush=True)

    # ---- load wild datasets via run_wild's own loaders -----------------------
    datasets = {}
    print("\n--- Loading datasets (run_wild.FILES_MAP) ---", flush=True)
    for label, filename in rw.FILES_MAP.items():
        try:
            resolved = rw.resolve_dataset_file(DATASETS_DIR, filename)
        except FileNotFoundError as e:
            print(f"[MISS] {label}: {e}", flush=True)
            continue
        docs = rw.load_jsonl(str(resolved), limit=LIMIT_PER_DATASET)
        if docs:
            datasets[label] = {"file": str(resolved), "docs": docs}
            print(f"  [OK] {label}: {len(docs)} <- {resolved.name}", flush=True)
        else:
            print(f"  [EMPTY] {label}: parsed 0 rows from {resolved.name}", flush=True)
    if not datasets:
        raise RuntimeError("No wild datasets loaded.")

    model = tok = wrap = None
    try:
        print("\n--- Loading tokenizer + MoE model ---", flush=True)
        tok = moe.load_tokenizer()
        model = moe.load_moe_model(dtype=DTYPE, device_map=DEVICE_MAP,
                                   max_memory_per_gpu=MAX_MEMORY_PER_GPU,
                                   max_memory_cpu=MAX_MEMORY_CPU)
        eff_max_length = min(MAX_LENGTH, rw.get_model_ctx_limit(model, tok))
        wrap = rw.ModelWrapper(
            model, tok, max_length=eff_max_length, stride=STRIDE,
            prompt_prefix_tokens=PROMPT_PREFIX_TOKENS,
            max_answer_tokens=MAX_ANSWER_TOKENS,
            answer_truncation_side=ANSWER_TRUNCATION_SIDE)
        print(f"[info] effective max_length = {eff_max_length}", flush=True)

        output = {
            "model_name": moe.MODEL_NAME,
            "datasets_dir": str(DATASETS_DIR),
            "max_length": eff_max_length,
            "max_length_requested": MAX_LENGTH,
            "max_answer_tokens": MAX_ANSWER_TOKENS,
            "answer_truncation_side": ANSWER_TRUNCATION_SIDE,
            "stride": STRIDE,
            "prompt_prefix_tokens": PROMPT_PREFIX_TOKENS,
            "limit_per_dataset": LIMIT_PER_DATASET,
            "dtype": DTYPE,
            "files_map": rw.FILES_MAP,
            "results": {},
        }
        out_path = OUT_DIR / f"{rw.safe_name(moe.MODEL_NAME.split('/')[-1])}__wild_losses.json"

        print("\n--- Computing losses ---", flush=True)
        for label, info in datasets.items():
            docs = info["docs"]
            rows, valid = [], []
            for i, doc in enumerate(docs):
                loss = wrap.get_loss(doc)
                rows.append({"idx": i, "loss": loss, "ok": loss is not None})
                if loss is not None:
                    valid.append(loss)
                if (i + 1) % 25 == 0:
                    print(f"   {label}: {i+1}/{len(docs)}", flush=True)
            output["results"][label] = {
                "file": info["file"],
                "summary": {
                    "count": len(rows),
                    "valid_count": len(valid),
                    "mean_loss": float(np.mean(valid)) if valid else None,
                    "std_loss": float(np.std(valid)) if valid else None,
                },
                "rows": rows,
            }
            print(f"  [done] {label}: valid {len(valid)}/{len(rows)}  "
                  f"mean_loss={output['results'][label]['summary']['mean_loss']}", flush=True)
            # checkpoint after every dataset
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(output, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Saved losses to: {out_path}", flush=True)
    finally:
        for obj in (model, tok, wrap):
            if obj is not None:
                del obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print("\n✅ DONE.", flush=True)


if __name__ == "__main__":
    main()
