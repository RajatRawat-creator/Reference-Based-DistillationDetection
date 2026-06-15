#!/usr/bin/env python3
"""
Single-model (loss-only) run of deepseek-ai/deepseek-moe-16b-base over the o1
ASCII-vs-Unicode datasets, reusing run_o1_ascii_unicode.py's machinery (FILES_MAP
with the per-file escape transform, load_jsonl, and the stride ModelWrapper) so
the numbers are comparable to the other models' o1-detection runs.

NOTE on context: MoE-16B's context window is 4096, far below the 32768 the larger
o1-detection models used. get_model_ctx_limit caps the effective max_length to
MoE's 4096 automatically (same behaviour as the R1TestGitHub MoE o1 run); MoE's
per-token answer NLL is therefore scored over a 4096 window via the stride loop.

Run:  python run_moe16b_o1_loss.py     (LIMIT_PER_DATASET=2 for a smoke test)
"""

import os
import sys
import gc
import json
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))   # DistillDetectRelease/
import moe16b_loader as moe                    # noqa: E402  (env + patches FIRST)
sys.path.insert(0, str(SCRIPT_DIR))            # o1_detection/
import run_o1_ascii_unicode as ro1             # noqa: E402  reuse FILES_MAP + ModelWrapper

# ---- settings (match run_o1_ascii_unicode.sh) --------------------------------
DATASETS_DIR = SCRIPT_DIR.parent / "data" / "o1"
OUT_DIR = SCRIPT_DIR.parent / "outputs" / "o1_detection_moe16b"
LIMIT_PER_DATASET = int(os.environ.get("LIMIT_PER_DATASET", "200"))
MAX_LENGTH = 32768           # requested; capped to MoE ctx (4096) by get_model_ctx_limit
MAX_ANSWER_TOKENS = 4000
ANSWER_TRUNCATION_SIDE = "right"
STRIDE = 512
PROMPT_PREFIX_TOKENS = 3000
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
    print(f"[info] limit        = {LIMIT_PER_DATASET}  dtype={DTYPE}", flush=True)

    # ---- load o1 datasets via run_o1_ascii_unicode's loader (with transform) -
    datasets = {}
    print("\n--- Loading o1 datasets (run_o1_ascii_unicode.FILES_MAP) ---", flush=True)
    for label, spec in ro1.FILES_MAP.items():
        filename = spec["file"]
        escape = (spec.get("transform") == "escape")
        try:
            resolved = ro1.resolve_dataset_file(DATASETS_DIR, filename)
        except FileNotFoundError as e:
            print(f"[MISS] {label}: {e}", flush=True)
            continue
        docs, stats = ro1.load_jsonl(str(resolved), limit=LIMIT_PER_DATASET,
                                     escape_nonascii=escape)
        if docs:
            datasets[label] = {"file": str(resolved), "escape": escape, "docs": docs}
            print(f"  [OK] {label}: {len(docs)} <- {resolved.name} "
                  f"(escape={escape}, nonascii_after_rows={stats['rows_nonascii_after']})", flush=True)
        else:
            print(f"  [EMPTY] {label}: parsed 0 rows from {resolved.name}", flush=True)
    if not datasets:
        raise RuntimeError("No o1 datasets loaded.")

    model = tok = wrap = None
    try:
        print("\n--- Loading tokenizer + MoE model ---", flush=True)
        tok = moe.load_tokenizer()
        model = moe.load_moe_model(dtype=DTYPE, device_map=DEVICE_MAP,
                                   max_memory_per_gpu=MAX_MEMORY_PER_GPU,
                                   max_memory_cpu=MAX_MEMORY_CPU)
        eff_max_length = min(MAX_LENGTH, ro1.get_model_ctx_limit(model, tok))
        wrap = ro1.ModelWrapper(
            model, tok, max_length=eff_max_length, stride=STRIDE,
            prompt_prefix_tokens=PROMPT_PREFIX_TOKENS,
            max_answer_tokens=MAX_ANSWER_TOKENS,
            answer_truncation_side=ANSWER_TRUNCATION_SIDE)
        print(f"[info] effective max_length = {eff_max_length} "
              f"(requested {MAX_LENGTH}, MoE ctx caps it)", flush=True)

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
            "files_map": ro1.FILES_MAP,
            "results": {},
        }
        out_path = OUT_DIR / f"{ro1.safe_name(moe.MODEL_NAME.split('/')[-1])}__o1_ascii_unicode_losses.json"

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
                "escape": info["escape"],
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
