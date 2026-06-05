#!/usr/bin/env python3
# Single-Target Reference-Normalized Loss MIA (Server)
# Matches the Colab reference logic, with two added features:
# - optional truncation of the teacher answer to max_answer_tokens
#   BEFORE scoring, excluding the prompt tokens.
# - per-file non-ASCII re-escaping so *_default.jsonl (which stored chars as
#   \uXXXX on disk) produces a DIFFERENT tokenization from *_unicode.jsonl
#   (which stored the raw UTF-8 chars). Without this, json.loads collapses
#   both forms to the same Python str and the MIA scores come out identical.

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse
import gc
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# Each entry maps a display label to:
#   file:      the source .jsonl filename (under --datasets_dir)
#   transform: what to do to question/response strings after json.loads
#              (omitted / None => leave as json.loads returned)
#
# Transforms:
#   None      : raw Unicode chars stay as they are (• × ₁₁ ² ...)
#   "escape"  : re-escape non-ASCII to literal \uXXXX form. json.loads
#               normally decodes "\u2022" back to "•"; this undoes that,
#               so the tokenizer sees the 6 ASCII chars '\u2022' instead
#               of the single char '•'. Result is pure ASCII but not
#               natural-looking text.
FILES_MAP = {
    # o1 — two variants of the same responses
    #   1) raw Unicode chars as stored (• × ₁₁ ² ...) — no transform
    #   2) same file parsed then re-escaped so non-ASCII chars become their
    #      literal \uXXXX form. The tokenizer sees pure ASCII; e.g. '•'
    #      becomes the 6 chars '\u2022'. This is the only way to preserve
    #      the on-disk distinction between *_unicode.jsonl and
    #      *_default.jsonl, because json.loads collapses both forms to the
    #      same Python str otherwise.
    "o1 (OMI, Unicode)": {"file": "o1_openmath__responses_unicode.jsonl"},
    "o1 (OMI, ASCII)": {"file": "o1__responses_default.jsonl", "transform": "escape"},
}


def safe_name(s: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_+.") else "_" for c in s)[:180]


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def pretty_label(old: str) -> str:
    s = old
    s = s.replace("(OM,", "(OMI,")
    s = s.replace("(OM ,", "(OMI,")
    if s.startswith("GPT"):
        s = s.replace("GPT(", "GPT-OSS-120B (", 1)
    if s.startswith("Llama"):
        s = s.replace("Llama(", "Llama-3.3-70B-Instruct (", 1)
    if s.startswith("Qwen"):
        s = s.replace("Qwen(", "Qwen-3-8B (", 1)
    if "/" in s:
        s = s.split("/")[-1]
    return s


def scores_to_probabilities(scores_dict: dict):
    all_scores = []
    for v in scores_dict.values():
        if v is None:
            continue
        arr = np.asarray(v, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size:
            all_scores.append(arr)

    if not all_scores:
        return {k: [] for k in scores_dict.keys()}, 0.0, 1.0

    all_scores = np.concatenate(all_scores)
    mu = float(all_scores.mean())
    sigma = float(all_scores.std() + 1e-8)

    probs = {}
    for k, v in scores_dict.items():
        arr = np.asarray(v, dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            probs[k] = []
            continue
        z = (arr - mu) / sigma
        probs[k] = sigmoid(-z)
    return probs, mu, sigma


def get_input_device(model):
    if hasattr(model, "hf_device_map") and isinstance(model.hf_device_map, dict):
        dmap = model.hf_device_map

        preferred_keys = [
            "model.embed_tokens",
            "model.model.embed_tokens",
            "transformer.wte",
            "model.tok_embeddings",
            "model.model.tok_embeddings",
        ]
        for k in preferred_keys:
            if k in dmap:
                dev = dmap[k]
                if dev == "disk":
                    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
                if isinstance(dev, int):
                    return torch.device(f"cuda:{dev}")
                return torch.device(dev)

        for dev in dmap.values():
            if isinstance(dev, int):
                return torch.device(f"cuda:{dev}")
            if isinstance(dev, str) and dev.startswith("cuda"):
                return torch.device(dev)

        return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

    return next(model.parameters()).device


def get_model_ctx_limit(model, tokenizer, default=4096):
    vals = []
    for attr in ["n_positions", "max_position_embeddings"]:
        v = getattr(model.config, attr, None)
        if isinstance(v, int) and v > 0:
            vals.append(v)
    tv = getattr(tokenizer, "model_max_length", None)
    if isinstance(tv, int) and 0 < tv < 100000:
        vals.append(tv)
    return min(vals) if vals else default


def resolve_dataset_file(datasets_dir: Path, filename: str) -> Path:
    p = datasets_dir / filename
    if p.exists():
        return p

    stem = filename[:-6] if filename.endswith(".jsonl") else filename
    pattern = re.compile(rf"^{re.escape(stem)}(\s*\(\d+\))?\.jsonl$")
    matches = [x for x in datasets_dir.iterdir() if x.is_file() and pattern.match(x.name)]
    if not matches:
        raise FileNotFoundError(f"Missing dataset file: {p} (and no '(N)' variant found)")
    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    print(f"[warn] Using variant for missing {filename}: {matches[0].name}")
    return matches[0]


def _escape_nonascii(s: str) -> str:
    """
    Mirror json.dumps(..., ensure_ascii=True) for NON-ASCII ONLY.
    ASCII chars (incl. \\n, \\t, etc.) are preserved as-is so whitespace and
    control-char tokenization is unaffected. Non-ASCII code points become
    their literal \\uXXXX form (surrogate pair for code points > U+FFFF).

    Rationale: files saved with ensure_ascii=True store e.g. '•' as the 6
    literal chars '\\u2022'. json.loads decodes that back to the single
    character '•', making the on-disk distinction disappear by the time the
    tokenizer runs. Applying this re-escape after parsing restores the
    escape-sequence form so the tokenizer actually sees different input.
    """
    out = []
    for c in s:
        cp = ord(c)
        if cp < 128:
            out.append(c)
        elif cp <= 0xFFFF:
            out.append(f"\\u{cp:04x}")
        else:
            cp -= 0x10000
            hi = 0xD800 + (cp >> 10)
            lo = 0xDC00 + (cp & 0x3FF)
            out.append(f"\\u{hi:04x}\\u{lo:04x}")
    return "".join(out)


def load_jsonl(
    filename: str,
    limit: int,
    escape_nonascii: bool = False,
):
    """
    Load a JSONL file of {question, response} (or prompt/answer/input/output)
    records into "Problem: ...\\nSolution: ..." strings.

    Optional post-parse transform (applied to both question and response):
      - escape_nonascii: re-escape non-ASCII code points to literal \\uXXXX
        form (useful to keep *_default.jsonl distinguishable from
        *_unicode.jsonl after json.loads collapses them).

    Returns (data, stats) where stats summarizes the conversion so the caller
    can verify that escape-mode output is actually pure ASCII.
    """
    data = []
    stats = {
        "rows_total": 0,
        "rows_kept": 0,
        "nonascii_before": set(),   # unique non-ASCII chars in raw input
        "nonascii_after": set(),    # unique non-ASCII chars AFTER transform
        "rows_nonascii_before": 0,
        "rows_nonascii_after": 0,
    }
    try:
        with open(filename, "r", encoding="utf-8") as f:
            for line in f:
                if len(data) >= limit:
                    break
                try:
                    item = json.loads(line)
                except Exception:
                    continue
                stats["rows_total"] += 1
                q = item.get("question") or item.get("prompt") or item.get("input") or ""
                a = item.get("response") or item.get("answer") or item.get("output") or ""

                combined_before = q + a
                bad_before = {c for c in combined_before if ord(c) >= 128}
                if bad_before:
                    stats["rows_nonascii_before"] += 1
                    stats["nonascii_before"] |= bad_before

                if escape_nonascii:
                    q = _escape_nonascii(q)
                    a = _escape_nonascii(a)

                combined_after = q + a
                bad_after = {c for c in combined_after if ord(c) >= 128}
                if bad_after:
                    stats["rows_nonascii_after"] += 1
                    stats["nonascii_after"] |= bad_after

                if q and a:
                    data.append(f"Problem: {q}\nSolution: {a}")
                    stats["rows_kept"] += 1
    except FileNotFoundError:
        pass

    # Sort char sets so we can print them deterministically
    stats["nonascii_before"] = sorted(stats["nonascii_before"])
    stats["nonascii_after"] = sorted(stats["nonascii_after"])
    return data, stats


class ModelWrapper:
    """
    Same scoring logic as the Colab version, with optional answer truncation.
    """
    def __init__(
        self,
        model,
        tokenizer,
        max_length: int,
        stride: int = 512,
        prompt_prefix_tokens: int = 1000,
        max_answer_tokens: int | None = None,
        answer_truncation_side: str = "right",
        debug: bool = False,
        debug_max_print: int = 20,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = get_input_device(model)

        self.max_length = int(max_length)
        self.stride = int(stride)
        self.prompt_prefix_tokens = int(prompt_prefix_tokens)
        self.max_answer_tokens = max_answer_tokens
        self.answer_truncation_side = answer_truncation_side

        self.debug = debug
        self.debug_max_print = debug_max_print
        self._dbg_prints = 0

    def _split_problem_solution(self, document: str):
        split_str = "Solution:"
        if split_str not in document:
            return None, None

        before, after = document.split(split_str, 1)
        prompt_text = before + split_str
        answer_text = " " + after.strip()

        if len(answer_text.strip()) == 0:
            return None, None

        return prompt_text, answer_text

    def _truncate_answer_ids(self, answer_ids):
        if self.max_answer_tokens is None:
            return answer_ids
        if len(answer_ids) <= self.max_answer_tokens:
            return answer_ids

        if self.answer_truncation_side == "left":
            return answer_ids[-self.max_answer_tokens:]
        return answer_ids[:self.max_answer_tokens]

    def _tokenize_prompt_answer(self, document: str):
        prompt_text, answer_text = self._split_problem_solution(document)
        if prompt_text is None:
            return None, None

        prompt_ids = self.tokenizer(
            prompt_text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]

        answer_ids = self.tokenizer(
            answer_text,
            add_special_tokens=False,
            truncation=False,
        )["input_ids"]

        if len(answer_ids) == 0:
            return None, None

        answer_ids = self._truncate_answer_ids(answer_ids)

        if len(answer_ids) == 0:
            return None, None

        return prompt_ids, answer_ids

    def get_loss(self, document: str):
        prompt_ids, answer_ids = self._tokenize_prompt_answer(document)
        if prompt_ids is None:
            return None

        max_len = self.max_length
        stride = self.stride
        prompt_anchor_len = min(self.prompt_prefix_tokens, len(prompt_ids))
        prompt_anchor = prompt_ids[:prompt_anchor_len]

        total_nll = 0.0
        total_scored_tokens = 0

        for i in range(0, len(answer_ids), stride):
            end = min(i + stride, len(answer_ids))
            new_answer_ids = answer_ids[i:end]
            trg_len = len(new_answer_ids)

            if trg_len == 0:
                continue

            remaining = max_len - trg_len
            if remaining <= 0:
                raise ValueError(
                    f"stride={stride} is too large for max_length={max_len}. "
                    f"Need max_length > stride."
                )

            keep_anchor = min(len(prompt_anchor), remaining)
            anchor_ids = prompt_anchor[:keep_anchor]
            remaining_after_anchor = remaining - len(anchor_ids)

            prev_answer_keep = min(i, remaining_after_anchor)
            prev_answer_ids = answer_ids[i - prev_answer_keep:i]
            remaining_after_prev = remaining_after_anchor - len(prev_answer_ids)

            if remaining_after_prev > 0:
                prompt_tail_ids = prompt_ids[-remaining_after_prev:]
            else:
                prompt_tail_ids = []

            if anchor_ids:
                anchor_set_cut = len(anchor_ids)
                if len(prompt_ids) <= anchor_set_cut:
                    merged_prompt_ids = anchor_ids
                else:
                    overlap_start_idx = max(
                        0,
                        anchor_set_cut - (len(prompt_ids) - len(prompt_tail_ids))
                    )
                    if overlap_start_idx > 0 and len(prompt_tail_ids) > 0:
                        trimmed_tail = prompt_tail_ids[overlap_start_idx:]
                    else:
                        trimmed_tail = prompt_tail_ids
                    merged_prompt_ids = anchor_ids + trimmed_tail
            else:
                merged_prompt_ids = prompt_tail_ids

            input_ids_list = merged_prompt_ids + prev_answer_ids + new_answer_ids
            labels_list = (
                [-100] * (len(merged_prompt_ids) + len(prev_answer_ids))
                + new_answer_ids
            )

            if self.debug and self._dbg_prints < self.debug_max_print:
                print(
                    f"[DBG] i={i} end={end} "
                    f"prompt_anchor={len(anchor_ids)} "
                    f"prompt_tail={len(prompt_tail_ids)} "
                    f"prev_answer={len(prev_answer_ids)} "
                    f"new_tokens={trg_len} "
                    f"total_input={len(input_ids_list)}"
                )
                self._dbg_prints += 1

            input_ids = torch.tensor([input_ids_list], device=self.device)
            labels = torch.tensor([labels_list], device=self.device)
            attention_mask = torch.ones_like(input_ids)

            with torch.inference_mode():
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            loss = out.loss
            if loss is None or (not torch.isfinite(loss)):
                return None

            total_nll += float(loss.detach().cpu()) * trg_len
            total_scored_tokens += trg_len

        if total_scored_tokens == 0:
            return None

        return total_nll / total_scored_tokens


def main():
    # Force single-threaded tensor materialization BEFORE any HF import side-effects
    # to prevent race-condition illegal memory access on multi-GPU model loading.
    os.environ.setdefault("TRANSFORMERS_NUM_WORKERS_MATERIALIZE", "1")

    ap = argparse.ArgumentParser()
    ap.add_argument("--target_model", required=True, help="HF id or local path")
    ap.add_argument("--ref_model", required=True, help="HF id or local path")
    ap.add_argument("--datasets_dir", default=None, help="Default: ./MIADatasetsR1Test next to this script")
    ap.add_argument("--out_dir", required=True)

    # Keep Colab-like total context default unless you explicitly change it.
    ap.add_argument("--max_length", type=int, default=4096)

    # New: limit only the teacher response tokens, excluding prompt.
    ap.add_argument("--max_answer_tokens", type=int, default=2048)

    ap.add_argument("--answer_truncation_side", choices=["left", "right"], default="right")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--prompt_prefix_tokens", type=int, default=1000)
    ap.add_argument("--limit_per_dataset", type=int, default=200)

    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="float16")
    ap.add_argument("--device_map", default="auto")
    ap.add_argument("--ref_device_map", default="cuda:0")

    # Per-GPU memory cap for device_map=auto on the TARGET model.
    # Prevents the placer from over-committing a GPU, which causes
    # cudaErrorIllegalAddress during tensor materialization.
    # Example: "72GiB" for A100-80GB cards.
    ap.add_argument("--max_memory_per_gpu", type=str, default=None,
                    help="Per-GPU memory cap for target model, e.g. '72GiB'. "
                         "Strongly recommended when loading models >=70B across multiple GPUs.")
    ap.add_argument("--max_memory_cpu", type=str, default="200GiB",
                    help="CPU memory cap used alongside --max_memory_per_gpu (default: 200GiB).")

    ap.add_argument("--trust_remote_code_ref", action="store_true", default=True)
    ap.add_argument("--trust_remote_code_tgt", action="store_true", default=True)
    ap.add_argument("--local_files_only_tgt", action="store_true", default=False)
    ap.add_argument("--load_in_4bit", action="store_true", default=False)
    ap.add_argument("--bnb_4bit_quant_type", choices=["nf4", "fp4"], default="nf4")
    ap.add_argument("--bnb_4bit_use_double_quant", action="store_true", default=True)
    ap.add_argument("--bnb_compute_dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")

    # NOTE: per-label text transforms (None / escape) are now encoded
    # directly in FILES_MAP at the top of this file, not via CLI flags.
    # Edit FILES_MAP to change what each label does.

    args = ap.parse_args()

    # ==========================================
    # WAKE UP ALL GPUS
    # ==========================================
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            _ = torch.zeros(1, device=f"cuda:{i}")
    # ==========================================

    os.makedirs(args.out_dir, exist_ok=True)
    script_dir = Path(__file__).resolve().parent
    datasets_dir = Path(args.datasets_dir).expanduser() if args.datasets_dir else (script_dir / "MIADatasetsR1Test")

    torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    bnb_compute_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.bnb_compute_dtype]

    quant_config = None
    if args.load_in_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=args.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=bnb_compute_dtype,
        )

    # Build max_memory dict for target model to prevent GPU over-commitment.
    target_max_memory = None
    if args.max_memory_per_gpu is not None:
        n_gpu = torch.cuda.device_count()
        target_max_memory = {i: args.max_memory_per_gpu for i in range(n_gpu)}
        target_max_memory["cpu"] = args.max_memory_cpu
        print(f"[INFO] target max_memory: {target_max_memory}")

    # Map FILES_MAP transform strings -> escape_nonascii arg for load_jsonl.
    TRANSFORM_DISPATCH = {
        None:     False,
        "escape": True,
    }

    datasets = {}
    print("--- Loading Datasets ---")
    for label, spec in FILES_MAP.items():
        filename = spec["file"]
        transform = spec.get("transform")
        if transform not in TRANSFORM_DISPATCH:
            raise ValueError(
                f"Unknown transform {transform!r} for label {label!r}. "
                f"Valid: {list(TRANSFORM_DISPATCH)}"
            )
        escape_flag = TRANSFORM_DISPATCH[transform]

        try:
            resolved = resolve_dataset_file(datasets_dir, filename)
        except FileNotFoundError as e:
            print(f"[MISS] {label}: {e}")
            continue

        loaded, stats = load_jsonl(
            str(resolved),
            limit=args.limit_per_dataset,
            escape_nonascii=escape_flag,
        )
        if not loaded:
            print(f"  ❌ {label}: found file but parsed 0 usable rows")
            continue

        datasets[label] = loaded
        transform_tag = f" [{transform}]" if transform else ""
        print(f"  ✅ {label}: {len(loaded)}{transform_tag}")

        # --- Conversion verification -----------------------------------
        # Report what the transform did. When transform="escape", hard-fail
        # if any non-ASCII char survived (because \uXXXX re-escaping must
        # produce pure ASCII by construction).
        n_before = stats["rows_nonascii_before"]
        n_after = stats["rows_nonascii_after"]
        uniq_before = len(stats["nonascii_before"])
        uniq_after = len(stats["nonascii_after"])
        print(
            f"     conversion: rows_with_nonascii "
            f"{n_before}/{stats['rows_kept']} -> {n_after}/{stats['rows_kept']}  "
            f"| unique non-ASCII chars {uniq_before} -> {uniq_after}"
        )
        # Only the 'escape' transform is required to produce pure ASCII.
        # The default (no transform) leaves text untouched.
        must_be_pure_ascii = transform == "escape"
        if must_be_pure_ascii and n_after > 0:
            sample = stats["nonascii_after"][:30]
            raise RuntimeError(
                f"[ASCII CHECK FAILED] {label}: after transform={transform!r}, "
                f"{n_after} rows still contain non-ASCII chars. "
                f"Unique survivors ({uniq_after}): {sample}"
            )
        if must_be_pure_ascii:
            print(f"     ✅ ASCII verified: 0 non-ASCII chars remain")

    if not datasets:
        raise RuntimeError("No datasets loaded.")

    print("\n==================================================")
    print(f"📥 PHASE 1: PRE-COMPUTING REFERENCE SCORES ({args.ref_model})")
    print("==================================================")

    ref_tok = AutoTokenizer.from_pretrained(args.ref_model, trust_remote_code=args.trust_remote_code_ref)
    ref_tok.truncation_side = "left"
    ref_tok.padding_side = "left"
    if ref_tok.pad_token is None:
        ref_tok.pad_token = ref_tok.eos_token

    if args.load_in_4bit:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model,
            device_map=args.ref_device_map,
            quantization_config=quant_config,
            trust_remote_code=args.trust_remote_code_ref,
            low_cpu_mem_usage=True,
        )
    else:
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.ref_model,
            device_map=args.ref_device_map,
            torch_dtype=torch_dtype,
            trust_remote_code=args.trust_remote_code_ref,
            low_cpu_mem_usage=True,
        )
    ref_model.eval()

    ref_max_length = min(args.max_length, get_model_ctx_limit(ref_model, ref_tok))
    ref_wrap = ModelWrapper(
        ref_model,
        ref_tok,
        max_length=ref_max_length,
        stride=args.stride,
        prompt_prefix_tokens=args.prompt_prefix_tokens,
        max_answer_tokens=args.max_answer_tokens,
        answer_truncation_side=args.answer_truncation_side,
    )
    print(f"[INFO] ref max_length = {ref_max_length}")
    print(f"[INFO] max_answer_tokens = {args.max_answer_tokens}")

    ref_scores_cache = {k: [] for k in datasets.keys()}
    for ds_label, samples in datasets.items():
        for text in samples:
            ref_scores_cache[ds_label].append(ref_wrap.get_loss(text))
        print(f"   ✅ Processed {ds_label}")

    # Force CPU offload before deletion to avoid dangling CUDA pointers
    del ref_model, ref_tok, ref_wrap
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    target_name = Path(args.target_model).name if ("/" not in args.target_model) else args.target_model.split("/")[-1]
    print("==================================================")
    print(f"🚀 PROCESSING TARGET: {target_name}")
    print("==================================================")

    model = None
    tok = None
    wrap = None

    try:
        try:
            tok = AutoTokenizer.from_pretrained(
                args.target_model,
                trust_remote_code=args.trust_remote_code_tgt,
                local_files_only=args.local_files_only_tgt,
                fix_mistral_regex=True,
            )
        except Exception:
            tok = AutoTokenizer.from_pretrained(
                args.target_model,
                trust_remote_code=args.trust_remote_code_tgt,
                local_files_only=args.local_files_only_tgt,
            )

        tok.truncation_side = "left"
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token

        if args.load_in_4bit:
            model = AutoModelForCausalLM.from_pretrained(
                args.target_model,
                device_map=args.device_map,
                quantization_config=quant_config,
                trust_remote_code=args.trust_remote_code_tgt,
                local_files_only=args.local_files_only_tgt,
                low_cpu_mem_usage=True,
                max_memory=target_max_memory,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                args.target_model,
                device_map=args.device_map,
                torch_dtype=torch_dtype,
                trust_remote_code=args.trust_remote_code_tgt,
                local_files_only=args.local_files_only_tgt,
                low_cpu_mem_usage=True,
                max_memory=target_max_memory,
            )
        model.eval()

        tgt_max_length = min(args.max_length, get_model_ctx_limit(model, tok))
        wrap = ModelWrapper(
            model,
            tok,
            max_length=tgt_max_length,
            stride=args.stride,
            prompt_prefix_tokens=args.prompt_prefix_tokens,
            max_answer_tokens=args.max_answer_tokens,
            answer_truncation_side=args.answer_truncation_side,
        )
        print(f"[INFO] target max_length = {tgt_max_length}")
        print(f"[INFO] max_answer_tokens = {args.max_answer_tokens}")

        results = {"Ref-Norm Loss": {ds: [] for ds in datasets}}
        fail_counts = {ds: 0 for ds in datasets}

        for ds_label, samples in datasets.items():
            ref_list = ref_scores_cache[ds_label]
            for i, text in enumerate(samples):
                ref_loss = ref_list[i]
                if ref_loss is None:
                    fail_counts[ds_label] += 1
                    continue

                target_loss = wrap.get_loss(text)
                if target_loss is None or not np.isfinite(target_loss):
                    fail_counts[ds_label] += 1
                    continue

                results["Ref-Norm Loss"][ds_label].append(target_loss - ref_loss)

        raw_path = Path(args.out_dir) / f"{safe_name(target_name)}__results.json"
        with raw_path.open("w") as f:
            json.dump(
                {
                    "model_name": target_name,
                    "ref_model": args.ref_model,
                    "max_length": args.max_length,
                    "max_answer_tokens": args.max_answer_tokens,
                    "answer_truncation_side": args.answer_truncation_side,
                    "stride": args.stride,
                    "prompt_prefix_tokens": args.prompt_prefix_tokens,
                    # Record each label's (file, transform) so it's easy to
                    # tell later what produced each result list.
                    "files_map": {
                        label: {
                            "file": spec["file"],
                            "transform": spec.get("transform"),
                        }
                        for label, spec in FILES_MAP.items()
                    },
                    "results": results,
                    "fail_counts": fail_counts,
                },
                f,
                indent=2,
            )
        print("✅ Saved raw results:", raw_path)

        cm = plt.get_cmap("tab20")
        probs_dict, mu, sigma = scores_to_probabilities(results["Ref-Norm Loss"])

        plt.figure(figsize=(12, 8))
        has_data = False
        labels_sorted = sorted(probs_dict.keys())

        for idx, label in enumerate(labels_sorted):
            vals = probs_dict[label]
            if vals is None or len(vals) == 0:
                continue
            arr = np.asarray(vals, dtype=np.float64)
            if arr.size == 0:
                continue

            has_data = True
            arr.sort()
            x = np.linspace(0, 100, arr.size) if arr.size > 1 else np.array([50.0])
            plt.plot(x, arr, label=pretty_label(label), linewidth=2, color=cm(idx % 20))

        if has_data:
            plt.title(f"{pretty_label(target_name)}\nReference Attack CDF (Target - Ref)", fontsize=16)
            plt.xlabel("Percentile", fontsize=12)
            plt.ylabel("Normalized Membership Likelihood", fontsize=12)
            plt.ylim(0, 1)
            plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            fig_path = Path(args.out_dir) / f"{safe_name(target_name)}__ref_attack_cdf.png"
            plt.savefig(fig_path, dpi=200, bbox_inches="tight")
            plt.close()
            print("✅ Saved plot:", fig_path)
        else:
            print("⚠️ No valid data to plot.")

    finally:
        if model is not None:
            del model
        if tok is not None:
            del tok
        if wrap is not None:
            del wrap
        gc.collect()
        torch.cuda.empty_cache()

    print("\n✅ DONE. Outputs in:", args.out_dir)


if __name__ == "__main__":
    main()