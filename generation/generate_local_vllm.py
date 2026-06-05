#!/usr/bin/env python3
"""
Generate responses from local models via vLLM on questions from a JSONL file.

Supports regular chat models (Gemma, Llama, etc.) and GPT-OSS harmony format.
For GPT-OSS, splits the analysis (reasoning) and final (response) channels.

Usage:
  python generate_local_vllm.py \
      --model openai/gpt-oss-120b \
      --questions_path questions.jsonl \
      --out_dir outputs \
      --max_new_tokens 8192
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

from vllm import LLM, SamplingParams
from transformers import AutoTokenizer


# ---------------------------------------------------------------
# Harmony format parser for GPT-OSS
# ---------------------------------------------------------------
def is_gpt_oss_model(model_name: str) -> bool:
    return "gpt-oss" in model_name.lower()


def parse_harmony_output(text: str) -> Tuple[str, Optional[str]]:
    """
    Parse GPT-OSS harmony format output into (response, reasoning).

    Harmony format:
      <|channel|>analysis<|message|>...reasoning...<|end|>
      <|start|>assistant<|channel|>final<|message|>...response...<|return|>
    """
    analysis_parts = []
    final_parts = []

    # Extract analysis blocks
    for m in re.finditer(
        r'<\|channel\|>analysis<\|message\|>(.*?)(?:<\|end\|>|$)',
        text, re.DOTALL
    ):
        t = m.group(1).strip()
        if t:
            analysis_parts.append(t)

    # Extract final blocks
    for m in re.finditer(
        r'<\|channel\|>final<\|message\|>(.*?)(?:<\|return\|>|<\|end\|>|$)',
        text, re.DOTALL
    ):
        t = m.group(1).strip()
        if t:
            final_parts.append(t)

    reasoning = "\n".join(analysis_parts) if analysis_parts else None
    response = "\n".join(final_parts) if final_parts else text.strip()

    return response, reasoning


# ---------------------------------------------------------------
# Load questions
# ---------------------------------------------------------------
def load_questions(path: Path, question_field: str = "question", limit: int = 0) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            q = row.get(question_field) or ""
            if not q.strip():
                continue
            items.append(row)
            if limit > 0 and len(items) >= limit:
                break
    return items


def count_done(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


# ---------------------------------------------------------------
# Format prompts with chat template
# ---------------------------------------------------------------
def format_prompts(
    questions: List[Dict[str, Any]],
    tokenizer,
    question_field: str = "question",
    system_prompt: Optional[str] = None,
) -> List[str]:
    prompts = []
    for item in questions:
        q = item.get(question_field, "").strip()
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": q})

        try:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            prompt = q

        prompts.append(prompt)
    return prompts


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Generate responses from local models via vLLM")
    ap.add_argument("--model", required=True, help="HF model ID")
    ap.add_argument("--questions_path", type=Path, required=True)
    ap.add_argument("--out_dir", type=Path, default=Path("./outputs"))
    ap.add_argument("--question_field", default="question")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--system_prompt", type=str, default=None,
                    help="System prompt. For GPT-OSS, defaults to 'Reasoning: high'")

    # Generation
    ap.add_argument("--max_new_tokens", type=int, default=4096,
                    help="Token cap — the ONLY sampling param we override; "
                         "everything else comes from the model's generation_config.json")
    ap.add_argument("--temperature", type=float, default=None,
                    help="Optional override. If unset, uses the model's default temperature.")
    ap.add_argument("--min_tokens", type=int, default=0,
                    help="Force at least this many generated tokens before EOS is allowed. "
                         "Useful for untuned base models that otherwise return empty. 0 = no minimum.")

    # vLLM
    ap.add_argument("--dtype", default="auto")
    ap.add_argument("--trust_remote_code", action="store_true", default=True)
    ap.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    ap.add_argument("--tensor_parallel_size", type=int, default=1)
    ap.add_argument("--max_model_len", type=int, default=8192)

    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gpt_oss = is_gpt_oss_model(args.model)
    if gpt_oss and args.system_prompt is None:
        args.system_prompt = "Reasoning: high"
        print(f"GPT-OSS detected — using system prompt: '{args.system_prompt}'")

    # ---------------------------------------------------------------
    # Load questions
    # ---------------------------------------------------------------
    print(f"Loading questions from {args.questions_path}...")
    questions = load_questions(args.questions_path, question_field=args.question_field, limit=args.limit)
    print(f"  Loaded {len(questions)} questions.")

    # Output path
    model_slug = args.model.replace("/", "__")
    out_path = args.out_dir / f"{model_slug}__responses.jsonl"

    # Resume
    done = count_done(out_path)
    if done >= len(questions):
        print(f"[Done] All {len(questions)} already generated at {out_path}.")
        return
    if done > 0:
        print(f"  Resuming from question {done}.")
    remaining = questions[done:]

    # ---------------------------------------------------------------
    # Load model
    # ---------------------------------------------------------------
    print(f"\nLoading model: {args.model}...")
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=args.trust_remote_code,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
        generation_config="auto",  # load the model's own generation_config.json defaults
        # For multi-GPU TP, fall back to NCCL all-reduce. vLLM's custom all-reduce
        # kernel errors out on some node topologies (custom_all_reduce.cuh:453
        # 'invalid argument'); NCCL is marginally slower but robust.
        disable_custom_all_reduce=(args.tensor_parallel_size > 1),
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )

    # Start from the model's default sampling params (temperature, top_p, top_k,
    # repetition_penalty, etc. from generation_config.json) and override ONLY the
    # token cap. Optionally override temperature if the user passed --temperature.
    sampling_params = llm.get_default_sampling_params()
    sampling_params.max_tokens = args.max_new_tokens
    if args.temperature is not None:
        sampling_params.temperature = args.temperature
    if args.min_tokens > 0:
        sampling_params.min_tokens = args.min_tokens
    print(f"Sampling params (model defaults + token cap): {sampling_params}")

    # ---------------------------------------------------------------
    # Format prompts
    # ---------------------------------------------------------------
    print("Formatting prompts...")
    prompts = format_prompts(remaining, tokenizer,
                              question_field=args.question_field,
                              system_prompt=args.system_prompt)

    if prompts:
        print(f"\n  Sample prompt (first):\n{prompts[0][:600]}")
        if len(prompts[0]) > 600:
            print("...")
        print()

    # ---------------------------------------------------------------
    # Generate
    # ---------------------------------------------------------------
    print(f"Generating {len(prompts)} responses (max_tokens={args.max_new_tokens}, "
          f"temp={sampling_params.temperature}, gpt_oss={gpt_oss})...")
    t0 = time.time()
    outputs = llm.generate(prompts, sampling_params)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(prompts)/elapsed:.1f} q/s)")

    # ---------------------------------------------------------------
    # Write results
    # ---------------------------------------------------------------
    n_empty = 0
    with out_path.open("a", encoding="utf-8") as f:
        for i, (item, output) in enumerate(zip(remaining, outputs)):
            raw_text = output.outputs[0].text

            if gpt_oss:
                response, reasoning = parse_harmony_output(raw_text)
            else:
                response = raw_text
                reasoning = None

            if not response.strip():
                n_empty += 1

            record = {
                "model": args.model,
                "prompt_id": done + i,
                "question": item.get(args.question_field, ""),
                "response": response,
                "reasoning": reasoning,
                "num_tokens": len(output.outputs[0].token_ids),
                "finish_reason": output.outputs[0].finish_reason,
            }

            # Carry over metadata
            for key in ["tbd_score", "problem_source", "prompt_id", "teacher_model",
                         "usage", "cost_credits"]:
                if key in item and key not in record:
                    record[key] = item[key]

            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        f.flush()

    print(f"\nWrote {len(remaining)} responses to {out_path}")
    if n_empty:
        print(f"  WARNING: {n_empty} empty responses")
    if gpt_oss:
        # Count how many had reasoning
        with out_path.open("r") as rf:
            recs = [json.loads(l) for l in rf if l.strip()]
        has_reasoning = sum(1 for r in recs if r.get("reasoning"))
        print(f"  GPT-OSS: {has_reasoning}/{len(recs)} records have analysis (reasoning) trace")

    # Quick stats
    token_counts = [len(o.outputs[0].token_ids) for o in outputs]
    token_counts.sort()
    n = len(token_counts)
    if n:
        print(f"  Token counts: min={token_counts[0]} median={token_counts[n//2]} max={token_counts[-1]}")
        truncated = sum(1 for o in outputs if o.outputs[0].finish_reason == "length")
        if truncated:
            print(f"  WARNING: {truncated} responses hit max_tokens (truncated)")

    print(f"\nDone. Output at: {out_path}")


if __name__ == "__main__":
    main()