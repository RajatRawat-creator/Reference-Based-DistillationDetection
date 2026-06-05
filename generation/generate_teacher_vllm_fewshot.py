#!/usr/bin/env python3
import os
os.environ["VLLM_USE_V1"] = "0"
from huggingface_hub import login

# -----------------------------------------------------------------------------
# 0. Setup & Authentication
# -----------------------------------------------------------------------------
_hf_token = os.environ.get("HF_TOKEN")
if _hf_token:
    login(token=_hf_token)

import argparse
import json
import re
from pathlib import Path
from typing import List

from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, GenerationConfig


def load_questions(input_source: str, openmath_path: Path) -> List[str]:
    if input_source == "s1k":
        print("Loading s1K dataset from Hugging Face...")
        ds = load_dataset("simplescaling/s1K", split="train")
        return [str(x["question"]) for x in ds]

    if input_source == "openmath_jsonl":
        if not openmath_path.exists():
            raise FileNotFoundError(f"OpenMath file not found at: {openmath_path}")

        print(f"Loading OpenMath questions from {openmath_path}...")
        qs = []
        with open(openmath_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    q = data.get("problem") or data.get("question")
                    if q:
                        qs.append(str(q))
                except json.JSONDecodeError:
                    continue
        return qs

    raise ValueError(f"Unknown input_source: {input_source}")


def count_done(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for _ in f)


def safe_slug(text: str) -> str:
    text = re.sub(r"[^\w.\-]+", "_", text)
    return text[:220]


def load_fewshot_template(prompt_file: Path) -> str:
    if not prompt_file.exists():
        raise FileNotFoundError(f"Few-shot prompt file not found: {prompt_file}")
    text = prompt_file.read_text(encoding="utf-8")

    if "<TARGET QUESTION HERE>" not in text:
        raise ValueError(
            f"Prompt file {prompt_file} does not contain '<TARGET QUESTION HERE>'"
        )
    return text


def fill_fewshot_template(template: str, question: str) -> str:
    return template.replace("<TARGET QUESTION HERE>", question)


def should_use_chat_template(model_name: str, tokenizer) -> bool:
    name = model_name.lower()

    # Keep base Qwen as raw text prompt.
    if "qwen3-8b" in name and "instruct" not in name and "base" in name:
        return False

    return bool(getattr(tokenizer, "chat_template", None))


def apply_chat_with_optional_thinking(tokenizer, messages, model_name: str) -> str:
    """
    Try to request visible thinking traces for models that support it.
    Fall back cleanly if the tokenizer/template does not accept the arg.
    """
    lower_name = model_name.lower()

    want_thinking = (
        "gpt-oss" in lower_name
        or "qwen" in lower_name
    )

    if want_thinking:
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,
            )
        except TypeError:
            pass
        except Exception as e:
            print(f"Warning: enable_thinking=True failed for {model_name}: {e}")

    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def prepare_prompts(
    questions: List[str],
    fewshot_template: str,
    tokenizer,
    model_name: str
) -> List[str]:
    prompts = []
    use_chat = should_use_chat_template(model_name, tokenizer)
    print(f"Using chat template: {use_chat}")

    for q in questions:
        filled = fill_fewshot_template(fewshot_template, q)

        if use_chat:
            messages = [{"role": "user", "content": filled}]
            full_prompt = apply_chat_with_optional_thinking(
                tokenizer=tokenizer,
                messages=messages,
                model_name=model_name,
            )
        else:
            full_prompt = filled

        prompts.append(full_prompt)

    return prompts


def get_model_default_params(
    model_name: str,
    max_tokens_override: int = None,
    temperature_override: float = None
) -> SamplingParams:
    print(f"Fetching generation config for: {model_name}...")
    stop_token_ids = None
    temperature = 1.0
    top_p = 1.0

    try:
        gen_config = GenerationConfig.from_pretrained(model_name)
        if getattr(gen_config, "temperature", None) is not None:
            temperature = gen_config.temperature
        if getattr(gen_config, "top_p", None) is not None:
            top_p = gen_config.top_p

        eos_ids = getattr(gen_config, "eos_token_id", None)
        if isinstance(eos_ids, int):
            stop_token_ids = [eos_ids]
        elif isinstance(eos_ids, list):
            stop_token_ids = eos_ids
    except Exception as e:
        print(f"Warning: could not load generation config: {e}")
        print("Using fallback defaults: temperature=1.0, top_p=1.0")

    if temperature_override is not None:
        temperature = temperature_override

    max_tokens = max_tokens_override or 2048

    print(
        f"Sampling params -> temperature={temperature}, top_p={top_p}, "
        f"max_tokens={max_tokens}, stop_token_ids={stop_token_ids}"
    )

    return SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        stop_token_ids=stop_token_ids
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", required=True)
    parser.add_argument("--input_source", choices=["s1k", "openmath_jsonl"], required=True)
    parser.add_argument("--fewshot_prompt_file", type=Path, required=True)
    parser.add_argument("--openmath_path", type=Path, default=Path("DATASETS/OpenMathInstruct1KQuestions.jsonl"))
    parser.add_argument("--out_dir", type=Path, required=True)
    parser.add_argument("--tp", type=int, default=2)
    parser.add_argument("--gpu_util", type=float, default=0.90)
    parser.add_argument("--max_model_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--limit", type=int, default=200)

    args = parser.parse_args()

    model_slug = safe_slug(args.teacher_model.replace("/", "__"))
    prompt_slug = safe_slug(args.fewshot_prompt_file.stem)

    # Make one folder per prompt file
    prompt_subdir = args.out_dir / prompt_slug
    prompt_subdir.mkdir(parents=True, exist_ok=True)

    out_filename = f"{model_slug}__{args.input_source}__fewshot__vllm.jsonl"
    out_path = prompt_subdir / out_filename

    questions = load_questions(args.input_source, args.openmath_path)
    if args.limit is not None:
        questions = questions[:args.limit]

    template = load_fewshot_template(args.fewshot_prompt_file)

    done = count_done(out_path)
    if done >= len(questions):
        print(f"[Done] All {len(questions)} items already generated for {out_path}")
        return

    print(f"Total questions to run: {len(questions)}")
    print(f"Resuming from index {done}...")
    print(f"Saving outputs to: {out_path}")
    remaining_qs = questions[done:]

    print(f"Initializing vLLM [Model: {args.teacher_model}, TP: {args.tp}]...")
    llm = LLM(
        model=args.teacher_model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_model_len,
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        args.teacher_model,
        trust_remote_code=True
    )

    sampling_params = get_model_default_params(
        args.teacher_model,
        max_tokens_override=args.max_tokens,
        temperature_override=args.temperature
    )

    print(f"Formatting {len(remaining_qs)} prompts...")
    prompts = prepare_prompts(
        remaining_qs,
        template,
        tokenizer,
        args.teacher_model
    )

    print("Starting generation...")
    start_idx = done

    with out_path.open("a", encoding="utf-8") as f:
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_qs = remaining_qs[i:i + args.batch_size]

            outputs = llm.generate(batch_prompts, sampling_params)

            for j, output in enumerate(outputs):
                original_idx = start_idx + i + j
                response_text = output.outputs[0].text if output.outputs else ""

                record = {
                    "teacher_model": args.teacher_model,
                    "input_source": args.input_source,
                    "prompt_id": original_idx,
                    "prompt_format": "fewshot_from_file",
                    "fewshot_prompt_file": str(args.fewshot_prompt_file),
                    "prompt_subdir": str(prompt_subdir),
                    "question": batch_qs[j],
                    "response": response_text,
                    "formatted_prompt": output.prompt,
                    "sampling_params": str(sampling_params),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

    print(f"Done! Saved to {out_path}")


if __name__ == "__main__":
    main()