# Teacher response generation

`generate_local_vllm.py` generates responses from a local model via vLLM on
a JSONL of `question` records. Handles both plain chat models (Gemma, Llama,
Qwen) and the GPT-OSS harmony format (analysis vs final channels parsed
separately).

It is one-model-at-a-time — you launch it once per (teacher, prompt set):

| Teacher                              | Prompts            | Suggested wrapper              |
|--------------------------------------|--------------------|--------------------------------|
| `google/gemma-3-27b-it`              | OMI, s1            | `run_local_gen.sh`             |
| `nvidia/Llama-3.3-70B-Instruct-FP8`  | OMI, s1            | `run_local_gen.sh`             |
| `Qwen/Qwen3-8B`                      | OMI, s1            | adapt `run_local_gen.sh`       |
| `openai/gpt-oss-120b`                | OMI, s1            | `run_local_gen_GPTOSS.sh`      |

The `.sh` wrappers are Slurm sbatch launchers. Export `HF_TOKEN` in the
submitting shell:

```bash
export HF_TOKEN=hf_...
sbatch generation/run_local_gen.sh
sbatch generation/run_local_gen_GPTOSS.sh
```

## Output

One JSONL per (model, questions file) under `--out_dir`. Each row:

```json
{"model_name": "...", "question": "...", "response": "...", "reasoning": "..."}
```

The output JSONL is the input to `training/` (SFT) and is also reused by
`reference_mia/` and `normal_mia/` as the scoring corpus.

## OMI-CoT few-shot generation

For the OMI-CoT students (the four base models distilled on the Llama
OMI(918) OMI-CoT SFT data), reference-based MIA is improved by scoring against
candidate-teacher responses generated with **in-context few-shot exemplars
drawn from each student** ("+ In-context exemplars from S"). `generate_teacher_vllm_fewshot.py`
produces those per-student candidate datasets.

- `FewShotPrompts/` — one `.txt` per OMI-CoT student. Each holds that student's
  last-15 boxed solutions as few-shot exemplars plus a single
  `<TARGET QUESTION HERE>` placeholder.
- `run_fewshot_gen.sh` — sweeps all four candidate teachers
  (Llama-3.3-70B-FP8, Qwen-3-8B, Gemma-3-27B-it, GPT-OSS-120B) over every
  prompt in `FewShotPrompts/`.

```bash
export HF_TOKEN=hf_...
sbatch generation/run_fewshot_gen.sh          # INPUT_SOURCE=s1k by default
# INPUT_SOURCE=openmath_jsonl OPENMATH_PATH=... sbatch generation/run_fewshot_gen.sh
```

Output: `<OUT_DIR>/<prompt-stem>/<teacher>__<source>__fewshot__vllm.jsonl`,
where the prompt-stem subfolder name equals the few-shot prompt file name
(`Student=` → `Student_`). With the default `OUT_DIR=../data/omi_cot_fewshot`
this regenerates exactly the shipped `data/omi_cot_fewshot/` layout. These
per-student candidate sets are then used as the `--datasets_dir` for the
OMI-CoT reference MIA (the `Reference_Fewshot` condition).

## Wild generation

Wild model generation (DeepSeek R1 distills, QwQ, etc.) used external API
services (OpenRouter, Fireworks) and is not included in this release. The
prepared MIA inputs are in `data/wild/` and are scored directly by
`reference_mia/run_wild.py`.
