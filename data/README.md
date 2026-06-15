# Data

All datasets ship inside this repo. No external download step.

## Folders

| Folder              | Contents                                                                  | Consumers                                |
|---------------------|---------------------------------------------------------------------------|------------------------------------------|
| `training/`         | Teacher response JSONLs used for student SFT, **also** used as MIA inputs | `training/`, `reference_mia/` |
| `wild/`             | Wild-model MIA scoring inputs: 10-teacher pool + Qwen-3-235B (for XCoder) and `_files_map_xcoder.json` | `reference_mia/run_wild.py` |
| `o1/`               | o1 default-vs-unicode paired JSONLs                                       | `o1_detection/run_o1_ascii_unicode.py`   |
| `MIADatasets/`      | Controlled-model reference-MIA candidate responses (4 teachers × OMI/s1, 200 rows, cleaned) | `reference_mia/run_controlled.py` |
| `omi_cot_fewshot/`  | Per-student few-shot reference-MIA candidate responses for OMI-CoT students | OMI-CoT reference MIA (Reference_Fewshot) |

> The pre-computed reference-MIA **result** JSONs (formerly `data/reference_mia/`
> and `data/classifier/`) now live under `../ReferenceMIAResults/`, alongside the
> CPU-only reproduction scripts. This `data/` folder holds only the **inputs**
> (candidate responses, SFT/probe data) consumed by the GPU runners.

## `training/` provenance

Each `Teacher=<T>_Data=<D>_Template=<TPL>.jsonl` is the verbatim output of
the corresponding teacher run from `generation/generate_local_vllm.py`. Rows
follow `{question, response, ...}`.

Files in this release:

- `Teacher=Nvidia-Llama-3.3-70B-Instruct_Data=OMI(1K)_Template=Chat.jsonl` (the corrected/active Nvidia-Llama dataset)
- `Teacher=Nvidia-Llama-3.3-70B-Instruct_Data=OMI(918)_Template=OMI_COT.jsonl` (the OMI CoT-templated variant)
- `Teacher=Nvidia-Llama-3.3-70B-Instruct_Data=S1_Template=Chat.jsonl`
- `Teacher=GPT-OSS-120B_Data=OMI(1K)_Template=Chat.jsonl`
- `Teacher=GPT-OSS-120B_Data=S1_Template=Chat.jsonl`
- `Teacher=Qwen-3-8B_Data=OMI(1K)_Template=Chat-TraceOnly.jsonl`
- `Teacher=Qwen-3-8B_Data=S1_Template=Chat-TraceOnly.jsonl`

## `wild/` provenance

The teacher-candidate pool for the **wild** reference MIA — a 10-way
classification over these 10 teacher models (11-way for X-Coder, which adds
Qwen-3-235B), each with an OMI and an s1 response set (200 rows each). All files
follow a uniform `<model>_omi(200).jsonl` / `<model>_s1(200).jsonl` naming. This
folder holds exactly the files consumed by `reference_mia/run_wild.py` (its
`FILES_MAP`, 20 files) plus the two Qwen-3-235B files used for the X-Coder
11-way pool via `_files_map_xcoder.json` — **22 jsonl files total**:

| Teacher | files (`_omi(200).jsonl` / `_s1(200).jsonl`) | Source |
|---|---|---|
| Gemma-3-27B-it | `Gemma-3-27B-it_*` | **content = `data/MIADatasets/`** (identical to controlled) |
| Llama-3.3-70B-Instruct | `Llama-3.3-70B-Instruct_*` | **content = `data/MIADatasets/`** (FP8) |
| GPT-OSS-120B | `GPT-OSS-120B_*` | **content = `data/MIADatasets/`** |
| Claude 3.5 Sonnet | `Claude-Sonnet-3.5_*` | `collected_outputs` |
| Claude Opus 4.5 | `Claude-Opus-4.5_*` | `collected_outputs` (trace + response) |
| Claude Opus 4.6 | `Claude-Opus-4.6_*` | `collected_outputs` (trace + response) |
| DeepSeek R1 | `DeepSeek-R1_*` | `collected_outputs` (reasoning + response) |
| o1 | `o1_*` | `collected_outputs` (trace + response) |
| o3 | `o3_*` | `collected_outputs` (trace + response) |
| QwQ-32B-Preview | `QwQ-32B-Preview_*` | `collected_outputs` |
| Qwen-3-235B *(X-Coder pool only)* | `Qwen-3-235B_*` | `collected_outputs` (reasoning + response) |

All 22 files have the full 200 rows, share the same 200 OMI / 200 s1 questions,
and carry a non-empty response (trace/reasoning + response, or response). The
Gemma / Llama / GPT-OSS files are **content-identical** to the same three teachers
in `data/MIADatasets/` (only the data/wild copies were renamed to this uniform
scheme; the `data/MIADatasets/` copies keep their original `__…__chat__vllm`
names, since the controlled runner reads those). The closed-API outputs (Claude,
OpenAI) were generated via OpenRouter / direct API and are included verbatim —
redistributability is subject to each vendor's TOS, the user's responsibility.

> **Note — truncation.** Reference MIA scores only the **first 2,048 tokens** of
> each teacher answer (`--max_answer_tokens 2048` in the `reference_mia/` runners),
> so for long reasoning traces the responses stored here are effectively truncated
> at scoring time — only their leading 2,048 answer tokens contribute to the
> reference-normalized loss.

## `MIADatasets/` provenance

The controlled candidate-teacher responses (4 teachers × OMI/s1, 200 rows each),
consumed by `reference_mia/run_controlled.py`.

> **Note — format-marker stripping.** For two teachers we strip the format
> scaffolding from the responses so detection can't lean on an obvious stylistic
> giveaway (making the task harder / more realistic): the **Qwen-3-8B** files
> (`Qwen3-8B__…__vllm_cleaned.jsonl`) have the `<think>…</think>` tags removed,
> and the **GPT-OSS-120B** files (`gpt-oss-120b__…__vllmCLEAN.jsonl`) have the
> harmony `assistantfinal` / channel markers removed. The same cleaned content is
> what appears for these teachers in `data/wild/`.

## `o1/` provenance

Two JSONLs that differ only in non-ASCII encoding (escape vs raw UTF-8) —
used by `o1_detection/` to detect o1-distillation via the
`\uXXXX` escape signature.

## `omi_cot_fewshot/` provenance

Reference-MIA candidate datasets for the **OMI-CoT student models** — i.e. the
four base models (Qwen-2.5-1.5B, Qwen-2.5-3B, Llama-3.2-3B-Instruct,
Gemma-3-4B-PT) distilled on the Llama OMI(918) OMI-CoT SFT data
(`training/Teacher=Nvidia-Llama-3.3-70B-Instruct_Data=OMI(918)_Template=OMI_COT.jsonl`).

Layout — one subfolder per OMI-CoT student, each holding the four candidate
teachers' responses generated with that student's own last-15 boxed outputs as
in-context few-shot exemplars (200 rows each):

```
omi_cot_fewshot/
  Student_<base>_Teacher_Llama-3.3-70B-Instruct_Data_OMI_918__Template_..._fewshot_prompt/
      google__gemma-3-27b-it__s1k__fewshot__vllm.jsonl
      nvidia__Llama-3.3-70B-Instruct-FP8__s1k__fewshot__vllm.jsonl
      Qwen__Qwen3-8B__s1k__fewshot__vllm.jsonl
      openai__gpt-oss-120b__s1k__fewshot__vllm.jsonl
```

Because the few-shot exemplars come from the student, **each file is
student-specific** (all 16 are distinct). Feeding these as the candidate
responses in reference-based MIA is the "+ In-context exemplars from S"
condition (`Reference_Fewshot`) and improves teacher identification over the
plain reference MIA (`Reference_NonFewshot`).

Byte-identical to `scripts/teacher_gen_outputs_s1_fewshot/<same subdirs>/`.
Generated by `scripts/generate_teacher_vllm_fewshot.py`. The downstream result
JSONs (and all other pre-computed reference-MIA results) live under
`../ReferenceMIAResults/` and are analyzed by its `scripts/reproduce_*.py`.

## Pre-computed results

The reference-MIA **result** JSONs are not under `data/` — they are gathered in
`../ReferenceMIAResults/` (`controlled/`, `OMI_COT/`, `ModelsInTheWild/`,
`OpenQuestions/`) together with the CPU-only reproduction scripts that recompute
every paper number from them.
