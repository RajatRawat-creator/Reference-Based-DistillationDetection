# Data

All datasets ship inside this repo. No external download step.

## Folders

| Folder              | Contents                                                                  | Consumers                                |
|---------------------|---------------------------------------------------------------------------|------------------------------------------|
| `training/`         | Teacher response JSONLs used for student SFT, **also** used as MIA inputs | `training/`, `reference_mia/`, `normal_mia/` |
| `wild/`             | MIA scoring inputs for wild-model evaluation                              | `reference_mia/run_wild.py`              |
| `o1/`               | o1 default-vs-unicode paired JSONLs                                       | `o1_detection/run_o1_ascii_unicode.py`   |
| `classifier/`       | `controlled_dataset_10f.csv` — 10-feature classifier dataset from paper   | `classifier/apply_threshold.py`          |
| `reference_mia/`    | Pre-computed reference-MIA result JSONs (controlled students + wild models) | `classifier/build_threshold.py`, `classifier/apply_threshold.py` |
| `MIADatasets/`      | Controlled-model reference-MIA candidate responses (4 teachers × OMI/s1, 200 rows, cleaned) | `reference_mia/run_controlled.py` |
| `omi_cot_fewshot/`  | Per-student few-shot reference-MIA candidate responses for OMI-CoT students | OMI-CoT reference MIA (Reference_Fewshot) |

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

### Known gap — Gemma-3-27B-it teacher data

The paper's 4th teacher (`google/gemma-3-27b-it`) does **not** appear as a
`Teacher=Gemma-3-27B-it_*.jsonl` here. Its response JSONLs were stored
under the wild-MIA naming convention instead:

- `wild/gemma-3-27b-it__s1k__chat__vllm.jsonl`
- `wild/gemma-3-27b-it__openmath_jsonl__chat__vllm (1).jsonl`

If you want to retrain the controlled students from scratch on a Gemma
teacher, copy those two files into `training/` and rename them to the
`Teacher=Gemma-3-27B-it_Data=<...>_Template=Chat.jsonl` convention.

## `wild/` provenance

The teacher-candidate pool for the **wild** reference MIA — a 10-way
classification over these 10 teacher models, each with an OMI and an s1
response set (200 rows each, 20 files total), consumed by
`reference_mia/run_wild.py` (its `FILES_MAP`):

| Teacher | OMI / s1 files | Source |
|---|---|---|
| Gemma-3-27B-it | `gemma-3-27b-it__…__vllm.jsonl` | **= `data/MIADatasets/`** (identical to controlled) |
| Llama-3.3-70B-Instruct | `Llama-3.3-70B-Instruct-FP8__…__chat__vllm.jsonl` | **= `data/MIADatasets/`** |
| GPT-OSS-120B | `gpt-oss-120b__…__vllmCLEAN.jsonl` | **= `data/MIADatasets/`** |
| Claude 3.5 Sonnet | `Claude-Sonnet-3.5_{omi,s1}(200).jsonl` | `collected_outputs` |
| Claude Opus 4.5 | `claude-opus-4-5-…_reasoning_only__REPAIRED.jsonl` | `collected_outputs` |
| Claude Opus 4.6 | `Claude-Opus-4.6_{omi,s1}(200)_reasoningonly.jsonl` | `collected_outputs` |
| DeepSeek R1 | `R1_{omi,s1}(200)_reasoningonly.jsonl` | `collected_outputs` |
| o1 | `o1_{omi,s1}(200).jsonl` | `collected_outputs` |
| o3 | `o3_{omi,s1}(200).jsonl` | `collected_outputs` |
| QwQ-32B-Preview | `QwQ-32B-Preview_{omi,s1}(200).jsonl` | `collected_outputs` |

The Gemma / Llama / GPT-OSS files are **byte-identical** to the same three
teachers in `data/MIADatasets/`, so the controlled and wild pools share an
identical candidate distribution for those teachers. The closed-API outputs
(Claude, OpenAI) were generated via OpenRouter / direct API and are included
verbatim — redistributability is subject to each vendor's TOS, the user's
responsibility.

## `o1/` provenance

Two JSONLs that differ only in non-ASCII encoding (escape vs raw UTF-8) —
used by `o1_detection/` to detect o1-distillation via the
`\uXXXX` escape signature.

## `classifier/` provenance

`controlled_dataset_10f.csv` is identical to the canonical paper artifact at
`scripts/FINALGITHUBALLSCRIPTS/Classifier/controlled_dataset_10f.csv`. It
is regenerable from controlled MIA outputs via `classifier/build_threshold.py`.

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
Generated by `scripts/generate_teacher_vllm_fewshot.py`; the downstream result
JSONs are analyzed by
`scripts/FINALGITHUBALLSCRIPTS/MethodEvaluationScripts/eval_omicot_reference.py`.

## `reference_mia/` provenance

Pre-computed reference-normalized loss MIA result JSONs, so the classifier
stage runs without re-running stage 3.

- `reference_mia/controlled/` — 19 JSONs, byte-identical to
  `scripts/FINALGITHUBALLSCRIPTS/ReferenceMIAResults/*.json`. These are the
  default input to `classifier/build_threshold.py` (which rebuilds
  `classifier/controlled_dataset_10f.csv` from them — verified byte-identical).
- `reference_mia/wild/<TARGET>__REF__<REF>/<MODEL>__results.json` — 17 JSONs,
  byte-identical to `NewScripts/ModelsinTheWildOutputs/<...>/*__results.json`.
  These are the default input to `classifier/apply_threshold.py` (which scores
  the 7 R1-distill / s1.1 wild models in its `R1_SUBDIRS` list).

Both default paths can be overridden: `REF_MIA_RESULTS_DIR` /
`--input-dir` for the build step, `DD_WILD_DIR` / `--wild-dir` for apply —
e.g. point them at a fresh `outputs/reference_mia_*` run.
