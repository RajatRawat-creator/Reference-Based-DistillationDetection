# Distillation Detection via Reference-Normalized Loss MIA

This repository is the **minimal release** of the code used in our paper on
detecting whether a student LLM was distilled from a given teacher (e.g.
DeepSeek R1, GPT-OSS-120B, Gemma-3-27B-it, Llama-3.3-70B-Instruct, Qwen-3-8B,
etc.).

The pipeline runs end-to-end in four stages:

1. **Generate teacher responses** (`generation/`) — one of four base
   teachers (Gemma-3-27B-it, GPT-OSS-120B, Qwen-3-8B, Nvidia-Llama-3.3-70B-Instruct)
   answers an OMI / s1 prompt set via vLLM.
2. **Train controlled student models** (`training/`) — SFT a base model on
   the teacher responses for each (student × teacher × prompt-set) combo.
3. **Reference-normalized loss MIA** (`reference_mia/`) — for each target
   model (controlled student OR wild model), score it against every
   candidate teacher's responses, normalized by a base reference model.
4. **o1 ASCII / Unicode detection** (`o1_detection/`) — paired
   default-vs-unicode MIA exploiting o1's `\uXXXX` escape signature.

**Results & significance reproduction** (`ReferenceMIAResults/`) — every
teacher-identification, threshold-generalization (LOSO / leave-one-teacher-out),
and o1 number in the paper is reproduced **CPU-only** from the gathered ref-norm
JSONs by the self-checking scripts in `ReferenceMIAResults/scripts/` (see
`ReferenceMIAResults/README.md`). This replaces the old `classifier/` and
`significance/` folders.

All datasets ship in `data/`.

## Layout

```
DistillDetectRelease/
├── README.md
├── requirements.txt
├── generation/                       # Stage 1: teacher response generation (vLLM)
│   ├── generate_local_vllm.py
│   ├── run_local_gen.sh              #   Gemma + Llama wrapper
│   ├── run_local_gen_GPTOSS.sh       #   GPT-OSS wrapper (separate; longer ctx)
│   ├── generate_teacher_vllm_fewshot.py  # OMI-CoT few-shot candidate generation
│   ├── run_fewshot_gen.sh           #   all-teachers few-shot sweep
│   └── FewShotPrompts/              #   per-student few-shot prompt templates (.txt)
├── training/                         # Stage 2: SFT controlled students
│   ├── train_qwen.py                 #   Qwen-2.5-1.5B + Qwen-2.5-3B
│   ├── train_qwen.sh
│   ├── train_llama_gemma.py          #   Llama-3.2-3B-Instruct + Gemma-3-4B-PT
│   └── train_llama_gemma.sh
├── reference_mia/                    # Stage 3: reference-normalized MIA
│   ├── run_controlled.py             #   For controlled students
│   ├── run_controlled.sh
│   ├── run_wild.py                   #   For wild R1-distill / open-weight models
│   ├── run_wild.sh
│   ├── run_omi_cot.py                #   For OMI-CoT students (few-shot probe)
│   ├── run_omi_cot.sh
│   ├── pairs_controlled_data.csv     #   target,reference manifest (controlled students)
│   ├── pairs_wild.csv                #   target,reference manifest (R1-distills, s1.1, QwQ, GPT-OSS)
│   ├── pairs_xcoder.csv              #   X-Coder only (activates 11-teacher pool, +Qwen-3-235B)
│   └── pairs_omi_cot.csv             #   target,reference,fewshot_subdir (4 OMI-CoT students)
├── o1_detection/                     # Stage 4: o1 ASCII-vs-Unicode detection
│   ├── run_o1_ascii_unicode.py
│   └── run_o1_ascii_unicode.sh
├── ReferenceMIAResults/              # Gathered ref-norm results + CPU-only reproduction
│   ├── controlled/  OMI_COT/  ModelsInTheWild/  OpenQuestions/   # all ref-norm JSONs
│   ├── scripts/                      #   reproduce_*.py (tables, LOSO/LOTO, o1, open-world) + plot_figures.py
│   └── README.md
└── data/                             # ALL datasets (no external download)
    ├── README.md
    ├── training/                     #   Teacher SFT data (reused as MIA input)
    ├── wild/                         #   wild MIA inputs (10 teachers + Qwen-3-235B for XCoder; _files_map_xcoder.json)
    ├── o1/                           #   o1 default-vs-unicode pairs
    ├── MIADatasets/                  #   controlled reference-MIA candidate responses (200 rows)
    └── omi_cot_fewshot/              #   per-student few-shot candidate responses (OMI-CoT students)
```

## Environment

```bash
pip install -r requirements.txt
export HF_TOKEN=hf_...    # for gated models (Llama, Gemma)
```

## Quick start

### 1. (Optional) Regenerate teacher responses

The teacher JSONLs in `data/training/` are pre-computed. To regenerate
from scratch (e.g. with a different sampling config):

```bash
sbatch generation/run_local_gen.sh           # Gemma + Llama
sbatch generation/run_local_gen_GPTOSS.sh    # GPT-OSS-120B
sbatch generation/run_fewshot_gen.sh         # OMI-CoT few-shot candidates (all teachers)
```

### 2. Train controlled students

```bash
sbatch training/train_qwen.sh           # Qwen 1.5B + 3B
sbatch training/train_llama_gemma.sh    # Llama-3.2-3B-Instruct + Gemma-3-4B-PT
```

Checkpoints land under `checkpoints/...` (override via `SFT_OUTPUT_DIR`).

### 3. Reference-normalized MIA

```bash
sbatch reference_mia/run_controlled.sh       # iterates pairs_controlled_data.csv (data/MIADatasets)
sbatch reference_mia/run_wild.sh             # iterates pairs_wild.csv (data/wild, 10-teacher pool)
sbatch reference_mia/run_omi_cot.sh          # OMI-CoT students (data/omi_cot_fewshot)
```

Each (target, reference) pair writes `<target>__results.json` under
`outputs/reference_mia_{controlled,wild,omi_cot}/`.

### 4. o1 detection

```bash
sbatch o1_detection/run_o1_ascii_unicode.sh
```

### 5. Reproduce every paper number (CPU-only, no GPU)

All teacher-identification, threshold-generalization (LOSO / leave-one-teacher-out),
and o1 results reproduce from the gathered ref-norm JSONs — each script ends with a
self-checking `VERIFICATION: N/N PASS`:

```bash
cd ReferenceMIAResults/scripts
python reproduce_tables.py                    # controlled + real-world accuracy, top-1 significance
python reproduce_threshold_generalization.py  # margin threshold + LOSO / leave-one-teacher-out + significance
python reproduce_o1_ascii_unicode.py          # controlled o1 ASCII-vs-Unicode table
python reproduce_open_questions.py             # open-world o1 significance + same-family-excluded rankings
python plot_figures.py                         # replot the ranked CDF figures
```

## File-by-file equivalence with the original codebase

All Python files are either **byte-identical** copies of the original
sources or differ **only** in (a) path constants, (b) Hugging Face token
handling (env var instead of hard-coded literal), and (c) the `FILES_MAP`
candidate dictionary in the reference-MIA runners (which teacher response
files are scored — data wiring, not algorithm). No changes to the MIA / scoring
/ training logic.

| Release path                                       | Original path                                                              |
|----------------------------------------------------|----------------------------------------------------------------------------|
| `generation/generate_local_vllm.py`                | `NewScripts/generate_local_vllm.py`                                        |
| `generation/run_local_gen.sh`                      | `NewScripts/run_local_gen.sh`                                              |
| `generation/run_local_gen_GPTOSS.sh`               | `NewScripts/run_local_gen_GPTOSS.sh`                                       |
| `generation/generate_teacher_vllm_fewshot.py`      | `scripts/generate_teacher_vllm_fewshot.py`                                 |
| `generation/run_fewshot_gen.sh`                    | `scripts/sbatch_run_.sbatch` (+ `scripts/sbatch_generateData_fewshot.sbatch`) |
| `generation/FewShotPrompts/*.txt`                  | `scripts/FewShotPrompts/*.txt`                                             |
| `data/omi_cot_fewshot/*`                           | `scripts/teacher_gen_outputs_s1_fewshot/*`                                 |
| `training/train_qwen.py`                           | `scripts/FINALGITHUBALLSCRIPTS/training_script/run_a.py`                   |
| `training/train_qwen.sh`                           | `scripts/FINALGITHUBALLSCRIPTS/training_script/run_a.sh`                   |
| `training/train_llama_gemma.py`                    | `scripts/FINALGITHUBALLSCRIPTS/training_script/run_b.py`                   |
| `training/train_llama_gemma.sh`                    | `scripts/FINALGITHUBALLSCRIPTS/training_script/run_b.sh`                   |
| `reference_mia/run_controlled.py`                  | `NewScripts/run_reference_attack_server.py`                                |
| `reference_mia/run_wild.py`                        | `NewScripts/run_reference_attack_server.py`                                |
| `reference_mia/run_omi_cot.py`                     | `NewScripts/run_reference_attack_server.py` (few-shot `FILES_MAP` only)    |
| `o1_detection/run_o1_ascii_unicode.py`             | `NewScripts/run_reference_attack_o1_controlled.py`                         |
| `o1_detection/run_o1_ascii_unicode.sh`             | `NewScripts/run_reference_attack_o1_controlled.sh`                         |
| `data/training/Teacher=*.jsonl`                    | `scripts/FINALGITHUBALLSCRIPTS/SFTDatasets{,OLDD}/Teacher=*.jsonl`         |
| `data/MIADatasets/*` (controlled candidates)       | cleaned 200-row teacher responses (Gemma/Llama/GPT-OSS/Qwen, OMI+s1)       |
| `data/wild/*` (Gemma/Llama/GPT-OSS)                | identical to `data/MIADatasets/` (same 3 teachers)                        |
| `data/wild/*` (Claude/R1/o1/o3/QwQ-Preview/Qwen-3-235B) | `scripts/collected_outputs/` + `NewScripts/R1testUnicode_upload/collected_outputs/` |
| `data/o1/*`                                        | `NewScripts/O1_UnicodeandASCII/*`                                          |
| `ReferenceMIAResults/controlled/*.json`            | `scripts/FINALGITHUBALLSCRIPTS/ReferenceMIAResults/*.json`                 |
| `ReferenceMIAResults/ModelsInTheWild/...*__results.json` | `NewScripts/ModelsinTheWildOutputs/*__REF__*/*__results.json`        |

## Reproducing each headline result

Every result is reproducible from the shipped data without re-running GPU
jobs (the pre-computed MIA outputs are included). To regenerate from models,
run the corresponding stage first.

| Result | Command(s) | Reads | Notes |
|---|---|---|---|
| **Controlled + real-world accuracy, top-1 significance** | `python ReferenceMIAResults/scripts/reproduce_tables.py` | `ReferenceMIAResults/` | CPU; 27/27 checks |
| **Margin threshold + LOSO / leave-one-teacher-out + significance** | `python ReferenceMIAResults/scripts/reproduce_threshold_generalization.py` | `ReferenceMIAResults/` | CPU; fold-fit τ (max-F1 ≈ 0.067); 38/38 checks |
| **o1 ASCII-vs-Unicode table (controlled)** | `python ReferenceMIAResults/scripts/reproduce_o1_ascii_unicode.py` | `ReferenceMIAResults/` | CPU; 54/54 checks |
| **Open-world o1 significance + same-family-excluded rankings** | `python ReferenceMIAResults/scripts/reproduce_open_questions.py` | `ReferenceMIAResults/` | CPU; 25/25 checks |
| **Reference MIA — controlled students** | `sbatch reference_mia/run_controlled.sh` | `data/MIADatasets/`, checkpoints | GPU; writes `outputs/reference_mia_controlled/` |
| **Reference MIA — wild models** (10-teacher pool) | `sbatch reference_mia/run_wild.sh` | `data/wild/` | GPU; downloads wild target models from HF |
| **Reference MIA — XCoder** (11-teacher pool, +Qwen-3-235B) | `DATASETS_DIR=../data/wild PAIRS_CSV=pairs_xcoder.csv sbatch reference_mia/run_wild.sh` | `data/wild/` (`_files_map_xcoder.json`) | GPU; auto-activates the 11-way pool |
| **Reference MIA — OMI-CoT few-shot** (Reference_Fewshot) | `sbatch reference_mia/run_omi_cot.sh` | `data/omi_cot_fewshot/`, checkpoints | GPU; "+ in-context exemplars from S" |
| **o1 ASCII-vs-Unicode detection** | `sbatch o1_detection/run_o1_ascii_unicode.sh` | `data/o1/` | GPU |
| **(Re)generate teacher responses** | `sbatch generation/run_local_gen.sh` (+ `_GPTOSS.sh`) | your question JSONL | GPU; optional — data ships pre-computed |
| **(Re)generate OMI-CoT few-shot candidates** | `sbatch generation/run_fewshot_gen.sh` | `generation/FewShotPrompts/` | GPU; rebuilds `data/omi_cot_fewshot/` |

GPU stages write to `outputs/` (gitignored); the reproduction scripts above read the
shipped `ReferenceMIAResults/` JSONs and need no GPU. To score a fresh wild run,
copy its `*__results.json` into `ReferenceMIAResults/ModelsInTheWild/ReferenceMIA/`.

## Cluster / environment notes

The GPU `.sh` launchers were written for a SLURM cluster. They no longer hardcode
any machine-specific path — the spots you may need to change are marked with
`# EDIT:` comments and/or honor environment-variable overrides:

- **`HF_TOKEN`** — set it in your environment (`export HF_TOKEN=hf_...`), or paste
  it into the `hf_PASTE_YOUR_TOKEN_HERE` placeholder in each launcher. Required for
  gated models (Llama, Gemma); the launchers fail fast if unset.
- **`PYTHON`** — defaults to `python`; set it to your env's interpreter if needed
  (`PYTHON=/path/to/env/bin/python sbatch ...`).
- **`#SBATCH` directives** — partition (`-p`), node (`--nodelist`), QOS, `--gpus`,
  `--time` are placeholders/cluster-specific; set them for your scheduler.
- **`HF_HOME`** — defaults to `$HOME/.cache/huggingface`; override for a different
  cache location.
- **`CHECKPOINTS_DIR`** — where `training/` wrote the student checkpoints (default
  `../checkpoints`).
- **Running under `sbatch`** — SLURM copies the script to a spool dir, so the
  default `cd "$(dirname "$0")"` can't find `../data`; the launchers note the
  absolute `cd` to substitute (or run them directly with `bash`).

The CPU-only reproduction (`ReferenceMIAResults/scripts/*.py`) needs none of this —
just `pip install -r requirements.txt`.

- **Wild-model + closed-API generation** (R1, Claude, o1/o3, QwQ) used external
  services (OpenRouter / vendor APIs) and is not scripted here; their responses
  ship pre-generated in `data/wild/` (redistribution subject to vendor TOS).
