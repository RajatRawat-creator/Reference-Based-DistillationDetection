# Distillation Detection via Reference-Normalized Loss MIA

This repository is the **minimal release** of the code used in our paper on
detecting whether a student LLM was distilled from a given teacher (e.g.
DeepSeek R1, GPT-OSS-120B, Gemma-3-27B-it, Llama-3.3-70B-Instruct, Qwen-3-8B,
etc.).

The pipeline runs end-to-end in six stages:

1. **Generate teacher responses** (`generation/`) — one of four base
   teachers (Gemma-3-27B-it, GPT-OSS-120B, Qwen-3-8B, Nvidia-Llama-3.3-70B-Instruct)
   answers an OMI / s1 prompt set via vLLM.
2. **Train controlled student models** (`training/`) — SFT a base model on
   the teacher responses for each (student × teacher × prompt-set) combo.
3. **Reference-normalized loss MIA** (`reference_mia/`) — for each target
   model (controlled student OR wild model), score it against every
   candidate teacher's responses, normalized by a base reference model.
4. **Normal (non-reference) MIA** (`normal_mia/`) — Min-K% baseline,
   controlled models only, for comparison with stage 3.
5. **Margin-threshold classifier** (`classifier/`) — learn a threshold
   from the controlled MIA results, then classify wild models.
6. **o1 ASCII / Unicode detection** (`o1_detection/`) — paired
   default-vs-unicode MIA exploiting o1's `\uXXXX` escape signature.

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
│   ├── pairs_controlled.csv          #   target,reference manifest (20 students)
│   ├── pairs_wild.csv                #   target,reference manifest (18 wild models)
│   └── pairs_omi_cot.csv             #   target,reference,fewshot_subdir (4 OMI-CoT students)
├── normal_mia/                       # Stage 4: Min-K% baseline (controlled only)
│   ├── run_controlled.py
│   └── run_controlled.sh
├── classifier/                       # Stage 5: margin-threshold classifier
│   ├── build_threshold.py            #   Assembles classifier-ready CSV from MIA results
│   ├── apply_threshold.py            #   Learns threshold + scores wild models
│   └── utils.py
├── o1_detection/                     # Stage 6: o1 ASCII-vs-Unicode detection
│   ├── run_o1_ascii_unicode.py
│   └── run_o1_ascii_unicode.sh
└── data/                             # ALL datasets (no external download)
    ├── README.md
    ├── training/                     #   Teacher SFT data (reused as MIA input)
    ├── wild/                         #   MIA scoring inputs for wild models
    ├── o1/                           #   o1 default-vs-unicode pairs
    ├── classifier/                   #   controlled_dataset_10f.csv
    ├── reference_mia/                #   pre-computed MIA result JSONs (controlled + wild)
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
sbatch reference_mia/run_controlled.sh       # iterates pairs_controlled.csv (data/MIADatasets)
sbatch reference_mia/run_wild.sh             # iterates pairs_wild.csv (data/wild, 10-teacher pool)
sbatch reference_mia/run_omi_cot.sh          # OMI-CoT students (data/omi_cot_fewshot)
```

Each (target, reference) pair writes `<target>__results.json` under
`outputs/reference_mia_{controlled,wild,omi_cot}/`.

### 4. (Optional) Normal MIA baseline

```bash
sbatch normal_mia/run_controlled.sh
```

### 5. Apply the threshold classifier

```bash
python classifier/build_threshold.py                 # builds data/classifier/controlled_dataset_10f.csv from MIA outputs
python classifier/apply_threshold.py                 # default max-F1 threshold + wild scoring
python classifier/apply_threshold.py --threshold p95
python classifier/apply_threshold.py --eval-controlled
```

If you're only consuming the released CSV, skip `build_threshold.py` —
`data/classifier/controlled_dataset_10f.csv` is already populated.

### 6. o1 detection

```bash
sbatch o1_detection/run_o1_ascii_unicode.sh
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
| `normal_mia/run_controlled.py`                     | `scripts/run_reference_attack_server_MINK.py`                              |
| `o1_detection/run_o1_ascii_unicode.py`             | `NewScripts/run_reference_attack_o1_controlled.py`                         |
| `o1_detection/run_o1_ascii_unicode.sh`             | `NewScripts/run_reference_attack_o1_controlled.sh`                         |
| `classifier/build_threshold.py`                    | `scripts/FINALGITHUBALLSCRIPTS/Classifier/build_dataset_10f.py`            |
| `classifier/apply_threshold.py`                    | `scripts/FINALGITHUBALLSCRIPTS/Classifier/eval_threshold_r1distill.py`     |
| `classifier/utils.py`                              | `scripts/FINALGITHUBALLSCRIPTS/Classifier/utils.py`                        |
| `data/training/Teacher=*.jsonl`                    | `scripts/FINALGITHUBALLSCRIPTS/SFTDatasets{,OLDD}/Teacher=*.jsonl`         |
| `data/MIADatasets/*` (controlled candidates)       | cleaned 200-row teacher responses (Gemma/Llama/GPT-OSS/Qwen, OMI+s1)       |
| `data/wild/*` (Gemma/Llama/GPT-OSS)                | identical to `data/MIADatasets/` (same 3 teachers)                        |
| `data/wild/*` (Claude/R1/o1/o3/QwQ-Preview)        | `scripts/collected_outputs/` + `NewScripts/R1testUnicode_upload/collected_outputs/` |
| `data/o1/*`                                        | `NewScripts/O1_UnicodeandASCII/*`                                          |
| `data/classifier/controlled_dataset_10f.csv`       | `scripts/FINALGITHUBALLSCRIPTS/Classifier/controlled_dataset_10f.csv`      |
| `data/reference_mia/controlled/*.json`             | `scripts/FINALGITHUBALLSCRIPTS/ReferenceMIAResults/*.json`                 |
| `data/reference_mia/wild/*__REF__*/*__results.json`| `NewScripts/ModelsinTheWildOutputs/*__REF__*/*__results.json`             |

## Reproducing each headline result

Every result is reproducible from the shipped data without re-running GPU
jobs (the pre-computed MIA outputs are included). To regenerate from models,
run the corresponding stage first.

| Result | Command(s) | Reads | Notes |
|---|---|---|---|
| **Controlled classifier threshold** (max-F1 ≈ 0.067) | `python classifier/apply_threshold.py --eval-controlled` | `data/classifier/controlled_dataset_10f.csv` | learns + reports on controlled set |
| **Wild 10-way detection accuracy** | `python classifier/apply_threshold.py` | `data/reference_mia/wild/` | applies the threshold to the wild R1-distill / s1.1 models |
| **Rebuild the controlled feature CSV** | `python classifier/build_threshold.py` | `data/reference_mia/controlled/` | reproduces `data/classifier/controlled_dataset_10f.csv` byte-for-byte |
| **Reference MIA — controlled students** | `sbatch reference_mia/run_controlled.sh` | `data/MIADatasets/`, checkpoints | GPU; writes `outputs/reference_mia_controlled/` |
| **Reference MIA — wild models** (10-teacher pool) | `sbatch reference_mia/run_wild.sh` | `data/wild/` | GPU; downloads wild target models from HF |
| **Reference MIA — OMI-CoT few-shot** (Reference_Fewshot) | `sbatch reference_mia/run_omi_cot.sh` | `data/omi_cot_fewshot/`, checkpoints | GPU; "+ in-context exemplars from S" |
| **Non-reference MIA baseline** (Min-K%) | `sbatch normal_mia/run_controlled.sh` | `data/MIADatasets/`, checkpoints | GPU; baseline for comparison |
| **o1 ASCII-vs-Unicode detection** | `sbatch o1_detection/run_o1_ascii_unicode.sh` | `data/o1/` | GPU |
| **(Re)generate teacher responses** | `sbatch generation/run_local_gen.sh` (+ `_GPTOSS.sh`) | your question JSONL | GPU; optional — data ships pre-computed |
| **(Re)generate OMI-CoT few-shot candidates** | `sbatch generation/run_fewshot_gen.sh` | `generation/FewShotPrompts/` | GPU; rebuilds `data/omi_cot_fewshot/` |

GPU stages write to `outputs/` (gitignored). To feed a fresh wild run into the
classifier: `python classifier/apply_threshold.py --wild-dir outputs/reference_mia_wild`.

## Cluster / environment notes

The `.sh` / `sbatch` launchers were written for a SLURM cluster (Berkeley NLP).
Before running on your machine, adjust per launcher:

- the `conda activate <env>` line and the `#SBATCH` directives (`-p`, `--gpus`, …);
- the cache exports (`HF_HOME`, `VLLM_CACHE_ROOT`, `XDG_CACHE_HOME`) — point them
  at writable paths or delete them;
- input/output path variables (`QUESTIONS`, `OUT_DIR`, `CHECKPOINTS_DIR`,
  `DATASETS_DIR`) — most honor env-var overrides.

`export HF_TOKEN=hf_...` is required for gated models (Llama, Gemma); every
launcher fails fast if it is unset.

## Known gaps

- **Wild-model + closed-API generation** (R1, Claude, o1/o3, QwQ) used external
  services (OpenRouter / vendor APIs) and is not scripted here; their responses
  ship pre-generated in `data/wild/` (redistribution subject to vendor TOS).
