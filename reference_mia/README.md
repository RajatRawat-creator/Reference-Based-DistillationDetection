# Reference-normalized loss MIA

Two scripts, both implementing the same single-target reference-normalized
membership-inference attack:

```
score(x) = loss_target(x) - loss_ref(x)
```

The target is the model we want to test. The reference is a base model
known not to have been trained on the candidate teacher's responses.
Lower (more negative) `score(x)` means the target memorized the
distribution more strongly than the reference — a stronger signal that
the target was distilled from the teacher who produced `x`.

| Script                  | Use case                                        | Pairs manifest          |
|-------------------------|-------------------------------------------------|-------------------------|
| `run_controlled.py`     | Scoring the controlled SFT students             | `pairs_controlled_data.csv` |
| `run_wild.py`           | Scoring wild R1-distill / open-weight models    | `pairs_wild.csv`        |
| `run_omi_cot.py`        | Scoring the OMI-CoT students (few-shot probe)   | `pairs_omi_cot.csv`     |

All three share the same MIA core (`ModelWrapper` / stride loss / `load_jsonl`)
derived from `NewScripts/run_reference_attack_server.py`; they differ in their
`FILES_MAP` (candidate pool) and minor options. `run_wild.py` is the most current
(supports `--dtype auto`, `--attn_implementation`, and a `_files_map.json` drop-in
pool override); `run_controlled.py` / `run_omi_cot.py` are earlier forks with the
controlled / few-shot `FILES_MAP`. They could be unified into one runner + a
`_files_map.json` per pool, but are kept separate for now.

## Pairs manifests

`pairs_controlled_data.csv` and `pairs_wild.csv` are two-column CSVs
(`target,reference`) read by the `.sh` launchers. For XCoder, use
`pairs_xcoder.csv` with `DATASETS_DIR=../data/wild` — `run_wild.sh` then
auto-activates the 11-teacher pool (adds Qwen-3-235B) via
`data/wild/_files_map_xcoder.json`.

- **Controlled**: target is a trained student checkpoint directory name
  (resolved under `$CHECKPOINTS_DIR`). Reference is the base model that
  student family was fine-tuned from.
- **Wild**: target and reference are both HF model ids.

## Required args (both scripts)

- `--target_model`: HF id or local path of the target model.
- `--ref_model`: HF id or local path of the reference model.
- `--out_dir`: where to write `<target>__results.json` and the CDF plot.

## Useful options

- `--max_length 4096`           total context window
- `--max_answer_tokens 2048`    truncate teacher answers before scoring
- `--stride 512`                sliding window stride
- `--load_in_4bit`              4-bit quantized loading (bitsandbytes)
- `--max_memory_per_gpu 72GiB`  recommended for 70B targets on 80GB cards

The output JSON contains the per-sample reference-normalized losses
keyed by FILES_MAP label. These result JSONs are the inputs to the CPU-only
reproduction scripts in `ReferenceMIAResults/scripts/`.

## Running

```bash
export HF_TOKEN=hf_...

# Controlled students
sbatch reference_mia/run_controlled.sh

# Wild models (R1 distills, etc.)
sbatch reference_mia/run_wild.sh
```

## OMI-CoT few-shot variant

For students distilled on the Llama OMI(918) OMI-CoT SFT data, the reference
MIA is run against **per-student** candidate datasets in
`data/omi_cot_fewshot/` — each teacher's s1 responses generated with that
student's own in-context exemplars ("+ In-context exemplars from S"). This is
the `Reference_Fewshot` condition and improves teacher identification.

```bash
export HF_TOKEN=hf_...
sbatch reference_mia/run_omi_cot.sh        # iterates pairs_omi_cot.csv
```

`pairs_omi_cot.csv` has three columns — `target,reference,fewshot_subdir` —
where `fewshot_subdir` is the student's folder under `$FEWSHOT_BASE`
(default `../data/omi_cot_fewshot`). The launcher points `--datasets_dir` at
that subfolder per student. Regenerate the candidate datasets with
`generation/run_fewshot_gen.sh`. Downstream, the result JSONs correspond to the
paper's `OMICoT/Reference_Fewshot` outputs (analyzer:
`scripts/FINALGITHUBALLSCRIPTS/MethodEvaluationScripts/eval_omicot_reference.py`).

## DeepSeek-MoE-16B reference (open-world DeepSeek-R1)

`run_moe16b.sh` + `run_moe16b_wild_loss.py` score **DeepSeek-MoE-16B-Base** as the
reference model for the open-world DeepSeek-R1 analysis (R1 has no clean
same-family pre-distillation checkpoint, so MoE-16B is used as the base reference).
They reuse `run_wild.py`'s `FILES_MAP` + `ModelWrapper`, but load the model through
the shared **`../moe16b_loader.py`** rather than a plain `from_pretrained`.

Why a separate loader: `deepseek-ai/deepseek-moe-16b-base` ships **custom remote
modeling code** that no longer runs on current Transformers / torch. `moe16b_loader`
applies the required compatibility shims **before `transformers` is imported** —
e.g. restoring `is_torch_fx_available`, patching the FP8 integration internals,
adding torch-2.6 PEP-585 schema-inference support, and normalizing the model's
`rope_scaling` config. Loading MoE-16B the same way as the other models raises
import/attribute errors, hence the dedicated loader (also used by
`o1_detection/run_moe16b_o1_loss.py`).
