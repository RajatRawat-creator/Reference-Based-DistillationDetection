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
| `run_controlled.py`     | Scoring the controlled SFT students             | `pairs_controlled.csv`  |
| `run_wild.py`           | Scoring wild R1-distill / open-weight models    | `pairs_wild.csv`        |
| `run_omi_cot.py`        | Scoring the OMI-CoT students (few-shot probe)   | `pairs_omi_cot.csv`     |

`run_controlled.py` and `run_wild.py` are byte-identical copies of the
original `NewScripts/run_reference_attack_server.py`. `run_omi_cot.py` is the
same script with **only** its `FILES_MAP` changed to the four few-shot
candidate files. The split is organizational — each file's `FILES_MAP` (the
candidate teacher set scored per run) can diverge independently.

## Pairs manifests

`pairs_controlled.csv` and `pairs_wild.csv` are two-column CSVs
(`target,reference`) read by the `.sh` launchers and also by
`classifier/apply_threshold.py` to know which result JSON each
wild model corresponds to.

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
keyed by FILES_MAP label, and is consumed by
`classifier/apply_threshold.py`.

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
