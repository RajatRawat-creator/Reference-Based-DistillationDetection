# Margin-threshold classifier

Two scripts. Run in order:

| Script               | Reads                                                | Writes                                  |
|----------------------|------------------------------------------------------|------------------------------------------|
| `build_threshold.py` | `../outputs/reference_mia_controlled/*__results.json` (controlled MIA output) | `data/controlled_dataset_10f.csv`        |
| `apply_threshold.py` | `data/controlled_dataset_10f.csv` (to learn threshold) + `../outputs/reference_mia_wild/*/*__results.json` (to classify) | per-row predictions + accuracy summaries (stdout) |

`build_threshold.py` extracts the 10 per-candidate features (paper §5.1)
from each controlled student's reference-MIA JSON and writes the
classifier-ready CSV.

`apply_threshold.py` then:

1. **Learns a margin threshold** from `controlled_dataset_10f.csv`. Three modes:
   - `--threshold f1`   (default) — pick the margin that maximizes F1 on
     the controlled set
   - `--threshold p95`            — pick the 95th-percentile margin of
     the label-0 (not-distilled) rows
   - `--threshold <float>`        — fixed value
2. **Scores wild models** by reading their `<target>__results.json` files
   from `../outputs/reference_mia_wild/`. For each (model, prompt, condition):
   - `margin = top_candidate_score − second_best_candidate_score`
   - `pred   = "DISTILLED" if margin >= threshold else "NOT DISTILLED"`
3. **Prints per-row results** and accuracy summaries split by prompt
   (OMI / s1) and condition (with_true / without_true).

## Pool exclusions

Hard-coded at the top of `apply_threshold.py`:

- `EXCLUDED_CANDIDATES = {"QwQ-32B", "Qwen-3-235B-A22B-Thinking-2507"}`
  — removed from the candidate pool when computing margins.
- `EXCLUDED_FROM_METRICS = {"DeepSeek-R1-Distill-Qwen-1.5B"}`
  — still scored but excluded from the final accuracy summaries
  (too small; Gemma-3-27B-it outscores R1 on it).

`apply_threshold.py` also reads `../reference_mia/pairs_wild.csv` for the
wild-target list. `R1_SUBDIRS` at the top of the script is the legacy
hard-coded fallback list of subdirectory names that match the
`<target>__REF__<reference>` convention.

## Usage

```bash
# Step 1 — assemble the classifier-ready CSV from controlled MIA outputs
python classifier/build_threshold.py

# Step 2 — learn threshold + score the wild models
python classifier/apply_threshold.py                     # default max-F1
python classifier/apply_threshold.py --threshold p95
python classifier/apply_threshold.py --threshold 0.086

# Evaluate on the controlled CSV instead of wild outputs
python classifier/apply_threshold.py --eval-controlled

# Point at a non-default wild output tree
python classifier/apply_threshold.py --wild-dir /path/to/MIA/outputs
```

## Reproducibility note

If you only want to evaluate the released CSV without re-running MIA,
skip step 1 — `data/controlled_dataset_10f.csv` is already populated
from the paper run.
