# Normal (non-reference) MIA — controlled models only

This folder hosts the baseline MIA scoring functions that don't depend
on the reference-normalization signal used by `reference_mia/`. They
exist as a baseline comparison in the paper.

## Single script — Min-K%

`run_controlled.py` is a byte-identical copy of
`scripts/run_reference_attack_server_MINK.py` (only the hardcoded HF
token line is swapped for `HF_TOKEN` env-var lookup). It computes the
Min-K% probability MIA score per sample for each candidate teacher in
`FILES_MAP`.

The original Min-K, ZLIB, and RawLoss baseline results in the paper
live in `scripts/FINALGITHUBALLSCRIPTS/NonReferenceMIAResults/`.

## Related sibling scripts (in original tree, not copied here)

If you want the ZLIB or full reverse-baseline variants, see:
- `scripts/run_reference_attack_server_ZLIB.py` — ZLIB-normalized variant
- `scripts/reverse_baseline_mia_server.py` — combined loss + Min-K + ZLIB

These are not copied into the release by default; they can be added
later if the paper's tables require regenerating those columns.

## Running

```bash
export HF_TOKEN=hf_...
sbatch normal_mia/run_controlled.sh
```

The launcher reuses `../reference_mia/pairs_controlled.csv` for its
target list, so adding/removing a controlled student in one place
updates both pipelines.

## Output

`<target>__results.json` written to `--out_dir`, with the same shape
as the reference-MIA outputs but containing Min-K% scores instead of
ref-norm losses.
