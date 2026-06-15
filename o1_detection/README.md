# o1 ASCII vs Unicode reference MIA

Detects o1-distillation by exploiting that OpenAI's o1 chat API returns
non-ASCII codepoints as `\uXXXX` escape sequences. We compare two serializations
of the same outputs: **ASCII** (each non-ASCII codepoint left as its literal
`\uXXXX` escape; file `o1__responses_ascii.jsonl`) vs **Unicode** (raw UTF-8;
`o1_openmath__responses_unicode.jsonl`). A model distilled from o1 scores the
ASCII (`\uXXXX`-escaped) variant better than the raw-UTF-8 variant; a
non-distilled model won't.

## Scripts

| File                          | Notes                                              |
|-------------------------------|----------------------------------------------------|
| `run_o1_ascii_unicode.py`     | Byte-identical copy of `NewScripts/run_reference_attack_o1_controlled.py` |
| `run_o1_ascii_unicode.sh`     | Slurm wrapper iterating the same target/reference sweep used in the paper |

Per-file `transform` in `FILES_MAP`:
- `None`     — leave the parsed text as-is (e.g. `o1_..._unicode.jsonl`)
- `"escape"` — re-escape every non-ASCII codepoint back to its literal
              `\uXXXX` form. Without this, `json.loads` collapses
              `*_ascii.jsonl` (which stored characters as `\uXXXX`)
              and `*_unicode.jsonl` (raw UTF-8) to the *same* Python
              string, and the MIA scores come out identical.

The escape transform also asserts the output is pure ASCII so silent
non-conversions are caught early.

## Running

```bash
export HF_TOKEN=hf_...
sbatch o1_detection/run_o1_ascii_unicode.sh
```

The sweep iterates the same (target, reference) pairs used in the paper
(R1-distills, s1.1, gemma family, gpt-oss). Edit the `MODELS=( ... )`
array in the `.sh` to add or remove pairs.

Inputs come from `../data/o1/` (override via `DATASETS_DIR=...`).
Outputs land per-pair under `../outputs/o1_detection/` (override via
`BASE_OUTDIR=...`).
