# ReferenceMIAResults

All reference-normalized-loss result JSONs for the paper, gathered into one tree,
plus self-contained scripts that **reproduce every teacher-identification and
threshold-generalization number exactly** and replot the ranked CDF figures.

These JSONs are the canonical result files for the analysis. The wild ones mirror
the GPU runs written to `../outputs/reference_mia_*` (gitignored); the controlled /
OMI-CoT / open-question results live only here. Everything the reproduction needs
is in this folder — the scripts read nothing outside it.

## Layout

```
ReferenceMIAResults/
├── controlled/                     19 controlled-student JSONs (4-teacher pool)
├── OMI_COT/                        highly-customized-instruction (OMI-COT) students
│   ├── WithFewShotPrompting/         + in-context exemplars from S   (4)
│   └── WithoutFewShotPrompting/      default                         (4)
├── ModelsInTheWild/
│   ├── ReferenceMIA/               6 DeepSeek-R1 distills + s1.1-32B  (7)
│   └── o1AsciiUnicode/             o1 ASCII-vs-Unicode probes:
│                                     4 controlled o1-SFT students
│                                     + R1-distill / gemma / llama targets (no GPT)
└── OpenQuestions/
    ├── ReferenceMIA/               QwQ-32B, DeepSeek-R1 (ref MoE-16B), GPT-OSS (3)
    └── o1AsciiUnicode/             R1 o1 gap + 3 GPT-OSS o1 probes
```

Each file is a reference-MIA result: `results["Ref-Norm Loss"][candidate] =
[per-probe stored losses]`, where `stored = loss_student - loss_reference`
(lower = more like that candidate's teacher).

## Scripts (`scripts/`)

```bash
cd scripts
python reproduce_tables.py                   # teacher-identification section (Ours)
python reproduce_threshold_generalization.py # threshold-generalization section (Ours)
python reproduce_o1_ascii_unicode.py         # o1 ASCII-vs-Unicode table (controlled, Ours)
python reproduce_open_questions.py           # open-world o1_sig table (Ours)
python plot_figures.py                       # replot ranked CDF figures -> ../figures/
python gather_results.py                     # rebuild THIS tree from raw GPU outputs
```

Only **our** (reference-based) results are verified by these scripts — baselines
(length-matching, string methods, non-reference MIA, R1-answers-only) are out of
scope and not recomputed here.

### `gather_results.py` — rebuild this tree from raw GPU outputs
The `reference_mia/` and `o1_detection/` scripts write per-pair `*__results.json`
under `outputs/...` with producer-side names; this curated tree uses fixed folders
and filenames. `gather_results.py` rebuilds the exact curated layout: for each raw
output it reads the file's own `model_name` / `ref_model` / candidate-key list,
looks the resulting content-key up in the committed `results_manifest.json`, and
copies the file to its checked-in relative path **verbatim** (filenames and all).
Nothing is renamed by rule and no consumer script changes — the rebuilt tree is
byte-identical, so the `reproduce_*.py` / `plot_figures.py` paths keep resolving.

```bash
python gather_results.py                       # scan ../../outputs -> ../../ReferenceMIAResults_rebuilt
python gather_results.py --inputs DIR [DIR..]  # scan specific output dirs
python gather_results.py --in-place            # write into ReferenceMIAResults/ itself
```

By default it writes to a sibling `ReferenceMIAResults_rebuilt/` (gitignored) so you
can `diff -r` against the committed tree before trusting it. It reports placed /
unmatched / not-yet-produced counts. `results_manifest.json` is regenerated only
when the curated layout changes, via `_gen_results_manifest.py`.

**Two files are NOT gathered** (manifest `source="combine"`): the DeepSeek-R1
results referenced to DeepSeek-MoE-16B —
`OpenQuestions/ReferenceMIA/.../DeepSeek-R1__results.json` and
`OpenQuestions/o1AsciiUnicode/DeepSeek-R1__o1_ascii_unicode__results.json`. These
are built by `_build_from_collected_losses.py` from the OLD MoE-16B collected
losses and are intentionally static (a fresh MoE-16B run is not comparable; see the
builder's header). They ship checked-in and are left untouched.

The reproduce scripts read **only** from this tree and end with a
`VERIFICATION: N/N checks PASS` line that asserts each cell against the paper.

### Every paper result → how to get it

| Paper result | Command | Checks |
|---|---|---|
| Table: controlled teacher-ID, **Raw likelihood (Ours)** row | `reproduce_tables.py` | 28/28 |
| Table: `top1_controlled` (per-probe p-ranges) | `reproduce_tables.py` | ↑ |
| Table: customized-instructions (Ours + few-shot) | `reproduce_tables.py` | ↑ |
| Table: real-world avg, **Raw likelihood (Ours)** (OMI) | `reproduce_tables.py` | ↑ |
| Table: `top1_wild` (R1 p-range) | `reproduce_tables.py` | ↑ |
| X-Coder ranking (Qwen-3-235B + R1 top) | `reproduce_tables.py` | ↑ |
| §threshold-generalization: τ=0.067 (85.5%/53.6%), **both CV tables** (LOSO/LOTO, detection p\*) | `reproduce_threshold_generalization.py` | 38/38 |
| Table: controlled o1 ASCII-vs-Unicode (`o1_significance`, all 13 targets, Δ + CI + p) | `reproduce_o1_ascii_unicode.py` | 54/54 |
| Table: open-world o1 (`o1_sig`, R1 + GPT-OSS) + open-world rankings | `reproduce_open_questions.py` | 25/25 |
| **All p-values** (top-1 controlled/wild, threshold detection p\*, o1 controlled + open-world) | the four scripts above | — |
| **ASCII-vs-Unicode gap for every target** | `reproduce_o1_ascii_unicode.py` (13 controlled) + `reproduce_open_questions.py` (4 open-world) | — |
| All CDF figures | `plot_figures.py` → `figures/` | — |

### `reproduce_tables.py` — teacher identification
Reproduces, exactly:
- `teacher_identification_controlled`, **Raw likelihood (Ours)** row: Llama
  100.0/100.0, Qwen 100.0/100.0, GPT-OSS 100.0/99.9 (Agg. / Per-sample).
- `top1_controlled`: per-probe top-1 exact-binomial p-ranges vs 1/4 — GPT-OSS
  3.9e-121–2.3e-118, Qwen 3.9e-121–1.4e-113, Llama 3.9e-121–3.5e-71 (19/19 sig).
- `teacher_identification_controlled_customized_instructions` (full): Random
  25/25; Ours Default 100.0/100.0, Customized 50.0/65.9; + in-context 75.0/67.1.
- `teacher_identification_real_world_avg`, **Raw likelihood (Ours)** (OMI, 7
  models): 100.0/97.2.
- `top1_wild`: DeepSeek-R1 p-range vs 1/10 = 1.0e-200–3.4e-138 (14/14 sig).

### `reproduce_threshold_generalization.py` — threshold generalization
Uses **fold-fit τ** (and in-sample τ=0.067); it deliberately does **not** use
τ*=0.1009 (that is the previous section's headline, not this one). Reproduces:
- In-sample τ=0.067 → 85.5% (65/76) controlled; same τ unchanged on real-world →
  53.6% (15/28).
- `threshold-generalization-controlled` per teacher — Detected, Detection p*,
  Acc(S-out), Acc(T-out), FP(S-out), FP(T-out):
  GPT-OSS 6/6 `<1e-26` 24/24 15/24 0/6 4/6; Qwen 6/6 `<1e-20` 22/24 15/24 0/6
  1/6; Llama 1/7 `~1e-34–1.0` 17/28 16/28 0/7 0/7; Overall 13/19, 63/76 (82.9%),
  46/76 (60.5%), 0/19, 5/19.
  (Llama's lower p* prints as the true value 6.1e-34; the paper rounds to 1e-34.)
- `threshold-generalization-wild` (leave-one-R1-student-out, fold τ≈0.176):
  DeepSeek-R1 included 13/14, detected 6/7, p* 1.2e-26–8.0e-3; removed 13/14,
  0/7 false detections, p*≥0.078; overall 26/28 (92.9%). The lone miss is
  s1.1-32B (OMI margin 0.105 < fold τ).

Metric definitions (per-sample = sorted/CDF comparison; Agg = max of mean-score
and sorted-vote; detection = normality-aware one-sided test vs τ, Bonferroni
×(K−1), both prompts naming the same teacher) are documented in each script's
header and match the paper's evaluation code.

### `reproduce_o1_ascii_unicode.py` — o1 ASCII vs Unicode (Table `o1_significance`)
All 13 rows are our method (Δ_ASCII = mean(L_ASCII) − mean(L_Unicode) via reference
MIA): 4 o1-distilled SFT positives + 9 controls. Reproduces every Δ_ASCII, 95%
bootstrap CI (seed 12345, B=10,000), p-value (one-sided, t or Wilcoxon, null at 0,
K=2 so no Bonferroni), and significance flag — 4/4 positives + the 3 Gemma-3
controls significant; smallest distilled gap (+0.0573) exceeds the largest control
(+0.0373) by 54%. `VERIFICATION: 54/54 PASS`.

### `reproduce_open_questions.py` — open-world o1 diagnostic (Table `o1_sig`)
The only significance result in the open-questions section (the ranking figures
there are qualitative, no p-values). δ_{Uni−ASCII} = mean(L_ASCII)−mean(L_Unicode)
with bootstrap CI + normality-aware test (null 0) for DeepSeek-R1 (ref MoE-16B,
+0.937, p=8.5e-16) and the three GPT-OSS configs. `VERIFICATION: 20/20 PASS`.
(The paper labels the GPT-OSS-120B/ref-GPT-OSS-20B row "t", but its diffs are
non-normal and the reported p=3.5e-8 is the Wilcoxon value — governing test is
Wilcoxon.)

### `plot_figures.py` — figures (→ `figures/`)
Self-contained (styling vendored in `_ranked_style.py`; reads only this tree).
Writes 11 figures:

| Paper Fig. | Figure | What it shows |
|---|---|---|
| **Fig. 2** | `controlled_qwen2.5-3b_from_gptoss120b_omi_ranked` | Controlled example: Qwen-2.5-3B ← GPT-OSS-120B, OMI probe — representative ref-MIA CDF |
| **Fig. 4** | `OMI_Fewshot_Qwen-2.5-3B` | OMI-CoT teacher-ID for Qwen-2.5-3B: **Without** \| **With** few-shot, side-by-side w/ legend (4 candidates) |
| **Fig. 5** | `s11_32b_wild_s1_ranked` | Real-world s1.1-32B, **s1** true data — R1 ranked top |
| **Fig. 6** | `gemma3_o1_ascii_vs_unicode_ranked` | o1 ASCII/Unicode: Gemma-3 non-distilled \| o1-distilled |
| **Fig. 7** | `qwq32b_ranked` | Open-world QwQ-32B (ref QwQ-Preview), s1 \| OMI — R1 top |
| **Fig. 8** | `r1_refmia_s1_with_o1_ascii_ranked` | DeepSeek-R1 ref-MIA (s1, ref MoE-16B) \| R1 ASCII/Unicode (p=8.5e-16) |
| **Fig. 9** | `gptoss_120b_ascii_with_20b_omi_ranked` | GPT-OSS-120B, **ref GPT-2 XL for both panels**: candidate ranking \| ASCII/Unicode (p=2.7e-16) |
| appendix | `gptoss_20b_vs_120b_omi_ranked` | Open-world GPT-OSS-20B \| 120B, OMI (ref GPT-2 XL) |
| appendix | `gptoss_120b_ref_20b_ranked` | Open-world GPT-OSS-120B (ref GPT-OSS-20B), s1 \| OMI |
| appendix | `gptoss_20b_wild_omi_with_ascii_ranked` | GPT-OSS-20B candidates (ref GPT-2 XL) \| GPT-OSS-120B ASCII/Unicode |
| appendix | `llama31_8b_o1_ascii_vs_unicode_ranked` | o1 ASCII/Unicode controls: Llama-3.1-8B \| -8B-Instruct |

Note: `gptoss_120b_ascii_with_20b_omi_ranked`'s `with_20b` is a legacy filename —
**both** of its panels are GPT-OSS-120B with **GPT-2 XL** as reference (left = its
reference-MIA candidate ranking, OMI; right = its o1 ASCII-vs-Unicode gap).

The ranked GPT-OSS/QwQ ones + `llama31` reproduce the published PNGs byte-for-byte;
`gemma3`/`gptoss_120b_ascii` match the paper figures (built from the canonical
table-consistent JSONs here); `r1_refmia` is curve-identical (AA only). Figures are
gitignored (`*.png`/`*.pdf`) — regenerated on demand.
```
