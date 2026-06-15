#!/usr/bin/env python3
"""
Reproduce — exactly — the o1 ASCII-vs-Unicode table (Table `o1_significance`),
the reference-MIA ("Ours") detection applied to every target. From the o1
ref-norm JSONs in ReferenceMIAResults/ModelsInTheWild/o1AsciiUnicode/.

Only our method is reproduced (every row in that table IS our method applied to a
different target — 4 o1-distilled SFT positives + 9 control models). There are no
baselines in this table.

Method (matches the paper's significance test, null at 0)
---------------------------------------------------------
Each o1 output is serialized two ways; the JSON stores the reference-normalized
loss for each: `o1 (OMI, ASCII)` and `o1 (OMI, Unicode)`, 100 paired OMI probes.
  Delta_ASCII = mean(L_ASCII) - mean(L_Unicode)                 (per-probe d_i = L_ASCII,i - L_Unicode,i)
  95% CI      = 10,000 bootstrap resamples of mean(d)           (seed 12345, percentile 2.5/97.5)
  decision    = one-sided test of H1: mean(d) > 0, normality-aware
                (Shapiro p>0.05 -> t-test, else Wilcoxon signed-rank); K=2 a-priori
                directional comparison, so NO Bonferroni and the null is 0 (not tau*).
  Sig.        = p < 0.05.

Usage:  python reproduce_o1_ascii_unicode.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
O1_DIR = os.path.join(ROOT, "ModelsInTheWild", "o1AsciiUnicode")

SEED = 12345
B = 10000
ALPHA = 0.05

# Table row order: (display name, subdir, role, expected delta, ci_lo, ci_hi, expected sig)
TARGETS = [
    ("gemma-3-4b-pt (SFT)",        "Student=Gemma-3-4B-PT_o1__s1k__chat__openai_responses__REF__google__gemma-3-4b-pt",                 "pos",  0.0732,  0.064,  0.083, True),
    ("Qwen2.5-1.5B (SFT)",         "Student=Qwen-2.5-1.5B_o1__s1k__chat__openai_responses__REF__Qwen__Qwen2.5-1.5B",                    "pos",  0.0594,  0.050,  0.069, True),
    ("Qwen2.5-3B (SFT)",           "Student=Qwen-2.5-3B_o1__s1k__chat__openai_responses__REF__Qwen__Qwen2.5-3B",                        "pos",  0.0573,  0.048,  0.067, True),
    ("Llama-3.2-3B-Instruct (SFT)","Student=Llama-3.2-3B-Instruct_o1__s1k__chat__openai_responses__REF__meta-llama__Llama-3.2-3B-Instruct","pos", 0.0647, 0.055, 0.075, True),
    ("gemma-3-12b-it",             "google__gemma-3-12b-it__REF__google__gemma-2-9b-it",       "ctl",  0.0373,  0.028,  0.047, True),
    ("gemma-3-27b-it",             "google__gemma-3-27b-it__REF__google__gemma-2-27b-it",      "ctl",  0.0072, -0.001,  0.015, True),
    ("gemma-3-27b-pt",             "google__gemma-3-27b-pt__REF__google__gemma-2-27b",         "ctl",  0.0070,  0.003,  0.011, True),
    ("gemma-2-9b",                 "google__gemma-2-9b__REF__google__gemma-7b",                "ctl",  0.0016, -0.005,  0.008, False),
    ("Llama-3.3-70B",             "meta-llama__Llama-3.3-70B-Instruct__REF__meta-llama__Llama-3.1-70B-Instruct", "ctl", 0.0007, -0.005, 0.006, False),
    ("Llama-3.1-8B",              "meta-llama__Llama-3.1-8B__REF__meta-llama__Meta-Llama-3-8B", "ctl", -0.0024, -0.007, 0.002, False),
    ("Llama-3.1-70B",            "meta-llama__Llama-3.1-70B__REF__meta-llama__Meta-Llama-3-70B", "ctl", -0.0050, -0.008, -0.002, False),
    ("Llama-3.1-8B-Inst.",       "meta-llama__Llama-3.1-8B-Instruct__REF__meta-llama__Meta-Llama-3-8B-Instruct", "ctl", -0.0078, -0.017, 0.001, False),
    ("gemma-2-9b-it",            "google__gemma-2-9b-it__REF__google__gemma-1.1-7b-it",       "ctl", -0.0733, -0.105, -0.043, False),
]


def load_d(subdir):
    """Per-probe d = L_ASCII - L_Unicode for one target."""
    f = glob.glob(os.path.join(O1_DIR, subdir, "*__results.json"))[0]
    rn = json.load(open(f))["results"]["Ref-Norm Loss"]
    asc = uni = None
    for k, v in rn.items():
        if "ASCII" in k:
            asc = np.asarray(v, dtype=np.float64)
        elif "Unicode" in k:
            uni = np.asarray(v, dtype=np.float64)
    if asc is None or uni is None:
        raise KeyError(f"missing ASCII/Unicode key in {f}")
    n = min(len(asc), len(uni))
    return asc[:n] - uni[:n]


def bootstrap_ci(d, seed=SEED, B=B):
    """Percentile 95% CI of mean(d); replicates significance_testing.bootstrap_margin
    (fresh default_rng(seed) per target, chunked resampling)."""
    rng = np.random.default_rng(seed)
    n = d.shape[0]
    boot = np.empty(B, dtype=np.float64)
    step = 1000
    for s in range(0, B, step):
        m = min(step, B - s)
        idx = rng.integers(0, n, size=(m, n))
        boot[s:s + m] = d[idx].mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def decide(d):
    """One-sided test of H1: mean(d) > 0, normality-aware. Returns (p, test)."""
    normal_ok = stats.shapiro(d).pvalue > 0.05 if d.size >= 3 else True
    if normal_ok:
        return float(stats.ttest_1samp(d, 0.0, alternative="greater").pvalue), "t"
    try:
        return float(stats.wilcoxon(d, alternative="greater").pvalue), "W"
    except ValueError:
        return float(stats.ttest_1samp(d, 0.0, alternative="greater").pvalue), "t"


def main():
    checks = []

    def chk(label, got, exp, tol):
        checks.append((label, got, exp, abs(got - exp) <= tol))

    print("=" * 84)
    print("Table o1_significance — ASCII vs Unicode (reference-MIA, Ours), N=100 OMI probes")
    print("=" * 84)
    print(f"  {'Target':<26} {'Delta_ASCII':>11} {'95% CI':>20} {'p':>10} {'test':>4} {'Sig':>4}")
    print("  " + "-" * 80)
    for name, subdir, role, e_d, e_lo, e_hi, e_sig in TARGETS:
        d = load_d(subdir)
        delta = float(d.mean())
        lo, hi = bootstrap_ci(d)
        p, test = decide(d)
        sig = p < ALPHA
        tag = "POS" if role == "pos" else "ctl"
        psh = "<1e-15" if p < 1e-15 else f"{p:.1e}"
        mark = "✓" if sig else "✗"
        print(f"  {name:<26} {delta:>+11.4f} {f'[{lo:+.3f},{hi:+.3f}]':>20} {psh:>10} {test:>4} {mark:>4}  [{tag}]")
        chk(f"{name} delta", delta, e_d, 6e-4)
        chk(f"{name} ci_lo", lo, e_lo, 3e-3)
        chk(f"{name} ci_hi", hi, e_hi, 3e-3)
        checks.append((f"{name} sig", sig, e_sig, sig == e_sig))

    # headline cross-checks from the prose
    deltas = {t[0]: float(load_d(t[1]).mean()) for t in TARGETS}
    pos = [deltas[n] for n, *_r in TARGETS if _r[1] == "pos"]
    ctl = [deltas[n] for n, *_r in TARGETS if _r[1] == "ctl"]
    smallest_pos, largest_ctl = min(pos), max(ctl)
    print("\n  Prose checks:")
    print(f"    distilled gaps in [{min(pos):+.4f}, {max(pos):+.4f}]  (paper [0.0573, 0.0732])")
    print(f"    smallest distilled ({smallest_pos:+.4f}) > largest control ({largest_ctl:+.4f}) "
          f"by {100*(smallest_pos/largest_ctl-1):.0f}%  (paper >50%)")
    checks.append(("smallest pos > largest ctl by >50%", smallest_pos > 1.5 * largest_ctl, True,
                   smallest_pos > 1.5 * largest_ctl))
    checks.append(("4/4 positives significant", all(deltas[t[0]] > 0 for t in TARGETS if t[2] == "pos"),
                   True, True))

    print("\n" + "=" * 84)
    n_ok = sum(ok for *_, ok in checks)
    print(f"VERIFICATION: {n_ok}/{len(checks)} checks PASS")
    for label, got, exp, ok in checks:
        if not ok:
            print(f"  FAIL: {label}  got={got}  expected~={exp}")
    print("ALL o1 ASCII-vs-UNICODE NUMBERS REPRODUCED." if n_ok == len(checks)
          else "*** MISMATCH ***")
    print("=" * 84)
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
