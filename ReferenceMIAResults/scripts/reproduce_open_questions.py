#!/usr/bin/env python3
"""
Reproduce — exactly — the open-questions significance table (Table `o1_sig`):
the o1 Unicode-vs-ASCII diagnostic applied to the open-world targets
(DeepSeek-R1 and the GPT-OSS family), from the ref-norm JSONs in
ReferenceMIAResults/OpenQuestions/o1AsciiUnicode/.

This is the ONLY significance result in the open-questions section — the
reference-MIA ranking figures there (QwQ-32B, DeepSeek-R1, GPT-OSS) are reported
qualitatively with NO p-values (single-teacher / reference-inadequacy caveats),
so nothing else in that section has a significance number to reproduce.

Method (matches the paper's significance test, null at 0; same as the controlled
o1 table, just on the open-world targets)
------------------------------------------------------------------------------
delta_{Uni-ASCII} = mean(f_Unicode) - mean(f_ASCII) = mean(L_ASCII) - mean(L_Unicode),
the within-model ASCII-minus-Unicode gap in reference-normalized loss; per-probe
d_i = L_ASCII,i - L_Unicode,i over 100 OMI probes.
  95% CI   = 10,000 bootstrap resamples of mean(d)   (seed 12345, percentile 2.5/97.5)
  decision = one-sided H1: mean(d) > 0, normality-aware (Shapiro p>0.05 -> t-test,
             else Wilcoxon signed-rank). K=2 a-priori directional -> no Bonferroni.
  Sig.     = p < 0.05.

Usage:  python reproduce_open_questions.py
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
O1_DIR = os.path.join(ROOT, "OpenQuestions", "o1AsciiUnicode")
RMIA_DIR = os.path.join(ROOT, "OpenQuestions", "ReferenceMIA")

SEED = 12345
B = 10000
ALPHA = 0.05

# Same-family exclusion for the open-world reference-MIA ranking: drop the
# near-variant of the target from the candidate pool to avoid a trivial
# same-family match. ONLY these two (per the paper): testing GPT-OSS (20B or
# 120B) drops GPT-OSS-120B; testing QwQ-32B drops QwQ-32B-Preview. DeepSeek-R1
# uses the full pool (no near-variant in the candidate set).
SAME_FAMILY_EXCLUDE = {"qwq-32b": "QwQ-32B Preview", "gpt-oss": "GPT-OSS-120B"}


def excluded_candidate(target_name):
    t = target_name.lower()
    if "qwq-32b" in t:
        return SAME_FAMILY_EXCLUDE["qwq-32b"]
    if "gpt-oss" in t:
        return SAME_FAMILY_EXCLUDE["gpt-oss"]
    return None


# Open-world reference-MIA configs (target dir under OpenQuestions/ReferenceMIA, label).
# Reported as RANKINGS only (qualitative) — the paper attaches NO p-values here.
OPEN_WORLD_RANK = [
    ("Qwen__QwQ-32B__REF__Qwen__QwQ-32B-Preview",                 "QwQ-32B (ref QwQ-32B-Preview)"),
    ("deepseek-ai__DeepSeek-R1__REF__deepseek-ai__deepseek-moe-16b-base", "DeepSeek-R1 (ref MoE-16B)"),
    ("openai__gpt-oss-120b__REF__openai-community__gpt2-xl",      "GPT-OSS-120B (ref GPT-2 XL)"),
    ("openai__gpt-oss-120b__REF__openai__gpt-oss-20b",            "GPT-OSS-120B (ref GPT-OSS-20B)"),
    ("openai__gpt-oss-20b__REF__openai-community__gpt2-xl",       "GPT-OSS-20B (ref GPT-2 XL)"),
]


def rank_candidates(subdir, label, probe):
    """Top candidates by mean teacher-likeness (higher = -mean stored loss),
    after the same-family exclusion. Returns ordered list of teacher names."""
    f = glob.glob(os.path.join(RMIA_DIR, subdir, "*__results.json"))[0]
    rn = json.load(open(f))["results"]["Ref-Norm Loss"]
    drop = excluded_candidate(label)
    scores = {}
    for k, v in rn.items():
        # tolerant: matches both "(OMI, …)" and the collapsed "(OMI)"
        if f"({probe}," not in k and f"({probe})" not in k:
            continue
        t = k.split(" (")[0]
        if drop is not None and t == drop:
            continue
        scores[t] = -float(np.mean(v))
    return sorted(scores, key=scores.get, reverse=True)

# (display name, results-json path relative to O1_DIR, expected delta, ci_lo, ci_hi, expected p)
TARGETS = [
    ("DeepSeek-R1 (ref DeepSeek-MoE-16B-Base)", "DeepSeek-R1__o1_ascii_unicode__results.json",
     0.937, 0.746, 1.141, 8.5e-16),
    ("GPT-OSS-120B (ref GPT-OSS-20B)", "openai__gpt-oss-120b__REF__openai__gpt-oss-20b/gpt-oss-120b__results.json",
     0.321, 0.206, 0.439, 3.5e-8),
    ("GPT-OSS-120B (ref GPT-2 XL)", "openai__gpt-oss-120b__REF__openai-community__gpt2-xl/gpt-oss-120b__results.json",
     0.831, 0.690, 0.980, 2.7e-16),
    ("GPT-OSS-20B (ref GPT-2 XL)", "openai__gpt-oss-20b__REF__openai-community__gpt2-xl/gpt-oss-20b__results.json",
     0.510, 0.391, 0.641, 1.3e-13),
]


def load_d(relpath):
    f = glob.glob(os.path.join(O1_DIR, relpath))[0]
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

    # ---- Open-world reference-MIA RANKING (qualitative; same-family exclusion) ----
    print("=" * 88)
    print("Open-world reference-MIA ranking (qualitative — no p-values per paper)")
    print("Same-family exclusion: QwQ-32B drops QwQ-32B-Preview; GPT-OSS drops GPT-OSS-120B")
    print("=" * 88)
    for subdir, label in OPEN_WORLD_RANK:
        drop = excluded_candidate(label)
        omi = rank_candidates(subdir, label, "OMI")
        s1 = rank_candidates(subdir, label, "s1")
        print(f"  {label}   (excluded: {drop or 'none'})")
        print(f"     OMI top3: {omi[:3]}")
        print(f"     s1  top3: {s1[:3]}")
    # checks backing the qualitative claims
    qwq_omi = rank_candidates(OPEN_WORLD_RANK[0][0], OPEN_WORLD_RANK[0][1], "OMI")
    qwq_s1 = rank_candidates(OPEN_WORLD_RANK[0][0], OPEN_WORLD_RANK[0][1], "s1")
    checks.append(("QwQ-32B top=DeepSeek R1 (OMI)", qwq_omi[0], "DeepSeek R1", qwq_omi[0] == "DeepSeek R1"))
    checks.append(("QwQ-32B top=DeepSeek R1 (s1)", qwq_s1[0], "DeepSeek R1", qwq_s1[0] == "DeepSeek R1"))
    checks.append(("QwQ-32B-Preview excluded from pool", "QwQ-32B Preview" not in qwq_omi, True,
                   "QwQ-32B Preview" not in qwq_omi))
    gpt_tops = {rank_candidates(s, l, "OMI")[0] for s, l in OPEN_WORLD_RANK if "GPT-OSS" in l}
    checks.append(("GPT-OSS top shifts with reference (>1 distinct top)", len(gpt_tops) > 1, True, len(gpt_tops) > 1))
    gpt_excl_ok = all("GPT-OSS-120B" not in rank_candidates(s, l, "OMI")
                      for s, l in OPEN_WORLD_RANK if "GPT-OSS" in l)
    checks.append(("GPT-OSS-120B excluded from GPT-OSS pools", gpt_excl_ok, True, gpt_excl_ok))

    print("\n" + "=" * 88)
    print("Table o1_sig — open-world o1 Unicode-vs-ASCII diagnostic (reference-MIA, Ours), N=100")
    print("=" * 88)
    print(f"  {'Target':<40} {'delta':>8} {'95% CI':>20} {'test':>5} {'p':>10} {'Sig':>4}")
    print("  " + "-" * 84)
    for name, rel, e_d, e_lo, e_hi, e_p in TARGETS:
        d = load_d(rel)
        delta = float(d.mean())
        lo, hi = bootstrap_ci(d)
        p, test = decide(d)
        sig = p < ALPHA
        print(f"  {name:<40} {delta:>+8.3f} {f'[{lo:+.3f},{hi:+.3f}]':>20} {test:>5} {p:>10.1e} {('✓' if sig else '✗'):>4}")
        chk(f"{name} delta", delta, e_d, 2e-3)
        chk(f"{name} ci_lo", lo, e_lo, 5e-3)
        chk(f"{name} ci_hi", hi, e_hi, 5e-3)
        chk(f"{name} log10(p)", np.log10(p), np.log10(e_p), 0.3)
        checks.append((f"{name} sig", sig, True, sig is True))

    print("\n" + "=" * 88)
    n_ok = sum(ok for *_, ok in checks)
    print(f"VERIFICATION: {n_ok}/{len(checks)} checks PASS")
    for label, got, exp, ok in checks:
        if not ok:
            print(f"  FAIL: {label}  got={got}  expected~={exp}")
    print("ALL OPEN-QUESTIONS o1_sig NUMBERS REPRODUCED." if n_ok == len(checks)
          else "*** MISMATCH ***")
    print("Note: the paper labels the GPT-OSS-120B (ref GPT-OSS-20B) row 'test=t', but its")
    print("per-probe diffs are non-normal (Shapiro p=7e-4); the reported p=3.5e-8 is the")
    print("Wilcoxon value (t-test would give 1.9e-7), so the governing test is Wilcoxon.")
    print("=" * 88)
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
