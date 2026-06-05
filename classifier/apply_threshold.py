"""
Apply the margin-threshold classifier to the R1-distill real-world models.

Pipeline:
  1. Learn a threshold from the controlled benchmark CSV
     (data/controlled_dataset_10f.csv) using either the max-F1 search
     (default) or the 95th percentile of label=0 margins (--threshold p95).
  2. Score each real-world model by reading its
     `*__results.json` file produced by `mia/run_reference_attack.py`.
  3. Print per-row predictions and accuracy summaries split by prompt
     (OMI vs s1) and condition (with_true vs without_true).

Excluded from candidate pool : QwQ-32B, Qwen-3-235B-A22B-Thinking-2507
Excluded from metrics        : DeepSeek-R1-Distill-Qwen-1.5B

Usage:
    python3 eval_threshold.py
    python3 eval_threshold.py --threshold 0.086
    python3 eval_threshold.py --threshold p95
    python3 eval_threshold.py --eval-controlled       # report on controlled CSV
    python3 eval_threshold.py --wild-dir /path/to/ReferenceMIA/outputs

Equivalent to the original FINALGITHUBALLSCRIPTS/Classifier/eval_threshold_r1distill.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Defaults --------------------------------------------------------------
# Where reference-attack `*__results.json` files live for the real-world
# (a.k.a. "wild") models. Override at the CLI with --wild-dir.
WILD_BASE = os.environ.get(
    "DD_WILD_DIR",
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "reference_mia", "wild",
    ),
)

# Controlled benchmark CSV lives under repo-root/data/classifier/.
DEFAULT_CONTROLLED = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "classifier",
    "controlled_dataset_10f.csv",
)

EXCLUDED_CANDIDATES = {"QwQ-32B", "Qwen-3-235B-A22B-Thinking-2507"}
EXCLUDED_FROM_METRICS = {"DeepSeek-R1-Distill-Qwen-1.5B"}

R1_SUBDIRS = [
    "deepseek-ai__DeepSeek-R1-Distill-Llama-70B__REF__meta-llama__Llama-3.3-70B-Instruct",
    "deepseek-ai__DeepSeek-R1-Distill-Llama-8B__REF__meta-llama__Llama-3.1-8B",
    "deepseek-ai__DeepSeek-R1-Distill-Qwen-14B__REF__Qwen__Qwen2.5-14B",
    "deepseek-ai__DeepSeek-R1-Distill-Qwen-1.5B__REF__Qwen__Qwen2.5-Math-1.5B",
    "deepseek-ai__DeepSeek-R1-Distill-Qwen-32B__REF__Qwen__Qwen2.5-32B",
    "deepseek-ai__DeepSeek-R1-Distill-Qwen-7B__REF__Qwen__Qwen2.5-Math-7B",
    "simplescaling__s1.1-32B__REF__Qwen__Qwen2.5-32B-Instruct",
]


def is_excluded(key: str) -> bool:
    return any(key.startswith(p + " (") for p in EXCLUDED_CANDIDATES)


def compute_margin(ref_norm: dict, prompt: str, remove_r1: bool = False):
    excl_prefixes = set(EXCLUDED_CANDIDATES)
    if remove_r1:
        excl_prefixes.add("DeepSeek R1")
    cands = [k for k in ref_norm if f"({prompt}," in k and
             not any(k.startswith(p + " (") for p in excl_prefixes)]
    scores = {c: -float(np.mean(ref_norm[c])) for c in cands}
    ranked = sorted(scores, key=lambda c: scores[c], reverse=True)
    margin = scores[ranked[0]] - scores[ranked[1]]
    top = ranked[0].split(" (")[0]
    return margin, top, len(cands)


def find_threshold_f1(df: pd.DataFrame) -> float:
    margins = df["margin"].values
    y = df["label"].values
    best_t, best_f1 = float(margins.min()) - 1e-9, -1.0
    for t in np.unique(margins):
        preds = (margins >= t).astype(int)
        s = f1_score(y, preds, zero_division=0)
        if s >= best_f1:
            best_f1, best_t = s, t
    return best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--threshold", default="f1",
        help="Threshold value, 'f1' (max-F1 search on controlled), or 'p95' (95th pct of label=0 margins). Default: f1"
    )
    ap.add_argument("--controlled", default=DEFAULT_CONTROLLED)
    ap.add_argument("--wild-dir", default=WILD_BASE)
    ap.add_argument(
        "--eval-controlled", action="store_true",
        help="Evaluate threshold on the controlled dataset instead of wild models."
    )
    args = ap.parse_args()

    # ── learn threshold from controlled data ──────────────────────────────────
    df_ctrl = pd.read_csv(args.controlled)
    if args.threshold == "f1":
        threshold = find_threshold_f1(df_ctrl)
        thresh_label = f"max-F1 on controlled = {threshold:.4f}"
    elif args.threshold == "p95":
        threshold = float(df_ctrl[df_ctrl.label == 0]["margin"].quantile(0.95))
        thresh_label = f"p95 of label=0 margins = {threshold:.4f}"
    else:
        threshold = float(args.threshold)
        thresh_label = f"manual = {threshold:.4f}"

    print(f"Threshold : {thresh_label}")
    print(f"Excluded candidates : {sorted(EXCLUDED_CANDIDATES)}")
    print(f"Excluded from metrics: {sorted(EXCLUDED_FROM_METRICS)}")
    print()

    # ── controlled evaluation mode ────────────────────────────────────────────
    if args.eval_controlled:
        df = df_ctrl.copy()
        df["pred"] = (df["margin"] >= threshold).astype(int)
        df["correct"] = df["pred"] == df["label"]

        print(f"{'Student':<25} {'Teacher family':<25} {'Prompt':>5}  {'Condition':<13}  {'Margin':>7}  {'Pred':>3}  {'True':>4}  OK")
        print("-" * 105)
        prev = None
        for _, r in df.iterrows():
            if r.student != prev and prev is not None:
                print()
            pred_str = "DISTILLED" if r.pred == 1 else "NOT DIST."
            true_str = "DISTILLED" if r.label == 1 else "NOT DIST."
            ok = "YES" if r.correct else "FAIL <--"
            print(f"{r.student:<25} {r.true_teacher_family:<25} {r.probe_prompt:>5}  {r.candidate_setting:<13}  {r.margin:>7.4f}  {pred_str:<9}  {true_str:<9}  {ok}")
            prev = r.student

        print()
        print("=" * 50)
        total_c = df.correct.sum()
        total_n = len(df)
        print(f"  OVERALL: {total_c}/{total_n} = {total_c/total_n:.1%}")
        for lbl, lbl_name in [(1, "with_true (label=1)"), (0, "without_true (label=0)")]:
            sub = df[df.label == lbl]
            print(f"  {lbl_name:<22}: {sub.correct.sum()}/{len(sub)} = {sub.correct.mean():.1%}")
        for prompt in ["OMI", "s1"]:
            sub = df[df.probe_prompt == prompt]
            print(f"  {prompt:<22}: {sub.correct.sum()}/{len(sub)} = {sub.correct.mean():.1%}")
        fp = df[(df.label == 0) & (df.pred == 1)]
        fn = df[(df.label == 1) & (df.pred == 0)]
        print()
        print(f"  False positives (label=0 predicted distilled): {len(fp)}")
        for _, r in fp.iterrows():
            print(f"    {r.student:<25} {r.true_teacher_family:<25} {r.probe_prompt}  margin={r.margin:.4f}")
        print(f"  False negatives (label=1 predicted not distilled): {len(fn)}")
        for _, r in fn.iterrows():
            print(f"    {r.student:<25} {r.true_teacher_family:<25} {r.probe_prompt}  margin={r.margin:.4f}")
        return

    # ── collect rows ──────────────────────────────────────────────────────────
    rows = []
    for subdir in R1_SUBDIRS:
        jsons = glob.glob(os.path.join(args.wild_dir, subdir, "*__results.json"))
        if not jsons:
            print(f"[WARN] no results.json in {subdir}")
            continue
        with open(jsons[0]) as f:
            blob = json.load(f)
        student = blob["model_name"]
        ref = blob["results"]["Ref-Norm Loss"]
        excluded = student in EXCLUDED_FROM_METRICS

        for prompt in ["OMI", "s1"]:
            for remove_r1, cond, true_lbl in [
                (False, "with_true",    "DISTILLED"),
                (True,  "without_true", "NOT DISTILLED"),
            ]:
                margin, top, n_cands = compute_margin(ref, prompt, remove_r1)
                pred = "DISTILLED" if margin >= threshold else "NOT DISTILLED"
                correct = pred == true_lbl
                rows.append({
                    "model": student, "prompt": prompt, "condition": cond,
                    "top_candidate": top, "margin": margin,
                    "pred": pred, "true": true_lbl,
                    "correct": correct, "excluded": excluded,
                    "n_cands": n_cands,
                })

    df = pd.DataFrame(rows)

    # ── full table ────────────────────────────────────────────────────────────
    print(f"{'Model':<35} {'Prompt':>5}  {'Condition':<13}  {'Top candidate':<25} {'Margin':>7}  {'Pred':<14}  {'True':<14}  OK")
    print("-" * 120)
    prev = None
    for _, r in df.iterrows():
        if r.model != prev and prev is not None:
            print()
        excl_tag = " (excl)" if r.excluded else ""
        ok = "YES" if r.correct else "FAIL <--"
        print(f"{r.model+excl_tag:<35} {r.prompt:>5}  {r.condition:<13}  {r.top_candidate:<25} {r.margin:>7.4f}  {r.pred:<14}  {r['true']:<14}  {ok}")
        prev = r.model

    # ── summary tables ────────────────────────────────────────────────────────
    metric = df[~df.excluded]
    print()
    print("=" * 55)
    print("ACCURACY SPLIT BY PROMPT  (Qwen-1.5B excluded)")
    print("=" * 55)
    for prompt in ["OMI", "s1"]:
        sub = metric[metric.prompt == prompt]
        print(f"  {prompt}: {sub.correct.sum()}/{len(sub)} = {sub.correct.mean():.1%}")

    print()
    print("ACCURACY BY PROMPT x CONDITION")
    print("-" * 45)
    for prompt in ["OMI", "s1"]:
        for cond in ["with_true", "without_true"]:
            sub = metric[(metric.prompt == prompt) & (metric.condition == cond)]
            print(f"  {prompt} {cond:<13}: {sub.correct.sum()}/{len(sub)} = {sub.correct.mean():.1%}")

    total_c = metric.correct.sum()
    total_n = len(metric)
    print()
    print(f"  OVERALL: {total_c}/{total_n} = {total_c/total_n:.1%}")


if __name__ == "__main__":
    main()
