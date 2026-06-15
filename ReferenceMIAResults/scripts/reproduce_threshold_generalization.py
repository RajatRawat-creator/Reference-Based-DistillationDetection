#!/usr/bin/env python3
"""
Reproduce — exactly — every number in the paper's threshold-generalization
section ("This section evaluates how the margin-threshold detector ...
generalizes beyond the models used to calibrate it"), from the
reference-normalized-loss JSONs gathered in ReferenceMIAResults/.

Self-contained: reads ONLY from ReferenceMIAResults/. NOTHING ELSE is computed —
in particular this section uses fold-fit tau (and the in-sample tau=0.067); it
does NOT use tau* = 0.1009 (that is the teacher-identification headline of the
previous section, not this one).

Numbers reproduced
------------------
* In-sample max-F1 threshold tau = 0.067 -> 85.5% (65/76) on the controlled set;
  the SAME tau applied unchanged to the 28 real-world cells -> 53.6% (15/28).
* Table `threshold-generalization-controlled` (per true teacher):
      Detected, Detection p*, Acc (S-out), Acc (T-out), FP (S-out), FP (T-out)
      GPT-OSS-120B           6/6  <1e-26          24/24  15/24  0/6  4/6
      Qwen-3-8B              6/6  <1e-20          22/24  15/24  0/6  1/6
      Llama-3.3-70B-Instruct 1/7  1e-34 -- 1.0    17/28  16/28  0/7  0/7
      Overall               13/19 ---             63/76  46/76  0/19 5/19
  (S-out overall 82.9% = 63/76; T-out overall 60.5% = 46/76.)
* Table `threshold-generalization-wild` (leave-one-R1-student-out, fold-fit tau):
      DeepSeek-R1 included   Acc 13/14  Detected 6/7  p* 1.2e-26 -- 8.0e-3
      DeepSeek-R1 removed    Acc 13/14  Detected 0/7  p* >= 0.078 (n.s.)
      Overall                Acc 26/28 (92.9%)
  The lone missed positive is s1.1-32B (OMI margin 0.105 < fold tau 0.176).

Definitions (identical to the paper)
------------------------------------
Per evaluation cell (distillation x prompt x condition) we form the per-probe
paired difference d_i = f(t1) - f(t2), where f = -(stored ref-norm loss), t1 is
the top candidate by mean f and t2 the runner-up over the candidate pool
(y=1: true teacher present; y=0: true teacher removed). margin = mean(d).
  * Acc (margin prediction): correct iff (margin >= tau) == y, with tau re-fit
    (max-F1 on the training fold's margins) per fold.
        - S-out = leave-one-base-student-out (4 controlled folds; 7 wild folds)
        - T-out = leave-one-teacher-out      (3 controlled folds)
  * Detection test (Detected / FP / Detection p*): one-sided normality-aware
    threshold test of H1: mean(d) > tau (t-test if Shapiro p>0.05 else Wilcoxon
    on d-tau), p* = min(1, p*(K-1)); CALL iff p* < 0.05 on BOTH prompts AND both
    prompts name the same top teacher t1. Detected = called with the true teacher
    present (y=1); FP = a substitute called when the true teacher is removed (y=0).
    Detection p* = range of the per-prompt p* values (fold-fit tau).

Usage
-----
    python reproduce_threshold_generalization.py
"""
from __future__ import annotations

import glob
import json
import os
import re
from collections import defaultdict

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from scipy import stats

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CTRL_DIR = os.path.join(ROOT, "controlled")
WILD_DIR = os.path.join(ROOT, "ModelsInTheWild", "ReferenceMIA")

CANDIDATE_TEACHERS = ["GPT-OSS-120B", "Llama-3.3-70B-Instruct", "Qwen-3-8B", "Gemma-3-27B-it"]
CTRL_TEACHERS = ["GPT-OSS-120B", "Qwen-3-8B", "Llama-3.3-70B-Instruct"]  # print order
TEACHER_CANON = {"Nvidia-Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct"}
PROMPTS = ["OMI", "s1"]

WILD_STUDENTS = [
    "DeepSeek-R1-Distill-Llama-70B", "DeepSeek-R1-Distill-Llama-8B",
    "DeepSeek-R1-Distill-Qwen-1.5B", "DeepSeek-R1-Distill-Qwen-7B",
    "DeepSeek-R1-Distill-Qwen-14B", "DeepSeek-R1-Distill-Qwen-32B", "s1.1-32B",
]
WILD_TEACHER = "DeepSeek R1"


# --------------------------------------------------------------------------- #
# Shared statistics (identical to the paper's CV code)
# --------------------------------------------------------------------------- #
def fit_tau(margins, labels):
    """max-F1 threshold; ties broken by accuracy then lower tau."""
    m = np.asarray(margins, float)
    y = np.asarray(labels, int)
    cand = sorted(set(m.tolist()))
    grid = [-1e9] + [(cand[i] + cand[i + 1]) / 2 for i in range(len(cand) - 1)] + [1e9]
    best = None
    for t in grid:
        pr = (m >= t).astype(int)
        tp = int(((pr == 1) & (y == 1)).sum())
        fp = int(((pr == 1) & (y == 0)).sum())
        fn = int(((pr == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        key = (f1, float((pr == y).mean()), -t)
        if best is None or key > best[0]:
            best = (key, t)
    return best[1]


def sig_pstar(d, K, tau):
    """Normality-aware one-sided threshold test vs tau, Bonferroni x (K-1)."""
    normal_ok = stats.shapiro(d).pvalue > 0.05 if d.size >= 3 else True
    if normal_ok:
        p = stats.ttest_1samp(d, tau, alternative="greater").pvalue
    else:
        try:
            p = stats.wilcoxon(d - tau, alternative="greater").pvalue
        except ValueError:
            p = stats.ttest_1samp(d, tau, alternative="greater").pvalue
    return min(1.0, p * (K - 1))


def top2_d(arrs_by_cand):
    """arrs_by_cand: {name: f-array}. Returns (t1_name, d=f(t1)-f(t2), K)."""
    names = list(arrs_by_cand)
    F = np.column_stack([arrs_by_cand[n] for n in names])
    F = F[np.all(np.isfinite(F), axis=1)]
    order = np.argsort(F.mean(axis=0))[::-1]
    t1, t2 = int(order[0]), int(order[1])
    return names[t1], F[:, t1] - F[:, t2], F.shape[1]


# --------------------------------------------------------------------------- #
# Cell construction
# --------------------------------------------------------------------------- #
def controlled_cells():
    """One dict per (distillation x prompt x condition). f = -(stored loss),
    candidates use the SAME prompt's keys (not held-out cross-probe)."""
    cells = []
    pat = os.path.join(CTRL_DIR, "Student=*_Teacher=*_Data=*(1K)_Template=Chat.json")
    for fp in sorted(glob.glob(pat)):
        b = os.path.basename(fp)
        student = re.search(r"Student=([^_]+)", b).group(1)
        traw = re.search(r"Teacher=(.+?)_Data=", b).group(1)
        teacher = TEACHER_CANON.get(traw, traw)
        data = re.search(r"Data=([^(_]+)", b).group(1)
        rn = json.load(open(fp))["results"]["Ref-Norm Loss"]
        distill = f"{student}|{teacher}|{data}"
        for prompt in PROMPTS:
            for y, cset in ((1, list(CANDIDATE_TEACHERS)),
                            (0, [c for c in CANDIDATE_TEACHERS if c != teacher])):
                arrs = {c: -np.asarray(rn[f"{c} ({prompt}, Chat)"], float) for c in cset}
                t1, d, K = top2_d(arrs)
                cells.append(dict(student=student, teacher=teacher, data=data,
                                  distill=distill, prompt=prompt, y=y,
                                  d=d, K=K, t1=t1, margin=float(d.mean())))
    return cells


def wild_cells():
    by_name = {}
    for f in glob.glob(os.path.join(WILD_DIR, "*", "*__results.json")):
        by_name[json.load(open(f)).get("model_name")] = f
    cells = []
    for s in WILD_STUDENTS:
        rn = json.load(open(by_name[s]))["results"]["Ref-Norm Loss"]
        for prompt in PROMPTS:
            for y, drop in ((1, None), (0, WILD_TEACHER)):
                arrs = {}
                for k in rn:
                    # tolerant: matches both "(OMI, …)" and the collapsed "(OMI)"
                    if f"({prompt}," not in k and f"({prompt})" not in k:
                        continue
                    nm = k.split(" (")[0]
                    if drop and nm == drop:
                        continue
                    arrs[nm] = -np.asarray(rn[k], float)
                t1, d, K = top2_d(arrs)
                cells.append(dict(student=s, prompt=prompt, y=y,
                                  d=d, K=K, t1=t1, margin=float(d.mean())))
    return cells


# --------------------------------------------------------------------------- #
# Margin-accuracy cross-validation
# --------------------------------------------------------------------------- #
def margin_cv(cells, group_key):
    """Per-cell correctness of (margin>=tau)==y, tau re-fit per fold."""
    m = np.array([c["margin"] for c in cells])
    y = np.array([c["y"] for c in cells])
    grp = np.array([c[group_key] for c in cells])
    corr = np.zeros(len(cells), int)
    taus = {}
    for g in sorted(set(grp.tolist())):
        te = grp == g
        tau = fit_tau(m[~te], y[~te])
        taus[g] = tau
        corr[te] = ((m[te] >= tau).astype(int) == y[te]).astype(int)
    return corr, taus


def detect_by_distill(cells, tau_for, label):
    """{distill: (called, t1, [per-prompt p*])} for the given condition label.
    tau_for(cell) -> fold tau for that cell."""
    by = defaultdict(dict)
    for c in cells:
        if c["y"] == label:
            by[c["distill"]][c["prompt"]] = c
    out = {}
    for dl, byp in by.items():
        ps = {p: sig_pstar(byp[p]["d"], byp[p]["K"], tau_for(byp[p])) for p in PROMPTS}
        t1s = {p: byp[p]["t1"] for p in PROMPTS}
        called = all(ps[p] < 0.05 for p in PROMPTS) and len(set(t1s.values())) == 1
        out[dl] = (called, t1s["OMI"], list(ps.values()))
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def main():
    checks = []

    def chk(label, got, exp, tol=0):
        ok = (got == exp) if tol == 0 else (abs(got - exp) <= tol)
        checks.append((label, got, exp, ok))

    cc = controlled_cells()
    wc = wild_cells()
    cm = np.array([c["margin"] for c in cc]); cy = np.array([c["y"] for c in cc])

    # ---- Part 1: in-sample tau and unchanged transfer -----------------------
    tau0 = fit_tau(cm, cy)
    in_acc = int(((cm >= tau0).astype(int) == cy).sum())
    wild_unchanged = sum(int((c["margin"] >= 0.067) == c["y"]) for c in wc)
    print("=" * 80)
    print("Threshold generalization (fold-fit tau; NOT tau*=0.1009)")
    print("=" * 80)
    print(f"In-sample max-F1 tau = {tau0:.4f} -> controlled acc {in_acc}/{len(cc)} "
          f"= {100*in_acc/len(cc):.1f}%   (paper 0.067, 85.5%, 65/76)")
    print(f"Same tau=0.067 applied UNCHANGED to real-world: {wild_unchanged}/28 "
          f"= {100*wild_unchanged/28:.1f}%   (paper 53.6%, 15/28)")
    chk("tau0", round(tau0, 3), 0.067, tol=0.001)
    chk("in-sample acc", in_acc, 65); chk("wild unchanged", wild_unchanged, 15)

    # ---- Part 2: controlled CV per-teacher table ----------------------------
    loso_c, loso_tau = margin_cv(cc, "student")
    loto_c, loto_tau = margin_cv(cc, "teacher")
    tea = np.array([c["teacher"] for c in cc])
    det_sout = detect_by_distill(cc, lambda c: loso_tau[c["student"]], 1)
    fp_sout  = detect_by_distill(cc, lambda c: loso_tau[c["student"]], 0)
    fp_tout  = detect_by_distill(cc, lambda c: loto_tau[c["teacher"]], 0)

    # group distillations by teacher
    dist_teacher = {c["distill"]: c["teacher"] for c in cc}
    by_teacher = defaultdict(list)
    for dl, t in dist_teacher.items():
        by_teacher[t].append(dl)

    print("\n" + "=" * 80)
    print("Table: threshold-generalization-controlled  (by true teacher)")
    print("=" * 80)
    print(f"  {'True teacher':<24} {'Det.':>5} {'Detection p*':>16} "
          f"{'Acc(S)':>7} {'Acc(T)':>7} {'FP(S)':>6} {'FP(T)':>6}")
    exp_ctrl = {
        "GPT-OSS-120B":           dict(det=(6,6), accS=(24,24), accT=(15,24), fpS=(0,6), fpT=(4,6)),
        "Qwen-3-8B":              dict(det=(6,6), accS=(22,24), accT=(15,24), fpS=(0,6), fpT=(1,6)),
        "Llama-3.3-70B-Instruct": dict(det=(1,7), accS=(17,28), accT=(16,28), fpS=(0,7), fpT=(0,7)),
    }
    tot = dict(det=0, n=0, accS=0, accT=0, nS=0, fpS=0, fpT=0, nstud=0)
    for t in CTRL_TEACHERS:
        dls = by_teacher[t]
        # Detected = called (p*<0.05 both prompts, same teacher) AND names the TRUE teacher.
        ndet = sum(1 for dl in dls if det_sout[dl][0] and det_sout[dl][1] == t)
        # per-prompt p* range over ALL y=1 cells of this teacher (S-out tau)
        pstars = [p for dl in dls for p in det_sout[dl][2]]
        idx = tea == t
        accS = int(loso_c[idx].sum()); accT = int(loto_c[idx].sum()); nc = int(idx.sum())
        nfpS = sum(fp_sout[dl][0] for dl in dls)
        nfpT = sum(fp_tout[dl][0] for dl in dls)
        plo, phi = min(pstars), max(pstars)
        if phi < 1e-4:  # all cells highly significant -> report upper bound (next power of 10)
            prange = f"<1e{int(np.ceil(np.log10(phi)))}"
        else:
            prange = f"{plo:.0e}--{phi:.1f}"
        print(f"  {t:<24} {f'{ndet}/{len(dls)}':>5} {prange:>16} "
              f"{f'{accS}/{nc}':>7} {f'{accT}/{nc}':>7} "
              f"{f'{nfpS}/{len(dls)}':>6} {f'{nfpT}/{len(dls)}':>6}")
        e = exp_ctrl[t]
        chk(f"{t} detected", ndet, e["det"][0]); chk(f"{t} ndist", len(dls), e["det"][1])
        chk(f"{t} accS", accS, e["accS"][0]); chk(f"{t} accT", accT, e["accT"][0])
        chk(f"{t} fpS", nfpS, e["fpS"][0]); chk(f"{t} fpT", nfpT, e["fpT"][0])
        # p* sanity
        if t == "GPT-OSS-120B": chk("GPT p* max<1e-26", int(phi < 1e-26), 1)
        if t == "Qwen-3-8B":    chk("Qwen p* max<1e-20", int(phi < 1e-20), 1)
        if t == "Llama-3.3-70B-Instruct":
            chk("Llama p* min~1e-34", int(1e-35 < plo < 1e-33), 1)
            chk("Llama p* max==1.0", round(phi, 6), 1.0)
        tot["det"] += ndet; tot["n"] += len(dls)
        tot["accS"] += accS; tot["accT"] += accT; tot["nS"] += nc
        tot["fpS"] += nfpS; tot["fpT"] += nfpT; tot["nstud"] += len(dls)
    o_det = f"{tot['det']}/{tot['n']}"
    o_accS = f"{tot['accS']}/{tot['nS']}"; o_accT = f"{tot['accT']}/{tot['nS']}"
    o_fpS = f"{tot['fpS']}/{tot['nstud']}"; o_fpT = f"{tot['fpT']}/{tot['nstud']}"
    print(f"  {'Overall':<24} {o_det:>5} {'---':>16} "
          f"{o_accS:>7} {o_accT:>7} {o_fpS:>6} {o_fpT:>6}")
    print(f"  S-out overall {tot['accS']}/{tot['nS']} = {100*tot['accS']/tot['nS']:.1f}%   "
          f"T-out overall {tot['accT']}/{tot['nS']} = {100*tot['accT']/tot['nS']:.1f}%")
    chk("overall detected", tot["det"], 13); chk("overall accS", tot["accS"], 63)
    chk("overall accT", tot["accT"], 46); chk("overall fpS", tot["fpS"], 0)
    chk("overall fpT", tot["fpT"], 5)

    # ---- Part 3: wild leave-one-student-out table ---------------------------
    wm = [(c["student"], c["y"], c["margin"]) for c in wc]
    inc_acc = rem_acc = det = fp = 0
    called_pstars = []
    removed_pstars_all = []
    fold_taus = []
    for s in WILD_STUDENTS:
        trm = np.array([m for (ss, l, m) in wm if ss != s])
        try_ = np.array([l for (ss, l, m) in wm if ss != s])
        tau = fit_tau(trm, try_); fold_taus.append(tau)
        scells = [c for c in wc if c["student"] == s]
        byp1 = {c["prompt"]: c for c in scells if c["y"] == 1}
        byp0 = {c["prompt"]: c for c in scells if c["y"] == 0}
        for p in PROMPTS:
            inc_acc += int((byp1[p]["margin"] >= tau) == 1)
            rem_acc += int((byp0[p]["margin"] >= tau) == 0)
        ps1 = {p: sig_pstar(byp1[p]["d"], byp1[p]["K"], tau) for p in PROMPTS}
        t1s = {p: byp1[p]["t1"] for p in PROMPTS}
        called = all(ps1[p] < 0.05 for p in PROMPTS) and len(set(t1s.values())) == 1 and t1s["OMI"] == WILD_TEACHER
        ps0 = {p: sig_pstar(byp0[p]["d"], byp0[p]["K"], tau) for p in PROMPTS}
        t0s = {p: byp0[p]["t1"] for p in PROMPTS}
        fp_fire = all(ps0[p] < 0.05 for p in PROMPTS) and len(set(t0s.values())) == 1
        det += int(called); fp += int(fp_fire)
        if called:
            called_pstars += list(ps1.values())  # per-prompt p* of called students
        removed_pstars_all += list(ps0.values())  # all per-prompt p* of removed cells

    print("\n" + "=" * 80)
    print("Table: threshold-generalization-wild  (leave-one-R1-student-out, fold tau)")
    print("=" * 80)
    print(f"  fold tau range: {min(fold_taus):.4f}--{max(fold_taus):.4f}  (paper ~0.176)")
    clo, chi = min(called_pstars), max(called_pstars)
    print(f"  {'Held-out condition':<24} {'Accuracy':>9} {'Detected':>9} {'Detection p*':>22}")
    print(f"  {'DeepSeek-R1 included':<24} {f'{inc_acc}/14':>9} {f'{det}/7':>9} "
          f"{f'{clo:.1e} -- {chi:.1e}':>22}")
    rem_min = min(removed_pstars_all)
    print(f"  {'DeepSeek-R1 removed':<24} {f'{rem_acc}/14':>9} {f'{fp}/7':>9} "
          f"{f'>= {rem_min:.3f} (n.s.)':>22}")
    print(f"  {'Overall':<24} {f'{inc_acc+rem_acc}/28':>9} "
          f"= {100*(inc_acc+rem_acc)/28:.1f}%   (paper 26/28, 92.9%)")
    chk("wild inc acc", inc_acc, 13); chk("wild rem acc", rem_acc, 13)
    chk("wild overall acc", inc_acc + rem_acc, 26)
    chk("wild detected", det, 6); chk("wild FP", fp, 0)
    chk("wild p* lo~1.2e-26", int(1e-27 < clo < 1e-25), 1)
    chk("wild p* hi~8.0e-3", int(abs(chi - 8.0e-3) < 5e-4), 1)
    chk("wild removed p*>=0.078", int(abs(rem_min - 0.078) < 5e-4), 1)

    # ---- Verification -------------------------------------------------------
    print("\n" + "=" * 80)
    n_ok = sum(ok for *_, ok in checks)
    print(f"VERIFICATION: {n_ok}/{len(checks)} checks PASS")
    for label, got, exp, ok in checks:
        if not ok:
            print(f"  FAIL: {label}  got={got}  expected={exp}")
    print("ALL THRESHOLD-GENERALIZATION NUMBERS REPRODUCED."
          if n_ok == len(checks) else "*** MISMATCH ***")
    print("=" * 80)
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
