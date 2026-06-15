#!/usr/bin/env python3
"""
Replot the paper's reference-MIA / ASCII-vs-Unicode CDF figures from the
reference-normalized-loss JSONs in ReferenceMIAResults/ (self-contained — reads
nothing outside this folder; styling is vendored in `_ranked_style.py`).

Figures written to ReferenceMIAResults/figures/:

  Ranked CDF (2-panel, side legend):
    qwq32b_ranked                          QwQ-32B (ref QwQ-Preview), s1|OMI
    gptoss_20b_vs_120b_omi_ranked          GPT-OSS-20B|120B targets, OMI (ref GPT-2 XL)
    gptoss_120b_ref_20b_ranked             GPT-OSS-120B (ref GPT-OSS-20B), s1|OMI
    gptoss_20b_wild_omi_with_ascii_ranked  GPT-OSS-20B candidates | GPT-OSS-120B ASCII/Unicode
    gptoss_120b_ascii_with_20b_omi_ranked  GPT-OSS-120B candidates | GPT-OSS-120B ASCII/Unicode (p=2.7e-16)
    r1_refmia_s1_with_o1_ascii_ranked      DeepSeek-R1 ref-MIA (s1) | R1 ASCII/Unicode (p=8.5e-16)
    s11_32b_wild_s1_ranked                 s1.1-32B (s1 probe)
    controlled_qwen2.5-3b_from_gptoss120b_omi_ranked   controlled example

  o1 ASCII-vs-Unicode (2-panel, no side legend):
    gemma3_o1_ascii_vs_unicode_ranked      Gemma-3 non-distilled | o1-distilled
    llama31_8b_o1_ascii_vs_unicode_ranked  Llama-3.1-8B | Llama-3.1-8B-Instruct (both non-distilled)

  OMI-CoT few-shot (2-panel, top legend):
    OMI_Fewshot_Qwen-2.5-3B                Qwen-2.5-3B teacher-ID: Without | With few-shot

Run:  python plot_figures.py
"""
import glob
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                              # ReferenceMIAResults/
# self-contained: the styling module is vendored next to this script.
sys.path.insert(0, str(HERE))
import _ranked_style as mod  # noqa: E402

OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
mod.OUT_DIR = OUT

RMIA_W = ROOT / "ModelsInTheWild" / "ReferenceMIA"
RMIA_OQ = ROOT / "OpenQuestions" / "ReferenceMIA"
O1_OQ = ROOT / "OpenQuestions" / "o1AsciiUnicode"
O1_MITW = ROOT / "ModelsInTheWild" / "o1AsciiUnicode"
CTRL = ROOT / "controlled"
OMICOT = ROOT / "OMI_COT"


def one(pat):
    """Resolve a glob to a single path (the JSON we want)."""
    hits = glob.glob(str(pat))
    if not hits:
        raise FileNotFoundError(pat)
    return hits[0]


# --------------------------------------------------------------------------- #
# o1 ASCII-vs-Unicode two-panel figure (vendored from make_o1_ascii_extra_figures)
# --------------------------------------------------------------------------- #
def two_panel_ascii(out_name, left_json, left_title, right_json, right_title):
    left = mod.load_ascii_panel(left_json)
    right = mod.load_ascii_panel(right_json)
    fig = plt.figure(figsize=(10.0, 4.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.07)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1], sharey=ax0)
    mod.draw_ascii(ax0, left, left_title)
    mod.draw_ascii(ax1, right, right_title)
    ax0.set_ylabel("Score", fontsize=11)
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.subplots_adjust(top=0.9, bottom=0.155, left=0.07, right=0.98, wspace=0.09)
    fig.savefig(OUT / f"{out_name}.png", dpi=160)
    fig.savefig(OUT / f"{out_name}.pdf")
    plt.close(fig)
    print(f"[saved] {out_name}")


# --------------------------------------------------------------------------- #
# OMI-CoT few-shot figure (Qwen-2.5-3B): Without | With few-shot, top legend
# --------------------------------------------------------------------------- #
# Four candidate teachers, fixed order + matplotlib-default colors (matches the
# paper figure). The few-shot file labels Llama as "Nvidia-Llama-…" -> normalize.
OMI_TEACHERS = ["GPT-OSS-120B", "Gemma-3-27B-it", "Llama-3.3-70B-Instruct", "Qwen-3-8B"]
OMI_COLORS = {"GPT-OSS-120B": "#1f77b4", "Gemma-3-27B-it": "#ff7f0e",
              "Llama-3.3-70B-Instruct": "#2ca02c", "Qwen-3-8B": "#d62728"}


def _omicot_s1_panel(json_path):
    """{teacher: sorted membership curve} over the s1 candidates, joint-normalized."""
    rn = json.load(open(json_path))["results"]["Ref-Norm Loss"]
    arrs = {}
    for k, v in rn.items():
        if "(s1" not in k:
            continue
        t = k.split(" (")[0]
        if t == "Nvidia-Llama-3.3-70B-Instruct":
            t = "Llama-3.3-70B-Instruct"
        a = np.asarray(v, dtype=np.float64)
        arrs[t] = a[np.isfinite(a)]
    allv = np.concatenate(list(arrs.values()))
    mu, sigma = allv.mean(), allv.std() + 1e-8
    return {t: np.sort(mod.sigmoid(-(a - mu) / sigma)) for t, a in arrs.items()}


def omi_fewshot_qwen3b():
    without = _omicot_s1_panel(one(OMICOT / "WithoutFewShotPrompting" / "*Qwen-2.5-3B*.json"))
    withfs = _omicot_s1_panel(one(OMICOT / "WithFewShotPrompting" / "*Qwen-2.5-3B*.json"))
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10.0, 3.2), sharey=True)
    for ax, panel, title in [(ax0, without, "Without Few Shot"), (ax1, withfs, "With Few Shot")]:
        for t in OMI_TEACHERS:
            vals = panel[t]
            ax.plot(np.linspace(0, 100, vals.size), vals, color=OMI_COLORS[t], linewidth=2.0)
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Percentile", fontsize=11)
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.25)
    ax0.set_ylabel("Score", fontsize=11)
    handles = [Line2D([0], [0], color=OMI_COLORS[t], lw=2.5) for t in OMI_TEACHERS]
    fig.legend(handles, OMI_TEACHERS, loc="upper center", ncol=4, frameon=False,
               fontsize=10, bbox_to_anchor=(0.5, 1.02))
    plt.subplots_adjust(top=0.83, bottom=0.16, left=0.07, right=0.98, wspace=0.05)
    fig.savefig(OUT / "OMI_Fewshot_Qwen-2.5-3B.png", dpi=160, bbox_inches="tight")
    fig.savefig(OUT / "OMI_Fewshot_Qwen-2.5-3B.pdf", bbox_inches="tight")
    plt.close(fig)
    print("[saved] OMI_Fewshot_Qwen-2.5-3B")


def main():
    # ---- ranked CDF figures (2-panel, side legend) ----
    qwq = mod.load_candidate_panels(
        one(RMIA_OQ / "Qwen__QwQ-32B__REF__Qwen__QwQ-32B-Preview/*__results.json"),
        exclude={"QwQ-32B Preview"})
    mod.make_figure("qwq32b_ranked", [qwq["s1"], qwq["OMI"]],
                    ["Prompts: s1", "Prompts: OMI"], "Reference: QwQ-32B Preview")

    p20 = mod.load_candidate_panels(
        one(RMIA_OQ / "openai__gpt-oss-20b__REF__openai-community__gpt2-xl/*__results.json"),
        exclude={"GPT-OSS-120B"})
    p120 = mod.load_candidate_panels(
        one(RMIA_OQ / "openai__gpt-oss-120b__REF__openai-community__gpt2-xl/*__results.json"),
        exclude={"GPT-OSS-120B"})
    mod.make_figure("gptoss_20b_vs_120b_omi_ranked", [p20["OMI"], p120["OMI"]],
                    ["GPT-OSS-20B", "GPT-OSS-120B"], "Reference: GPT-2 XL")

    g = mod.load_candidate_panels(
        one(RMIA_OQ / "openai__gpt-oss-120b__REF__openai__gpt-oss-20b/*__results.json"),
        exclude={"GPT-OSS-120B"})
    mod.make_figure("gptoss_120b_ref_20b_ranked", [g["s1"], g["OMI"]],
                    ["Prompts: s1", "Prompts: OMI"], "Reference: GPT-OSS-20B")

    # GPT-OSS-120B o1 ASCII-vs-Unicode panel (ref GPT-2 XL); shared by both figs below
    g120_ascii = mod.load_ascii_panel(
        one(O1_OQ / "openai__gpt-oss-120b__REF__openai-community__gpt2-xl/*__results.json"))

    # GPT-OSS-20B candidates | GPT-OSS-120B ASCII vs Unicode
    mod.make_figure("gptoss_20b_wild_omi_with_ascii_ranked",
                    [p20["OMI"], g120_ascii], ["GPT-OSS-20B", "GPT-OSS-120B"],
                    "Reference: GPT-2 XL", ascii_index=1)

    # Fig. 9: GPT-OSS-120B — BOTH panels are GPT-OSS-120B w/ GPT-2 XL reference:
    #   left = its ref-MIA candidate ranking (OMI); right = its ASCII/Unicode gap.
    mod.make_figure("gptoss_120b_ascii_with_20b_omi_ranked",
                    [p120["OMI"], g120_ascii], ["GPT-OSS-120B", "GPT-OSS-120B"],
                    "Reference: GPT-2 XL", ascii_index=1,
                    ascii_pvalue=r"$p = 2.7 \times 10^{-16}$")

    # DeepSeek-R1 ref-MIA (s1) | R1 ASCII vs Unicode (p=8.5e-16)
    r1_left = mod.load_candidate_panels(
        one(RMIA_OQ / "deepseek-ai__DeepSeek-R1__REF__deepseek-ai__deepseek-moe-16b-base/*__results.json"),
        exclude={"DeepSeek R1"})["s1"]
    r1_ascii = mod.load_ascii_panel(one(O1_OQ / "DeepSeek-R1__o1_ascii_unicode__results.json"))
    mod.make_figure("r1_refmia_s1_with_o1_ascii_ranked", [r1_left, r1_ascii],
                    ["", ""], "Reference: DeepSeek-MoE-16B", ascii_index=1,
                    ascii_pvalue=r"$p = 8.5 \times 10^{-16}$")

    # s1.1-32B wild (s1 probe)
    s11 = mod.load_candidate_panels(
        one(RMIA_W / "simplescaling__s1.1-32B__REF__Qwen__Qwen2.5-32B-Instruct/*__results.json"))
    mod.make_figure("s11_32b_wild_s1_ranked", [s11["s1"]], ["Prompts: s1"],
                    "Reference: Qwen2.5-32B-Instruct")

    # Controlled Qwen-2.5-3B distilled from GPT-OSS-120B (s1 train) probed OMI
    ctrl = mod.load_candidate_panels(
        one(CTRL / "Student=Qwen-2.5-3B_Teacher=GPT-OSS-120B_Data=s1(1K)_Template=Chat.json"))
    mod.make_figure("controlled_qwen2.5-3b_from_gptoss120b_omi_ranked",
                    [ctrl["OMI"]], ["Prompts: OMI"], "Reference: Qwen2.5-3B")

    # ---- o1 ASCII-vs-Unicode two-panel figures ----
    two_panel_ascii(
        "gemma3_o1_ascii_vs_unicode_ranked",
        one(O1_MITW / "google__gemma-3-27b-pt__REF__google__gemma-2-27b/*__results.json"),
        "Gemma 3 (non-distilled)",
        one(O1_MITW / "Student=Gemma-3-4B-PT_o1*__REF__google__gemma-3-4b-pt/*__results.json"),
        "Gemma 3 (o1-distilled)")
    two_panel_ascii(
        "llama31_8b_o1_ascii_vs_unicode_ranked",
        one(O1_MITW / "meta-llama__Llama-3.1-8B__REF__meta-llama__Meta-Llama-3-8B/*__results.json"),
        "Llama 3.1-8B (non-distilled)",
        one(O1_MITW / "meta-llama__Llama-3.1-8B-Instruct__REF__meta-llama__Meta-Llama-3-8B-Instruct/*__results.json"),
        "Llama 3.1-8B-Instruct (non-distilled)")

    # ---- OMI-CoT few-shot (Qwen-2.5-3B) ----
    omi_fewshot_qwen3b()

    print(f"\n[done] wrote {len(list(OUT.glob('*.png')))} PNGs (+PDFs) to {OUT}")


if __name__ == "__main__":
    main()
