#!/usr/bin/env python3
"""Render the ranked-side-panel style across a set of figures, all the same
size / colors / legend format, into one folder:

  1. qwq32b_ranked                  - QwQ-32B (Ref: QwQ-32B Preview), s1 | OMI
  2. gptoss_20b_vs_120b_omi_ranked  - GPT-OSS-20B | GPT-OSS-120B targets, OMI
  3. gptoss_120b_ref_20b_ranked     - GPT-OSS-120B (Ref: GPT-OSS-20B), s1 | OMI
  4. gptoss_20b_wild_omi_with_ascii_ranked
                                    - GPT-OSS-20B OMI candidates | GPT-OSS-120B
                                      ASCII vs Unicode (direct-labelled)

Shared style: no top legend; a ranked side panel (ordered by avg score@90 over
the data panels); stable per-model ColorBrewer-Paired colors (dark = focal,
light = answers-only baselines drawn thinner/underneath); short names; a
"* answers only" + reference note.
"""
import json
import math
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Output directory; the caller (plot_figures.py) overrides this. Default: a
# `figures/` folder next to ReferenceMIAResults/.
OUT_DIR = Path(__file__).resolve().parent.parent / "figures"

# ---- canonical (stable across every figure) -------------------------------
COLOR = {
    "DeepSeek R1":            "#e31a1c",  # dark red
    "QwQ-32B Preview":        "#6a3d9a",  # dark purple
    "Qwen-3-235B":            "#b15928",  # brown (XCoder candidate only)
    "GPT-OSS-120B":           "#1f78b4",  # dark blue
    "Claude Opus 4.5":        "#33a02c",  # dark green
    "Claude Opus 4.6":        "#ff7f00",  # dark orange
    "o1":                     "#cab2d6",  # light purple
    "o3":                     "#fdbf6f",  # light orange
    "Gemma-3-27B-it":         "#a6cee3",  # light blue   (answers only)
    "Llama-3.3-70B-Instruct": "#b2df8a",  # light green  (answers only)
    "Claude-3.5-Sonnet":      "#fb9a99",  # light red    (answers only)
}
FOCAL = {"DeepSeek R1", "QwQ-32B Preview", "Qwen-3-235B", "GPT-OSS-120B",
         "Claude Opus 4.5", "Claude Opus 4.6", "o1", "o3"}
ANSWERS_ONLY = {"Gemma-3-27B-it", "Llama-3.3-70B-Instruct", "Claude-3.5-Sonnet"}
# models marked with a "*": these were scored on summarized thinking traces
# instead of the original traces
STARRED = {"Claude Opus 4.5", "Claude Opus 4.6", "o1", "o3"}
SHORT = {
    "DeepSeek R1": "DeepSeek R1", "QwQ-32B Preview": "QwQ 32B Preview",
    "Qwen-3-235B": "Qwen 3 235B",
    "GPT-OSS-120B": "GPT-OSS 120B", "Claude Opus 4.5": "Claude Opus 4.5",
    "Claude Opus 4.6": "Claude Opus 4.6", "o1": "o1", "o3": "o3",
    "Gemma-3-27B-it": "Gemma 3 27B", "Llama-3.3-70B-Instruct": "Llama 3.3 70B Instruct",
    "Claude-3.5-Sonnet": "Claude 3.5 Sonnet",
}

# ASCII/Unicode are token-type categories, not models -> colors deliberately
# OUTSIDE the candidate palette (so no color means two things across the set;
# e.g. #1f78b4 stays reserved for GPT-OSS-120B, which is a candidate elsewhere).
ASCII_COLOR = {"ASCII": "#17becf", "Unicode": "#000000"}

# width matches qwq32b_two_panel.png (1763 px @ 160 dpi); height unchanged
FIGSIZE = (1763 / 160, 3.7)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def base_and_prompt(name):
    import re
    m = re.match(r"^(.*?)\s*\((s1|OMI)(?:,\s*.*?)?\)\s*$", name)
    return (m.group(1).strip(), m.group(2).strip()) if m else (None, None)


def load_candidate_panels(json_path, exclude=()):
    """Return {prompt: {model: sorted curve}}, jointly normalized over the
    file's kept candidates (s1 + OMI together, matching the originals)."""
    raw = json.load(open(json_path))["results"]["Ref-Norm Loss"]
    kept = {}
    for name, vals in raw.items():
        base, prompt = base_and_prompt(name)
        if base in COLOR and base not in exclude:
            kept[(base, prompt)] = np.asarray(vals, float)
    allv = np.concatenate([v[np.isfinite(v)] for v in kept.values()])
    mu, sigma = allv.mean(), allv.std() + 1e-8
    out = {"s1": {}, "OMI": {}}
    for (base, prompt), v in kept.items():
        v = v[np.isfinite(v)]
        out[prompt][base] = np.sort(sigmoid(-(v - mu) / sigma))
    return out


def _rows_by_idx(json_path):
    """{key: {idx: loss}} from a *_collected_losses.json (raw rows format)."""
    res = json.load(open(json_path))["results"]
    out = {}
    for key, blk in res.items():
        m = {}
        for r in (blk.get("rows", []) if isinstance(blk, dict) else []):
            if not r.get("ok", True):
                continue
            loss = r.get("loss")
            try:
                v = float(loss)
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                m[int(r.get("idx", len(m)))] = v
        out[key] = m
    return out


def load_paired_candidate_panel(ref_json, tgt_json, probe):
    """Reference-norm MIA from raw rows: pair tgt-ref per idx, joint-normalize,
    return {model: sorted curve} for the requested prompt group."""
    ref, tgt = _rows_by_idx(ref_json), _rows_by_idx(tgt_json)
    ref_norm = {}
    for key in set(ref) & set(tgt):
        base, _ = base_and_prompt(key)
        if base is None or base not in COLOR or base == "DeepSeek R1":
            continue
        idxs = sorted(set(ref[key]) & set(tgt[key]))
        paired = np.array([tgt[key][i] - ref[key][i] for i in idxs], float)
        if paired.size:
            ref_norm[key] = paired
    allv = np.concatenate(list(ref_norm.values()))
    mu, sigma = allv.mean(), allv.std() + 1e-8
    by_base = {}
    for key, vals in ref_norm.items():
        base, p = base_and_prompt(key)
        if p == probe:
            by_base.setdefault(base, []).append((key, vals))
    panel = {}
    for base, items in by_base.items():
        # prefer the "Trace + Response" view when several subtypes exist
        items.sort(key=lambda kv: 0 if "Trace + Response" in kv[0] else 1)
        panel[base] = np.sort(sigmoid(-(items[0][1] - mu) / sigma))
    return panel


def load_ascii_panel(json_path):
    raw = json.load(open(json_path))["results"]["Ref-Norm Loss"]
    arrays = {}
    for name, vals in raw.items():
        label = "ASCII" if "ASCII" in name else "Unicode"
        arrays[label] = np.asarray(vals, float)
    allv = np.concatenate([a[np.isfinite(a)] for a in arrays.values()])
    mu, sigma = allv.mean(), allv.std() + 1e-8
    return {k: np.sort(sigmoid(-(a[np.isfinite(a)] - mu) / sigma))
            for k, a in arrays.items()}


def p90(vals):
    return float(vals[int(0.9 * (len(vals) - 1))])


def draw_candidates(ax, panel, title):
    order = [m for m in panel if m not in FOCAL] + \
            [m for m in panel if m in FOCAL]
    for m in order:
        vals = panel[m]
        x = np.linspace(0, 100, vals.size)
        # uniform line width for every model; focal models still read clearly
        # via their saturated colors and being drawn on top (higher zorder)
        ax.plot(x, vals, color=COLOR[m], linewidth=2.4,
                zorder=3 if m in FOCAL else 2)
    _style_axis(ax, title)


def draw_ascii(ax, panel, title):
    for label in ("ASCII", "Unicode"):
        vals = panel[label]
        x = np.linspace(0, 100, vals.size)
        ax.plot(x, vals, color=ASCII_COLOR[label], linewidth=2.4, zorder=3)
    # direct labels at x=55 (lines are well separated there, unlike the x=100 spike)
    i = int(0.55 * (panel["Unicode"].size - 1))
    ax.text(55, panel["Unicode"][i] + 0.05, "Unicode", color=ASCII_COLOR["Unicode"],
            fontsize=10, fontweight="bold", ha="center", va="bottom")
    j = int(0.55 * (panel["ASCII"].size - 1))
    ax.text(55, panel["ASCII"][j] - 0.05, "ASCII", color=ASCII_COLOR["ASCII"],
            fontsize=10, fontweight="bold", ha="center", va="top")
    _style_axis(ax, title)


def _style_axis(ax, title):
    ax.set_title(title, fontsize=12, pad=4)
    ax.set_xlabel("Percentile", fontsize=11)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.tick_params(labelsize=9)
    ax.grid(True, alpha=0.22)


def draw_side(side, ranked):
    side.set_xlim(0, 1)
    side.set_ylim(0, 1)
    side.axis("off")
    side.text(0.0, 0.99, "Ranked by mean score",
              fontsize=10.8, fontweight="bold", va="top")
    n = len(ranked)
    # extra gap below the header so it isn't crowding entry "1."
    y_top, y_bot = 0.86, 0.10
    step = (y_top - y_bot) / max(n - 1, 1)
    for i, m in enumerate(ranked):
        y = y_top - i * step
        side.add_line(Line2D([0.0, 0.10], [y, y], color=COLOR[m],
                             linewidth=3.0, solid_capstyle="round"))
        name = SHORT[m] + (" *" if m in STARRED else "")
        side.text(0.14, y, f"{i + 1}. {name}", fontsize=9.8, va="center")


def make_figure(out_name, axis_panels, axis_titles, reference_note,
                ascii_index=None, ascii_pvalue=None):
    """axis_panels: list of panel dicts (one per data axis). The axis at
    `ascii_index` is an ASCII/Unicode panel; the rest are candidate panels.
    Ranking uses only the candidate panels.

    `ascii_pvalue`, if given, is a preformatted (mathtext) string drawn in the
    bottom-right whitespace of the ASCII/Unicode panel (e.g. the Unicode-vs-ASCII
    separation p-value)."""
    n = len(axis_panels)
    fig = plt.figure(figsize=FIGSIZE)
    gs = fig.add_gridspec(1, n + 1, width_ratios=[3.0] * n + [1.65], wspace=0.07)
    axes = []
    for i in range(n):
        ax = fig.add_subplot(gs[0, i], sharey=axes[0] if axes else None)
        axes.append(ax)
    side = fig.add_subplot(gs[0, n])

    cand_panels = [p for i, p in enumerate(axis_panels) if i != ascii_index]

    def rank_metric(m):
        # rank by the mean score (averaged across the candidate panels)
        vs = [float(np.mean(p[m])) for p in cand_panels if m in p]
        return np.mean(vs) if vs else -np.inf

    models = sorted(set().union(*[set(p) for p in cand_panels]))
    ranked = sorted(models, key=rank_metric, reverse=True)

    for i, (ax, panel, title) in enumerate(zip(axes, axis_panels, axis_titles)):
        if i == ascii_index:
            draw_ascii(ax, panel, title)
            if ascii_pvalue:
                ax.text(0.97, 0.06, ascii_pvalue, transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=14, color="0.15")
        else:
            draw_candidates(ax, panel, title)
    axes[0].set_ylabel("Score", fontsize=11)
    for ax in axes[1:]:
        plt.setp(ax.get_yticklabels(), visible=False)

    draw_side(side, ranked)
    side.text(0.0, 0.05, "* summarized thinking traces\n   instead of original",
              fontsize=8.8, linespacing=1.4,
              style="italic", color="0.35", va="top")
    # wrap on "·" so a long reference note stacks instead of running off the edge
    ref_lines = "\n".join(s.strip() for s in reference_note.split("·"))
    side.text(0.0, -0.10, ref_lines, fontsize=8.6, linespacing=1.5,
              style="italic", color="0.35", va="top")

    # fixed margins (no tight bbox) so every figure is exactly the same size
    plt.subplots_adjust(top=0.88, bottom=0.155, left=0.06, right=0.995, wspace=0.07)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUT_DIR / f"{out_name}.png"
    pdf = OUT_DIR / f"{out_name}.pdf"
    fig.savefig(png, dpi=160)
    fig.savefig(pdf)
    plt.close(fig)
    print(f"[saved] {png.name}  ranked={[SHORT[m] for m in ranked]}")


# Styling/helper module only — figure generation lives in plot_figures.py, which
# imports this module and calls load_candidate_panels / load_ascii_panel / make_figure
# with paths into ReferenceMIAResults/. (Original driver `main()` removed when vendored.)
