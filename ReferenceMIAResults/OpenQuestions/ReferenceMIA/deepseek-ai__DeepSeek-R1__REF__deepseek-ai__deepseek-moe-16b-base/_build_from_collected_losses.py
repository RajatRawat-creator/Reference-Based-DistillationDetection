#!/usr/bin/env python3
"""
Provenance / rebuild script for the R1 (ref: DeepSeek-MoE-16B-Base) wild result.

This subfolder holds the data behind the published two-panel figure
`outputs/ranked_style_figures/r1_refmia_s1_with_o1_ascii_ranked.png`
(left = DeepSeek-R1 reference-MIA, s1 prompts; right = o1 ASCII-vs-Unicode gap).

IMPORTANT — this is the OLD MoE-16B scoring run. Per project note
[[moe16b_o1_scoring_incompat]], re-running the MoE-16B losses with the current
DistillDetectRelease scripts produces values that are NOT comparable to this old
R1 data, so the published figure must be reproduced from THIS data, not from a
fresh reference_mia run. `DeepSeek-R1__results.json` below is therefore derived
purely by subtracting the OLD per-row losses (no model is re-run): it only
precomputes the exact `target - reference` step the plotter already did at draw
time, paired by row index.

Source files (in the working tree, outside the release):
  target : R1testUnicode/outputs_r1_collected/DeepSeek-R1__collected_losses.json
  ref    : NewScripts/outputs_deepseekmoe16b_collected/deepseek-moe-16b-base__collected_losses.json

Each `*_collected_losses.json` stores, per candidate teacher key,
`results[key]["rows"] = [{idx, loss, ok, ...}]`. We keep rows with ok==True and
a finite loss, intersect target/ref on idx per key, and store
`ref_norm = target_loss - ref_loss` (the standard wild `__results.json` schema:
results["Ref-Norm Loss"][key] = [per-probe values]).

Run from the repo root to regenerate `DeepSeek-R1__results.json`:
    python DistillDetectRelease/ReferenceMIAResults/OpenQuestions/ReferenceMIA/\
deepseek-ai__DeepSeek-R1__REF__deepseek-ai__deepseek-moe-16b-base/_build_from_collected_losses.py
"""
import json
import math
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
# repo root = .../distill_detect  (5 levels up from this file)
ROOT = HERE.parents[4]

TGT = ROOT / "R1testUnicode/outputs_r1_collected/DeepSeek-R1__collected_losses.json"
REF = ROOT / "NewScripts/outputs_deepseekmoe16b_collected/deepseek-moe-16b-base__collected_losses.json"
OUT = HERE / "DeepSeek-R1__results.json"


def rows_by_idx(path):
    """{candidate_key: {idx: loss}} keeping ok rows with finite loss."""
    res = json.load(open(path))["results"]
    out = {}
    for key, blk in res.items():
        m = {}
        rows = blk.get("rows", []) if isinstance(blk, dict) else []
        for r in rows:
            if not r.get("ok", True):
                continue
            try:
                v = float(r.get("loss"))
            except (TypeError, ValueError):
                continue
            if math.isfinite(v):
                m[int(r.get("idx", len(m)))] = v
        out[key] = m
    return out


def main():
    tgt = rows_by_idx(TGT)
    ref = rows_by_idx(REF)

    # Candidate-label convention: collapse the source's "(OMI, Trace + Response)"
    # / "(s1, Response Only)" / "(…, Response)" qualifiers down to just "(OMI)" /
    # "(s1)", matching the rest of ReferenceMIAResults (o1 ASCII/Unicode labels are
    # not in this file, so nothing else is affected).
    import re as _re
    _collapse = lambda k: _re.sub(r"\((OMI|s1),[^)]*\)", lambda m: f"({m.group(1)})", k)

    ref_norm = {}
    for key in sorted(set(tgt) & set(ref)):
        idxs = sorted(set(tgt[key]) & set(ref[key]))
        ref_norm[_collapse(key)] = [tgt[key][i] - ref[key][i] for i in idxs]

    blob = {
        "model_name": "DeepSeek-R1",
        "ref_model": "deepseek-ai/deepseek-moe-16b-base",
        "note": "Ref-norm losses derived from OLD collected-losses runs; see "
                "_build_from_collected_losses.py. Not comparable to fresh "
                "DistillDetectRelease MoE-16B runs.",
        "source_target_json": str(TGT.relative_to(ROOT)),
        "source_ref_json": str(REF.relative_to(ROOT)),
        "results": {"Ref-Norm Loss": ref_norm},
        "fail_counts": {k: 0 for k in ref_norm},
    }
    with open(OUT, "w") as f:
        json.dump(blob, f, indent=2)
    print(f"[ok] wrote {OUT}")
    print(f"[ok] {len(ref_norm)} candidate keys; "
          f"n per key (sample): "
          + ", ".join(f"{k.split(' (')[0]}={len(v)}" for k, v in list(ref_norm.items())[:4]))


if __name__ == "__main__":
    main()
