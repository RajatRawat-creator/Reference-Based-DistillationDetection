#!/usr/bin/env python3
"""
Gather/rename raw GPU outputs into the curated ReferenceMIAResults/ layout.

The run_*.py scripts write per-pair `*__results.json` files under outputs/... with
producer-side names; the curated tree uses fixed folders and filenames. This step
rebuilds that exact layout: for each raw `*__results.json` it reads the file's own
`model_name` / `ref_model` / candidate-key list, looks the resulting content-key up
in `results_manifest.json`, and copies the file to the committed relative path
(verbatim filename, including quirks like `... (6).json`). No naming rules are
encoded and no consumer script changes -- the rebuilt tree is byte-identical, so
reproduce_*.py / plot_figures.py keep working unchanged.

Usage:
    python gather_results.py                       # scan ../../outputs, write ../../ReferenceMIAResults_rebuilt
    python gather_results.py --inputs DIR [DIR..]  # scan specific output dirs
    python gather_results.py --dest PATH           # where to write the rebuilt tree
    python gather_results.py --in-place            # write into ReferenceMIAResults/ itself

Two DeepSeek-R1 ref:MoE-16B files are combine artifacts (source="combine" in the
manifest): they are built by their own `_build_from_collected_losses.py` from OLD
out-of-tree losses and are intentionally NOT regenerated here (see project note
moe16b_o1_scoring_incompat). gather lists them but never (re)writes them.
"""
import argparse
import glob
import hashlib
import json
import os
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
RMIA = HERE.parent                      # ReferenceMIAResults/
REPO = RMIA.parent                      # DistillDetectRelease/
MANIFEST = HERE / "results_manifest.json"


def candidate_keys(d):
    r = d.get("results", {})
    blk = r.get("Ref-Norm Loss", r) if isinstance(r, dict) else {}
    return sorted(blk.keys()) if isinstance(blk, dict) else []


def content_key(model_name, ref_model, ckeys):
    payload = json.dumps([model_name, ref_model, sorted(ckeys)], sort_keys=True)
    return hashlib.sha1(payload.encode()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", default=[str(REPO / "outputs")],
                    help="output dirs to scan recursively for *__results.json")
    ap.add_argument("--dest", default=str(REPO / "ReferenceMIAResults_rebuilt"),
                    help="destination root for the rebuilt tree")
    ap.add_argument("--in-place", action="store_true",
                    help="write into ReferenceMIAResults/ itself (overrides --dest)")
    args = ap.parse_args()

    if not MANIFEST.exists():
        raise SystemExit(f"missing {MANIFEST} -- run _gen_results_manifest.py first")
    entries = json.load(open(MANIFEST))["entries"]
    dest_root = RMIA if args.in_place else Path(args.dest)

    # index raw output files
    raw = []
    for d in args.inputs:
        raw += glob.glob(os.path.join(d, "**", "*__results.json"), recursive=True)
    raw = sorted(set(raw))

    placed, unmatched, skipped_combine = [], [], []
    produced_keys = set()
    for fp in raw:
        try:
            d = json.load(open(fp))
        except Exception as e:
            unmatched.append((fp, f"unreadable: {e}")); continue
        if "model_name" not in d:
            unmatched.append((fp, "no model_name")); continue
        key = content_key(d.get("model_name"), d.get("ref_model"), candidate_keys(d))
        ent = entries.get(key)
        if ent is None:
            unmatched.append((fp, "no manifest entry")); continue
        produced_keys.add(key)
        if ent["source"] == "combine":
            skipped_combine.append((fp, ent["dest"])); continue
        out = dest_root / ent["dest"]
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(fp, out)
        placed.append(ent["dest"])

    # manifest run_script entries with no matching raw output
    missing = [e["dest"] for k, e in entries.items()
               if e["source"] == "run_script" and k not in produced_keys]

    print(f"\nGather complete -> {dest_root}")
    print(f"  placed         : {len(placed)} files")
    print(f"  unmatched raw  : {len(unmatched)} (no manifest entry / unreadable)")
    print(f"  combine (skip) : {len(skipped_combine)} (static MoE-16B artifacts)")
    print(f"  not produced   : {len(missing)} run_script entries had no raw output")
    if unmatched:
        print("\n  -- unmatched raw files --")
        for fp, why in unmatched:
            print(f"     {why}: {os.path.relpath(fp, REPO)}")
    if missing:
        print("\n  -- manifest entries with no raw output (run those pairs to fill) --")
        for m in missing:
            print(f"     {m}")
    if skipped_combine:
        print("\n  -- combine artifacts (rebuild via their _build_from_collected_losses.py) --")
        for fp, dest in skipped_combine:
            print(f"     {dest}")


if __name__ == "__main__":
    main()
