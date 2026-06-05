"""
Build the controlled-experiment classifier dataset with 10 features (paper §5.1).

Identical logic to build_dataset.py, but each row carries 10 features extracted
from the full per-candidate ref-norm loss distributions instead of just 2.

Output CSV: controlled_dataset_10f.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    ALL_10_FEATURES,
    CANDIDATE_TEACHERS,
    PROMPT_MODES,
    TEACHER_NAME_MAP,
    candidate_key,
    features_10_from_candidate_data,
    parse_filename,
)

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT  = os.path.dirname(_SCRIPT_DIR)
DEFAULT_INPUT_DIR = os.environ.get(
    "REF_MIA_RESULTS_DIR",
    os.path.join(_REPO_ROOT, "data", "reference_mia", "controlled"),
)
DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "data", "classifier", "controlled_dataset_10f.csv")

EXPECTED_N_STUDENTS = 19


def build(input_dir: str, output_path: str) -> None:
    pattern = os.path.join(
        input_dir, "Student=*_Teacher=*_Data=*(1K)_Template=Chat.json"
    )
    files = sorted(glob.glob(pattern))
    print(f"[build_dataset_10f] found {len(files)} json files in {input_dir}")
    if len(files) != EXPECTED_N_STUDENTS:
        print(
            f"[build_dataset_10f] WARNING: expected {EXPECTED_N_STUDENTS} controlled "
            f"students, got {len(files)} (pipeline will still run)"
        )

    rows = []
    for fpath in files:
        meta = parse_filename(fpath)
        student = meta["student"]
        teacher_raw = meta["teacher"]
        if teacher_raw not in TEACHER_NAME_MAP:
            raise ValueError(
                f"Unknown teacher '{teacher_raw}' in {os.path.basename(fpath)}. "
                f"Add it to TEACHER_NAME_MAP in utils.py."
            )
        true_teacher = TEACHER_NAME_MAP[teacher_raw]
        distill_data = meta["data"]

        with open(fpath) as f:
            blob = json.load(f)
        ref_norm = blob["results"]["Ref-Norm Loss"]

        # Validate all keys exist before building rows.
        for cand in CANDIDATE_TEACHERS:
            for prompt in PROMPT_MODES:
                key = candidate_key(cand, prompt)
                if key not in ref_norm:
                    raise KeyError(
                        f"missing candidate key '{key}' in {os.path.basename(fpath)}"
                    )

        student_id = f"{student}__teacher={true_teacher}__data={distill_data}"

        for prompt in PROMPT_MODES:
            full_set = list(CANDIDATE_TEACHERS)
            removed_set = [t for t in CANDIDATE_TEACHERS if t != true_teacher]

            for label, cset, tag in (
                (1, full_set, "with_true"),
                (0, removed_set, "without_true"),
            ):
                raw_for_prompt = {
                    cand: ref_norm[candidate_key(cand, prompt)] for cand in cset
                }
                feats = features_10_from_candidate_data(raw_for_prompt, cset)

                row = {
                    "student_id": student_id,
                    "student": student,
                    "true_teacher_family": true_teacher,
                    "distill_data": distill_data,
                    "probe_prompt": prompt,
                    "candidate_setting": tag,
                    "n_candidates": len(cset),
                }
                row.update(feats)
                row["label"] = label
                rows.append(row)

    print(f"[build_dataset_10f] built {len(rows)} data points")
    pos = sum(r["label"] for r in rows)
    print(f"[build_dataset_10f] positives (y=1): {pos}, negatives (y=0): {len(rows) - pos}")

    families = {}
    for r in rows:
        families.setdefault(r["true_teacher_family"], 0)
        families[r["true_teacher_family"]] += 1
    print("[build_dataset_10f] rows per teacher family:")
    for fam, n in sorted(families.items()):
        print(f"    {fam}: {n}")

    meta_cols = [
        "student_id", "student", "true_teacher_family", "distill_data",
        "probe_prompt", "candidate_setting", "n_candidates",
    ]
    fieldnames = meta_cols + ALL_10_FEATURES + ["label"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[build_dataset_10f] wrote {output_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    p.add_argument("--output", default=DEFAULT_OUTPUT)
    args = p.parse_args()
    build(args.input_dir, args.output)


if __name__ == "__main__":
    main()
