"""
Shared helpers for the controlled-experiment classifier pipeline (paper §5.1).

- Parse the per-student JSON filenames in ReferenceMIAResults/.
- Normalize teacher-name spellings between filenames and JSON candidate keys.
- Convert reference-normalized loss arrays into a single "candidate score"
  (higher = more likely teacher).
- Extract the two features used by the paper: top score and gap to second.
"""
from __future__ import annotations

import os
import re
from typing import Dict, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Name handling
# ---------------------------------------------------------------------------

# Filenames use "Teacher=Nvidia-Llama-3.3-70B-Instruct" while the JSON keys
# inside Ref-Norm Loss use "Llama-3.3-70B-Instruct". Map filename form -> JSON form.
TEACHER_NAME_MAP: Dict[str, str] = {
    "GPT-OSS-120B": "GPT-OSS-120B",
    "Nvidia-Llama-3.3-70B-Instruct": "Llama-3.3-70B-Instruct",
    "Qwen-3-8B": "Qwen-3-8B",
}

# The four candidate teachers used for probing in the controlled experiments,
# in their JSON-key form. (Gemma-3-27B-it appears in the candidate pool but is
# never the true teacher for any controlled student.)
CANDIDATE_TEACHERS: List[str] = [
    "GPT-OSS-120B",
    "Llama-3.3-70B-Instruct",
    "Qwen-3-8B",
    "Gemma-3-27B-it",
]

# Probe prompt sets that each student is evaluated under.
PROMPT_MODES: List[str] = ["s1", "OMI"]


_FILENAME_RE = re.compile(
    r"^Student=(?P<student>.+?)"
    r"_Teacher=(?P<teacher>.+?)"
    r"_Data=(?P<data>.+?)\(1K\)"
    r"_Template=Chat\.json$"
)


def parse_filename(path: str) -> Dict[str, str]:
    """Pull (student, teacher, data) out of a ReferenceMIAResults filename."""
    base = os.path.basename(path)
    m = _FILENAME_RE.match(base)
    if not m:
        raise ValueError(f"Could not parse filename: {base}")
    return m.groupdict()


def candidate_key(teacher_canonical: str, prompt: str) -> str:
    """Build the dict key used inside results['Ref-Norm Loss']."""
    return f"{teacher_canonical} ({prompt}, Chat)"


# ---------------------------------------------------------------------------
# Scoring + features
# ---------------------------------------------------------------------------

def candidate_score(ref_norm_losses: List[float]) -> float:
    """
    Convert a list of reference-normalized losses into a single scalar where
    HIGHER means MORE LIKELY to be the true teacher.

    A more-negative ref-normalized loss means the student's loss dropped more
    relative to the reference model on that candidate's probe outputs, which
    is stronger evidence of distillation. Negating the mean gives a "score"
    that we can read in the natural direction (max = top candidate).
    """
    if not ref_norm_losses:
        raise ValueError("empty ref-norm loss list")
    return -sum(ref_norm_losses) / len(ref_norm_losses)


def features_from_candidate_scores(scores: List[float]) -> Tuple[float, float]:
    """
    The two features from §5.1 (Feature Extraction):
      (i)  top_score = score of the highest-scoring candidate (the most likely teacher)
      (ii) gap       = top_score - second-highest score
    """
    if len(scores) < 2:
        raise ValueError("need at least two candidates to compute a gap")
    s_sorted = sorted(scores, reverse=True)
    return s_sorted[0], s_sorted[0] - s_sorted[1]


# ---------------------------------------------------------------------------
# Real-world distilled models (paper §5.1, second bullet)
# ---------------------------------------------------------------------------
# All seven real-world distilled students share DeepSeek R1 as the true teacher
# (six R1-Distill checkpoints + s1.1-32B, which is trained on R1 traces).
REALWORLD_TRUE_TEACHER = "DeepSeek R1"

# JSON-key form of the candidates that probe these students. We drop QwQ-32B
# per the controlled experiment description (QwQ-32B-Preview is kept).
REALWORLD_CANDIDATES: List[str] = [
    "DeepSeek R1",
    "Qwen-3-235B-A22B-Thinking-2507",
    "QwQ-32B-Preview",
]
REALWORLD_EXCLUDED_CANDIDATES: List[str] = ["QwQ-32B"]

# Students excluded from summary accuracy metrics (predictions still computed
# and shown in per-student breakdowns, but omitted from overall/per-cell numbers).
REALWORLD_EXCLUDE_FROM_METRICS: List[str] = ["DeepSeek-R1-Distill-Qwen-1.5B"]

REALWORLD_PROBE_TYPE = "Trace + Response"

# Map: model_name as written in the JSON's "model_name" field -> true teacher
REALWORLD_STUDENT_TO_TEACHER: Dict[str, str] = {
    "DeepSeek-R1-Distill-Llama-70B": REALWORLD_TRUE_TEACHER,
    "DeepSeek-R1-Distill-Llama-8B":  REALWORLD_TRUE_TEACHER,
    "DeepSeek-R1-Distill-Qwen-14B":  REALWORLD_TRUE_TEACHER,
    "DeepSeek-R1-Distill-Qwen-1.5B": REALWORLD_TRUE_TEACHER,
    "DeepSeek-R1-Distill-Qwen-32B":  REALWORLD_TRUE_TEACHER,
    "DeepSeek-R1-Distill-Qwen-7B":   REALWORLD_TRUE_TEACHER,
    "s1.1-32B":                      REALWORLD_TRUE_TEACHER,
}


def realworld_candidate_key(teacher: str, prompt: str) -> str:
    """Build the dict key used inside results['Ref-Norm Loss'] for real-world JSONs."""
    return f"{teacher} ({prompt}, {REALWORLD_PROBE_TYPE})"


# ---------------------------------------------------------------------------
# 10-feature extraction (extended feature set)
# ---------------------------------------------------------------------------

ALL_10_FEATURES: List[str] = [
    "top_score",
    "second_score",
    "margin",
    "top_vote_frac",
    "vote_margin",
    "top_pairwise_winrate",
    "score_range",
    "paired_diff_skew",
    "paired_diff_p10",
    "paired_diff_p90",
]


def _skewness(x: np.ndarray) -> float:
    """Fisher-Pearson standardized moment skewness. Returns 0 if std==0."""
    s = x.std()
    if s == 0:
        return 0.0
    return float(((x - x.mean()) ** 3).mean() / s ** 3)


def features_10_from_candidate_data(
    raw_losses: Dict[str, List[float]],
    candidates: List[str],
) -> Dict[str, float]:
    """
    Extract 10 features from raw per-candidate ref-norm loss arrays.

    Dropped from the previous set (poor discriminating power on real-world data):
      - second_vote_frac: redundant with top_vote_frac / vote_margin.
      - score_dispersion: N-dependent std that collapses when N changes.
      - normalized_entropy: near-constant (~0.997) across all real-world rows.

    Added per-probe shape features (top-vs-second paired loss differences):
      - paired_diff_skew: skewness of (second_arr - top_arr) per probe.
          Genuine distillation → top wins many probes by a lot; some big losses
          create a long left tail → strongly negative skew.
          Spurious/architectural wins → differences are uniformly moderate → skew ≈ 0.
      - paired_diff_p10 / paired_diff_p90: 10th / 90th percentile of paired diffs.
          Capture the tails of per-probe competition without being as sensitive to
          individual outliers as min/max.

    N-invariance notes:
      - top_vote_frac / vote_margin: fraction of P probes, N-invariant.
      - top_pairwise_winrate: averaged over N-1 opponents, N-invariant.
      - paired_diff_*: computed only between top and second (2 candidates), N-invariant.
      - score_range: spread top-to-weakest, N-sensitive but still informative.
      - At N=2: top_vote_frac = top_pairwise_winrate = vote_margin+0.5,
        paired_diff features equal (second_arr - top_arr) stats.

    Args:
        raw_losses: mapping from candidate name to its list of raw ref-norm loss
                    values (lower = stronger distillation signal).
        candidates: which candidates to include in this row (2, 3, or 4 elements).
    """
    if len(candidates) < 2:
        raise ValueError("need at least 2 candidates")

    arrs = {c: np.array(raw_losses[c], dtype=float) for c in candidates}
    scores = {c: float(-arrs[c].mean()) for c in candidates}
    scores_arr = np.array([scores[c] for c in candidates])

    sorted_cands = sorted(candidates, key=lambda c: scores[c], reverse=True)
    top_cand = sorted_cands[0]
    second_cand = sorted_cands[1]
    top_arr = arrs[top_cand]
    second_arr = arrs[second_cand]
    N = len(candidates)

    # 1. top_score: mean signal strength of best candidate
    top_score = scores[top_cand]

    # 2. second_score: mean signal strength of runner-up
    second_score = scores[second_cand]

    # 3. margin: gap between top and second mean scores
    margin = top_score - second_score

    # 4. vote fractions: fraction of probes where top candidate is the argmin
    stacked = np.stack([arrs[c] for c in candidates], axis=0)  # (N, P)
    argmin_per_probe = np.argmin(stacked, axis=0)              # (P,)
    top_local_idx = candidates.index(top_cand)
    second_local_idx = candidates.index(second_cand)
    top_vote_frac = float(np.mean(argmin_per_probe == top_local_idx))
    second_vote_frac_tmp = float(np.mean(argmin_per_probe == second_local_idx))

    # 5. vote_margin: dominance of top over second in per-probe votes
    vote_margin = top_vote_frac - second_vote_frac_tmp

    # 6. top_pairwise_winrate: fraction of (probe, opponent) pairs where top wins;
    #    averaged over N-1 opponents to be N-invariant
    pairwise_wins = []
    for opp in candidates:
        if opp == top_cand:
            continue
        pairwise_wins.append(float(np.mean(top_arr < arrs[opp])))
    top_pairwise_winrate = float(np.mean(pairwise_wins))

    # 7. score_range: spread from top to weakest candidate
    score_range = top_score - float(scores_arr.min())

    # 8-10. per-probe shape features: top-vs-second paired loss differences.
    paired_diffs = second_arr - top_arr   # positive when top wins that probe
    paired_diff_skew = _skewness(paired_diffs)
    paired_diff_p10 = float(np.percentile(paired_diffs, 10))
    paired_diff_p90 = float(np.percentile(paired_diffs, 90))

    return {
        "top_score": top_score,
        "second_score": second_score,
        "margin": margin,
        "top_vote_frac": top_vote_frac,
        "vote_margin": vote_margin,
        "top_pairwise_winrate": top_pairwise_winrate,
        "score_range": score_range,
        "paired_diff_skew": paired_diff_skew,
        "paired_diff_p10": paired_diff_p10,
        "paired_diff_p90": paired_diff_p90,
    }