# -*- coding: utf-8 -*-
"""
3.6.6 / 3.7  Non-task-mask references.

Two references are used whose evaluation domain is NOT the task mask:

    * mask-free human preference for the G0.5 vs. D0.5 comparison, asked
      WITHOUT showing a mask ("Which image better separates the two styles
      overall, regardless of where the boundary falls?");
    * PB-IoU, computed between classifier-recovered style regions and an
      independently perceived boundary-derived region partition obtained by
      intersecting the two annotators' boundary-derived region masks.

PB-IoU is NOT independent of the classifier-recovered style regions. It is a
non-task-mask reference, not ground truth and not an independent anchor.
"""

import numpy as np

from config import PREFERENCE_QUESTIONS
from metrics.common import region_iou, boundary_f1, mean_absolute_boundary_distance


# ============================================================
# PB-IoU
# ============================================================
def consensus_partition(ann_a_boundary: np.ndarray,
                        ann_b_boundary: np.ndarray,
                        side: str = "top_left") -> np.ndarray:
    """
    Consensus boundary-derived region partition.

    The two annotators' boundary-derived region masks are INTERSECTED, i.e. a
    pixel belongs to region A only if both annotators place it in region A.
    Only the two boundaries are needed to build the set of boundary-derived
    region masks; ``side`` selects which side of the boundary is region A.
    """
    a = np.asarray(ann_a_boundary, dtype=bool)
    b = np.asarray(ann_b_boundary, dtype=bool)

    reg_a = _side_of_boundary(a, side)
    reg_b = _side_of_boundary(b, side)
    return reg_a & reg_b


def _side_of_boundary(boundary: np.ndarray, side: str) -> np.ndarray:
    """Region mask implied by a boundary: rows above it are region A."""
    h, w = boundary.shape
    region = np.zeros((h, w), dtype=bool)
    for x in range(w):
        col = np.where(boundary[:, x])[0]
        if col.size == 0:
            continue
        y0 = int(col.min())
        if side == "top_left":
            region[:y0, x] = True
        else:
            region[y0 + 1:, x] = True
    return region


def pb_iou(classifier_region: np.ndarray,
           consensus_region: np.ndarray) -> float:
    """
    Perceived-Boundary IoU: pixel-level IoU between the classifier-recovered
    style regions and the independently perceived boundary-derived region
    partition.
    """
    return region_iou(classifier_region, consensus_region)


def inter_annotator_stats(ann_a_boundary: np.ndarray,
                          ann_b_boundary: np.ndarray,
                          tol_px: int = 4):
    """
    Inter-annotator agreement for the boundary annotations: boundary F1 and the
    mean absolute boundary distance (manuscript: F1 = 0.86, MAD = 4.1 px).
    """
    return {
        "boundary_f1": boundary_f1(ann_a_boundary, ann_b_boundary, tol_px),
        "mean_absolute_boundary_distance_px":
            mean_absolute_boundary_distance(ann_a_boundary, ann_b_boundary),
    }


# ============================================================
# Split preference
# ============================================================
def preference_proportion(responses) -> float:
    """
    Proportion of judgments selecting the reference condition.

    ``responses``: iterable of 1/0 (or True/False). Condition-level preference
    proportions are DESCRIPTIVE summaries of the observed judgment set; no
    sampling model is assumed and no p-value is derived.
    """
    r = np.asarray(list(responses), dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return float("nan")
    return float(r.mean())


def split_preference_summary(responses_by_question: dict):
    """
    Summarise the three separate questions of Section 3.6.7.

    Expected keys (see PREFERENCE_QUESTIONS):
        mask_free          - no mask shown
        overall            - "Which image do you prefer overall?"
        forced_compliance  - mask-based, ignores naturalness and seam quality
    """
    out = {}
    for q in PREFERENCE_QUESTIONS:
        r = responses_by_question.get(q)
        out[q] = None if r is None else {
            "n": int(np.isfinite(np.asarray(list(r), dtype=float)).sum()),
            "proportion": preference_proportion(r),
        }
    return out


__all__ = [
    "consensus_partition", "pb_iou", "inter_annotator_stats",
    "preference_proportion", "split_preference_summary",
]
