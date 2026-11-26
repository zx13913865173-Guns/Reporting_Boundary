# -*- coding: utf-8 -*-
"""
3.6.6 / 3.6.7 / 3.7  Human data: split preference, 1-5 ratings and PB-IoU
annotations.

Split preference (mask-free reference for the G0.5 vs. D0.5 comparison)
    304 paired comparisons, each with three separate binary preference
    responses  ->  912 binary preference responses.

Per-question wording (Section 3.6.7):
    mask_free         no mask shown: "Which image better separates the two
                      styles overall, regardless of where the boundary falls?"
    overall           "Which image do you prefer overall?"
    forced_compliance "Which image follows the indicated region boundary more
                      closely, ignoring naturalness and seam quality?"

Ratings
    Two separate 1-5 scales (naturalness, boundary artifact) on the 10-seed
    subset. Mask-only was not rated for naturalness.

PB-IoU annotations
    40 images x 2 annotators = 80 annotations; only the boundary stroke is
    annotated and the two annotators' boundary-derived region masks are
    intersected.

Preference proportions are DESCRIPTIVE only: no p-values and no population-level
claims. The three questions describe the same assigned image pairs under
different instructions.
"""

import csv
import json

import numpy as np

from config import (
    PREFERENCE_PAIRS, PREFERENCE_QUESTIONS, PREFERENCE_COMPARISON,
    PB_IOU_IMAGES, PB_IOU_ANNOTATORS, RATING_SCALES, RATING_MIN, RATING_MAX,
    ANNOT_DIR,
)
from metrics.non_task_mask import (
    preference_proportion, split_preference_summary, consensus_partition,
    pb_iou, inter_annotator_stats,
)


# ============================================================
# Paired-comparison schedule
# ============================================================
def build_paired_schedule(n_pairs: int = PREFERENCE_PAIRS,
                          pair=PREFERENCE_COMPARISON,
                          seed: int = 42):
    """
    Build the assigned comparison list for the mask-free reference.

    Each row is one image pair (side-by-side view, no mask overlay). Which
    condition is shown on the left/right is randomised per pair.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_pairs):
        left_first = bool(rng.integers(0, 2))
        rows.append(dict(pair_id=i + 1,
                         left=pair[0] if left_first else pair[1],
                         right=pair[1] if left_first else pair[0],
                         condition=pair[0], comparator=pair[1]))
    return rows


def write_paired_schedule(rows=None, path=None):
    rows = rows or build_paired_schedule()
    path = path or (ANNOT_DIR / "paired_comparisons.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"[human_study] paired schedule ({len(rows)} pairs) -> {path}")
    return path


def load_responses(path):
    """
    Read the judgment file. Expected columns:
        pair_id, question, response   (response in {0, 1})
    """
    out = {q: [] for q in PREFERENCE_QUESTIONS}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            q = row["question"]
            if q in out:
                out[q].append(float(row["response"]))
    return out


def preference_report(responses_by_question: dict, path=None) -> dict:
    """
    Summarise the three questions. Proportions are descriptive only; no
    sampling model, no p-value and no population-level claim is attached.
    """
    summary = split_preference_summary(responses_by_question)
    total = int(sum(v["n"] for v in summary.values() if v))
    report = {
        "design": {
            "paired_comparisons": PREFERENCE_PAIRS,
            "questions": PREFERENCE_QUESTIONS,
            "binary_responses": total,
            "manuscript_expected_responses": PREFERENCE_PAIRS * len(PREFERENCE_QUESTIONS),
            "reference_condition": PREFERENCE_COMPARISON[0],
        },
        "per_question": summary,
        "interpretation": "descriptive proportions only; no p-values and no "
                          "population-level claims",
    }
    path = path or (ANNOT_DIR / "preference_report.json")
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[human_study] preference report -> {path}")
    return report


def preference_proportions(responses_by_question: dict) -> dict:
    """Convenience: {question: proportion}."""
    return {q: preference_proportion(r)
            for q, r in responses_by_question.items()
            if r is not None and len(list(r)) > 0}


# ============================================================
# 1-5 ratings
# ============================================================
def summarise_ratings(ratings: dict) -> dict:
    """
    Summarise 1-5 scale ratings.

    ``ratings``: {scale_name: {condition: [values]}} with scale_name in
    RATING_SCALES. The two scales are reported separately and are never
    combined into a single "quality" score, because the study measures
    distinct constructs.
    """
    out = {}
    for scale in RATING_SCALES:
        per_cond = ratings.get(scale, {})
        out[scale] = {
            c: {
                "n": len(v),
                "mean": float(np.mean(v)) if v else float("nan"),
                "median": float(np.median(v)) if v else float("nan"),
            }
            for c, v in per_cond.items()
        }
    return out


def check_rating_domain(values) -> bool:
    """Ratings must sit on the declared 1-5 integer scale."""
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return True
    return bool(np.all(np.isfinite(v))
                and np.all(v >= RATING_MIN) and np.all(v <= RATING_MAX))


# ============================================================
# PB-IoU annotations
# ============================================================
def pb_iou_schedule(n_images: int = PB_IOU_IMAGES):
    """40 images x 2 annotators = 80 boundary annotations."""
    return [dict(image_id=i + 1, annotator=a)
            for i in range(n_images)
            for a in range(1, PB_IOU_ANNOTATORS + 1)]


def pb_iou_report(annotations: dict, classifier_regions: dict) -> dict:
    """
    PB-IoU summary.

    ``annotations``: {image_id: {annotator: boundary_bool_array}}
    ``classifier_regions``: {image_id: classifier-recovered region-A bool array}

    Only the two strokes are annotated; the region partition used as reference
    is the INTERSECTION of the two annotators' boundary-derived region masks.
    """
    per_image, f1s, mads = {}, [], []
    for img_id, by_ann in annotations.items():
        if len(by_ann) < 2:
            continue
        a, b = (by_ann[k] for k in sorted(by_ann)[:2])
        stats = inter_annotator_stats(a, b)
        f1s.append(stats["boundary_f1"])
        mads.append(stats["mean_absolute_boundary_distance_px"])

        consensus = consensus_partition(a, b)
        cls_region = classifier_regions.get(img_id)
        if cls_region is not None:
            per_image[img_id] = pb_iou(cls_region, consensus)

    return {
        "n_images": len(per_image),
        "n_annotations": sum(len(v) for v in annotations.values()),
        "pb_iou_per_image": per_image,
        "pb_iou_mean": float(np.mean(list(per_image.values()))) if per_image else float("nan"),
        "inter_annotator_boundary_f1_mean": float(np.nanmean(f1s)) if f1s else float("nan"),
        "inter_annotator_mad_px_mean": float(np.nanmean(mads)) if mads else float("nan"),
        "notes": "PB-IoU uses classifier-recovered style regions; it is a "
                 "non-task-mask reference, not ground truth.",
    }


if __name__ == "__main__":
    sch = build_paired_schedule()
    print(f"paired comparisons: {len(sch)}; "
          f"three questions each = {len(sch) * len(PREFERENCE_QUESTIONS)} responses")
    print(f"PB-IoU schedule: {len(pb_iou_schedule())} annotations")
