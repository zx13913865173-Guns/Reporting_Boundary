# -*- coding: utf-8 -*-
"""
Figures 1-5 of the manuscript.

    Fig. 1  construct separation: target mask, compositing mask, generation
            masks and classifier evaluation domain are four different objects
    Fig. 2  the eight synthetic boundary tasks
    Fig. 3  six-condition profiles of TMA-IoU_C and the mask-defined
            descriptive metrics, with lower-is-better metrics flagged
    Fig. 4  CRSC interior with the direction-specific classifier noise floor
            and the boundary-band leakage reported separately
    Fig. 5  the two non-task-mask references (mask-free preference, PB-IoU)

Data source: paper_values.py by default (so that the figures can be rebuilt
without a GPU); pass ``from_csv=True`` to read the tables produced by the
pipeline instead.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from config import (
    FIG_DIR, FIG_SIZE, DPI, CONDITION_ORDER,
    BLUE_DARK, BLUE_MID, BLUE_LIGHT, BLUE_PALE, BLUE_GRAY,
)
from masks import build_mask, overlap_generation_masks, mask_to_grid, grid_shape
import paper_values as PV

_METRIC_COLORS = {
    "TMA_IoU_C": BLUE_DARK,
    "PRSC_A": BLUE_MID,
    "CRSC_A_interior": BLUE_GRAY,
    "CMMD_A": BLUE_LIGHT,
    "masked_purity_A": BLUE_PALE,
    "PB_IoU": BLUE_MID,
}


def _style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


def _save(fig, name: str, dpi: int = None):
    out = FIG_DIR / name
    fig.savefig(out, dpi=dpi or DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"[figures] {name}")
    return out


# ============================================================
# Fig 1 - construct separation
# ============================================================
def fig1(dpi: int = None):
    _style()
    task = "horizontal_split"
    m_a = build_mask(task)
    m_a_gen, m_b_gen, alpha = overlap_generation_masks(task, 16.0)
    k, _ = grid_shape()
    grid = mask_to_grid(m_a)

    fig, axes = plt.subplots(1, 4, figsize=FIG_SIZE)
    panels = [
        ("Target mask $M_A$", m_a),
        ("Compositing mask (G/D = $M_A$)", m_a),
        ("Generation masks (H/I, $\\delta=16$)", m_a_gen),
    ]
    for ax, (title, img) in zip(axes[:3], panels):
        ax.imshow(img, cmap="Blues", vmin=0, vmax=1)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    ax = axes[3]
    ax.imshow(np.kron(grid, np.ones((8, 8))), cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"Classifier evaluation domain\n({k} x {k} patch grid)", fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("Four different mask objects: task mask, compositing mask, "
                 "generation masks, classifier evaluation domain", fontsize=12)
    return _save(fig, "Fig1_construct_separation.png", dpi)


# ============================================================
# Fig 2 - the eight tasks
# ============================================================
def fig2(dpi: int = None):
    from config import TASKS, TASK_LABELS

    _style()
    fig, axes = plt.subplots(2, 4, figsize=FIG_SIZE)
    for ax, task in zip(axes.ravel(), TASKS):
        m = build_mask(task)
        rgb = np.zeros((*m.shape, 3), dtype=np.float32)
        rgb[m] = (0.79, 0.87, 0.94)        # region A - ink-wash tone
        rgb[~m] = (0.04, 0.24, 0.42)       # region B - cyberpunk tone
        ax.imshow(rgb)
        ax.set_title(TASK_LABELS[task], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Eight synthetic boundary tasks: region A (ink-wash) vs. "
                 "region B (cyberpunk)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "Fig2_tasks.png", dpi)


# ============================================================
# Fig 3 - six-condition profiles
# ============================================================
def fig3(dpi: int = None):
    _style()
    metrics = {
        "TMA_IoU_C": PV.TMA_IOU_C,
        "PRSC_A": PV.PRSC_A,
        "CRSC_A_interior": PV.CRSC_A_INTERIOR,
        "CMMD_A": PV.CMMD_A,
        "masked_purity_A": PV.MASKED_PURITY_A,
        "PB_IoU": PV.PB_IOU,
    }
    lower_better = {"CRSC_A_interior", "CMMD_A"}

    fig, axes = plt.subplots(2, 3, figsize=FIG_SIZE)
    x = np.arange(len(CONDITION_ORDER))
    for ax, (name, values) in zip(axes.ravel(), metrics.items()):
        y = [values[c] for c in CONDITION_ORDER]
        ax.bar(x, y, color=_METRIC_COLORS[name], edgecolor="#333333", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITION_ORDER, fontsize=9)
        ax.set_ylim(0, 1.0 if max(y) <= 1.0 else max(y) * 1.2)
        flag = "  (lower is better)" if name in lower_better else ""
        ax.set_title(f"{name}{flag}", fontsize=10)
        for xi, yi in zip(x, y):
            ax.text(xi, yi + 0.01, f"{yi:.2f}", ha="center", fontsize=8)
    fig.suptitle("Six-condition descriptive profiles "
                 "(construction-defined conditions only)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "Fig3_condition_profiles.png", dpi)


# ============================================================
# Fig 4 - CRSC interior and the classifier noise floor
# ============================================================
def fig4(dpi: int = None):
    from config import CLASSIFIER_FPR

    _style()
    labels = PV.CRSC_EXTENDED_ORDER
    x = np.arange(len(labels))
    w = 0.38

    fig, ax = plt.subplots(figsize=FIG_SIZE)
    ax.bar(x - w / 2, PV.CRSC_A_EXTENDED, w, color=BLUE_DARK,
           edgecolor="#333333", linewidth=0.6, label="CRSC$_A$ interior")
    ax.bar(x + w / 2, PV.CRSC_B_EXTENDED, w, color=BLUE_LIGHT,
           edgecolor="#333333", linewidth=0.6, label="CRSC$_B$ interior")
    ax.axhline(CLASSIFIER_FPR["FPR_A_on_B"], color=BLUE_GRAY, linestyle="--",
               linewidth=1.2, label="FPR$_A$ on B = 0.03")
    ax.axhline(CLASSIFIER_FPR["FPR_B_on_A"], color=BLUE_MID, linestyle=":",
               linewidth=1.2, label="FPR$_B$ on A = 0.05")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("CRSC interior")
    ax.set_ylim(0, 0.5)
    ax.set_title("CRSC interior with the direction-specific classifier noise floor\n"
                 "values near or below the floor are not distinguishable from "
                 "classifier error", fontsize=11)
    ax.legend(loc="upper right", ncol=2, fontsize=9)
    return _save(fig, "Fig4_crsc_noise_floor.png", dpi)


# ============================================================
# Fig 5 - non-task-mask references
# ============================================================
def fig5(dpi: int = None):
    _style()
    fig, axes = plt.subplots(1, 2, figsize=FIG_SIZE)

    ax = axes[0]
    names = ["mask-free", "overall", "forced compliance"]
    vals = [PV.MASK_FREE_PREFERENCE["proportion"],
            PV.OVERALL_PREFERENCE["proportion"],
            PV.FORCED_COMPLIANCE_PREFERENCE["proportion"]]
    x = np.arange(len(names))
    ax.bar(x, vals, 0.5, color=BLUE_MID, edgecolor="#333333", linewidth=0.6)
    ax.axhline(0.5, color=BLUE_GRAY, linestyle="--", linewidth=1.0)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Proportion preferring G0.5")
    ax.set_title("Mask-free reference: descriptive preference proportions\n"
                 "(304 paired comparisons; no p-values)", fontsize=10)
    for xi, vi in zip(x, vals):
        ax.text(xi, vi + 0.02, f"{vi:.3f}", ha="center", fontsize=9)

    ax = axes[1]
    y = [PV.PB_IOU[c] for c in CONDITION_ORDER]
    x = np.arange(len(CONDITION_ORDER))
    ax.bar(x, y, 0.5, color=BLUE_DARK, edgecolor="#333333", linewidth=0.6)
    ax.set_xticks(x); ax.set_xticklabels(CONDITION_ORDER, fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("PB-IoU")
    ax.set_title("PB-IoU against the consensus perceived boundary\n"
                 "(40 images x 2 annotators = 80 annotations)", fontsize=10)
    for xi, yi in zip(x, y):
        ax.text(xi, yi + 0.02, f"{yi:.2f}", ha="center", fontsize=9)

    fig.suptitle("Non-task-mask references (references, not ground truth)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return _save(fig, "Fig5_non_task_mask_references.png", dpi)


# ============================================================
# Driver
# ============================================================
def generate_all(dpi: int = None):
    return [fig1(dpi), fig2(dpi), fig3(dpi), fig4(dpi), fig5(dpi)]


if __name__ == "__main__":
    generate_all()
