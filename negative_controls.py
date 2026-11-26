# -*- coding: utf-8 -*-
"""
3.6.5  Core negative controls.

Three construction-sensitivity tests. In all three the GENERATED REGIONS ARE
FIXED; only the construction around them is perturbed.

    mask_swap          incorrect compositing mask; evaluated against the
                       correct task mask
    label_swap         swapped style labels; original task mask
    compositing_mask_perturbation
                       +/-8 / +/-16 px boundary-normal translation of the
                       compositing mask, plus 8 px dilation/erosion

Design: 3 controls x 4 G/D conditions x 10 prompts = 120 composites, plus
3 controls x 2 H/I conditions x 10 prompts = 60 composites = 180 composites in
total (the core negative controls).

Reported quantity: mean DECREASE in TMA-IoU_C relative to the condition's
unperturbed composite. The controls show that TMA-IoU_C responds to construction
perturbations; they do NOT by themselves establish that TMA-IoU_C is invalid as
a quality metric.
"""

import numpy as np
import pandas as pd
from PIL import Image

from config import (
    NEG_CONTROLS, NEG_CONTROL_PROMPTS, NEG_CONTROL_GD_CONDITIONS,
    NEG_CONTROL_HI_CONDITIONS, PERTURB_TRANSLATIONS_PX, PERTURB_MORPH_PX,
    TASKS, SEEDS,
)
from masks import (
    load_mask, dilate_px, erode_px, signed_distance, mask_to_grid,
)
from metrics.tma_iou import tma_iou_directions


# ============================================================
# Perturbation operators (operate on the COMPOSITING mask only)
# ============================================================
def translate_boundary(mask_a: np.ndarray, task: str, shift_px: int) -> np.ndarray:
    """
    Shift the compositing boundary along its local normal by ``shift_px``.

    Positive values move the boundary INTO region B (region A grows).
    """
    d = signed_distance(task)
    return d > -float(shift_px)


def morph_mask(mask_a: np.ndarray, task: str, morph_px: int) -> np.ndarray:
    """8 px dilation (+morph_px) or erosion (-morph_px) of the region-A mask."""
    return dilate_px(mask_a, morph_px) if morph_px > 0 else erode_px(mask_a, -morph_px)


def swap_masks(mask_a: np.ndarray) -> np.ndarray:
    """Incorrect compositing mask: region A and region B are exchanged."""
    return ~mask_a


# ============================================================
# Compositing under a perturbed construction
# ============================================================
def composite_with_mask(out_a: Image.Image, out_b: Image.Image,
                        mask_a: np.ndarray) -> Image.Image:
    """Composite two FIXED region outputs through an arbitrary binary mask."""
    a = np.array(out_a.convert("RGB"))
    b = np.array(out_b.convert("RGB"))
    m3 = np.repeat(mask_a[:, :, None], 3, axis=2)
    return Image.fromarray(np.where(m3, a, b).astype(np.uint8))


def _tma_iou_c_of_image(image: Image.Image, task: str, prob_fn) -> float:
    """TMA-IoU_C of a composited image (symmetric average)."""
    prob_a_grid = prob_fn(image, task)
    m_grid = mask_to_grid(load_mask(task))
    iou_a, iou_b = tma_iou_directions(prob_a_grid, m_grid)
    return 0.5 * (iou_a + iou_b)


# ============================================================
# Individual controls
# ============================================================
def run_mask_swap(task: str, out_a: Image.Image, out_b: Image.Image,
                  prob_fn) -> float:
    """Decrease in TMA-IoU_C when the compositing mask is swapped."""
    from generation import compose_hard

    base = _tma_iou_c_of_image(compose_hard(out_a, out_b, task), task, prob_fn)
    swapped = composite_with_mask(out_a, out_b, swap_masks(load_mask(task)))
    pert = _tma_iou_c_of_image(swapped, task, prob_fn)
    return float(base - pert)


def run_label_swap(task: str, out_a: Image.Image, out_b: Image.Image,
                   prob_fn) -> float:
    """Decrease in TMA-IoU_C when the two region outputs' style labels swap."""
    from generation import compose_hard

    base = _tma_iou_c_of_image(compose_hard(out_a, out_b, task), task, prob_fn)
    # labels swapped: region B output is composited into region A and vice versa
    swapped = composite_with_mask(out_b, out_a, load_mask(task))
    pert = _tma_iou_c_of_image(swapped, task, prob_fn)
    return float(base - pert)


def run_compositing_mask_perturbation(task: str, out_a: Image.Image,
                                      out_b: Image.Image, prob_fn) -> float:
    """
    Decrease in TMA-IoU_C under the strongest compositing-mask perturbation
    (mean over the +/-8 / +/-16 px translations and the 8 px morphologies).
    """
    from generation import compose_hard

    base = _tma_iou_c_of_image(compose_hard(out_a, out_b, task), task, prob_fn)
    m_a = load_mask(task)
    deltas = []

    for shift in PERTURB_TRANSLATIONS_PX:
        pert_mask = translate_boundary(m_a, task, shift)
        img = composite_with_mask(out_a, out_b, pert_mask)
        deltas.append(base - _tma_iou_c_of_image(img, task, prob_fn))

    for morph in (PERTURB_MORPH_PX, -PERTURB_MORPH_PX):
        pert_mask = morph_mask(m_a, task, morph)
        img = composite_with_mask(out_a, out_b, pert_mask)
        deltas.append(base - _tma_iou_c_of_image(img, task, prob_fn))

    return float(np.mean(deltas))


CONTROL_FUNCS = {
    "mask_swap": run_mask_swap,
    "label_swap": run_label_swap,
    "compositing_mask_perturbation": run_compositing_mask_perturbation,
}


# ============================================================
# Core-control sweep
# ============================================================
def run_core_controls(unit_provider, prob_fn, tasks=None,
                      conditions=None, prompts=None):
    """
    Run the three controls over the core-control design.

    ``unit_provider(task, condition, prompt_index)`` must return
    (out_a, out_b) with the generated regions FIXED across controls.
    ``prob_fn(image, task)`` returns the calibrated style-A probability grid.

    Returns a tidy DataFrame: task, condition, control, prompt, decrease.
    """
    tasks = tasks or [t for t in TASKS if t != "frame_border"]
    conditions = conditions or (NEG_CONTROL_GD_CONDITIONS + NEG_CONTROL_HI_CONDITIONS)
    prompts = prompts or range(NEG_CONTROL_PROMPTS)

    rows = []
    for task in tasks:
        for condition in conditions:
            for pi in prompts:
                out_a, out_b = unit_provider(task, condition, pi)
                for control in NEG_CONTROLS:
                    dec = CONTROL_FUNCS[control](task, out_a, out_b, prob_fn)
                    rows.append(dict(task=task, condition=condition,
                                     control=control, prompt=pi,
                                     decrease=dec))

    df = pd.DataFrame(rows)
    df.to_csv(_out_path(), index=False)
    return df


def summarise_controls(df: pd.DataFrame) -> pd.DataFrame:
    """Mean TMA-IoU_C decrease per condition x control (manuscript Table 1)."""
    return (df.groupby(["condition", "control"])["decrease"]
              .mean().unstack("control")[[*NEG_CONTROLS]])


def control_design_counts() -> dict:
    """The 120 + 60 = 180 composite design."""
    return {
        "GD_conditions": len(NEG_CONTROL_GD_CONDITIONS),
        "HI_conditions": len(NEG_CONTROL_HI_CONDITIONS),
        "prompts_per_condition": NEG_CONTROL_PROMPTS,
        "GD_composites": len(NEG_CONTROL_GD_CONDITIONS) * len(NEG_CONTROLS) * NEG_CONTROL_PROMPTS,
        "HI_composites": len(NEG_CONTROL_HI_CONDITIONS) * len(NEG_CONTROLS) * NEG_CONTROL_PROMPTS,
        "total_composites": (len(NEG_CONTROL_GD_CONDITIONS) + len(NEG_CONTROL_HI_CONDITIONS))
                            * len(NEG_CONTROLS) * NEG_CONTROL_PROMPTS,
    }


def _out_path():
    from config import METRICS_DIR
    return METRICS_DIR / "negative_controls.csv"


if __name__ == "__main__":
    print("core negative-control design:", control_design_counts())
