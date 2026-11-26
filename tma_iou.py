# -*- coding: utf-8 -*-
"""
3.6.1  TMA-IoU_C as a CONSTRUCTION-CONSISTENCY DIAGNOSTIC.

For a two-style task with styles A and B the direction-specific soft
probability IoU after compositing is

    IoU_soft^A = ( sum_i min(p_i^A, m_i^A) ) / ( sum_i max(p_i^A, m_i^A) )

with p_i^A the calibrated probability of style A at patch i and m_i^A the task
mask for style A on the same patch grid. Unless otherwise noted the reported
value is the symmetric average

    TMA-IoU_C = 1/2 ( IoU_soft^A + IoU_soft^B )

Per-direction values and uncertain-assigned / uncertain-excluded variants ship
here as sensitivity analyses.

TMA-IoU_C shares the task mask with the compositing construction for G/D
hard-mask conditions, so its value is construction-dependent. This module
computes it; it does not license any quality ordering.
"""

import numpy as np

from masks import mask_to_grid
from metrics.common import soft_iou, uncertain_grid


# ============================================================
# Direction-specific and symmetric TMA-IoU_C
# ============================================================
def tma_iou_directions(prob_a_grid: np.ndarray,
                       mask_a_grid: np.ndarray,
                       uncertain: np.ndarray = None):
    """
    Return (iou_soft_A, iou_soft_B).

    ``prob_a_grid`` holds the calibrated probability of style A on the patch
    grid; the probability of style B is its complement 1 - p^A. The style-B
    task mask is the complement of the style-A task mask.
    """
    p_a = np.asarray(prob_a_grid, dtype=np.float64)
    m_a = np.asarray(mask_a_grid, dtype=np.float64)

    p_b = 1.0 - p_a
    m_b = 1.0 - m_a

    valid = None
    if uncertain is not None:
        valid = ~np.asarray(uncertain, dtype=bool)

    return soft_iou(p_a, m_a, valid), soft_iou(p_b, m_b, valid)


def tma_iou_c(prob_a_grid: np.ndarray,
              mask_a_grid: np.ndarray,
              uncertain: np.ndarray = None,
              variant: str = "main"):
    """
    Symmetric average TMA-IoU_C plus the per-direction values.

    variant = "main"               : calibrated probabilities, all patches
    variant = "uncertain_excluded" : patches in the 0.4-0.6 band removed
    variant = "uncertain_assigned" : uncertain patches kept (same as main but
                                     documented separately in the outputs)

    Returns a dict with iou_soft_A, iou_soft_B, tma_iou_c, n_patches_used.
    """
    p_a = np.asarray(prob_a_grid, dtype=np.float64)
    m_a = np.asarray(mask_a_grid, dtype=np.float64)

    if variant == "main" or variant == "uncertain_assigned":
        valid = np.ones_like(p_a, dtype=bool)
    elif variant == "uncertain_excluded":
        valid = ~uncertain_grid(p_a)
    else:
        raise ValueError(f"Unknown variant: {variant}")

    iou_a = soft_iou(p_a, m_a, valid)
    iou_b = soft_iou(1.0 - p_a, 1.0 - m_a, valid)
    sym = 0.5 * (iou_a + iou_b)

    return dict(iou_soft_A=iou_a, iou_soft_B=iou_b, tma_iou_c=sym,
                variant=variant, n_patches_used=int(valid.sum()),
                n_patches_total=int(p_a.size))


def tma_iou_c_bundle(prob_a_grid: np.ndarray, mask_a_grid: np.ndarray):
    """All three reported variants in one call."""
    return {v: tma_iou_c(prob_a_grid, mask_a_grid, variant=v)
            for v in ("main", "uncertain_assigned", "uncertain_excluded")}


# ============================================================
# Convenience wrapper used by the pipeline
# ============================================================
def evaluate_composite(prob_a_pixel_or_grid, task: str, is_grid: bool = True):
    """
    Compute TMA-IoU_C for one composite.

    ``prob_a_pixel_or_grid`` is the calibrated style-A probability map produced
    by the classifier, either already on the patch grid (is_grid=True) or at
    pixel resolution (is_grid=False); in the latter case it is averaged down to
    the grid.
    """
    from config import MASK_TO_GRID_PRIMARY
    from masks import load_mask

    if is_grid:
        p_grid = np.asarray(prob_a_pixel_or_grid, dtype=np.float64)
    else:
        p_grid = _pixel_to_grid(prob_a_pixel_or_grid)

    mask_a = load_mask(task)
    m_grid = mask_to_grid(mask_a, rule=MASK_TO_GRID_PRIMARY)
    return tma_iou_c_bundle(p_grid, m_grid)


def _pixel_to_grid(prob_pixel: np.ndarray) -> np.ndarray:
    """Average a pixel-resolution probability map onto the patch grid."""
    from config import CLASSIFIER_CONFIG
    from masks import grid_shape

    patch = CLASSIFIER_CONFIG["patch_size"]
    stride = CLASSIFIER_CONFIG["stride"]
    n = prob_pixel.shape[0]
    k, _ = grid_shape(n, patch, stride)
    out = np.zeros((k, k), dtype=np.float64)
    i = 0
    for y in range(0, n - patch + 1, stride):
        for x in range(0, n - patch + 1, stride):
            out[i // k, i % k] = prob_pixel[y:y + patch, x:x + patch].mean()
            i += 1
    return out
