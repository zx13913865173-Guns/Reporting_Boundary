# -*- coding: utf-8 -*-
"""
3.6.2 / 3.6.3 / 3.6.4  Mask-defined descriptive metrics.

Every metric in this module has its evaluation domain defined by the task mask.
None of them is treated as a non-task-mask quality anchor.

    PRSC_A           proportion of non-uncertain patches classified as style A
                     inside erode(M_A, 32 px), computed on the PRE-COMPOSITE
                     region-A output (uncertain patches EXCLUDED).
    CRSC_A interior  proportion of patches in erode(M_B, 32 px) assigned to
                     style A by the calibrated classifier at the 0.5 threshold
                     (uncertain patches RETAINED).

                     PRSC excludes uncertain patches while CRSC retains them;
                     the asymmetry is a stated limitation.

    Boundary-band leakage  computed separately for patches whose centres lie
                     within BOUNDARY_BAND_PX of the boundary; NOT part of the
                     main CRSC.
    Masked purity / CMMD   CLIP-embedding diagnostics of style-level distance
                     to a reference distribution.
"""

import numpy as np

from config import (
    PRSC_ERODE_PX, CRSC_ERODE_PX, CRSC_THRESHOLD, BOUNDARY_BAND_PX,
    CLASSIFIER_CONFIG, CLASSIFIER_FPR,
)
from masks import (
    load_mask, mask_to_grid, erode_px, signed_distance, grid_shape,
)
from metrics.common import (
    uncertain_grid, style_purity, cmmd, masked_region_image,
)
from metrics.tma_iou import _pixel_to_grid


# ============================================================
# Patch-grid selection helpers
# ============================================================
def _grid_center_select(region_pixel: np.ndarray) -> np.ndarray:
    """Patches whose CENTRE lies inside a pixel-level region."""
    return mask_to_grid(region_pixel, rule="patch_center") > 0.5


def _band_select(task: str, band_px: int = BOUNDARY_BAND_PX) -> np.ndarray:
    """Patches whose CENTRES lie within ``band_px`` of the boundary."""
    d = np.abs(signed_distance(task))
    return _grid_center_select(d < band_px)


# ============================================================
# 3.6.2  PRSC
# ============================================================
def prsc(prob_a_pixel: np.ndarray, task: str, style: str = "A",
         erode: int = PRSC_ERODE_PX,
         uncertain_excluded: bool = True) -> float:
    """
    PRSC for one pre-composite region output.

    ``prob_a_pixel`` is the calibrated style-A probability map of the
    PRE-COMPOSITE region output for the region carrying ``style``.

    style = "A": domain erode(M_A, erode), patches classified as style A
    style = "B": domain erode(M_B, erode), patches classified as style B
    """
    mask_a = load_mask(task)
    domain = erode_px(mask_a if style == "A" else ~mask_a, erode)

    sel = _grid_center_select(domain)
    p_grid = _pixel_to_grid(prob_a_pixel)
    p_style = p_grid if style == "A" else 1.0 - p_grid

    if uncertain_excluded:
        sel = sel & ~uncertain_grid(p_grid)

    if sel.sum() == 0:
        return float("nan")
    return float((p_style[sel] >= CLASSIFIER_CONFIG["threshold"]).mean())


def prsc_uncertain_retained(prob_a_pixel: np.ndarray, task: str,
                           style: str = "A", erode: int = PRSC_ERODE_PX) -> float:
    """Sensitivity variant: uncertain patches retained (Section 4.5)."""
    return prsc(prob_a_pixel, task, style=style, erode=erode,
                uncertain_excluded=False)


# ============================================================
# 3.6.3  CRSC interior and boundary-band leakage
# ============================================================
def crsc_interior(prob_a_pixel: np.ndarray, task: str, style: str = "A",
                  erode: int = CRSC_ERODE_PX,
                  exclude_boundary_band: bool = True) -> float:
    """
    CRSC interior for one composite.

    Domain: erode(M_B, erode) for style A, erode(M_A, erode) for style B.
    Uncertain patches are RETAINED. Boundary-band patches are excluded from the
    main CRSC and reported separately (boundary_band_leakage).
    """
    mask_a = load_mask(task)
    domain = erode_px(~mask_a if style == "A" else mask_a, erode)
    sel = _grid_center_select(domain)
    if exclude_boundary_band:
        sel = sel & ~_band_select(task)

    p_grid = _pixel_to_grid(prob_a_pixel)
    p_style = p_grid if style == "A" else 1.0 - p_grid

    if sel.sum() == 0:
        return float("nan")
    return float((p_style[sel] >= CRSC_THRESHOLD).mean())


def boundary_band_leakage(prob_a_pixel: np.ndarray, task: str,
                          style: str = "A") -> float:
    """
    Boundary-band leakage: proportion of patches inside the boundary band that
    the classifier assigns to the opposite style. Descriptive only; not part of
    the main CRSC.
    """
    sel = _band_select(task)
    p_grid = _pixel_to_grid(prob_a_pixel)
    p_style = p_grid if style == "A" else 1.0 - p_grid
    if sel.sum() == 0:
        return float("nan")
    return float((p_style[sel] >= CRSC_THRESHOLD).mean())


def crsc_noise_floor(direction: str) -> float:
    """
    Direction-specific classifier noise floor: FPR_A on B = 0.03 and
    FPR_B on A = 0.05 (Section 4.4). CRSC values near or below this floor are
    described as "not distinguishable from classifier error".
    """
    return CLASSIFIER_FPR["FPR_A_on_B"] if direction == "A" else CLASSIFIER_FPR["FPR_B_on_A"]


def interpret_crsc(value: float, direction: str = "A") -> str:
    """Descriptive label relative to the direction-specific noise floor."""
    floor = crsc_noise_floor(direction)
    if not np.isfinite(value):
        return "undefined"
    if value <= floor:
        return "not distinguishable from classifier error"
    return "above the direction-specific classifier noise floor"


# ============================================================
# 3.6.4  Masked style purity and CMMD
# ============================================================
def masked_purity(image, region_mask: np.ndarray, ref_paths) -> float:
    """Masked style purity on a neutral-filled masked region (Section 4.6)."""
    region = masked_region_image(image, region_mask)
    return style_purity(region, ref_paths)


def masked_cmmd(image, region_mask: np.ndarray, ref_paths) -> float:
    """CMMD between the masked region and the style reference set."""
    region = masked_region_image(image, region_mask)
    return cmmd(region, ref_paths)


# ============================================================
# 4.10  Transition width and base contribution
# ============================================================
def transition_width_median(prob_pixel: np.ndarray,
                            smooth_px: int = 3) -> float:
    """
    Median transition width in pixels: the distance over which the style
    probability moves from 0.1 to 0.9, measured along horizontal scanlines of a
    segmenter-smoothed probability map. This is a smoothed UPPER BOUND, not a
    geometric width (Limitations).
    """
    from scipy.ndimage import gaussian_filter

    if smooth_px and smooth_px > 0:
        prob_pixel = gaussian_filter(prob_pixel.astype(np.float32), smooth_px)

    widths = []
    for y in range(prob_pixel.shape[0]):
        row = prob_pixel[y]
        lo = np.where(row >= 0.1)[0]
        hi = np.where(row >= 0.9)[0]
        if lo.size == 0 or hi.size == 0:
            continue
        w = abs(int(hi[0]) - int(lo[0]))
        if w > 0:
            widths.append(w)
    return float(np.median(widths)) if widths else float("nan")


def base_contribution(composite, base_image, band_mask: np.ndarray,
                      tol: int = 2) -> float:
    """
    Base-image contribution inside the boundary band.

    NOTE: the manuscript does not state the exact definition of this quantity.
    The implementation below defines it as the fraction of band pixels where the
    composite still equals the base image within a per-channel tolerance of
    ``tol`` grey levels, which is consistent with the reported values
    (G1.0 = 0.000, H = 0.009, I = 0.010) under a fixed inpainting strength of
    0.80. The threshold is descriptive only and carries no confidence interval.
    """
    a = np.asarray(composite.convert("RGB"), dtype=np.int16)
    b = np.asarray(base_image.convert("RGB"), dtype=np.int16)
    diff = np.abs(a - b).max(axis=2)
    if band_mask.sum() == 0:
        return float("nan")
    return float((diff[band_mask] <= tol).mean())


__all__ = [
    "prsc", "prsc_uncertain_retained", "crsc_interior",
    "boundary_band_leakage", "crsc_noise_floor", "interpret_crsc",
    "masked_purity", "masked_cmmd", "transition_width_median",
    "base_contribution",
]
