# -*- coding: utf-8 -*-
"""
Metric families for the reporting-boundary reproduction.

    (i)   construction-consistency diagnostic : tma_iou
    (ii)  mask-defined descriptive metrics     : mask_defined
    (iii) non-task-mask references             : non_task_mask

Only family (iii) has an evaluation domain that is not the task mask, and even
there PB-IoU is a reference, not ground truth.
"""

from metrics.common import (
    soft_iou, uncertain_grid, region_iou, boundary_f1,
    mean_absolute_boundary_distance, masked_region_image, style_purity, cmmd,
    mmd_rbf, clip_embed_image,
)
from metrics.tma_iou import (
    tma_iou_directions, tma_iou_c, tma_iou_c_bundle, evaluate_composite,
)
from metrics.mask_defined import (
    prsc, prsc_uncertain_retained, crsc_interior, boundary_band_leakage,
    crsc_noise_floor, interpret_crsc, masked_purity, masked_cmmd,
    transition_width_median, base_contribution,
)
from metrics.non_task_mask import (
    consensus_partition, pb_iou, inter_annotator_stats,
    preference_proportion, split_preference_summary,
)

__all__ = [
    "soft_iou", "uncertain_grid", "region_iou", "boundary_f1",
    "mean_absolute_boundary_distance", "masked_region_image", "style_purity",
    "cmmd", "mmd_rbf", "clip_embed_image",
    "tma_iou_directions", "tma_iou_c", "tma_iou_c_bundle", "evaluate_composite",
    "prsc", "prsc_uncertain_retained", "crsc_interior",
    "boundary_band_leakage", "crsc_noise_floor", "interpret_crsc",
    "masked_purity", "masked_cmmd", "transition_width_median",
    "base_contribution",
    "consensus_partition", "pb_iou", "inter_annotator_stats",
    "preference_proportion", "split_preference_summary",
]
