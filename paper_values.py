# -*- coding: utf-8 -*-
"""
Numeric values as reported in the current manuscript.

Kept in one place so that figures and consistency checks can be run without a
GPU: every number below is a string-quoted transcription from Sections 4.1-4.12
and the Data Availability paragraph. Nothing here is estimated or interpolated.

Grouped by the manuscript's own metric families:

    (i)   construction-consistency diagnostic : TMA-IoU_C
    (ii)  mask-defined descriptive metrics    : PRSC, CRSC interior,
                                                masked purity, CMMD
    (iii) non-task-mask references            : mask-free preference, PB-IoU
"""

# ============================================================
# Condition labels (order used by Fig. 3 and all profile tables)
# ============================================================
CONDITIONS = ["G0.5", "D0.5", "G1.0", "D1.0", "H", "I"]

# ============================================================
# 4.3  Complete condition-level values
# ============================================================
TMA_IOU_C = {"G0.5": 0.69, "D0.5": 0.68, "G1.0": 0.67,
             "D1.0": 0.72, "H": 0.71, "I": 0.70}

PRSC_A = {"G0.5": 0.84, "D0.5": 0.80, "G1.0": 0.72,
          "D1.0": 0.65, "H": 0.86, "I": 0.86}

CRSC_A_INTERIOR = {"G0.5": 0.06, "D0.5": 0.24, "G1.0": 0.03,
                   "D1.0": 0.18, "H": 0.04, "I": 0.05}

CMMD_A = {"G0.5": 0.12, "D0.5": 0.14, "G1.0": 0.11,
          "D1.0": 0.18, "H": 0.13, "I": 0.10}

MASKED_PURITY_A = {"G0.5": 0.80, "D0.5": 0.82, "G1.0": 0.75,
                   "D1.0": 0.84, "H": 0.86, "I": 0.85}

# CRSC_A and CMMD_A are lower-is-better (Fig. 3 caption).
LOWER_IS_BETTER = ["CRSC_A_interior", "CMMD_A", "boundary_band_leakage_A"]

# ============================================================
# 4.4  CRSC interior, boundary-band leakage and the classifier noise floor
# ============================================================
# Order: A, B, G0.5, D0.5, G1.0, D1.0, H, I
CRSC_EXTENDED_ORDER = ["A", "B", "G0.5", "D0.5", "G1.0", "D1.0", "H", "I"]

CRSC_A_EXTENDED = [0.38, 0.41, 0.06, 0.24, 0.03, 0.18, 0.04, 0.05]
CRSC_B_EXTENDED = [0.38, 0.41, 0.07, 0.25, 0.04, 0.19, 0.05, 0.06]
BOUNDARY_BAND_LEAKAGE = {"A": 0.42, "B": 0.45, "G0.5": 0.10, "D0.5": 0.27,
                         "G1.0": 0.12, "D1.0": 0.21, "H": 0.15, "I": 0.18}

# ============================================================
# 4.1 / Table 1  Core negative controls (mean TMA-IoU_C decrease)
# ============================================================
NEG_CONTROL_DECREASE = {
    "mask_swap": {"G0.5": 0.31, "D0.5": 0.30, "G1.0": 0.29,
                  "D1.0": 0.33, "H": 0.28, "I": 0.27},
    "label_swap": {"G0.5": 0.27, "D0.5": 0.26, "G1.0": 0.24,
                   "D1.0": 0.29, "H": 0.23, "I": 0.22},
    "compositing_mask_perturbation": {"G0.5": 0.12, "D0.5": 0.11, "G1.0": 0.10,
                                      "D1.0": 0.14, "H": 0.09, "I": 0.08},
}
NEG_CONTROL_DESIGN = {
    "mask_swap": "Incorrect compositing mask; generated regions fixed; "
                 "evaluated against correct task mask",
    "label_swap": "Swapped style labels; original task mask; "
                  "generated regions fixed",
    "compositing_mask_perturbation": "+/-8 / +/-16 px boundary-normal "
                                     "translation; 8 px dilation/erosion; "
                                     "generated regions fixed",
}
NEG_CONTROL_COUNTS = {"GD_composites": 120, "HI_composites": 60,
                      "total_composites": 180, "prompts_per_condition": 10}

# ============================================================
# 4.7 / 4.8  Non-task-mask references
# ============================================================
MASK_FREE_PREFERENCE = {"condition": "G0.5", "proportion": 0.586,
                        "n_success": 178, "n_pairs": 304}
OVERALL_PREFERENCE = {"condition": "G0.5", "proportion": 0.559,
                      "n_success": 170, "n_pairs": 304}
FORCED_COMPLIANCE_PREFERENCE = {"condition": "G0.5", "proportion": 0.641,
                                "n_success": 195, "n_pairs": 304}

PB_IOU = {"G0.5": 0.62, "D0.5": 0.60, "G1.0": 0.59,
          "D1.0": 0.65, "H": 0.67, "I": 0.66}
PB_IOU_META = {"images": 40, "annotators": 2, "annotations": 80,
               "inter_annotator_boundary_f1": 0.86,
               "mean_absolute_boundary_distance_px": 4.1}

# ============================================================
# 4.9  Naturalness and boundary artifacts (10-seed subset)
# ============================================================
NATURALNESS = {"G0.5": 3.60, "D0.5": 3.50}
BOUNDARY_ARTIFACT = {"G0.5": 3.70, "D0.5": 3.65}

# ============================================================
# 4.10  Overlap generation, transition width, base contribution
# ============================================================
OVERLAP_PROFILE = {
    "G1.0": {"base_contribution": 0.000, "transition_width_median_px": 18.4},
    "H": {"base_contribution": 0.009, "transition_width_median_px": 27.6},
    "I": {"base_contribution": 0.010, "transition_width_median_px": 36.2},
}
OVERLAP_HI_BOUNDARY_BAND = {"H": {"naturalness": 3.8, "artifact": 3.9},
                            "I": {"naturalness": 3.9, "artifact": 4.0}}

# ============================================================
# 4.11  Generalization profiles (descriptive only)
# ============================================================
GENERALIZATION_PROFILE = {
    "RLIMC (G1.0)":         {"PRSC_A": 0.72, "CRSC_A_interior": 0.03,
                             "naturalness": 3.6, "artifact": 3.7},
    "attention-supervised": {"PRSC_A": 0.78, "CRSC_A_interior": 0.07,
                             "naturalness": 3.4, "artifact": 3.3},
    "latent composition":   {"PRSC_A": 0.72, "CRSC_A_interior": 0.10,
                             "naturalness": 3.9, "artifact": 3.8},
    "prompt layout":        {"PRSC_A": 0.64, "CRSC_A_interior": 0.18,
                             "naturalness": 3.5, "artifact": 3.4},
    "global LoRA":          {"PRSC_A": 0.70, "CRSC_A_interior": 0.41,
                             "naturalness": 3.3, "artifact": 3.2},
}

# ============================================================
# 4.12  Classifier validation
# ============================================================
CLASSIFIER_META = {
    "backbone": "ResNet-50",
    "patch": "32 x 32, stride 8",
    "train_images": "200 ink-wash + 200 cyberpunk (independent of LoRA sets)",
    "uncertain_band": "0.4 - 0.6",
    "fpr_a_on_b": 0.03,
    "fpr_b_on_a": 0.05,
    "cohens_kappa_lora_screening": 0.88,
}

# ============================================================
# Limitations  Design sensitivity of the preference study
# ============================================================
DESIGN_SENSITIVITY = {"alpha": 0.05, "n": 304, "critical_success_count": 167,
                      "power": {"0.65": ">0.99", "0.60": "~0.97"}}
