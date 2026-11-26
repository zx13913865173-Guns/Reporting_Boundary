# -*- coding: utf-8 -*-
"""
Global configuration for the reporting-boundary reproduction.

This file encodes the design of the CURRENT manuscript:

    "Construction-Dependent Agreement Is Not, by Itself, Independent Quality
     Evidence: A Reporting-Boundary Analysis for Mask-Specified Regional
     Style Generation"

Everything here is taken from the manuscript text (Sections 3.1-3.9 and the
Data Availability statement). Where the manuscript does not specify an
implementation detail, the item is marked with NOTE: and the chosen default is
recorded in docs/REPRODUCTION_NOTES.md as a to-be-confirmed item.

Differences with respect to the earlier method-oriented draft (and the public
repository it released) are flagged with LEDGER: so that the whole set can be
re-verified against the Data Availability paragraph at once.
"""

from pathlib import Path

# ============================================================
# Paths
# ============================================================
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
FIG_DIR = ROOT / "figures"

MASK_DIR = OUTPUT_DIR / "target_masks"
GEN_DIR = OUTPUT_DIR / "generated"          # region outputs + composites
CKPT_DIR = OUTPUT_DIR / "checkpoints"
METRICS_DIR = OUTPUT_DIR / "metrics"
ANNOT_DIR = OUTPUT_DIR / "annotations"

for _d in (DATA_DIR, OUTPUT_DIR, FIG_DIR, MASK_DIR, GEN_DIR,
           CKPT_DIR, METRICS_DIR, ANNOT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ============================================================
# 3.1  Style-boundary tasks and target masks
# ============================================================
# Eight tasks, two styles, varying only in boundary shape and location.
# Geometry is transcribed from Section 3.1 (1024 x 1024 canvas).
#
# LEDGER: the earlier released masks.py used a different geometry and a
# different task<->shape assignment:
#     - irregular_sine amplitude  : released 256 px  -> manuscript 128 px
#     - frame border thickness    : released 81 px   -> manuscript 20 px
#     - horizontal_split          : released split along x -> manuscript y = 512
#     - vertical_split            : released split along y -> manuscript x = 512
# The geometry below follows the manuscript.

IMG_SIZE = 1024
CENTER = IMG_SIZE // 2                      # 512

TASKS = [
    "horizontal_split",     # boundary y = 512
    "vertical_split",       # boundary x = 512
    "diagonal_45",          # boundary y = x
    "diagonal_135",         # boundary y = 1024 - x
    "center_circle",        # centre (512, 512), radius 256
    "corner_quadrant",      # top-left quadrant isolation
    "irregular_sine",       # y = 512 + 128 sin(2 pi x / 1024)
    "frame_border",         # 20 px outer frame
]

TASK_LABELS = {
    "horizontal_split": "Horizontal split (y = 512)",
    "vertical_split":   "Vertical split (x = 512)",
    "diagonal_45":      "Diagonal 45 deg (y = x)",
    "diagonal_135":     "Diagonal 135 deg (y = 1024 - x)",
    "center_circle":    "Centre circle (r = 256)",
    "corner_quadrant":  "Top-left quadrant",
    "irregular_sine":   "Irregular sine (A = 128)",
    "frame_border":     "Frame border (20 px)",
}

SINE_AMPLITUDE = 128        # manuscript: y = 512 + 128 sin(2 pi x / 1024)
FRAME_THICKNESS = 20        # manuscript: 20 px outer frame

# The Frame task is excluded from the main aggregate because its 20 px border
# is smaller than the 32 x 32 classifier patch; it is a sensitivity check only.
FRAME_EXCLUDED_FROM_MAIN = True
MAIN_TASKS = [t for t in TASKS if t != "frame_border"]

# ============================================================
# 3.2  Condition matrix and construction status
# ============================================================
# Six construction-defined descriptive conditions:
#   G/D hard-mask subgroup  : G0.5, D0.5, G1.0, D1.0
#   H/I overlap-generation  : H (delta = 8), I (delta = 16)
CONDITION_ORDER = ["G0.5", "D0.5", "G1.0", "D1.0", "H", "I"]

CONSTRUCTION_SUBGROUP = {
    "G0.5": "GD", "D0.5": "GD", "G1.0": "GD", "D1.0": "GD",
    "H": "HI", "I": "HI",
}

# Non-construction conditions: retained for descriptive reference only and NOT
# part of the six-condition construction-defined descriptive set.
#
#   A = No-Control
#   B = Full-image Global LoRA
#   C = third non-construction reference
#
# NOTE: the manuscript names A, B and C as "descriptive reference only" but
# only defines A and B explicitly (Section 4.4). C is left unspecified; the
# pipeline exposes it as a pluggable factory (see generation.py) and it is
# never pooled with the six construction-defined conditions.
NON_CONSTRUCTION_CONDITIONS = ["A", "B", "C"]
NON_CONSTRUCTION_DEFINED = {
    "A": "No-Control (no LoRA, no mask)",
    "B": "Full-image Global LoRA",
    "C": "third non-construction reference - NOTE: not defined in manuscript",
}

# Mask-only diagnostic condition (no style label). Used only for
# classifier-response diagnostics; NOT part of the condition matrix.
MASK_ONLY_LABEL = "Mask-only"

# ============================================================
# 3.4  Overlap-generation variants H and I
# ============================================================
# D = signed distance to the target boundary, positive inside region A.
#   M_A,gen = (D > -delta)      M_B,gen = (D < delta)
# Both passes generate inside the overlap band; the compositing alpha blends
# the two generated results inside the band.
OVERLAP_DELTA = {"H": 8.0, "I": 16.0}       # manuscript: H delta = 8, I delta = 16
BASE_CONTRIBUTION_THRESHOLD = 0.01          # pre-specified; descriptive only

# ============================================================
# 3.5  LoRA training and generation
# ============================================================
STYLES = ["ink_wash", "cyberpunk"]
STYLE_A = "ink_wash"       # region A style
STYLE_B = "cyberpunk"      # region B style

LORA_CONFIG = {
    "base_model": "stabilityai/stable-diffusion-xl-base-1.0",
    "rank": 32,             # manuscript: rank 32
    "alpha": 1.0,           # manuscript: alpha = 1.0
    "steps": 3000,          # manuscript: 3000 steps
    "lr": 1e-4,             # manuscript: learning rate 1e-4
    "resolution": IMG_SIZE,
}

# 50 training images per style, screened by two domain experts on a 3-point
# scale; only images rated 3 by both experts retained. Cohen's kappa = 0.88.
LORA_TRAIN_IMAGES_PER_STYLE = 50
SCREENING_SCALE_MAX = 3
SCREENING_COHENS_KAPPA = 0.88

GEN_CONFIG = {
    "num_inference_steps": 30,      # manuscript: 30 inference steps
    "guidance_scale": 7.5,          # manuscript: CFG scale 7.5
    "scheduler": "DPM++ 2M Karras",
    "resolution": IMG_SIZE,
    "inpainting_strength": 0.80,    # manuscript: fixed inpainting strength 0.80
    "dtype": "float16",
}

# seed 42 plus 19 additional seeds
SEEDS = [42] + list(range(100, 119))        # 20 seeds
HUMAN_RATING_SEEDS = SEEDS[:10]             # human ratings on the 10-seed subset

# LoRA dose per condition (target-dose-matched comparison G0.5 vs D0.5).
#
# NOTE: the manuscript states that D1.0 belongs to the "D dual-LoRA subgroup"
# but does not spell out the exact difference between the G and D passes.
# The two passes are therefore configured explicitly and flagged in
# docs/REPRODUCTION_NOTES.md.
CONDITION_SETUP = {
    # name    kind          dose_A  dose_B  generator_mask
    "G0.5":  ("hard_mask",   0.5,    0.5,    "hard"),
    "D0.5":  ("dual_lora",   0.5,    0.5,    "hard"),
    "G1.0":  ("hard_mask",   1.0,    1.0,    "hard"),
    "D1.0":  ("dual_lora",   1.0,    1.0,    "hard"),
    "H":     ("overlap",     1.0,    1.0,    "overlap_8"),
    "I":     ("overlap",     1.0,    1.0,    "overlap_16"),
}

# ============================================================
# 3.6.1 / 4.12  Style classifier
# ============================================================
CLASSIFIER_CONFIG = {
    "backbone": "resnet50",
    "patch_size": 32,           # manuscript: 32 x 32 patches
    "stride": 8,                # manuscript: stride 8
    "train_images_per_style": 200,   # 200 ink-wash + 200 cyberpunk
    "epochs": 50,
    "batch_size": 32,
    "lr": 1e-4,
    "threshold": 0.5,           # 0.5 probability threshold
    "uncertain_lo": 0.4,        # probabilities in [0.4, 0.6] are uncertain
    "uncertain_hi": 0.6,
    "calibration": "temperature",    # calibrated probabilities for the main IoU
}

# Direction-specific classifier false-positive rates.
# CRSC_A is compared with FPR_A on B and CRSC_B with FPR_B on A. Values near or
# below the noise floor are described as "not distinguishable from classifier
# error", never as zero leakage or mechanism-specific suppression.
CLASSIFIER_FPR = {"FPR_A_on_B": 0.03, "FPR_B_on_A": 0.05}

# ============================================================
# 3.6.1  Mask -> patch-grid mapping
# ============================================================
# NOTE: the manuscript explicitly states that it does not specify whether the
# pixel-level task mask is mapped to the classifier patch grid by patch-centre
# sampling or by patch-area coverage, and that the exact rule should be taken
# from the released implementation. The released implementation below uses
# patch-centre sampling as the primary rule and ships patch-area coverage as a
# sensitivity variant.
MASK_TO_GRID_PRIMARY = "patch_center"       # {"patch_center", "patch_area"}

# ============================================================
# 3.6.2 / 3.6.3  PRSC and CRSC interior
# ============================================================
PRSC_ERODE_PX = 32          # PRSC uses erode(M_A, delta = 32 px)
CRSC_ERODE_PX = 32          # CRSC uses erode(M_B, delta = 32 px)
CRSC_THRESHOLD = 0.5        # calibrated 0.5 probability threshold
# PRSC excludes uncertain patches, CRSC retains them; the asymmetry is a
# stated limitation and both variants are shipped (see metrics/mask_defined.py).
BOUNDARY_BAND_PX = 32       # boundary-band leakage band; NOT in the main CRSC

# ============================================================
# 3.6.4  Masked style purity and CMMD
# ============================================================
STYLE_REF_IMAGES = 200      # 200 reference images of the target style
REGION_FILL = "neutral"     # masked regions use a neutral fill; no bbox crops
CMMD_KERNEL_SIGMA_SCALE = 1.0

# ============================================================
# 3.6.5  Core negative controls
# ============================================================
NEG_CONTROL_PROMPTS = 10    # 10 prompts per condition
NEG_CONTROLS = ["mask_swap", "label_swap", "compositing_mask_perturbation"]
PERTURB_TRANSLATIONS_PX = [-16, -8, 8, 16]  # boundary-normal translations
PERTURB_MORPH_PX = 8                        # dilation / erosion by 8 px

# G/D core controls : 3 controls x 4 conditions x 10 prompts = 120 composites
# H/I sensitivity   : 3 controls x 2 conditions x 10 prompts =  60 composites
#                                                       total = 180 composites
NEG_CONTROL_GD_CONDITIONS = ["G0.5", "D0.5", "G1.0", "D1.0"]
NEG_CONTROL_HI_CONDITIONS = ["H", "I"]

# ============================================================
# 3.6.6 / 3.7  Non-task-mask references
# ============================================================
PREFERENCE_COMPARISON = ("G0.5", "D0.5")    # mask-free reference comparison
PREFERENCE_PAIRS = 304                      # 304 paired comparisons
PREFERENCE_QUESTIONS = ["mask_free", "overall", "forced_compliance"]
PREFERENCE_RESPONSES = PREFERENCE_PAIRS * len(PREFERENCE_QUESTIONS)   # 912

PB_IOU_IMAGES = 40          # 40 images annotated by both annotators
PB_IOU_ANNOTATORS = 2       # 80 annotations in total
PB_IOU_ANNOTATIONS = PB_IOU_IMAGES * PB_IOU_ANNOTATORS   # 80
PB_IOU_INTER_ANNOTATOR_F1 = 0.86
PB_IOU_MEAN_ABS_DISTANCE_PX = 4.1

# ============================================================
# 3.6.7 / 4.9  Human ratings
# ============================================================
RATING_SCALES = ["naturalness", "boundary_artifact"]   # two separate 1-5 scales
RATING_MIN, RATING_MAX = 1, 5

# ============================================================
# 3.8  Statistical analysis
# ============================================================
# The manuscript reports NO condition-level SRCC, NO p-values, NO confidence
# intervals and NO Fisher-transformed intervals, and NO mixed-effects models.
# statistics.py asserts this boundary instead of silently computing them.
REPORT_SRCC = False
REPORT_P_VALUES = False
REPORT_CONFIDENCE_INTERVALS = False
REPORT_MIXED_EFFECTS = False

# Design-sensitivity calculation (Limitations): hypothetical one-sided exact
# binomial test, alpha = 0.05, critical success count c = 167, power > 0.99 at
# p1 = 0.65 and approximately 0.97 at p1 = 0.60. Characterises design
# sensitivity only - no p-value is derived from it.
DESIGN_SENSITIVITY = {
    "alpha": 0.05,
    "n": PREFERENCE_PAIRS,
    "critical_success_count": 167,
    "p1_grid": [0.60, 0.65],
}

# ============================================================
# Figures
# ============================================================
FIG_SIZE = (16, 9)
DPI = 900

BLUE_DARK = "#0B3C6B"
BLUE_MID = "#1F6FB2"
BLUE_LIGHT = "#7FB3D5"
BLUE_PALE = "#C9DDF0"
BLUE_GRAY = "#5D6D7E"

# ============================================================
# Reproducibility ledger (Data Availability paragraph)
# ============================================================
# Three condition-level values that differ between the earlier
# method-oriented draft and the present manuscript.
LEDGER_DIFFERENCES = [
    ("masked purity A, G0.5 vs D0.5",
     "0.80 vs 0.82 (this manuscript)", "0.84 vs 0.80 (earlier draft)"),
    ("PRSC_A, G1.0",
     "0.72 (this manuscript)", "0.86 (earlier draft)"),
    ("PRSC_A difference, G0.5 vs D0.5",
     "+0.04 (this manuscript)", "+0.06 [0.03, 0.09] (earlier draft)"),
]

REPO_URL = ("https://github.com/zx13913865173-Guns/"
            "Style-Boundary-Delineation-in-Human-AI-Co-Creation")
