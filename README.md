# Reporting-Boundary Analysis for Mask-Specified Regional Style Generation

Official reproduction materials (code + workflow) for the manuscript:

> **Construction-Dependent Agreement Is Not, by Itself, Independent Quality
> Evidence: A Reporting-Boundary Analysis for Mask-Specified Regional Style
> Generation**

---

## 2. Directory Structure

```text
Reporting_Boundary_Reproduction/
├── README.md                      # This file
├── requirements.txt               # Dependency list
├── DATA_AVAILABILITY.md           # Data availability + numerical discrepancies
├── config.py                      # Global configuration: tasks, condition matrix, hyperparameters
├── paper_values.py                # All manuscript values (for offline plotting/verification)
├── masks.py                       # 8 task masks, signed distance, patch grid mapping
├── lora_train.py                  # Style LoRA training + expert screening
├── classifier.py                  # 32x32 patch style classifier, temperature calibration
├── generation.py                  # RLIMC two-stage generation + condition matrix
├── negative_controls.py           # Three core negative controls (180 composites)
├── human_study.py                 # Paired preference, 1-5 scale, PB-IoU annotation
├── statistics.py                  # Descriptive tables + design sensitivity
├── figures.py                     # Figures 1-5 generation scripts
├── run_all.py                     # End-to-end driver (9 stages)
├── selftest.py                    # CPU-only smoke test (14 items)
├── metrics/
│   ├── __init__.py
│   ├── common.py                  # Patch grid, soft IoU, CLIP embedding, CMMD kernel
│   ├── tma_iou.py                 # TMA-IoU_C implementation
│   ├── mask_defined.py            # PRSC, CRSC interior, purity, CMMD
│   └── non_task_mask.py           # PB-IoU, consensus partition, preference proportions
├── docs/
│   ├── WORKFLOW.md                # 7-stage reproduction workflow
│   ├── CONDITION_MATRIX.md        # Six conditions x eight tasks x 20 seeds design table
│   └── REPRODUCTION_NOTES.md      # Item-by-item discrepancy checklist (must-read)
├── data/                          # Training images, annotations (not distributed)
└── outputs/                       # Masks, generation results, metric CSVs, figures
---


# 0) Dependencies
pip install -r requirements.txt

# 1) CPU-only self-test: geometry conventions, metric pipelines, negative controls, statistical boundaries, five figures
python selftest.py

# 2) Stage-by-stage execution (any stage can be run independently)
python run_all.py --stages masks
python run_all.py --stages lora
python run_all.py --stages classifier
python run_all.py --stages generate
python run_all.py --stages metrics controls
python run_all.py --stages human stats figures

# 3) Verify pipeline connectivity without GPU/weights
python run_all.py --dry-run --stages masks metrics figures --dpi 150
见 `DATA_AVAILABILITY.md`。
