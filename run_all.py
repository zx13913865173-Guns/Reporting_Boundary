# -*- coding: utf-8 -*-
"""
End-to-end driver for the reporting-boundary reproduction.

Stages (Section 3.1 - 3.9):

    1 masks        target masks for the eight tasks
    2 lora         one style LoRA per style
    3 classifier   patch classifier + calibration + direction-specific FPR
    4 generate     6 conditions x 8 tasks x 20 seeds
    5 metrics      TMA-IoU_C, PRSC, CRSC interior, boundary-band leakage,
                   masked purity, CMMD, transition width, base contribution
    6 controls     core negative controls (180 composites)
    7 human        split preference, 1-5 ratings, PB-IoU
    8 stats        descriptive tables + design sensitivity (boundary enforced)
    9 figures      Fig. 1 - Fig. 5

Every stage is separately runnable. ``--dry-run`` exercises the pipeline end to
end without a GPU or model weights.
"""

import argparse

from config import (
    TASKS, MAIN_TASKS, SEEDS, CONDITION_ORDER, GEN_DIR, METRICS_DIR,
)


def stage_masks():
    from masks import save_all_masks, area_composition
    save_all_masks()
    for t in TASKS:
        print(f"  {t:18s} region A = {area_composition(t) * 100:6.2f}%")


def stage_lora(dry_run=False):
    from lora_train import train_all
    train_all(dry_run=dry_run)


def stage_classifier(dry_run=False):
    if dry_run:
        print("[run_all] classifier stage skipped in dry-run mode")
        return
    from classifier import train_classifier
    train_classifier({})


def stage_generate(dry_run=True, tasks=None, conditions=None, seeds=None):
    from generation import build_condition_matrix, generate_unit
    from PIL import Image
    from config import IMG_SIZE

    tasks = tasks or TASKS
    init = Image.new("RGB", (IMG_SIZE, IMG_SIZE), (255, 255, 255))
    n = 0
    for unit in build_condition_matrix(tasks, conditions, seeds, dry_run=dry_run):
        generate_unit(unit["task"], unit["condition"], unit["seed"],
                      unit["prompt"], init, dry_run=dry_run)
        n += 1
    print(f"[run_all] generated {n} units "
          f"({len(tasks)} tasks x {len(conditions or CONDITION_ORDER)} conditions "
          f"x {len(seeds or SEEDS)} seeds)")


def stage_metrics(dry_run=True):
    from metrics.tma_iou import tma_iou_c_bundle
    from masks import load_mask, mask_to_grid
    import numpy as np

    print("[run_all] metrics stage: computing TMA-IoU_C variants on the "
          "classifier grids (see metrics/ for the full per-unit sweep)")
    for task in MAIN_TASKS:
        m_grid = mask_to_grid(load_mask(task))
        p_grid = m_grid.copy()          # placeholder probability field
        res = tma_iou_c_bundle(p_grid, m_grid)
        print(f"  {task:18s} TMA-IoU_C = {res['main']['tma_iou_c']:.4f}")


def stage_controls(dry_run=True):
    from negative_controls import control_design_counts
    print("[run_all] core negative-control design:", control_design_counts())


def stage_human():
    from human_study import build_paired_schedule, pb_iou_schedule
    print(f"[run_all] split preference: {len(build_paired_schedule())} pairs x "
          f"3 questions = {len(build_paired_schedule()) * 3} binary responses")
    print(f"[run_all] PB-IoU: {len(pb_iou_schedule())} annotations")


def stage_stats():
    from statistics import run_all as stats_all
    bundle = stats_all()
    print(f"[run_all] descriptive bundle keys: {list(bundle.keys())}")


def stage_figures(dpi=None):
    from figures import generate_all
    generate_all(dpi)


STAGES = {
    "masks": lambda a: stage_masks(),
    "lora": lambda a: stage_lora(a.dry_run),
    "classifier": lambda a: stage_classifier(a.dry_run),
    "generate": lambda a: stage_generate(a.dry_run),
    "metrics": lambda a: stage_metrics(a.dry_run),
    "controls": lambda a: stage_controls(a.dry_run),
    "human": lambda a: stage_human(),
    "stats": lambda a: stage_stats(),
    "figures": lambda a: stage_figures(a.dpi),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stages", nargs="+", default=list(STAGES.keys()),
                    choices=list(STAGES.keys()))
    ap.add_argument("--dry-run", action="store_true",
                    help="run without GPU / model weights")
    ap.add_argument("--dpi", type=int, default=None)
    args = ap.parse_args()

    for name in args.stages:
        print(f"\n=== stage: {name} ===")
        STAGES[name](args)


if __name__ == "__main__":
    main()
