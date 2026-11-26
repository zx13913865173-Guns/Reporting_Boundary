# -*- coding: utf-8 -*-
"""
Smoke self-test - runs the whole GPU-free part of the reproduction on
synthetic data and asserts the geometric and metric conventions.

Run from the repository root:

    python selftest.py

Everything checked here can be reproduced without model weights: the mask
geometry, the patch-grid mapping, the soft IoU, TMA-IoU_C, PRSC / CRSC interior,
the negative-control operators, the human-study schedules, the reporting
boundary guards, the design-sensitivity calculation and all five figures.
"""

import sys
import traceback

import numpy as np
from PIL import Image

PASS, FAIL = [], []


def check(name, fn):
    try:
        fn()
        PASS.append(name)
        print(f"  PASS  {name}")
    except Exception as exc:                       # noqa: BLE001
        FAIL.append((name, exc))
        print(f"  FAIL  {name}: {exc}")
        traceback.print_exc(limit=2)


# ============================================================
def test_mask_geometry():
    from masks import build_mask
    n = 1024

    m = build_mask("horizontal_split")
    assert m[100, 700] and not m[900, 700], "horizontal split must use y = 512"

    m = build_mask("vertical_split")
    assert m[700, 100] and not m[700, 900], "vertical split must use x = 512"

    m = build_mask("irregular_sine")
    assert m[639, 256] and not m[641, 256], "sine amplitude must be 128 px"

    m = build_mask("frame_border")
    assert m[19, 500] and not m[20, 500], "frame thickness must be 20 px"
    assert not m[500, 500], "frame must not fill the interior"

    m = build_mask("center_circle")
    assert m[512, 512] and not m[769, 512], "centre circle radius must be 256 px"

    m = build_mask("corner_quadrant")
    assert m[10, 10] and not m[900, 900], "top-left quadrant only"


def test_patch_grid():
    from masks import patch_coords, grid_shape, mask_to_grid, build_mask
    k, _ = grid_shape()
    assert k == 125, f"expected a 125 x 125 grid, got {k}"
    assert len(patch_coords()) == 125 * 125

    grid = mask_to_grid(build_mask("horizontal_split"), rule="patch_center")
    assert grid.shape == (125, 125)
    assert set(np.unique(grid)).issubset({0.0, 1.0})

    area = mask_to_grid(build_mask("horizontal_split"), rule="patch_area")
    assert area.min() >= 0.0 and area.max() <= 1.0


def test_soft_iou():
    from metrics.common import soft_iou
    a = np.array([[1.0, 0.0], [0.5, 0.5]])
    assert abs(soft_iou(a, a.copy()) - 1.0) < 1e-9
    z = np.zeros((2, 2)); o = np.ones((2, 2))
    assert soft_iou(o, z) == 0.0


def test_tma_iou_c():
    from metrics.tma_iou import tma_iou_c, tma_iou_c_bundle
    m = np.zeros((4, 4)); m[:2] = 1.0

    perfect = tma_iou_c(m, m)
    assert abs(perfect["tma_iou_c"] - 1.0) < 1e-9
    assert abs(perfect["iou_soft_A"] - 1.0) < 1e-9

    inverted = tma_iou_c(1.0 - m, m)
    assert inverted["tma_iou_c"] < 0.05

    # uncertain-excluded must drop fully-uncertain patches
    p = np.full((4, 4), 0.5)
    ex = tma_iou_c(p, m, variant="uncertain_excluded")
    assert np.isnan(ex["tma_iou_c"]), "all-uncertain grid must yield no patches"

    b = tma_iou_c_bundle(0.7 * np.ones((4, 4)), m)
    assert set(b) == {"main", "uncertain_assigned", "uncertain_excluded"}


def test_signed_distance_and_overlap():
    from masks import signed_distance, overlap_generation_masks
    d = signed_distance("horizontal_split")
    assert d[100, 500] > 0, "D must be positive inside region A"
    assert d[900, 500] < 0, "D must be negative outside region A"

    m_a_gen, m_b_gen, alpha = overlap_generation_masks("horizontal_split", 16.0)
    assert m_a_gen[512 + 8, 500], "M_A,gen must extend 16 px into region B"
    assert not m_a_gen[512 + 40, 500]
    assert alpha.min() >= 0.0 and alpha.max() <= 1.0
    assert alpha[100, 500] == 1.0 and alpha[900, 500] == 0.0


def test_prsc_and_crsc():
    from masks import build_mask
    from metrics.mask_defined import (
        prsc, crsc_interior, boundary_band_leakage, crsc_noise_floor,
        interpret_crsc,
    )
    m_a = build_mask("horizontal_split")
    p_ideal = np.where(m_a, 1.0, 0.0).astype(np.float64)   # perfect region A

    assert abs(prsc(p_ideal, "horizontal_split", style="A") - 1.0) < 1e-9
    assert abs(prsc(p_ideal, "horizontal_split", style="B") - 1.0) < 1e-9

    # CRSC_A on a perfectly composited image must be ~0 (no A inside B domain)
    crsc_a = crsc_interior(p_ideal, "horizontal_split", style="A")
    assert crsc_a < 0.05, f"clean composite should not leak, got {crsc_a}"

    leaky = np.ones_like(p_ideal)                           # everything style A
    assert crsc_interior(leaky, "horizontal_split", style="A") > 0.9
    assert 0.0 <= boundary_band_leakage(leaky, "horizontal_split") <= 1.0

    assert crsc_noise_floor("A") == 0.03 and crsc_noise_floor("B") == 0.05
    assert "not distinguishable" in interpret_crsc(0.02, "A")


def test_transition_and_base():
    from metrics.mask_defined import transition_width_median, base_contribution
    # a horizontal ramp crosses 0.1 and 0.9 along every scanline
    ramp = np.tile(np.linspace(0.0, 1.0, 1024, dtype=np.float32), (1024, 1))
    w = transition_width_median(ramp, smooth_px=0)
    assert np.isfinite(w) and w > 0, f"ramp must give a finite width, got {w}"

    img = Image.fromarray(np.full((64, 64, 3), 200, dtype=np.uint8))
    band = np.zeros((64, 64), dtype=bool); band[:32] = True
    assert abs(base_contribution(img, img, band) - 1.0) < 1e-9

    other = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    assert base_contribution(img, other, band) == 0.0


def test_non_task_mask():
    from metrics.non_task_mask import (
        region_iou, consensus_partition, pb_iou, preference_proportion,
    )
    a = np.zeros((8, 8), dtype=bool); a[:4] = True
    assert abs(region_iou(a, a) - 1.0) < 1e-9

    b1 = np.zeros((8, 8), dtype=bool); b1[4, :] = True
    b2 = np.zeros((8, 8), dtype=bool); b2[4, :] = True
    cons = consensus_partition(b1, b2)
    assert cons[:4].all() and not cons[4:].any()
    assert abs(pb_iou(a, cons) - 1.0) < 1e-9

    r = [1, 1, 0, 1, 0]
    assert abs(preference_proportion(r) - 0.6) < 1e-9


def test_negative_controls():
    from negative_controls import (
        run_mask_swap, run_label_swap,
        run_compositing_mask_perturbation, control_design_counts,
    )
    from masks import build_mask
    from metrics.tma_iou import _pixel_to_grid

    m_a = build_mask("horizontal_split")
    bright = Image.fromarray((np.where(m_a, 255, 0)).astype(np.uint8)[:, :, None]
                             .repeat(3, axis=2))
    dark = Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8))

    def prob_fn(image, task):
        px = np.array(image.convert("L"), dtype=np.float64) / 255.0
        return _pixel_to_grid(px)

    d_swap = run_mask_swap("horizontal_split", bright, dark, prob_fn)
    d_label = run_label_swap("horizontal_split", bright, dark, prob_fn)
    d_pert = run_compositing_mask_perturbation(
        "horizontal_split", bright, dark, prob_fn)

    assert d_swap > 0.2, f"mask swap must reduce TMA-IoU_C, got {d_swap}"
    assert d_label > 0.2, f"label swap must reduce TMA-IoU_C, got {d_label}"
    assert 0.0 < d_pert < d_swap, (
        f"perturbation must be milder than swap ({d_pert} vs {d_swap})")

    counts = control_design_counts()
    assert counts["GD_composites"] == 120
    assert counts["HI_composites"] == 60
    assert counts["total_composites"] == 180


def test_human_study():
    from human_study import (
        build_paired_schedule, pb_iou_schedule, check_rating_domain,
        summarise_ratings,
    )
    sch = build_paired_schedule()
    assert len(sch) == 304
    assert len(pb_iou_schedule()) == 80
    assert check_rating_domain([1, 2, 3, 4, 5]) and not check_rating_domain([0, 6])
    s = summarise_ratings({"naturalness": {"G0.5": [3.6, 3.6]}})
    assert "naturalness" in s and s["naturalness"]["G0.5"]["n"] == 2


def test_statistics_boundary():
    from statistics import (
        spearman_rank_correlation, p_value, confidence_interval,
        mixed_effects_model, assert_reporting_boundary, design_sensitivity,
    )
    for fn in (spearman_rank_correlation, p_value, confidence_interval,
               mixed_effects_model):
        try:
            fn()
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"{fn.__name__} must refuse to compute")

    assert all(v is False for v in assert_reporting_boundary().values())

    ds = design_sensitivity()
    p65 = float(ds.loc[ds["p1"] == 0.65, "power"].iloc[0])
    p60 = float(ds.loc[ds["p1"] == 0.60, "power"].iloc[0])
    assert p65 > 0.99, f"power at p1=0.65 must exceed 0.99, got {p65}"
    assert 0.95 < p60 < 0.99, f"power at p1=0.60 must be about 0.97, got {p60}"


def test_classifier_metrics():
    from classifier import _auc, _ece
    p = np.array([0.9, 0.8, 0.2, 0.1])
    y = np.array([1, 1, 0, 0])
    assert abs(_auc(p, y) - 1.0) < 1e-9
    assert _ece(p, y) < 0.2


def test_paper_values_consistency():
    import paper_values as PV
    from config import CONDITION_ORDER
    for name, d in (("TMA_IOU_C", PV.TMA_IOU_C), ("PRSC_A", PV.PRSC_A),
                    ("CRSC_A_INTERIOR", PV.CRSC_A_INTERIOR),
                    ("CMMD_A", PV.CMMD_A), ("MASKED_PURITY_A", PV.MASKED_PURITY_A),
                    ("PB_IOU", PV.PB_IOU)):
        assert set(d) == set(CONDITION_ORDER), f"{name} has incomplete conditions"

    assert PV.MASK_FREE_PREFERENCE["n_pairs"] == 304
    assert PV.PB_IOU_META["annotations"] == 80
    assert abs(PV.MASK_FREE_PREFERENCE["n_success"] / 304 - 0.586) < 0.002


def test_figures_low_dpi():
    import figures
    outs = figures.generate_all(dpi=40)
    assert len(outs) == 5
    for out in outs:
        assert out.exists() and out.stat().st_size > 0


# ============================================================
def main():
    print("=" * 72)
    print("smoke self-test - reporting-boundary reproduction")
    print("=" * 72)

    # the target masks must exist on disk for every load_mask() consumer
    from masks import save_all_masks
    save_all_masks()
    print("-" * 72)

    for name, fn in [
        ("mask geometry (manuscript Section 3.1)", test_mask_geometry),
        ("patch grid and mask->grid mapping", test_patch_grid),
        ("soft probability IoU", test_soft_iou),
        ("TMA-IoU_C directions and variants", test_tma_iou_c),
        ("signed distance and overlap masks (H/I)", test_signed_distance_and_overlap),
        ("PRSC / CRSC interior / noise floor", test_prsc_and_crsc),
        ("transition width and base contribution", test_transition_and_base),
        ("non-task-mask references", test_non_task_mask),
        ("core negative controls", test_negative_controls),
        ("human-study schedules", test_human_study),
        ("reporting boundary and design sensitivity", test_statistics_boundary),
        ("classifier validation helpers", test_classifier_metrics),
        ("paper_values internal consistency", test_paper_values_consistency),
        ("figures 1-5 (low dpi)", test_figures_low_dpi),
    ]:
        check(name, fn)

    print("-" * 72)
    print(f"passed {len(PASS)}   failed {len(FAIL)}")
    if FAIL:
        for name, exc in FAIL:
            print(f"  FAILED: {name} -> {exc}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
