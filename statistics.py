# -*- coding: utf-8 -*-
"""
3.8  Descriptive analysis with an explicit reporting boundary.

The manuscript reports NO condition-level SRCC, NO p-values, NO confidence
intervals or Fisher-transformed intervals, and NO mixed-effects models. This
module therefore does three things and deliberately refuses a fourth:

    1. builds the descriptive condition-level tables that Section 4.3 reports;
    2. summarises the descriptive preference proportions (Section 3.7);
    3. computes the DESIGN-SENSITIVITY calculation of the Limitations section
       (a hypothetical one-sided exact binomial test, used to characterise the
       sensitivity of the design, not to derive a p-value);
    4. refuses to compute rank correlations, p-values, confidence intervals,
       Fisher-transformed intervals or mixed-effects models - the guard
       functions below raise rather than silently producing them.
"""

import json

import numpy as np
import pandas as pd

from config import (
    METRICS_DIR, CONDITION_ORDER, DESIGN_SENSITIVITY,
    REPORT_SRCC, REPORT_P_VALUES, REPORT_CONFIDENCE_INTERVALS,
    REPORT_MIXED_EFFECTS,
)
import paper_values as PV


# ============================================================
# Guards - the reporting boundary is enforced, not merely documented
# ============================================================
def _refuse(feature: str, reason: str):
    raise NotImplementedError(f"{feature} is OUT of scope for this manuscript: {reason}")


def spearman_rank_correlation(*_a, **_k):
    _refuse("Condition-level SRCC",
            "no common ordering is defined across the six conditions, so no "
            "condition-level rank correlation is licensed.")


def p_value(*_a, **_k):
    _refuse("p-values",
            "the preference proportions are descriptive summaries of the "
            "observed judgment set; no sampling model is assumed.")


def confidence_interval(*_a, **_k):
    _refuse("confidence / Fisher-transformed intervals",
            "no population-level estimate is claimed.")


def mixed_effects_model(*_a, **_k):
    _refuse("mixed-effects models",
            "the design is reported descriptively; no model-based inference "
            "is claimed.")


def assert_reporting_boundary() -> dict:
    """Return the boundary flags and confirm they are all off."""
    flags = {
        "report_srcc": REPORT_SRCC,
        "report_p_values": REPORT_P_VALUES,
        "report_confidence_intervals": REPORT_CONFIDENCE_INTERVALS,
        "report_mixed_effects": REPORT_MIXED_EFFECTS,
    }
    if any(flags.values()):
        raise AssertionError(f"reporting boundary violated: {flags}")
    return flags


# ============================================================
# 4.3  Descriptive condition-level table
# ============================================================
def condition_table() -> pd.DataFrame:
    """
    The complete condition-level descriptive values of Section 4.3.

    TMA-IoU_C is the construction-consistency diagnostic; PRSC, CRSC interior,
    CMMD and masked purity are mask-defined descriptive metrics. No column is a
    quality anchor and no rank correlation between columns is computed.
    """
    df = pd.DataFrame({
        "TMA_IoU_C": [PV.TMA_IOU_C[c] for c in CONDITION_ORDER],
        "PRSC_A": [PV.PRSC_A[c] for c in CONDITION_ORDER],
        "CRSC_A_interior": [PV.CRSC_A_INTERIOR[c] for c in CONDITION_ORDER],
        "CMMD_A": [PV.CMMD_A[c] for c in CONDITION_ORDER],
        "masked_purity_A": [PV.MASKED_PURITY_A[c] for c in CONDITION_ORDER],
        "PB_IoU": [PV.PB_IOU[c] for c in CONDITION_ORDER],
    }, index=CONDITION_ORDER)
    df.index.name = "condition"
    return df


def write_condition_table(path=None) -> pd.DataFrame:
    df = condition_table()
    path = path or (METRICS_DIR / "condition_level_descriptive.csv")
    df.to_csv(path)
    print(f"[statistics] condition-level table -> {path}")
    return df


def direction_notes() -> pd.DataFrame:
    """
    Per-metric direction labels. Lower-is-better applies to CRSC interior,
    CMMD and boundary-band leakage; higher-is-better to PRSC, masked purity,
    TMA-IoU_C and PB-IoU. These labels are for reading the profile figures and
    do NOT define a common ranking.
    """
    back = {c: PV.CRSC_A_EXTENDED[i] for i, c in enumerate(PV.CRSC_EXTENDED_ORDER)}
    return pd.DataFrame({
        "metric": ["TMA_IoU_C", "PRSC_A", "CRSC_A_interior", "CMMD_A",
                   "masked_purity_A", "PB_IoU"],
        "direction": ["higher-is-better", "higher-is-better",
                      "lower-is-better", "lower-is-better",
                      "higher-is-better", "higher-is-better"],
        "family": ["construction-consistency diagnostic", "mask-defined",
                   "mask-defined", "mask-defined", "mask-defined",
                   "non-task-mask reference"],
    })


# ============================================================
# Limitations  Design sensitivity
# ============================================================
def design_sensitivity(n: int = None, critical: int = None,
                       p1_grid=None, alpha: float = None) -> pd.DataFrame:
    """
    Power of a hypothetical one-sided exact binomial test.

    This characterises the sensitivity of the preference design under
    pre-specified assumptions. It is NOT a test of the observed data and no
    p-value is derived from it.
    """
    from scipy.stats import binom

    n = n or DESIGN_SENSITIVITY["n"]
    critical = critical or DESIGN_SENSITIVITY["critical_success_count"]
    p1_grid = p1_grid or DESIGN_SENSITIVITY["p1_grid"]
    alpha = alpha or DESIGN_SENSITIVITY["alpha"]

    rows = []
    for p1 in p1_grid:
        # power = P(X >= critical | X ~ Binomial(n, p1)) for a one-sided test
        power = float(binom.sf(critical - 1, n, p1))
        rows.append(dict(n=n, alpha=alpha, p1=p1,
                         critical_success_count=critical,
                         power=power))
    return pd.DataFrame(rows)


# ============================================================
# Negative-control summary
# ============================================================
def negative_control_table() -> pd.DataFrame:
    """Mean TMA-IoU_C decrease per condition x control (manuscript Table 1)."""
    data = PV.NEG_CONTROL_DECREASE
    return pd.DataFrame(data, index=list(data["mask_swap"].keys())).T


# ============================================================
# Bundle
# ============================================================
def run_all(out_dir=None) -> dict:
    out_dir = out_dir or METRICS_DIR
    flags = assert_reporting_boundary()

    bundle = {
        "reporting_boundary": flags,
        "condition_table": condition_table().reset_index().to_dict("records"),
        "negative_controls": negative_control_table().to_dict(),
        "design_sensitivity": design_sensitivity().to_dict("records"),
        "preference": {
            "mask_free": PV.MASK_FREE_PREFERENCE,
            "overall": PV.OVERALL_PREFERENCE,
            "forced_compliance": PV.FORCED_COMPLIANCE_PREFERENCE,
        },
        "pb_iou": PV.PB_IOU,
    }
    (out_dir / "descriptive_bundle.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8")
    return bundle


if __name__ == "__main__":
    print(condition_table().to_string())
    print()
    print(negative_control_table().to_string())
    print()
    print(design_sensitivity().to_string(index=False))
