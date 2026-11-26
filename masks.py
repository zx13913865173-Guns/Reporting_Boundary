# -*- coding: utf-8 -*-
"""
3.1 / 3.4  Task-specified target masks and overlap-generation geometry.

Target masks are generated parametrically from task specifications, NOT traced
by annotators. Region A carries style A (ink-wash); region B carries style B
(cyberpunk).

Geometry follows the CURRENT manuscript (Section 3.1). See config.py for the
LEDGER entries describing where the earlier released implementation differed.

Also provides:
    * the signed distance field D to the target boundary (positive inside A),
      used by the H/I overlap-generation variants (Section 3.4);
    * the mask -> classifier patch-grid mappings (patch-centre sampling and
      patch-area coverage), which the manuscript leaves unspecified and which
      therefore ship as primary + sensitivity variants (Section 3.6.1).
"""

import numpy as np
from PIL import Image

from config import (
    IMG_SIZE, CENTER, TASKS, TASK_LABELS, SINE_AMPLITUDE, FRAME_THICKNESS,
    MASK_DIR, CLASSIFIER_CONFIG, MASK_TO_GRID_PRIMARY,
)


# ============================================================
# Target masks
# ============================================================
def build_mask(task: str, n: int = IMG_SIZE) -> np.ndarray:
    """
    Return the binary region-A mask for ``task`` as a bool array.

    Region A = True (style A, ink-wash), region B = False (style B, cyberpunk).
    """
    xx, yy = np.meshgrid(np.arange(n), np.arange(n))
    c = n // 2

    if task == "horizontal_split":
        # manuscript: boundary y = 512  ->  region A is y < 512
        m = (yy < c)
    elif task == "vertical_split":
        # manuscript: boundary x = 512  ->  region A is x < 512
        m = (xx < c)
    elif task == "diagonal_45":
        # manuscript: boundary y = x
        m = (yy < xx)
    elif task == "diagonal_135":
        # manuscript: boundary y = 1024 - x
        m = (yy < (n - 1 - xx))
    elif task == "center_circle":
        # manuscript: centre (512, 512), radius 256
        m = ((xx - c) ** 2 + (yy - c) ** 2) <= (n // 4) ** 2
    elif task == "corner_quadrant":
        # manuscript: top-left quadrant isolation
        m = (xx < c) & (yy < c)
    elif task == "irregular_sine":
        # manuscript: y = 512 + 128 sin(2 pi x / 1024)
        boundary = c + SINE_AMPLITUDE * np.sin(2.0 * np.pi * xx / float(n))
        m = (yy < boundary)
    elif task == "frame_border":
        # manuscript: 20 px outer frame
        t = FRAME_THICKNESS
        m = np.zeros((n, n), dtype=bool)
        m[:t, :] = True
        m[-t:, :] = True
        m[:, :t] = True
        m[:, -t:] = True
    else:
        raise ValueError(f"Unknown task: {task}")

    return m


def build_mask_pair(task: str, n: int = IMG_SIZE):
    """Return (M_A, M_B) as bool arrays with M_A | M_B == all pixels."""
    m_a = build_mask(task, n)
    return m_a, ~m_a


def save_all_masks():
    """Write all 8 target masks to MASK_DIR as 0/255 PNGs (255 = region A)."""
    paths = {}
    for task in TASKS:
        arr = (build_mask(task).astype(np.uint8) * 255)
        out = MASK_DIR / f"{task}.png"
        Image.fromarray(arr).save(out)
        paths[task] = out
        print(f"[masks] {task:18s} -> {out.name}")
    return paths


def load_mask(task: str) -> np.ndarray:
    """Load the saved target mask as a bool array (True = region A)."""
    return np.array(Image.open(MASK_DIR / f"{task}.png")) == 255


def area_composition(task: str) -> float:
    """Fraction of pixels belonging to region A (style A, ink-wash)."""
    return float(build_mask(task).mean())


# ============================================================
# 3.4  Signed distance to the target boundary
# ============================================================
def _edge(mask: np.ndarray) -> np.ndarray:
    """Inner boundary of a binary mask (4-neighbourhood)."""
    er = (mask
          & np.roll(mask, 1, 0) & np.roll(mask, -1, 0)
          & np.roll(mask, 1, 1) & np.roll(mask, -1, 1))
    return mask & ~er


def signed_distance(task: str, n: int = IMG_SIZE) -> np.ndarray:
    """
    Signed distance field D to the target boundary, positive inside region A.

    D is used by the overlap-generation variants (Section 3.4):
        M_A,gen = (D > -delta),   M_B,gen = (D < delta)
    """
    from scipy.ndimage import distance_transform_edt

    m_a = build_mask(task, n)
    boundary = _edge(m_a) | _edge(~m_a)
    d = distance_transform_edt(~boundary).astype(np.float32)
    return np.where(m_a, d, -d).astype(np.float32)


def boundary_pixel_mask(task: str, n: int = IMG_SIZE) -> np.ndarray:
    """Pixels lying on the target boundary."""
    m_a = build_mask(task, n)
    return _edge(m_a) | _edge(~m_a)


# ============================================================
# Overlap-generation variants (H, I)
# ============================================================
def overlap_generation_masks(task: str, delta: float, n: int = IMG_SIZE):
    """
    Return (M_A_gen, M_B_gen, alpha) for an overlap-generation condition.

    M_A_gen = D > -delta, M_B_gen = D < delta (both passes generate inside the
    overlap band). ``alpha`` blends the two generated results inside the band,
    with alpha = 1 for pure A and alpha = 0 for pure B.
    """
    d = signed_distance(task, n)
    m_a_gen = d > -delta
    m_b_gen = d < delta
    alpha = np.clip((d + delta) / (2.0 * delta), 0.0, 1.0).astype(np.float32)
    return m_a_gen, m_b_gen, alpha


def overlap_band(task: str, delta: float, n: int = IMG_SIZE) -> np.ndarray:
    """Pixels inside the overlap band (|D| < delta)."""
    return np.abs(signed_distance(task, n)) < delta


# ============================================================
# 3.6.1  Mask -> classifier patch grid
# ============================================================
def patch_coords(n: int = IMG_SIZE, patch: int = None, stride: int = None):
    """
    Top-left coordinates of every classifier patch on the sliding grid,
    in row-major order (y outer, x inner).
    """
    patch = patch or CLASSIFIER_CONFIG["patch_size"]
    stride = stride or CLASSIFIER_CONFIG["stride"]
    return [(x, y)
            for y in range(0, n - patch + 1, stride)
            for x in range(0, n - patch + 1, stride)]


def grid_shape(n: int = IMG_SIZE, patch: int = None, stride: int = None):
    patch = patch or CLASSIFIER_CONFIG["patch_size"]
    stride = stride or CLASSIFIER_CONFIG["stride"]
    k = (n - patch) // stride + 1
    return k, k


def mask_to_grid(mask: np.ndarray, rule: str = None,
                 patch: int = None, stride: int = None) -> np.ndarray:
    """
    Map a pixel-level binary mask to the classifier patch grid.

    rule = "patch_center" : value of the mask at the patch centre  -> {0, 1}
    rule = "patch_area"   : fraction of mask pixels inside the patch -> [0, 1]

    NOTE: Section 3.6.1 states that the manuscript does not specify which of
    these two rules is used and that the exact rule should be taken from the
    released implementation. ``MASK_TO_GRID_PRIMARY`` in config.py selects the
    primary rule used for the reported TMA-IoU_C values; the other rule is a
    sensitivity variant. Returns a (k, k) float array.
    """
    rule = rule or MASK_TO_GRID_PRIMARY
    patch = patch or CLASSIFIER_CONFIG["patch_size"]
    stride = stride or CLASSIFIER_CONFIG["stride"]
    k, _ = grid_shape(mask.shape[0], patch, stride)
    half = patch // 2

    out = np.zeros((k, k), dtype=np.float32)
    i = 0
    for y in range(0, mask.shape[0] - patch + 1, stride):
        for x in range(0, mask.shape[1] - patch + 1, stride):
            if rule == "patch_center":
                out[i // k, i % k] = float(mask[y + half, x + half])
            elif rule == "patch_area":
                out[i // k, i % k] = float(mask[y:y + patch, x:x + patch].mean())
            else:
                raise ValueError(f"Unknown grid rule: {rule}")
            i += 1
    return out


def grid_to_pixel(grid: np.ndarray, n: int = IMG_SIZE,
                  patch: int = None, stride: int = None,
                  thresh: float = 0.5) -> np.ndarray:
    """
    Back-project a patch-grid field to a pixel map by averaging overlapping
    patch contributions (used for transition-width and region visualisation).
    """
    patch = patch or CLASSIFIER_CONFIG["patch_size"]
    stride = stride or CLASSIFIER_CONFIG["stride"]
    acc = np.zeros((n, n), dtype=np.float32)
    cnt = np.zeros((n, n), dtype=np.float32)
    k = grid.shape[0]
    i = 0
    for y in range(0, n - patch + 1, stride):
        for x in range(0, n - patch + 1, stride):
            acc[y:y + patch, x:x + patch] += float(grid[i // k, i % k])
            cnt[y:y + patch, x:x + patch] += 1.0
            i += 1
    pix = acc / np.maximum(cnt, 1.0)
    return (pix >= thresh) if thresh is not None else pix


def erode_px(mask: np.ndarray, px: int) -> np.ndarray:
    """Binary erosion by a square structuring element of side 2*px + 1."""
    from scipy.ndimage import binary_erosion

    if px <= 0:
        return mask.copy()
    st = np.ones((2 * px + 1, 2 * px + 1), dtype=bool)
    return binary_erosion(mask, structure=st, border_value=0)


def dilate_px(mask: np.ndarray, px: int) -> np.ndarray:
    """Binary dilation by a square structuring element of side 2*px + 1."""
    from scipy.ndimage import binary_dilation

    if px <= 0:
        return mask.copy()
    st = np.ones((2 * px + 1, 2 * px + 1), dtype=bool)
    return binary_dilation(mask, structure=st, border_value=0)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    save_all_masks()
    print("\nArea composition (region A fraction):")
    for t in TASKS:
        print(f"  {t:18s} {area_composition(t) * 100:6.2f}%  "
              f"[{TASK_LABELS[t]}]")
    k, _ = grid_shape()
    print(f"\nPatch grid: {k} x {k} patches "
          f"(patch {CLASSIFIER_CONFIG['patch_size']}, "
          f"stride {CLASSIFIER_CONFIG['stride']})")
    print(f"Primary mask->grid rule: {MASK_TO_GRID_PRIMARY}")
