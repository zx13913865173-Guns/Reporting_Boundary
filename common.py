# -*- coding: utf-8 -*-
"""
Shared utilities for the metric families.

    (i)   construction-consistency diagnostic : metrics/tma_iou.py
    (ii)  mask-defined descriptive metrics     : metrics/mask_defined.py
    (iii) non-task-mask references             : metrics/non_task_mask.py

Nothing in this module treats a mask-defined quantity as a non-mask anchor.
"""

import numpy as np

from config import (
    CLASSIFIER_CONFIG, STYLE_REF_IMAGES, CMMD_KERNEL_SIGMA_SCALE, REGION_FILL,
)


# ============================================================
# Soft-probability IoU (Section 3.6.1)
# ============================================================
def soft_iou(p: np.ndarray, m: np.ndarray, valid: np.ndarray = None) -> float:
    """
    Direction-specific soft-probability IoU on the classifier patch grid:

        IoU_soft = sum_i min(p_i, m_i) / sum_i max(p_i, m_i)

    ``p`` is the calibrated probability of the direction's target style and
    ``m`` is the task mask on the same patch grid. ``valid`` optionally
    restricts the summation (used by the uncertain-excluded variant).
    """
    p = np.asarray(p, dtype=np.float64)
    m = np.asarray(m, dtype=np.float64)
    if valid is not None:
        v = np.asarray(valid, dtype=bool)
        p, m = p[v], m[v]
    if p.size == 0:
        return float("nan")
    denom = np.maximum(p, m).sum()
    if denom <= 0:
        return float("nan")
    return float(np.minimum(p, m).sum() / denom)


def uncertain_grid(prob_grid: np.ndarray) -> np.ndarray:
    """Patches whose calibrated probability lies in the uncertain band."""
    lo, hi = CLASSIFIER_CONFIG["uncertain_lo"], CLASSIFIER_CONFIG["uncertain_hi"]
    return (prob_grid >= lo) & (prob_grid <= hi)


# ============================================================
# Distance / band helpers
# ============================================================
def mean_absolute_boundary_distance(pred_boundary: np.ndarray,
                                   ref_boundary: np.ndarray) -> float:
    """Mean absolute distance (px) from a predicted boundary to a reference."""
    from scipy.ndimage import distance_transform_edt

    if pred_boundary.sum() == 0 or ref_boundary.sum() == 0:
        return float("nan")
    dt = distance_transform_edt(~ref_boundary)
    return float(dt[pred_boundary].mean())


def boundary_f1(pred_boundary: np.ndarray, ref_boundary: np.ndarray,
                tol_px: int = 4) -> float:
    """Boundary F1 with a tolerance band (inter-annotator agreement)."""
    from scipy.ndimage import binary_dilation

    st = np.ones((2 * tol_px + 1, 2 * tol_px + 1), dtype=bool)
    ref_d = binary_dilation(ref_boundary, structure=st)
    pred_d = binary_dilation(pred_boundary, structure=st)

    tp = float((pred_boundary & ref_d).sum())
    fp = float((pred_boundary & ~ref_d).sum())
    fn = float((ref_boundary & ~pred_d).sum())
    if tp == 0:
        return 0.0
    prec, rec = tp / (tp + fp), tp / (tp + fn)
    return float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0


# ============================================================
# Pixel-level IoU (PB-IoU uses region partitions, not the task mask)
# ============================================================
def region_iou(pred_region: np.ndarray, ref_region: np.ndarray) -> float:
    """Pixel-level IoU between two binary region partitions."""
    pred_region = np.asarray(pred_region, dtype=bool)
    ref_region = np.asarray(ref_region, dtype=bool)
    union = np.logical_or(pred_region, ref_region).sum()
    if union == 0:
        return float("nan")
    return float(np.logical_and(pred_region, ref_region).sum() / union)


# ============================================================
# CLIP embeddings, masked style purity and CMMD
# ============================================================
_CLIP_CACHE = {}


def load_clip(device: str = None):
    """
    Load CLIP ViT-B/32 lazily. Returns (model, preprocess, device).

    The manuscript uses CLIP embeddings for masked style purity and for CMMD
    (Section 3.6.4). Raises a clear error when the optional dependency is
    absent so that GPU-free runs can skip those two metrics explicitly.
    """
    if "clip" in _CLIP_CACHE:
        return _CLIP_CACHE["clip"]

    try:
        import torch
        import clip
    except ImportError as exc:      # pragma: no cover - environment dependent
        raise RuntimeError(
            "masked style purity and CMMD need the optional 'clip' + 'torch' "
            "dependencies (see requirements.txt). Set SKIP_CLIP=1 to skip "
            "these two metrics."
        ) from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    _CLIP_CACHE["clip"] = (model, preprocess, device)
    return _CLIP_CACHE["clip"]


def masked_region_image(image, mask_bool: np.ndarray, fill=REGION_FILL):
    """
    Return the masked region of ``image`` on a neutral fill.

    Section 4.6: purity and CMMD are computed on masked region representations
    with neutral fill; bounding-box crops are NOT used for non-rectangular
    regions.
    """
    from PIL import Image

    arr = np.array(image.convert("RGB"))
    out = np.full_like(arr, 127) if fill == "neutral" else np.zeros_like(arr)
    m3 = np.repeat(mask_bool[:, :, None], 3, axis=2)
    out = np.where(m3, arr, out)
    return Image.fromarray(out.astype(np.uint8))


def clip_embed_image(pil_image) -> np.ndarray:
    """L2-normalised CLIP image embedding as a 1-D float array."""
    import torch

    model, preprocess, device = load_clip()
    with torch.no_grad():
        x = preprocess(pil_image).unsqueeze(0).to(device)
        e = model.encode_image(x)
        e = e / e.norm(dim=-1, keepdim=True)
    return e.squeeze(0).cpu().numpy()


def clip_embed_reference_set(ref_paths) -> np.ndarray:
    """Stack the embeddings of a reference set into an (N, D) array."""
    embs = [clip_embed_image(_open_rgb(p)) for p in ref_paths]
    if not embs:
        raise ValueError("empty reference set")
    return np.vstack(embs)


def style_purity(region_pil, ref_paths) -> float:
    """
    Masked style purity: mean CLIP cosine similarity between the masked
    generated region and the STYLE_REF_IMAGES reference images of that style
    (Section 3.6.4).
    """
    refs = clip_embed_reference_set(ref_paths)
    if refs.shape[0] > STYLE_REF_IMAGES:
        refs = refs[:STYLE_REF_IMAGES]
    e = clip_embed_image(region_pil)
    return float(np.mean(refs @ e))


def mmd_rbf(X: np.ndarray, Y: np.ndarray, sigma: float = None) -> float:
    """Maximum mean discrepancy with a Gaussian RBF kernel."""
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if sigma is None:
        Z = np.vstack([X, Y])
        d = np.linalg.norm(Z[:, None, :] - Z[None, :, :], axis=-1)
        off = d[np.triu_indices_from(d, k=1)]
        sigma = float(np.median(off)) if off.size else 1.0
        sigma = max(sigma, 1e-6) * CMMD_KERNEL_SIGMA_SCALE

    def k(A, B):
        d2 = ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)
        return np.exp(-d2 / (2.0 * sigma ** 2))

    return float(k(X, X).mean() + k(Y, Y).mean() - 2.0 * k(X, Y).mean())


def cmmd(region_pil, ref_paths) -> float:
    """
    CMMD between the masked region representation and the reference set, using
    CLIP embeddings and MMD with a Gaussian RBF kernel (Section 3.6.4).

    Masked purity and CMMD both derive from CLIP embeddings; they are reported
    as separate descriptive diagnostics, never as mutually confirming validity
    tests.
    """
    refs = clip_embed_reference_set(ref_paths)
    if refs.shape[0] > STYLE_REF_IMAGES:
        refs = refs[:STYLE_REF_IMAGES]
    e = clip_embed_image(region_pil)[None, :]
    return mmd_rbf(e, refs)


def _open_rgb(p):
    from PIL import Image

    return Image.open(p).convert("RGB")
