# -*- coding: utf-8 -*-
"""
3.6.1 / 4.12  Two-class style classifier for the patch grid.

ResNet-50 on 32 x 32 patches with stride 8, calibrated patch-level
probabilities, and an uncertain band of 0.4-0.6. Training set: 200 ink-wash and
200 cyberpunk patches, independent of the LoRA training sets.

DIRECTION CONVENTION (this is the part the released draft got wrong):
    ``predict_prob_map_A`` returns the calibrated probability of STYLE A
    (ink-wash), i.e. p_i^A. TMA-IoU_C for direction A must compare p_i^A with
    the style-A task mask m_i^A. The released draft took the level-set of
    P(cyberpunk) >= 0.5 and intersected it with the ink-wash (255) mask, which
    is the opposite direction and changes every IoU conclusion by construction.

Full validation metrics (accuracy, F1, AUC, ECE, mean uncertain, FNR, confusion
matrix) and the direction-specific FPRs are produced by ``validate``. Classifier
accuracy is reported ONLY to describe the residual classifier contribution to
TMA-IoU_C; it is not evidence that TMA-IoU_C is a valid quality metric.
"""

from pathlib import Path

import numpy as np

from config import (
    CLASSIFIER_CONFIG, DATA_DIR, CKPT_DIR, METRICS_DIR,
    CLASSIFIER_FPR, STYLE_A,
)

STYLE_INDEX_A = 0       # label 0 = style A (ink_wash)
STYLE_INDEX_B = 1       # label 1 = style B (cyberpunk)


# ============================================================
# Data
# ============================================================
def iter_patches(image_rgb: np.ndarray, patch: int = None, stride: int = None):
    """Yield (y, x, patch_array) over the sliding grid."""
    patch = patch or CLASSIFIER_CONFIG["patch_size"]
    stride = stride or CLASSIFIER_CONFIG["stride"]
    for y in range(0, image_rgb.shape[0] - patch + 1, stride):
        for x in range(0, image_rgb.shape[1] - patch + 1, stride):
            yield y, x, image_rgb[y:y + patch, x:x + patch]


def build_patch_dataset(images_by_style: dict):
    """
    Build (X, y) patch arrays from {style_name: [rgb arrays]}.

    Label 0 is style A and label 1 is style B, matching STYLE_INDEX_A / _B.
    """
    label = {STYLE_A: STYLE_INDEX_A}
    label.setdefault("cyberpunk", STYLE_INDEX_B)

    X, y = [], []
    for style, images in images_by_style.items():
        for img in images:
            for _, _, p in iter_patches(np.asarray(img)):
                X.append(p.astype(np.float32) / 127.5 - 1.0)
                y.append(label.get(style, STYLE_INDEX_B))
    return np.stack(X), np.asarray(y, dtype=np.int64)


# ============================================================
# Model
# ============================================================
def build_classifier(num_classes: int = 2):
    """ResNet-50 with a two-way head (lazy torch import)."""
    import torch.nn as nn
    from torchvision import models

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_classifier(images_by_style: dict, out_path: Path = None,
                     device: str = None):
    """Train the patch classifier on the 200 + 200 patch set."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X, y = build_patch_dataset(images_by_style)
    ds = TensorDataset(torch.from_numpy(X).permute(0, 3, 1, 2), torch.from_numpy(y))
    dl = DataLoader(ds, batch_size=CLASSIFIER_CONFIG["batch_size"], shuffle=True)

    model = build_classifier().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=CLASSIFIER_CONFIG["lr"])
    lossf = nn.CrossEntropyLoss()

    model.train()
    for ep in range(CLASSIFIER_CONFIG["epochs"]):
        tot, n = 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += float(loss) * xb.size(0)
            n += xb.size(0)
        print(f"[classifier] epoch {ep + 1:02d}/{CLASSIFIER_CONFIG['epochs']} "
              f"loss {tot / max(n, 1):.4f}")

    out_path = out_path or (CKPT_DIR / "style_classifier.pt")
    torch.save(model.state_dict(), out_path)
    print(f"[classifier] saved -> {out_path}")
    return out_path


def load_classifier(ckpt_path: Path = None, device: str = None):
    """Load the trained patch classifier (with its temperature if present)."""
    import torch

    ckpt_path = ckpt_path or (CKPT_DIR / "style_classifier.pt")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = build_classifier().to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    temp_path = Path(str(ckpt_path).replace(".pt", "_temperature.json"))
    temperature = 1.0
    if temp_path.exists():
        import json
        temperature = json.loads(temp_path.read_text(encoding="utf-8"))["temperature"]
    return model, temperature, device


# ============================================================
# Calibration and inference
# ============================================================
def calibrate_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """
    Single-parameter temperature scaling on a held-out split, minimising NLL.
    Returns the fitted temperature (1.0 means no rescaling).
    """
    from scipy.optimize import minimize_scalar

    def nll(t):
        z = logits / max(t, 1e-3)
        z = z - z.max(axis=1, keepdims=True)
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return -np.log(np.maximum(p[np.arange(len(labels)), labels], 1e-12)).mean()

    res = minimize_scalar(nll, bounds=(0.05, 10.0), method="bounded")
    return float(res.x)


def _softmax_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = logits / max(temperature, 1e-3)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_prob_map_A(image_rgb: np.ndarray, model=None, temperature: float = 1.0,
                       device: str = None):
    """
    Calibrated style-A probability map.

    Returns (prob_pixel_A, prob_grid_A):
        prob_pixel_A : (H, W) float, averaged over overlapping patches
        prob_grid_A  : (k, k) float, the classifier patch grid used by TMA-IoU_C

    Both are probabilities of STYLE A (ink-wash) as required by the direction
    convention above.
    """
    import torch

    from masks import grid_shape, grid_to_pixel

    model = model or load_classifier()[0]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    patch = CLASSIFIER_CONFIG["patch_size"]
    stride = CLASSIFIER_CONFIG["stride"]
    n = image_rgb.shape[0]
    k, _ = grid_shape(n, patch, stride)

    batch, coords = [], []
    for y, x, p in iter_patches(image_rgb):
        batch.append(p.astype(np.float32) / 127.5 - 1.0)
        coords.append((y, x))

    xb = torch.from_numpy(np.stack(batch)).permute(0, 3, 1, 2).to(device)
    with torch.no_grad():
        logits = model(xb).cpu().numpy()
    probs = _softmax_temperature(logits, temperature)[:, STYLE_INDEX_A]

    grid = np.zeros((k, k), dtype=np.float64)
    for i, (y, x) in enumerate(coords):
        grid[i // k, i % k] = probs[i]

    pixel = grid_to_pixel(grid, n=n, patch=patch, stride=stride, thresh=None)
    return pixel.astype(np.float64), grid


def uncertain_mask_grid(prob_grid: np.ndarray) -> np.ndarray:
    """Patches in the 0.4-0.6 uncertain band."""
    lo, hi = CLASSIFIER_CONFIG["uncertain_lo"], CLASSIFIER_CONFIG["uncertain_hi"]
    return (prob_grid >= lo) & (prob_grid <= hi)


# ============================================================
# 4.12  Validation
# ============================================================
def validate(prob_grid_A: np.ndarray, true_style_A_grid: np.ndarray,
             out_csv: Path = None) -> dict:
    """
    Full validation metrics on a labelled patch grid, plus the direction-specific
    false-positive rates used as the CRSC noise floor.
    """
    import pandas as pd

    p = np.asarray(prob_grid_A).ravel()
    y = np.asarray(true_style_A_grid).ravel().astype(int)
    pred = (p >= CLASSIFIER_CONFIG["threshold"]).astype(int)

    tp = int(((pred == 1) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    fnr = fn / max(fn + tp, 1)

    # AUC via rank statistic (no sklearn dependency).
    auc = _auc(p, y)

    # Expected calibration error with 10 equal-width bins.
    ece = _ece(p, y, bins=10)

    fpr_a_on_b = fp / max(fp + tn, 1)          # A called on B patches
    fpr_b_on_a = fn / max(fn + tp, 1)          # B called on A patches

    out = {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "auc": auc, "ece": ece, "fnr": fnr,
        "mean_uncertain":
            float(uncertain_mask_grid(np.asarray(prob_grid_A)).mean()),
        "FPR_A_on_B": fpr_a_on_b, "FPR_B_on_A": fpr_b_on_a,
        "confusion_matrix": {"TP": tp, "TN": tn, "FP": fp, "FN": fn},
        "manuscript_FPR": CLASSIFIER_FPR,
    }
    if out_csv:
        pd.DataFrame([{k: v for k, v in out.items()
                       if not isinstance(v, dict)}]).to_csv(out_csv, index=False)
    return out


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _ece(p: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece, n = 0.0, len(p)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1])
        if m.sum() == 0:
            continue
        conf = p[m].mean()
        acc = y[m].mean()
        ece += (m.sum() / n) * abs(conf - acc)
    return float(ece)


def default_dataset_dir(style: str) -> Path:
    return DATA_DIR / "classifier" / style
