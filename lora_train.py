# -*- coding: utf-8 -*-
"""
3.5  LoRA training (kohya-ss / sd-scripts wrapper).

One LoRA per style, trained on the manuscript's settings:

    rank 32, alpha 1.0, 3000 steps, learning rate 1e-4

Training images: 50 per style, screened by two domain experts on a 3-point
scale; only images rated 3 by BOTH experts are retained. Screening agreement is
reported with Cohen's kappa (kappa = 0.88 for the two-expert screening).

LEDGER: the released method-oriented draft shipped the same wrapper but with a
different parameter block; the hyper-parameters below follow the manuscript.
"""

import json
import subprocess
from pathlib import Path

import numpy as np

from config import (
    DATA_DIR, CKPT_DIR, LORA_CONFIG, STYLES,
    LORA_TRAIN_IMAGES_PER_STYLE, SCREENING_SCALE_MAX, SCREENING_COHENS_KAPPA,
)

KOHYA_TRAIN = "accelerate launch sd-scripts/train_network.py"


# ============================================================
# Training-image screening (two experts, 3-point scale)
# ============================================================
def screen_training_images(ratings_expert_1: dict,
                           ratings_expert_2: dict):
    """
    Keep images rated 3 (SCREENING_SCALE_MAX) by BOTH experts.

    ``ratings_expert_*``: {image_filename: rating in 1..3}
    Returns (kept, dropped, cohens_kappa_observed).
    """
    kept, dropped = [], []
    for name in sorted(set(ratings_expert_1) | set(ratings_expert_2)):
        r1 = ratings_expert_1.get(name)
        r2 = ratings_expert_2.get(name)
        if r1 == SCREENING_SCALE_MAX and r2 == SCREENING_SCALE_MAX:
            kept.append(name)
        else:
            dropped.append(name)
    return kept, dropped, cohens_kappa(ratings_expert_1, ratings_expert_2)


def cohens_kappa(a: dict, b: dict) -> float:
    """Cohen's kappa between two raters over their shared items."""
    labels = sorted(set(a.values()) | set(b.values()))
    idx = {v: i for i, v in enumerate(labels)}
    names = [n for n in sorted(set(a) & set(b)) if a[n] is not None and b[n] is not None]
    if not names:
        return float("nan")

    k = len(labels)
    obs = np.zeros((k, k), dtype=float)
    for n in names:
        obs[idx[a[n]], idx[b[n]]] += 1
    obs /= obs.sum()
    pe = float((obs.sum(0) * obs.sum(1)).sum())
    po = float(np.trace(obs))
    return (po - pe) / (1.0 - pe) if pe < 1.0 else float("nan")


# ============================================================
# LoRA training
# ============================================================
def train_lora(style: str, base_model: str = None, lr: float = None,
               steps: int = None, dry_run: bool = False):
    """Train one LoRA for ``style`` on the screened training set."""
    base_model = base_model or LORA_CONFIG["base_model"]
    lr = lr or LORA_CONFIG["lr"]
    steps = steps or LORA_CONFIG["steps"]

    train_dir = DATA_DIR / "train" / style
    out_dir = CKPT_DIR / style
    out_dir.mkdir(parents=True, exist_ok=True)

    if not train_dir.exists():
        raise FileNotFoundError(
            f"missing screened training directory: {train_dir}\n"
            f"expected about {LORA_TRAIN_IMAGES_PER_STYLE} images "
            f"(manuscript: 50 per style after screening)."
        )

    cmd = [
        *KOHYA_TRAIN.split(),
        f"--pretrained_model_name_or_path={base_model}",
        f"--train_data_dir={train_dir}",
        f"--output_dir={out_dir}",
        f"--output_name={style}",
        "--network_module=networks.lora",
        f"--network_dim={LORA_CONFIG['rank']}",
        f"--network_alpha={LORA_CONFIG['alpha']}",
        f"--max_train_steps={steps}",
        f"--learning_rate={lr}",
        f"--resolution={LORA_CONFIG['resolution']},{LORA_CONFIG['resolution']}",
        "--mixed_precision=fp16",
        "--save_model_as=safetensors",
        "--optimizer_type=AdamW8bit",
        "--lr_scheduler=cosine",
        "--seed=42",
    ]

    if dry_run:
        print("[lora_train] dry-run command:\n  " + " ".join(cmd))
        return out_dir / f"{style}.safetensors"

    print(f"[lora_train] training {style} "
          f"(rank {LORA_CONFIG['rank']}, alpha {LORA_CONFIG['alpha']}, "
          f"{steps} steps, lr {lr}) ...")
    subprocess.run(cmd, check=True)
    out = out_dir / f"{style}.safetensors"
    print(f"[lora_train] done: {out}")
    return out


def train_all(base_model: str = None, lr: float = None, dry_run: bool = False):
    """Train one LoRA per style."""
    return {s: train_lora(s, base_model, lr, dry_run=dry_run) for s in STYLES}


def write_screening_report(kept: dict, dropped: dict, kappa: float,
                           out_path: Path = None):
    """Persist the screening outcome for the repository record."""
    out_path = out_path or (CKPT_DIR / "screening_report.json")
    payload = {
        "kept_per_style": {k: len(v) for k, v in kept.items()},
        "dropped_per_style": {k: len(v) for k, v in dropped.items()},
        "cohens_kappa": kappa,
        "manuscript_kappa": SCREENING_COHENS_KAPPA,
    }
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[lora_train] screening report -> {out_path}")
    return out_path


if __name__ == "__main__":
    train_all()
