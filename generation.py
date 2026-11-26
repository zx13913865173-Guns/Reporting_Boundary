# -*- coding: utf-8 -*-
"""
3.3 / 3.4 / 3.5  Regional generation, RLIMC compositing and the condition matrix.

RLIMC (Region-Localized Inpainting with Masked Compositing) works in two
stages:

    stage 1  generate region A and region B outputs in pixel space by two
             region-localised inpainting passes, each with its dedicated
             style LoRA; the region-A output is produced BEFORE any cross-region
             compositing (this pre-composite output is what PRSC is computed
             on);
    stage 2  composite the two pixel-space outputs using a mask specification.

Compositing specifications:
    hard mask        composite = out_A where M_A else out_B
    overlap (H, I)   M_A,gen = D > -delta, M_B,gen = D < delta, and
                     composite = alpha * out_A + (1 - alpha) * out_B with the
                     alpha ramp of masks.overlap_generation_masks

Generation settings follow the manuscript: 30 inference steps, CFG 7.5,
DPM++ 2M Karras, 1024 x 1024, fixed inpainting strength 0.80, 20 seeds.

NOTE (recorded in docs/REPRODUCTION_NOTES.md): the manuscript places G0.5,
D0.5, G1.0 and D1.0 in the same hard-mask subgroup and states that D1.0 belongs
to the "D dual-LoRA subgroup", but does not spell out the exact construction
difference between the G and D passes. The implementation below realises

    G*  single generation pass, no LoRA switching, hard-mask compositing
    D*  two region passes, each with its region's dedicated LoRA (dual-LoRA),
        hard-mask compositing

with the LoRA adapter weight set to the condition dose (0.5 or 1.0) so that the
G0.5 vs. D0.5 comparison is target-dose-matched. This is the only reading that
makes "dual-LoRA subgroup" literal; it must be confirmed against the authors'
released implementation.
"""

from pathlib import Path

import numpy as np
from PIL import Image

from config import (
    GEN_CONFIG, SEEDS, STYLES, STYLE_A, STYLE_B, IMG_SIZE,
    LORA_CONFIG, OVERLAP_DELTA, CONDITION_SETUP, GEN_DIR, CKPT_DIR,
    NON_CONSTRUCTION_CONDITIONS,
)
from masks import (
    load_mask, build_mask, signed_distance, overlap_generation_masks,
)


# ============================================================
# Pipeline
# ============================================================
def load_pipeline(base_model: str = None, lora_paths: dict = None,
                  device: str = None):
    """
    Build the SDXL inpainting pipeline (DPM++ 2M Karras) and attach the style
    LoRAs. ``lora_paths`` maps style name -> safetensors path.
    """
    import torch
    from diffusers import StableDiffusionXLInpaintPipeline, DPMSolverMultistepScheduler

    base_model = base_model or LORA_CONFIG["base_model"]
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if GEN_CONFIG["dtype"] == "float16" else torch.float32

    pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
        base_model, torch_dtype=dtype, use_safetensors=True)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config, use_karras_sigmas=True)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    if lora_paths:
        for style, path in lora_paths.items():
            if path and Path(path).exists():
                pipe.load_lora_weights(str(path), adapter_name=style)
        pipe.set_adapters([s for s in lora_paths if lora_paths[s]])
    return pipe


def set_lora_dose(pipe, style: str, dose: float):
    """Set the adapter weight of one style LoRA (the condition dose)."""
    if style is None:
        return
    try:
        pipe.set_adapters([style], adapter_weights=[float(dose)])
    except Exception as exc:      # adapter not loaded -> no-op for dry runs
        print(f"[generation] adapter {style} unavailable ({exc}); dose ignored")


# ============================================================
# Region-localised inpainting
# ============================================================
def generate_region(pipe, prompt: str, init_image, region_mask: np.ndarray,
                    seed: int, negative_prompt: str = "",
                    strength: float = None, steps: int = None,
                    guidance: float = None):
    """
    One region-localised inpainting pass.

    ``region_mask``: bool array, True where the pass writes the region. The mask
    can extend beyond the target region (the overlap variants pass
    M_A,gen / M_B,gen).
    """
    import torch

    strength = GEN_CONFIG["inpainting_strength"] if strength is None else strength
    steps = GEN_CONFIG["num_inference_steps"] if steps is None else steps
    guidance = GEN_CONFIG["guidance_scale"] if guidance is None else guidance

    gen = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
    gen.manual_seed(int(seed))

    mask_img = Image.fromarray((region_mask.astype(np.uint8) * 255), mode="L")
    out = pipe(prompt=prompt, negative_prompt=negative_prompt,
               image=init_image.convert("RGB"), mask_image=mask_img,
               height=IMG_SIZE, width=IMG_SIZE,
               num_inference_steps=steps, guidance_scale=guidance,
               strength=strength, generator=gen).images[0]
    return out


def region_passes(task: str, condition: str):
    """
    Return [(style, dose, generation_mask)] for the condition's passes.

    For hard-mask conditions the generation mask equals the target region; for
    the overlap conditions it is the extended mask of Section 3.4.
    """
    kind, dose_a, dose_b, gm = CONDITION_SETUP[condition]
    m_a = build_mask(task)

    if gm == "hard":
        return [(STYLE_A, dose_a, m_a), (STYLE_B, dose_b, ~m_a)]

    delta = OVERLAP_DELTA[condition]
    m_a_gen, m_b_gen, _ = overlap_generation_masks(task, delta)
    return [(STYLE_A, dose_a, m_a_gen), (STYLE_B, dose_b, m_b_gen)]


# ============================================================
# Compositing
# ============================================================
def compose_hard(out_a: Image.Image, out_b: Image.Image, task: str) -> Image.Image:
    """Hard-mask compositing: region A from out_A, region B from out_B."""
    m_a = load_mask(task)
    a = np.array(out_a.convert("RGB"))
    b = np.array(out_b.convert("RGB"))
    m3 = np.repeat(m_a[:, :, None], 3, axis=2)
    return Image.fromarray(np.where(m3, a, b).astype(np.uint8))


def compose_overlap(out_a: Image.Image, out_b: Image.Image, task: str,
                    delta: float) -> Image.Image:
    """Overlap compositing with the alpha ramp inside the overlap band."""
    _, _, alpha = overlap_generation_masks(task, delta)
    a = np.array(out_a.convert("RGB")).astype(np.float32)
    b = np.array(out_b.convert("RGB")).astype(np.float32)
    al = alpha[:, :, None].astype(np.float32)
    return Image.fromarray(np.clip(al * a + (1.0 - al) * b, 0, 255).astype(np.uint8))


def compose(out_a: Image.Image, out_b: Image.Image, task: str,
            condition: str) -> Image.Image:
    if CONDITION_SETUP.get(condition, (None,))[3] == "hard":
        return compose_hard(out_a, out_b, task)
    return compose_overlap(out_a, out_b, task, OVERLAP_DELTA[condition])


# ============================================================
# Non-construction references
# ============================================================
def generate_no_control(pipe, prompt: str, base_image: Image.Image,
                        seed: int) -> Image.Image:
    """Condition A: no LoRA, no mask - plain text-to-image initialisation."""
    set_lora_dose(pipe, STYLE_A, 0.0)
    set_lora_dose(pipe, STYLE_B, 0.0)
    full = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)
    return generate_region(pipe, prompt, base_image, full, seed)


def generate_global_lora(pipe, prompt: str, base_image: Image.Image,
                         seed: int, dose: float = 1.0) -> Image.Image:
    """Condition B: Full-image Global LoRA, no mask."""
    set_lora_dose(pipe, STYLE_A, dose)
    set_lora_dose(pipe, STYLE_B, dose)
    full = np.ones((IMG_SIZE, IMG_SIZE), dtype=bool)
    return generate_region(pipe, prompt, base_image, full, seed)


def condition_C(pipe, prompt: str, base_image: Image.Image, seed: int):
    """
    Condition C: third non-construction reference.

    NOTE: the manuscript lists A, B and C as "descriptive reference only" but
    only defines A and B. C is exposed here as a pluggable factory so that the
    pipeline stays complete without inventing a construction. It is never
    pooled with the six construction-defined conditions.
    """
    raise NotImplementedError(
        "Condition C is not defined in the manuscript. Provide an "
        "implementation here (e.g. attention-supervised, latent composition or "
        "prompt-layout generation) and record it in "
        "docs/REPRODUCTION_NOTES.md before use."
    )


def generate_mask_only(base_image: Image.Image, task: str) -> Image.Image:
    """Mask-only diagnostic: paste the target mask geometry, no generation."""
    m_a = load_mask(task)
    arr = np.array(base_image.convert("RGB"))
    overlay = arr.copy()
    overlay[m_a] = np.clip(arr[m_a].astype(np.int16) + 40, 0, 255).astype(np.uint8)
    return Image.fromarray(overlay)


# ============================================================
# One (task, condition, seed) unit
# ============================================================
def generate_unit(task: str, condition: str, seed: int, prompt: str,
                  init_image: Image.Image, pipe=None, dry_run: bool = False,
                  save: bool = True) -> dict:
    """
    Generate one (task, condition, seed) unit and return the image bundle.

    Returned keys:
        region_A, region_B   pre-composite region outputs (PRSC domain)
        composite            final composited image (TMA-IoU_C, CRSC domain)
        generation_masks     the per-pass generation masks actually used
    """
    kind, dose_a, dose_b, gm = CONDITION_SETUP[condition]
    passes = region_passes(task, condition)

    if dry_run:
        out_a = _placeholder(task, "A", seed)
        out_b = _placeholder(task, "B", seed)
    else:
        if pipe is None:
            raise ValueError("a pipeline is required unless dry_run=True")
        # G*: single pass without LoRA switching. D*: one pass per region with
        # its own LoRA (dual-LoRA subgroup).
        if kind == "dual_lora":
            set_lora_dose(pipe, STYLE_A, dose_a)
            out_a = generate_region(pipe, prompt, init_image, passes[0][2], seed)
            set_lora_dose(pipe, STYLE_B, dose_b)
            out_b = generate_region(pipe, prompt, init_image, passes[1][2], seed)
        else:
            set_lora_dose(pipe, STYLE_A, dose_a)
            out_a = generate_region(pipe, prompt, init_image, passes[0][2], seed)
            out_b = generate_region(pipe, prompt, init_image, passes[1][2], seed)

    composite = compose(out_a, out_b, task, condition)

    bundle = {
        "region_A": out_a, "region_B": out_b, "composite": composite,
        "generation_masks": [p[2] for p in passes],
        "task": task, "condition": condition, "seed": seed, "kind": kind,
    }
    if save:
        d = GEN_DIR / task / condition
        d.mkdir(parents=True, exist_ok=True)
        out_a.save(d / f"seed{seed:03d}_regionA.png")
        out_b.save(d / f"seed{seed:03d}_regionB.png")
        composite.save(d / f"seed{seed:03d}_composite.png")
    return bundle


def _placeholder(task: str, which: str, seed: int) -> Image.Image:
    """Deterministic placeholder used for dry runs and smoke tests."""
    rng = np.random.default_rng(abs(hash((task, which, seed))) % (2 ** 32))
    base = np.full((IMG_SIZE, IMG_SIZE, 3), 255 if which == "A" else 0, dtype=np.uint8)
    noise = rng.integers(0, 24, size=base.shape, dtype=np.uint8)
    return Image.fromarray(np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8))


# ============================================================
# Full condition matrix
# ============================================================
def build_condition_matrix(tasks, conditions=None, seeds=None, prompt_map=None,
                           dry_run: bool = True):
    """
    Yield the full generation schedule.

    6 conditions x N tasks x 20 seeds, plus the Mask-only diagnostic and the
    non-construction references A/B when requested.
    """
    conditions = conditions or list(CONDITION_SETUP.keys())
    seeds = seeds or SEEDS
    for task in tasks:
        for condition in conditions:
            for seed in seeds:
                yield dict(task=task, condition=condition, seed=seed,
                           prompt=(prompt_map or {}).get(condition, ""),
                           dry_run=dry_run)


def expected_unit_count(tasks, conditions=None, seeds=None) -> int:
    """6 x N_tasks x 20 units (manuscript: 6 x 8 x 20 = 960)."""
    conditions = conditions or list(CONDITION_SETUP.keys())
    seeds = seeds or SEEDS
    return len(tasks) * len(conditions) * len(seeds)


if __name__ == "__main__":
    n = expected_unit_count(["horizontal_split", "vertical_split",
                             "diagonal_45", "diagonal_135", "center_circle",
                             "corner_quadrant", "irregular_sine", "frame_border"])
    print(f"condition-matrix units (8 tasks x 6 conditions x 20 seeds) = {n}")
    print("non-construction references:", NON_CONSTRUCTION_CONDITIONS)
