from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable, Dict, Mapping

import numpy as np
from PIL import Image, ImageFilter

Sample = Mapping[str, Any]
PreprocessFn = Callable[[Sample], Image.Image]


# -----------------------------
# I/O helpers
# -----------------------------
def _load_image_from_sample(sample: Sample) -> Image.Image:
    """Load an image using sample['data_dir'] and sample['img_path']."""
    data_dir = sample.get("data_dir")
    img_path = sample.get("img_path")
    if not data_dir or not img_path:
        raise KeyError("sample must contain 'data_dir' and 'img_path'")

    full_path = Path(str(data_dir)) / str(img_path)
    with Image.open(full_path) as im:
        return im.copy()


# -----------------------------
# Image utilities
# -----------------------------
def _to_uint8_rgb(image: Image.Image) -> Image.Image:
    """Convert input image to uint8 RGB with min-max normalization (CXR-friendly)."""
    arr = np.array(image, dtype=np.float32)
    arr_min, arr_max = float(arr.min()), float(arr.max())
    if arr_max - arr_min > 0:
        arr = (arr - arr_min) / (arr_max - arr_min)
    else:
        arr = np.zeros_like(arr)
    arr = (arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr).convert("RGB")


def _clip_bbox(bbox: list[int], w: int, h: int) -> tuple[int, int, int, int] | None:
    """Clip [x1,y1,x2,y2] to image bounds. Returns None if invalid/empty."""
    if len(bbox) != 4:
        return None
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(int(x1), w))
    x2 = max(0, min(int(x2), w))
    y1 = max(0, min(int(y1), h))
    y2 = max(0, min(int(y2), h))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _pick_random_region_bbox(sample: Sample, w: int, h: int) -> tuple[int, int, int, int] | None:
    """Randomly select one bbox from sample['regions'] and clip it to image bounds."""
    regions = sample.get("regions") or []
    if not isinstance(regions, list) or len(regions) == 0:
        return None

    valid: list[tuple[int, int, int, int]] = []
    for r in regions:
        if not isinstance(r, dict):
            continue
        bbox = r.get("bbox")
        if isinstance(bbox, list) and len(bbox) == 4:
            clipped = _clip_bbox(bbox, w, h)
            if clipped is not None:
                valid.append(clipped)

    if not valid:
        return None

    idx = int(np.random.randint(0, len(valid)))
    return valid[idx]


def _robust_center_scale_rgb(
    pixels_rgb: np.ndarray,
    *,
    low_q: float = 1.0,
    high_q: float = 90.0,
    min_std: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute robust per-channel center (median) and scale (std) after trimming outliers.

    pixels_rgb: array of shape [N,3]
    - Trim pixels outside [low_q, high_q] percentiles per channel
    - Center = median of trimmed pixels per channel
    - Scale = std of trimmed pixels per channel (clamped to min_std)
    """
    px = pixels_rgb.astype(np.float32)

    lo = np.percentile(px, low_q, axis=0)
    hi = np.percentile(px, high_q, axis=0)

    keep = np.ones((px.shape[0],), dtype=bool)
    keep &= (px[:, 0] >= lo[0]) & (px[:, 0] <= hi[0])
    keep &= (px[:, 1] >= lo[1]) & (px[:, 1] <= hi[1])
    keep &= (px[:, 2] >= lo[2]) & (px[:, 2] <= hi[2])

    trimmed = px[keep]
    if trimmed.shape[0] < 32:
        trimmed = px

    center = np.median(trimmed, axis=0).astype(np.float32)
    scale = trimmed.std(axis=0).astype(np.float32)
    scale = np.maximum(scale, float(min_std))
    return center, scale


def _matched_correlated_noise_fill(
    img_arr: np.ndarray,
    mask: np.ndarray,
    blur_radius: float = 2.0,
) -> np.ndarray:
    """
    Build a correlated noise fill matching robust stats of pixels OUTSIDE the mask.

    - Robust center: median
    - Robust trimming: ignore pixels below 1st percentile and above 90th percentile
    - Scale: std computed on the trimmed set

    img_arr: uint8 RGB image, shape [H,W,3]
    mask: boolean mask, True where we will fill (occlude)
    Returns uint8 RGB fill image array, shape [H,W,3]
    """
    h, w, c = img_arr.shape
    assert c == 3

    outside = ~mask
    if outside.sum() < 32:
        outside = np.ones((h, w), dtype=bool)

    out_pixels = img_arr[outside]  # [N,3]
    center, scale = _robust_center_scale_rgb(out_pixels, low_q=1.0, high_q=90.0, min_std=1.0)

    noise = np.random.randn(h, w, 3).astype(np.float32) * scale + center
    noise = np.clip(noise, 0.0, 255.0).astype(np.uint8)

    noise_img = Image.fromarray(noise, mode="RGB").filter(ImageFilter.GaussianBlur(radius=blur_radius))
    return np.array(noise_img, dtype=np.uint8)


def _blend_region(
    img_arr: np.ndarray,
    fill_arr: np.ndarray,
    region_mask: np.ndarray,
    p: float,
    feather_radius: int = 6,
) -> np.ndarray:
    """Blend original and fill within region_mask using strength p, with feathered edges."""
    p = float(np.clip(p, 0.0, 1.0))
    if p <= 0.0:
        return img_arr

    base_alpha = region_mask.astype(np.float32) * p

    alpha_u8 = (base_alpha * 255.0).astype(np.uint8)
    alpha_img = Image.fromarray(alpha_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=float(feather_radius)))
    alpha = np.array(alpha_img, dtype=np.float32) / 255.0

    out = img_arr.astype(np.float32) * (1.0 - alpha[..., None]) + fill_arr.astype(np.float32) * alpha[..., None]
    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _parse_p_from_experiment(experiment: str) -> float | None:
    """Parse p from strings like ObjectClassOcclusion_p25 and return p in [0,1]."""
    m = re.fullmatch(r"ObjectClassOcclusion_p(\d{1,3})", experiment)
    if not m:
        return None
    p_int = max(0, min(int(m.group(1)), 100))
    return p_int / 100.0


# -----------------------------
# Preprocess implementations (sample-only)
# -----------------------------
def baseline(sample: Sample) -> Image.Image:
    """Baseline preprocessing: normalize to [0,1], convert to uint8, convert to RGB."""
    image = _load_image_from_sample(sample)
    return _to_uint8_rgb(image)


def all_noise(sample: Sample) -> Image.Image:
    """Replace all pixels with uniform random noise (same spatial size as the loaded image)."""
    image = _load_image_from_sample(sample)
    w, h = image.size
    noise = np.random.randint(0, 256, size=(h, w, 3), dtype=np.uint8)
    return Image.fromarray(noise, mode="RGB")


def all_noise_mean(sample: Sample) -> Image.Image:
    """
    Apply matched correlated noise to the entire image.
    Noise stats are robustly estimated from the image itself.
    """
    image = _to_uint8_rgb(_load_image_from_sample(sample))
    img_arr = np.array(image, dtype=np.uint8)
    h, w, _ = img_arr.shape

    mask = np.zeros((h, w), dtype=bool)
    fill = _matched_correlated_noise_fill(img_arr, mask, blur_radius=2.0)

    out = _blend_region(img_arr, fill, np.ones((h, w), dtype=bool), p=1.0, feather_radius=0)
    return Image.fromarray(out, mode="RGB")


def object_class_occlusion(sample: Sample, *, p: float) -> Image.Image:
    """
    Randomly pick one bbox from sample['regions'] and apply matched correlated noise within that bbox.
    """
    image = _to_uint8_rgb(_load_image_from_sample(sample))
    img_arr = np.array(image, dtype=np.uint8)
    h, w, _ = img_arr.shape

    bbox = _pick_random_region_bbox(sample, w=w, h=h)
    if bbox is None or p <= 0.0:
        return image

    x1, y1, x2, y2 = bbox
    region_mask = np.zeros((h, w), dtype=bool)
    region_mask[y1:y2, x1:x2] = True

    fill = _matched_correlated_noise_fill(img_arr, region_mask, blur_radius=2.0)
    out = _blend_region(img_arr, fill, region_mask, p=p, feather_radius=6)
    return Image.fromarray(out, mode="RGB")


# -----------------------------
# Registry + resolver
# -----------------------------
PREPROCESS: Dict[str, PreprocessFn] = {
    "baseline": baseline,
    "all_noise": all_noise,
    "all_noise_mean": all_noise_mean,
}


def _resolve_preprocess(experiment: str) -> PreprocessFn:
    """
    Resolve an experiment string into a preprocessing callable (sample-only).
    Supports:
      - baseline
      - all_noise
      - all_noise_mean
      - ObjectClassOcclusion_pXX  (XX in 0..100)
    """
    if experiment in PREPROCESS:
        return PREPROCESS[experiment]

    p = _parse_p_from_experiment(experiment)
    if p is not None:
        return lambda sample: object_class_occlusion(sample, p=p)

    raise KeyError(f"Preprocess key not found: {experiment}")
