"""
Render a standalone OCO-style noise patch for use in figures.

The patch replicates exactly the noise texture used in Object Class Occlusion (OCO):
  1. Load a real CXR and convert to uint8 RGB.
  2. Estimate robust per-channel stats (median + trimmed std) from the whole image.
  3. Generate i.i.d. Gaussian noise scaled to those stats.
  4. Apply a Gaussian blur (radius 2.0) to introduce spatial correlation.
  5. Save the cropped patch at the requested size.

Usage
-----
python src/analysis/image_examples/plot_oco_noise_patch.py \
    [--img  path/to/image.png]       # real CXR for stat estimation
    [--size 256]                     # patch side in pixels (square)
    [--out  output/oco_noise_patch.png]
    [--dpi  300]
    [--seed 42]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from PIL import Image, ImageFilter

rcParams["font.family"] = "sans-serif"


# ---------------------------------------------------------------------------
# Noise generation (mirrors preprocess._matched_correlated_noise_fill)
# ---------------------------------------------------------------------------

def _robust_stats_gray(img_arr: np.ndarray) -> tuple[float, float]:
    """Robust grayscale center (median) and scale (std) after 1-99 percentile trim."""
    # Collapse to luminance if RGB, otherwise use as-is
    if img_arr.ndim == 3:
        px = (0.2126 * img_arr[..., 0] + 0.7152 * img_arr[..., 1]
              + 0.0722 * img_arr[..., 2]).ravel().astype(np.float32)
    else:
        px = img_arr.ravel().astype(np.float32)
    lo, hi = np.percentile(px, 1.0), np.percentile(px, 99.0)
    trimmed = px[(px >= lo) & (px <= hi)]
    if trimmed.size < 32:
        trimmed = px
    center = float(np.median(trimmed))
    scale  = float(max(trimmed.std(), 1.0))
    return center, scale


def make_oco_noise_patch(
    img_arr: np.ndarray,
    size: int,
    block: int = 32,
) -> np.ndarray:
    """
    Generate a grayscale blocky noise patch matching the robust stats of img_arr.

    Noise is generated at (size // block) resolution then upscaled with
    nearest-neighbour interpolation so each logical pixel covers block×block px.

    Returns a uint8 2-D array of shape (size, size).
    """
    center, scale = _robust_stats_gray(img_arr)
    small = max(1, size // block)
    noise = np.random.randn(small, small).astype(np.float32) * scale + center
    noise = np.clip(noise, 0.0, 255.0).astype(np.uint8)
    patch = Image.fromarray(noise, mode="L").resize(
        (size, size), resample=Image.NEAREST
    )
    return np.array(patch, dtype=np.uint8)


# ---------------------------------------------------------------------------
# Image loading (mirrors preprocess._to_uint8_rgb)
# ---------------------------------------------------------------------------

def _load_uint8_rgb(img_path: Path) -> np.ndarray:
    with Image.open(img_path) as im:
        arr = np.array(im, dtype=np.float32)
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    arr = (arr * 255.0).astype(np.uint8)
    return np.array(Image.fromarray(arr).convert("RGB"), dtype=np.uint8)


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_patch(
    patch: np.ndarray,
    dpi: int = 600,
    output_path: Path | None = None,
) -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=dpi)
    fig.patch.set_facecolor("white")
    ax.imshow(patch, cmap="gray", vmin=0, vmax=255, aspect="equal", interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white",
                    format="png", pil_kwargs={"compress_level": 1})
        print(f"Saved → {output_path}")

    return fig


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _find_first_image(img_dir: Path) -> Path:
    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    if not imgs:
        raise FileNotFoundError(f"No images found in {img_dir}")
    return imgs[0]


def main() -> None:
    default_img_dir = (
        Path(__file__).parents[3]
        / "data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
    )
    default_out = Path(__file__).parent / "output" / "oco_noise_patch.png"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img", default=None,
                   help="CXR image used to estimate noise statistics. "
                        "If omitted, picks the first PNG in --img-dir.")
    p.add_argument("--img-dir", default=str(default_img_dir))
    p.add_argument("--size", type=int, default=1024,
                   help="Patch side length in pixels (square, default: 1024)")
    p.add_argument("--block", type=int, default=32,
                   help="Block size in pixels — each noise value covers block×block px (default: 32)")
    p.add_argument("--out", default=str(default_out))
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    np.random.seed(args.seed)

    if args.img:
        img_path = Path(args.img)
    else:
        img_path = _find_first_image(Path(args.img_dir))
        print(f"Using image for stat estimation: {img_path.name}")

    img_arr = _load_uint8_rgb(img_path)
    patch   = make_oco_noise_patch(img_arr, size=args.size, block=args.block)

    fig = plot_patch(patch, dpi=args.dpi, output_path=Path(args.out))
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
