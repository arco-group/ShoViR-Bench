"""
Standalone demo figure: one X-ray image with three non-overlapping bounding boxes.
  1. Disease region   — red solid border
  2. Co-disease region — blue dashed border
  3. Random region    — green dotted border

Usage
-----
python src/analysis/image_examples/plot_demo_three_bbox.py \
    [--img  path/to/image.png] \
    [--out  output/demo_three_bbox.png] \
    [--dpi  300]

If --img is omitted the script picks the first PNG it finds under
data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from PIL import Image

rcParams["font.family"] = "sans-serif"

# ---------------------------------------------------------------------------
# Three non-overlapping boxes (normalised [x1, y1, x2, y2])
# Verified non-overlapping:
#   ① x=[0.05,0.42] y=[0.06,0.44]   top-left
#   ② x=[0.54,0.92] y=[0.52,0.88]   bottom-right
#   ③ x=[0.08,0.44] y=[0.58,0.92]   bottom-left
# ---------------------------------------------------------------------------
REGIONS = [
    {
        "label": "Disease",
        "category": "Cardiomegaly",
        "anatomy": "Heart",
        "type": "disease",
        "color": "#EB4C4C",          # red
        "linestyle": "-",
        "linewidth": 10,
        "box": [0.05, 0.11, 0.42, 0.44],
    },
    {
        "label": "Co-disease",
        "category": "Pleural Effusion",
        "anatomy": "Right pleural space",
        "type": "co",
        "color": "#0992C2",          # blue
        "linestyle": "-",
        "linewidth": 10,
        "box": [0.54, 0.52, 0.92, 0.88],
    },
    {
        "label": "Random region",
        "category": "Random",
        "anatomy": "Left lung",
        "type": "random",
        "color": "#A8DF8E",          # green
        "linestyle": "-",
        "linewidth": 10,
        "box": [0.08, 0.58, 0.44, 0.92],
    },
]


def _load_image(img_path: Path) -> np.ndarray:
    """Load any PIL-readable image as a normalised float32 grayscale array."""
    img = Image.open(img_path)
    # Convert multi-channel images to grayscale first
    if img.mode not in ("L", "I", "I;16", "I;16B", "F"):
        img = img.convert("RGB")
        arr = np.array(img, dtype=np.float32)
        # luminosity weights
        arr = 0.2126 * arr[..., 0] + 0.7152 * arr[..., 1] + 0.0722 * arr[..., 2]
    else:
        arr = np.array(img, dtype=np.float32)
    # Normalise to [0, 1]
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo)
    return arr


def plot_demo(img_path: Path, dpi: int = 300, output_path: Path | None = None,
              draw_boxes: bool = True) -> plt.Figure:
    img_array = _load_image(img_path)
    ih, iw = img_array.shape

    fig = plt.figure(figsize=(20, 20), dpi=dpi)
    fig.patch.set_facecolor("white")

    # Image fills the whole figure
    ax_img = fig.add_axes([0.0, 0.0, 1.0, 1.0])

    ax_img.imshow(img_array, cmap="gray", vmin=0, vmax=1, aspect="equal")
    ax_img.axis("off")

    if not draw_boxes:
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white",
                        format="png", pil_kwargs={"compress_level": 1})
            print(f"Saved → {output_path}")
        return fig

    for ri, reg in enumerate(REGIONS):
        x1, y1, x2, y2 = reg["box"]
        px1, py1 = x1 * iw, y1 * ih
        pw,  ph  = (x2 - x1) * iw, (y2 - y1) * ih
        color    = reg["color"]

        # border only — no fill
        ax_img.add_patch(mpatches.Rectangle(
            (px1, py1), pw, ph,
            linewidth=reg["linewidth"], edgecolor=color,
            facecolor="none", linestyle=reg["linestyle"],
        ))

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white",
                    format="png", pil_kwargs={"compress_level": 1})
        print(f"Saved → {output_path}")

    return fig


def _find_first_image(img_dir: Path) -> Path:
    imgs = sorted(img_dir.glob("*.png"))
    if not imgs:
        imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.jpeg"))
    if not imgs:
        raise FileNotFoundError(f"No PNG/JPG images found in {img_dir}")
    return imgs[0]


def main() -> None:
    default_img_dir = (
        Path(__file__).parents[3]
        / "data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
    )
    default_out = Path(__file__).parent / "output" / "demo_three_bbox.png"

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--img", default=None,
                   help="Path to a single X-ray image. "
                        "If omitted, uses the first PNG in --img-dir.")
    p.add_argument("--img-dir", default=str(default_img_dir),
                   help="Image directory to auto-pick from (default: PadChest_GR_images)")
    p.add_argument("--out", default=str(default_out), help="Output PNG path")
    p.add_argument("--dpi", type=int, default=600)
    p.add_argument("--no-boxes", action="store_true", help="Save the plain image without any bounding boxes")
    args = p.parse_args()

    if args.img:
        img_path = Path(args.img)
    else:
        img_path = _find_first_image(Path(args.img_dir))
        print(f"Auto-selected image: {img_path.name}")

    fig = plot_demo(img_path, dpi=args.dpi, output_path=Path(args.out),
                    draw_boxes=not args.no_boxes)
    plt.close(fig)
    print("Done.")


if __name__ == "__main__":
    main()
