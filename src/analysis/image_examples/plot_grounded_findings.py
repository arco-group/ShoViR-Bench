"""
Visualize PadChest-GR grounded findings — mimics the reference figure style:
  • Left:  X-ray with numbered, colored bounding boxes (no text labels on image)
  • Right: "Grounded Findings" legend with finding category, locations and
            progression for every finding (boxed or not)

Usage
-----
python src/analysis/image_examples/plot_grounded_findings.py \
    [--json data/padchest-gr/BIMCV-Padchest-GR\ /grounded_reports_20240819.json] \
    [--img-dir "data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"] \
    [--indices 0 5 12]          # which samples to plot (default: first N with ≥1 box)
    [--n 3]                     # how many samples to auto-pick
    [--out output/grounded]     # output prefix (one PNG per sample)
    [--dpi 300]
"""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from PIL import Image

# ---------------------------------------------------------------------------
# Aesthetics
# ---------------------------------------------------------------------------
rcParams["font.family"] = "sans-serif"

# Palette: visually distinct, colorblind-safe-ish, high contrast on dark bg
_PALETTE = [
    "#4C9BE8",   # blue
    "#F5A623",   # orange
    "#4CAF50",   # green
    "#E05C5C",   # red
    "#9B59B6",   # purple
    "#00BCD4",   # cyan
    "#FF8F00",   # amber
    "#EC407A",   # pink
]


def _color(idx: int) -> str:
    return _PALETTE[idx % len(_PALETTE)]


# ---------------------------------------------------------------------------
# Core plot function
# ---------------------------------------------------------------------------

def plot_sample(
    sample: dict,
    img_dir: Path,
    dpi: int = 300,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot one sample: X-ray (left) + findings legend (right).

    Parameters
    ----------
    sample      : one entry from grounded_reports_20240819.json
    img_dir     : directory that contains the PNG files
    dpi         : output resolution
    output_path : if given, save figure to this path
    """
    image_id = sample["ImageID"]
    findings = sample["findings"]
    n = len(findings)

    # ── load image ──────────────────────────────────────────────────────────
    img_path = img_dir / image_id
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_array = np.array(Image.open(img_path))
    ih, iw = img_array.shape[:2]

    # ── figure height scales with number of findings ─────────────────────────
    fig_h = max(8.0, n * 1.6)
    fig = plt.figure(figsize=(18, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    # left panel (image), right panel (legend)
    ax_img = fig.add_axes([0.01, 0.02, 0.50, 0.96])
    ax_leg = fig.add_axes([0.53, 0.02, 0.46, 0.96])

    # ── left: image with boxes ───────────────────────────────────────────────
    ax_img.imshow(img_array, cmap="gray", aspect="equal")
    ax_img.set_title("Current image", fontsize=13, color="#555555", pad=8)
    ax_img.axis("off")

    finding_colors: list[str] = []

    for fi, finding in enumerate(findings):
        color = _color(fi)
        finding_colors.append(color)
        boxes = finding.get("boxes", [])

        for box in boxes:
            x1 = box[0] * iw
            y1 = box[1] * ih
            x2 = box[2] * iw
            y2 = box[3] * ih
            w, h = x2 - x1, y2 - y1

            # semi-transparent fill
            ax_img.add_patch(mpatches.Rectangle(
                (x1, y1), w, h,
                linewidth=0, facecolor=color, alpha=0.18,
            ))
            # border
            ax_img.add_patch(mpatches.Rectangle(
                (x1, y1), w, h,
                linewidth=2.5, edgecolor=color, facecolor="none",
            ))
            # number badge (circle in data coords — always works)
            badge_r = min(iw, ih) * 0.028
            cx = x1 + badge_r + 3
            cy = y1 + badge_r + 3
            ax_img.add_patch(plt.Circle(
                (cx, cy), badge_r, color=color, zorder=5,
            ))
            ax_img.text(
                cx, cy, str(fi + 1),
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white", zorder=6,
            )

    # ── right: legend panel — work in DATA coords with xlim/ylim=(0,1) ───────
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.set_facecolor("white")
    ax_leg.axis("off")

    # header
    HEADER_Y = 0.97
    ax_leg.text(
        0.0, HEADER_Y,
        "Grounded Findings",
        ha="left", va="top",
        fontsize=15, fontweight="bold", color="#222222",
    )
    # separator under header
    sep_y = HEADER_Y - 0.045
    ax_leg.plot([0, 1], [sep_y, sep_y], color="#cccccc", linewidth=1.0, clip_on=False)

    # distribute findings evenly in remaining vertical space
    content_top = sep_y - 0.01
    slot_h = content_top / max(n, 1)

    BADGE_R   = 0.025          # circle radius in data coords
    BADGE_X   = BADGE_R + 0.01
    TEXT_X    = BADGE_X * 2 + BADGE_R + 0.02
    LINE_GAP  = 0.033          # vertical gap between text lines

    for fi, finding in enumerate(findings):
        color = finding_colors[fi]

        # top of this finding's slot
        slot_top = content_top - fi * slot_h
        badge_y  = slot_top - slot_h * 0.30

        # ── colored circle badge (in data coords — reliable) ─────────────────
        circle = plt.Circle(
            (BADGE_X, badge_y), BADGE_R,
            color=color, zorder=5, clip_on=False,
        )
        ax_leg.add_patch(circle)
        ax_leg.text(
            BADGE_X, badge_y, str(fi + 1),
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=6,
        )

        # ── text lines ───────────────────────────────────────────────────────
        labels     = finding.get("labels", [])
        locations  = finding.get("locations", [])
        progression = finding.get("progression")

        lines: list[tuple[str, dict]] = []

        cat_text = ", ".join(labels) if labels else "negative"
        lines.append((
            f"Finding category: {cat_text}",
            {"size": 13, "weight": "semibold", "color": "#111111"},
        ))
        if locations:
            lines.append((
                f"Locations: {', '.join(locations)}",
                {"size": 11, "color": "#444444"},
            ))
        lines.append((
            f"Progression label: {progression if progression is not None else 'None'}",
            {"size": 11, "color": "#888888"},
        ))

        text_y = slot_top - slot_h * 0.08
        for li, (txt, fmt) in enumerate(lines):
            wrapped = textwrap.shorten(txt, width=60, placeholder="…")
            ax_leg.text(
                TEXT_X, text_y - li * LINE_GAP,
                wrapped,
                ha="left", va="top",
                fontsize=fmt.get("size", 11),
                fontweight=fmt.get("weight", "normal"),
                color=fmt.get("color", "#333333"),
            )

        # thin rule between findings
        if fi < n - 1:
            rule_y = slot_top - slot_h
            ax_leg.plot(
                [0.01, 0.99], [rule_y, rule_y],
                color="#eeeeee", linewidth=0.8, clip_on=False,
            )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {output_path}")

    return fig


# ---------------------------------------------------------------------------
# chexpert-by-label format
# ---------------------------------------------------------------------------

def plot_chexpert_sample(
    sample: dict,
    img_dir: Path,
    label: str = "",
    dpi: int = 300,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot one sample from chexpert-by-label/<Label>_samples.json.

    Disease regions are drawn with a solid border; co-occurrence regions with a
    dashed border. All regions appear in the right-hand legend with their
    CheXpert category and anatomy.

    Parameters
    ----------
    sample      : one entry from e.g. Atelectasis_samples.json
    img_dir     : directory that contains the PNG files
    label       : the primary label name (e.g. "Atelectasis"), used for header
    dpi         : output resolution
    output_path : if given, save figure to this path
    """
    image_id = sample["img_path"]

    # Build a flat list of regions: disease first, then co-occurrence
    disease_regions = sample.get("disease_regions", [])
    co_regions      = sample.get("co_occurrence_regions", [])
    all_regions = [
        (r, "disease")    for r in disease_regions
    ] + [
        (r, "co")         for r in co_regions
    ]
    n = len(all_regions)

    # ── load image ──────────────────────────────────────────────────────────
    img_path = img_dir / image_id
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    img_array = np.array(Image.open(img_path))
    ih, iw = img_array.shape[:2]

    # ── figure ──────────────────────────────────────────────────────────────
    fig_h = max(8.0, n * 1.6)
    fig = plt.figure(figsize=(18, fig_h), dpi=dpi)
    fig.patch.set_facecolor("white")

    ax_img = fig.add_axes([0.01, 0.02, 0.50, 0.96])
    ax_leg = fig.add_axes([0.53, 0.02, 0.46, 0.96])

    # ── left: image ─────────────────────────────────────────────────────────
    ax_img.imshow(img_array, cmap="gray", aspect="equal")
    ax_img.axis("off")

    region_colors: list[str] = []

    for ri, (region, kind) in enumerate(all_regions):
        color = _color(ri)
        region_colors.append(color)
        linestyle = "-" if kind == "disease" else "--"
        linewidth = 2.5 if kind == "disease" else 2.0

        for box in region.get("bbox", []):
            x1 = box[0] * iw
            y1 = box[1] * ih
            x2 = box[2] * iw
            y2 = box[3] * ih
            w, h = x2 - x1, y2 - y1

            ax_img.add_patch(mpatches.Rectangle(
                (x1, y1), w, h,
                linewidth=0, facecolor=color, alpha=0.18,
            ))
            ax_img.add_patch(mpatches.Rectangle(
                (x1, y1), w, h,
                linewidth=linewidth, edgecolor=color,
                facecolor="none", linestyle=linestyle,
            ))
            # numbered badge
            badge_r = min(iw, ih) * 0.028
            cx = x1 + badge_r + 3
            cy = y1 + badge_r + 3
            ax_img.add_patch(plt.Circle(
                (cx, cy), badge_r, color=color, zorder=5,
            ))
            ax_img.text(
                cx, cy, str(ri + 1),
                ha="center", va="center",
                fontsize=10, fontweight="bold",
                color="white", zorder=6,
            )

    # ── right: legend ────────────────────────────────────────────────────────
    ax_leg.set_xlim(0, 1)
    ax_leg.set_ylim(0, 1)
    ax_leg.set_facecolor("white")
    ax_leg.axis("off")

    content_top = 0.98
    slot_h  = content_top / max(n, 1)
    BADGE_R  = 0.025
    BADGE_X  = BADGE_R + 0.01
    TEXT_X   = BADGE_X * 2 + BADGE_R + 0.02
    LINE_GAP = 0.033

    for ri, (region, kind) in enumerate(all_regions):
        color = region_colors[ri]
        slot_top = content_top - ri * slot_h
        badge_y  = slot_top - slot_h * 0.30

        # badge
        ax_leg.add_patch(plt.Circle(
            (BADGE_X, badge_y), BADGE_R,
            color=color, zorder=5, clip_on=False,
        ))
        ax_leg.text(
            BADGE_X, badge_y, str(ri + 1),
            ha="center", va="center",
            fontsize=11, fontweight="bold", color="white", zorder=6,
        )

        # text lines
        cats    = region.get("chexpert_categories", [])
        anatomy = region.get("anatomy", "")

        cat_text = ", ".join(cats) if cats else "unknown"

        lines: list[tuple[str, dict]] = [
            (
                f"Chexpert Label: {cat_text}",
                {"size": 13, "weight": "semibold", "color": "#111111"},
            ),
        ]
        if anatomy:
            lines.append((
                f"Anatomy: {anatomy}",
                {"size": 11, "color": "#444444"},
            ))

        text_y = slot_top - slot_h * 0.08
        for li, (txt, fmt) in enumerate(lines):
            wrapped = textwrap.shorten(txt, width=62, placeholder="…")
            ax_leg.text(
                TEXT_X, text_y - li * LINE_GAP,
                wrapped,
                ha="left", va="top",
                fontsize=fmt.get("size", 11),
                fontweight=fmt.get("weight", "normal"),
                color=fmt.get("color", "#333333"),
            )

        if ri < n - 1:
            rule_y = slot_top - slot_h
            ax_leg.plot(
                [0.01, 0.99], [rule_y, rule_y],
                color="#eeeeee", linewidth=0.8, clip_on=False,
            )

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        print(f"Saved → {output_path}")

    return fig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def pick_samples(data: list[dict], n: int) -> list[dict]:
    """Return the first n samples that have at least 1 finding with boxes."""
    out = []
    for s in data:
        boxed = sum(1 for f in s["findings"] if f.get("boxes"))
        if boxed >= 1:
            out.append(s)
            if len(out) == n:
                break
    return out


def pick_chexpert_samples(data: list[dict], n: int) -> list[dict]:
    """Return the first n samples that have at least 1 disease region and 1 co-occurrence region."""
    out = []
    for s in data:
        if s.get("disease_regions") and s.get("co_occurrence_regions"):
            out.append(s)
            if len(out) == n:
                break
    # fall back to samples with any region
    if not out:
        for s in data:
            if s.get("disease_regions") or s.get("co_occurrence_regions"):
                out.append(s)
                if len(out) == n:
                    break
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    default_img_dir = (
        Path(__file__).parents[3]
        / "data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images"
    )
    default_json = (
        Path(__file__).parents[3]
        / "data/padchest-gr/BIMCV-Padchest-GR /grounded_reports_20240819.json"
    )
    default_chexpert_dir = (
        Path(__file__).parents[3]
        / "data/padchest-gr/chexpert-by-label"
    )

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", choices=["grounded", "chexpert"], default="chexpert",
                   help="Data source format (default: chexpert)")
    # grounded-format args
    p.add_argument("--json",    default=str(default_json),    help="[grounded] Path to grounded_reports JSON")
    # chexpert-format args
    p.add_argument("--label",   default="Atelectasis",
                   help="[chexpert] Label name, e.g. Atelectasis (used to find <label>_samples.json)")
    p.add_argument("--chexpert-dir", default=str(default_chexpert_dir),
                   help="[chexpert] Directory containing <Label>_samples.json files")
    # shared
    p.add_argument("--img-dir", default=str(default_img_dir), help="Image directory")
    p.add_argument("--indices", nargs="+", type=int, default=None,
                   help="Explicit sample indices to plot (0-based). Overrides --n.")
    p.add_argument("--n",  type=int, default=3, help="Number of samples to auto-pick")
    p.add_argument("--out", default=None, help="Output path prefix (no extension). Default: output/<label>")
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    img_dir = Path(args.img_dir)

    if args.source == "chexpert":
        samples_file = Path(args.chexpert_dir) / f"{args.label}_samples.json"
        print(f"Loading {samples_file} …")
        with open(samples_file) as f:
            data = json.load(f)
        print(f"  {len(data)} samples loaded.")

        if args.indices is not None:
            samples = [data[i] for i in args.indices]
        else:
            samples = pick_chexpert_samples(data, args.n)
            if not samples:
                samples = data[: args.n]

        out_prefix = args.out or str(
            Path(__file__).parent / "output" / args.label.lower()
        )
        print(f"Plotting {len(samples)} samples …")
        for i, sample in enumerate(samples):
            out = Path(f"{out_prefix}_{i:02d}.png")
            fig = plot_chexpert_sample(
                sample, img_dir,
                label=args.label,
                dpi=args.dpi,
                output_path=out,
            )
            plt.close(fig)

    else:  # grounded
        print(f"Loading {args.json} …")
        with open(args.json) as f:
            data = json.load(f)
        print(f"  {len(data)} studies loaded.")

        if args.indices is not None:
            samples = [data[i] for i in args.indices]
        else:
            samples = pick_samples(data, args.n)
            if not samples:
                samples = data[: args.n]

        out_prefix = args.out or str(Path(__file__).parent / "output" / "grounded")
        print(f"Plotting {len(samples)} samples …")
        for i, sample in enumerate(samples):
            out = Path(f"{out_prefix}_{i:02d}.png")
            fig = plot_sample(sample, img_dir, dpi=args.dpi, output_path=out)
            plt.close(fig)

    print("Done.")


if __name__ == "__main__":
    main()
