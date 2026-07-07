"""
Visualize PadChest images with bounding boxes and annotations.

Creates nicely formatted plots showing:
- Original X-ray image
- Bounding boxes with colored borders
- Labels and sentence descriptions
"""

import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import numpy as np


# Color palette for different categories (colorblind-friendly)
CATEGORY_COLORS = {
    'Cardiomegaly': '#E64B35',      # Red
    'Fracture': '#4DBBD5',          # Cyan
    'Atelectasis': '#00A087',       # Teal
    'Consolidation': '#3C5488',     # Blue
    'Edema': '#F39B7F',             # Salmon
    'Lung Lesion': '#8491B4',       # Purple-gray
    'Lung Opacity': '#91D1C2',      # Light teal
    'Pleural Effusion': '#DC0000',  # Dark red
    'Pneumothorax': '#7E6148',      # Brown
    'Support Devices': '#B09C85',   # Tan
}

# Default image directory
IMAGE_DIR = Path("/mimer/NOBACKUP/groups/naiss2023-6-336/Shortcut-RRG/Shortcut-Learning-RRG/data/padchest-gr/BIMCV-Padchest-GR /PadChest_GR_images")


def load_samples(samples_json_path: str) -> list:
    """Load samples from a JSON file."""
    with open(samples_json_path, 'r') as f:
        return json.load(f)


def plot_single_image_with_bbox(
    sample: dict,
    image_dir: Path = IMAGE_DIR,
    figsize: tuple = (10, 10),
    ax=None
) -> plt.Figure:
    """
    Plot a single image with its bounding box and annotation.

    Args:
        sample: Sample dictionary containing image_info and main_bbox
        image_dir: Directory containing the images
        figsize: Figure size if creating new figure
        ax: Existing axes to plot on (optional)

    Returns:
        Figure object
    """
    image_id = sample['image_info']['ImageID']
    bbox_data = sample['main_bbox']

    # Load image
    image_path = image_dir / image_id
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]

    # Get bounding box coordinates (normalized -> pixel)
    box = bbox_data['box']
    x1 = box[0] * img_width
    y1 = box[1] * img_height
    x2 = box[2] * img_width
    y2 = box[3] * img_height

    # Get category color
    category = bbox_data['chexpert_category']
    color = CATEGORY_COLORS.get(category, '#FF6B6B')

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure

    # Display image
    ax.imshow(img_array, cmap='gray')

    # Draw bounding box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=3,
        edgecolor=color,
        facecolor='none',
        linestyle='-'
    )
    ax.add_patch(rect)

    # Add label background
    label_text = f"{category}"
    padchest_label = ', '.join(bbox_data.get('padchest_labels', []))

    # Add label at top of bbox
    ax.text(
        x1, y1 - 10,
        label_text,
        fontsize=12,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9),
        verticalalignment='bottom'
    )

    # Add PadChest label below the main label
    if padchest_label:
        ax.text(
            x1, y1 + 20,
            f"({padchest_label})",
            fontsize=9,
            color='white',
            bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.7),
            verticalalignment='top'
        )

    ax.axis('off')

    return fig


def plot_image_with_sentence(
    sample: dict,
    image_dir: Path = IMAGE_DIR,
    figsize: tuple = (12, 10)
) -> plt.Figure:
    """
    Plot an image with bounding box and sentence description below.

    Args:
        sample: Sample dictionary
        image_dir: Directory containing images
        figsize: Figure size

    Returns:
        Figure object
    """
    image_id = sample['image_info']['ImageID']
    bbox_data = sample['main_bbox']

    # Load image
    image_path = image_dir / image_id
    img = Image.open(image_path)
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]

    # Get bounding box
    box = bbox_data['box']
    x1 = box[0] * img_width
    y1 = box[1] * img_height
    x2 = box[2] * img_width
    y2 = box[3] * img_height

    # Get category and color
    category = bbox_data['chexpert_category']
    color = CATEGORY_COLORS.get(category, '#FF6B6B')

    # Create figure with space for text
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(6, 1, height_ratios=[5, 0.3, 0.3, 0.2, 0.2, 0.2])

    ax_img = fig.add_subplot(gs[0])

    # Display image
    ax_img.imshow(img_array, cmap='gray')

    # Draw bounding box
    rect = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=4,
        edgecolor=color,
        facecolor=color,
        alpha=0.2,
        linestyle='-'
    )
    ax_img.add_patch(rect)

    # Draw border
    rect_border = patches.Rectangle(
        (x1, y1), x2 - x1, y2 - y1,
        linewidth=4,
        edgecolor=color,
        facecolor='none',
        linestyle='-'
    )
    ax_img.add_patch(rect_border)

    # Add category label inside bbox
    ax_img.text(
        (x1 + x2) / 2, y1 + 30,
        category,
        fontsize=14,
        fontweight='bold',
        color='white',
        bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.95),
        horizontalalignment='center',
        verticalalignment='top'
    )

    ax_img.axis('off')
    ax_img.set_title(f'Image: {image_id[:40]}...', fontsize=10, color='gray')

    # Add text annotations below the image
    ax_text1 = fig.add_subplot(gs[2])
    ax_text1.axis('off')

    # English sentence
    sentence_en = bbox_data.get('sentence_en', 'N/A')
    ax_text1.text(
        0.5, 0.5,
        f'EN: "{sentence_en}"',
        fontsize=11,
        fontweight='medium',
        color='#2C3E50',
        horizontalalignment='center',
        verticalalignment='center',
        wrap=True,
        transform=ax_text1.transAxes
    )

    # Spanish sentence
    ax_text2 = fig.add_subplot(gs[3])
    ax_text2.axis('off')
    sentence_es = bbox_data.get('sentence_es', 'N/A')
    ax_text2.text(
        0.5, 0.5,
        f'ES: "{sentence_es}"',
        fontsize=10,
        fontstyle='italic',
        color='#7F8C8D',
        horizontalalignment='center',
        verticalalignment='center',
        wrap=True,
        transform=ax_text2.transAxes
    )

    # PadChest labels
    ax_text3 = fig.add_subplot(gs[4])
    ax_text3.axis('off')
    padchest_labels = ', '.join(bbox_data.get('padchest_labels', []))
    ax_text3.text(
        0.5, 0.5,
        f'PadChest Labels: [{padchest_labels}]',
        fontsize=9,
        color='#95A5A6',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax_text3.transAxes
    )

    plt.tight_layout()
    return fig


def create_comparison_grid(
    samples: list,
    title: str = "PadChest Image Examples with Bounding Boxes",
    image_dir: Path = IMAGE_DIR,
    ncols: int = 3,
    figsize_per_image: tuple = (5, 6),
    output_path: str = None
) -> plt.Figure:
    """
    Create a grid of images with bounding boxes.

    Args:
        samples: List of sample dictionaries
        title: Main title for the figure
        image_dir: Directory containing images
        ncols: Number of columns in the grid
        figsize_per_image: Size per image cell
        output_path: Path to save the figure (optional)

    Returns:
        Figure object
    """
    n_samples = len(samples)
    nrows = (n_samples + ncols - 1) // ncols

    fig_width = figsize_per_image[0] * ncols
    fig_height = figsize_per_image[1] * nrows + 1  # Extra space for title

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    if nrows == 1:
        axes = [axes] if ncols == 1 else axes
    if ncols == 1:
        axes = [[ax] for ax in axes]

    # Flatten axes for easy iteration
    axes_flat = [ax for row in axes for ax in (row if isinstance(row, np.ndarray) else [row])]

    for idx, (sample, ax) in enumerate(zip(samples, axes_flat)):
        image_id = sample['image_info']['ImageID']
        bbox_data = sample['main_bbox']

        # Load image
        image_path = image_dir / image_id
        if not image_path.exists():
            ax.text(0.5, 0.5, f'Image not found:\n{image_id[:30]}...',
                   ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        img = Image.open(image_path)
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        # Get bounding box
        box = bbox_data['box']
        x1 = box[0] * img_width
        y1 = box[1] * img_height
        x2 = box[2] * img_width
        y2 = box[3] * img_height

        # Get category and color
        category = bbox_data['chexpert_category']
        color = CATEGORY_COLORS.get(category, '#FF6B6B')

        # Display image
        ax.imshow(img_array, cmap='gray')

        # Draw filled bounding box with transparency
        rect_fill = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0,
            facecolor=color,
            alpha=0.25
        )
        ax.add_patch(rect_fill)

        # Draw border
        rect_border = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=3,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect_border)

        # Add label
        ax.text(
            x1 + 5, y1 + 25,
            category,
            fontsize=11,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9)
        )

        # Add sentence below
        sentence = bbox_data.get('sentence_en', '')
        if len(sentence) > 60:
            sentence = sentence[:57] + '...'
        ax.set_xlabel(sentence, fontsize=8, wrap=True, color='#2C3E50')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused axes
    for idx in range(len(samples), len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def plot_image_with_multiple_bboxes(
    sample: dict,
    image_dir: Path = IMAGE_DIR,
    figsize: tuple = (12, 14),
    output_path: str = None
) -> plt.Figure:
    """
    Plot an image with multiple bounding boxes (main + other).

    Args:
        sample: Sample dictionary with main_bbox and other_bboxes
        image_dir: Directory containing images
        figsize: Figure size
        output_path: Path to save the figure (optional)

    Returns:
        Figure object
    """
    image_id = sample['image_info']['ImageID']
    main_bbox = sample['main_bbox']
    other_bboxes = sample.get('other_bboxes', [])

    # Load image
    image_path = image_dir / image_id
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    img = Image.open(image_path)
    img_array = np.array(img)
    img_height, img_width = img_array.shape[:2]

    # Create figure
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(8, 1, height_ratios=[6, 0.4, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3])

    ax_img = fig.add_subplot(gs[0])

    # Display image
    ax_img.imshow(img_array, cmap='gray')

    # Collect all bboxes with their info
    all_bboxes = []

    # Add main bbox
    main_category = main_bbox['chexpert_category']
    main_color = CATEGORY_COLORS.get(main_category, '#FF6B6B')
    all_bboxes.append({
        'box': main_bbox['box'],
        'category': main_category,
        'color': main_color,
        'padchest_labels': main_bbox.get('padchest_labels', []),
        'sentence_en': main_bbox.get('sentence_en', ''),
        'is_main': True
    })

    # Add other bboxes
    for other in other_bboxes:
        # Get category from other bbox
        categories = other.get('chexpert_categories', [])
        category = categories[0] if categories else 'Unknown'
        color = CATEGORY_COLORS.get(category, '#9B59B6')  # Purple for unknown

        all_bboxes.append({
            'box': other['box'],
            'category': category,
            'color': color,
            'padchest_labels': other.get('padchest_labels', []),
            'sentence_en': other.get('sentence_en', ''),
            'is_main': False
        })

    # Draw all bounding boxes
    legend_handles = []
    for i, bbox_info in enumerate(all_bboxes):
        box = bbox_info['box']
        x1 = box[0] * img_width
        y1 = box[1] * img_height
        x2 = box[2] * img_width
        y2 = box[3] * img_height

        color = bbox_info['color']
        category = bbox_info['category']
        is_main = bbox_info['is_main']

        # Different line styles for main vs other
        linestyle = '-' if is_main else '--'
        linewidth = 4 if is_main else 3

        # Draw filled bbox
        rect_fill = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=0,
            facecolor=color,
            alpha=0.2
        )
        ax_img.add_patch(rect_fill)

        # Draw border
        rect_border = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=linewidth,
            edgecolor=color,
            facecolor='none',
            linestyle=linestyle
        )
        ax_img.add_patch(rect_border)

        # Add label
        label_suffix = " (main)" if is_main else ""
        ax_img.text(
            x1 + 5, y1 + 25 + (i * 30 if not is_main else 0),
            f"{category}{label_suffix}",
            fontsize=10,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=color, alpha=0.9)
        )

        # Add to legend
        legend_handles.append(patches.Patch(
            facecolor=color, alpha=0.5,
            edgecolor=color, linewidth=2,
            label=f"{category}{label_suffix}"
        ))

    ax_img.axis('off')
    ax_img.set_title(
        f'Image with {len(all_bboxes)} bounding boxes\n{image_id[:50]}...',
        fontsize=11, color='gray'
    )

    # Add legend
    ax_img.legend(
        handles=legend_handles,
        loc='upper right',
        fontsize=9,
        framealpha=0.9
    )

    # Add text annotations
    ax_legend = fig.add_subplot(gs[1])
    ax_legend.axis('off')
    ax_legend.text(
        0.5, 0.5,
        f"Total bounding boxes: {sample.get('total_bboxes_in_image', len(all_bboxes))}  |  "
        f"Categories: {', '.join(set(b['category'] for b in all_bboxes))}",
        fontsize=10,
        fontweight='bold',
        color='#2C3E50',
        horizontalalignment='center',
        verticalalignment='center',
        transform=ax_legend.transAxes
    )

    # Add sentences for each bbox
    for i, bbox_info in enumerate(all_bboxes[:4]):  # Limit to 4 sentences
        ax_text = fig.add_subplot(gs[2 + i])
        ax_text.axis('off')

        sentence = bbox_info.get('sentence_en', 'N/A')
        if len(sentence) > 100:
            sentence = sentence[:97] + '...'

        color = bbox_info['color']
        category = bbox_info['category']
        main_marker = "★ " if bbox_info['is_main'] else "  "

        ax_text.text(
            0.02, 0.5,
            f"{main_marker}",
            fontsize=12,
            color=color,
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax_text.transAxes
        )
        ax_text.text(
            0.05, 0.5,
            f"[{category}] {sentence}",
            fontsize=9,
            color='#2C3E50',
            horizontalalignment='left',
            verticalalignment='center',
            wrap=True,
            transform=ax_text.transAxes
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def create_multi_bbox_grid(
    samples: list,
    title: str = "Images with Multiple Bounding Boxes",
    image_dir: Path = IMAGE_DIR,
    ncols: int = 3,
    figsize_per_image: tuple = (6, 7),
    output_path: str = None
) -> plt.Figure:
    """
    Create a grid of images showing all their bounding boxes.

    Args:
        samples: List of sample dictionaries (with multiple bboxes)
        title: Main title
        image_dir: Directory containing images
        ncols: Number of columns
        figsize_per_image: Size per cell
        output_path: Path to save

    Returns:
        Figure object
    """
    n_samples = len(samples)
    nrows = (n_samples + ncols - 1) // ncols

    fig_width = figsize_per_image[0] * ncols
    fig_height = figsize_per_image[1] * nrows + 1

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height))

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    axes_flat = axes.flatten()

    for idx, (sample, ax) in enumerate(zip(samples, axes_flat)):
        image_id = sample['image_info']['ImageID']
        main_bbox = sample['main_bbox']
        other_bboxes = sample.get('other_bboxes', [])

        # Load image
        image_path = image_dir / image_id
        if not image_path.exists():
            ax.text(0.5, 0.5, f'Image not found', ha='center', va='center', transform=ax.transAxes)
            ax.axis('off')
            continue

        img = Image.open(image_path)
        img_array = np.array(img)
        img_height, img_width = img_array.shape[:2]

        # Display image
        ax.imshow(img_array, cmap='gray')

        # Draw main bbox
        main_category = main_bbox['chexpert_category']
        main_color = CATEGORY_COLORS.get(main_category, '#FF6B6B')
        box = main_bbox['box']
        x1, y1 = box[0] * img_width, box[1] * img_height
        x2, y2 = box[2] * img_width, box[3] * img_height

        rect_fill = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=main_color, alpha=0.25)
        ax.add_patch(rect_fill)
        rect_border = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=3, edgecolor=main_color, facecolor='none')
        ax.add_patch(rect_border)
        ax.text(x1 + 5, y1 + 20, main_category, fontsize=9, fontweight='bold', color='white',
                bbox=dict(boxstyle='round,pad=0.2', facecolor=main_color, alpha=0.9))

        # Draw other bboxes
        for other in other_bboxes:
            categories = other.get('chexpert_categories', [])
            category = categories[0] if categories else 'Unknown'
            color = CATEGORY_COLORS.get(category, '#9B59B6')

            box = other['box']
            x1, y1 = box[0] * img_width, box[1] * img_height
            x2, y2 = box[2] * img_width, box[3] * img_height

            rect_fill = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, facecolor=color, alpha=0.2)
            ax.add_patch(rect_fill)
            rect_border = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor=color, facecolor='none', linestyle='--')
            ax.add_patch(rect_border)
            ax.text(x1 + 5, y1 + 20, category, fontsize=8, fontweight='bold', color='white',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor=color, alpha=0.85))

        # Title with bbox count
        total_bboxes = sample.get('total_bboxes_in_image', 1 + len(other_bboxes))
        ax.set_title(f"{total_bboxes} bboxes", fontsize=10, fontweight='bold')

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    # Hide unused axes
    for idx in range(len(samples), len(axes_flat)):
        axes_flat[idx].axis('off')

    fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")

    return fig


def find_samples_with_multiple_bboxes(samples: list, min_bboxes: int = 2) -> list:
    """Find samples that have at least min_bboxes bounding boxes."""
    return [s for s in samples if s.get('total_bboxes_in_image', 1) >= min_bboxes]


def main():
    """Main function to create example visualizations."""
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Paths to samples
    base_data_dir = Path("/mimer/NOBACKUP/groups/naiss2023-6-336/Shortcut-RRG/Shortcut-Learning-RRG/data/padchest-gr/chexpert-by-label")

    cardiomegaly_path = base_data_dir / "Cardiomegaly" / "samples.json"
    fracture_path = base_data_dir / "Fracture" / "samples.json"

    # Load samples
    print("Loading Cardiomegaly samples...")
    cardiomegaly_samples = load_samples(cardiomegaly_path)
    print(f"  Loaded {len(cardiomegaly_samples)} samples")

    print("Loading Fracture samples...")
    fracture_samples = load_samples(fracture_path)
    print(f"  Loaded {len(fracture_samples)} samples")

    # Create comparison grid for Cardiomegaly (3 examples)
    print("\nCreating Cardiomegaly examples...")
    fig_cardio = create_comparison_grid(
        cardiomegaly_samples[:3],
        title="Cardiomegaly Examples from PadChest-GR",
        output_path=str(output_dir / "cardiomegaly_examples.png")
    )
    plt.close(fig_cardio)

    # Create comparison grid for Fracture (3 examples)
    print("Creating Fracture examples...")
    fig_fracture = create_comparison_grid(
        fracture_samples[:3],
        title="Fracture Examples from PadChest-GR",
        output_path=str(output_dir / "fracture_examples.png")
    )
    plt.close(fig_fracture)

    # Create mixed comparison (2 from each category)
    print("Creating mixed comparison...")
    mixed_samples = cardiomegaly_samples[:2] + fracture_samples[:2]
    fig_mixed = create_comparison_grid(
        mixed_samples,
        title="PadChest-GR Examples: Cardiomegaly vs Fracture",
        ncols=2,
        figsize_per_image=(6, 7),
        output_path=str(output_dir / "mixed_comparison.png")
    )
    plt.close(fig_mixed)

    # Create detailed single image plots
    print("\nCreating detailed single-image plots...")

    # Cardiomegaly detailed
    fig_cardio_detail = plot_image_with_sentence(
        cardiomegaly_samples[0],
        figsize=(10, 12)
    )
    fig_cardio_detail.savefig(
        output_dir / "cardiomegaly_detailed.png",
        dpi=300, bbox_inches='tight', facecolor='white'
    )
    print(f"Saved: {output_dir / 'cardiomegaly_detailed.png'}")
    plt.close(fig_cardio_detail)

    # Fracture detailed
    fig_fracture_detail = plot_image_with_sentence(
        fracture_samples[0],
        figsize=(10, 12)
    )
    fig_fracture_detail.savefig(
        output_dir / "fracture_detailed.png",
        dpi=300, bbox_inches='tight', facecolor='white'
    )
    print(f"Saved: {output_dir / 'fracture_detailed.png'}")
    plt.close(fig_fracture_detail)

    # --- NEW: Images with multiple bounding boxes ---
    print("\n" + "="*60)
    print("Creating visualizations for images with MULTIPLE bboxes...")
    print("="*60)

    # Find samples with 2+ bboxes
    cardio_multi = find_samples_with_multiple_bboxes(cardiomegaly_samples, min_bboxes=2)
    fracture_multi = find_samples_with_multiple_bboxes(fracture_samples, min_bboxes=2)

    print(f"\nCardiomegaly samples with 2+ bboxes: {len(cardio_multi)}")
    print(f"Fracture samples with 2+ bboxes: {len(fracture_multi)}")

    # Create grid for Cardiomegaly with multiple bboxes
    if cardio_multi:
        print("\nCreating Cardiomegaly multi-bbox examples...")
        fig_cardio_multi = create_multi_bbox_grid(
            cardio_multi[:3],
            title="Cardiomegaly Images with Multiple Bounding Boxes",
            output_path=str(output_dir / "cardiomegaly_multi_bbox.png")
        )
        plt.close(fig_cardio_multi)

        # Create detailed view for first multi-bbox sample
        print("Creating detailed multi-bbox Cardiomegaly view...")
        fig_cardio_multi_detail = plot_image_with_multiple_bboxes(
            cardio_multi[0],
            figsize=(12, 14),
            output_path=str(output_dir / "cardiomegaly_multi_bbox_detailed.png")
        )
        plt.close(fig_cardio_multi_detail)

    # Create grid for Fracture with multiple bboxes
    if fracture_multi:
        print("\nCreating Fracture multi-bbox examples...")
        fig_fracture_multi = create_multi_bbox_grid(
            fracture_multi[:3],
            title="Fracture Images with Multiple Bounding Boxes",
            output_path=str(output_dir / "fracture_multi_bbox.png")
        )
        plt.close(fig_fracture_multi)

        # Create detailed view
        print("Creating detailed multi-bbox Fracture view...")
        fig_fracture_multi_detail = plot_image_with_multiple_bboxes(
            fracture_multi[0],
            figsize=(12, 14),
            output_path=str(output_dir / "fracture_multi_bbox_detailed.png")
        )
        plt.close(fig_fracture_multi_detail)

    # Mixed multi-bbox comparison
    mixed_multi = cardio_multi[:2] + fracture_multi[:2] if cardio_multi and fracture_multi else []
    if mixed_multi:
        print("\nCreating mixed multi-bbox comparison...")
        fig_mixed_multi = create_multi_bbox_grid(
            mixed_multi,
            title="Multi-Bbox Examples: Cardiomegaly & Fracture",
            ncols=2,
            figsize_per_image=(7, 8),
            output_path=str(output_dir / "mixed_multi_bbox.png")
        )
        plt.close(fig_mixed_multi)

    print("\n" + "="*60)
    print("All visualizations created successfully!")
    print(f"Output directory: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
