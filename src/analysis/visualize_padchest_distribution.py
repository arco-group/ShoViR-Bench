"""
Create visualization plots for PadChest-GR dataset distribution with CheXpert labels.
"""

import json
import sys
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from padchest_to_chexpert_mapping import convert_image_to_chexpert_labels


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)


def analyze_and_visualize(json_path, output_dir="src/analysis/plots"):
    """Create comprehensive visualizations of the dataset."""

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Total images: {len(data)}\n")

    # CheXpert categories
    chexpert_categories = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
    ]

    # Data collection
    category_counts = {cat: 0 for cat in chexpert_categories}
    category_bbox_counts = {cat: 0 for cat in chexpert_categories}
    category_bbox_distribution = {cat: [] for cat in chexpert_categories}
    images_with_n_bboxes = Counter()
    images_with_n_bboxes_per_category = {cat: Counter() for cat in chexpert_categories}

    # Images with 2+ bboxes per category
    images_2plus_bboxes = {cat: 0 for cat in chexpert_categories}
    images_3plus_bboxes = {cat: 0 for cat in chexpert_categories}

    total_bboxes_per_image = []

    print("Processing images...")
    for idx, image_data in enumerate(data):
        # Convert to CheXpert
        chexpert_result = convert_image_to_chexpert_labels(image_data)

        # Count total bboxes
        total_bboxes = sum(info['bbox_count'] for info in chexpert_result.values())
        total_bboxes_per_image.append(total_bboxes)
        images_with_n_bboxes[total_bboxes] += 1

        # Per category statistics
        for category, info in chexpert_result.items():
            if info['present']:
                category_counts[category] += 1
                bbox_count = info['bbox_count']
                category_bbox_counts[category] += bbox_count
                category_bbox_distribution[category].append(bbox_count)
                images_with_n_bboxes_per_category[category][bbox_count] += 1

                if bbox_count >= 2:
                    images_2plus_bboxes[category] += 1
                if bbox_count >= 3:
                    images_3plus_bboxes[category] += 1

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} images...")

    # Print statistics
    print("\n" + "="*80)
    print("BBOX STATISTICS BY CHEXPERT CATEGORY")
    print("="*80)
    print(f"\n{'Category':<30} {'Images':>8} {'Total BBox':>10} {'2+ BBox':>8} {'3+ BBox':>8} {'Avg BBox':>10}")
    print("-"*80)

    for cat in chexpert_categories:
        if category_counts[cat] > 0:
            avg_bbox = category_bbox_counts[cat] / category_counts[cat]
            print(f"{cat:<30} {category_counts[cat]:>8} {category_bbox_counts[cat]:>10} "
                  f"{images_2plus_bboxes[cat]:>8} {images_3plus_bboxes[cat]:>8} {avg_bbox:>10.2f}")

    # Create visualizations
    create_visualizations(
        data,
        category_counts,
        category_bbox_counts,
        category_bbox_distribution,
        images_with_n_bboxes,
        images_with_n_bboxes_per_category,
        images_2plus_bboxes,
        images_3plus_bboxes,
        total_bboxes_per_image,
        chexpert_categories,
        output_dir
    )

    return {
        'category_counts': category_counts,
        'category_bbox_counts': category_bbox_counts,
        'images_2plus_bboxes': images_2plus_bboxes,
        'images_3plus_bboxes': images_3plus_bboxes
    }


def create_visualizations(data, category_counts, category_bbox_counts,
                         category_bbox_distribution, images_with_n_bboxes,
                         images_with_n_bboxes_per_category,
                         images_2plus_bboxes, images_3plus_bboxes,
                         total_bboxes_per_image, chexpert_categories, output_dir):
    """Create all visualization plots."""

    # 1. Category prevalence bar chart
    print("\nCreating Category Prevalence plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Sort by count
    sorted_cats = sorted([(cat, category_counts[cat]) for cat in chexpert_categories
                         if cat != 'No Finding'], key=lambda x: x[1], reverse=True)
    cats, counts = zip(*sorted_cats) if sorted_cats else ([], [])

    colors = sns.color_palette("husl", len(cats))
    ax1.barh(range(len(cats)), counts, color=colors)
    ax1.set_yticks(range(len(cats)))
    ax1.set_yticklabels(cats)
    ax1.set_xlabel('Number of Images')
    ax1.set_title('CheXpert Category Prevalence (excluding No Finding)')
    ax1.invert_yaxis()

    # Add counts to bars
    for i, count in enumerate(counts):
        ax1.text(count + 5, i, str(count), va='center')

    # Total bboxes per category
    sorted_bbox_cats = sorted([(cat, category_bbox_counts[cat]) for cat in chexpert_categories
                               if cat != 'No Finding' and category_bbox_counts[cat] > 0],
                              key=lambda x: x[1], reverse=True)
    cats_bbox, bbox_counts = zip(*sorted_bbox_cats) if sorted_bbox_cats else ([], [])

    colors = sns.color_palette("husl", len(cats_bbox))
    ax2.barh(range(len(cats_bbox)), bbox_counts, color=colors)
    ax2.set_yticks(range(len(cats_bbox)))
    ax2.set_yticklabels(cats_bbox)
    ax2.set_xlabel('Total Number of Bounding Boxes')
    ax2.set_title('Total Bounding Boxes by CheXpert Category')
    ax2.invert_yaxis()

    for i, count in enumerate(bbox_counts):
        ax2.text(count + 5, i, str(count), va='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/category_prevalence.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/category_prevalence.png")
    plt.close()

    # 2. BBox distribution per category
    print("\nCreating BBox Distribution per Category plot...")
    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    for idx, cat in enumerate(chexpert_categories):
        ax = axes[idx]
        if category_counts[cat] > 0:
            bbox_dist = category_bbox_distribution[cat]
            unique, counts = np.unique(bbox_dist, return_counts=True)

            ax.bar(unique, counts, color=sns.color_palette("husl", 1)[0], alpha=0.7)
            ax.set_xlabel('Number of BBoxes')
            ax.set_ylabel('Number of Images')
            ax.set_title(f'{cat}\n({category_counts[cat]} images, {category_bbox_counts[cat]} bboxes)')
            ax.grid(axis='y', alpha=0.3)

            # Add percentage labels
            total = len(bbox_dist)
            for u, c in zip(unique, counts):
                if c > 0:
                    pct = c / total * 100
                    ax.text(u, c, f'{pct:.1f}%', ha='center', va='bottom', fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(cat)
            ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/bbox_distribution_per_category.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/bbox_distribution_per_category.png")
    plt.close()

    # 3. Images with 2+ and 3+ bboxes
    print("\nCreating Multi-BBox Images plot...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # 2+ bboxes
    sorted_2plus = sorted([(cat, images_2plus_bboxes[cat]) for cat in chexpert_categories
                          if images_2plus_bboxes[cat] > 0], key=lambda x: x[1], reverse=True)
    if sorted_2plus:
        cats_2, counts_2 = zip(*sorted_2plus)
        colors = sns.color_palette("viridis", len(cats_2))
        ax1.barh(range(len(cats_2)), counts_2, color=colors)
        ax1.set_yticks(range(len(cats_2)))
        ax1.set_yticklabels(cats_2)
        ax1.set_xlabel('Number of Images')
        ax1.set_title('Images with 2+ Bounding Boxes per Category')
        ax1.invert_yaxis()

        for i, count in enumerate(counts_2):
            ax1.text(count + 2, i, str(count), va='center')

    # 3+ bboxes
    sorted_3plus = sorted([(cat, images_3plus_bboxes[cat]) for cat in chexpert_categories
                          if images_3plus_bboxes[cat] > 0], key=lambda x: x[1], reverse=True)
    if sorted_3plus:
        cats_3, counts_3 = zip(*sorted_3plus)
        colors = sns.color_palette("viridis", len(cats_3))
        ax2.barh(range(len(cats_3)), counts_3, color=colors)
        ax2.set_yticks(range(len(cats_3)))
        ax2.set_yticklabels(cats_3)
        ax2.set_xlabel('Number of Images')
        ax2.set_title('Images with 3+ Bounding Boxes per Category')
        ax2.invert_yaxis()

        for i, count in enumerate(counts_3):
            ax2.text(count + 1, i, str(count), va='center')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/multi_bbox_images.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/multi_bbox_images.png")
    plt.close()

    # 4. Overall bbox distribution histogram
    print("\nCreating Overall BBox Distribution plot...")
    fig, ax = plt.subplots(figsize=(12, 6))

    bbox_counts = [k for k, v in sorted(images_with_n_bboxes.items()) for _ in range(v)]
    if bbox_counts:
        unique, counts = np.unique(bbox_counts, return_counts=True)

        # Limit to first 15 bins for readability
        unique = unique[:15]
        counts = counts[:15]

        ax.bar(unique, counts, color=sns.color_palette("rocket", 1)[0], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Number of Bounding Boxes per Image')
        ax.set_ylabel('Number of Images')
        ax.set_title('Distribution of Bounding Boxes per Image (All Categories Combined)')
        ax.grid(axis='y', alpha=0.3)

        # Add count labels
        for u, c in zip(unique, counts):
            ax.text(u, c, str(c), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/overall_bbox_distribution.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/overall_bbox_distribution.png")
    plt.close()

    # 5. Heatmap of bbox counts per category
    print("\nCreating BBox Count Heatmap...")
    fig, ax = plt.subplots(figsize=(14, 8))

    # Create matrix: categories x bbox_counts (0-10)
    max_bbox = 10
    matrix = np.zeros((len(chexpert_categories), max_bbox + 1))

    for cat_idx, cat in enumerate(chexpert_categories):
        for bbox_count, image_count in images_with_n_bboxes_per_category[cat].items():
            if bbox_count <= max_bbox:
                matrix[cat_idx, bbox_count] = image_count

    sns.heatmap(matrix, annot=True, fmt='g', cmap='YlOrRd',
                xticklabels=list(range(max_bbox + 1)),
                yticklabels=chexpert_categories,
                cbar_kws={'label': 'Number of Images'},
                ax=ax)
    ax.set_xlabel('Number of Bounding Boxes')
    ax.set_ylabel('CheXpert Category')
    ax.set_title('Heatmap: Images by Category and BBox Count')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/bbox_count_heatmap.png", dpi=300, bbox_inches='tight')
    print(f"  Saved: {output_dir}/bbox_count_heatmap.png")
    plt.close()

    print("\nAll visualizations created successfully!")


if __name__ == "__main__":
    json_path = "data/padchest-gr/BIMCV-Padchest-GR /grounded_reports_20240819.json"

    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    results = analyze_and_visualize(json_path)
