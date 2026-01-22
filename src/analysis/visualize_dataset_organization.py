"""
Visualization of PadChest-GR dataset organization results.

Creates plots for:
1. BBox count distribution
2. CheXpert label distribution
3. Co-occurrence heatmap of labels
4. CheXpert labeler verification metrics
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Output directory for plots
OUTPUT_DIR = Path("data/padchest-gr/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# CheXpert categories (for consistent ordering)
CHEXPERT_CATEGORIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumothorax', 'Support Devices'
]


def load_summaries():
    """Load all summary files."""
    bbox_summary_path = Path("data/padchest-gr/chexpert-separated/summary.json")
    label_summary_path = Path("data/padchest-gr/chexpert-by-label/summary.json")
    verification_path = Path("data/padchest-gr/chexpert-separated/bbox_verification_results.json")

    summaries = {}

    if bbox_summary_path.exists():
        with open(bbox_summary_path) as f:
            summaries['bbox'] = json.load(f)

    if label_summary_path.exists():
        with open(label_summary_path) as f:
            summaries['label'] = json.load(f)

    if verification_path.exists():
        with open(verification_path) as f:
            summaries['verification'] = json.load(f)

    # Load category-specific summaries for co-occurrence data
    summaries['categories'] = {}
    label_dir = Path("data/padchest-gr/chexpert-by-label")
    for cat in CHEXPERT_CATEGORIES:
        cat_dir = label_dir / cat.replace(' ', '_')
        summary_file = cat_dir / "summary.json"
        if summary_file.exists():
            with open(summary_file) as f:
                summaries['categories'][cat] = json.load(f)

    return summaries


def plot_bbox_count_distribution(bbox_summary):
    """
    Plot distribution of images by bbox count.
    Creates both a bar chart and a cumulative line chart.
    """
    distribution = bbox_summary['bbox_count_distribution']
    cumulative = bbox_summary['cumulative_counts']

    # Convert to sorted lists
    bbox_counts = sorted([int(k) for k in distribution.keys()])
    counts = [distribution[str(bc)] for bc in bbox_counts]
    cumulative_counts = [cumulative[str(bc)] for bc in bbox_counts]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart for distribution
    ax1 = axes[0]
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(bbox_counts)))
    bars = ax1.bar(bbox_counts, counts, color=colors, edgecolor='navy', linewidth=0.5)
    ax1.set_xlabel('Number of BBoxes per Image', fontsize=12)
    ax1.set_ylabel('Number of Images', fontsize=12)
    ax1.set_title('Distribution of Images by BBox Count', fontsize=14, fontweight='bold')
    ax1.set_xticks(bbox_counts)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        if count > 0:
            ax1.annotate(f'{count}',
                        xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                        ha='center', va='bottom', fontsize=9)

    # Cumulative line chart
    ax2 = axes[1]
    ax2.fill_between(bbox_counts, cumulative_counts, alpha=0.3, color='steelblue')
    ax2.plot(bbox_counts, cumulative_counts, 'o-', color='steelblue', linewidth=2, markersize=8)
    ax2.set_xlabel('Minimum BBox Count', fontsize=12)
    ax2.set_ylabel('Number of Images with N+ BBoxes', fontsize=12)
    ax2.set_title('Cumulative Distribution: Images with N+ BBoxes', fontsize=14, fontweight='bold')
    ax2.set_xticks(bbox_counts)

    # Add annotations for key points
    for i, (bc, cc) in enumerate(zip(bbox_counts, cumulative_counts)):
        if bc <= 5 or cc < 50:
            ax2.annotate(f'{cc}', xy=(bc, cc), xytext=(0, 8),
                        textcoords='offset points', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'bbox_count_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'bbox_count_distribution.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'bbox_count_distribution.png'}")
    plt.close()


def plot_chexpert_label_distribution(label_summary):
    """
    Plot distribution of samples per CheXpert category.
    """
    samples_per_cat = label_summary['samples_per_category']

    # Sort by count
    sorted_cats = sorted(samples_per_cat.items(), key=lambda x: x[1], reverse=True)
    categories = [c[0] for c in sorted_cats]
    counts = [c[1] for c in sorted_cats]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Create horizontal bar chart
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(categories)))
    bars = ax.barh(categories, counts, color=colors, edgecolor='darkslategray', linewidth=0.5)

    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_title('Samples per CheXpert Category', fontsize=14, fontweight='bold')
    ax.invert_yaxis()  # Largest at top

    # Add value labels
    for bar, count in zip(bars, counts):
        ax.annotate(f'{count}',
                   xy=(bar.get_width(), bar.get_y() + bar.get_height()/2),
                   xytext=(5, 0), textcoords='offset points',
                   ha='left', va='center', fontsize=10)

    # Add statistics text
    stats = label_summary['statistics']
    stats_text = (
        f"Total images: {stats['total_images_in_source']}\n"
        f"Images with valid bboxes: {stats['images_with_valid_bboxes']}\n"
        f"Total bboxes: {stats['bboxes_valid']}\n"
        f"Filtered (negated): {stats['bboxes_filtered_negated']}"
    )
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chexpert_label_distribution.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'chexpert_label_distribution.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'chexpert_label_distribution.png'}")
    plt.close()


def plot_cooccurrence_heatmap(category_summaries):
    """
    Create a heatmap showing co-occurrence of different labels.
    """
    # Build co-occurrence matrix
    categories = [cat for cat in CHEXPERT_CATEGORIES if cat in category_summaries]
    n_cats = len(categories)
    cooccurrence = np.zeros((n_cats, n_cats))

    for i, cat in enumerate(categories):
        summary = category_summaries[cat]
        cooccur_dict = summary.get('co_occurring_categories', {})
        for j, other_cat in enumerate(categories):
            if other_cat in cooccur_dict:
                cooccurrence[i, j] = cooccur_dict[other_cat]
            elif i == j:
                # Diagonal: samples with same category bbox in image
                cooccurrence[i, j] = summary.get('samples_with_same_category_other', 0)

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Create heatmap
    mask = cooccurrence == 0
    sns.heatmap(cooccurrence, annot=True, fmt='.0f', cmap='YlOrRd',
                xticklabels=categories, yticklabels=categories,
                ax=ax, mask=mask, linewidths=0.5,
                cbar_kws={'label': 'Co-occurrence Count'})

    ax.set_title('Co-occurrence of CheXpert Labels in Same Image', fontsize=14, fontweight='bold')
    ax.set_xlabel('Co-occurring Category', fontsize=12)
    ax.set_ylabel('Main Category', fontsize=12)

    # Rotate x labels
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'label_cooccurrence_heatmap.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'label_cooccurrence_heatmap.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'label_cooccurrence_heatmap.png'}")
    plt.close()

    # Also create a normalized version (by row)
    fig, ax = plt.subplots(figsize=(12, 10))

    # Normalize by row (main category total)
    row_sums = cooccurrence.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # Avoid division by zero
    cooccurrence_norm = cooccurrence / row_sums * 100

    sns.heatmap(cooccurrence_norm, annot=True, fmt='.1f', cmap='YlOrRd',
                xticklabels=categories, yticklabels=categories,
                ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Co-occurrence Percentage (%)'})

    ax.set_title('Normalized Co-occurrence of CheXpert Labels (% of Main Category)',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Co-occurring Category', fontsize=12)
    ax.set_ylabel('Main Category', fontsize=12)

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'label_cooccurrence_normalized.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'label_cooccurrence_normalized.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'label_cooccurrence_normalized.png'}")
    plt.close()


def plot_verification_metrics(verification_summary):
    """
    Plot CheXpert labeler verification results (precision, recall, F1).
    """
    metrics = verification_summary['category_metrics']

    # Filter to categories with expected samples (exclude No Finding and Pneumonia which have 0 expected)
    valid_cats = [cat for cat in CHEXPERT_CATEGORIES
                  if cat in metrics and metrics[cat]['total_expected'] > 0]

    # Extract metrics
    precision = [metrics[cat]['precision'] for cat in valid_cats]
    recall = [metrics[cat]['recall'] for cat in valid_cats]
    f1 = [metrics[cat]['f1_score'] for cat in valid_cats]

    # Sort by F1 score
    sorted_indices = np.argsort(f1)[::-1]
    valid_cats = [valid_cats[i] for i in sorted_indices]
    precision = [precision[i] for i in sorted_indices]
    recall = [recall[i] for i in sorted_indices]
    f1 = [f1[i] for i in sorted_indices]

    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(14, 7))

    x = np.arange(len(valid_cats))
    width = 0.25

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2ecc71', edgecolor='darkgreen')
    bars2 = ax.bar(x, recall, width, label='Recall', color='#3498db', edgecolor='darkblue')
    bars3 = ax.bar(x + width, f1, width, label='F1 Score', color='#e74c3c', edgecolor='darkred')

    ax.set_xlabel('CheXpert Category', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('CheXpert Labeler Verification: Precision, Recall, and F1 Score',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(valid_cats, rotation=45, ha='right')
    ax.legend(loc='lower left')
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='0.5 threshold')

    # Add value labels on top of bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width()/2, height),
                           xytext=(0, 3), textcoords='offset points',
                           ha='center', va='bottom', fontsize=7, rotation=90)

    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chexpert_verification_metrics.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'chexpert_verification_metrics.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'chexpert_verification_metrics.png'}")
    plt.close()

    # Create confusion-style summary
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Match statistics pie chart
    ax1 = axes[0]
    match_stats = verification_summary['verification_stats']
    labels = ['Exact Match', 'Partial Match', 'No Match', 'Predicted Superset', 'Predicted Subset']
    sizes = [match_stats['exact_match'], match_stats['partial_match'],
             match_stats['no_match'], match_stats['predicted_superset'], match_stats['predicted_subset']]
    colors = ['#2ecc71', '#f1c40f', '#e74c3c', '#9b59b6', '#3498db']
    explode = (0.05, 0.02, 0.05, 0.02, 0.02)

    wedges, texts, autotexts = ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
                                        autopct='%1.1f%%', shadow=True, startangle=90)
    ax1.set_title('Verification Match Statistics', fontsize=14, fontweight='bold')

    # Add total count in center
    total = sum(sizes)
    ax1.text(0, 0, f'Total\n{total}', ha='center', va='center', fontsize=12, fontweight='bold')

    # TP/FP/FN bar chart for each category
    ax2 = axes[1]

    tp = [metrics[cat]['true_positives'] for cat in valid_cats]
    fp = [metrics[cat]['false_positives'] for cat in valid_cats]
    fn = [metrics[cat]['false_negatives'] for cat in valid_cats]

    x = np.arange(len(valid_cats))
    width = 0.25

    ax2.bar(x - width, tp, width, label='True Positives', color='#2ecc71')
    ax2.bar(x, fp, width, label='False Positives', color='#e74c3c')
    ax2.bar(x + width, fn, width, label='False Negatives', color='#f39c12')

    ax2.set_xlabel('CheXpert Category', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('True Positives, False Positives, False Negatives', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(valid_cats, rotation=45, ha='right')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'chexpert_verification_summary.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'chexpert_verification_summary.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'chexpert_verification_summary.png'}")
    plt.close()


def plot_category_details(category_summaries, label_summary):
    """
    Create detailed plots for category statistics.
    """
    categories = [cat for cat in CHEXPERT_CATEGORIES if cat in category_summaries]

    # Collect data
    total_samples = []
    unique_images = []
    with_other_bboxes = []
    uncertain = []
    minimal = []

    for cat in categories:
        summary = category_summaries[cat]
        total_samples.append(summary['total_samples'])
        unique_images.append(summary['unique_images'])
        with_other_bboxes.append(summary['samples_with_other_bboxes'])
        uncertain.append(summary['samples_uncertain'])
        minimal.append(summary['samples_minimal'])

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Samples vs Unique Images
    ax1 = axes[0, 0]
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax1.bar(x - width/2, total_samples, width, label='Total Samples', color='steelblue')
    bars2 = ax1.bar(x + width/2, unique_images, width, label='Unique Images', color='coral')
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Count')
    ax1.set_title('Total Samples vs Unique Images per Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()

    # 2. Samples with other bboxes percentage
    ax2 = axes[0, 1]
    with_other_pct = [w/t*100 if t > 0 else 0 for w, t in zip(with_other_bboxes, total_samples)]
    colors = plt.cm.RdYlGn(np.array(with_other_pct)/100)
    bars = ax2.barh(categories, with_other_pct, color=colors)
    ax2.set_xlabel('Percentage (%)')
    ax2.set_title('% of Samples with Other BBoxes in Same Image')
    ax2.set_xlim(0, 100)
    ax2.invert_yaxis()
    for bar, pct in zip(bars, with_other_pct):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=9)

    # 3. Uncertain vs Minimal samples
    ax3 = axes[1, 0]
    x = np.arange(len(categories))
    width = 0.35
    bars1 = ax3.bar(x - width/2, uncertain, width, label='Uncertain', color='#f39c12')
    bars2 = ax3.bar(x + width/2, minimal, width, label='Minimal', color='#95a5a6')
    ax3.set_xlabel('Category')
    ax3.set_ylabel('Count')
    ax3.set_title('Uncertain and Minimal Samples per Category')
    ax3.set_xticks(x)
    ax3.set_xticklabels(categories, rotation=45, ha='right')
    ax3.legend()

    # 4. Average other bboxes per sample
    ax4 = axes[1, 1]
    avg_other = [category_summaries[cat]['avg_other_bboxes_per_sample'] for cat in categories]
    sorted_idx = np.argsort(avg_other)[::-1]
    sorted_cats = [categories[i] for i in sorted_idx]
    sorted_avg = [avg_other[i] for i in sorted_idx]

    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(sorted_cats)))
    bars = ax4.barh(sorted_cats, sorted_avg, color=colors)
    ax4.set_xlabel('Average Other BBoxes per Sample')
    ax4.set_title('Average Number of Other BBoxes per Sample')
    ax4.invert_yaxis()
    for bar, avg in zip(bars, sorted_avg):
        ax4.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                f'{avg:.2f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'category_details.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'category_details.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'category_details.png'}")
    plt.close()


def create_summary_dashboard(summaries):
    """
    Create a comprehensive summary dashboard.
    """
    fig = plt.figure(figsize=(16, 12))

    # Title
    fig.suptitle('PadChest-GR Dataset Organization Summary', fontsize=16, fontweight='bold', y=0.98)

    # Create grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # 1. BBox count distribution (top left)
    if 'bbox' in summaries:
        ax1 = fig.add_subplot(gs[0, 0])
        distribution = summaries['bbox']['bbox_count_distribution']
        bbox_counts = sorted([int(k) for k in distribution.keys()])
        counts = [distribution[str(bc)] for bc in bbox_counts]
        ax1.bar(bbox_counts, counts, color='steelblue', edgecolor='navy')
        ax1.set_xlabel('BBox Count')
        ax1.set_ylabel('Images')
        ax1.set_title('BBox Count Distribution')

    # 2. CheXpert label distribution (top middle and right)
    if 'label' in summaries:
        ax2 = fig.add_subplot(gs[0, 1:])
        samples = summaries['label']['samples_per_category']
        sorted_items = sorted(samples.items(), key=lambda x: x[1], reverse=True)
        cats = [x[0] for x in sorted_items]
        vals = [x[1] for x in sorted_items]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(cats)))
        ax2.barh(cats, vals, color=colors)
        ax2.set_xlabel('Samples')
        ax2.set_title('Samples per CheXpert Category')
        ax2.invert_yaxis()

    # 3. Verification metrics (middle row)
    if 'verification' in summaries:
        ax3 = fig.add_subplot(gs[1, :])
        metrics = summaries['verification']['category_metrics']
        valid_cats = [cat for cat in CHEXPERT_CATEGORIES
                      if cat in metrics and metrics[cat]['total_expected'] > 0]
        f1_scores = [(cat, metrics[cat]['f1_score']) for cat in valid_cats]
        f1_scores.sort(key=lambda x: x[1], reverse=True)
        cats = [x[0] for x in f1_scores]
        f1 = [x[1] for x in f1_scores]
        precision = [metrics[cat]['precision'] for cat in cats]
        recall = [metrics[cat]['recall'] for cat in cats]

        x = np.arange(len(cats))
        width = 0.25
        ax3.bar(x - width, precision, width, label='Precision', color='#2ecc71')
        ax3.bar(x, recall, width, label='Recall', color='#3498db')
        ax3.bar(x + width, f1, width, label='F1', color='#e74c3c')
        ax3.set_xticks(x)
        ax3.set_xticklabels(cats, rotation=45, ha='right')
        ax3.set_ylabel('Score')
        ax3.set_title('CheXpert Labeler Verification Metrics')
        ax3.legend(loc='lower left')
        ax3.set_ylim(0, 1.1)

    # 4. Match statistics (bottom left)
    if 'verification' in summaries:
        ax4 = fig.add_subplot(gs[2, 0])
        stats = summaries['verification']['verification_stats']
        labels = ['Exact', 'Partial', 'None']
        sizes = [stats['exact_match'], stats['partial_match'], stats['no_match']]
        colors = ['#2ecc71', '#f1c40f', '#e74c3c']
        ax4.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax4.set_title('Match Statistics')

    # 5. Key statistics text (bottom middle)
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis('off')

    stats_text = "Key Statistics\n" + "="*30 + "\n\n"
    if 'bbox' in summaries:
        stats_text += f"Total images: {summaries['bbox']['total_images_in_dataset']}\n"
        stats_text += f"Images with bboxes: {summaries['bbox']['total_images_with_chexpert_bboxes']}\n"
        stats_text += f"Max bboxes/image: {summaries['bbox']['max_bbox_count']}\n\n"
    if 'label' in summaries:
        s = summaries['label']['statistics']
        stats_text += f"Valid bboxes: {s['bboxes_valid']}\n"
        stats_text += f"Filtered (negated): {s['bboxes_filtered_negated']}\n"
        stats_text += f"Categories: {len(summaries['label']['categories_available'])}\n"

    ax5.text(0.1, 0.9, stats_text, transform=ax5.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # 6. Co-occurrence mini heatmap (bottom right)
    if 'categories' in summaries and summaries['categories']:
        ax6 = fig.add_subplot(gs[2, 2])
        categories = [cat for cat in CHEXPERT_CATEGORIES[:6] if cat in summaries['categories']]
        n = len(categories)
        cooc = np.zeros((n, n))
        for i, cat in enumerate(categories):
            cooc_dict = summaries['categories'][cat].get('co_occurring_categories', {})
            for j, other in enumerate(categories):
                if other in cooc_dict:
                    cooc[i, j] = cooc_dict[other]

        sns.heatmap(cooc, annot=True, fmt='.0f', cmap='YlOrRd',
                   xticklabels=[c[:8] for c in categories],
                   yticklabels=[c[:8] for c in categories],
                   ax=ax6, cbar=False)
        ax6.set_title('Co-occurrence (Top 6)')
        ax6.tick_params(axis='x', rotation=45)

    plt.savefig(OUTPUT_DIR / 'summary_dashboard.png', dpi=150, bbox_inches='tight')
    plt.savefig(OUTPUT_DIR / 'summary_dashboard.pdf', bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR / 'summary_dashboard.png'}")
    plt.close()


def main():
    """Generate all visualizations."""
    print("="*60)
    print("PadChest-GR Dataset Organization Visualization")
    print("="*60)

    # Load all summaries
    print("\nLoading summary files...")
    summaries = load_summaries()

    if not summaries:
        print("Error: No summary files found!")
        return

    print(f"Loaded: {list(summaries.keys())}")

    # Generate plots
    if 'bbox' in summaries:
        print("\n1. Generating BBox count distribution plot...")
        plot_bbox_count_distribution(summaries['bbox'])

    if 'label' in summaries:
        print("\n2. Generating CheXpert label distribution plot...")
        plot_chexpert_label_distribution(summaries['label'])

    if 'categories' in summaries and summaries['categories']:
        print("\n3. Generating co-occurrence heatmap...")
        plot_cooccurrence_heatmap(summaries['categories'])

        if 'label' in summaries:
            print("\n4. Generating category details plots...")
            plot_category_details(summaries['categories'], summaries['label'])

    if 'verification' in summaries:
        print("\n5. Generating verification metrics plots...")
        plot_verification_metrics(summaries['verification'])

    print("\n6. Generating summary dashboard...")
    create_summary_dashboard(summaries)

    print("\n" + "="*60)
    print(f"All plots saved to: {OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
