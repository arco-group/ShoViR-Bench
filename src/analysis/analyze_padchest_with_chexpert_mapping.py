"""
Analyze PadChest-GR dataset using the CheXpert label mapping.

This script converts PadChest labels to CheXpert categories and provides statistics
on the dataset distribution in terms of the 14 CheXpert labels.
"""

import json
import sys
from collections import Counter
import numpy as np
from pathlib import Path

# Import the mapping
from padchest_to_chexpert_mapping import (
    convert_image_to_chexpert_labels,
    CHEXPERT_TO_PADCHEST_MAPPING,
    map_padchest_to_chexpert
)


def analyze_with_chexpert_mapping(json_path):
    """Analyze PadChest-GR dataset using CheXpert label mapping."""

    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Total number of images: {len(data)}\n")

    # CheXpert category counters
    chexpert_categories = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
    ]

    # Count images with each category
    category_counts = {cat: 0 for cat in chexpert_categories}
    category_bbox_counts = {cat: 0 for cat in chexpert_categories}
    category_label_sources = {cat: Counter() for cat in chexpert_categories}

    # Co-occurrence matrix
    co_occurrence = {cat: Counter() for cat in chexpert_categories}

    # Track unmapped labels
    unmapped_label_counts = Counter()
    total_unmapped = 0

    print("Processing images...")
    for idx, image_data in enumerate(data):
        # Convert to CheXpert labels
        chexpert_result = convert_image_to_chexpert_labels(image_data)

        # Track which categories are present in this image
        present_categories = []

        for category, info in chexpert_result.items():
            if info['present']:
                category_counts[category] += 1
                category_bbox_counts[category] += info['bbox_count']
                present_categories.append(category)

                # Track which PadChest labels contributed
                for padchest_label in info['padchest_labels']:
                    category_label_sources[category][padchest_label] += 1

        # Update co-occurrence matrix
        for cat1 in present_categories:
            for cat2 in present_categories:
                if cat1 != cat2:
                    co_occurrence[cat1][cat2] += 1

        # Track unmapped labels
        for finding in image_data.get('findings', []):
            if finding.get('abnormal', False):
                for label in finding.get('labels', []):
                    if not map_padchest_to_chexpert(label):
                        unmapped_label_counts[label] += 1
                        total_unmapped += 1

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} images...")

    # Print results
    print("\n" + "="*80)
    print("CHEXPERT CATEGORY DISTRIBUTION (AFTER MAPPING)")
    print("="*80)

    print(f"\nTotal images: {len(data)}")
    print("\nCategory prevalence:")
    print("-" * 80)

    # Sort by count
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    for category, count in sorted_categories:
        percentage = count / len(data) * 100
        avg_bbox = category_bbox_counts[category] / count if count > 0 else 0
        total_bbox = category_bbox_counts[category]
        print(f"{category:30s}: {count:5d} images ({percentage:5.2f}%) | {total_bbox:5d} bboxes (avg: {avg_bbox:.2f})")

    # Print label sources for each category
    print("\n" + "="*80)
    print("PADCHEST LABEL SOURCES FOR EACH CHEXPERT CATEGORY")
    print("="*80)

    for category in chexpert_categories:
        sources = category_label_sources[category]
        if sources:
            print(f"\n{category} ({category_counts[category]} images):")
            for label, count in sources.most_common():
                print(f"  {label:40s}: {count:5d}")

    # Co-occurrence analysis
    print("\n" + "="*80)
    print("CATEGORY CO-OCCURRENCE (Top 5 pairs for each category)")
    print("="*80)

    for category in chexpert_categories:
        if category_counts[category] > 0:
            top_cooccur = co_occurrence[category].most_common(5)
            if top_cooccur:
                print(f"\n{category}:")
                for co_cat, count in top_cooccur:
                    percentage = count / category_counts[category] * 100
                    print(f"  + {co_cat:30s}: {count:4d} images ({percentage:5.1f}%)")

    # Unmapped labels
    print("\n" + "="*80)
    print("UNMAPPED PADCHEST LABELS (Top 30)")
    print("="*80)
    print(f"\nTotal unmapped label mentions: {total_unmapped}")
    print(f"Unique unmapped labels: {len(unmapped_label_counts)}\n")

    for label, count in unmapped_label_counts.most_common(30):
        print(f"  {label:40s}: {count:5d}")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    images_with_findings = len(data) - category_counts['No Finding']
    images_with_mapped_findings = sum(1 for cat, count in category_counts.items()
                                       if cat != 'No Finding' and count > 0)

    total_mapped_labels = sum(sum(sources.values()) for sources in category_label_sources.values())
    total_labels_in_dataset = total_mapped_labels + total_unmapped

    print(f"\nImages with any finding: {images_with_findings} ({images_with_findings/len(data)*100:.2f}%)")
    print(f"Images with 'No Finding': {category_counts['No Finding']} ({category_counts['No Finding']/len(data)*100:.2f}%)")
    print(f"\nCheXpert categories with at least one image: {images_with_mapped_findings}/14")
    print(f"\nTotal label mentions: {total_labels_in_dataset}")
    print(f"  Mapped to CheXpert: {total_mapped_labels} ({total_mapped_labels/total_labels_in_dataset*100:.2f}%)")
    print(f"  Unmapped: {total_unmapped} ({total_unmapped/total_labels_in_dataset*100:.2f}%)")

    # Average categories per image
    total_categories_mentioned = sum(category_counts.values()) - category_counts['No Finding']
    avg_categories = total_categories_mentioned / len(data)
    print(f"\nAverage CheXpert categories per image: {avg_categories:.2f}")

    return {
        'category_counts': category_counts,
        'category_bbox_counts': category_bbox_counts,
        'category_label_sources': category_label_sources,
        'co_occurrence': co_occurrence,
        'unmapped_label_counts': unmapped_label_counts
    }


if __name__ == "__main__":
    # Default path
    json_path = "data/padchest-gr/BIMCV-Padchest-GR /grounded_reports_20240819.json"

    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    results = analyze_with_chexpert_mapping(json_path)
