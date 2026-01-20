"""
Analyze the distribution of the PadChest-GR dataset.

This script analyzes:
- Location distribution
- Label distribution (and mapping to 14 CheXpert labels used in RRG)
- Total bounding boxes per image
"""

import json
import sys
from collections import Counter, defaultdict
import numpy as np
from pathlib import Path

# 14 CheXpert labels used in RRG
CHEXPERT_LABELS = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]


def analyze_dataset(json_path):
    """Analyze the PadChest-GR dataset."""

    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"\nTotal number of images: {len(data)}")

    # Counters
    location_counter = Counter()
    label_counter = Counter()
    bbox_per_image = []
    extra_bbox_per_image = []

    # Track unique labels and locations
    unique_labels = set()
    unique_locations = set()

    # Statistics
    images_with_boxes = 0
    images_with_extra_boxes = 0
    images_with_abnormal_findings = 0
    images_with_normal_findings = 0
    total_findings = 0
    abnormal_findings = 0
    normal_findings = 0

    # Label matching analysis
    label_to_chexpert_mapping = defaultdict(set)
    unmatched_labels = set()

    print("\nAnalyzing images...")
    for img_idx, image_data in enumerate(data):
        image_id = image_data['ImageID']
        findings = image_data.get('findings', [])

        total_findings += len(findings)

        # Count bboxes for this image
        image_bbox_count = 0
        image_extra_bbox_count = 0
        has_abnormal = False
        has_normal = False

        for finding in findings:
            # Check if abnormal
            is_abnormal = finding.get('abnormal', False)

            if is_abnormal:
                abnormal_findings += 1
                has_abnormal = True

                # Count locations
                locations = finding.get('locations', [])
                for loc in locations:
                    location_counter[loc] += 1
                    unique_locations.add(loc)

                # Count labels
                labels = finding.get('labels', [])
                for label in labels:
                    label_counter[label] += 1
                    unique_labels.add(label)

                    # Try to match to CheXpert labels
                    matched = False
                    for chexpert_label in CHEXPERT_LABELS:
                        if chexpert_label.lower() in label.lower() or label.lower() in chexpert_label.lower():
                            label_to_chexpert_mapping[label].add(chexpert_label)
                            matched = True

                    if not matched:
                        unmatched_labels.add(label)

                # Count bounding boxes
                boxes = finding.get('boxes', [])
                extra_boxes = finding.get('extra_boxes', [])

                image_bbox_count += len(boxes)
                image_extra_bbox_count += len(extra_boxes)
            else:
                normal_findings += 1
                has_normal = True

        # Image-level statistics
        bbox_per_image.append(image_bbox_count)
        extra_bbox_per_image.append(image_extra_bbox_count)

        if image_bbox_count > 0:
            images_with_boxes += 1
        if image_extra_bbox_count > 0:
            images_with_extra_boxes += 1
        if has_abnormal:
            images_with_abnormal_findings += 1
        if has_normal:
            images_with_normal_findings += 1

        # Progress
        if (img_idx + 1) % 1000 == 0:
            print(f"  Processed {img_idx + 1} images...")

    # Print results
    print("\n" + "="*80)
    print("DATASET STATISTICS")
    print("="*80)

    print(f"\nTotal images: {len(data)}")
    print(f"Images with abnormal findings: {images_with_abnormal_findings}")
    print(f"Images with normal findings: {images_with_normal_findings}")
    print(f"Total findings: {total_findings}")
    print(f"  - Abnormal findings: {abnormal_findings}")
    print(f"  - Normal findings: {normal_findings}")

    print("\n" + "-"*80)
    print("BOUNDING BOX STATISTICS")
    print("-"*80)

    bbox_array = np.array(bbox_per_image)
    extra_bbox_array = np.array(extra_bbox_per_image)

    print(f"\nImages with bounding boxes: {images_with_boxes} ({images_with_boxes/len(data)*100:.2f}%)")
    print(f"Images with extra bounding boxes: {images_with_extra_boxes} ({images_with_extra_boxes/len(data)*100:.2f}%)")

    print(f"\nBounding boxes per image:")
    print(f"  Mean: {bbox_array.mean():.2f}")
    print(f"  Median: {np.median(bbox_array):.2f}")
    print(f"  Min: {bbox_array.min()}")
    print(f"  Max: {bbox_array.max()}")
    print(f"  Std: {bbox_array.std():.2f}")
    print(f"  Total bboxes: {bbox_array.sum()}")

    print(f"\nExtra bounding boxes per image:")
    print(f"  Mean: {extra_bbox_array.mean():.2f}")
    print(f"  Median: {np.median(extra_bbox_array):.2f}")
    print(f"  Min: {extra_bbox_array.min()}")
    print(f"  Max: {extra_bbox_array.max()}")
    print(f"  Std: {extra_bbox_array.std():.2f}")
    print(f"  Total extra bboxes: {extra_bbox_array.sum()}")

    print("\n" + "-"*80)
    print("LOCATION DISTRIBUTION")
    print("-"*80)

    print(f"\nTotal unique locations: {len(unique_locations)}")
    print(f"Total location mentions: {sum(location_counter.values())}")
    print(f"\nTop 20 locations:")
    for loc, count in location_counter.most_common(20):
        print(f"  {loc}: {count}")

    print("\n" + "-"*80)
    print("LABEL DISTRIBUTION")
    print("-"*80)

    print(f"\nTotal unique labels: {len(unique_labels)}")
    print(f"Total label mentions: {sum(label_counter.values())}")
    print(f"\nTop 30 labels:")
    for label, count in label_counter.most_common(30):
        print(f"  {label}: {count}")

    print("\n" + "-"*80)
    print("CHEXPERT LABEL MAPPING ANALYSIS")
    print("-"*80)

    print(f"\nCheXpert labels used in RRG (14 labels):")
    for i, label in enumerate(CHEXPERT_LABELS, 1):
        print(f"  {i:2d}. {label}")

    print(f"\n\nPadChest labels that match CheXpert labels:")
    matched_count = 0
    for padchest_label in sorted(label_to_chexpert_mapping.keys()):
        chexpert_matches = label_to_chexpert_mapping[padchest_label]
        count = label_counter[padchest_label]
        matched_count += count
        print(f"  '{padchest_label}' ({count}) -> {chexpert_matches}")

    print(f"\n\nPadChest labels that DON'T match CheXpert labels:")
    unmatched_count = 0
    for label in sorted(unmatched_labels):
        count = label_counter[label]
        unmatched_count += count
        print(f"  '{label}' ({count})")

    print(f"\n\nMatching summary:")
    print(f"  Matched label mentions: {matched_count} ({matched_count/sum(label_counter.values())*100:.2f}%)")
    print(f"  Unmatched label mentions: {unmatched_count} ({unmatched_count/sum(label_counter.values())*100:.2f}%)")
    print(f"  Unique matched labels: {len(label_to_chexpert_mapping)}")
    print(f"  Unique unmatched labels: {len(unmatched_labels)}")

    # Distribution of bboxes per image (histogram)
    print("\n" + "-"*80)
    print("BOUNDING BOX DISTRIBUTION")
    print("-"*80)

    print("\nHistogram of bounding boxes per image:")
    unique, counts = np.unique(bbox_array, return_counts=True)
    for val, cnt in zip(unique[:20], counts[:20]):  # Show first 20 bins
        print(f"  {int(val):3d} bboxes: {cnt:5d} images")
    if len(unique) > 20:
        print(f"  ... (showing first 20 out of {len(unique)} unique values)")

    return {
        'total_images': len(data),
        'location_counter': location_counter,
        'label_counter': label_counter,
        'bbox_per_image': bbox_array,
        'extra_bbox_per_image': extra_bbox_array,
        'unique_labels': unique_labels,
        'unique_locations': unique_locations,
        'label_to_chexpert_mapping': dict(label_to_chexpert_mapping),
        'unmatched_labels': unmatched_labels
    }


if __name__ == "__main__":
    # Default path
    json_path = "data/padchest-gr/BIMCV-Padchest-GR /grounded_reports_20240819.json"

    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    results = analyze_dataset(json_path)
