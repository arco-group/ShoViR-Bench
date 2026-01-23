"""
Organize PadChest-GR dataset by CheXpert labels for evaluation.

This script creates files organized by CheXpert label where:
- Each sample has a "main_bbox" with the target CheXpert label
- Each sample includes "other_bboxes" in the same image (can be same or different labels)
- False negatives are filtered out using text analysis of bbox annotations

Output structure:
    data/padchest-gr/chexpert-by-label/
    ├── Atelectasis/
    │   ├── samples.json        # All samples with Atelectasis as main bbox
    │   └── summary.json        # Statistics for this category
    ├── Cardiomegaly/
    │   └── ...
    └── summary.json            # Overall statistics
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional

from padchest_to_chexpert_mapping import convert_image_to_chexpert_labels, PADCHEST_TO_CHEXPERT_MAPPING


# CheXpert categories (excluding No Finding for evaluation)
CHEXPERT_CATEGORIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]


class NegationDetector:
    """
    Detects negated findings in radiology report text.
    Uses pattern matching to identify false negatives.
    """

    def __init__(self):
        # Negation patterns that indicate absence of finding
        self.negation_patterns = [
            # Direct negation
            r'\bno\s+(?:evidence\s+of\s+)?',
            r'\bno\b',
            r'\bnot\b',
            r'\bwithout\b',
            r'\babsent\b',
            r'\babsence\s+of\b',
            r'\bnegative\s+for\b',
            r'\brule\s*out\b',
            r'\br/o\b',
            r'\bdenies\b',
            r'\bdenied\b',
            r'\bfree\s+of\b',
            r'\bclear\s+of\b',
            r'\bunremarkable\b',
            r'\bnormal\b',

            # Resolved/improved conditions
            r'\bresolved\b',
            r'\bhas\s+resolved\b',
            r'\bcleared\b',
            r'\bimproved\b',
            r'\bno\s+longer\s+present\b',
            r'\bno\s+longer\s+seen\b',
            r'\bno\s+longer\s+visible\b',

            # Excluded conditions
            r'\bexcluded\b',
            r'\bexcludes\b',
            r'\brules\s+out\b',

            # Comparison suggesting absence
            r'\bwithout\s+evidence\b',
            r'\bno\s+definite\b',
            r'\bno\s+significant\b',
            r'\bno\s+acute\b',
            r'\bno\s+new\b',
        ]

        # Uncertainty patterns that suggest possible false positive
        self.uncertainty_patterns = [
            r'\bpossible\b',
            r'\bprobable\b',
            r'\blikely\b',
            r'\bquestionable\b',
            r'\bsuspected\b',
            r'\bsuspicious\b',
            r'\bcannot\s+exclude\b',
            r'\bcannot\s+rule\s+out\b',
            r'\bmay\s+represent\b',
            r'\bmay\s+be\b',
            r'\bif\s+clinical\b',
            r'\bcorrelate\b',
        ]

        # Minimal/borderline patterns
        self.minimal_patterns = [
            r'\bminimal\b',
            r'\bminor\b',
            r'\bsubtle\b',
            r'\btrace\b',
            r'\btiny\b',
            r'\bsmall\b',
            r'\bborderline\b',
            r'\bat\s+the\s+(?:upper|lower)\s+limit\s+of\s+normal\b',
            r'\bwithin\s+normal\s+limits\b',
        ]

    def is_negated(self, text: str) -> Tuple[bool, str]:
        """
        Check if the text indicates negation of a finding.

        Args:
            text: The sentence describing the finding

        Returns:
            Tuple of (is_negated, reason)
        """
        text_lower = text.lower()

        for pattern in self.negation_patterns:
            if re.search(pattern, text_lower):
                return True, f"Negation pattern: {pattern}"

        return False, ""

    def is_uncertain(self, text: str) -> Tuple[bool, str]:
        """
        Check if the text indicates uncertainty about a finding.

        Args:
            text: The sentence describing the finding

        Returns:
            Tuple of (is_uncertain, reason)
        """
        text_lower = text.lower()

        for pattern in self.uncertainty_patterns:
            if re.search(pattern, text_lower):
                return True, f"Uncertainty pattern: {pattern}"

        return False, ""

    def is_minimal(self, text: str) -> Tuple[bool, str]:
        """
        Check if the text indicates minimal/borderline finding.

        Args:
            text: The sentence describing the finding

        Returns:
            Tuple of (is_minimal, reason)
        """
        text_lower = text.lower()

        for pattern in self.minimal_patterns:
            if re.search(pattern, text_lower):
                return True, f"Minimal pattern: {pattern}"

        return False, ""

    def analyze_finding(self, text: str) -> Dict:
        """
        Full analysis of a finding text.

        Returns dict with:
            - is_valid: True if finding should be included
            - is_negated: True if negation detected
            - is_uncertain: True if uncertainty detected
            - is_minimal: True if minimal finding
            - reason: Explanation for any flags
        """
        is_negated, neg_reason = self.is_negated(text)
        is_uncertain, unc_reason = self.is_uncertain(text)
        is_minimal, min_reason = self.is_minimal(text)

        # A finding is valid if not negated
        # Uncertain and minimal findings are still valid but flagged
        is_valid = not is_negated

        reasons = []
        if neg_reason:
            reasons.append(neg_reason)
        if unc_reason:
            reasons.append(unc_reason)
        if min_reason:
            reasons.append(min_reason)

        return {
            'is_valid': is_valid,
            'is_negated': is_negated,
            'is_uncertain': is_uncertain,
            'is_minimal': is_minimal,
            'reason': '; '.join(reasons) if reasons else 'Clean finding'
        }


def extract_bboxes_with_labels(image_data: Dict, negation_detector: NegationDetector) -> List[Dict]:
    """
    Extract all bboxes from an image with their CheXpert labels and validation status.

    Args:
        image_data: The image data dictionary
        negation_detector: NegationDetector instance

    Returns:
        List of bbox dictionaries with labels and validation info
    """
    bboxes = []

    for finding in image_data.get('findings', []):
        # Skip non-abnormal findings
        if not finding.get('abnormal', False):
            continue

        # Skip findings without boxes
        boxes = finding.get('boxes', [])
        if not boxes:
            continue

        # Get CheXpert labels for this finding
        finding_labels = finding.get('labels', [])
        chexpert_cats = set()
        for label in finding_labels:
            if label in PADCHEST_TO_CHEXPERT_MAPPING:
                chexpert_cat = PADCHEST_TO_CHEXPERT_MAPPING[label]
                if chexpert_cat != 'No Finding':
                    chexpert_cats.add(chexpert_cat)

        if not chexpert_cats:
            continue

        # Analyze text for negation/uncertainty
        sentence_en = finding.get('sentence_en', '')
        text_analysis = negation_detector.analyze_finding(sentence_en)

        # Create bbox entry for each box in this finding
        for box_idx, box in enumerate(boxes):
            bbox_entry = {
                'box': box,
                'chexpert_categories': list(chexpert_cats),
                'padchest_labels': finding_labels,
                'sentence_en': sentence_en,
                'sentence_es': finding.get('sentence_es', ''),
                'locations': finding.get('locations', []),
                'is_valid': text_analysis['is_valid'],
                'is_negated': text_analysis['is_negated'],
                'is_uncertain': text_analysis['is_uncertain'],
                'is_minimal': text_analysis['is_minimal'],
                'validation_reason': text_analysis['reason'],
                'box_index': box_idx,
                'total_boxes_in_finding': len(boxes)
            }
            bboxes.append(bbox_entry)

    return bboxes


def organize_by_chexpert_label(
    json_path: str,
    output_dir: str = "data/padchest-gr/chexpert-by-label",
    filter_negated: bool = True,
    filter_uncertain: bool = False,
    include_minimal: bool = True
):
    """
    Organize dataset into separate directories by CheXpert label.

    Each sample contains:
    - main_bbox: The bbox with the target CheXpert label (for evaluation)
    - other_bboxes: Other bboxes in the same image (context)
    - image_info: Image metadata

    Args:
        json_path: Path to the original PadChest-GR JSON
        output_dir: Output directory for organized files
        filter_negated: If True, filter out negated findings (default: True)
        filter_uncertain: If True, filter out uncertain findings (default: False)
        include_minimal: If True, include minimal/borderline findings (default: True)
    """

    print("=" * 80)
    print("ORGANIZING DATASET BY CHEXPERT LABEL")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Filter negated findings: {filter_negated}")
    print(f"  Filter uncertain findings: {filter_uncertain}")
    print(f"  Include minimal findings: {include_minimal}")

    # Load dataset
    print(f"\nLoading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Total images: {len(data)}")

    # Initialize
    negation_detector = NegationDetector()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Storage for samples by category
    samples_by_category = defaultdict(list)

    # Statistics
    stats = {
        'total_images': len(data),
        'images_with_valid_bboxes': 0,
        'total_bboxes_extracted': 0,
        'bboxes_filtered_negated': 0,
        'bboxes_filtered_uncertain': 0,
        'bboxes_valid': 0,
        'samples_per_category': Counter(),
        'co_occurrence': defaultdict(Counter),  # category -> other categories in same image
        'filtering_examples': {
            'negated': [],
            'uncertain': [],
            'valid': []
        }
    }

    print("\nProcessing images...")
    for idx, image_data in enumerate(data):
        # Extract all bboxes with validation
        all_bboxes = extract_bboxes_with_labels(image_data, negation_detector)
        stats['total_bboxes_extracted'] += len(all_bboxes)

        if not all_bboxes:
            continue

        # Filter bboxes based on settings
        valid_bboxes = []
        for bbox in all_bboxes:
            # Track filtering
            if bbox['is_negated']:
                stats['bboxes_filtered_negated'] += 1
                if len(stats['filtering_examples']['negated']) < 10:
                    stats['filtering_examples']['negated'].append({
                        'sentence': bbox['sentence_en'],
                        'categories': bbox['chexpert_categories'],
                        'reason': bbox['validation_reason']
                    })
                if filter_negated:
                    continue

            if bbox['is_uncertain']:
                stats['bboxes_filtered_uncertain'] += 1
                if len(stats['filtering_examples']['uncertain']) < 10:
                    stats['filtering_examples']['uncertain'].append({
                        'sentence': bbox['sentence_en'],
                        'categories': bbox['chexpert_categories'],
                        'reason': bbox['validation_reason']
                    })
                if filter_uncertain:
                    continue

            if bbox['is_minimal'] and not include_minimal:
                continue

            valid_bboxes.append(bbox)
            stats['bboxes_valid'] += 1

            if len(stats['filtering_examples']['valid']) < 10 and bbox['is_valid']:
                stats['filtering_examples']['valid'].append({
                    'sentence': bbox['sentence_en'],
                    'categories': bbox['chexpert_categories'],
                    'reason': bbox['validation_reason']
                })

        if not valid_bboxes:
            continue

        stats['images_with_valid_bboxes'] += 1

        # Get all categories in this image for co-occurrence tracking
        all_categories_in_image = set()
        for bbox in valid_bboxes:
            all_categories_in_image.update(bbox['chexpert_categories'])

        # Create samples: one for each bbox as "main" bbox
        image_info = {
            'ImageID': image_data.get('ImageID', ''),
            'StudyID': image_data.get('StudyID', ''),
            'PreviousImageID': image_data.get('PreviousImageID'),
            'PreviousStudyID': image_data.get('PreviousStudyID'),
        }

        for main_idx, main_bbox in enumerate(valid_bboxes):
            # Create list of other bboxes in the image
            other_bboxes = [
                bbox for bbox_idx, bbox in enumerate(valid_bboxes)
                if bbox_idx != main_idx
            ]

            # Create sample for each CheXpert category this bbox belongs to
            for category in main_bbox['chexpert_categories']:
                sample = {
                    'image_info': image_info,
                    'main_bbox': {
                        'box': main_bbox['box'],
                        'chexpert_category': category,
                        'all_chexpert_categories': main_bbox['chexpert_categories'],
                        'padchest_labels': main_bbox['padchest_labels'],
                        'sentence_en': main_bbox['sentence_en'],
                        'sentence_es': main_bbox['sentence_es'],
                        'locations': main_bbox['locations'],
                        'is_uncertain': main_bbox['is_uncertain'],
                        'is_minimal': main_bbox['is_minimal'],
                    },
                    'other_bboxes': [
                        {
                            'box': ob['box'],
                            'chexpert_categories': ob['chexpert_categories'],
                            'padchest_labels': ob['padchest_labels'],
                            'sentence_en': ob['sentence_en'],
                            'locations': ob['locations'],
                            'same_category': category in ob['chexpert_categories']
                        }
                        for ob in other_bboxes
                    ],
                    'total_bboxes_in_image': len(valid_bboxes),
                    'other_categories_in_image': list(all_categories_in_image - {category})
                }

                samples_by_category[category].append(sample)
                stats['samples_per_category'][category] += 1

                # Track co-occurrence
                for other_cat in all_categories_in_image:
                    if other_cat != category:
                        stats['co_occurrence'][category][other_cat] += 1

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1} images...")

    print(f"\nProcessing complete!")
    print(f"Images with valid bboxes: {stats['images_with_valid_bboxes']}")
    print(f"Total valid bboxes: {stats['bboxes_valid']}")

    # Save files for each category
    print(f"\nSaving organized files to: {output_path}")

    for category in CHEXPERT_CATEGORIES:
        samples = samples_by_category[category]

        if not samples:
            print(f"  {category}: No samples found")
            continue

        # Create category directory
        cat_dir = output_path / category.replace(' ', '_')
        cat_dir.mkdir(parents=True, exist_ok=True)

        # Save samples
        samples_file = cat_dir / "samples.json"
        with open(samples_file, 'w') as f:
            json.dump(samples, f, indent=2)

        # Calculate category statistics
        n_samples = len(samples)
        n_with_other_bboxes = sum(1 for s in samples if len(s['other_bboxes']) > 0)
        n_with_same_cat_other = sum(
            1 for s in samples
            if any(ob['same_category'] for ob in s['other_bboxes'])
        )
        n_uncertain = sum(1 for s in samples if s['main_bbox']['is_uncertain'])
        n_minimal = sum(1 for s in samples if s['main_bbox']['is_minimal'])

        avg_other_bboxes = sum(len(s['other_bboxes']) for s in samples) / n_samples if n_samples > 0 else 0

        # Unique images
        unique_images = len(set(s['image_info']['ImageID'] for s in samples))

        cat_summary = {
            'category': category,
            'total_samples': n_samples,
            'unique_images': unique_images,
            'samples_with_other_bboxes': n_with_other_bboxes,
            'samples_with_same_category_other': n_with_same_cat_other,
            'samples_uncertain': n_uncertain,
            'samples_minimal': n_minimal,
            'avg_other_bboxes_per_sample': round(avg_other_bboxes, 2),
            'co_occurring_categories': dict(stats['co_occurrence'][category].most_common(10))
        }

        summary_file = cat_dir / "summary.json"
        with open(summary_file, 'w') as f:
            json.dump(cat_summary, f, indent=2)

        print(f"  {category}: {n_samples} samples ({unique_images} unique images)")

    # Save overall summary
    overall_summary = {
        'configuration': {
            'filter_negated': filter_negated,
            'filter_uncertain': filter_uncertain,
            'include_minimal': include_minimal,
            'source_file': json_path
        },
        'statistics': {
            'total_images_in_source': stats['total_images'],
            'images_with_valid_bboxes': stats['images_with_valid_bboxes'],
            'total_bboxes_extracted': stats['total_bboxes_extracted'],
            'bboxes_filtered_negated': stats['bboxes_filtered_negated'],
            'bboxes_filtered_uncertain': stats['bboxes_filtered_uncertain'],
            'bboxes_valid': stats['bboxes_valid'],
        },
        'samples_per_category': dict(stats['samples_per_category']),
        'filtering_examples': stats['filtering_examples'],
        'categories_available': [
            cat for cat in CHEXPERT_CATEGORIES
            if stats['samples_per_category'][cat] > 0
        ]
    }

    overall_summary_file = output_path / "summary.json"
    with open(overall_summary_file, 'w') as f:
        json.dump(overall_summary, f, indent=2)

    print(f"\nOverall summary saved to: {overall_summary_file}")

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"\nFiltering Statistics:")
    print(f"  Total bboxes extracted: {stats['total_bboxes_extracted']}")
    print(f"  Bboxes filtered (negated): {stats['bboxes_filtered_negated']}")
    print(f"  Bboxes filtered (uncertain): {stats['bboxes_filtered_uncertain']}")
    print(f"  Valid bboxes: {stats['bboxes_valid']}")

    print(f"\nSamples by Category:")
    print("-" * 50)
    for category in CHEXPERT_CATEGORIES:
        count = stats['samples_per_category'][category]
        if count > 0:
            print(f"  {category:<30}: {count:>5} samples")

    total_samples = sum(stats['samples_per_category'].values())
    print("-" * 50)
    print(f"  {'TOTAL':<30}: {total_samples:>5} samples")

    print(f"\nAll files saved to: {output_path}")

    return overall_summary


def main():
    """Main entry point."""
    json_path = "data/padchest-gr/BIMCV-Padchest-GR /grounded_reports_20240819.json"

    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Run organization with default settings:
    # - Filter negated (false negatives)
    # - Keep uncertain (but flag them)
    # - Include minimal findings
    summary = organize_by_chexpert_label(
        json_path,
        filter_negated=True,
        filter_uncertain=False,
        include_minimal=True
    )

    return summary


if __name__ == "__main__":
    main()
