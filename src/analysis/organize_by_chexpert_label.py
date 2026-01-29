"""
Organize PadChest-GR dataset by CheXpert labels for evaluation.

This script creates files organized by CheXpert label where:
- Each sample has a "main_bbox" with the target CheXpert label
- Each sample includes "other_bboxes" in the same image (can be same or different labels)
- False negatives are filtered out using text analysis of bbox annotations
- CheXbert verifies the PadChest→CheXpert label remapping before saving

Output structure:
    data/padchest-gr/chexpert-by-label/
    ├── verified_samples.json       # All verified samples (all categories)
    ├── discarded_samples.json      # Samples discarded by verifier
    ├── {Category}_samples.json     # Samples per category
    └── summary.json                # Overall statistics
"""

import json
import sys
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple, Optional

# Add parent paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "evalutations"))

from padchest_to_chexpert_mapping import convert_image_to_chexpert_labels, PADCHEST_TO_CHEXPERT_MAPPING


# CheXpert categories for classification (matching CheXbert order)
CHEXPERT_CATEGORIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices'
]


class CheXbertLabelVerifier:
    """
    CheXbert-based verifier for PadChest→CheXpert label mappings.
    Uses the pre-trained CheXbert model to extract labels from sentences
    and verify that the mapped label matches.
    """

    def __init__(self, device: str = None):
        """
        Initialize the CheXbert verifier.

        Args:
            device: Device to use ('cuda', 'cpu', or None for auto)
        """
        from rrg_eval.chexbert import CheXbert
        from rrg_eval.factuality_utils import CheXbert_CONDITIONS, map_to_binary

        self.model = CheXbert()
        self.conditions = CheXbert_CONDITIONS
        self.map_to_binary = map_to_binary

        # Mapping from our category names to CheXbert condition indices
        self.category_to_idx = {
            'Enlarged Cardiomediastinum': 0,
            'Cardiomegaly': 1,
            'Lung Opacity': 2,
            'Lung Lesion': 3,
            'Edema': 4,
            'Consolidation': 5,
            'Pneumonia': 6,
            'Atelectasis': 7,
            'Pneumothorax': 8,
            'Pleural Effusion': 9,
            'Pleural Other': 10,
            'Fracture': 11,
            'Support Devices': 12,
        }

        print("CheXbert verifier loaded successfully")

    def verify_batch(
        self,
        sentences: List[str],
        chexpert_categories: List[str],
    ) -> List[Tuple[bool, str, float]]:
        """
        Verify a batch of sentences against their expected CheXpert categories.

        Args:
            sentences: List of finding sentences
            chexpert_categories: List of expected CheXpert categories

        Returns:
            List of (is_verified, reason, confidence) tuples
        """
        if not sentences:
            return []

        # Run CheXbert on all sentences
        # Output shape: [14 conditions, num_sentences]
        outputs = self.model(sentences)

        results = []
        for i, (sentence, expected_cat) in enumerate(zip(sentences, chexpert_categories)):
            if not sentence.strip():
                results.append((False, "Empty sentence", 0.0))
                continue

            if expected_cat not in self.category_to_idx:
                results.append((False, f"Unknown category: {expected_cat}", 0.0))
                continue

            cat_idx = self.category_to_idx[expected_cat]

            # Get the prediction for this category
            # CheXbert outputs: 0=blank, 1=positive, 2=negative, 3=uncertain
            pred = outputs[cat_idx][i]

            # Map to binary (positive=1, negative/blank/uncertain=0)
            is_positive = self.map_to_binary(pred) == 1

            if is_positive:
                # CheXbert confirms the label
                confidence = 1.0 if pred == 1 else 0.7  # Lower if uncertain
                results.append((True, f"CheXbert confirms: {expected_cat}", confidence))
            else:
                # CheXbert does not detect this label
                pred_labels = []
                for cat, idx in self.category_to_idx.items():
                    if self.map_to_binary(outputs[idx][i]) == 1:
                        pred_labels.append(cat)

                if pred_labels:
                    results.append((
                        False,
                        f"CheXbert predicts: {pred_labels}, not {expected_cat}",
                        0.0
                    ))
                else:
                    results.append((
                        False,
                        f"CheXbert: No findings detected, expected {expected_cat}",
                        0.0
                    ))

        return results

    def verify_mapping(
        self,
        padchest_labels: List[str],
        chexpert_category: str,
        sentence_en: str,
        sentence_es: str = "",
    ) -> Tuple[bool, str, float]:
        """
        Verify if the PadChest→CheXpert mapping is correct using CheXbert.

        Args:
            padchest_labels: Original PadChest labels
            chexpert_category: Mapped CheXpert category
            sentence_en: English sentence describing the finding
            sentence_es: Spanish sentence (fallback)

        Returns:
            Tuple of (is_verified, reason, confidence_score)
        """
        text = sentence_en or sentence_es or ""

        if not text.strip():
            return False, "Empty sentence - cannot verify", 0.0

        results = self.verify_batch([text], [chexpert_category])
        return results[0]


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
    include_minimal: bool = True,
    verify_mappings: bool = True,
    device: str = None
):
    """
    Organize dataset by CheXpert label with CheXbert verification.

    Each sample contains:
    - main_bbox: The bbox with the target CheXpert label (for evaluation)
    - other_bboxes: Other bboxes in the same image (context)
    - image_info: Image metadata
    - verification: CheXbert verification results for the label mapping

    Only saves samples where CheXbert confirms the remapped label.
    Discarded samples are saved separately for analysis.

    Args:
        json_path: Path to the original PadChest-GR JSON
        output_dir: Output directory for organized files
        filter_negated: If True, filter out negated findings (default: True)
        filter_uncertain: If True, filter out uncertain findings (default: False)
        include_minimal: If True, include minimal/borderline findings (default: True)
        verify_mappings: If True, use CheXbert to verify label mappings (default: True)
        device: Device for CheXbert ('cuda', 'cpu', or None for auto)
    """

    print("=" * 80)
    print("ORGANIZING DATASET BY CHEXPERT LABEL (WITH CHEXBERT VERIFICATION)")
    print("=" * 80)
    print(f"\nConfiguration:")
    print(f"  Filter negated findings: {filter_negated}")
    print(f"  Filter uncertain findings: {filter_uncertain}")
    print(f"  Include minimal findings: {include_minimal}")
    print(f"  Verify with CheXbert: {verify_mappings}")

    # Load dataset
    print(f"\nLoading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)
    print(f"Total images: {len(data)}")

    # Initialize
    negation_detector = NegationDetector()

    # Initialize CheXbert label verifier
    if verify_mappings:
        print("\nLoading CheXbert model...")
        label_verifier = CheXbertLabelVerifier(device=device)
    else:
        label_verifier = None

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Storage for verified and discarded samples (by category)
    verified_samples_by_category = defaultdict(list)
    discarded_samples = []

    # Statistics
    stats = {
        'total_images': len(data),
        'images_with_valid_bboxes': 0,
        'total_bboxes_extracted': 0,
        'bboxes_filtered_negated': 0,
        'bboxes_filtered_uncertain': 0,
        'bboxes_valid': 0,
        'samples_verified': 0,
        'samples_discarded': 0,
        'verified_per_category': Counter(),
        'discarded_per_category': Counter(),
        'co_occurrence': defaultdict(Counter),
        'verification_reasons': {
            'verified': [],
            'discarded': []
        },
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
                # Verify the label mapping using CheXbert
                if verify_mappings:
                    is_verified, reason, confidence = label_verifier.verify_mapping(
                        padchest_labels=main_bbox['padchest_labels'],
                        chexpert_category=category,
                        sentence_en=main_bbox['sentence_en'],
                        sentence_es=main_bbox['sentence_es']
                    )
                else:
                    is_verified, reason, confidence = True, "Verification disabled", 1.0

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
                    'other_categories_in_image': list(all_categories_in_image - {category}),
                    'verification': {
                        'is_verified': is_verified,
                        'reason': reason,
                        'confidence': confidence
                    }
                }

                if is_verified:
                    # Save to verified samples
                    verified_samples_by_category[category].append(sample)
                    stats['samples_verified'] += 1
                    stats['verified_per_category'][category] += 1

                    # Track co-occurrence
                    for other_cat in all_categories_in_image:
                        if other_cat != category:
                            stats['co_occurrence'][category][other_cat] += 1

                    # Track verification examples
                    if len(stats['verification_reasons']['verified']) < 20:
                        stats['verification_reasons']['verified'].append({
                            'category': category,
                            'padchest_labels': main_bbox['padchest_labels'],
                            'sentence': main_bbox['sentence_en'][:200],
                            'reason': reason,
                            'confidence': confidence
                        })
                else:
                    # Discard sample
                    discarded_samples.append(sample)
                    stats['samples_discarded'] += 1
                    stats['discarded_per_category'][category] += 1

                    # Track discard examples
                    if len(stats['verification_reasons']['discarded']) < 20:
                        stats['verification_reasons']['discarded'].append({
                            'category': category,
                            'padchest_labels': main_bbox['padchest_labels'],
                            'sentence': main_bbox['sentence_en'][:200],
                            'reason': reason,
                            'confidence': confidence
                        })

        if (idx + 1) % 500 == 0:
            print(f"  Processed {idx + 1} images... (verified: {stats['samples_verified']}, discarded: {stats['samples_discarded']})")

    print(f"\nProcessing complete!")
    print(f"Images with valid bboxes: {stats['images_with_valid_bboxes']}")
    print(f"Total valid bboxes: {stats['bboxes_valid']}")
    print(f"Samples verified: {stats['samples_verified']}")
    print(f"Samples discarded: {stats['samples_discarded']}")

    # Save files (not directories)
    print(f"\nSaving files to: {output_path}")

    # Save all verified samples (combined file)
    all_verified_samples = []
    for category in CHEXPERT_CATEGORIES:
        all_verified_samples.extend(verified_samples_by_category[category])

    verified_all_file = output_path / "verified_samples.json"
    with open(verified_all_file, 'w') as f:
        json.dump(all_verified_samples, f, indent=2)
    print(f"  Saved {len(all_verified_samples)} verified samples to: verified_samples.json")

    # Save discarded samples
    discarded_file = output_path / "discarded_samples.json"
    with open(discarded_file, 'w') as f:
        json.dump(discarded_samples, f, indent=2)
    print(f"  Saved {len(discarded_samples)} discarded samples to: discarded_samples.json")

    # Save per-category files
    print(f"\n  Per-category files:")
    for category in CHEXPERT_CATEGORIES:
        samples = verified_samples_by_category[category]

        if not samples:
            continue

        # Save category-specific file
        cat_filename = f"{category.replace(' ', '_')}_samples.json"
        cat_file = output_path / cat_filename
        with open(cat_file, 'w') as f:
            json.dump(samples, f, indent=2)

        # Calculate category statistics
        n_samples = len(samples)
        n_discarded = stats['discarded_per_category'][category]
        unique_images = len(set(s['image_info']['ImageID'] for s in samples))
        avg_confidence = sum(s['verification']['confidence'] for s in samples) / n_samples if n_samples else 0

        print(f"    {category}: {n_samples} verified, {n_discarded} discarded ({unique_images} unique images, avg conf: {avg_confidence:.2f})")

    # Save overall summary
    overall_summary = {
        'configuration': {
            'filter_negated': filter_negated,
            'filter_uncertain': filter_uncertain,
            'include_minimal': include_minimal,
            'verify_mappings': verify_mappings,
            'verifier': 'CheXbert' if verify_mappings else None,
            'source_file': json_path
        },
        'statistics': {
            'total_images_in_source': stats['total_images'],
            'images_with_valid_bboxes': stats['images_with_valid_bboxes'],
            'total_bboxes_extracted': stats['total_bboxes_extracted'],
            'bboxes_filtered_negated': stats['bboxes_filtered_negated'],
            'bboxes_filtered_uncertain': stats['bboxes_filtered_uncertain'],
            'bboxes_valid': stats['bboxes_valid'],
            'samples_verified': stats['samples_verified'],
            'samples_discarded': stats['samples_discarded'],
            'verification_rate': stats['samples_verified'] / (stats['samples_verified'] + stats['samples_discarded']) if (stats['samples_verified'] + stats['samples_discarded']) > 0 else 0
        },
        'verified_per_category': dict(stats['verified_per_category']),
        'discarded_per_category': dict(stats['discarded_per_category']),
        'verification_examples': stats['verification_reasons'],
        'filtering_examples': stats['filtering_examples'],
        'categories_available': [
            cat for cat in CHEXPERT_CATEGORIES
            if stats['verified_per_category'][cat] > 0
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

    print(f"\nVerification Statistics:")
    print(f"  Samples verified: {stats['samples_verified']}")
    print(f"  Samples discarded: {stats['samples_discarded']}")
    verification_rate = stats['samples_verified'] / (stats['samples_verified'] + stats['samples_discarded']) * 100 if (stats['samples_verified'] + stats['samples_discarded']) > 0 else 0
    print(f"  Verification rate: {verification_rate:.1f}%")

    print(f"\nVerified Samples by Category:")
    print("-" * 60)
    print(f"  {'Category':<30} {'Verified':>10} {'Discarded':>10}")
    print("-" * 60)
    for category in CHEXPERT_CATEGORIES:
        verified = stats['verified_per_category'][category]
        discarded = stats['discarded_per_category'][category]
        if verified > 0 or discarded > 0:
            print(f"  {category:<30} {verified:>10} {discarded:>10}")

    print("-" * 60)
    print(f"  {'TOTAL':<30} {stats['samples_verified']:>10} {stats['samples_discarded']:>10}")

    print(f"\nFiles saved to: {output_path}")
    print(f"  - verified_samples.json (all verified samples)")
    print(f"  - discarded_samples.json (samples not verified)")
    print(f"  - {{Category}}_samples.json (per-category files)")
    print(f"  - summary.json (statistics)")

    return overall_summary


def main():
    """Main entry point."""
    json_path = "data/padchest-gr/BIMCV-Padchest-GR /grounded_reports_20240819.json"

    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Run organization with CheXbert verification:
    # - Filter negated (false negatives)
    # - Filter uncertain findings
    # - Include minimal findings
    # - Use CheXbert to verify label mappings
    # - Only save samples where CheXbert confirms the remapped label
    summary = organize_by_chexpert_label(
        json_path,
        filter_negated=True,
        filter_uncertain=True,
        include_minimal=True,
        verify_mappings=True,
        device=None  # Auto-detect (cuda if available)
    )

    return summary


if __name__ == "__main__":
    main()
