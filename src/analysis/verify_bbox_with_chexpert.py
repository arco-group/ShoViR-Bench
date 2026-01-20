"""
Verify bbox text labels using CheXpert labeler.

This script uses a CheXpert label extraction approach to verify that the text
in each bbox generates the expected CheXpert labels.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
import re


class SimpleCheXpertLabeler:
    """
    Simple rule-based CheXpert labeler using keyword matching.
    This is a lightweight alternative to full transformer models.
    """

    def __init__(self):
        # Define keywords for each CheXpert category
        self.category_keywords = {
            'Atelectasis': [
                'atelectasis', 'atelectatic', 'collapse', 'collapsed',
                'volume loss', 'subsegmental'
            ],
            'Cardiomegaly': [
                'cardiomegaly', 'enlarged heart', 'cardiac enlargement',
                'heart size increased', 'cardiomegalia'
            ],
            'Consolidation': [
                'consolidation', 'consolidated', 'airspace opacity',
                'airspace opacification', 'alveolar', 'infiltrate'
            ],
            'Edema': [
                'edema', 'oedema', 'pulmonary edema', 'interstitial edema',
                'vascular congestion', 'fluid overload', 'congestive',
                'kerley', 'ground glass'
            ],
            'Enlarged Cardiomediastinum': [
                'mediastinal enlargement', 'widened mediastinum', 'mediastinal widening',
                'cardiomediastinal', 'mediastinal contour', 'superior mediastinum'
            ],
            'Fracture': [
                'fracture', 'fractured', 'broken', 'rib fracture',
                'clavicle fracture', 'vertebral fracture', 'compression fracture'
            ],
            'Lung Lesion': [
                'lesion', 'nodule', 'mass', 'nodular', 'granuloma',
                'calcified granuloma', 'cavitation', 'cavity'
            ],
            'Lung Opacity': [
                'opacity', 'opacification', 'infiltrate', 'density',
                'increased density', 'interstitial', 'reticular',
                'reticulonodular', 'hazy', 'ground glass'
            ],
            'Pleural Effusion': [
                'pleural effusion', 'effusion', 'fluid', 'pleural fluid',
                'hydrothorax', 'hemothorax'
            ],
            'Pleural Other': [
                'pleural thickening', 'pleural plaque', 'pleural mass',
                'costophrenic angle blunting', 'pleural calcification',
                'fissure thickening'
            ],
            'Pneumonia': [
                'pneumonia', 'pneumonic', 'infectious', 'infection'
            ],
            'Pneumothorax': [
                'pneumothorax', 'ptx', 'air collection', 'collapsed lung'
            ],
            'Support Devices': [
                'tube', 'line', 'catheter', 'device', 'pacemaker',
                'endotracheal', 'nasogastric', 'ng tube', 'ett',
                'central line', 'port', 'tracheostomy'
            ],
            'No Finding': [
                'normal', 'clear', 'no acute', 'unremarkable',
                'within normal limits', 'no significant'
            ]
        }

        # Negation patterns
        self.negation_patterns = [
            r'\bno\b', r'\bnot\b', r'\bwithout\b', r'\babsent\b',
            r'\bdenies\b', r'\bdenied\b', r'\bnegative for\b',
            r'\brule out\b', r'\br/o\b'
        ]

    def label_text(self, text: str) -> dict:
        """
        Label a text string with CheXpert categories.

        Args:
            text: The text to label (finding sentence)

        Returns:
            dict mapping category names to presence (True/False/None)
            None indicates uncertain
        """
        text_lower = text.lower()

        # Check for negations
        has_negation = any(re.search(pattern, text_lower) for pattern in self.negation_patterns)

        labels = {}

        for category, keywords in self.category_keywords.items():
            # Check if any keyword matches
            matches = [kw for kw in keywords if kw in text_lower]

            if matches:
                # If negation present, mark as negative (False)
                # Otherwise mark as positive (True)
                labels[category] = not has_negation
            else:
                labels[category] = None  # Not mentioned

        return labels

    def get_positive_labels(self, text: str) -> list:
        """Get list of categories that are positively labeled."""
        labels = self.label_text(text)
        return [cat for cat, val in labels.items() if val is True]


def verify_dataset_labels(json_path: str, output_file: str = "data/padchest-gr/chexpert-separated/bbox_verification_results.json"):
    """
    Verify all bbox labels in the organized dataset files.
    """

    print("="*80)
    print("CHEXPERT LABEL VERIFICATION")
    print("="*80)

    # Initialize labeler
    labeler = SimpleCheXpertLabeler()

    # Load the organized dataset
    organized_dir = Path("data/padchest-gr/chexpert-separated")
    all_bboxes_file = organized_dir / "all-with-bboxes.json"

    if not all_bboxes_file.exists():
        print(f"Error: Organized dataset not found at {all_bboxes_file}")
        print("Run organize_by_bbox_count.py first!")
        return

    print(f"\nLoading organized dataset from: {all_bboxes_file}")
    with open(all_bboxes_file, 'r') as f:
        data = json.load(f)

    print(f"Total images to verify: {len(data)}\n")

    # Verification results
    verification_results = {
        'total_images': len(data),
        'total_findings': 0,
        'verification_stats': {
            'exact_match': 0,
            'partial_match': 0,
            'no_match': 0,
            'predicted_superset': 0,
            'predicted_subset': 0
        },
        'category_stats': defaultdict(lambda: {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'total_expected': 0,
            'total_predicted': 0
        }),
        'examples': {
            'exact_match': [],
            'partial_match': [],
            'no_match': []
        }
    }

    print("Verifying labels...")
    for img_idx, image in enumerate(data):
        for finding in image.get('findings_with_chexpert', []):
            verification_results['total_findings'] += 1

            sentence = finding['sentence_en']
            expected_labels = set(finding['chexpert_categories'])
            predicted_labels = set(labeler.get_positive_labels(sentence))

            # Calculate matches
            matches = expected_labels & predicted_labels
            false_positives = predicted_labels - expected_labels
            false_negatives = expected_labels - predicted_labels

            # Classify verification result
            if matches == expected_labels == predicted_labels:
                verification_results['verification_stats']['exact_match'] += 1
                result_type = 'exact_match'
            elif len(matches) > 0:
                verification_results['verification_stats']['partial_match'] += 1
                result_type = 'partial_match'
            else:
                verification_results['verification_stats']['no_match'] += 1
                result_type = 'no_match'

            if predicted_labels > expected_labels:
                verification_results['verification_stats']['predicted_superset'] += 1
            elif predicted_labels < expected_labels and len(matches) > 0:
                verification_results['verification_stats']['predicted_subset'] += 1

            # Update per-category stats
            for cat in expected_labels:
                verification_results['category_stats'][cat]['total_expected'] += 1
                if cat in predicted_labels:
                    verification_results['category_stats'][cat]['true_positives'] += 1
                else:
                    verification_results['category_stats'][cat]['false_negatives'] += 1

            for cat in predicted_labels:
                verification_results['category_stats'][cat]['total_predicted'] += 1
                if cat not in expected_labels:
                    verification_results['category_stats'][cat]['false_positives'] += 1

            # Store examples (first 5 of each type)
            if len(verification_results['examples'][result_type]) < 5:
                verification_results['examples'][result_type].append({
                    'sentence': sentence,
                    'expected': list(expected_labels),
                    'predicted': list(predicted_labels),
                    'matches': list(matches),
                    'false_positives': list(false_positives),
                    'false_negatives': list(false_negatives)
                })

        if (img_idx + 1) % 100 == 0:
            print(f"  Verified {img_idx + 1} images...")

    # Calculate metrics for each category
    category_metrics = {}
    for cat, stats in verification_results['category_stats'].items():
        tp = stats['true_positives']
        fp = stats['false_positives']
        fn = stats['false_negatives']

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        category_metrics[cat] = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
            'total_expected': stats['total_expected'],
            'total_predicted': stats['total_predicted']
        }

    verification_results['category_metrics'] = category_metrics

    # Print results
    print("\n" + "="*80)
    print("VERIFICATION RESULTS")
    print("="*80)

    total = verification_results['total_findings']
    stats = verification_results['verification_stats']

    print(f"\nTotal findings verified: {total}")
    print(f"\nOverall Statistics:")
    print(f"  Exact matches:     {stats['exact_match']:4d} ({stats['exact_match']/total*100:5.1f}%)")
    print(f"  Partial matches:   {stats['partial_match']:4d} ({stats['partial_match']/total*100:5.1f}%)")
    print(f"  No matches:        {stats['no_match']:4d} ({stats['no_match']/total*100:5.1f}%)")
    print(f"  Predicted more:    {stats['predicted_superset']:4d} ({stats['predicted_superset']/total*100:5.1f}%)")
    print(f"  Predicted less:    {stats['predicted_subset']:4d} ({stats['predicted_subset']/total*100:5.1f}%)")

    print("\n" + "-"*80)
    print("Per-Category Metrics:")
    print("-"*80)
    print(f"{'Category':<30} {'Precision':>10} {'Recall':>10} {'F1':>10} {'TP':>6} {'FP':>6} {'FN':>6}")
    print("-"*80)

    for cat in sorted(category_metrics.keys()):
        metrics = category_metrics[cat]
        if metrics['total_expected'] > 0 or metrics['total_predicted'] > 0:
            print(f"{cat:<30} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f} "
                  f"{metrics['f1_score']:>10.3f} {metrics['true_positives']:>6} "
                  f"{metrics['false_positives']:>6} {metrics['false_negatives']:>6}")

    # Calculate overall metrics
    total_tp = sum(m['true_positives'] for m in category_metrics.values())
    total_fp = sum(m['false_positives'] for m in category_metrics.values())
    total_fn = sum(m['false_negatives'] for m in category_metrics.values())

    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0

    print("-"*80)
    print(f"{'OVERALL':<30} {overall_precision:>10.3f} {overall_recall:>10.3f} "
          f"{overall_f1:>10.3f} {total_tp:>6} {total_fp:>6} {total_fn:>6}")
    print("-"*80)

    # Print example mismatches
    print("\n" + "="*80)
    print("EXAMPLE CASES")
    print("="*80)

    for match_type, examples in verification_results['examples'].items():
        if examples:
            print(f"\n{match_type.upper().replace('_', ' ')} (showing {len(examples)} examples):")
            print("-"*80)
            for ex in examples:
                print(f"\nSentence: {ex['sentence']}")
                print(f"Expected: {ex['expected']}")
                print(f"Predicted: {ex['predicted']}")
                if ex['matches']:
                    print(f"✓ Matches: {ex['matches']}")
                if ex['false_positives']:
                    print(f"✗ False Positives: {ex['false_positives']}")
                if ex['false_negatives']:
                    print(f"✗ False Negatives: {ex['false_negatives']}")

    # Save results
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(verification_results, f, indent=2)

    print(f"\n\nVerification results saved to: {output_file}")

    return verification_results


if __name__ == "__main__":
    json_path = "data/padchest-gr/BIMCV-Padchest-GR /grounded_reports_20240819.json"

    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    results = verify_dataset_labels(json_path)
