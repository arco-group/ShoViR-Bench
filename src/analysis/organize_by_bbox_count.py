"""
Organize PadChest-GR dataset by bbox count with CheXpert labels.
Also verify bbox text labels using CheXpert labeler from HuggingFace.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List

from padchest_to_chexpert_mapping import convert_image_to_chexpert_labels


def organize_dataset_by_bbox_count(json_path: str, output_dir: str = "data/padchest-gr/chexpert-separated"):
    """
    Organize dataset into separate JSON files based on bbox count with CheXpert labels.

    Creates files like:
    - 2bbox-images.json (images with at least 2 bboxes with CheXpert labels)
    - 3bbox-images.json (images with at least 3 bboxes with CheXpert labels)
    - etc.
    """

    print(f"Loading dataset from: {json_path}")
    with open(json_path, 'r') as f:
        data = json.load(f)

    print(f"Total images: {len(data)}\n")

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_path}\n")

    # Organize images by bbox count
    images_by_bbox_count = defaultdict(list)
    bbox_count_stats = Counter()

    # Statistics
    total_images_with_bboxes = 0
    max_bbox_count = 0

    print("Processing images...")
    for idx, image_data in enumerate(data):
        # Convert to CheXpert labels
        chexpert_result = convert_image_to_chexpert_labels(image_data)

        # Count total bboxes with CheXpert labels
        total_bboxes = sum(info['bbox_count'] for cat, info in chexpert_result.items()
                          if info['present'] and cat != 'No Finding')

        if total_bboxes > 0:
            total_images_with_bboxes += 1
            max_bbox_count = max(max_bbox_count, total_bboxes)
            bbox_count_stats[total_bboxes] += 1

            # Create enhanced image data with CheXpert labels
            enhanced_image = {
                'original_data': image_data,
                'chexpert_labels': {},
                'total_chexpert_bboxes': total_bboxes,
                'findings_with_chexpert': []
            }

            # Add CheXpert category information
            for category, info in chexpert_result.items():
                if info['present'] and category != 'No Finding':
                    enhanced_image['chexpert_labels'][category] = {
                        'bbox_count': info['bbox_count'],
                        'padchest_labels': info['padchest_labels'],
                        'locations': info['locations']
                    }

            # Extract findings with bboxes and their CheXpert mappings
            for finding in image_data.get('findings', []):
                if finding.get('abnormal', False) and len(finding.get('boxes', [])) > 0:
                    # Determine which CheXpert categories this finding maps to
                    finding_chexpert_cats = set()
                    for label in finding.get('labels', []):
                        for cat, info in chexpert_result.items():
                            if label in info['padchest_labels']:
                                finding_chexpert_cats.add(cat)

                    if finding_chexpert_cats:
                        enhanced_finding = {
                            'sentence_en': finding.get('sentence_en', ''),
                            'sentence_es': finding.get('sentence_es', ''),
                            'labels': finding.get('labels', []),
                            'locations': finding.get('locations', []),
                            'boxes': finding.get('boxes', []),
                            'chexpert_categories': list(finding_chexpert_cats)
                        }
                        enhanced_image['findings_with_chexpert'].append(enhanced_finding)

            # Add to appropriate bbox count bins
            images_by_bbox_count[total_bboxes].append(enhanced_image)

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx + 1} images...")

    print(f"\nProcessing complete!")
    print(f"Total images with CheXpert bboxes: {total_images_with_bboxes}")
    print(f"Max bbox count: {max_bbox_count}\n")

    # Print distribution
    print("Distribution of images by bbox count:")
    print("-" * 50)
    for bbox_count in sorted(bbox_count_stats.keys()):
        count = bbox_count_stats[bbox_count]
        cumulative = sum(bbox_count_stats[bc] for bc in bbox_count_stats if bc >= bbox_count)
        print(f"{bbox_count:2d} bboxes: {count:4d} images | {bbox_count}+ bboxes: {cumulative:4d} images")

    # Generate JSON files for each bbox count (2+)
    print(f"\nGenerating JSON files...")
    generated_files = []

    for min_bbox_count in range(2, max_bbox_count + 1):
        # Collect all images with at least min_bbox_count bboxes
        images_for_file = []
        for bbox_count, images in images_by_bbox_count.items():
            if bbox_count >= min_bbox_count:
                images_for_file.extend(images)

        if images_for_file:
            output_file = output_path / f"{min_bbox_count}bbox-images.json"

            with open(output_file, 'w') as f:
                json.dump(images_for_file, f, indent=2)

            generated_files.append(output_file)
            print(f"  Created: {output_file} ({len(images_for_file)} images)")

    # Also create a comprehensive file with all images that have bboxes
    all_images_with_bboxes = []
    for images in images_by_bbox_count.values():
        all_images_with_bboxes.extend(images)

    all_file = output_path / "all-with-bboxes.json"
    with open(all_file, 'w') as f:
        json.dump(all_images_with_bboxes, f, indent=2)
    print(f"  Created: {all_file} ({len(all_images_with_bboxes)} images)")

    # Create a summary file
    summary = {
        'total_images_in_dataset': len(data),
        'total_images_with_chexpert_bboxes': total_images_with_bboxes,
        'max_bbox_count': max_bbox_count,
        'bbox_count_distribution': dict(bbox_count_stats),
        'generated_files': [str(f) for f in generated_files + [all_file]],
        'cumulative_counts': {
            str(min_count): sum(bbox_count_stats[bc] for bc in bbox_count_stats if bc >= min_count)
            for min_count in range(1, max_bbox_count + 1)
        }
    }

    summary_file = output_path / "summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Created: {summary_file}")

    print(f"\nAll files saved to: {output_path}")

    return summary


def verify_bbox_text_with_chexpert_labeler(
    json_path: str,
    output_file: str = "data/padchest-gr/chexpert-separated/bbox_verification.json"
):
    """
    Verify bbox text labels using CheXpert labeler from HuggingFace.
    This checks if the text in each bbox generates the expected CheXpert labels.

    Note: This requires the CheXpert labeler model from HuggingFace.
    """
    print("\n" + "="*80)
    print("CHEXPERT LABELER VERIFICATION")
    print("="*80)

    try:
        # Try to import transformers and load the CheXpert labeler
        from transformers import AutoTokenizer, AutoModelForSequenceClassification
        import torch

        print("\nLoading CheXpert labeler model from HuggingFace...")
        print("Note: This may take a while on first run...")

        # Use CheXbert or similar model
        # For now, we'll create a placeholder for the verification logic
        print("\n[INFO] CheXpert labeler integration placeholder.")
        print("To fully implement, you need to:")
        print("  1. Install: pip install transformers torch")
        print("  2. Load CheXbert model (stanford/chexbert or similar)")
        print("  3. Run inference on bbox sentences")

        # Placeholder for actual implementation
        verification_results = {
            'note': 'This is a placeholder. Implement CheXbert model loading and inference.',
            'model': 'stanford/chexbert (to be implemented)',
            'instructions': [
                'Install transformers: pip install transformers torch',
                'Load model: AutoModelForSequenceClassification.from_pretrained("stanford/chexbert")',
                'For each bbox sentence, run inference',
                'Compare predicted labels with mapped CheXpert labels'
            ]
        }

        # Save placeholder results
        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, 'w') as f:
            json.dump(verification_results, f, indent=2)

        print(f"\nVerification placeholder saved to: {output_file}")
        print("\nTo implement full verification:")
        print("  1. Uncomment and implement the CheXbert model loading code below")
        print("  2. Run inference on each finding's sentence_en")
        print("  3. Compare with the mapped CheXpert categories")

    except ImportError as e:
        print(f"\n[WARNING] transformers library not installed: {e}")
        print("To enable CheXpert labeler verification:")
        print("  pip install transformers torch")


def create_verification_script_template():
    """Create a template script for CheXpert label verification."""

    template = '''"""
CheXpert Label Verification Script Template

This script verifies that bbox text generates expected CheXpert labels using a labeler model.
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

# CheXpert categories
CHEXPERT_CATEGORIES = [
    'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
    'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
    'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
    'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
]

def load_chexpert_labeler():
    """Load CheXpert labeler model."""
    # Option 1: Use CheXbert (if available)
    # model_name = "stanford/chexbert"

    # Option 2: Use alternative CheXpert labeler
    # model_name = "your-model-name"

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # return tokenizer, model
    pass

def verify_finding_labels(sentence: str, expected_labels: list, tokenizer, model):
    """
    Verify that the sentence generates the expected CheXpert labels.

    Args:
        sentence: The finding sentence in English
        expected_labels: List of expected CheXpert categories
        tokenizer: HuggingFace tokenizer
        model: HuggingFace model

    Returns:
        dict with verification results
    """
    # Tokenize
    # inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

    # Run inference
    # with torch.no_grad():
    #     outputs = model(**inputs)
    #     predictions = torch.sigmoid(outputs.logits)

    # Convert predictions to labels (threshold at 0.5)
    # predicted_labels = []
    # for idx, prob in enumerate(predictions[0]):
    #     if prob > 0.5:
    #         predicted_labels.append(CHEXPERT_CATEGORIES[idx])

    # Compare with expected
    # matches = set(predicted_labels) & set(expected_labels)
    # false_positives = set(predicted_labels) - set(expected_labels)
    # false_negatives = set(expected_labels) - set(predicted_labels)

    # return {
    #     'sentence': sentence,
    #     'expected': expected_labels,
    #     'predicted': predicted_labels,
    #     'matches': list(matches),
    #     'false_positives': list(false_positives),
    #     'false_negatives': list(false_negatives),
    #     'accuracy': len(matches) / max(len(expected_labels), 1)
    # }
    pass

if __name__ == "__main__":
    print("CheXpert Label Verification")
    print("Implement the verification logic above")
'''

    template_file = "src/analysis/chexpert_verification_template.py"
    with open(template_file, 'w') as f:
        f.write(template)

    print(f"\nCreated verification template: {template_file}")
    return template_file


if __name__ == "__main__":
    json_path = "data/padchest-gr/BIMCV-Padchest-GR /grounded_reports_20240819.json"

    if len(sys.argv) > 1:
        json_path = sys.argv[1]

    if not Path(json_path).exists():
        print(f"Error: File not found: {json_path}")
        sys.exit(1)

    # Organize dataset by bbox count
    summary = organize_dataset_by_bbox_count(json_path)

    # Create verification template
    create_verification_script_template()

    # Placeholder for verification (implement when model is available)
    verify_bbox_text_with_chexpert_labeler(json_path)

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total images: {summary['total_images_in_dataset']}")
    print(f"Images with CheXpert bboxes: {summary['total_images_with_chexpert_bboxes']}")
    print(f"\nCumulative counts (images with N+ bboxes):")
    for min_count, count in sorted(summary['cumulative_counts'].items(), key=lambda x: int(x[0])):
        print(f"  {min_count}+ bboxes: {count} images")
