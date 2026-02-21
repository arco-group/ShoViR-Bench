"""
Mapping dictionary from PadChest-GR labels (154 unique labels) to CheXpert labels (14 categories).

This mapping is based on clinical knowledge and semantic similarity between labels.
Some PadChest labels may map to multiple CheXpert categories or none at all.

CheXpert Categories:
1. Atelectasis
2. Cardiomegaly
3. Consolidation
4. Edema
5. Enlarged Cardiomediastinum
6. Fracture
7. Lung Lesion
8. Lung Opacity
9. Pleural Effusion
10. Pleural Other
11. Pneumonia
12. Pneumothorax
13. Support Devices
14. No Finding
"""

# Direct 1-to-1 mappings
PADCHEST_TO_CHEXPERT_MAPPING = {
    # ============================================================================
    # ATELECTASIS - Collapse or incomplete expansion of lung tissue
    # ============================================================================
    'atelectasis': 'Atelectasis',
    'atelectasis basal': 'Atelectasis',
    'laminar atelectasis': 'Atelectasis',
    'lobar atelectasis': 'Atelectasis',
    'segmental atelectasis': 'Atelectasis',
    'total atelectasis': 'Atelectasis',

    # ============================================================================
    # CARDIOMEGALY - Enlarged heart
    # ============================================================================
    'cardiomegaly': 'Cardiomegaly',

    # ============================================================================
    # CONSOLIDATION - Filling of airspaces with fluid, pus, blood, or cells
    # ============================================================================
    'consolidation': 'Consolidation',
    'alveolar pattern': 'Consolidation',  # Often indicates consolidation

    # ============================================================================
    # EDEMA - Fluid accumulation in lungs
    # ============================================================================
    # Note: No direct "edema" label in PadChest, but related findings:
    'hilar congestion': 'Edema',  # Can indicate pulmonary edema
    'kerley lines': 'Edema',  # Classic sign of interstitial edema
    'ground glass pattern': 'Edema',  # Can indicate edema or other pathology

    # ============================================================================
    # ENLARGED CARDIOMEDIASTINUM - Widened mediastinum
    # ============================================================================
    'mediastinal enlargement': 'Enlarged Cardiomediastinum',
    'superior mediastinal enlargement': 'Enlarged Cardiomediastinum',
    'mediastinic lipomatosis': 'Enlarged Cardiomediastinum',
    'aortic button enlargement': 'Enlarged Cardiomediastinum',

    # ============================================================================
    # FRACTURE - Bone fractures
    # ============================================================================
    'fracture': 'Fracture',
    'callus rib fracture': 'Fracture',
    'rib fracture': 'Fracture',
    'clavicle fracture': 'Fracture',
    'humeral fracture': 'Fracture',
    'vertebral fracture': 'Fracture',
    'vertebral compression': 'Fracture',
    'vertebral anterior compression': 'Fracture',

    # ============================================================================
    # LUNG LESION - Abnormal lung masses or nodules
    # ============================================================================
    'nodule': 'Lung Lesion',
    'multiple nodules': 'Lung Lesion',
    'mass': 'Lung Lesion',
    'pulmonary mass': 'Lung Lesion',
    'calcified granuloma': 'Lung Lesion',
    'granuloma': 'Lung Lesion',
    'cavitation': 'Lung Lesion',
    'cyst': 'Lung Lesion',
    'abscess': 'Lung Lesion',

    # ============================================================================
    # LUNG OPACITY - Increased density in lungs (non-specific)
    # ============================================================================
    'increased density': 'Lung Opacity',
    'infiltrates': 'Lung Opacity',
    'interstitial pattern': 'Lung Opacity',
    'reticular interstitial pattern': 'Lung Opacity',
    'reticulonodular interstitial pattern': 'Lung Opacity',
    'miliary opacities': 'Lung Opacity',
    'air bronchogram': 'Lung Opacity',  # Often seen with opacity/consolidation

    # ============================================================================
    # PLEURAL EFFUSION - Fluid in pleural space
    # ============================================================================
    'pleural effusion': 'Pleural Effusion',
    'loculated pleural effusion': 'Pleural Effusion',
    'loculated fissural effusion': 'Pleural Effusion',
    'hydropneumothorax': 'Pleural Effusion',  # Contains both air and fluid

    # ============================================================================
    # PLEURAL OTHER - Other pleural abnormalities
    # ============================================================================
    'pleural thickening': 'Pleural Other',
    'apical pleural thickening': 'Pleural Other',
    'calcified pleural thickening': 'Pleural Other',
    'calcified pleural plaques': 'Pleural Other',
    'fissure thickening': 'Pleural Other',
    'major fissure thickening': 'Pleural Other',
    'minor fissure thickening': 'Pleural Other',
    'costophrenic angle blunting': 'Pleural Other',  # Often pleural thickening/scarring
    'pleural mass': 'Pleural Other',

    # ============================================================================
    # PNEUMONIA - Lung infection/inflammation
    # ============================================================================
    # Note: No direct "pneumonia" label in PadChest
    # Pneumonia would typically show as consolidation or infiltrates

    # ============================================================================
    # PNEUMOTHORAX - Air in pleural space
    # ============================================================================
    'pneumothorax': 'Pneumothorax',
    # 'hydropneumothorax' already mapped to Pleural Effusion (contains both)

    # ============================================================================
    # SUPPORT DEVICES - Medical devices
    # ============================================================================
    'pacemaker': 'Support Devices',
    'dual chamber device': 'Support Devices',
    'single chamber device': 'Support Devices',
    'NSG tube': 'Support Devices',  # Nasogastric tube
    'endotracheal tube': 'Support Devices',
    'tracheostomy tube': 'Support Devices',
    'chest drain tube': 'Support Devices',
    'central venous catheter': 'Support Devices',
    'central venous catheter via jugular vein': 'Support Devices',
    'central venous catheter via subclavian vein': 'Support Devices',
    'reservoir central venous catheter': 'Support Devices',
    'catheter': 'Support Devices',
    'electrical device': 'Support Devices',
    'ventriculoperitoneal drain tube': 'Support Devices',
    'gastrostomy tube': 'Support Devices',
}

# Labels that don't map to any CheXpert category (mostly chronic/structural findings)
PADCHEST_NO_CHEXPERT_MAPPING = {
    # Chronic/Degenerative Changes
    'chronic changes',
    'vertebral degenerative changes',
    'non axial articular degenerative changes',
    'axial hyperostosis',
    'osteopenia',
    'osteoporosis',

    # Vascular/Cardiac Chronic Changes
    'aortic elongation',
    'aortic atheromatosis',
    'ascendent aortic elongation',
    'descendent aortic elongation',
    'supra aortic elongation',
    'aortic aneurysm',
    'pulmonary artery enlargement',
    'vascular hilar enlargement',
    'hilar enlargement',
    'vascular redistribution',
    'central vascular redistribution',

    # Skeletal Deformities
    'scoliosis',
    'kyphosis',
    'pectum excavatum',
    'pectum carinatum',
    'thoracic cage deformation',
    'cervical rib',

    # Hernias and Anatomical Variants
    'hiatal hernia',
    'diaphragmatic eventration',
    'hemidiaphragm elevation',
    'azygos lobe',

    # Chronic Lung Changes
    'air trapping',
    'bullas',
    'bronchiectasis',
    'fibrotic band',
    'volume loss',
    'hyperinflated lung',
    'flattened diaphragm',
    'hypoexpansion',
    'lung vascular paucity',

    # Benign Findings
    'pseudonodule',
    'calcified densities',
    'nipple shadow',
    'end on vessel',
    'gynecomastia',
    'goiter',
    'obesity',

    # Surgical/Post-procedural
    'surgery',
    'surgery heart',
    'surgery lung',
    'surgery breast',
    'surgery neck',
    'surgery humeral',
    'sternotomy',
    'mastectomy',
    'suture material',
    'osteosynthesis material',

    # Prostheses (not support devices in acute care sense)
    'mammary prosthesis',
    'humeral prosthesis',
    'prosthesis',
    'artificial heart valve',
    'artificial aortic heart valve',
    'artificial mitral heart valve',
    'heart valve calcified',
    'aortic endoprosthesis',
    'endoprosthesis',

    # Mediastinal Findings (not clearly enlarged cardiomediastinum)
    'mediastinal mass',
    'mediastinal shift',
    'azygoesophageal recess shift',
    'tracheal shift',

    # Other Soft Tissue/Bone
    'calcified fibroadenoma',
    'calcified adenopathy',
    'adenopathy',
    'blastic bone lesion',
    'lytic bone lesion',
    'sclerotic bone lesion',
    'metal',
    'soft tissue mass',
    'abnormal foreign body',

    # Miscellaneous
    'bronchovascular markings',
    'costochondral junction hypertrophy',
    'sternoclavicular junction hypertrophy',
    'subacromial space narrowing',
    'subcutaneous emphysema',
    'pneumoperitoneo',
    'pericardial effusion',
    'Chilaiditi sign',
    'dai',  # Diffuse axonal injury (not typical for chest X-ray)
    'air fluid level',
}

# Create reverse mapping: CheXpert -> list of PadChest labels
CHEXPERT_TO_PADCHEST_MAPPING = {}
for padchest_label, chexpert_label in PADCHEST_TO_CHEXPERT_MAPPING.items():
    if chexpert_label not in CHEXPERT_TO_PADCHEST_MAPPING:
        CHEXPERT_TO_PADCHEST_MAPPING[chexpert_label] = []
    CHEXPERT_TO_PADCHEST_MAPPING[chexpert_label].append(padchest_label)


def map_padchest_to_chexpert(padchest_label: str) -> str:
    """
    Map a PadChest label to a CheXpert category.

    Args:
        padchest_label: Label from PadChest-GR dataset

    Returns:
        CheXpert category name, or None if no mapping exists
    """
    return PADCHEST_TO_CHEXPERT_MAPPING.get(padchest_label, None)


def get_chexpert_labels_from_finding(finding_labels: list) -> set:
    """
    Convert a list of PadChest labels from a finding to CheXpert categories.

    Args:
        finding_labels: List of PadChest labels for a single finding

    Returns:
        Set of CheXpert category names
    """
    chexpert_labels = set()
    for label in finding_labels:
        mapped = map_padchest_to_chexpert(label)
        if mapped:
            chexpert_labels.add(mapped)
    return chexpert_labels


def convert_image_to_chexpert_labels(image_data: dict) -> dict:
    """
    Convert all findings in an image to CheXpert categories.

    Args:
        image_data: Image data dictionary from PadChest-GR JSON

    Returns:
        Dictionary mapping CheXpert categories to:
            - 'present': boolean indicating if category is present
            - 'padchest_labels': list of original PadChest labels that mapped to this category
            - 'locations': list of anatomical locations
            - 'bbox_count': number of bounding boxes for this category
    """
    # Initialize all CheXpert categories
    chexpert_categories = {
        'Atelectasis': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Cardiomegaly': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Consolidation': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Edema': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Enlarged Cardiomediastinum': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Fracture': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Lung Lesion': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Lung Opacity': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Pleural Effusion': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Pleural Other': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Pneumonia': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Pneumothorax': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'Support Devices': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
        'No Finding': {'present': False, 'padchest_labels': [], 'locations': [], 'bbox_count': 0},
    }

    has_any_abnormal = False

    for finding in image_data.get('findings', []):
        is_abnormal = finding.get('abnormal', False)

        if is_abnormal:
            has_any_abnormal = True
            labels = finding.get('labels', [])
            locations = finding.get('locations', [])
            boxes = finding.get('boxes', [])

            for label in labels:
                chexpert_label = map_padchest_to_chexpert(label)
                if chexpert_label:
                    chexpert_categories[chexpert_label]['present'] = True
                    chexpert_categories[chexpert_label]['padchest_labels'].append(label)
                    chexpert_categories[chexpert_label]['locations'].extend(locations)
                    chexpert_categories[chexpert_label]['bbox_count'] += len(boxes)

    # If no abnormal findings, mark as "No Finding"
    if not has_any_abnormal:
        chexpert_categories['No Finding']['present'] = True

    return chexpert_categories


def print_mapping_statistics():
    """Print statistics about the mapping."""
    print("="*80)
    print("PADCHEST TO CHEXPERT MAPPING STATISTICS")
    print("="*80)

    print(f"\nTotal PadChest labels mapped to CheXpert: {len(PADCHEST_TO_CHEXPERT_MAPPING)}")
    print(f"Total PadChest labels NOT mapped: {len(PADCHEST_NO_CHEXPERT_MAPPING)}")
    print(f"Total unique PadChest labels: {len(PADCHEST_TO_CHEXPERT_MAPPING) + len(PADCHEST_NO_CHEXPERT_MAPPING)}")

    print("\n" + "-"*80)
    print("CheXpert Categories Coverage:")
    print("-"*80)

    for category in sorted(CHEXPERT_TO_PADCHEST_MAPPING.keys()):
        padchest_labels = CHEXPERT_TO_PADCHEST_MAPPING[category]
        print(f"\n{category} ({len(padchest_labels)} PadChest labels):")
        for label in sorted(padchest_labels):
            print(f"  - {label}")

    # Check for CheXpert categories with no mappings
    all_chexpert = [
        'Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion',
        'Lung Opacity', 'Pleural Effusion', 'Pleural Other',
        'Pneumonia', 'Pneumothorax', 'Support Devices', 'No Finding'
    ]

    unmapped_chexpert = [cat for cat in all_chexpert if cat not in CHEXPERT_TO_PADCHEST_MAPPING]

    if unmapped_chexpert:
        print("\n" + "-"*80)
        print("CheXpert Categories with NO PadChest mappings:")
        print("-"*80)
        for cat in unmapped_chexpert:
            print(f"  - {cat}")


if __name__ == "__main__":
    print_mapping_statistics()
