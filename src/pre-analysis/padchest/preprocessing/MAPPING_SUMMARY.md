# PadChest-GR to CheXpert Mapping Summary

**Created:** 2026-01-20
**Dataset:** BIMCV-Padchest-GR grounded_reports_20240819.json (4,555 images)

---

## Executive Summary

A comprehensive mapping has been created to convert PadChest-GR's 154 unique labels into the 14 CheXpert categories used in the RRG (Region-based Radiology Report Generation) framework.

### Key Findings

- **Mapping Coverage:** 69 of 154 PadChest labels (44.8%) can be mapped to CheXpert categories
- **Label Conversion Rate:** 43.01% of label mentions in the dataset map to CheXpert
- **CheXpert Coverage:** 12 of 14 CheXpert categories have corresponding PadChest labels
- **Unmapped Labels:** 85 labels (primarily chronic/structural findings) don't map to CheXpert

---

## CheXpert Category Distribution (After Mapping)

| Rank | Category | Images | % of Dataset | Total BBoxes | Avg BBoxes/Image |
|------|----------|--------|--------------|--------------|------------------|
| 1 | **No Finding** | 1,456 | 31.96% | 0 | 0.00 |
| 2 | **Cardiomegaly** | 498 | 10.93% | 495 | 0.99 |
| 3 | **Lung Opacity** | 452 | 9.92% | 631 | 1.40 |
| 4 | **Pleural Other** | 429 | 9.42% | 623 | 1.45 |
| 5 | **Support Devices** | 274 | 6.02% | 462 | 1.69 |
| 6 | **Atelectasis** | 259 | 5.69% | 270 | 1.04 |
| 7 | **Lung Lesion** | 248 | 5.44% | 353 | 1.42 |
| 8 | **Pleural Effusion** | 205 | 4.50% | 273 | 1.33 |
| 9 | **Fracture** | 190 | 4.17% | 197 | 1.04 |
| 10 | **Consolidation** | 182 | 4.00% | 254 | 1.40 |
| 11 | **Enlarged Cardiomediastinum** | 96 | 2.11% | 94 | 0.98 |
| 12 | **Edema** | 91 | 2.00% | 162 | 1.78 |
| 13 | **Pneumothorax** | 10 | 0.22% | 8 | 0.80 |
| 14 | **Pneumonia** | 0 | 0.00% | 0 | 0.00 |

### Statistics
- **Images with findings:** 3,099 (68.04%)
- **Average CheXpert categories per image:** 0.64

---

## Detailed Mapping: PadChest → CheXpert

### 1. Atelectasis (259 images, 270 bboxes)
**6 PadChest labels mapped:**

| PadChest Label | Count |
|----------------|-------|
| laminar atelectasis | 154 |
| atelectasis | 87 |
| lobar atelectasis | 11 |
| atelectasis basal | 6 |
| segmental atelectasis | 4 |
| total atelectasis | 1 |

**Clinical Note:** All variants of atelectasis (laminar, lobar, segmental, total) are mapped to this category.

---

### 2. Cardiomegaly (498 images, 495 bboxes)
**1 PadChest label mapped:**

| PadChest Label | Count |
|----------------|-------|
| cardiomegaly | 498 |

**Clinical Note:** Direct 1:1 mapping. This is the second most common CheXpert finding in the dataset.

---

### 3. Consolidation (182 images, 254 bboxes)
**2 PadChest labels mapped:**

| PadChest Label | Count |
|----------------|-------|
| alveolar pattern | 146 |
| consolidation | 42 |

**Clinical Note:** Alveolar pattern typically indicates consolidation process.

**Co-occurrence:** 51.1% of consolidation cases appear with Lung Opacity.

---

### 4. Edema (91 images, 162 bboxes)
**3 PadChest labels mapped:**

| PadChest Label | Count |
|----------------|-------|
| hilar congestion | 78 |
| ground glass pattern | 11 |
| kerley lines | 4 |

**Clinical Note:** No direct "edema" label in PadChest. Mapped based on radiological signs of pulmonary edema (hilar congestion, Kerley lines, ground glass pattern).

**Co-occurrence:** 37.4% appear with both Lung Opacity and Cardiomegaly (suggesting cardiac origin).

---

### 5. Enlarged Cardiomediastinum (96 images, 94 bboxes)
**4 PadChest labels mapped:**

| PadChest Label | Count |
|----------------|-------|
| aortic button enlargement | 38 |
| superior mediastinal enlargement | 23 |
| mediastinic lipomatosis | 21 |
| mediastinal enlargement | 17 |

**Clinical Note:** Various causes of mediastinal widening are mapped here.

**Co-occurrence:** 22.9% co-occur with Cardiomegaly.

---

### 6. Fracture (190 images, 197 bboxes)
**8 PadChest labels mapped:**

| PadChest Label | Count |
|----------------|-------|
| callus rib fracture | 108 |
| vertebral anterior compression | 41 |
| rib fracture | 22 |
| humeral fracture | 13 |
| clavicle fracture | 10 |
| vertebral fracture | 3 |
| vertebral compression | 3 |
| fracture | 1 |

**Clinical Note:** Includes all bone fractures visible on chest X-ray (ribs, vertebrae, clavicle, humerus).

---

### 7. Lung Lesion (248 images, 353 bboxes)
**9 PadChest labels mapped:**

| PadChest Label | Count |
|----------------|-------|
| calcified granuloma | 100 |
| nodule | 89 |
| pulmonary mass | 24 |
| granuloma | 20 |
| cavitation | 14 |
| multiple nodules | 13 |
| mass | 4 |
| cyst | 2 |
| abscess | 1 |

**Clinical Note:** Includes focal lung lesions (nodules, masses, granulomas, cysts, cavities).

---

### 8. Lung Opacity (452 images, 631 bboxes)
**7 PadChest labels mapped:**

| PadChest Label | Count |
|----------------|-------|
| interstitial pattern | 195 |
| infiltrates | 154 |
| increased density | 112 |
| reticular interstitial pattern | 10 |
| reticulonodular interstitial pattern | 4 |
| air bronchogram | 2 |
| miliary opacities | 2 |

**Clinical Note:** Non-specific increased density in lungs. Third most common category.

**Co-occurrence:** Frequently appears with Consolidation (20.6%), Support Devices (16.6%), and Cardiomegaly (16.4%).

---

### 9. Pleural Effusion (205 images, 273 bboxes)
**4 PadChest labels mapped:**

| PadChest Label | Count |
|----------------|-------|
| pleural effusion | 208 |
| loculated pleural effusion | 4 |
| loculated fissural effusion | 3 |
| hydropneumothorax | 2 |

**Clinical Note:** All types of pleural fluid collections.

**Co-occurrence:** 32.7% with Lung Opacity, 32.2% with Support Devices, 27.3% with Consolidation.

---

### 10. Pleural Other (429 images, 623 bboxes)
**9 PadChest labels mapped:**

| PadChest Label | Count |
|----------------|-------|
| apical pleural thickening | 205 |
| costophrenic angle blunting | 191 |
| pleural thickening | 20 |
| minor fissure thickening | 16 |
| calcified pleural thickening | 11 |
| calcified pleural plaques | 6 |
| fissure thickening | 2 |
| major fissure thickening | 2 |
| pleural mass | 1 |

**Clinical Note:** Pleural abnormalities other than effusion. Fourth most common category.

---

### 11. Pneumonia (0 images)
**No PadChest labels mapped.**

**Clinical Note:** ⚠️ **CRITICAL GAP** - No pneumonia labels in PadChest-GR. This is a major limitation for training models on this condition.

---

### 12. Pneumothorax (10 images, 8 bboxes)
**1 PadChest label mapped:**

| PadChest Label | Count |
|----------------|-------|
| pneumothorax | 10 |

**Clinical Note:** Very rare in this dataset (0.22% of images).

---

### 13. Support Devices (274 images, 462 bboxes)
**15 PadChest labels mapped:**

| PadChest Label | Count |
|----------------|-------|
| pacemaker | 104 |
| NSG tube | 52 |
| central venous catheter via jugular vein | 46 |
| endotracheal tube | 39 |
| dual chamber device | 35 |
| tracheostomy tube | 26 |
| single chamber device | 19 |
| central venous catheter via subclavian vein | 17 |
| reservoir central venous catheter | 15 |
| central venous catheter | 15 |
| chest drain tube | 7 |
| electrical device | 4 |
| catheter | 3 |
| gastrostomy tube | 1 |
| ventriculoperitoneal drain tube | 1 |

**Clinical Note:** Comprehensive coverage of medical devices and tubes.

---

### 14. No Finding (1,456 images)
**Automatically assigned when no abnormal findings present.**

**Clinical Note:** 31.96% of the dataset has normal findings.

---

## Category Co-occurrence Analysis

### High Co-occurrence Pairs (>25%)

| Primary Category | Co-occurring Category | Images | Percentage |
|------------------|----------------------|--------|------------|
| Consolidation | Lung Opacity | 93 | 51.1% |
| Edema | Lung Opacity | 34 | 37.4% |
| Edema | Cardiomegaly | 34 | 37.4% |
| Pleural Effusion | Lung Opacity | 67 | 32.7% |
| Pleural Effusion | Support Devices | 66 | 32.2% |
| Pneumothorax | Pleural Effusion | 4 | 40.0% |
| Support Devices | Lung Opacity | 75 | 27.4% |
| Consolidation | Pleural Effusion | 56 | 30.8% |
| Consolidation | Support Devices | 49 | 26.9% |

**Clinical Insights:**
- Consolidation and Lung Opacity frequently co-occur (overlap in radiological appearance)
- Edema associates with cardiomegaly and lung opacity (cardiac edema pattern)
- Support Devices commonly appear with acute findings (critically ill patients)

---

## Unmapped Labels

**Total:** 85 unique labels, 4,178 mentions (56.99% of all label mentions)

### Top 20 Unmapped Labels

| Label | Count | Category |
|-------|-------|----------|
| chronic changes | 493 | Chronic |
| aortic elongation | 464 | Vascular/Chronic |
| scoliosis | 408 | Skeletal |
| vertebral degenerative changes | 269 | Skeletal/Chronic |
| aortic atheromatosis | 243 | Vascular/Chronic |
| air trapping | 225 | Lung/Chronic |
| vascular hilar enlargement | 193 | Vascular |
| fibrotic band | 153 | Lung/Chronic |
| kyphosis | 103 | Skeletal |
| pseudonodule | 103 | Benign |
| hiatal hernia | 86 | Anatomical |
| diaphragmatic eventration | 84 | Anatomical |
| hemidiaphragm elevation | 81 | Anatomical |
| sternotomy | 76 | Post-surgical |
| volume loss | 75 | Lung/Chronic |
| suture material | 74 | Post-surgical |
| hilar enlargement | 69 | Vascular |
| nipple shadow | 68 | Benign |
| metal | 55 | Foreign body |
| gynecomastia | 53 | Benign |

### Categories of Unmapped Labels

1. **Chronic/Degenerative Changes** (1,034 mentions)
   - chronic changes, vertebral degenerative changes, osteopenia, osteoporosis, etc.

2. **Vascular/Cardiac Chronic** (967 mentions)
   - aortic elongation, aortic atheromatosis, vascular hilar enlargement, etc.

3. **Skeletal Deformities** (511 mentions)
   - scoliosis, kyphosis, thoracic cage deformation, etc.

4. **Chronic Lung Changes** (656 mentions)
   - air trapping, fibrotic band, bronchiectasis, volume loss, etc.

5. **Post-surgical/Prostheses** (363 mentions)
   - sternotomy, mastectomy, suture material, prostheses, etc.

6. **Benign/Normal Variants** (393 mentions)
   - nipple shadow, pseudonodule, gynecomastia, azygos lobe, etc.

7. **Anatomical Variants** (251 mentions)
   - hiatal hernia, diaphragmatic eventration, hemidiaphragm elevation, etc.

---

## Dataset Characteristics

### Strengths

1. **Well-represented Categories:**
   - Cardiomegaly (498 images, 10.93%)
   - Lung Opacity (452 images, 9.92%)
   - Pleural Other (429 images, 9.42%)
   - Support Devices (274 images, 6.02%)

2. **Grounded Annotations:**
   - 66% of images have bounding boxes
   - Average 1.70 bboxes per image (for abnormal findings)

3. **Chronic Findings:**
   - Excellent coverage of chronic/structural pathology
   - Useful for comprehensive chest X-ray interpretation

### Limitations

1. **Missing CheXpert Categories:**
   - **Pneumonia:** 0 images (major gap)
   - **Pneumothorax:** Only 10 images (very rare)

2. **Acute Pathology Underrepresented:**
   - Limited acute consolidation (182 images vs chronic changes)
   - Minimal pneumothorax
   - No pneumonia

3. **Label Distribution Skew:**
   - 56.99% of label mentions are unmapped (chronic findings)
   - Dataset biased toward chronic/outpatient imaging

4. **Low Multi-label Density:**
   - Average 0.64 CheXpert categories per image
   - Many images have single or no CheXpert findings

---

## Recommendations for RRG Integration

### 1. Training Strategy

**Option A: Selective Training (Recommended)**
- Train only on well-represented categories (>100 images)
- Skip Pneumonia and Pneumothorax for PadChest-GR
- Use MIMIC-CXR for these categories instead

**Option B: Augmented Training**
- Combine PadChest-GR with MIMIC-CXR
- Use PadChest for chronic findings, MIMIC for acute pathology
- Weighted sampling to balance distribution

### 2. Data Filtering

Consider filtering images by:
- Exclude "No Finding" if focusing on abnormality detection (saves 31.96%)
- Focus on images with bboxes (66% of dataset) for region-grounded training
- Filter images with only unmapped chronic findings

### 3. Label Augmentation

**Chronic Context:**
- Consider creating a "Chronic Changes" meta-category
- Use unmapped labels as additional context features
- May improve report generation naturalness

**Multi-label Handling:**
- Average 0.64 CheXpert categories per image is low
- Consider hierarchical labeling (CheXpert + PadChest-specific)

### 4. Complementary Use

**PadChest-GR strengths complement MIMIC-CXR:**

| Dataset | Strength |
|---------|----------|
| MIMIC-CXR | Acute pathology, pneumonia, ICU findings |
| PadChest-GR | Chronic findings, skeletal, cardiac, grounded regions |

**Recommended Split:**
- Use PadChest-GR for: Cardiomegaly, Fracture, Pleural Other, Support Devices
- Use MIMIC-CXR for: Pneumonia, Edema, Consolidation, acute Pneumothorax

---

## Usage

### Python Scripts Available

1. **`padchest_to_chexpert_mapping.py`**
   - Core mapping dictionary
   - Conversion functions
   - Usage:
     ```python
     from padchest_to_chexpert_mapping import convert_image_to_chexpert_labels

     chexpert_result = convert_image_to_chexpert_labels(image_data)
     ```

2. **`analyze_padchest_distribution.py`**
   - Original PadChest label analysis
   - Usage: `python analyze_padchest_distribution.py`

3. **`analyze_padchest_with_chexpert_mapping.py`**
   - CheXpert-mapped analysis
   - Usage: `python analyze_padchest_with_chexpert_mapping.py`

### Example: Convert Image to CheXpert

```python
from padchest_to_chexpert_mapping import convert_image_to_chexpert_labels

# Load your PadChest image data
image_data = {
    "ImageID": "...",
    "findings": [
        {
            "abnormal": True,
            "labels": ["cardiomegaly", "pleural effusion"],
            "locations": ["cardiac", "left"],
            "boxes": [[0.1, 0.2, 0.3, 0.4]]
        }
    ]
}

# Convert to CheXpert categories
result = convert_image_to_chexpert_labels(image_data)

# Result structure:
# {
#     'Cardiomegaly': {
#         'present': True,
#         'padchest_labels': ['cardiomegaly'],
#         'locations': ['cardiac'],
#         'bbox_count': 1
#     },
#     'Pleural Effusion': {
#         'present': True,
#         'padchest_labels': ['pleural effusion'],
#         'locations': ['left'],
#         'bbox_count': 1
#     },
#     ...
# }
```

---

## Conclusion

The comprehensive mapping enables integration of PadChest-GR into the RRG framework with 12/14 CheXpert categories covered. The dataset's strength lies in chronic/structural findings and grounded annotations, making it complementary to acute-focused datasets like MIMIC-CXR.

**Key Takeaway:** Use PadChest-GR as a **complement** rather than a **replacement** for existing CheXpert-labeled datasets, leveraging its unique coverage of chronic findings and bounding box annotations.
