# PadChest-GR Dataset Distribution Analysis

**Analysis Date:** 2026-01-20
**Dataset:** BIMCV-Padchest-GR grounded_reports_20240819.json

---

## Dataset Overview

- **Total Images:** 4,555
- **Images with Abnormal Findings:** 3,099 (68.0%)
- **Images with Normal Findings:** 2,786 (61.2%)
- **Total Findings:** 10,479
  - Abnormal: 7,037 (67.1%)
  - Normal: 3,442 (32.9%)

---

## Bounding Box Statistics

### Coverage
- **Images with Bounding Boxes:** 3,008 (66.04%)
- **Images with Extra Bounding Boxes:** 2,775 (60.92%)

### Bounding Boxes per Image
| Metric | Value |
|--------|-------|
| Mean | 1.70 |
| Median | 1.00 |
| Min | 0 |
| Max | 14 |
| Std | 1.90 |
| **Total** | **7,733** |

### Extra Bounding Boxes per Image
| Metric | Value |
|--------|-------|
| Mean | 1.38 |
| Median | 1.00 |
| Min | 0 |
| Max | 13 |
| Std | 1.63 |
| **Total** | **6,265** |

### Distribution
- 0 bboxes: 1,547 images (34.0%)
- 1 bbox: 1,057 images (23.2%)
- 2 bboxes: 761 images (16.7%)
- 3 bboxes: 462 images (10.1%)
- 4+ bboxes: 728 images (16.0%)

---

## Location Distribution

- **Total Unique Locations:** 101
- **Total Location Mentions:** 8,834

### Top 20 Locations
| Rank | Location | Count |
|------|----------|-------|
| 1 | right | 1,072 |
| 2 | left | 775 |
| 3 | aortic | 737 |
| 4 | pleural | 575 |
| 5 | cardiac | 520 |
| 6 | hilar | 394 |
| 7 | basal | 385 |
| 8 | bilateral | 380 |
| 9 | diaphragm | 228 |
| 10 | costophrenic angle | 202 |
| 11 | apical | 202 |
| 12 | hemithorax | 187 |
| 13 | rib | 175 |
| 14 | basal bilateral | 132 |
| 15 | bone | 130 |
| 16 | right upper lobe | 120 |
| 17 | pectoral | 117 |
| 18 | tracheal | 111 |
| 19 | bronchi | 105 |
| 20 | dorsal vertebrae | 103 |

---

## Label Distribution

- **Total Unique Labels:** 154
- **Total Label Mentions:** 7,331

### Top 30 Labels
| Rank | Label | Count |
|------|-------|-------|
| 1 | cardiomegaly | 498 |
| 2 | chronic changes | 493 |
| 3 | aortic elongation | 464 |
| 4 | scoliosis | 408 |
| 5 | vertebral degenerative changes | 269 |
| 6 | aortic atheromatosis | 243 |
| 7 | air trapping | 225 |
| 8 | pleural effusion | 208 |
| 9 | apical pleural thickening | 205 |
| 10 | interstitial pattern | 195 |
| 11 | vascular hilar enlargement | 193 |
| 12 | costophrenic angle blunting | 191 |
| 13 | infiltrates | 154 |
| 14 | laminar atelectasis | 154 |
| 15 | fibrotic band | 153 |
| 16 | alveolar pattern | 146 |
| 17 | increased density | 112 |
| 18 | callus rib fracture | 108 |
| 19 | pacemaker | 104 |
| 20 | kyphosis | 103 |
| 21 | pseudonodule | 103 |
| 22 | calcified granuloma | 100 |
| 23 | nodule | 89 |
| 24 | atelectasis | 87 |
| 25 | hiatal hernia | 86 |
| 26 | diaphragmatic eventration | 84 |
| 27 | hemidiaphragm elevation | 81 |
| 28 | hilar congestion | 78 |
| 29 | sternotomy | 76 |
| 30 | volume loss | 75 |

---

## CheXpert Label Mapping

### 14 CheXpert Labels Used in RRG
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

### Matching Summary

| Category | Count | Percentage |
|----------|-------|------------|
| **Matched Label Mentions** | 1,184 | 16.15% |
| **Unmatched Label Mentions** | 6,147 | 83.85% |
| **Unique Matched Labels** | 18 | - |
| **Unique Unmatched Labels** | 136 | - |

### PadChest Labels Matching CheXpert (18 labels)

| PadChest Label | Count | Maps to CheXpert |
|----------------|-------|------------------|
| cardiomegaly | 498 | Cardiomegaly |
| pleural effusion | 208 | Pleural Effusion |
| laminar atelectasis | 154 | Atelectasis |
| callus rib fracture | 108 | Fracture |
| atelectasis | 87 | Atelectasis |
| consolidation | 42 | Consolidation |
| rib fracture | 22 | Fracture |
| humeral fracture | 13 | Fracture |
| lobar atelectasis | 11 | Atelectasis |
| clavicle fracture | 10 | Fracture |
| pneumothorax | 10 | Pneumothorax |
| atelectasis basal | 6 | Atelectasis |
| segmental atelectasis | 4 | Atelectasis |
| loculated pleural effusion | 4 | Pleural Effusion |
| vertebral fracture | 3 | Fracture |
| hydropneumothorax | 2 | Pneumothorax |
| total atelectasis | 1 | Atelectasis |
| fracture | 1 | Fracture |

### Key Observations

**Coverage:**
- Only **16.15%** of PadChest label mentions directly match the 14 CheXpert labels
- The dataset contains **154 unique labels**, far more granular than the 14 CheXpert categories
- Many PadChest labels represent chronic/degenerative conditions not in CheXpert (e.g., scoliosis, aortic elongation, vertebral degenerative changes)

**CheXpert Labels Present in PadChest:**
- ✅ Atelectasis (265 total mentions across variants)
- ✅ Cardiomegaly (498 mentions)
- ✅ Consolidation (42 mentions)
- ✅ Fracture (167 total mentions across variants)
- ✅ Pleural Effusion (212 total mentions across variants)
- ✅ Pneumothorax (12 mentions)

**CheXpert Labels NOT Found or Rare in PadChest:**
- ❌ Edema (0 mentions)
- ❌ Enlarged Cardiomediastinum (0 mentions, but "mediastinal enlargement" exists with 17 mentions)
- ❌ Lung Lesion (0 mentions)
- ❌ Lung Opacity (0 mentions, but "increased density" and "opacities" exist)
- ❌ Pleural Other (0 mentions as such, but "pleural thickening" exists with 20 mentions)
- ❌ Pneumonia (0 mentions)
- ❌ Support Devices (0 mentions as category, but specific devices like "pacemaker", "NSG tube", etc. exist)
- ❌ No Finding (0 mentions as label)

**Dataset Characteristics:**
- PadChest-GR focuses heavily on chronic/structural findings rather than acute pathology
- Strong emphasis on cardiac (cardiomegaly, aortic changes) and skeletal findings (fractures, degenerative changes)
- Many device-related findings (pacemakers, prostheses, surgical materials)
- Limited acute findings like pneumonia, edema, or acute consolidation

---

## Recommendations

1. **Label Mapping Strategy:** Consider creating a comprehensive mapping dictionary from PadChest's 154 labels to the 14 CheXpert categories
2. **Missing CheXpert Categories:** Some categories (Edema, Pneumonia, Enlarged Cardiomediastinum) are essentially absent - this dataset may not be suitable for training models on these conditions
3. **Dataset Complement:** PadChest-GR appears complementary to MIMIC-CXR, with more chronic/structural findings vs. acute pathology
4. **Bounding Box Usage:** 66% of images have grounded bounding boxes, which is valuable for region-based reasoning
