# Project Methodology: Shortcut Learning in Vision-Language Models for Chest X-Ray Report Generation

## 1. Research Question

Do vision-language models (VLMs) for radiology report generation actually rely on pathology-specific visual features, or do they exploit shortcuts — spurious correlations, co-occurring findings, or language priors — to produce plausible-sounding reports?

We answer this by systematically occluding annotated disease regions in chest X-rays and measuring how report quality degrades (or fails to degrade) across 9 state-of-the-art VLMs.

---

## 2. Dataset: PadChest-GR

**PadChest-GR** (BIMCV-Padchest-GR, released August 2024) is a Spanish chest X-ray dataset with **grounded radiology reports** — each finding annotation is linked to specific anatomical regions with bounding boxes, enabling targeted occlusion experiments.

### Key Statistics

| Metric | Value |
|--------|-------|
| Total images | 4,555 |
| Images with abnormal findings | 3,099 (68.0%) |
| Images with normal findings | 2,786 (61.2%) |
| Total finding annotations | 10,479 (67.1% abnormal, 32.9% normal) |
| Images with bounding boxes | 3,008 (66.0%) |
| Total bounding boxes | 7,733 (mean 1.70/image, max 14) |
| Unique anatomical locations | 101 |
| Unique PadChest labels | 154 |

### Annotation Structure

Each image has:
- **Report**: combined radiology text assembled from per-region sentences
- **Labels**: 14-element binary vector following the CheXpert labeling scheme
- **Regions**: list of annotated findings, each with:
  - Anatomy (e.g., "lung", "heart")
  - Bounding boxes (normalized [0,1] coordinates)
  - Findings text and CheXpert category mapping
  - Per-region CheXpert labels

### CheXpert Label Scheme (14 labels)

Enlarged Cardiomediastinum, Cardiomegaly, Lung Opacity, Lung Lesion, Edema, Consolidation, Pneumonia, Atelectasis, Pneumothorax, Pleural Effusion, Pleural Other, Fracture, Support Devices, No Finding.

### Per-Category Stratification

For occlusion experiments, the dataset is stratified by CheXpert category. Eight categories with >= 50 samples are used: **Atelectasis, Cardiomegaly, Fracture, Lung Lesion, Lung Opacity, Pleural Effusion, Pleural Other, Support Devices**. Each sample is keyed as `"{image_path}::{category}"`, allowing the same image to contribute to multiple categories.

---

## 3. Models Under Evaluation

Nine VLMs spanning diverse architectures, parameter counts, and training paradigms:

| # | Model | HuggingFace ID | Params | Architecture | Decoding |
|---|-------|----------------|--------|--------------|----------|
| 1 | **MedGemma** | `google/medgemma-1.5-4b-it` | 4B | Gemma multimodal, instruction-tuned | Greedy |
| 2 | **MAIRA-2** | `microsoft/maira-2` | — | Custom multimodal with structured clinical context input (frontal, lateral, indication, technique) | Greedy |
| 3 | **CXRMateED** | `aehrc/cxrmate-rrg24` | — | Encoder-decoder with CXR-specific image encoder; outputs findings + impression sections | Beam search (4) |
| 4 | **NV-Reason-CXR** | `nvidia/NV-Reason-CXR-3B` | 3B | Qwen-based with chain-of-thought reasoning (2048 max tokens) | Greedy |
| 5 | **CheXOne** | `StanfordAIMI/CheXOne` | — | Qwen2.5-VL architecture | Greedy |
| 6 | **Libra** | `X-iZhang/libra-v1.0-7b` | 7B | LLaVA-based with temporal attention crossover (TAC) for prior image integration | Sampling (T=0.2) |
| 7 | **LLaVA-Rad** | `X-iZhang/libra-llava-rad` | — | LLaVA fine-tuned on radiology, always uses two-slot (current+prior) image format | Sampling (T=0.2) / Beam (5) |
| 8 | **CheXagent** | `StanfordAIMI/CheXagent-2-3b-srrg-findings` | 3B | Causal LM with chat template; images saved to disk for tokenizer path requirement | Greedy |
| 9 | **RaDialog** | `Chantal/RaDialog-...` | 7B | LLaVA v1.5 + LoRA; **two-stage pipeline**: (1) CheXpert classifier predicts finding labels from image, (2) labels prepended to prompt for conditional generation | Greedy |

### Model-Specific Prompts

| Model | Prompt |
|-------|--------|
| MedGemma | "Describe this X-ray." |
| MAIRA-2 | (empty — processor formats structured radiology input) |
| CXRMateED | (empty — autonomous generation) |
| NV-Reason-CXR | "Find abnormalities and support devices." |
| CheXOne | "Write an example findings section for the CXR." |
| Libra | "Provide a detailed description of the findings in the radiology image." |
| LLaVA-Rad | "Describe the findings in the radiology image." |
| CheXagent | "Structured Radiology Report Generation for Findings Section" |
| RaDialog | "You are to act as a radiologist and write the finding section of a chest x-ray radiology report..." + dynamically prepended predicted findings |

---

## 4. Image Preprocessing

### Base Normalization (all experiments)

1. Load image as float32 numpy array
2. **CXR-specific min-max normalization**: `(pixel - min) / (max - min)` (NOT ImageNet normalization)
3. Scale to uint8 [0, 255]
4. Convert to RGB

### Correlated Noise Generation

Used for all occlusion strategies (OCO, DOCO, RO). Generates noise that matches the local image statistics:

1. Identify background pixels (outside occlusion mask)
2. Compute **robust statistics**: per-channel median and std after trimming 1st/99th percentile outliers
3. Generate Gaussian noise: `noise = randn(H, W, 3) * scale + center`
4. Apply Gaussian blur (radius=2.0) for spatial correlation
5. Clip to [0, 255]

### Feathered Blending

Occlusion regions use soft-edged blending to avoid hard boundary artifacts:

1. Alpha map = `region_mask * occlusion_strength`
2. Gaussian blur alpha (radius=6 pixels) for feathered edges
3. Blend: `output = original * (1 - alpha) + noise * alpha`

---

## 5. Experiment Design

### Overview

```
                         ┌─────────────────────────────────────┐
                         │         Input: Chest X-Ray           │
                         │      + Annotated Disease Bboxes      │
                         └──────────────┬──────────────────────┘
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              │                         │                         │
     ┌────────▼────────┐    ┌──────────▼──────────┐   ┌─────────▼─────────┐
     │   No Occlusion   │    │  Targeted Occlusion  │   │  Control Occlusion │
     │                  │    │                      │   │                   │
     │  • baseline      │    │  • OCO  (all bboxes) │   │  • RO  (random    │
     │  (full image)    │    │  • DOCO (co-occur.)  │   │        locations) │
     │                  │    │  • ROCO (one bbox)   │   │  • all_noise      │
     └────────┬─────────┘    └──────────┬───────────┘   │  (full image)    │
              │                         │               └────────┬──────────┘
              │                         │                        │
              └─────────────────────────┼────────────────────────┘
                                        │
                                        ▼
                              ┌──────────────────┐
                              │   9 VLM Models    │
                              │   (inference)     │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │  Generated Report │
                              └────────┬─────────┘
                                       │
                                       ▼
                              ┌──────────────────┐
                              │   Evaluation      │
                              │  (6 metric types) │
                              └──────────────────┘
```

### Experiment Types

| Experiment | What Is Occluded | Purpose |
|------------|-----------------|---------|
| **baseline** | Nothing | Reference performance (upper bound) |
| **OCO** (Object Class Occlusion) | ALL annotated disease bounding boxes | Tests sensitivity to primary pathology regions |
| **DOCO** (Drop OCO) | ALL co-occurrence bounding boxes; samples without bboxes dropped | Tests reliance on co-occurring findings |
| **ROCO** (Random Object Class Occlusion) | ONE randomly selected annotated bbox | Gentler variant — partial region occlusion |
| **RO** (Random Occlusion) | N random bboxes in **unannotated** regions | Negative control — noise outside pathology areas |
| **all_noise** | Entire image replaced with uniform random noise | Tests pure language prior (no visual info) |
| **all_noise_mean** | Entire image replaced with structure-matched correlated noise | Destroys spatial info while preserving intensity distribution |

### Occlusion Strength Levels

Each occlusion experiment is tested at multiple blend strengths:

| Level | Strength | Meaning |
|-------|----------|---------|
| p00 | 0% | No occlusion (same as baseline) |
| p20 | 20% | 80% original + 20% noise |
| p40 | 40% | 60% original + 40% noise |
| p60 | 60% | 40% original + 60% noise |
| p80 | 80% | 20% original + 80% noise |
| p100 | 100% | Fully replaced by noise |

This produces a **dose-response curve** for each model — revealing how gracefully performance degrades as pathology information is progressively removed.

### Key Experimental Contrasts

1. **OCO vs. Baseline**: Does removing the pathology region hurt performance? If not → shortcut learning.
2. **OCO vs. RO**: Does it matter WHERE noise is placed? If OCO and RO produce similar degradation → model isn't region-specific.
3. **OCO vs. all_noise**: If full-image noise performs similarly to targeted OCO → model relies on language priors, not vision.
4. **DOCO analysis**: Does occluding co-occurring regions affect reports? Tests whether models use co-occurrence patterns as shortcuts.
5. **Dose-response (p20→p100)**: Smooth degradation = genuine visual reliance; abrupt cliff or plateau = threshold-based or shortcut behavior.

---

## 6. Inference Pipeline

### Flow

```
1. CLI argument parsing (model, experiment, dataset, seed)
        │
        ▼
2. Dataset loading
   • baseline: full verified_samples.json
   • OCO/DOCO/RO: per-category JSONs, composite keys "{path}::{category}"
   • DOCO: additional filtering — drop samples without bboxes
        │
        ▼
3. Parallel preprocessing (16 workers, multiprocessing)
   • Per-sample deterministic seed: (base_seed + sample_idx) % 2^32
   • Apply occlusion strategy → PIL Image
   • Cache to disk as .npy files
        │
        ▼
4. Batched model inference (PyTorch DataLoader)
   • Load cached images, batch by --num-images (model-specific)
   • model(images, prompt) → generated text
   • torch.no_grad(), bfloat16 precision
        │
        ▼
5. Output writing
   • JSON array: [{image_path, prediction, reference, label, target_category}, ...]
   • Path: outputs/{experiment}/{pXX}/{dataset}/{model_id}::seed={seed}.json
```

### Reproducibility

- Random seed fixed to **3** (configurable) for `numpy` and `random`
- Per-sample seeds ensure identical preprocessing regardless of worker parallelism or execution order
- Greedy decoding (most models) eliminates sampling variance

---

## 7. Evaluation Metrics

Six complementary metric types capture different aspects of report quality:

### 7.1 Lexical Metrics

| Metric | What It Measures | Method |
|--------|-----------------|--------|
| **BLEU-1** | Unigram precision | Corpus-level BLEU (sacrebleu) |
| **BLEU-4** | 4-gram precision with brevity penalty | Standard MT metric adapted for reports |
| **ROUGE-L** | Longest common subsequence F1 | Captures fluency and content overlap |

### 7.2 Clinical Factuality: CheXbert

A BERT-based classifier (`bert-base-uncased` + 14 linear heads) that extracts clinical findings from free text:

1. Run CheXbert on both prediction and reference texts
2. Extract 14 binary finding labels (two binarization modes: uncertain=negative, uncertain=positive)
3. Compute micro-F1 and macro-F1 over 14-label and 5-label subsets

**5-label subset**: Cardiomegaly, Edema, Consolidation, Atelectasis, Pleural Effusion (most clinically actionable).

**8 CheXbert metrics total**: {Micro, Macro} × {14-label, 5-label} × {uncertain=neg, uncertain=pos}

### 7.3 Clinical Entity Matching: RadGraph-F1

Uses a named entity recognition + relation extraction model to parse radiology reports into structured knowledge graphs:

1. Extract entities (anatomy, observation) and relations (located_at, suggestive_of) from both prediction and reference
2. **Partial matching**: entities without relations must match exactly (token + label); entities with relations must match token + label + has_relations (ignoring specific targets)
3. Compute F1 = 2PR/(P+R) between prediction and reference entity sets

### 7.4 LLM-Based: GREEN (optional)

Stanford AIMI's GREEN metric uses a fine-tuned LLaMA-2 7B model (`StanfordAIMI/GREEN-radllama2-7b`) to score report quality via natural language understanding.

### Confidence Intervals

All metrics support bootstrap confidence intervals: 500 resamples, 95% confidence, percentile method, seed=3.

---

## 8. Evaluation Pipeline

```
outputs/{experiment}/{pXX}/{dataset}/{model}.json
        │
        ▼
   run_eval.py --filepath <file> --output-mode per-experiment
        │
        ├── BLEU-1, BLEU-4 (sacrebleu, 500 bootstrap)
        ├── ROUGE-L (custom, 500 bootstrap)
        ├── F1-RadGraph partial (radgraph NER+RE, 500 bootstrap)
        ├── CheXbert 14-label & 5-label F1 (BERT classifier, 500 bootstrap)
        └── [optional] GREEN (LLaMA-2 7B)
        │
        ▼
   results/{experiment}/{dataset}/results.csv
   (one row per model, columns = metric values + CI bounds)
```

### Output Modes

- **per-file**: one CSV per input JSON, mirroring the outputs directory structure
- **per-experiment**: aggregated CSV with all models for one experiment/dataset combination

---

## 9. Computational Infrastructure

- **Cluster**: NAISS Alvis (Swedish national HPC)
- **SLURM project**: NAISS2025-5-662
- **GPU**: NVIDIA A40 (1 per job)
- **Precision**: bfloat16
- **Python**: 3.11.5
- **5 virtual environments** (due to dependency conflicts between model families)

### Scale

- 9 models × 6 occlusion types × 6 strength levels × 8 disease categories = **~2,592 experiment configurations**
- Each configuration produces one output JSON and one evaluation CSV
- Total SLURM jobs per full experiment sweep: ~54 per experiment type (9 models × 6 levels)

---

## 10. Summary: What the Methodology Graph Should Show

The methodology has four main stages:

1. **Data Preparation**: PadChest-GR dataset → per-category stratification → bounding box extraction
2. **Image Manipulation**: 6 occlusion strategies × 6 strength levels applied using correlated noise with feathered blending
3. **Model Inference**: 9 VLMs generate radiology reports from manipulated images
4. **Evaluation**: 6 metric types (BLEU, ROUGE, CheXbert, RadGraph, optionally GREEN) quantify report quality degradation

The core analysis compares metric scores across occlusion conditions to determine whether each model genuinely relies on pathology-specific visual features or exploits shortcuts.
