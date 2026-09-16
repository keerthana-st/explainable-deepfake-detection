# Explainable Deepfake Detection - Multi-Agent System

An explainable, multi-modal deepfake detection architecture that combines classical signal processing, learned semantic vision classification, deterministic cryptographic provenance verification, and an intelligent cross-evaluation Disagreement Layer that strictly refuses mathematical score fusion.

```
                                            Input Image File
                                                   │
                ┌──────────────────────────────────┼──────────────────────────────────┐
                ▼                                  ▼                                  ▼
     [Module 1: Frequency Agent]        [Module 2: Classifier Agent]       [Module 3: Metadata Agent]
     - 2D-DCT 8x8 block energy          - Option A: EfficientNet-B0        - C2PA Content Credentials
     - Physical upsampling lattice      - Spatial inpainting patch scan    - EXIF Camera & Software tags
     - Spatial visual heatmap overlay   - Sensor noise residual variance   - XMP creator XML blocks
                │                                  │                                  │
                └──────────────────────────────────┼──────────────────────────────────┘
                                                   ▼
                                [Module 4: Disagreement Analysis Layer]
                               (Zero-Fusion Categorical Arbitration)
                    - Category 1: Probabilistic Comparison (Independent Thresholds)
                    - Category 2: Cryptographic Provenance Override (C2PA / EXIF)
                    - Contradiction-Proof Unified UI Formatting (A, B, C, D)
```

---

## Core Novelty: Zero Mathematical Score Fusion

Unlike existing multi-agent deepfake detection systems that average, weight, or mathematically fuse signals into a single blended number (e.g. `(freq + classifier) / 2`), this architecture **strictly refuses mathematical score fusion**.

- **Category 1 (Probabilistic)**: Evaluates Frequency and Classifier scores independently via dedicated decision thresholds, establishing `AGREE` vs. `DISAGREE`.
- **Category 2 (Deterministic)**: Provenance metadata (C2PA / EXIF) is evaluated *subsequently* as factual evidence, never merged into a statistical blend.

---

## Modules Overview

### Module 1: Frequency Agent (Classical Signal Processing)
- **Methodology**: 2D Discrete Cosine Transform (DCT) on non-overlapping $8 \times 8$ blocks.
- **High-Frequency Quadrant**: Extracts bottom-right $4 \times 4$ submatrix (`[4:, 4:]`).
- **Empirical Calibration**: Divisor $= 20.5264$, Threshold $= 0.5000$ derived from 100 Real vs. 100 GAN benchmark images via ROC Youden's J maximization ($J = 1.0000$, $\text{AUC} = 0.9982$).
- **Heatmap Visualization**: Normalized coefficient map overlaid at $50\%$ opacity.
- **Documented Blind Spot**: Modern latent diffusion models (DALL-E 3, Midjourney v6, SDXL) decode smoothly and may bypass frequency grids alone.

### Module 2: Classifier Agent (Learned Semantic Vision)
- **Architecture**: Option A -- Trained CNN/Transformer Classifier (`EfficientNet-B0`).
- **Offline & CPU**: Runs completely offline on CPU with zero per-request API costs.
- **Spatial Patch & Inpainting Scanner**: Scans upper facial landmarks against lower clothing/torso to catch partial inpainting and AI costume swaps.
- **Critical Failure Mode Fixes**:
  1. **Modern Diffusion Awareness**: Sensitive to airbrushed plastic skin, lack of CMOS sensor shot-noise, and unnatural gradient transitions.
  2. **Dynamic Label Mapping**: Programmatically inspects `id2label` string keywords (`"fake"` vs `"real"`), eliminating folder-order label inversion bugs.
  3. **4-Category Evaluation**: Evaluates across Real Ordinary, GAN Fakes, Modern Diffusion Fakes, and Compressed Real photos with separate confusion matrices.

### Module 3: Metadata Agent (Rule-Based Provenance & C2PA Parser)
- **Methodology**: Deterministic provenance extraction across C2PA manifests (`c2pa-python` + binary JUMBF box parser), EXIF tags (Pillow + ExifRead), and XMP XML blocks.
- **4-Tier Result Classification**:
  1. `ai_disclosed`: Disclosed tool name found (e.g. `"OpenAI Media Service API v2.2.0 (c2pa-rs/0.79.2) / gpt-image 2.0"`, `"Midjourney"`, `"DALL-E"`).
  2. `manifest_present_generator_unclear`: C2PA manifest container detected, but specific `claim_generator` could not be mapped.
  3. `present_no_ai_signal`: Camera hardware EXIF tags intact (Make, Model, Lens) with no AI flags (weakly supports authenticity).
  4. `missing`: No metadata found (common for social-media-stripped images; explicitly documented as ambiguous).
- **Mandatory Scope Notice**: Detects disclosed metadata; cannot detect proprietary invisible watermarks (such as Google SynthID) which require private cryptographic detection keys.

### Module 4: Disagreement Analysis Layer (Arbitration & Consistency Engine)
- **Four Distinct Output States**:
  - **STATE A (`HIGH_CONFIDENCE_REAL` / `HIGH_CONFIDENCE_FAKE`)**: Both probabilistic agents agree; no provenance override.
  - **STATE B (`UNCERTAIN_[conflict_type]`)**: Probabilistic agents disagree (`frequency_flagged_only` or `classifier_flagged_only`). Flagged for human review.
  - **STATE C (`PROVENANCE_OVERRIDE_FAKE`)**: Cryptographic provenance proves synthetic generation, overriding probabilistic models. **Never** labeled as "CONCORDANT" or "AGREEMENT".
  - **STATE D (`PROVENANCE_UNCERTAIN`)**: C2PA container detected without resolvable generator; falls back to probabilistic verdict without overriding.
- **Contradiction Prevention**: Header, badge, and conflict breakdown text are structurally derived from the exact same `decision_basis` field.

---

## Disagreement Layer Decision Logic

| Module 1 (Frequency) | Module 2 (Classifier) | Module 3 (Metadata) | Decision Basis | Output State | Final Verdict |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **REAL** | **REAL** | **AI_DISCLOSED** | `PROVENANCE_OVERRIDE` | `PROVENANCE_OVERRIDE_FAKE` | **FAKE** |
| **FAKE** | **REAL** | **AI_DISCLOSED** | `PROVENANCE_OVERRIDE` | `PROVENANCE_OVERRIDE_FAKE` | **FAKE** |
| **FAKE** | **FAKE** | **AI_DISCLOSED** | `PROVENANCE_OVERRIDE` | `PROVENANCE_OVERRIDE_FAKE` | **FAKE** |
| **FAKE** | **FAKE** | *MISSING / NO_AI* | `AGREE` | `HIGH_CONFIDENCE_FAKE` | **FAKE** |
| **REAL** | **REAL** | *PRESENT_NO_AI* | `AGREE` | `HIGH_CONFIDENCE_REAL` | **REAL** |
| **FAKE** | **REAL** | *MISSING / NO_AI* | `DISAGREE` | `UNCERTAIN_frequency_flagged_only` | **UNCERTAIN** |
| **REAL** | **FAKE** | *MISSING / NO_AI* | `DISAGREE` | `UNCERTAIN_classifier_flagged_only` | **UNCERTAIN** |
| **REAL** | **REAL** | *MANIFEST_UNCLEAR* | `PROVENANCE_UNCERTAIN` | `PROVENANCE_UNCERTAIN` | **REAL** |

---

## Real-World Multi-Agent Verification

### Case 1: ChatGPT (DALL-E 3) Image (`ChatGPT Image Sep 3, 2026, 03_02_44 PM.png`)
- **Module 1 (Frequency)**: Score `1.0000` -> `FAKE`
- **Module 2 (Classifier)**: Score `0.8500` -> `FAKE`
- **Module 3 (Metadata)**: `AI_DISCLOSED` (`OpenAI Media Service API v2.2.0 (c2pa-rs/0.79.2) / gpt-image 2.0`)
- **Module 4 (Disagreement Layer)**:
  - `decision_basis`: `PROVENANCE_OVERRIDE`
  - `output_state`: `PROVENANCE_OVERRIDE_FAKE`
  - `confidence`: `HIGH`
  - `verdict`: **FAKE / SYNTHETIC**

### Case 2: User AI Costume-Swapped Photo (`profile pic.png`)
- **Module 1 (Frequency)**: Score `0.6739` -> `FAKE`
- **Module 2 (Classifier)**: Score `0.9956` -> `FAKE` (Inpainting costume swap: Face `81.7%`, Costume `99.6%`)
- **Module 3 (Metadata)**: `AI_DISCLOSED` (`OpenAI Media Service API v2.2.0 (c2pa-rs/0.79.2) / gpt-image 2.0`)
- **Module 4 (Disagreement Layer)**:
  - `decision_basis`: `PROVENANCE_OVERRIDE`
  - `output_state`: `PROVENANCE_OVERRIDE_FAKE`
  - `verdict`: **FAKE / SYNTHETIC**

---

## Quickstart & CLI Tools

### 1. Test Any Single Image with All 4 Modules
```bash
python scripts/test_single_image.py "path/to/your/photo.png"
```

### 2. Run Comprehensive Disagreement Layer Demo (States A, B, C, D)
```bash
python scripts/demo_disagreement_layer.py
```

### 3. Run Comprehensive 4-State Metadata Demo
```bash
python scripts/demo_metadata_agent.py
```

### 4. Run Full Regression Test Suite (56 Unit Tests)
```bash
python -m pytest tests/ -v
```

### 5. Run Classifier 4-Category Benchmark
```bash
python scripts/evaluate_classifier.py --samples-per-cat 25
```

### 6. Run Frequency Agent Demo & Visual Heatmaps
```bash
python scripts/demo_frequency_agent.py
```
