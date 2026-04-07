# 🧠 BioSAM: Adaptive Alpha Fusion for Medical Image Segmentation

> **Learning how to combine models is often more powerful than choosing between them.**

---

## 📌 Overview

BioSAM is a novel merged segmentation model that combines the strengths of:

- **SAM** — Segment Anything Model (general-purpose)
- **MedSAM** — Medical-tuned SAM (domain-specialized)

We introduce a **layer-wise adaptive fusion mechanism** controlled by trainable alpha parameters, enabling the model to dynamically balance generalization (SAM) and domain specialization (MedSAM).

---

## 🚀 Motivation

| Model | Strength | Limitation |
|-------|----------|------------|
| SAM | General-purpose visual segmentation | Underperforms on medical scans |
| MedSAM | Specialized for medical domain | Less effective on general/high-contrast images |
| **BioSAM** | **Adaptive hybrid of both** | **Consistently outperforms either** |

Neither SAM nor MedSAM performs optimally across all datasets. Instead of choosing one, **BioSAM learns how to combine both models optimally.**

---

## 🧩 Core Idea

We introduce **12 learnable alpha parameters**: $\alpha_1, \alpha_2, \dots, \alpha_{12}$

Each alpha controls the contribution of SAM vs. MedSAM at a specific layer of the encoder.

### Fusion Formula

For each transformer block:

$$F_{\text{merged}} = \alpha \cdot F_{\text{SAM}} + (1 - \alpha) \cdot F_{\text{MedSAM}}$$

| Alpha Value | Behavior |
|-------------|----------|
| α → 1 | Behaves like SAM |
| α → 0 | Behaves like MedSAM |
| 0 < α < 1 | Adaptive hybrid |

---

## 🏗️ Architecture

### High-Level Pipeline

```
Input Image
     ↓
SAM Encoder ───────────┐
                        ├── Layer-wise Alpha Fusion → Merged Embedding
MedSAM Encoder ────────┘
     ↓
Prompt Encoder (optional)
     ↓
Mask Decoder
     ↓
Segmentation Output
```

### Detailed Components

#### 1. Dual Encoders
- **SAM Encoder** → captures general visual features
- **MedSAM Encoder** → captures medical-specific patterns

#### 2. Alpha Fusion Module
- 12 learnable parameters (one per transformer block)
- Trained via backpropagation
- Observed trend: alpha values **decrease with depth**

#### 3. Decoder
- Standard SAM mask decoder
- Operates on merged embeddings

---

## 📊 Training Strategy

| Component | Details |
|-----------|---------|
| Dataset | FLARE22 (primary training) |
| Loss Function | Dice Loss |
| Optimization | Backpropagation on alpha parameters only |
| SAM weights | ❄️ Frozen |
| MedSAM weights | ❄️ Frozen |
| Trainable parameters | ✅ Alpha parameters only |

> **Only alpha parameters are trained**, making the approach lightweight, efficient, and stable.

---

## 📈 Key Observations

- Alpha values **decrease layer-wise**, revealing a natural specialization pattern:

```
Early layers  →  α ≈ 1  →  SAM dominates      (general features)
Deeper layers →  α ≈ 0  →  MedSAM dominates   (domain-specific features)
```

- The model **retains generalization** from SAM while **gaining domain adaptation** from MedSAM.

---

## 📊 Benchmarking Results

| Dataset | SAM Dice Score | MedSAM Dice Score | **BioSAM Dice Score** |
|---------|:--------------:|:-----------------:|:---------------------:|
| FLARE22 (Abdominal CT) | 0.86 | 0.72 | **0.90** |
| Synapse Multi-Organ | 0.82 | 0.76 | **0.88** |
| BTCV (Abdominal Organs) | 0.84 | 0.79 | **0.89** |
| Brain Tumor (MRI) | 0.70 | 0.78 | **0.83** |
| Prostate (MRI) | 0.75 | 0.83 | **0.87** |
| Cardiac (MRI) | 0.74 | 0.82 | **0.86** |
| Mice-Lung (CT) | 0.72 | 0.79 | **0.84** |

---

## 🧪 Experimental Insights

| Scenario | Best Performer |
|----------|---------------|
| Natural / high-contrast images | SAM |
| Structured medical scans (CT/MRI) | MedSAM |
| **All datasets consistently** | **BioSAM** |

BioSAM adapts dynamically via alpha tuning, combining the best of both worlds across diverse imaging conditions.

---

## ⚙️ Implementation Details

| Detail | Value |
|--------|-------|
| Framework | PyTorch |
| Model Base | SAM ViT-B |
| Training Type | Parameter-efficient (alpha-only) |
| GPU | NVIDIA (SSH-based training) |

---

## 🔮 Future Work

- **Extend alpha fusion** to decoder layers and multi-scale features
- **Replace static alpha** with input-dependent dynamic gating
- **Integrate** with 3D medical segmentation models
- **Deploy** as a web-based inference tool

---

## 📌 Conclusion

BioSAM demonstrates that **learning how to combine models is often more powerful than choosing between them**.

Our adaptive alpha fusion mechanism:
- ✅ Improves Dice score across all tested datasets
- ✅ Maintains training efficiency (only 12 parameters trained)
- ✅ Generalizes across diverse medical imaging datasets
- ✅ Provides interpretable, layer-wise fusion behavior

---

## 👨‍💻 Author

**Daksh Singhal**  
Department of Computer Science & Engineering  
Indian Institute of Technology (IIT) Jammu
