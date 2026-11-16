# ⚖️ MODEL_COMPARISON.md
# Systematic Benchmark Comparison of Model Architectures & Checkpoints
**Project**: Enveda CASMI 2026 — Molecule Identification From Mass Spectra  
**Document**: Model Benchmark Ledger & Comparative Analysis  

---

## 1. Comprehensive Model Benchmark Matrix

All models have been evaluated on our official 5-fold molecule-grouped validation split (`val_mrr_at_25`, `recall@1`, `recall@5`), measured on an NVIDIA GeForce RTX 3060 (12GB VRAM).

| Model Checkpoint | Architecture & Paradigm | Latent Dim / Outputs | Val MRR@25 | Recall@1 | Recall@5 | Inference Latency (ms/query) | VRAM Footprint | Best Use Case |
|---|---|:---:|:---:|:---:|:---:|:---:|:---:|---|
| **`FPNet v1`** | MS/MS Peak Transformer (6 layers, sinusoidal $m/z$) | 2,048 (Morgan bits) | `0.1450` | `0.098` | `0.210` | 18 ms | 1.1 GB | Baseline candidate retrieval |
| **`FPNet v2`** | Multi-Head Self-Attention + Precursor Energy Gating | 4,096 Morgan + 166 MACCS | `0.1820` | `0.124` | `0.265` | 32 ms | 1.8 GB | High-capacity fingerprint prior |
| **`Saner v1`** | Contrastive Bi-Encoder (MS/MS $\leftrightarrow$ Molecule) | 512-dim normalized | `0.4410` | `0.352` | `0.580` | 14 ms | 0.8 GB | Fast candidate pool filtering |
| **`Scorer v2`** | Deep Siamese Cross-Attention Scorer | 320-dim feature vector | `0.4780` | `0.389` | `0.634` | 45 ms | 1.4 GB | Core neural ranking backbone |
| **`Custom Scorer v1`** | 25-Epoch Full Refit on 2.43M Spectra (Local GPU) | 320-dim contrastive | **`0.6874`** | **`0.592`** | **`0.812`** | 42 ms | 1.4 GB | Superior domain-adapted scoring |
| **`ICEBERG`** | Forward Graph Neural Network (predicts peaks from SMILES) | Peak intensity vector | `0.3880` | `0.290` | `0.520` | 85 ms | 2.2 GB | Forward peak coverage rescoring |
| **`GLACIER`** | Neural Mass Fragmentation Tree Predictor | Tree loss distribution | `0.3950` | `0.301` | `0.535` | 110 ms | 2.6 GB | Neutral loss validation |
| **`Learned Logistic Reranker`**| 24-Feature Multimodal Ridge / Logistic Regressor | Scalar probability | `0.4880` | `0.405` | `0.655` | 3 ms | CPU | Multi-evidence ensemble fusion |
| **`LightGBM LambdaRank`** | Gradient Boosted Decision Tree (Pairwise Ranking) | Ranking score | **`0.4950`** | **`0.418`** | **`0.670`** | 5 ms | CPU | Optimal non-linear feature gating |

---

## 2. In-Depth Architectural Analysis

### 2.1 Inverse Models (Spectrum $\to$ Fingerprint / Structure)
- **`FPNet` Series**: Predicts molecular fingerprints from tandem mass spectra.
  - *Mechanism*: Treats MS/MS peaks as an unordered sequence of tokens $(m/z, I)$, projects $m/z$ with sinusoidal positional embeddings, and applies transformer self-attention.
  - *Finding*: While effective for retrieving candidate classes, pure fingerprint prediction hits a performance ceiling at MRR ~0.18 because cosine Tanimoto similarity over 2048 bits cannot reliably distinguish positional isomers with identical heavy-atom skeletons.
- **`Saner v1` vs `Scorer v2`**:
  - `Saner v1` maps both spectra and molecules into a shared 512-dimensional Euclidean sphere. Rapid to evaluate via matrix multiplication against 50,000 candidates.
  - `Scorer v2` utilizes a deep cross-attention layer between peak features and candidate atomic graphs, capturing specific fragment-to-substructure correspondences.
  - *Ensemble Synergy*: Standardized z-score combination $0.50 \cdot z(\text{Saner}) + 0.50 \cdot z(\text{Scorer v2})$ yields $+0.008$ over either model alone.

### 2.2 Forward Models (Candidate Molecule $\to$ Spectrum)
- **`ICEBERG` and `GLACIER`**:
  - Predict the expected MS/MS fragmentation spectrum of a candidate molecule and compare it against the experimental query spectrum.
  - *Advantage*: Forward models provide explicit physical explainability: if a candidate molecule cannot produce the observed base peak ($100\%$ relative intensity), its rank is penalized.
  - *Limitation*: Heavy computational overhead (~100 ms per candidate). Therefore, forward models are only applied to the Top 30 candidates retrieved by the inverse models.

### 2.3 Custom Scorer v1: Local GPU Refit Breakthrough
- Trained for 25 epochs on our local NVIDIA RTX 3060 (12GB VRAM) across the complete 2,431,940 spectrum dataset.
- Achieved **`0.6874` validation MRR@25** on the held-out molecule-grouped validation fold.
- Represents the highest-fidelity neural embedding model in the project repository. Integrated into Version 31 and planned for full tri-neural consensus in Version 32.
