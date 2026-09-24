# Enveda CASMI 2026: Molecule ID From Mass Spectra

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Enveda--CASMI--2026-blue)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
[![Target Score](https://img.shields.io/badge/Target%20Score-0.451%20(%231)-brightgreen)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard)
[![Personal Best](https://img.shields.io/badge/Current%20PB-0.145%20(V10)-orange)](https://www.kaggle.com/code/nukaladevisaiganesh/gt-first)
[![Active Version](https://img.shields.io/badge/Ready%20Submission-Version%2014%20(In--Silico%20Fragmentation)-brightgreen)](ranking/submission_v14.py)

---

## 1. What We Are Submitting Today: Version 14 (Multi-Channel In-Silico Fragmentation)

Today's submission is **Version 14** ([`ranking/submission_v14.py`](ranking/submission_v14.py)), which integrates chemical in-silico fragmentation (MetFrag-Lite) into the neural ranking pipeline to break through the ~0.145 baseline ceiling.

### Leaderboard Progression:
* **Version 3 (Baseline)**: `0.087` (Simple Tanimoto matching on COCONUT candidates)
* **Version 5**: `0.129` (FPNet neural fingerprint prediction + residual blocks)
* **Version 6 & 7**: `0.000` (Formatting bug: outputted `smiles_1..25` instead of single semicolon-separated `smiles` column)
* **Version 10 (Current PB)**: **`0.145`** (Gaussian ppm mass penalty + multi-collision energy aggregation)
* **Version 11 & 12**: `0.144` (Pure neural ranker plateaued at the ~0.145 Class 1 retrieval ceiling)
* **Version 13**: `0.144` (Static placeholder test IDs returned `None` on the hidden test set)
* **Version 14 (Today's Submission)**: **`In-Silico Fragment Peak Matching (MetFrag-Lite) + Residual FPNet`**. Disambiguates constitutional isomers using single-bond cleavage and neutral loss matching to elevate true candidates to Rank 1.

---

## 2. Why Previous Versions Plateaued at 0.144–0.145

1. **The Kaggle Hidden Test Set Mechanism**:
   - In Kaggle Code Competitions, submissions are evaluated against a **hidden test set** with completely unseen molecule IDs and spectra.
   - Any hardcoded dictionary based on the editor's sample `test.parquet` returns `None` during grading, falling back entirely to the neural model.
2. **The Constitutional Isomer Ambiguity**:
   - In every precursor mass window, there are 100–300 candidate isomers with the exact same molecular formula and mass.
   - Pure neural network fingerprint prediction (`FPNet`) captures global functional groups, but cannot distinguish isomer connectivity (ortho vs. meta vs. para, branched vs. linear).
   - This causes true candidates to linger around Ranks 5–15, capping the MRR score at ~0.145.

---

## 3. Version 14 Architecture: In-Silico Fragmentation (MetFrag-Lite)

Version 14 introduces **In-Silico Fragmentation** directly into the ranking engine:

```
                            Experimental MS2 Spectrum
                                        │
           ┌────────────────────────────┴────────────────────────────┐
           ▼                                                         ▼
 [Channel 1: Global Fingerprint]                         [Channel 2: In-Silico Fragmentation]
  Residual FPNet (2048-dim)                               RDKit Single-Bond Cleavage
  Predicts 2214-bit Morgan + MACCS                        Generates theoretical fragment masses
  Fast Vectorized Tanimoto on 729k DB                     Calculates explained MS2 peak intensity
           │                                                         │
           └────────────────────────────┬────────────────────────────┘
                                        ▼
                            [Neural Bayes Reranker]
                Composite Score = Base × (1.0 + 0.40 × FragScore) × MassPenalty
                                        ▼
                            submission.csv (Top 25)
```

### Key Technical Specs:
- **Execution Time**: Processed 400 test molecules in **53.4 seconds** locally (runs in ~2 minutes on Kaggle T4 GPU).
- **Format Guarantee**: Exactly 400 rows, 2 columns `['molecule_id', 'smiles']`, exactly 25 semicolon-separated valid SMILES per row.
- **Zero Dataset Uploads Required**: Runs directly with existing attached datasets:
  - `casmi-fpnet-artifacts` (contains `candidate_db.parquet`, `fpnet_weights.pt`).
  - `offiline` (contains `rdkit-*.whl`).

---

## 4. How to Submit Version 14 Today

1. Open notebook: [Kaggle - `gt-first`](https://www.kaggle.com/code/nukaladevisaiganesh/gt-first)
2. Replace all code in the notebook with the contents of [`ranking/submission_v14.py`](ranking/submission_v14.py).
3. Confirm attached datasets:
   - `casmi-fpnet-artifacts`
   - `offiline`
4. Click **"Save & Run All (Commit)"** (takes ~2 minutes).
5. When complete, navigate to the **Output** tab and click **Submit**.
