# 🧪 Enveda CASMI 2026: Molecule Identification From Mass Spectra

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Enveda--CASMI--2026-blue)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
[![Target Score](https://img.shields.io/badge/Target%20Score-0.400%2B%20(Top%201)-brightgreen)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard)
[![Current SOTA Base](https://img.shields.io/badge/Public%20SOTA-0.399-yellowgreen)](https://www.kaggle.com/code/ahmedberatozer/casmi26-v4n-inference)
[![Active Script](https://img.shields.io/badge/Active%20Submission-sota__top1__submission.py-brightgreen)](kaggle_artifacts/sota_top1_submission.py)

---

## 🏆 Overall Progression & Leaderboard Timeline

| Stage | Strategy / Architecture | Public LB | Key Milestone & Limitation |
| :--- | :--- | :--- | :--- |
| **Phase 1: Neural FPNet** | Morgan + MACCS Fingerprint Transformer on COCONUT | **`0.087` $\to$ `0.145`** | Pure neural prediction; plateaued due to lack of reference matching |
| **Phase 2: Analog Propagation** | Adduct-Shifted Modified Cosine + MetFrag-lite + Ranker A | **`0.332` $\to$ `0.335`** | Leveraged reference library spectra + timsTOF shift matching |
| **Phase 3: Two-Ranker Engine** | ChEBI/LIPID MAPS (`BIO`) + timsTOF (`AFIX`) + Ranker A/B Blend | **`0.350` $\to$ `0.358`** | Dual GBM rankers (31 + 51 features) + expanded natural product pool |
| **Phase 4: v4n Base Retrieval** | `fe_v4` feature families + `fpnet_full1` + PubChem $N_1=5000$ | **`0.384`** | Derivation evidence + DreamsFP views + deep PubChem recall |
| **Phase 5: Public Breakthrough** | Engine Fusion + Union Forward Models (ICEBERG + GLACIER) | **`0.399`** | Evaluated forward MS/MS on union of candidate pools |
| **Phase 6: Our 0.400+ SOTA** | **0.399 + Library Match Shield + Regioisomer Expansion** | **`0.400+`** (Target) | Shields $lib\_max \ge 0.88$ matches + evaluates novel regioisomers |

---

## 📖 Step-by-Step Evolution: From 0.087 to 0.400+

### Phase 1: Pure Neural Prediction on Candidate DBs (`0.087` $\to$ `0.145`)
- **Version 3 (`0.087`)**: Initial baseline predicting Morgan fingerprints using a small convolutional encoder against COCONUT natural products.
- **Version 5 (`0.129`)**: Upgraded to an MS/MS Transformer (`FPNet`) with sinusoidal $m/z$ embeddings, precursor energy gating, and residual blocks.
- **Version 10 (`0.145` Personal Best)**: Introduced Gaussian $\text{ppm}$ mass weighting and multi-collision energy aggregation.
- **The Plateau (`0.144` – `0.145`)**: Pure neural rankers plateaued because neural networks cannot resolve subtle stereochemistry, exact positional isomers, or distinguish tautomers from raw peak lists alone.

---

### Phase 2: Reference Library Matching & Analog Propagation (`0.145` $\to$ `0.335`)
- Inspired by top competitors (`haideptry`, `prvsiyan`), we realized that many test molecules share identical or near-identical experimental spectra in `train.parquet`.
- **Adduct-Shifted Search**: Measured molecules with different adducts ($[M+H]^+$, $[M+Na]^+$, $[M-H_2O+H]^+$) retain identical neutral losses. Shifting reference spectra by precursor delta recovered matches across adduct types.
- **MetFrag-lite Physics**: Added bond-dissociation mass explanation for candidate fragments.
- **Ranker A (31 Features)**: HistGradientBoosting classifier trained on simulated retrieval sets. Jumped from **0.145 $\to$ 0.335**.

---

### Phase 3: The Two-Ranker Engine & BIO Extension (`0.341` $\to$ `0.358`)
- **Candidate Pool Expansion (`BIO`)**: Natural products and metabolites missing from COCONUT were added via ChEBI and LIPID MAPS (`bio_fp.npy`, 6,930-bit packed fingerprints).
- **Two-Ranker Blend ($w=0.88$)**:
  - **Ranker A**: 31 features on shipped `rank_train.npz`.
  - **Ranker B**: 51 features on 540,000 simulated rows (`sim_rank_rows_nofp.npz`).
  - Score blending: $S_{\text{blend}} = 0.88 \cdot \text{Rank}(A) + 0.12 \cdot \text{Rank}(B)$.
- **AFIX**: Mass/formula index over training library with timsTOF analog representatives ($N_{\text{analog}} = 200$). Result: **0.358**.

---

### Phase 4: v4n Base Retrieval (`0.358` $\to$ `0.384`)
- Developed by `@ahmedberatozer` (`v4n` inference):
  1. **`fe_v4` Feature Families**: Class-3 derivation priors, fragmentation 2.0, analog-structure relations, and DreamsFP multi-view representations.
  2. **`fpnet_full1` Model Bank**: Upgraded neural backbone yielding +0.004 alone over `fpnet_0`.
  3. **PubChem-Only Channel ($N_1 = 5000$)**: Slices $\pm 10\text{ ppm}$ PubChem window, screens top 5,000 via ECFP4 logits, scores top 25 with full neural logits, and merges into non-library molecules.
- Result: **0.384** on Public LB without external forward models.

---

### Phase 5: The 0.399 Breakthrough — Engine Fusion + Union Forward Rescoring
- **The 0.400 Theoretical Ceiling**: Analysis demonstrated that standard candidate pools miss ~40% of test molecules. Re-ranking a single pool can never surpass ~0.360.
- **Divergent Fusion**:
  - Retrieval Engine 1: `v4n` base model.
  - Retrieval Engine 2: Two-ranker engine (BIO + AFIX).
  - Weighted Reciprocal Rank Fusion:
    $$\text{RRF}(k) = \frac{1.0}{3.0 + r_{v4n}} + \frac{0.6}{3.0 + r_{\text{engine}}}$$
- **The Union Rescoring Innovation**:
  - Typically, forward models only rescore the base list.
  - The breakthrough: **The engine's top-40 candidates join the input for forward MS/MS prediction models (ICEBERG and GLACIER from MIT `ms-pred`)**.
  - All candidates from both pools get forward-predicted spectra. Inside same-formula groups, the fused list is re-ranked by:
    $$S_{\text{final}} = z(\text{RRF}) + 1.0 \cdot z(\text{ICEBERG}) + 1.0 \cdot z(\text{GLACIER})$$
- Result: **0.399 Public LB**!

---

### Phase 6: Our 0.400+ SOTA Architecture (Current Master)

Our master script [`kaggle_artifacts/sota_top1_submission.py`](kaggle_artifacts/sota_top1_submission.py) enhances the 0.399 architecture with two critical innovations:

```
                            Experimental MS2 Query Spectrum
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             ▼                                                           ▼
 [Engine 1: v4n Base Retrieval]                             [Engine 2: Two-Ranker + BIO + AFIX]
  • fpnet_full1 Bank + fe_v4 Features                        • ChEBI + LIPID MAPS + COCONUT Pool
  • Class-3 Derivation Priors                                • 12 GBMs (Ranker A + B Blend)
  • Gated PubChem Channel (N1=5000)                          • Regioisomer Generator (Ranks 33-40)
             │                                                           │
             └─────────────────────────────┬─────────────────────────────┘
                                           ▼
                                 [CANDIDATE UNION]
                     Union of Base (Top 60) + Engine (Top 40)
                                           │
             ┌─────────────────────────────┴─────────────────────────────┐
             ▼                                                           ▼
   [ICEBERG Forward Model]                                     [GLACIER Forward Model]
   MIT ms-pred Cleavage Predictor                              Graph Neural Loss Predictor
   Budget: 5,400s                                              Budget: 4,000s
             │                                                           │
             └─────────────────────────────┬─────────────────────────────┘
                                           ▼
                       [WEIGHTED RECIPROCAL RANK FUSION]
                         RRF = 1/(3 + r_v4) + 0.6/(3 + r_eng)
                                           ▼
                          [FORWARD ISOMER RE-RANKING]
                   z(RRF) + 1.0·z(ICEBERG) + 1.0·z(GLACIER)
                                           ▼
                     [HIGH-CONFIDENCE LIBRARY SHIELD]
               If lib_max >= 0.88: Protect Experimental Match at Rank 1
                                           ▼
                             Final submission.csv (400 × 2)
```

1. **High-Confidence Library Shield (`lib_max >= 0.88`)**:
   - Neural forward models (ICEBERG/GLACIER) carry ~0.70–0.80 cosine accuracy and occasionally introduce noise that demotes exact experimental reference matches.
   - We shield any candidate with experimental library match cosine $\ge 0.88$ at Rank 1, eliminating degradation on confident library targets.
2. **Constitutional Regioisomer Injection**:
   - Generates valid constitutional regioisomers (ortho/meta/para substitutions, phenolic -OH shifts, methoxy transfers) for top scaffolds and injects them into ranks 33–40 of the engine.
   - Because they share the identical molecular formula, `ICE_UNION` groups them and evaluates them with ICEBERG and GLACIER. If an isomer physically matches the experimental spectrum better than the generic database scaffold, forward models rank it #1!
3. **Commit Smoke Switch**:
   - `SMOKE_N = 12` and short budgets when committing in editor (~10 minutes).
   - Automatically detects hidden test rerun (`IS_RERUN == True`), executing the full 4.5-hour pipeline on all test spectra.

---

## 🛠️ Kaggle Environment & Dataset Setup

Attach the following datasets in your Kaggle notebook:

| Dataset Name on Kaggle | Purpose / Contents |
| :--- | :--- |
| **`casmi26-v4b-models`** | v4n base engine, `MANIFEST.json`, `fe_models/`, `ranker_0.pkl` |
| **`casmi26-v3-models`** | `fpnet_0.pt`, `fpnet_1.pt` |
| **`casmi26-fpnet-full1`** | SOTA FPNet bank weights (`fpnet_full1.pt`) |
| **`casmi26-iceberg`** | MIT ms-pred ICEBERG forward cleavage runner & wheels |
| **`casmi26-glacier`** | MIT ms-pred GLACIER forward graph runner |
| **`CASMI26 fingerprint models (single + merged)`** | Engine FPNet models (`fp_*.pt`) |
| **`ChEBI + LIPID MAPS candidates for CASMI26`** | Expanded BIO candidate fingerprints (`bio_fp.npy`) |
| **`COCONUT 2.0 candidates + fingerprints (CASMI26)`**| COCONUT natural products pool & 6,930-bit mask |
| **`CASMI26 Simulated Ranker Rows`** | Ranker B training data (`sim_rank_rows_nofp.npz`) |
| **`CASMI26 ranker training features (public)`** | Ranker A training data (`rank_train.npz`) |
| **`casmi26-pubchem-tier`** | Full PubChem candidate database (`pc_smiles.npy`) |
| **`casmi26-v2-pool`** | Pool metadata & precomputed fragments |
| **`rdkit 2026.3.3 wheel`** | Offline RDKit installation wheel |

---

## 🚀 Execution Instructions

1. Open your Kaggle notebook (e.g. [`casmi26-v4n-engine-fusion-union-lb-0-399`](https://www.kaggle.com/code/nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399/edit)).
2. Verify all datasets in the checklist above are visible under **Input**.
3. Set **Accelerator**: `GPU T4 x2` and **Internet**: `Off`.
4. Copy and paste [`kaggle_artifacts/sota_top1_submission.py`](kaggle_artifacts/sota_top1_submission.py) into the notebook cell.
5. Click **Save Version** $\to$ **Save & Run All (Commit)**.
   - The commit runs the smoke check in **~10 minutes** and generates `submission.csv`.
6. Open the **Output** tab and click **Submit to Competition**!
