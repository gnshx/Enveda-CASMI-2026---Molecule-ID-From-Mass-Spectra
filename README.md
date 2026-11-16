# 🧪 Enveda CASMI 2026: Molecule Identification From Mass Spectra

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Enveda--CASMI--2026-blue)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
[![Personal Best](https://img.shields.io/badge/Personal%20Best-0.4800%20(Top%20Contender)-gold)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/submissions)
[![Target Score](https://img.shields.io/badge/Target%20Score-0.487%2B%20(Gold%20Tier)-brightgreen)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard)
[![Active Deployment](https://img.shields.io/badge/Active%20Deployment-Version%2031%20Frontier%20(N--1c%20+%20ExtLib)-brightgreen)](deploy_kernel_v31/casmi26-v4n-engine-fusion-union-lb-0-399.ipynb)
[![Progress Tracker](https://img.shields.io/badge/Roadmap%20%26%20Progress-RESEARCH__ROADMAP.md-blue)](RESEARCH_ROADMAP.md)
[![Daily Journey Diary](https://img.shields.io/badge/Daily%20Journey-progress%2FDAILY__JOURNEY.md-purple)](progress/DAILY_JOURNEY.md)

---

> 🏛️ **Master Autonomous Research Lab Documentation Suite**:
> | Document | Purpose & Description |
> |---|---|
> | 🔬 [COMPETITION_UNDERSTANDING.md](COMPETITION_UNDERSTANDING.md) | Official CASMI 2026 problem definition, MRR@25 math, InChIKey14 rules, 3 Discovery Regimes |
> | 📋 [PROJECT_AUDIT.md](PROJECT_AUDIT.md) | Inventory of data (2.43M spectra), models, code modules, GPU compute & forensic review |
> | 🚦 [PROJECT_STATUS.md](PROJECT_STATUS.md) | Real-time leaderboard standings (Top 1: 0.549, PB: 0.4800, active Ref 58442788) & telemetry |
> | 🗺️ [RESEARCH_ROADMAP.md](RESEARCH_ROADMAP.md) | 4-Milestone Strategic Roadmap to 0.550+ across Regimes A, B, and C |
> | 🧪 [RESEARCH_FINDINGS.md](RESEARCH_FINDINGS.md) | Literature analysis (MIST, SIRIUS, DreaMS, MetFrag), timsTOF physics, CID energies & adducts |
> | 📦 [DATA_PROVENANCE.md](DATA_PROVENANCE.md) | Lineage of 325k ExtLib spectra (MassBank, Takumu Hata), DBs (PubChem, COCONUT) & licenses |
> | ⚖️ [MODEL_COMPARISON.md](MODEL_COMPARISON.md) | Side-by-side benchmark matrix of all inverse, forward & contrastive neural models |
> | 📜 [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | Complete registry of all 31 Kaggle submissions and local validation breakthroughs |
> | 🔬 [ABLATION_RESULTS.md](ABLATION_RESULTS.md) | Isolated component impact table & cumulative progress waterfall (+0.108 gain) |
> | 🛑 [FAILURE_ANALYSIS.md](FAILURE_ANALYSIS.md) | Forensic autopsies of past regressions (V25 LOTUS, V26/V27 demotions, V28/V29 timeouts) |
> | 🧪 [VALIDATION_REPORT.md](VALIDATION_REPORT.md) | Zero-leakage molecule-grouped splits (`structure_split`, `scaffold_split`, `np_split`) |
> | 🚀 [SUBMISSION_READINESS.md](SUBMISSION_READINESS.md) | Automated 7-point pre-flight verification checklist, Python script & sandbox rules |
> | 👑 [FINAL_STRATEGY.md](FINAL_STRATEGY.md) | Master synthesis of the 3-regime sovereign architecture & dual-submission protocol |

---

## 🏆 Overall Progression & Leaderboard Timeline

| Stage | Strategy / Architecture | Public LB | Key Milestone & Ablation |
| :--- | :--- | :--- | :--- |
| **Phase 1: Neural FPNet** | Morgan + MACCS Fingerprint Transformer on COCONUT | **`0.087` $\to$ `0.145`** | Pure neural prediction; plateaued due to lack of reference matching |
| **Phase 2: Analog Propagation** | Adduct-Shifted Modified Cosine + MetFrag-lite + Ranker A | **`0.332` $\to$ `0.335`** | Leveraged reference library spectra + timsTOF shift matching |
| **Phase 3: Two-Ranker Engine** | ChEBI/LIPID MAPS (`BIO`) + timsTOF (`AFIX`) + Ranker A/B Blend | **`0.350` $\to$ `0.358`** | Dual GBM rankers (31 + 51 features) + expanded natural product pool |
| **Phase 4: v4n Base Retrieval** | `fe_v4` feature families + `fpnet_full1` + PubChem $N_1=5000$ | **`0.384`** | Derivation evidence + DreamsFP views + deep PubChem recall |
| **Phase 5: Public Breakthrough** | Engine Fusion + Union Forward Models (ICEBERG + GLACIER) | **`0.399`** | Evaluated forward MS/MS on union of candidate pools |
| **Phase 6: Popularity + Library Gate** | Popularity Prior (`mu=0.15`) + Gap Fragment Rescoring | **`0.404`** | PubChem substance/citation prior on isomers + gap coverage |
| **Phase 7: SOTA 0.409 Master** | Stronger Prior (`POP_MU=0.25`) + Library Gate (`ICE/GL=0`) | **`0.409`** | Optimal isomer separation + shields confident library matches |
| **Phase 8: PubChem Join SOTA** | PubChem Join (`PC_JOIN_N=50`) into Main Engine Ranker | **`0.420`** | Outside answers ranked #1 via 160-feature evaluation |
| **Phase 9: Parallax & Fragments** | Physical Bond-Cleavage Rescoring (5-Shield Gate) | **`0.472`** | MetFrag-lite fragment explanation elevates Class 1/2 structures |
| **Phase 10: Baidalin MIDX Dedup** | In-Pool InChIKey14 Skeletal Deduplication | **`0.479`** | Purges redundant stereoisomers from ranks 2–25; reclaims slots |
| **Phase 11: Dual-Neural Ensemble**| **Dual-Neural Z-Score Standardization + Rank 1 Shield** | **`0.4800` (PB)** | $0.50 \cdot z(\text{Saner}) + 0.50 \cdot z(\text{v2})$ + $w_{\text{top}}=0.00$ Rank 1 lock |
| **Phase 12: Forensic Diagnosis** | Root-Cause Autopsy on V25-V27 Regressions | **`0.472` $\to$ `0.467`** | Identified raw LOTUS decoy noise & $w_{\text{top}}=0.25$ rank flips |
| **Phase 13: Local GPU Refit** | 25-Epoch Custom Scorer Refit on RTX 3060 CUDA | **`0.6874` val MRR** | Full training on 2.43M spectra; published to Kaggle dataset |
| **Phase 14: Apex Sovereign ExtLib** | **Dual-Tier Reference Promotion + timsTOF Mass Prior** | **`0.487+` (Target)** | 324k external spectra + Gold/Silver promotion + timsTOF -1.5ppm |
| **Phase 15: Frontier Sovereign (V31)**| **N-1c In-Test Molecular Network + Calibrated ExtLib + Logistic Reranker** | **`0.488 – 0.492+` (Pending)** | In-test analog propagation ($\lambda=0.35$) + ExtLib $\tau=0.80$ + single-pass timeout guard |

---

## 📖 Step-by-Step Evolution: From 0.087 to 0.409 (Top 1)

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

### Phase 6, 7 & 8: The SOTA Breakthrough (0.409 $\to$ 0.420)

```
Step | Change Description                                    | Configuration                          | Public LB
0    | Reference v4n + Engine Fusion + Union                 | Defaults                               | 0.399
1    | Popularity prior + Fragment re-score in gap           | POP_MU=0.15, FRAG_LAM=0.5, MODE='gap'  | 0.404
2    | ICEBERG / GLACIER switched off for library hits        | ICE_LAM_LIB=0.0, GL_LAM_LIB=0.0       | 0.404 (Safety)
3    | Stronger popularity prior                             | POP_MU=0.25                            | 0.409
4    | PubChem Join (Main Engine Ranking)                    | PC_JOIN_N=50, PC_JOIN_MAX_LIB=0.7     | 0.420 (CURRENT SOTA)
```

#### Why Each Component Works:
1. **PubChem Join (`PC_JOIN_N = 50, PC_JOIN_MAX_LIB = 0.7`) — The 0.420 Breakthrough**:
   - The primary candidate pool has ~710,000 structures (training set + COCONUT).
   - In previous iterations, outside answers from PubChem were artificially confined to fixed slots (4, 8, 12, ...), completely barring any outside candidate from reaching **Rank 1**.
   - `pc_join.py` injects the top 50 PubChem-only candidates directly into the main `V1FE` engine.
   - Candidates are enriched with all 160 features (analog propagation, forward simulation, derivation priors) and ranked directly by the LightGBM ranker.
   - Molecules with confident library matches (`lib_max >= 0.7`) are protected with a gate to prevent decoy displacement.
   - **Result**: Leaps Public LB by **+0.011 to 0.420**!
2. **Popularity Prior (`POP_MU = 0.25`)**:
   - Differentiates isomers of the same formula using PubChem substance (SID) and literature (PMID) counts:
     $$f = z(\text{ranker}) + \mu \cdot \left(\log(1 + \text{substances}) + \log(1 + \text{PubMed})\right)$$
   - Only applied to candidate pool structures (`pid >= 0`).
3. **Library Gate for Forward Models (`ICE_LAM_LIB = 0.0, GL_LAM_LIB = 0.0`)**:
   - Protects verified experimental reference matches ($\text{lib\_max} \ge 0.90$) from simulated spectrum distortion.
4. **Fragment Re-Score in the Gap (`FRAG_MODE = 'gap'`, `FRAG_LAM = 0.5`)**:
   - Evaluates MetFrag bond-dissociation peak credit exclusively for gap adducts that ICEBERG/GLACIER cannot score.

---

## 🛠️ Kaggle Environment & Dataset Setup

Attach the following datasets in your Kaggle notebook:

| Dataset Name on Kaggle | Purpose / Contents |
| :--- | :--- |
| **`casmi26-pubchem-popularity-prior`** | **PubChem Substance & PubMed priors (`pool_lsid.npy`, `pool_lpmid.npy`)** |
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

## 🚀 Execution Instructions for Version 7 (0.420)

1. Open your Kaggle notebook (e.g. [`casmi26-v4n-engine-fusion-union-lb-0-399`](https://www.kaggle.com/code/nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399/edit)).
2. Under **Input** $\to$ **Add Input**, ensure all 14 required datasets are attached (especially `casmi26-pubchem-popularity-prior`).
3. Set **Accelerator**: `GPU T4 x2` and **Internet**: `Off`.
4. Copy and paste the complete contents of [`kaggle_artifacts/sota_top1_submission.py`](kaggle_artifacts/sota_top1_submission.py) (or [`kaggle_artifacts/kaggle_notebook.py`](kaggle_artifacts/kaggle_notebook.py)) into the notebook code cell.
5. Click **Save Version** $\to$ **Save & Run All (Commit)**.
   - The commit runs the smoke check on 12 molecules in **~15–20 minutes** and produces `submission.csv`.
   - Verify logs print: `PubChem join: N 50 order first max lib 0.7` and `popularity prior loaded`.
6. Go to the notebook version page $\to$ **Output** tab $\to$ click **Submit to Competition**!
7. The scoring rerun automatically processes the hidden test set to achieve **0.420 Public LB**!
