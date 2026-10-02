# 🗺️ CASMI 2026: Complete Journey & Progress Tracker
**Competition**: [Enveda CASMI 2026 — Molecule Identification From Mass Spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)  
**Primary Metric**: Mean Reciprocal Rank @ 25 (MRR@25) on Tautomer-Canonical InChIKey14  
**Current Milestone**: **`0.409` Public Leaderboard (Top 1 SOTA)**  
**Active Script**: [`kaggle_artifacts/sota_top1_submission.py`](../kaggle_artifacts/sota_top1_submission.py)

---

## 🏆 Overall Progression Timeline

```
[0.087] Pure CNN Morgan Fingerprint
   │
   ▼
[0.145] MS/MS Transformer (FPNet) + Peak Gating + Multi-CE Aggregation
   │
   ▼
[0.335] Analog Propagation + Library Modified Cosine + Ranker A (31 Feats)
   │
   ▼
[0.358] Two-Ranker Engine (BIO ChEBI/LIPID MAPS + AFIX timsTOF + 51 Feats)
   │
   ▼
[0.384] v4n Retrieval (fe_v4 Derivation Priors + fpnet_full1 + PubChem N1=5000)
   │
   ▼
[0.399] Engine Fusion + Candidate Union Forward Models (ICEBERG + GLACIER)
   │
   ▼
[0.404] PubChem Popularity Prior (mu=0.15) + Gap Fragment Rescoring
   │
   ▼
[0.409] SOTA Master: Stronger Prior (POP_MU=0.25) + Library Forward Gate
```

---

## 📊 Milestone Summary Table

| Phase | Architecture / Strategy | Public LB | Key Mechanism & Contribution | Primary Limitation Discovered |
| :---: | :--- | :---: | :--- | :--- |
| **1** | Pure Neural Fingerprint Predictor | **`0.087` $\to$ `0.145`** | MS/MS Transformer predicting Morgan + MACCS bits against COCONUT DB. | Cannot differentiate subtle stereoisomers or tautomers from peak lists alone. |
| **2** | Reference Library Matching | **`0.145` $\to$ `0.335`** | Adduct-shifted modified cosine search over reference library + MetFrag-lite + Ranker A. | Candidate pool capped at 499k COCONUT structures; missed non-plant natural products. |
| **3** | Two-Ranker Engine (`BIO` + `AFIX`) | **`0.350` $\to$ `0.358`** | ChEBI + LIPID MAPS (`bio_fp`) + timsTOF analog indexing + Ranker A/B blend ($w=0.88$). | Single retrieval pool missed novel chemotypes and synthetic drugs. |
| **4** | `v4n` Base Retrieval | **`0.358` $\to$ `0.384`** | `fe_v4` feature families (derivation priors, DreamsFP) + `fpnet_full1` + PubChem tier ($N_1=5000$). | Ranking relies heavily on statistical features; cannot forward-simulate spectra. |
| **5** | Engine Fusion + Union Forward Rescoring | **`0.384` $\to$ `0.399`** | Weighted Reciprocal Rank Fusion ($K=3, \alpha=0.6$) + Forward MS/MS (ICEBERG + GLACIER) on Candidate Union. | Candidate list contains correct molecular formulas but wrong constitutional isomers. |
| **6** | Popularity Prior + Gap Fragment Rescoring | **`0.399` $\to$ `0.404`** | PubChem substance/PubMed citation prior ($\mu=0.15$) + MetFrag bond-dissociation credit on gap adducts. | Forward models occasionally demoted exact experimental library matches. |
| **7** | SOTA Master (`v4b-libgate-pop025`) | **`0.404` $\to$ `0.409`** | Stronger popularity prior ($\mu=0.25$) + Library Gate (`ICE_LAM_LIB=0, GL_LAM_LIB=0`). | Answers outside both candidate pools and PubChem remain unreached. |

---

## 🔬 Deep Dive: Step-by-Step Evolution

### Phase 1: Pure Neural Prediction on Candidate DBs (`0.087` $\to$ `0.145`)
* **Hypothesis**: Can a deep neural network map MS/MS spectra directly into Morgan fingerprints and rank a database of candidate SMILES by Tanimoto similarity?
* **Architecture**:
  - Precursor mass sinusoidal embeddings + dynamic collision energy scaling.
  - Multi-head self-attention over precursor-normalized $(m/z, \text{intensity})$ pairs.
  - Linear projection to 4,096-bit Morgan fingerprints + 166-bit MACCS keys.
* **Findings**:
  - Progressed from 0.087 to 0.145 via Gaussian mass weighting ($\pm 10\text{ ppm}$) and multi-CE pooling.
  - **The Plateau**: Neural predictions lack peak-specific structural resolution. Positional isomers (ortho/meta/para substitutions) yield nearly identical predicted fingerprints.

---

### Phase 2: Reference Library Matching & Analog Propagation (`0.145` $\to$ `0.335`)
* **Hypothesis**: The competition dataset (`train.parquet`) contains 2.5M spectra of 275k compounds. Many test molecules are either already in the library or share structural cores with reference compounds.
* **Architecture**:
  - **Adduct-Shifted Search**: Query spectra shifted by $\Delta = \text{Precursor}_{\text{query}} - \text{Precursor}_{\text{lib}}$ to recover matches across adduct ionization states.
  - **MetFrag-lite Physics**: Untrained physical model calculating what fraction of peak intensity can be explained by single/double bond cleavages.
  - **Ranker A**: 31 features fed to a `HistGradientBoostingClassifier`.
* **Findings**: Public LB surged from **0.145 $\to$ 0.335**, proving that experimental reference matching is orders of magnitude more reliable than pure neural prediction.

---

### Phase 3: The Two-Ranker Engine & BIO Extension (`0.341` $\to$ `0.358`)
* **Hypothesis**: Natural products and metabolites from human/bacterial/fungal origins are absent from standard plant databases (COCONUT). Expanding the candidate pool with ChEBI and LIPID MAPS will improve candidate recall.
* **Architecture**:
  - `bio_fp.npy`: Injected 62,744 non-redundant structures from ChEBI and LIPID MAPS into the candidate pool (total: 774,943 structures).
  - `AFIX`: Prioritized timsTOF instrument representatives in analog matching.
  - **Dual Ranker Blend**:
    $$S_{\text{blend}} = 0.88 \cdot \text{Ranker}_A(31\text{ feats}) + 0.12 \cdot \text{Ranker}_B(51\text{ feats})$$
* **Findings**: Public LB reached **0.358**.

---

### Phase 4: `v4n` Base Retrieval (`0.358` $\to$ `0.384`)
* **Hypothesis**: Single-pool ranking fails when the target structure is novel or absent. A dedicated PubChem channel searching 105M structures can retrieve targets outside the primary pool.
* **Architecture**:
  - `fe_v4` Families: Derivation priors, fragmentation 2.0, analog-structure relations, and DreamsFP multi-view representations.
  - `fpnet_full1` Model Bank: Enhanced neural backbone.
  - PubChem-Only Channel ($N_1 = 5,000$): Slices $\pm 10\text{ ppm}$ PubChem window, screens top 5,000 candidates via ECFP4 logits, scores top 25 with full neural logits, and merges into non-library molecules via an aggressive/gentle gating mechanism.
* **Findings**: Standalone retrieval score hit **0.384** without any forward simulation.

---

### Phase 5: The 0.399 Breakthrough — Engine Fusion + Union Forward Rescoring
* **Hypothesis**: `v4n` and the Two-Ranker Engine explore different chemical regions. Fusing their candidate pools and evaluating their **union** with forward neural spectrum simulators will resolve same-formula isomer ambiguity.
* **Architecture**:
  - **Candidate Union**: Top-60 from `v4n` $\cup$ Top-40 from Engine 2.
  - **Forward Models**:
    - **ICEBERG**: Combinatorial bond-cleavage forward spectrum predictor (budget: 5,400s).
    - **GLACIER**: Graph neural network forward fragmentation loss predictor (budget: 4,000s).
  - **Weighted Reciprocal Rank Fusion**:
    $$\text{RRF}(k) = \frac{1.0}{3.0 + r_{v4n}} + \frac{0.6}{3.0 + r_{\text{engine}}}$$
  - **Forward Rescoring**: Re-ranks inside same-formula groups by $z(\text{RRF}) + 1.0 \cdot z(\text{ICE}) + 1.0 \cdot z(\text{GL})$.
* **Findings**: Public LB jumped to **0.399** (our first major breakthrough).

---

### Phase 6 & 7: The 0.409 SOTA Master (`v4b-libgate-pop025`)
* **Hypothesis**: 
  1. Most remaining errors are between constitutional isomers with identical molecular formulas where simulated spectra are indecisive. PubChem substance records and PubMed citation frequency reflect real-world natural products.
  2. For confident library matches ($\text{lib\_max} \ge 0.90$), forward models introduce simulation noise. Disabling them preserves ground truth.
  3. Molecules measured with non-standard adducts or in negative mode get no ICEBERG score. Adding MetFrag-parsimony only in this gap recovers lost signal.
* **Ablation Table**:

```
Step | Change Description                                    | Configuration                          | Public LB
0    | Reference v4n + Engine Fusion + Union                 | Defaults                               | 0.399
1    | Popularity prior + Fragment re-score in gap           | POP_MU=0.15, FRAG_LAM=0.5, MODE='gap'  | 0.404
2    | ICEBERG / GLACIER switched off for library hits        | ICE_LAM_LIB=0.0, GL_LAM_LIB=0.0       | 0.404 (Safety)
3    | Stronger popularity prior                             | POP_MU=0.25                            | 0.409 (SOTA)
```

* **What Worked**:
  - `POP_MU = 0.25`: Prior $f = z(\text{ranker}) + \mu \cdot (\log(1 + \text{SIDs}) + \log(1 + \text{PMIDs}))$ cleanly separates authentic metabolites from obscure synthetic combinations.
  - `ICE_LAM_LIB = 0.0, GL_LAM_LIB = 0.0`: Protected 18 out of 400 exact library hits from degradation.
  - `FRAG_MODE = 'gap', FRAG_LAM = 0.5`: Evaluated bond-cleavage intensity only for uncovered adducts.
* **What Failed (Negative Ablations)**:
  - Applying fragment re-score to *all* molecules: **`0.390`** (diluted forward model predictions).
  - Popularity prior inside the PubChem-only channel: **`0.393` to `0.405`** (hurt precision of novel compounds).

---

## 🗂️ Repository Structure Overview

```
Enveda-CASMI-2026/
├── README.md                                # Project overview, quickstart & dashboard
├── progress/
│   ├── ROADMAP_AND_PROGRESS.md              # [This file] Complete history & ablation details
│   └── FUTURE_EXPERIMENTS_PLAN.md           # Actionable plan to reach 0.450 - 0.500
├── kaggle_artifacts/
│   ├── sota_top1_submission.py              # Active master submission script (0.409 SOTA)
│   ├── kaggle_notebook.py                   # Kaggle notebook mirror
│   └── candidate_db.parquet                 # Local evaluation artifacts
├── experiments/                             # Experiment tracking & legacy validation logs
└── models/                                  # Checkpoints & serialized weights
```
