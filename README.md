# Enveda CASMI 2026: Molecule ID From Mass Spectra

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Enveda--CASMI--2026-blue)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)
[![Target Score](https://img.shields.io/badge/Target%20Score-0.451%20(%231)-brightgreen)](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra/leaderboard)
[![Personal Best](https://img.shields.io/badge/Current%20PB-0.145%20(V10)-orange)](https://www.kaggle.com/code/nukaladevisaiganesh/gt-first)
[![Active Version](https://img.shields.io/badge/Ready%20Submission-Version%2011%20(Hybrid)-purple)](ranking/hybrid_v11.py)

---

## 1. What We Are Doing Right Now

We are engineering a competitive solution to reach and surpass the **#1 Leaderboard Score (0.451)** held by *Ozymandias31415*.

### Leaderboard Progression:
* **Version 3 (Baseline)**: `0.087` (Simple Tanimoto matching on COCONUT candidates)
* **Version 5**: `0.129` (FPNet neural fingerprint prediction + residual blocks)
* **Version 6 & 7**: `0.000` (Formatting bug: outputted `smiles_1..25` instead of single semicolon-separated `smiles` column)
* **Version 10 (Current PB)**: **`0.145`** (Fixed formatting, added Gaussian ppm mass penalty + multi-collision energy aggregation)
* **Version 11 (Ready to Deploy)**: **`Hybrid Ranker`** combining **305 Peak-Verified Library Matches** (Tier 1) with our proven **V10 Neural Engine** (Tiers 2 & 3 fallback).

---

## 2. Competition Rules & Execution Constraints

To guarantee that submissions are never rejected or graded with 0.000, all code adheres to these strict rules:

| Rule | Requirement | Implementation in V11 |
| :--- | :--- | :--- |
| **Internet Access** | **Must be OFF (`Internet: Off`)** | Fully offline. Uses wheels and precomputed weights. |
| **Output File** | **`submission.csv` in `/kaggle/working/`** | Strictly outputted and verified locally. |
| **Column Names** | Exactly two columns: `molecule_id,smiles` | Verified via pandas assertion: `['molecule_id', 'smiles']`. |
| **SMILES Format** | Semicolon-delimited (`smi1;smi2;...;smi25`) | Exactly 25 unique, valid SMILES per row. |
| **Row Count** | Exactly 400 rows matching `test.parquet` | Verified 1-to-1 match against test `molecule_id` list. |
| **Execution Mode** | Must use **"Save & Run All (Commit)"** | Avoid "Quick Save" (Quick Save does not run the model or create `submission.csv`). |
| **Runtime Limits** | Max 9 hours GPU / 16 GB RAM | V11 executes in **~3 minutes** on CPU, using < 2 GB RAM. |

---

## 3. Evaluation Metric & Leaderboard Dynamics

Submissions are evaluated on **Mean Reciprocal Rank (MRR)** for the top 25 candidates:

$$\text{MRR} = \frac{1}{N} \sum_{u=1}^N \frac{1}{\text{Rank}_u}$$

* **Rank 1**: **1.000** point
* **Rank 2**: **0.500** points (50% drop if pushed from 1st to 2nd!)
* **Rank 25**: 0.040 points
* **Rank > 25**: 0.000 points

> [!CRITICAL]
> Because Rank 1 awards 1.0 point while Rank 2 awards only 0.5 points, **a false guess at Rank 1 halves the score of a correctly predicted molecule**.
> Version 11 only injects library matches when **$\ge 8$ identical MS/MS fragment peaks** are confirmed against `train.parquet`. For all other molecules, the ranking is decided purely by the neural engine.

---

## 4. Test Set Composition: The 3-Tier Discovery

Through deep spectral peak matching across all row groups of `train.parquet` (2.5 million spectra across ~275,000 unique compounds), we mapped the exact distribution of the 400 test molecules:

```
Test Set (400 molecules)
 ├── Class 1: Public Reference Matches (305 molecules, 76.25%)
 │    └── Confirmed with >= 8 identical MS/MS fragments in train.parquet
 │    └── Handled via in-memory Tier 1 base64 lookup -> Injected at Rank 1 (1.000 pts)
 ├── Class 2: Database Knowns (PubChem / COCONUT / ChEBI, ~60 molecules)
 │    └── Known natural products without reference spectra
 │    └── Handled via FPNet neural fingerprint prediction + Gaussian ppm mass penalty
 └── Class 3: Novel / Uncatalogued Analogs (~35 molecules)
      └── Handled via neural fallback & analog candidate pools
```

---

## 5. Active Pipeline: Version 11 Architecture

The standalone deployment script is [`ranking/hybrid_v11.py`](ranking/hybrid_v11.py):

```
       Input: test.parquet (400 molecules, ~1500 spectra)
                             │
                             ▼
              Check Test Molecule ID (mid)
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
       mid in Tier 1?                 mid NOT in Tier 1?
     (>= 8 shared peaks)              (< 8 shared peaks)
              │                             │
              ▼                             ▼
     Inject Reference SMILES          Neural FPNet Inference
           at Rank 1                  + Gaussian ppm Mass Penalty
              │                             │
              ▼                             ▼
     Backfill Ranks 2..25             Populate Ranks 1..25
     with Neural Ranked               with Neural Ranked
     COCONUT Candidates               COCONUT Candidates
              │                             │
              └──────────────┬──────────────┘
                             ▼
            Deduplication & Top-25 Formatting
                             ▼
        Output: submission.csv (400 rows, 2 columns)
```

### Dataset Requirements on Kaggle:
No new dataset uploads are needed for Version 11. It uses:
1. `casmi-fpnet-artifacts` (contains `candidate_db.parquet`, `fpnet_weights.pt`).
2. `offiline` (contains `rdkit-*.whl`).

---

## 6. Project Roadmap to Reach Score 0.451 (#1)

| Step | Milestone | Expected Score | Status |
| :---: | :--- | :---: | :---: |
| **V10** | Verified Kaggle pipeline + Gaussian ppm mass penalty | `0.145` | **Completed** |
| **V11** | Hybrid ranker with 305 verified library matches | `0.30 - 0.40+` | **Ready for Submission** |
| **V12** | Upload 812k Candidate Database (`candidate_db_packed.parquet`) | `+0.05` | Built locally (70 MB) |
| **V13** | Multi-Residual FPNet v2 + Neutral-Loss Encoder Ensemble | `+0.04` | Weights trained (152 MB) |
| **V14** | Analog Precursor Delta-Shift Propagation ($\Delta m$) | **`0.451+` (#1)** | In design |

---

## 7. Submission Instructions for Version 11

1. Open notebook: [Kaggle - `gt-first`](https://www.kaggle.com/code/nukaladevisaiganesh/gt-first)
2. Replace all code in the notebook with [`ranking/hybrid_v11.py`](ranking/hybrid_v11.py).
3. Confirm that attached datasets are:
   - `casmi-fpnet-artifacts`
   - `offiline`
4. Click **"Save & Run All (Commit)"**.
5. When complete, navigate to **Output** and click **Submit**.
