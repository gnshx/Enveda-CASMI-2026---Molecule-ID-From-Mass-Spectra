# Enveda CASMI 2026

Molecule identification from LC-MS/MS mass spectra — Kaggle Competition 2026.

## Project Overview

**Goal**: Predict the 2D chemical structure (SMILES) of unknown molecules from their MS/MS spectra.  
**Metric**: MRR@25 (Mean Reciprocal Rank at 25)  
**Prize**: $50,000  
**Competition**: [Enveda CASMI 2026](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)

---

## Architecture

```
MS/MS Spectrum(s)
      │
      ▼
 Preprocessing (peak_filter.py)
      │
   ┌──┴───────────────┬───────────────────┐
   ▼                  ▼                   ▼
Cosine             DreaMS            Chemical DB
Retrieval          FAISS             (COCONUT +
(matchms)          Search             PubChem)
   │                  │                   │
   └──────────────────┴───────────────────┘
                       │
                 Candidate Pool
                       │
               Feature Engineering
                (spectral.py +
                 chemical.py)
                       │
                Ensemble Ranker
            (LightGBM + XGBoost + MLP)
                       │
               Multi-Spectrum Aggregation
               (multi_spectrum.py)
                       │
                  Top-25 SMILES
```

---

## Setup

```bash
conda env create -f environment.yml
conda activate casmi2026
```

### Download data
```bash
kaggle competitions download -c enveda-CASMI26-molecule-id-mass-spectra
unzip enveda-CASMI26-molecule-id-mass-spectra.zip -d data/
```

---

## Project Structure

```
Enveda-CASMI-2026/
├── environment.yml             # Conda environment
├── data/                       # Raw data (not committed)
│   ├── train.parquet
│   ├── test.parquet
│   └── references/             # COCONUT, PubChem subsets
├── preprocessing/
│   ├── peak_filter.py          # Peak filtering + deisotoping
│   └── normalize.py            # Intensity transforms
├── retrieval/
│   ├── cosine.py               # matchms cosine search
│   ├── dreams.py               # DreaMS embeddings + FAISS
│   └── faiss_index.py          # FAISS index management
├── chemistry/
│   ├── adducts.py              # Neutral mass calculation
│   ├── formula.py              # Molecular formula utilities
│   ├── fingerprints.py         # Morgan, MACCS, RDKit FPs
│   └── rdkit_utils.py          # RDKit helpers
├── candidates/
│   ├── coconut.py              # COCONUT database lookup
│   ├── pubchem.py              # PubChem mass lookup
│   ├── training_structures.py  # Train SMILES database
│   └── merger.py               # Deduplicate candidate pools
├── features/
│   ├── spectral.py             # Cosine, fragment, neutral loss features
│   └── chemical.py             # Mass error, formula, fingerprint features
├── ranking/
│   ├── train_ranker.py         # LightGBM + XGBoost training
│   ├── neural_ranker.py        # MLP ranker
│   └── ensemble.py             # Ensemble + weight optimization
├── aggregation/
│   └── multi_spectrum.py       # Evidence fusion across spectra
├── validation/
│   ├── splitter.py             # Structure/library/NP splits
│   └── metrics.py              # MRR@25, Recall@K
├── submission/
│   └── build_submission.py     # Format + validate submission.csv
├── notebooks/                  # Jupyter EDA notebooks
└── experiments/                # Ablation logs and results
```

---

## 30-Day Roadmap

| Phase | Days | Target |
|-------|------|--------|
| 0: Foundations | 1–3 | EDA + validation framework |
| 1: Retrieval | 4–9 | MRR@25 ≥ 0.20 |
| 2: Chemistry | 10–15 | Recall@100 ≥ 0.60 |
| 3: Ranking | 16–21 | MRR@25 ≥ 0.40 |
| 4: Multi-spectrum | 22–25 | MRR@25 ≥ 0.50 |
| 5: Polish | 26–30 | Final submission |

---

## Key Insight

> Measure **Recall@100** before MRR@25.  
> If the true molecule is not in your candidate pool, your ranker cannot recover it.  
> **Fix recall first. Then fix ranking.**

---

## Novelty Classes

| Class | Description | Strategy |
|-------|-------------|----------|
| 1 | Molecule has public MS/MS spectrum | Spectral library search |
| 2 | Molecule in databases (no spectrum) | Mass + formula constrained DB search |
| 3 | Truly novel (not in PubChem) | De novo generation + formula guidance |

---

## External Resources

| Resource | URL | License | Use |
|----------|-----|---------|-----|
| COCONUT | coconut.naturalproducts.net | CC BY 4.0 | Candidate DB |
| DreaMS | github.com/pluskal-lab/DreaMS | Check license | Spectrum embeddings |
| matchms | github.com/matchms/matchms | Apache 2.0 | Cosine similarity |
| GNPS | gnps.ucsd.edu | CC BY 4.0 | Reference spectra |
| MIST-CF | github.com/samgoldman97/mist-cf | MIT | Formula prediction |
| RDKit | rdkit.org | BSD | Cheminformatics |
