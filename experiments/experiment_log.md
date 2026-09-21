# CASMI 2026 — Experiment Log

Format: Date | Experiment | Val MRR@25 | Recall@25 | Recall@100 | Notes
---

## Phase 1: Spectral Retrieval

| Date | Experiment | MRR@25 | R@1 | R@5 | R@10 | R@25 | R@100 | Class 1 MRR | Class 2 MRR | Retrieval Fail | Ranking Fail | Notes |
|------|------------|--------|-----|-----|------|------|-------|-------------|-------------|----------------|--------------|-------|
| 2026-09-21 | `baseline_v001`: Numba Cosine + ModCosine + Multi-Spec Agg | **0.3454** | 0.3200 | 0.3720 | 0.3800 | **0.3880** | 0.3920 | **0.6908** (R@25: 0.776) | 0.0000 | 60.8% | 0.4% | First reproducible baseline. 962k ref spectra. Evaluated on 250 timsTOF NP mols (1,183 spectra). Confirms ranking is near-perfect when retrieved; bottleneck is retrieval failure! |


## Phase 2: Chemistry Databases

| Date | Experiment | MRR@25 | R@25 | R@100 | Notes |
|------|-----------|--------|------|-------|-------|
| TBD  | + COCONUT candidates    | — | — | — | |
| TBD  | + PubChem NP subset     | — | — | — | |
| TBD  | + training structures   | — | — | — | |
| TBD  | + fingerprint features  | — | — | — | |
| TBD  | + fragment features     | — | — | — | |
| TBD  | + neutral loss features | — | — | — | |

## Phase 3: Learning-to-Rank

| Date | Experiment | MRR@25 | R@25 | R@100 | Notes |
|------|-----------|--------|------|-------|-------|
| TBD  | LightGBM ranker (all features) | — | — | — | |
| TBD  | XGBoost ranker                 | — | — | — | |
| TBD  | MLP ranker                     | — | — | — | |
| TBD  | Ensemble (LGBM+XGB)            | — | — | — | |
| TBD  | Ensemble (LGBM+XGB+MLP)        | — | — | — | |

## Phase 4: Multi-Spectrum

| Date | Experiment | MRR@25 | R@25 | R@100 | Notes |
|------|-----------|--------|------|-------|-------|
| TBD  | max aggregation   | — | — | — | |
| TBD  | mean aggregation  | — | — | — | |
| TBD  | ranker aggregation features | — | — | — | |
| TBD  | CE-stratified features | — | — | — | |

## Kaggle Public Leaderboard History

| Date | Configuration | Public Score | Notes |
|------|--------------|-------------|-------|
| TBD  | Phase 1 baseline | — | First submission |
| TBD  | Phase 2       | — | |
| TBD  | Phase 3       | — | |
| TBD  | Final         | — | |
