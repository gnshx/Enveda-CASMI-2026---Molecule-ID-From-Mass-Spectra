# V46 Independent timsTOF-Adapted Spectral Similarity Retrieval Report

- **Fold-4 Pairwise Tanimoto RMSE**: `0.088`

| Cohort | N | Binned Cosine Analog MRR@25 | V46 Spectral Embedding MRR@25 | Blend (Binned + V46) MRR@25 | Blend 95% CI vs Binned |
|---|---:|---:|---:|---:|---|
| `np_examples_purged` (Regime B/C) | 250 | 0.4745 | **0.2205** | **0.4035** | [-0.0987, -0.0443] (p_win=0.000) |
| `timstof_holdout` (Regime B) | 1000 | 0.5602 | **0.3502** | **0.5526** | [-0.0222, +0.0074] (p_win=0.170) |
