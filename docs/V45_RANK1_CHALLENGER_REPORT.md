# V45 Learned Rank-1 Challenger Report

- **OOF Gate Threshold**: `delta_min = 0.0` (lib lock `>= 0.85`)

## Out-of-Sample Holdout Results (Zero Neural/Feature/Ref Leakage)

| Holdout Cohort | N | Incumbent Neural MRR@25 | Full Rerank MRR@25 | Gated Challenger MRR@25 | Correct R1 Promotions | Incorrect R1 Demotions | Gated 95% CI vs Incumbent |
|---|---:|---:|---:|---:|---:|---:|---|
| `np_examples_purged` (Regime B/C) | 250 | 0.2871 | **0.4495** | **0.4495** | 48 | 10 | [+0.1203, +0.2042] (p_win=1.000) |
| `np_examples_purged` (Mixed 50A/50B) | 250 | 0.2871 | **0.5569** | **0.5569** | 78 | 7 | [+0.2228, +0.3180] (p_win=1.000) |
| `timstof_holdout` (Regime B) | 1000 | 0.5740 | **0.8149** | **0.8147** | 323 | 25 | [+0.2196, +0.2631] (p_win=1.000) |
