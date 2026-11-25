# V47 & V48 Ablation Report: Formula Consensus, Rare Fragments & CE Weighting

| Variant | OOF MRR@25 | `np_purged` (Regime B/C) | `np_purged` (Mixed 50A/50B) | `timstof_holdout` (Regime B) |
|---|---:|---:|---:|---:|
| `V45_control` | 0.8477 | **0.4495** (R1=0.300) | **0.5569** (R1=0.432) | **0.8147** (R1=0.718) |
| `V47_plus_formula_adduct` | 0.8488 | **0.4500** (R1=0.304) | **0.5451** (R1=0.412) | **0.8137** (R1=0.716) |
| `V48_plus_rare_ce` | 0.8495 | **0.4512** (R1=0.296) | **0.5514** (R1=0.420) | **0.8126** (R1=0.717) |
| `V48_plus_V46_emb` | 0.8505 | **0.4495** (R1=0.292) | **0.5519** (R1=0.424) | **0.8136** (R1=0.717) |
