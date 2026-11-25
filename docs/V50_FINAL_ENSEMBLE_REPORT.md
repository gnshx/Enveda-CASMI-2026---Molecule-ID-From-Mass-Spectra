# V50 Calibrated Ensemble Report

- **Selected Model**: `V50_3Model_Base42` (`weights={'lgb': 0.5, 'cb': 0.5, 'hgb': 0.0, 'delta_thr': 0.0}`)
- **Compact Deploy Payload Size**: `1,662,523` bytes (`models/v50_compact_lgb.json.zlib`)

| Variant | OOF MRR@25 | `np_purged` (Regime B/C) | `np_purged` (Mixed 50A/50B) | `timstof_holdout` (Regime B) |
|---|---:|---:|---:|---:|
| `V50_3Model_Base42` | 0.8478 | **0.4466** (R1=0.296, +47/-10) | **0.5572** (R1=0.432, +78/-7) | **0.8152** (R1=0.719, +324/-25) |
| `V50_3Model_Plus_CE_MultiAdd` | 0.8509 | **0.4460** (R1=0.288, +45/-10) | **0.5453** (R1=0.412, +73/-7) | **0.8135** (R1=0.717, +322/-25) |
