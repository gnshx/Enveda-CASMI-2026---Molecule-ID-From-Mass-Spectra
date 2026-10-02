# 🚀 CASMI 2026: Next-Level Experiments Plan (0.409 $\to$ 0.450+)
**Current SOTA**: **`0.409`** ([`kaggle_artifacts/sota_top1_submission.py`](../kaggle_artifacts/sota_top1_submission.py))  
**Objective**: Systematically optimize hyperparameters, candidate recall, and forward model ensembling to push beyond 0.409 toward **`0.450+`** and secure #1 on the leaderboard.

---

## 🎯 Executive Summary of Levers

```
                                    Target: 0.450+
                                          ▲
                  ┌───────────────────────┼───────────────────────┐
                  ▼                       ▼                       ▼
          [HYPERPARAMETER]       [CANDIDATE RECALL]      [FORWARD ENSEMBLING]
          • Fill-25 Backfill     • Regioisomer Gen       • Glacier Weight Boost
          • Popularity Tuning    • PubChem Expansion     • Adduct Neutral Losses
          • Stage-C RRF Tune     • ChEBI 2026 Update     • Loss Calibrator
```

---

## 📋 Prioritized Experiment Roadmap

### Experiment 1: The "Zero-Fallback" Backfill (`FILL_25 = True`)
* **Priority**: 🔴 **Highest (Immediate, Zero Risk)**
* **Estimated Gain**: **`+0.003 to +0.006`**
* **The Problem**:
  - In `fusion_core.py`, when a molecule has fewer than 25 valid candidate structures, remaining positions are filled with dummy fallback `'CCO'`.
  - For sparse queries (rare molecular masses), correct candidates in the pool ranked between 15 and 40 are dropped, forfeiting reciprocal rank credits ($1/16 \dots 1/25$).
* **The Solution**:
  - Turn on `FILL_25 = True` in `CFG`:
  ```python
  CFG.update({
      'FILL_25': True,
  })
  ```
  - When `FILL_25 = True`, short lists automatically backfill from:
    1. Engine-2 deep candidates (ranks 26–40).
    2. Gated PubChem candidates.
    3. Deeper base candidates (ranks 26–60).
  - All candidates are deduplicated by tautomer-canonical InChIKey14.

---

### Experiment 2: Dual Forward Model Variance Calibration (`ICE_LAM` vs `GL_LAM`)
* **Priority**: 🟠 **High**
* **Estimated Gain**: **`+0.004 to +0.008`**
* **The Problem**:
  - The current 0.409 baseline weights both forward models equally (`ICE_LAM = 1.0, GL_LAM = 1.0`).
  - **ICEBERG** uses combinatorial bond cleavage trees, which can produce high variance on complex macrocycles or steroids.
  - **GLACIER** uses a graph neural network with learned fragmentation embeddings, exhibiting lower variance and smoother similarity distributions.
* **The Solution**:
  - Perform grid evaluation on relative forward model weighting:
    - Variant 2A: `ICE_LAM = 0.8, GL_LAM = 1.2` (Prioritizes graph neural stability)
    - Variant 2B: `ICE_LAM = 0.6, GL_LAM = 1.4`
    - Variant 2C: `ICE_LAM = 1.0, GL_LAM = 1.5`
  - In `CFG`:
  ```python
  CFG.update({
      'ICE_LAM': 0.8,
      'GL_LAM': 1.2,
  })
  ```

---

### Experiment 3: Stage-C Post-Fusion Forward Rescoring Tuning
* **Priority**: 🟠 **High**
* **Estimated Gain**: **`+0.003 to +0.005`**
* **The Mechanism**:
  - In `fusion_core.py` (Stage C), the base list and Engine 2 list are fused via Reciprocal Rank Fusion:
    $$\text{RRF} = \frac{1}{K_{rr} + r_{v4}} + \alpha \cdot \frac{1}{K_{rr} + r_{\text{eng}}}$$
  - The fused list undergoes a second forward-model re-scoring pass. Currently `POST_ICE_LAM` and `POST_GL_LAM` default to `None` (reusing base weights).
* **The Solution**:
  - Test calibrated post-fusion re-ranking weights:
    - `POST_ICE_LAM = 0.8, POST_GL_LAM = 1.0`
    - `FUSE_ALPHA = 0.65` (boosts high-confidence analog propagation from Engine 2)
    - `FUSE_KRR = 2.5` (slightly sharper top-rank reward)

---

### Experiment 4: Constitutional Regioisomer In-Flight Injection
* **Priority**: 🟡 **Medium-High (Novel Chemotypes)**
* **Estimated Gain**: **`+0.005 to +0.012`**
* **The Problem**:
  - Novel natural products or newly synthesized compounds are completely absent from both candidate pools and PubChem. Database retrieval can never score $>0$ for these queries.
* **The Solution**:
  - For queries where $\text{lib\_max} < 0.50$ and the top candidate's neural score is high, generate valid constitutional regioisomers (ortho/meta/para migrations, phenolic transfers) on the top scaffold.
  - Inject these generated SMILES into slots 33–40 of Engine 2.
  - Because they share the identical formula, `ICE_UNION` groups them and evaluates them with ICEBERG and GLACIER.
  - If a generated isomer matches the measured spectrum better than the generic database scaffold, forward models rank it #1.

---

### Experiment 5: Fine-Grained Popularity Prior Scaling (`POP_MU`)
* **Priority**: 🟡 **Medium**
* **Estimated Gain**: **`+0.002 to +0.005`**
* **The Background**:
  - Step 1 ($\mu = 0.15$) $\to$ **`0.404`**
  - Step 3 ($\mu = 0.25$) $\to$ **`0.409`**
* **The Solution**:
  - Test fine-grained values around the sweet spot:
    - Variant 5A: `POP_MU = 0.28`
    - Variant 5B: `POP_MU = 0.32`
    - Variant 5C: Non-linear popularity scaling:
      $$f = z + \mu_1 \cdot \log(1 + \text{SIDs}) + \mu_2 \cdot \log(1 + \text{PMIDs})$$
      (weighting PubMed citations higher than raw substance catalog counts).

---

### Experiment 6: Hybrid Multi-Engine Tri-Fusion
* **Priority**: 🔵 **Exploratory**
* **Estimated Gain**: **`+0.008 to +0.015`**
* **The Architecture**:
  - Fusing 3 diverse retrieval engines:
    1. **Engine 1**: `v4n` base model (`fe_v4` + `fpnet_full1`).
    2. **Engine 2**: Two-ranker engine (BIO + AFIX + timsTOF analogs).
    3. **Engine 3**: Apex Deep PubChem channel or decoupled candidate generator (`haideptry` v38).
  - 3-Way Reciprocal Rank Fusion:
    $$\text{RRF}_3 = \frac{1.0}{3 + r_1} + \frac{0.6}{3 + r_2} + \frac{0.4}{3 + r_3}$$

---

## 📊 Experiment Tracking Ledger

| ID | Experiment Description | Key Config Changes | Expected Score | Actual LB | Status |
| :---: | :--- | :--- | :---: | :---: | :---: |
| **EXP-00** | Baseline 0.399 reference | Defaults | 0.399 | **0.399** | ✅ Done |
| **EXP-01** | Popularity prior + Gap frag | `POP_MU=0.15, FRAG_LAM=0.5, MODE='gap'` | 0.404 | **0.404** | ✅ Done |
| **EXP-02** | Library Gate | `ICE_LAM_LIB=0, GL_LAM_LIB=0` | 0.404 | **0.404** | ✅ Done |
| **EXP-03** | Stronger Popularity Prior | `POP_MU=0.25` | 0.409 | **0.409** | 🚀 Active |
| **EXP-04** | Fill-25 Backfill | `FILL_25=True` | 0.412 | TBD | ⏳ Queued |
| **EXP-05** | Forward Model Weight Rebalance | `ICE_LAM=0.8, GL_LAM=1.2` | 0.415 | TBD | ⏳ Queued |
| **EXP-06** | Popularity Tuning $\mu=0.30$ | `POP_MU=0.30, FILL_25=True` | 0.417 | TBD | ⏳ Queued |
| **EXP-07** | Regioisomer In-Flight Expansion | Regioisomer generator on Engine 2 | 0.425+ | TBD | ⏳ Queued |
| **EXP-08** | Tri-Engine Fusion | 3-way RRF consensus | 0.440+ | TBD | ⏳ Queued |

---

## 🛠️ Verification Protocol

1. **Safety First**: Every experiment is controlled by a single parameter in `CFG`.
2. **Local Commit Verification**: The commit run executes on 12 smoke molecules in ~15 minutes to guarantee zero runtime exceptions, zero schema violations, and complete reproducibility.
3. **Submission Verification**: Check that `submission.csv` contains exactly 400 rows, zero null values, no duplicated IDs, and max 25 valid SMILES per row.
