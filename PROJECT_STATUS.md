# 🚦 PROJECT_STATUS.md
# Real-Time Project Status & Live Execution Dashboard
**Status Timestamp**: 2026-11-17 05:32 IST (2026-11-17 00:02 UTC)  
**Competition Deadline**: 2026-12-14 23:59 UTC (27 days remaining)  

---

## 1. Live Competition Standings

| Rank / Position | Team / Competitor | Public Score (MRR@25) | Strategy / Known Approach |
|:---:|---|:---:|---|
| **#1** | `Shehab Alshehabi` | **`0.549`** | Multi-channel retrieval + deep analog propagation + ensemble reranking |
| **#2** | `democat` | **`0.540`** | Hybrid spectral library + molecular network fusion |
| **#3** | `Oleksandr Parfionov` | **`0.536`** | Learned candidate reranker + formula filtering |
| **#4** | `Savko` | **`0.528`** | Top-1 evidence gap closing + external libraries |
| **Top 20 Cutoff** | `huytngo` | **`0.503`** | Peaks to structures + network reranking |
| **Our Active Deployment** | **Version 32 (Kernel v33)** | **`Pending`** | **Sovereign Immutable Rank-1 Shield (0.00 w_top) + Calibrated Neural Reranker (0.90 w_mid) + NP Prior (0.25) + MIDX Dedup (Target: 0.490 – 0.495+)** |
| **Our Verified PB** | **Version 24 (Ref `57982390`)** | **`0.4800`** | Dual-Neural Z-Score Ensemble (Saner v1 + Scorer v2) + MIDX Dedup |

---

## 2. Active Run Telemetry: Version 32 (Kernel Version 33)

- **Kernel Slug**: `nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399`
- **Pushed At**: `2026-11-17 00:56:10 UTC`
- **Kaggle Status**: `KernelWorkerStatus.RUNNING` (Commit validation active on Kaggle cloud worker)
- **Monitoring Runner**: `scripts/deploy_and_submit_v32.py --submit-only` (PID `1344262`) actively polling every 15s to trigger competition submission upon commit completion.
- **Architectural Safeguards**:
  - `w_top = 0.00` & `PIN_OLD_LEADER = True`: 100% hard protection for all 305 verified Regime A reference library hits.
  - `w_mid = 0.90`: Simplex optimal weight for Scorer v2 on ranks 2–25 (+0.0541 Val MRR lift).
  - `POP_W = 0.25`, `NP_W = 0.10`: Natural product prior for out-of-pool candidate discovery (+0.4401 lift).
  - Metric-Exact Deduplication: `ENABLE_K14_DEDUP = True` via canonical tautomer enumeration.
- **Monitoring Daemon**: `scripts/autonomous_lab_daemon.py` (PID `1338448`) 24/7 master lab daemon actively polling every 45 seconds with hardware & system telemetry.

---

## 3. Daily Resource Budget

- **Daily Submission Quota**: **5 of 5 submissions available** (Reset at 00:00 UTC on 2026-11-17).
- **Local Compute**: NVIDIA GeForce RTX 3060 (12GB VRAM) — 100% idle, 28 MiB used, 39°C, ready for local model training and validation benchmarks.
- **Kaggle GPU Quota**: Healthy. Kernel Version 33 executing commit validation on Kaggle cloud.
- **Code Repository**: Synchronized with GitHub `main` at commit `d9c945c`.

---

## 4. Phase 28: Regime C Constitutional Isomer Disambiguation & Admission Guard

- **Module**: `ranking/regio_generator.py` upgraded with `to_canonical_inchikey14` and `admit_candidates_safely`.
- **Constitutional Regioisomer Generation**:
  - 100% elemental formula conservation (910/910 variants strictly match parent stoichiometry).
  - 100% unique InChIKey14 generation across aromatic migrations, N-alkyl shifts, and enol unsaturations.
- **MetFrag-lite Bond Cleavage Disambiguation**:
  - Lifts true isomer MRR from `0.3054` (random order) to `0.5751` (**+0.2697 lift**).
  - Top-1 Accuracy: 38.0%, Top-3: 67.8%, Top-5: 85.1%.
- **Candidate Expansion Benchmark on Depleted Pools (<25 candidates)**:
  - 100% true structure recovery (43/43 holdout queries).
  - Boosts Regime C Val MRR from `0.0000` to `0.1564` (**+0.1564 lift**) and up to `0.3159` under full consensus reranking (Recall@5 = 58.1%).
  - Audit Reports: `experiments/regime_c_isomer_disambiguation_report.json` and `experiments/regime_c_mrr_impact_report.json`.

---

## 4. Swarm Milestone 2 Integration & Pre-Flight Certification

- **Milestone 2 Unified Pipeline Module**: [pipelines/milestone2_pipeline.py](file:///home/dlcv/Enveda-CASMI-2026/pipelines/milestone2_pipeline.py)
  - Tri-Neural Consensus Weights: $0.10 \cdot z(\text{Saner}) + 0.90 \cdot z(\text{Scorer v2})$.
  - Bruker timsTOF Empirical Mass Calibration: $-1.50\text{ ppm}$ offset with $\sigma = 3.5\text{ ppm}$ Gaussian prior.
  - Multi-CE Stepped Collision Energy Consensus: Dynamic spectral pooling across stepped energies.
  - Diagnostic Neutral Loss Detection: Plausibility multiplier for $\text{H}_2\text{O}$, $\text{NH}_3$, $\text{CO}$, $\text{CO}_2$, $\text{HCOOH}$, hexose.
  - MetFrag-lite Bond Cleavage Consistency: $\le 2$ breaks ($w_1 = 1.0, w_2 = 0.6$).
  - Absolute Rank-1 Shield: Freezes confirmed Regime A gold reference hits (305 queries).
- **Pre-Flight Verification Status**: **100% PASSED (7 / 7 Checks Certified)**
  - Submission Target: [submissions/submission_milestone2.csv](file:///home/dlcv/Enveda-CASMI-2026/submissions/submission_milestone2.csv)
  - Dimensionality & Queries: Exactly 400 rows $\times$ 25 slots matching test queries in exact order.
  - SMILES Validity: 10,000 / 10,000 parsed by RDKit with zero syntax or valence errors.
  - InChIKey14 Skeletal Deduplication: Zero duplicate skeletons (0.00% redundancy).
  - Runtime Budget: Offline execution completed in 93.1 seconds (1.55 minutes, well under the 7.5h limit).
  - Audit Report: [experiments/milestone2_verification_report.json](file:///home/dlcv/Enveda-CASMI-2026/experiments/milestone2_verification_report.json).

