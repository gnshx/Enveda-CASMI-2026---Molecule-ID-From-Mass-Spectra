# 📜 EXPERIMENT_LOG.md
# Comprehensive Experiment Ledger: From Baseline to SOTA Frontier
**Project**: Enveda CASMI 2026 — Molecule Identification From Mass Spectra  
**Current Personal Best**: `0.4800` (Version 24, Ref `57982390`)  
**Active Frontier**: `Version 31` (Ref `58442788`, Target 0.488 – 0.492+)  

---

## 1. Complete Kaggle Submission Ledger

| Version | Submission Ref | Date (UTC) | Architecture & Core Hypothesis | Public LB | Status | Key Finding / Hypothesis Validation |
|:---:|:---:|:---:|---|:---:|:---:|---|
| **V3** | `56123401` | 2026-10-15 | Baseline CNN on Morgan 2048 fingerprints against COCONUT | **`0.087`** | COMPLETE | Initial proof of concept; confirmed data loader and format. |
| **V5** | `56245102` | 2026-10-18 | Transformer `FPNet v1` with sinusoidal $m/z$ embeddings | **`0.129`** | COMPLETE | Attention mechanism captures peak correlations; $+0.042$ gain. |
| **V10** | `56512398` | 2026-10-24 | Gaussian $\text{ppm}$ weighting + multi-collision energy pooling | **`0.145`** | COMPLETE | Fingerprint prediction hits ceiling; cannot resolve isomers. |
| **V11** | `56891230` | 2026-10-27 | Modified cosine spectral library search + timsTOF shift matching | **`0.332`** | COMPLETE | **Massive breakthrough**: Reference library matching jumps score $+0.187$. |
| **V14** | `57123980` | 2026-10-30 | Two-Ranker Engine (ChEBI/LIPID MAPS + timsTOF drift) | **`0.350`** | COMPLETE | Expansion of natural product pools improves coverage. |
| **V18** | `57451203` | 2026-11-01 | Quad-channel retrieval + Ranker A/B blending | **`0.358`** | COMPLETE | Multi-channel candidate ranking yields robust consensus. |
| **v4n** | `57650123` | 2026-11-02 | v4n base retrieval + PubChem $N_1=5000$ recall | **`0.384`** | COMPLETE | Deep candidate recall prevents missing true molecules. |
| **V20** | `57812904` | 2026-11-03 | Engine Fusion + Union Forward Models (ICEBERG/GLACIER) | **`0.399`** | COMPLETE | Evaluating forward MS/MS on union of candidate pools. |
| **V21** | `57890123` | 2026-11-03 | PubChem Popularity Prior (`mu = 0.15`) + Gap fragment rescoring | **`0.404`** | COMPLETE | Penalizing obscure synthetic compounds boosts true NP ranks. |
| **V22** | `57912345` | 2026-11-03 | Stronger Prior (`mu = 0.25`) + Library Gate (`ICE/GL = 0`) | **`0.409`** | COMPLETE | Highest single-pipeline public master baseline. |
| **V23** | `57951234` | 2026-11-04 | PubChem Join (`PC_JOIN_N = 50`) into main engine ranker | **`0.420`** | COMPLETE | Deep outside answers brought to rank 1. |
| **V24** | `57982390` | 2026-11-04 | **Dual-Neural Z-Score Ensemble (Saner + Scorer) + MIDX Dedup** | **`0.4800`** | **COMPLETE (PB)** | **Personal Best**: Standardized dual-neural score + skeletal dedup + rank 1 shield. |
| **V25** | `57983294` | 2026-11-04 | Tri-Neural Multi-Capacity + Raw LOTUS 150k KB + MIDX Dedup | **`0.472`** | COMPLETE | **Regression**: Unfiltered LOTUS introduced false-positive synthetic decoys. |
| **V26** | `57992067` | 2026-11-04 | Apex Sovereign Balanced Tri-Force + In-Pool InChIKey14 Dedup | **`0.470`** | COMPLETE | Aggressive neural re-weighting disturbed confident ExtLib rank-1 matches. |
| **V27** | `58021007` | 2026-11-05 | Sovereign Consensus + Locked Rank 1 Shield + LOTUS Backfill | **`0.467`** | COMPLETE | Candidate competition in intermediate ranks reduced MRR on Regime B. |
| **V28** | `58168204` | 2026-11-09 | Apex Sovereign Dual-Tier ExtLib (Gold/Silver) + timsTOF Prior | — | **ERROR (Timeout)** | Double outsider neural pass exceeded Kaggle 9.0h timeout; killed by system. |
| **V29** | `58168293` | 2026-11-09 | Apex Sovereign with Calibrated ExtLib + Out-of-Pool Scoring | — | **ERROR (Timeout)** | Second run also killed by 9.0h timeout; queue locked for 7 days. |
| **V31** | **`58442788`** | 2026-11-16 | **Apex Sovereign Frontier (N-1c Network + ExtLib 0.80 + Logistic Reranker)** | **Pending** | **PENDING (Active)** | Flushed zombie queue; single-pass outsider scoring; running cleanly. |

---

## 2. Key Local Validation Experiments & Breakthroughs

### Experiment EXP-042: Local 25-Epoch Custom Scorer Refit
- **Objective**: Adapt general chemical embeddings to Enveda natural product MS/MS.
- **Hardware**: Local NVIDIA GeForce RTX 3060 (12GB VRAM), batch size 256, mixed precision FP16.
- **Data**: Full 2,431,940 spectra in `train.parquet`.
- **Result**: Validation MRR@25 on held-out 20% fold increased from `0.4780` (Scorer v2) to **`0.6874`**.
- **Artifact**: `models/custom_scorer_v1/final_model.pt` (19.15 MB).

### Experiment EXP-048: In-Pool InChIKey14 Deduplication Geometry
- **Objective**: Validate mathematical theorem on eliminating redundant stereoisomers.
- **Test Set**: 400 validation query spectra.
- **Result**:
  - Raw candidate list (with stereoisomers): `val_mrr = 0.4731`
  - InChIKey14 deduplicated candidate list: `val_mrr = 0.4802` (+0.0071 MRR gain).
  - Confirmed: Evaluation ignores stereochemistry; redundant isomers steal slots.

### Experiment EXP-055: N-1c Test Molecular Graph Propagation
- **Objective**: Propagate structural evidence across related queries in the test set.
- **Graph Topology**: Cosine similarity threshold $\tau \ge 0.60$, minimum shared peaks $k \ge 6$.
- **Propagation Weight**: $\lambda = 0.35$.
- **Result**: Validation MRR increased by $+0.0042$, successfully resolving 6 previously missed natural product analogs.
