# 📋 PROJECT_AUDIT.md
# Comprehensive Repository and Asset Audit
**Audit Date**: 2026-11-17  
**Project Path**: `/home/dlcv/Enveda-CASMI-2026`  
**Host Architecture**: Linux x86_64, Ubuntu 24.04 LTS  
**Compute Resources**: NVIDIA GeForce RTX 3060 (12GB VRAM), Driver 595.99.02, CUDA 13.2  

---

## 1. Physical Assets & Data Inventory

### Data Files (`data/`)
| File | Size | Records / Description | Status |
|---|:---:|---|:---:|
| `data/train.parquet` | 2,892.77 MB | 2,431,940 tandem MS/MS spectra covering ~275k molecules. | Verified & intact |
| `data/test.parquet` | 4.62 MB | 400 test molecules (evaluation split with multi-CE spectra). | Verified & intact |
| `data/train_sample_1k.parquet` | 1.78 MB | 1,000 spectrum smoke test split for rapid iteration. | Verified & intact |
| `data/sample_submission.csv` | 0.04 MB | Official submission format template (400 molecules). | Verified & intact |
| `data/validation_folds/` | 22.58 MB | Pre-computed multi-domain validation folds (spectra, summary). | Verified & intact |

### Trained Checkpoints & Models (`models/`)
| Artifact | Size | Description | Origin / Architecture |
|---|:---:|---|---|
| `models/custom_scorer_v1/final_model.pt` | 19.15 MB | 25-epoch full refit on 2.43M spectra. Val MRR: **0.6874**. | Locally trained on RTX 3060 CUDA |
| `models/fpnet_v2_weights.pt` | 145.50 MB | Multi-head self-attention fingerprint network (4,096 Morgan bits). | Local training / FPNet |
| `models/fpnet_weights.pt` | 113.45 MB | Base fingerprint network checkpoint. | Public baseline |
| `models/scorer_kit/r1_final_model.pt` | 11.32 MB | Rank-1 calibrated neural candidate scorer. | Scorer kit bundle |

---

## 2. Codebase Modular Architecture Audit

| Directory | Core Modules | Purpose & Findings |
|---|---|---|
| `chemistry/` | `adducts.py` | Accurate precursor mass shift calculation across positive/negative adducts (`[M+H]+`, `[M-H]-`, `[M+Na]+`, `[M+K]+`, `[M+NH4]+`, `[M+HCOO]-`). Verified robust. |
| `validation/` | `metrics.py`, `splitter.py` | Official competition metric implementation (`smiles_to_inchikey14`, `mrr_at_25`, `recall_at_k`) + strict molecule-grouped validation splits (`structure_split`, `library_split`, `np_split`). |
| `retrieval/` | `cosine.py` | Peak alignment and square-root weighted modified cosine similarity search. |
| `candidates/` | `candidate_generator.py` | Formula-constrained candidate retrieval from local indices. |
| `features/` | `spectral_features.py` | 160-dimensional spectral feature extraction for candidate reranking. |
| `ranking/` | `evaluate_pipeline_v2.py`, `regio_generator.py` | Hybrid consensus pipelines, isomer reranking, candidate list fusion. |
| `scripts/` | `deploy_and_submit_v31.py`, `score_poller.py` | Autonomous deployment pipeline, Kaggle CLI interface, continuous score poller daemon. |
| `deploy_kernel_*/` | `v22` through `v31` | Standalone Kaggle submission notebook packages with embedded assets and configurations. |

---

## 3. Forensic Analysis: What Works vs What Caused Regressions

### ✅ What Has Been Proven to Work:
1. **Skeletal Deduplication (MIDX / InChIKey14)**: Deduplicating redundant stereoisomers in top-25 lists provided an immediate +0.007 LB jump (0.472 $\to$ 0.479).
2. **Rank 1 Shielding**: Protecting high-confidence library matches ($w_{\text{top}} = 0.00$ or strictly gated) preserves exact hits from being corrupted by noisy neural rerankers.
3. **Dual-Neural Z-Score Fusion**: Ensembling Saner v1 with a wider neural scorer ($0.65 / 0.35$ or $0.50 / 0.50$) achieved our PB of **0.4800**.
4. **Calibrated ExtLib Matching**: Lowering the reference spectral matching threshold from $\tau=0.85$ to $\tau=0.80$ with $\ge 4$ peaks yields a proven +0.004 LB gain.
5. **In-Test Molecular Network (`N-1c`)**: Propagating structural evidence across mutually similar query spectra yields an additional +0.004 LB gain (0.484 $\to$ 0.488).

### ❌ What Failed & Why:
1. **Unranked LOTUS Decoy Backfill (Version 25: 0.472)**: Injected 150k raw LOTUS candidate structures into empty slots without neural ranking. Irrelevant decoys displaced valid tail candidates, dropping MRR by -0.008.
2. **Displacing Rank 1 Leaders (Version 26 & 27: 0.470 & 0.467)**: Using $w_{\text{top}} = 0.25$ allowed noisy neural predictions to demote correct Rank-1 physical library hits to Rank 2 or 3 (costing 0.50 to 0.67 MRR per flip).
3. **Double-Outsider Neural Pass Timeout (Version 28 & 29: Zombie PENDING)**: Attempting to score outside candidates twice using two different neural models on Kaggle exceeded the 9.0-hour container timeout, causing the worker to be killed by SIGKILL.
