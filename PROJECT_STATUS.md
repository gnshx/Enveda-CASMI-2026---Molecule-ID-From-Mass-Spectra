# 🚦 PROJECT_STATUS.md
# Real-Time Project Status & Live Execution Dashboard
**Status Timestamp**: 2026-11-17 01:47 IST (2026-11-16 20:17 UTC)  
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
| **Our Active Submission** | **Version 31 (Ref `58442788`)** | **`PENDING`** | **N-1c Molecular Network + ExtLib $\tau=0.80$ + Logistic Reranker (Target: 0.488 – 0.492+)** |
| **Our Verified PB** | **Version 24 (Ref `57982390`)** | **`0.4800`** | Dual-Neural Z-Score Ensemble (Saner v1 + Scorer v2) + MIDX Dedup |

---

## 2. Active Run Telemetry: Version 31

- **Submission Reference**: `58442788`
- **Submitted At**: `2026-11-16 20:04:11 UTC`
- **Kaggle Status**: `SubmissionStatus.PENDING` (Worker active on hidden test evaluation)
- **Container Execution Guard**: Enforced `TIME_GUARD_HOURS = 7.5` and single-pass outsider scoring. Expected completion runtime: ~6.0 – 6.5 hours.
- **Monitoring Daemon**: `scripts/score_poller.py` daemon actively polling every 30 seconds.

---

## 3. Daily Resource Budget

- **Daily Submission Quota**: **4 of 5 submissions available** (Resets at 00:00 UTC daily).
- **Local Compute**: NVIDIA GeForce RTX 3060 (12GB VRAM) — 100% idle, 28 MiB used, ready for local model training and validation benchmarks.
- **Kaggle GPU Quota**: Healthy. Notebook commit executions are running in fast path (3 minutes).
- **Code Repository**: Synchronized with GitHub `main` at commit `1dd2a8b`.
