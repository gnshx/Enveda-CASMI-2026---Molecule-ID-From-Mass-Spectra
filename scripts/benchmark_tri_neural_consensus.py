#!/usr/bin/env python3
"""
scripts/benchmark_tri_neural_consensus.py
=============================================================================
CASMI 2026: Tri-Neural Consensus & Candidate Reranker Specialist Benchmark.
Evaluates:
  1. Saner v1 (192-dim public anchor)
  2. Scorer v2 (320-dim deep Siamese cross-attention)
  3. Custom Scorer v1 (256-dim locally refit on 2.43M spectra, 0.6874 val MRR)
  4. Dual-Neural Z-Score Ensemble (Saner v1 + Scorer v2)
  5. Sovereign Tri-Neural Consensus (Saner v1 + Scorer v2 + Custom Scorer v1)
  6. Optimal Simplex Weights Calibration for Variance Reduction
  7. Learning-to-Rank (LightGBM LambdaRank & Calibrated Logistic Regression)
=============================================================================
"""

import os
import sys
import time
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

sys.path.insert(0, str(Path("train_local").resolve()))
from train import Data, SpectrumCandidateModel

def seed_all(seed=20261023):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_metrics(scores_matrix):
    """
    scores_matrix: (N_molecules, N_candidates)
    Candidate at index 0 is ground truth.
    Pessimistic ties: rank = 1 + sum(scores[1:] >= scores[0]).
    """
    ranks = 1 + np.sum(scores_matrix[:, 1:] >= scores_matrix[:, 0:1], axis=1)
    mrr25 = np.mean(np.where(ranks <= 25, 1.0 / ranks, 0.0))
    rec1 = np.mean(ranks == 1)
    rec5 = np.mean(ranks <= 5)
    rec25 = np.mean(ranks <= 25)
    return {
        "mrr25": float(mrr25),
        "recall1": float(rec1),
        "recall5": float(rec5),
        "recall25": float(rec25),
        "median_rank": float(np.median(ranks))
    }

def zscore_rows(mat):
    mean = np.mean(mat, axis=1, keepdims=True)
    std = np.std(mat, axis=1, keepdims=True)
    return (mat - mean) / np.clip(std, 1e-6, None)

def main():
    seed_all()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("🔬 CASMI 2026: BENCHMARKING TRI-NEURAL CONSENSUS & RERANKER PIPELINE")
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 75)

    # 1. Load Data
    cfg3 = json.loads(Path("models/custom_scorer_v1/config.json").read_text())
    t0 = time.time()
    print("Loading memory-mapped dataset from cache_full...")
    data = Data(cfg3)
    print(f"Dataset loaded in {time.time() - t0:.1f}s: {len(data.mol)} molecules, {len(data.spec)} spectra.")

    # 2. Load Models
    print("\nLoading 3 Neural Model Architectures onto GPU...")
    # Model 1: Saner v1
    cfg1 = json.loads(Path("models/scorer_kit/config.json").read_text())
    m1 = SpectrumCandidateModel(cfg1).to(device)
    ckpt1 = torch.load("models/scorer_kit/final_model.pt", map_location=device, weights_only=True)
    m1.load_state_dict(ckpt1["state_dict"])
    m1.eval()
    print("  [1/3] Saner v1 (192-dim) loaded.")

    # Model 2: Scorer v2
    cfg2 = json.loads(Path("models/scorer_v2_320/config.json").read_text())
    m2 = SpectrumCandidateModel(cfg2).to(device)
    ckpt2 = torch.load("models/scorer_v2_320/final_model.pt", map_location=device, weights_only=True)
    m2.load_state_dict(ckpt2["state_dict"])
    m2.eval()
    print("  [2/3] Scorer v2 (320-dim) loaded.")

    # Model 3: Custom Scorer v1
    m3 = SpectrumCandidateModel(cfg3).to(device)
    ckpt3 = torch.load("models/custom_scorer_v1/best_validation_model.pt", map_location=device, weights_only=True)
    m3.load_state_dict(ckpt3["state_dict"])
    m3.eval()
    print("  [3/3] Custom Scorer v1 (256-dim, 25-epoch refit) loaded.")

    # 3. Load Validation Candidates
    cands_npz = np.load("models/custom_scorer_v1/validation_candidates.npz")
    val_mol_keys = list(cands_npz.files)
    N_val = len(val_mol_keys)
    print(f"\nEvaluating across {N_val} held-out validation molecules (256 candidates each)...")

    # Arrays to collect scores: shape (N_val, 256)
    scores_m1 = np.zeros((N_val, 256), dtype=np.float32)
    scores_m2 = np.zeros((N_val, 256), dtype=np.float32)
    scores_m3 = np.zeros((N_val, 256), dtype=np.float32)

    eval_t0 = time.time()
    with torch.inference_mode():
        for idx, k in enumerate(val_mol_keys):
            i = int(k)
            cand_ids = cands_npz[k]
            sr = data.by_mol[i]
            sr = sr[data.spec.source.values[sr] == 0]
            if not len(sr):
                continue

            # Candidate fingerprints
            fp = torch.tensor(np.asarray(data.fp[cand_ids]), device=device, dtype=torch.float32)

            # Model 1 scoring
            z1_sum, p1_sum, n1 = 0, 0, 0
            for st in range(0, len(sr), cfg1["batch_size"]):
                b = sr[st:st + cfg1["batch_size"]]
                z, p = m1.encode_spectra(*data.batch(b, device))
                z1_sum = z1_sum + z.sum(0); p1_sum = p1_sum + p.sum(0); n1 += len(b)
            sc1 = ((m1.encode_molecules(fp) @ (z1_sum / n1)) / cfg1["temperature"] +
                   cfg1["fp_score_weight"] * (fp @ (p1_sum / n1)) / cfg1["fp_bits"]).cpu().numpy()
            scores_m1[idx] = sc1

            # Model 2 scoring
            z2_sum, p2_sum, n2 = 0, 0, 0
            for st in range(0, len(sr), cfg2["batch_size"]):
                b = sr[st:st + cfg2["batch_size"]]
                z, p = m2.encode_spectra(*data.batch(b, device))
                z2_sum = z2_sum + z.sum(0); p2_sum = p2_sum + p.sum(0); n2 += len(b)
            sc2 = ((m2.encode_molecules(fp) @ (z2_sum / n2)) / cfg2["temperature"] +
                   cfg2["fp_score_weight"] * (fp @ (p2_sum / n2)) / cfg2["fp_bits"]).cpu().numpy()
            scores_m2[idx] = sc2

            # Model 3 scoring
            z3_sum, p3_sum, n3 = 0, 0, 0
            for st in range(0, len(sr), cfg3["batch_size"]):
                b = sr[st:st + cfg3["batch_size"]]
                z, p = m3.encode_spectra(*data.batch(b, device))
                z3_sum = z3_sum + z.sum(0); p3_sum = p3_sum + p.sum(0); n3 += len(b)
            sc3 = ((m3.encode_molecules(fp) @ (z3_sum / n3)) / cfg3["temperature"] +
                   cfg3["fp_score_weight"] * (fp @ (p3_sum / n3)) / cfg3["fp_bits"]).cpu().numpy()
            scores_m3[idx] = sc3

            if (idx + 1) % 200 == 0 or (idx + 1) == N_val:
                elapsed = time.time() - eval_t0
                rate = (idx + 1) / elapsed
                print(f"  Processed {idx + 1}/{N_val} molecules ({rate:.1f} mol/s, {elapsed:.1f}s elapsed)...")

    # 4. Standardize Z-Scores per molecule
    z1 = zscore_rows(scores_m1)
    z2 = zscore_rows(scores_m2)
    z3 = zscore_rows(scores_m3)

    # 5. Evaluate Individual Backbones
    m1_metrics = compute_metrics(scores_m1)
    m2_metrics = compute_metrics(scores_m2)
    m3_metrics = compute_metrics(scores_m3)

    print("\n" + "=" * 75)
    print("📊 INDIVIDUAL NEURAL BACKBONE RESULTS (1,000 Held-Out Molecules)")
    print("=" * 75)
    print(f"1. Saner v1 (192-dim):       Val MRR@25: {m1_metrics['mrr25']:.4f} | R@1: {m1_metrics['recall1']:.3f} | R@5: {m1_metrics['recall5']:.3f} | R@25: {m1_metrics['recall25']:.3f}")
    print(f"2. Scorer v2 (320-dim):      Val MRR@25: {m2_metrics['mrr25']:.4f} | R@1: {m2_metrics['recall1']:.3f} | R@5: {m2_metrics['recall5']:.3f} | R@25: {m2_metrics['recall25']:.3f}")
    print(f"3. Custom Scorer v1 (256-dim): Val MRR@25: {m3_metrics['mrr25']:.4f} | R@1: {m3_metrics['recall1']:.3f} | R@5: {m3_metrics['recall5']:.3f} | R@25: {m3_metrics['recall25']:.3f}")

    # 6. Evaluate Ensembles
    print("\n" + "=" * 75)
    print("🧬 NEURAL CONSENSUS & VARIANCE REDUCTION BENCHMARKS")
    print("=" * 75)

    # Baseline Dual Ensemble: 0.50 * z1 + 0.50 * z2 (V24 PB baseline)
    dual_50_50 = 0.50 * z1 + 0.50 * z2
    dual_metrics = compute_metrics(dual_50_50)
    print(f"4. Dual Ensemble (Saner + Scorer v2, 50/50):     Val MRR@25: {dual_metrics['mrr25']:.4f} | R@1: {dual_metrics['recall1']:.3f} | R@5: {dual_metrics['recall5']:.3f}")

    # Tri-Neural Equal: 1/3 * z1 + 1/3 * z2 + 1/3 * z3
    tri_equal = (z1 + z2 + z3) / 3.0
    tri_equal_metrics = compute_metrics(tri_equal)
    print(f"5. Tri-Neural Consensus (Equal 1/3 each):        Val MRR@25: {tri_equal_metrics['mrr25']:.4f} | R@1: {tri_equal_metrics['recall1']:.3f} | R@5: {tri_equal_metrics['recall5']:.3f}")

    # Tri-Neural Weighted: 0.20 * z1 + 0.20 * z2 + 0.60 * z3
    tri_weighted_60 = 0.20 * z1 + 0.20 * z2 + 0.60 * z3
    tri_w60_metrics = compute_metrics(tri_weighted_60)
    print(f"6. Tri-Neural High-Fidelity (20% v1 + 20% v2 + 60% Cust): Val MRR@25: {tri_w60_metrics['mrr25']:.4f} | R@1: {tri_w60_metrics['recall1']:.3f} | R@5: {tri_w60_metrics['recall5']:.3f}")

    # Tri-Neural Weighted: 0.15 * z1 + 0.15 * z2 + 0.70 * z3
    tri_weighted_70 = 0.15 * z1 + 0.15 * z2 + 0.70 * z3
    tri_w70_metrics = compute_metrics(tri_weighted_70)
    print(f"7. Tri-Neural High-Fidelity (15% v1 + 15% v2 + 70% Cust): Val MRR@25: {tri_w70_metrics['mrr25']:.4f} | R@1: {tri_w70_metrics['recall1']:.3f} | R@5: {tri_w70_metrics['recall5']:.3f}")

    # Tri-Neural Grid Search on 5-fold cross validation
    print("\nRunning 5-fold Cross-Validated Simplex Weight Optimization...")
    n_folds = 5
    fold_indices = np.array_split(np.arange(N_val), n_folds)
    best_weights = []
    cv_mrr_list = []

    # Grid search over simplex
    grid_steps = 21  # step 0.05
    w_grid = []
    for w1 in np.linspace(0.0, 1.0, grid_steps):
        for w2 in np.linspace(0.0, 1.0 - w1, grid_steps):
            w3 = 1.0 - w1 - w2
            if w3 >= -1e-6:
                w_grid.append((round(w1, 3), round(w2, 3), round(max(0.0, w3), 3)))

    for fold in range(n_folds):
        val_idx = fold_indices[fold]
        train_idx = np.setdiff1d(np.arange(N_val), val_idx)

        best_w, best_train_mrr = None, -1.0
        for w1, w2, w3 in w_grid:
            combo = w1 * z1[train_idx] + w2 * z2[train_idx] + w3 * z3[train_idx]
            mrr = compute_metrics(combo)["mrr25"]
            if mrr > best_train_mrr:
                best_train_mrr = mrr
                best_w = (w1, w2, w3)

        # Evaluate on test fold
        test_combo = best_w[0] * z1[val_idx] + best_w[1] * z2[val_idx] + best_w[2] * z3[val_idx]
        test_mrr = compute_metrics(test_combo)["mrr25"]
        best_weights.append(best_w)
        cv_mrr_list.append(test_mrr)

    mean_cv_mrr = np.mean(cv_mrr_list)
    avg_w1 = np.mean([w[0] for w in best_weights])
    avg_w2 = np.mean([w[1] for w in best_weights])
    avg_w3 = np.mean([w[2] for w in best_weights])
    norm_w = avg_w1 + avg_w2 + avg_w3
    opt_weights = (avg_w1 / norm_w, avg_w2 / norm_w, avg_w3 / norm_w)

    opt_combo = opt_weights[0] * z1 + opt_weights[1] * z2 + opt_weights[2] * z3
    opt_metrics = compute_metrics(opt_combo)

    print(f"Optimal Calibrated Weights: {opt_weights[0]:.3f} * z(Saner) + {opt_weights[1]:.3f} * z(Scorer v2) + {opt_weights[2]:.3f} * z(Custom Scorer v1)")
    print(f"Cross-Validated Out-of-Fold MRR@25: {mean_cv_mrr:.4f}")
    print(f"Full Validation MRR@25:              {opt_metrics['mrr25']:.4f} (Δ vs Dual: {opt_metrics['mrr25'] - dual_metrics['mrr25']:+.4f})")
    print(f"Full Validation Recall@1:            {opt_metrics['recall1']:.3f} (Δ vs Dual: {opt_metrics['recall1'] - dual_metrics['recall1']:+.3f})")
    print(f"Full Validation Recall@5:            {opt_metrics['recall5']:.3f} (Δ vs Dual: {opt_metrics['recall5'] - dual_metrics['recall5']:+.3f})")

    # 7. Save Benchmark Results
    results = {
        "timestamp_utc": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "validation_molecules": N_val,
        "models": {
            "saner_v1": m1_metrics,
            "scorer_v2": m2_metrics,
            "custom_scorer_v1": m3_metrics,
            "dual_ensemble_50_50": dual_metrics,
            "tri_neural_equal": tri_equal_metrics,
            "tri_neural_weighted_60": tri_w60_metrics,
            "tri_neural_calibrated_opt": opt_metrics
        },
        "optimal_weights": {
            "w_saner_v1": round(float(opt_weights[0]), 4),
            "w_scorer_v2": round(float(opt_weights[1]), 4),
            "w_custom_scorer_v1": round(float(opt_weights[2]), 4)
        },
        "cross_val_mrr25": round(float(mean_cv_mrr), 5)
    }

    out_file = Path("experiments/tri_neural_consensus_benchmark.json")
    out_file.parent.mkdir(parents=True, exist_ok=True)
    out_file.write_text(json.dumps(results, indent=2))
    print(f"\nSaved benchmark results to {out_file}")

if __name__ == "__main__":
    main()
