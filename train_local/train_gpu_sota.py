#!/usr/bin/env python3
"""
train_local/train_gpu_sota.py
End-to-End High-Throughput GPU Training on CUDA for Enveda CASMI 2026.
Uses NVIDIA GeForce RTX 3060 (12 GB VRAM) with mixed precision (AMP fp16).
Trains a multi-head Transformer SpectrumCandidateModel directly on data/train.parquet.
"""

import os, sys, json, time, shutil
from pathlib import Path
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prepare, train

def main():
    print("=" * 70)
    print("🚀 STARTING LOCAL CUDA GPU TRAINING PIPELINE")
    print("=" * 70)

    # 1. GPU Verification
    assert torch.cuda.is_available(), "CUDA GPU is required!"
    dev_name = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"Device: {dev_name} ({vram_gb:.2f} GB VRAM)")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")

    # 2. Configuration for SOTA High-Capacity Model
    cfg = {
        "seed": 20261023,
        "fp_bits": 2048,
        "max_peaks": 96,
        "min_peaks": 3,
        "width": 256,               # High-capacity width (up from 192)
        "heads": 8,                 # 8 attention heads
        "layers": 4,                # 4 Transformer Encoder layers
        "embedding": 256,
        "dropout": 0.1,
        "temperature": 0.1,
        "aux_weight": 0.5,
        "fp_score_weight": 2.0,
        "batch_size": 128,          # Optimized for 12GB RTX 3060
        "hard_negatives": 16,
        "negative_pool": 128,
        "epochs": 25,
        "steps_per_epoch": 1000,
        "patience": 6,
        "lr": 0.0002,
        "weight_decay": 0.01,
        "val_fraction": 0.05,       # 5% identity-disjoint holdout
        "val_candidates": 256,
        "max_validation_molecules": 1000,
        "holdout_hours": 12.0,
        "refit_hours": 12.0,
        "refit_full": True,
        "require_gpu": True,
        "output": str(Path("models/custom_scorer_v1").resolve()),
        "work": str(Path("train_local/cache_full").resolve()),
        "train_path": str(Path("data/train.parquet").resolve()),
        "external_paths": []
    }

    work_dir = Path(cfg["work"])
    out_dir = Path(cfg["output"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save active config
    (out_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Work Cache: {work_dir}")
    print(f"Model Output: {out_dir}")

    # 3. Stage 1: Data Preparation (Skip if already cached)
    if not (work_dir / "peaks.bin").exists() or not (work_dir / "molecules.parquet").exists():
        print("\n--- STAGE 1: Data Preparation & Streaming Spooling ---")
        t0 = time.time()
        audit = prepare.prepare(cfg)
        print(f"Data Preparation completed in {time.time() - t0:.1f}s!")
        print(f"Total spectra: {audit['spectra']:,}, Unique molecules: {audit['molecules']:,}")
    else:
        print("\n--- STAGE 1: Using Existing Cache ---")
        print(f"Found existing binary spools at {work_dir}")

    # 4. Stage 2: Transformer Training on CUDA GPU
    print("\n--- STAGE 2: Deep Spectrum-Candidate Training on CUDA GPU ---")
    t0 = time.time()
    result = train.run(cfg)
    print(f"\n🎉 Training Pipeline Completed in {(time.time() - t0)/60:.2f} minutes!")
    print(f"Best Epoch: {result.get('best_epoch')}")
    print(f"Best Validation Diagnostic MRR@25: {result.get('best_diagnostic_mrr25')}")
    print(f"Final Model saved to: {out_dir / 'final_model.pt'}")

if __name__ == "__main__":
    main()
