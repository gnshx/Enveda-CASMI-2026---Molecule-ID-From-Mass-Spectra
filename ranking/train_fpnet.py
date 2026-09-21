"""
ranking/train_fpnet.py

Train FPNet on GPU (RTX 3060) to predict 2,048-bit Morgan + 166-bit MACCS keys from MS/MS spectra.
Zero leakage: enveda-np-examples (250 validation molecules) strictly withheld for validation.
"""

import os
import sys
import time
import math
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
sys.path.insert(0, str(ROOT))

from ranking.fpnet import FPNet

warnings.filterwarnings("ignore")

T0 = time.time()
def log(msg: str):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)

def extract_float(val, default=30.0) -> float:
    if val is None:
        return default
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return default
        return float(val[0])
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default


# ── Fast Numba Spectral Binning ──────────────────────────────────────────────

@numba.njit(fastmath=True)
def bin_spectrum(mzs: np.ndarray, ints: np.ndarray, n_bins: int = 4096, min_mz: float = 50.0, max_mz: float = 1074.0) -> np.ndarray:
    bin_size = (max_mz - min_mz) / n_bins
    out = np.zeros(n_bins, dtype=np.float32)
    for i in range(len(mzs)):
        mz = mzs[i]
        if min_mz <= mz < max_mz:
            idx = int((mz - min_mz) / bin_size)
            if ints[i] > out[idx]:
                out[idx] = ints[i]
    mx = 0.0
    for i in range(n_bins):
        if out[i] > 0:
            out[i] = np.sqrt(out[i])
            if out[i] > mx: mx = out[i]
    if mx > 0:
        for i in range(n_bins):
            out[i] /= mx
    return out


# ── Fingerprint Cache Helper ──────────────────────────────────────────────────

def compute_compound_fingerprint(smiles: str) -> np.ndarray:
    """Compute 2048-bit Morgan + 166-bit MACCS keys as float32 binary array."""
    target = np.zeros(2048 + 166, dtype=np.float32)
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Morgan 2048
            fp_m = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            for bit in fp_m.GetOnBits():
                target[bit] = 1.0
            # MACCS 166 (bits 1 to 166)
            fp_c = MACCSkeys.GenMACCSKeys(mol)
            for bit in fp_c.GetOnBits():
                if 0 <= bit < 166:
                    target[2048 + bit] = 1.0
    except Exception:
        pass
    return target


# ── Dataset Definition ────────────────────────────────────────────────────────

class SpectralDataset(Dataset):
    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(features)
        self.targets = torch.from_numpy(targets)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ── Data Loading & Preparation ────────────────────────────────────────────────

def prepare_data(max_train_samples: int = 100_000):
    log("Loading and preparing training & validation data...")
    train_path = ROOT / "data" / "train.parquet"
    pf = pq.ParquetFile(str(train_path))

    COLS = [
        "ingest_lib", "normalized_smiles", "inchikey14", "precursor_mz",
        "ms2_mzs", "ms2_normalized_intensities", "num_peaks", "collision_energy_ev", "ionization_mode"
    ]
    INDEX_LIBS = {
        "pluskal_ms2", "riken", "gnps", "massbank", "mona",
        "msdial", "spectraverse", "drug_plus", "masaryk"
    }

    raw_train_specs = []
    raw_val_specs = []

    # 1. Load validation spectra strictly from row group 20 (enveda-np-examples)
    log("Loading validation spectra from row group 20...")
    tbl_val = pf.read_row_group(20, columns=COLS)
    p_val = tbl_val.to_pydict()
    for i in range(len(p_val["ingest_lib"])):
        lib = p_val["ingest_lib"][i]
        if lib != "enveda-np-examples":
            continue
        smi = p_val["normalized_smiles"][i]
        n_p = p_val["num_peaks"][i]
        if smi is None or n_p < 3:
            continue
        pm = extract_float(p_val["precursor_mz"][i], 300.0)
        ce = extract_float(p_val["collision_energy_ev"][i], 30.0)
        ion = 1.0 if str(p_val["ionization_mode"][i]).lower().startswith("pos") else 0.0
        mzs = np.asarray(p_val["ms2_mzs"][i], dtype=np.float32)
        ints = np.asarray(p_val["ms2_normalized_intensities"][i], dtype=np.float32)
        raw_val_specs.append({
            "mzs": mzs, "ints": ints, "pm": pm, "ce": ce, "ion": ion,
            "smiles": smi, "inchikey14": p_val["inchikey14"][i]
        })
    log(f"Loaded {len(raw_val_specs):,} validation spectra.")

    # 2. Load training spectra from index libraries (RG 9, 10, 11, etc.)
    log("Loading training spectra from index libraries...")
    TARGET_RGS = [9, 10, 11, 12, 13, 16, 17, 18, 0]
    for rg in TARGET_RGS:
        tbl = pf.read_row_group(rg, columns=COLS)
        p = tbl.to_pydict()
        for i in range(len(p["ingest_lib"])):
            lib = p["ingest_lib"][i]
            if lib not in INDEX_LIBS:
                continue
            smi = p["normalized_smiles"][i]
            n_p = p["num_peaks"][i]
            if smi is None or n_p < 3:
                continue
            pm = extract_float(p["precursor_mz"][i], 300.0)
            if not (150.0 <= pm <= 600.0):
                continue
            ce = extract_float(p["collision_energy_ev"][i], 30.0)
            ion = 1.0 if str(p["ionization_mode"][i]).lower().startswith("pos") else 0.0
            mzs = np.asarray(p["ms2_mzs"][i], dtype=np.float32)
            ints = np.asarray(p["ms2_normalized_intensities"][i], dtype=np.float32)
            raw_train_specs.append({
                "mzs": mzs, "ints": ints, "pm": pm, "ce": ce, "ion": ion,
                "smiles": smi, "inchikey14": p["inchikey14"][i]
            })
            if len(raw_train_specs) >= max_train_samples:
                break
        log(f"  Row group {rg}: {len(raw_train_specs):,} training spectra collected...")
        if len(raw_train_specs) >= max_train_samples:
            break

    log(f"Extracted {len(raw_train_specs):,} train spectra, {len(raw_val_specs):,} val spectra.")

    # Precompute fingerprints for unique SMILES
    all_smiles = list(set([s["smiles"] for s in raw_train_specs] + [s["smiles"] for s in raw_val_specs]))
    log(f"Precomputing fingerprints for {len(all_smiles):,} unique SMILES...")
    fp_cache = {}
    t_fp = time.time()
    for i, smi in enumerate(all_smiles):
        fp_cache[smi] = compute_compound_fingerprint(smi)
        if (i + 1) % 10000 == 0 or i + 1 == len(all_smiles):
            log(f"  {i+1}/{len(all_smiles)} fingerprints computed ({time.time()-t_fp:.1f}s)...")

    log("Building feature arrays...")
    def build_matrix(specs):
        X = np.zeros((len(specs), 4096 + 3), dtype=np.float32)
        Y = np.zeros((len(specs), 2048 + 166), dtype=np.float32)
        for idx, s in enumerate(specs):
            b = bin_spectrum(s["mzs"], s["ints"])
            X[idx, :4096] = b
            X[idx, 4096] = s["pm"] / 1000.0
            X[idx, 4097] = s["ce"] / 100.0
            X[idx, 4098] = s["ion"]
            Y[idx] = fp_cache[s["smiles"]]
        return X, Y

    X_train, Y_train = build_matrix(raw_train_specs)
    X_val, Y_val = build_matrix(raw_val_specs)

    log(f"Data ready: Train shape {X_train.shape}, Val shape {X_val.shape}")
    return X_train, Y_train, X_val, Y_val, raw_val_specs


# ── Training Loop ────────────────────────────────────────────────────────────

def train_model():
    log("=== Training Spectrum-to-Fingerprint Deep Network (FPNet) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    X_train, Y_train, X_val, Y_val, val_specs = prepare_data(max_train_samples=120_000)

    train_ds = SpectralDataset(X_train, Y_train)
    val_ds = SpectralDataset(X_val, Y_val)

    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    model = FPNet(in_dim=4099, hidden_dim=2048, out_dim=2214, dropout=0.2, num_res_blocks=2).to(device)
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 12
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler()

    best_val_loss = float("inf")
    best_weights_path = ROOT / "models" / "fpnet_weights.pt"
    best_weights_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = len(train_loader)

        t_epoch_start = time.time()
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad()
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        scheduler.step()
        avg_train_loss = train_loss / n_batches

        # Evaluate on validation
        model.eval()
        val_loss = 0.0
        cosine_sims = []
        with torch.no_grad():
            for x_v, y_v in val_loader:
                x_v = x_v.to(device)
                y_v = y_v.to(device)
                with autocast():
                    logits = model(x_v)
                    v_loss = criterion(logits, y_v)
                val_loss += v_loss.item()

                preds = torch.sigmoid(logits)
                # Compute cosine similarity between predicted and true fingerprint
                cos_sim = F.cosine_similarity(preds, y_v, dim=1)
                cosine_sims.extend(cos_sim.cpu().numpy().tolist())

        avg_val_loss = val_loss / len(val_loader)
        mean_cos = np.mean(cosine_sims)
        dt = time.time() - t_epoch_start

        log(f"Epoch {epoch:2d}/{epochs} [{dt:.1f}s] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val FP Cosine Sim: {mean_cos:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), str(best_weights_path))
            log(f"  ✓ Saved new best model checkpoint (Val Loss: {best_val_loss:.4f})")

    log(f"✓ Training finished. Best model saved to: {best_weights_path}")
    return model


if __name__ == "__main__":
    train_model()
