"""
ranking/generate_submission_v3.py

Champion V3 Inference & Submission Generator for Enveda CASMI 2026.
Unites:
  1. Tier 1 Direct Spectral Library Matcher (promotes train.parquet hits to Rank 1)
  2. FPNet-v2 (Residual MLP, 250k spectra)
  3. Precomputed Packed Fingerprint Database (812,170 molecules, 100% mass coverage)
  4. Information-Weighted (IDF) Tanimoto Metric
  5. Gaussian Precursor Mass Penalty (12 ppm)
  6. Bayesian Natural Product Prior (COCONUT & Train Scaffold Enrichment)
"""

import sys
import time
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent.parent

# --- 1. Fast Numba/Numpy Weighted Tanimoto & Gaussian Penalty ---
def weighted_tanimoto_batch(query_fp: np.ndarray, cand_fps: np.ndarray, weights_sq: np.ndarray) -> np.ndarray:
    """Computes Information-Weighted Tanimoto similarity over packed candidate batch."""
    q_w = query_fp * weights_sq
    intersection = np.dot(cand_fps, q_w)
    cand_norm_sq = np.dot(cand_fps, weights_sq)
    query_norm_sq = np.sum(query_fp * q_w)
    union = query_norm_sq + cand_norm_sq - intersection
    return np.clip(intersection / np.maximum(union, 1e-7), 0.0, 1.0)

def gaussian_mass_penalty(cand_masses: np.ndarray, neutral_mass: float, sigma_ppm: float = 12.0) -> np.ndarray:
    """Computes Gaussian ppm penalty around target neutral mass."""
    diff_ppm = (cand_masses - neutral_mass) / neutral_mass * 1e6
    return np.exp(-0.5 * (diff_ppm / sigma_ppm) ** 2)

# --- 2. Model Architecture ---
class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int = 2048, dropout: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))

class FPNet(nn.Module):
    def __init__(self, in_dim: int = 4099, hidden_dim: int = 2048, out_dim: int = 2214, dropout: float = 0.0, num_res_blocks: int = 3):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim, dropout) for _ in range(num_res_blocks)
        ])
        self.head = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.in_proj(x)
        for block in self.res_blocks:
            h = block(h)
        return self.head(h)

# --- 3. Feature Extraction ---
ADDUCT_DELTAS = {
    "[M+H]+": 1.007276,
    "[M+Na]+": 22.989218,
    "[M+K]+": 38.963158,
    "[M+NH4]+": 18.033823,
    "[M-H]-": -1.007276,
    "[M+Cl]-": 34.969402,
    "[M+FA-H]-": 44.998201,
}

def get_neutral_mass(precursor_mz: float, adduct: str, ion_mode: str) -> float:
    if adduct in ADDUCT_DELTAS:
        return max(10.0, precursor_mz - ADDUCT_DELTAS[adduct])
    if ion_mode == "positive":
        return max(10.0, precursor_mz - 1.007276)
    else:
        return max(10.0, precursor_mz + 1.007276)

def extract_features(mzs: np.ndarray, ints: np.ndarray, pm: float, ce: float, is_pos: float) -> np.ndarray:
    feat = np.zeros(4099, dtype=np.float32)
    idx = (mzs / 0.5).astype(int)
    valid = (idx >= 0) & (idx < 2048)
    for i, v in zip(idx[valid], ints[valid]):
        feat[i] = max(feat[i], v)
    nl = pm - mzs
    nl_idx = (nl / 0.5).astype(int)
    nl_valid = (nl_idx >= 0) & (nl_idx < 2048)
    for i, v in zip(nl_idx[nl_valid], ints[nl_valid]):
        feat[2048 + i] = max(feat[2048 + i], v)
    feat[4096] = pm / 1000.0
    feat[4097] = ce / 100.0
    feat[4098] = is_pos
    return feat

def extract_float(val, default: float = 0.0) -> float:
    if val is None:
        return default
    if hasattr(val, '__iter__') and not isinstance(val, (str, bytes)):
        return float(val[0]) if len(val) > 0 else default
    return float(val)

def main():
    print("=" * 70)
    print("STARTING CHAMPION V3 PIPELINE (HYBRID LIBRARY + NEURAL BAYES)")
    print("=" * 70)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Load Tier 1 Library Matches
    lib_matches_path = ROOT / "models" / "library_matches.json"
    tier1_matches = {}
    if lib_matches_path.exists():
        with open(lib_matches_path) as f:
            all_matches = json.load(f)
            # Only keep high confidence matches (Cosine >= 0.82)
            tier1_matches = {k: v for k, v in all_matches.items() if v.get("cosine", 0.0) >= 0.82}
        print(f"✓ Loaded {len(tier1_matches)} high-confidence Tier 1 matches from {lib_matches_path.name}")
    else:
        print("! Warning: library_matches.json not found, running pure neural mode.")
        
    # 2. Load FPNet v2
    model_path = ROOT / "models" / "fpnet_v2_weights.pt"
    print(f"Loading FPNet from {model_path}...")
    model = FPNet(in_dim=4099, hidden_dim=2048, out_dim=2214, dropout=0.0, num_res_blocks=3).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print("✓ Model loaded (3 residual blocks).")

    # 3. Load Bit Frequencies (IDF Weights)
    freq_path = ROOT / "models" / "bit_frequencies.npy"
    if freq_path.exists():
        bit_freqs = np.load(freq_path)
        weights = 1.0 / (np.sqrt(bit_freqs) + 0.05)
        weights /= np.mean(weights)
        print("✓ Loaded Information-Weighted IDF bit weights.")
    else:
        weights = np.ones(2214, dtype=np.float32)
    weights_sq = (weights ** 2).astype(np.float32)

    # 4. Load Precomputed Packed Candidate Database
    cand_path = ROOT / "candidates" / "candidate_db_packed.parquet"
    print(f"Loading packed candidate database from {cand_path}...")
    t_load = time.time()
    cand_tbl = pq.read_table(cand_path)
    all_smiles = cand_tbl["canonical_smiles"].to_pylist()
    all_masses = cand_tbl["exact_mass"].to_numpy()
    all_sources = cand_tbl["source"].to_pylist()
    packed_raw = cand_tbl["fp_packed"].to_numpy()
    all_packed = np.frombuffer(b"".join(packed_raw), dtype=np.uint8).reshape(len(all_masses), 277)
    
    # Pre-index sorted masses for fast binary search
    sort_order = np.argsort(all_masses)
    sorted_masses = all_masses[sort_order]
    sorted_packed = all_packed[sort_order]
    sorted_smiles = [all_smiles[i] for i in sort_order]
    sorted_sources = [all_sources[i] for i in sort_order]
    
    # Compute Bayesian Prior Array: Natural products (COCONUT & Train) get a prior boost
    source_priors = np.ones(len(sorted_sources), dtype=np.float32)
    for i, s in enumerate(sorted_sources):
        if s == "train":
            source_priors[i] = 1.25   # 25% boost for verified train molecules
        elif s == "coconut":
            source_priors[i] = 1.20   # 20% boost for verified COCONUT natural products
    print(f"✓ Loaded {len(all_masses):,} candidates with packed fingerprints & Bayesian priors ({time.time()-t_load:.2f}s).")

    # 5. Load Test Data
    test_path = ROOT / "data" / "test.parquet"
    print(f"Loading test data from {test_path}...")
    tbl_test = pq.read_table(test_path)
    p_test = tbl_test.to_pydict()

    molecules: Dict[str, Dict] = {}
    for i in range(len(p_test["molecule_id"])):
        mol_id = p_test["molecule_id"][i]
        pm = extract_float(p_test["precursor_mz"][i], 300.0)
        ce = extract_float(p_test["collision_energy_ev"][i], 30.0)
        add = str(p_test["adduct"][i])
        ion = str(p_test["ionization_mode"][i]).lower()
        mzs = np.asarray(p_test["ms2_mzs"][i], dtype=np.float32)
        ints = np.asarray(p_test["ms2_normalized_intensities"][i], dtype=np.float32)

        if mol_id not in molecules:
            molecules[mol_id] = {
                "molecule_id": mol_id,
                "precursor_mz": pm,
                "adduct": add,
                "ion": ion,
                "spectra": []
            }
        molecules[mol_id]["spectra"].append({"mzs": mzs, "ints": ints, "pm": pm, "ce": ce})

    print(f"✓ Loaded {len(molecules)} unique test molecules.")

    # 6. Predict, Rank, and Combine in Single Pass
    t0 = time.time()
    submission_rows = []
    pool_sizes = []
    tier1_count = 0

    for idx, (mol_id, mol_info) in enumerate(molecules.items()):
        neutral_mass = get_neutral_mass(mol_info["precursor_mz"], mol_info["adduct"], mol_info["ion"])

        # Average predicted fingerprint across all collision energy spectra
        feats = []
        for s in mol_info["spectra"]:
            feat = extract_features(
                s["mzs"], s["ints"], s["pm"], s["ce"],
                1.0 if mol_info["ion"].startswith("pos") else 0.0
            )
            feats.append(feat)

        feats_tensor = torch.from_numpy(np.stack(feats)).to(device)
        with torch.no_grad():
            logits = model(feats_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()
        avg_fp = np.mean(probs, axis=0)

        # Fast mass window candidate slice
        mass_tol = max(0.5, neutral_mass * 15e-6)
        low_idx = np.searchsorted(sorted_masses, neutral_mass - mass_tol)
        high_idx = np.searchsorted(sorted_masses, neutral_mass + mass_tol)

        cand_masses = sorted_masses[low_idx:high_idx]
        cand_smis = sorted_smiles[low_idx:high_idx]
        cand_priors = source_priors[low_idx:high_idx]
        packed_arr = sorted_packed[low_idx:high_idx]
        pool_sizes.append(len(cand_smis))

        if len(cand_smis) > 0:
            cand_matrix = np.unpackbits(packed_arr, axis=1)[:, :2214].astype(np.float32)

            tan_scores = weighted_tanimoto_batch(avg_fp, cand_matrix, weights_sq)
            mass_pen = gaussian_mass_penalty(cand_masses, neutral_mass, sigma_ppm=12.0)
            
            # Combine Tanimoto + Mass Penalty + Bayesian Prior
            combined_scores = tan_scores * (0.65 + 0.35 * mass_pen) * cand_priors
            sorted_order = np.argsort(-combined_scores)
            ranked_smiles = [cand_smis[i] for i in sorted_order]
        else:
            ranked_smiles = []

        # Deduplicate candidates
        seen = set()
        final_smiles = []
        
        # --- TIER 1 DIRECT INJECTION ---
        # If this test molecule matched train.parquet with high cosine, place it at Rank 1!
        if mol_id in tier1_matches:
            t1_smi = tier1_matches[mol_id]["smiles"]
            final_smiles.append(t1_smi)
            seen.add(t1_smi)
            tier1_count += 1

        for s in ranked_smiles:
            if s not in seen:
                seen.add(s)
                final_smiles.append(s)

        # Backfill if < 25
        if len(final_smiles) < 25:
            for s in cand_smis:
                if s not in seen:
                    seen.add(s)
                    final_smiles.append(s)
                if len(final_smiles) >= 25:
                    break

        if len(final_smiles) < 25:
            for s in all_smiles[:200]:
                if s not in seen:
                    seen.add(s)
                    final_smiles.append(s)
                if len(final_smiles) >= 25:
                    break

        final_25 = final_smiles[:25]
        assert len(final_25) == 25, f"Molecule {mol_id} only got {len(final_25)} candidates!"

        row = {
            "molecule_id": mol_id,
            "smiles": ";".join(final_25)
        }
        submission_rows.append(row)

        if (idx + 1) % 50 == 0 or idx + 1 == len(molecules):
            elapsed = time.time() - t0
            print(f"  Processed {idx+1:3d}/{len(molecules)} molecules | Mean Pool: {np.mean(pool_sizes):.0f} | Tier 1 Promoted: {tier1_count} | Elapsed: {elapsed:.2f}s", flush=True)

    out_csv = ROOT / "submission_v3.csv"
    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(out_csv, index=False)
    print("\n" + "=" * 70)
    print(f"✓ Successfully generated Champion V3 submission to: {out_csv}")
    print(f"  Rows: {len(df_sub)} | Columns: {len(df_sub.columns)}")
    print(f"  Total Tier 1 Matches injected at Rank 1: {tier1_count}/{len(molecules)} ({(tier1_count/len(molecules))*100:.1f}%)")
    print(f"  Mean candidate pool size: {np.mean(pool_sizes):.1f} candidates per molecule")
    print(f"  Total inference time: {time.time()-t0:.2f}s!")
    print("=" * 70)

if __name__ == "__main__":
    main()
