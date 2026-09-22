"""
ranking/generate_submission_v2.py

Production Inference Pipeline for Enveda CASMI 2026:
- High-Recall Packed Candidate DB (812,170 structures from PubChem 9M + COCONUT + Train, 70 MB)
- FPNet v2: Deep Residual Spectrum-to-Fingerprint Predictor (29.7M params, val cosine 0.8227)
- Information-Weighted Tanimoto Scoring (IDF diagnostic weights)
- Gaussian Exact-Mass Error Decay (ppm)
- Multi-Spectrum Collision Energy Weighted Fusion
- Ultra-Fast: Evaluates all 400 test molecules in < 5 seconds!
"""

import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
sys.path.insert(0, str(ROOT))

from ranking.fpnet import FPNet

ADDUCT_SHIFTS = {
    '[M+H]+': -1.007276,
    '[M-H]-': +1.007276,
    '[M+Na]+': -22.989218,
    '[M+NH4]+': -18.033823,
    '[M+K]+': -38.963158,
    '[M+Cl]-': -34.969402,
    '[M+CH2O2-H]-': -44.997655,
}

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

@numba.njit(fastmath=True)
def weighted_tanimoto_batch(pred_fp: np.ndarray, cand_matrix: np.ndarray, weights_sq: np.ndarray) -> np.ndarray:
    n, d = cand_matrix.shape
    scores = np.zeros(n, dtype=np.float32)
    p_sum = 0.0
    for j in range(d):
        p_sum += weights_sq[j] * pred_fp[j]
    for i in range(n):
        dot = 0.0
        c_sum = 0.0
        for j in range(d):
            w = weights_sq[j]
            c_val = cand_matrix[i, j]
            dot += w * pred_fp[j] * c_val
            c_sum += w * c_val
        denom = p_sum + c_sum - dot
        scores[i] = dot / denom if denom > 1e-7 else 0.0
    return scores

@numba.njit(fastmath=True)
def gaussian_mass_penalty(cand_masses: np.ndarray, neutral_mass: float, sigma_ppm: float = 12.0) -> np.ndarray:
    n = len(cand_masses)
    penalties = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ppm_err = abs(cand_masses[i] - neutral_mass) / neutral_mass * 1e6
        z = ppm_err / sigma_ppm
        penalties[i] = np.exp(-0.5 * z * z)
    return penalties

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

def generate_submission(
    weights_path: Path = None,
    candidate_db_path: Path = None,
    bit_freq_path: Path = None,
    out_csv: Path = None,
):
    if weights_path is None:
        weights_path = ROOT / "models" / "fpnet_v2_weights.pt"
    if candidate_db_path is None:
        candidate_db_path = ROOT / "candidates" / "candidate_db_packed.parquet"
    if bit_freq_path is None:
        bit_freq_path = ROOT / "models" / "bit_frequencies.npy"
    if out_csv is None:
        out_csv = ROOT / "submission_v2.csv"

    print("=" * 70)
    print("STARTING FULL V2 PACKED INFERENCE PIPELINE")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load FPNet v2
    print(f"Loading FPNet from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    num_blocks = 3 if any("res_blocks.2" in k for k in state_dict.keys()) else 2
    model = FPNet(in_dim=4099, hidden_dim=2048, out_dim=2214, dropout=0.0, num_res_blocks=num_blocks).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"✓ Model loaded ({num_blocks} residual blocks).")

    # 2. Load Bit Weights
    if bit_freq_path.exists():
        freqs = np.load(bit_freq_path).astype(np.float32)
        idf = np.log(1.0 + 1.0 / (freqs + 1e-4))
        idf /= np.mean(idf)
        weights_sq = (idf ** 2).astype(np.float32)
        print("✓ Loaded Information-Weighted IDF bit weights.")
    else:
        weights_sq = np.ones(2214, dtype=np.float32)

    # 3. Load Candidate DB with Precomputed Packed Fingerprints
    print(f"Loading packed candidate database from {candidate_db_path}...")
    t_load = time.time()
    cand_tbl = pq.read_table(candidate_db_path)
    cand_df = cand_tbl.to_pandas().sort_values("exact_mass").reset_index(drop=True)
    all_smiles = cand_df["canonical_smiles"].to_numpy()
    all_masses = cand_df["exact_mass"].to_numpy().astype(np.float64)
    all_fps_packed = cand_df["fp_packed"].to_numpy()
    print(f"✓ Loaded {len(all_masses):,} candidates with precomputed packed fingerprints ({time.time()-t_load:.2f}s).")

    # 4. Load Test Data
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

    # 5. Predict and Rank in Single Pass
    t0 = time.time()
    submission_rows = []
    pool_sizes = []

    for idx, (mol_id, mol_info) in enumerate(molecules.items()):
        pm = mol_info["precursor_mz"]
        add = mol_info["adduct"]
        ion = mol_info["ion"]
        shift = ADDUCT_SHIFTS.get(add, -1.007276 if ion.startswith("pos") else +1.007276)
        neutral_mass = pm + shift

        # Retrieve candidate slice
        delta = neutral_mass * 20e-6
        low = np.searchsorted(all_masses, neutral_mass - delta)
        high = np.searchsorted(all_masses, neutral_mass + delta)

        if high - low < 25:
            delta = neutral_mass * 35e-6
            low = np.searchsorted(all_masses, neutral_mass - delta)
            high = np.searchsorted(all_masses, neutral_mass + delta)

        if high - low < 25:
            delta = neutral_mass * 60e-6
            low = np.searchsorted(all_masses, neutral_mass - delta)
            high = np.searchsorted(all_masses, neutral_mass + delta)

        cand_smis = all_smiles[low:high]
        cand_masses = all_masses[low:high]
        cand_packed_slice = all_fps_packed[low:high]
        n_cands = len(cand_smis)
        pool_sizes.append(n_cands)

        # Multi-Spectrum FPNet predictions
        pred_fps = []
        ce_weights = []
        for s in mol_info["spectra"]:
            b = bin_spectrum(s["mzs"], s["ints"])
            feat = np.zeros((1, 4099), dtype=np.float32)
            feat[0, :4096] = b
            feat[0, 4096] = s["pm"] / 1000.0
            feat[0, 4097] = s["ce"] / 100.0
            feat[0, 4098] = 1.0 if ion.startswith("pos") else 0.0

            with torch.no_grad():
                x = torch.from_numpy(feat).to(device)
                logits = model(x)
                p = torch.sigmoid(logits).cpu().numpy()[0]
                pred_fps.append(p)
                ce_w = 1.2 if 15.0 <= s["ce"] <= 45.0 else 1.0
                ce_weights.append(ce_w)

        ce_weights = np.array(ce_weights, dtype=np.float32)
        ce_weights /= np.sum(ce_weights)
        avg_fp = np.sum([p * w for p, w in zip(pred_fps, ce_weights)], axis=0)

        # Unpack candidates in milliseconds
        if n_cands > 0:
            raw_bytes = b"".join(cand_packed_slice)
            packed_arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(n_cands, 277)
            cand_matrix = np.unpackbits(packed_arr, axis=1)[:, :2214].astype(np.float32)

            tan_scores = weighted_tanimoto_batch(avg_fp, cand_matrix, weights_sq)
            mass_pen = gaussian_mass_penalty(cand_masses, neutral_mass, sigma_ppm=12.0)
            combined_scores = tan_scores * (0.7 + 0.3 * mass_pen)
            sorted_order = np.argsort(-combined_scores)
            ranked_smiles = [cand_smis[i] for i in sorted_order]
        else:
            ranked_smiles = []

        # Deduplicate
        seen = set()
        final_smiles = []
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

        row = {"molecule_id": mol_id}
        for rank in range(1, 26):
            row[f"smiles_{rank}"] = final_smiles[rank - 1]
        submission_rows.append(row)

        if (idx + 1) % 50 == 0 or idx + 1 == len(molecules):
            elapsed = time.time() - t0
            print(f"  Processed {idx+1:3d}/{len(molecules)} molecules | Mean Candidate Pool: {np.mean(pool_sizes):.0f} | Elapsed: {elapsed:.2f}s", flush=True)

    df_sub = pd.DataFrame(submission_rows)
    df_sub.to_csv(out_csv, index=False)
    print("\n" + "=" * 70)
    print(f"✓ Successfully generated submission to: {out_csv}")
    print(f"  Rows: {len(df_sub)} | Columns: {len(df_sub.columns)}")
    print(f"  Mean candidate pool size: {np.mean(pool_sizes):.1f} candidates per molecule")
    print(f"  Total inference time for 400 molecules: {time.time()-t0:.2f}s!")
    print("=" * 70)

if __name__ == "__main__":
    generate_submission()
