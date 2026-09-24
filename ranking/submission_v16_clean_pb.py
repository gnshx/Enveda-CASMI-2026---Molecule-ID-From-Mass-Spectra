"""
ranking/submission_v16_clean_pb.py

Direct restoration of Personal Best Version 10 (Leaderboard: 0.145).
Fixes the 1-bit MACCS offset bug and fragmentation noise that degraded V14 & V15 to 0.023.

Pipeline:
  1. FPNet (2048 Morgan + 166 MACCS bits) neural prediction per spectrum.
  2. Average predicted fingerprint across multiple collision energies.
  3. Precursor neutral mass candidate filtering with progressive ppm windows (±15, ±35, ±70 ppm).
  4. Exact MACCS bit indexing (arr[2048 + b] = 1.0 for b in GetOnBits()) strictly matching model training.
  5. Fast batch Tanimoto similarity.
  6. Calibrated Gaussian ppm mass penalty: combined = tanimoto * (0.75 + 0.25 * mass_penalties).
  7. Top 25 deduplicated ranking per molecule.
  8. Strictly generates 400 rows, 2 columns ('molecule_id', 'smiles').
"""

import os
import sys
import glob
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn as nn

try:
    import rdkit
except ImportError:
    whls = glob.glob("/kaggle/input/**/rdkit*.whl", recursive=True)
    if whls:
        os.system(f"pip install --no-index {whls[0]} -q")
    else:
        os.system("pip install rdkit -q")

from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

# 1. Model Definition (Identical to Version 10 / Version 5)
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
    def __init__(
        self,
        in_dim: int = 4096 + 3,
        hidden_dim: int = 2048,
        out_dim: int = 2048 + 166,
        dropout: float = 0.0,
        num_res_blocks: int = 2,
    ):
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

# 2. Spectral Binning (4096 bins, 50 to 1074 Da)
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
            if out[i] > mx:
                mx = out[i]
    if mx > 0:
        for i in range(n_bins):
            out[i] /= mx
    return out

# 3. Exact Fingerprint Encoding (STRICTLY matching training alignment)
def compute_fingerprint_bits(smiles: str):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        arr = np.zeros(2048 + 166, dtype=np.float32)
        fp_m = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        for b in fp_m.GetOnBits():
            arr[b] = 1.0
        fp_c = MACCSkeys.GenMACCSKeys(mol)
        for b in fp_c.GetOnBits():
            if 0 <= b < 166:
                arr[2048 + b] = 1.0
        return arr
    except Exception:
        return None

# 4. Fast Vectorized Tanimoto Batch
def fast_tanimoto_batch(query_fp: np.ndarray, cand_fps: np.ndarray) -> np.ndarray:
    dots = cand_fps @ query_fp
    q_sq = np.sum(query_fp ** 2)
    c_sq = np.sum(cand_fps ** 2, axis=1)
    denom = q_sq + c_sq - dots
    denom = np.maximum(denom, 1e-7)
    return dots / denom

# 5. Calibrated Gaussian ppm Mass Penalty (sigma = 15.0 ppm)
def gaussian_mass_penalty(cand_masses: np.ndarray, neutral_mass: float, sigma_ppm: float = 15.0) -> np.ndarray:
    diff_ppm = (cand_masses - neutral_mass) / neutral_mass * 1e6
    return np.exp(-0.5 * (diff_ppm / sigma_ppm) ** 2)

ADDUCT_SHIFTS = {
    '[M+H]+': -1.007276,
    '[M+NH4]+': -18.033823,
    '[M+Na]+': -22.989218,
    '[M+K]+': -38.963158,
    '[M-H2O+H]+': +17.003289,
    '[M-2H2O+H]+': +35.013854,
    '[M-H]-': +1.007276,
    '[M+Cl]-': -34.969402,
    '[M+CH2O2-H]-': -44.997655,
    '[M-H2O-H]-': +19.017841,
}

def extract_float(val, default=30.0) -> float:
    if val is None:
        return default
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return default
        return float(val[0])
    try:
        return float(val)
    except Exception:
        return default

# 6. Main Runner
def run_kaggle_inference(output_csv: str = "submission.csv"):
    print("\n" + "=" * 60)
    print("STARTING CLEAN PB SUBMISSION (VERSION 10 BASELINE - 0.145)")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # File discovery (robust across Kaggle folders and local paths)
    test_files = glob.glob("/kaggle/input/**/test.parquet", recursive=True) or glob.glob("**/data/test.parquet", recursive=True)
    cand_files = glob.glob("/kaggle/input/**/candidate_db.parquet", recursive=True) or glob.glob("**/candidate_db.parquet", recursive=True)
    weight_files = glob.glob("/kaggle/input/**/fpnet_weights.pt", recursive=True) or glob.glob("**/fpnet_weights.pt", recursive=True)

    if not test_files:
        raise FileNotFoundError("Could not find test.parquet")
    if not cand_files:
        raise FileNotFoundError("Could not find candidate_db.parquet")
    if not weight_files:
        raise FileNotFoundError("Could not find fpnet_weights.pt")

    print(f"  Test data:     {test_files[0]}")
    print(f"  Candidate DB:  {cand_files[0]}")
    print(f"  Weights:       {weight_files[0]}")

    # Load Model
    model = FPNet(in_dim=4099, hidden_dim=2048, out_dim=2214, dropout=0.0, num_res_blocks=2).to(device)
    model.load_state_dict(torch.load(weight_files[0], map_location=device))
    model.eval()
    print("✓ Model loaded successfully.")

    # Load Candidates
    print("Loading candidate database...")
    cand_tbl = pq.read_table(cand_files[0], columns=["canonical_smiles", "inchikey14", "exact_mass"])
    cand_df = cand_tbl.to_pandas().sort_values("exact_mass").reset_index(drop=True)
    all_smiles = cand_df["canonical_smiles"].to_numpy()
    all_masses = cand_df["exact_mass"].to_numpy()
    print(f"✓ Loaded {len(all_masses):,} candidates.")

    fp_cache: Dict[str, np.ndarray] = {}

    # Load Test Data
    print("Loading test data...")
    test_tbl = pq.read_table(test_files[0])
    p_test = test_tbl.to_pydict()

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

    print(f"✓ Loaded {len(molecules)} test molecules.")

    # Predict and Rank
    t0 = time.time()
    submission_rows = []

    for idx, (mol_id, mol_info) in enumerate(molecules.items()):
        pm = mol_info["precursor_mz"]
        add = mol_info["adduct"]
        shift = ADDUCT_SHIFTS.get(add, -1.007276 if mol_info["ion"].startswith("pos") else +1.007276)
        neutral_mass = pm + shift

        # Progressive Window Slice
        delta = neutral_mass * 15e-6
        low = np.searchsorted(all_masses, neutral_mass - delta)
        high = np.searchsorted(all_masses, neutral_mass + delta)
        cand_pool = all_smiles[low:high]
        cand_mass_slice = all_masses[low:high]

        if len(cand_pool) < 25:
            delta = neutral_mass * 35e-6
            low = np.searchsorted(all_masses, neutral_mass - delta)
            high = np.searchsorted(all_masses, neutral_mass + delta)
            cand_pool = all_smiles[low:high]
            cand_mass_slice = all_masses[low:high]

        if len(cand_pool) < 25:
            delta = neutral_mass * 70e-6
            low = np.searchsorted(all_masses, neutral_mass - delta)
            high = np.searchsorted(all_masses, neutral_mass + delta)
            cand_pool = all_smiles[low:high]
            cand_mass_slice = all_masses[low:high]

        # Predict FPNet fingerprint across spectra
        pred_fps = []
        for s in mol_info["spectra"]:
            b = bin_spectrum(s["mzs"], s["ints"])
            feat = np.zeros((1, 4099), dtype=np.float32)
            feat[0, :4096] = b
            feat[0, 4096] = s["pm"] / 1000.0
            feat[0, 4097] = s["ce"] / 100.0
            feat[0, 4098] = 1.0 if mol_info["ion"].startswith("pos") else 0.0

            with torch.no_grad():
                x = torch.from_numpy(feat).to(device)
                logits = model(x)
                p = torch.sigmoid(logits).cpu().numpy()[0]
                pred_fps.append(p)

        avg_fp = np.mean(pred_fps, axis=0)

        # Get candidate fingerprints
        valid_smis = []
        valid_masses = []
        fps_list = []
        for smi_idx, smi in enumerate(cand_pool):
            if smi in fp_cache:
                fp = fp_cache[smi]
            else:
                fp = compute_fingerprint_bits(smi)
                if fp is not None:
                    fp_cache[smi] = fp
            if fp is not None:
                valid_smis.append(smi)
                valid_masses.append(cand_mass_slice[smi_idx])
                fps_list.append(fp)

        if len(valid_smis) > 0:
            cand_matrix = np.stack(fps_list)
            tan_scores = fast_tanimoto_batch(avg_fp, cand_matrix)
            mass_penalties = gaussian_mass_penalty(np.array(valid_masses), neutral_mass, sigma_ppm=15.0)
            
            # Combine Tanimoto + Gaussian ppm accuracy penalty (boosts exact mass matches)
            combined_scores = tan_scores * (0.75 + 0.25 * mass_penalties)
            sorted_order = np.argsort(-combined_scores)
            ranked_smiles = [valid_smis[i] for i in sorted_order]
        else:
            ranked_smiles = []

        # Deduplicate
        seen = set()
        final_smiles = []
        for s in ranked_smiles:
            if s not in seen:
                seen.add(s)
                final_smiles.append(s)

        # Backfill if < 25 from candidate slice
        if len(final_smiles) < 25:
            for s in cand_pool:
                if s not in seen:
                    seen.add(s)
                    final_smiles.append(s)
                if len(final_smiles) >= 25:
                    break

        # Fallback if still < 25
        DEFAULT_FALLBACK = [
            "CCO", "CC(=O)O", "C1CCCCC1", "c1ccccc1", "Oc1ccccc1",
            "CC(C)O", "CC(=O)C", "c1ccncc1", "COC(=O)C", "CCNCC",
            "NCCO", "CCOCC", "c1cnccn1", "c1ncccn1", "c1ncc[nH]1",
            "C1CCOCC1", "C1CCNCC1", "CC(=O)N", "CSC", "CS(=O)(=O)C",
            "c1ccc2ccccc2c1", "OC(=O)c1ccccc1", "c1ccc(O)cc1", "c1ccc(N)cc1", "c1ccoc1"
        ]
        fb_idx = 0
        while len(final_smiles) < 25 and fb_idx < len(DEFAULT_FALLBACK):
            s = DEFAULT_FALLBACK[fb_idx]
            if s not in seen:
                seen.add(s)
                final_smiles.append(s)
            fb_idx += 1

        final_25 = final_smiles[:25]
        assert len(final_25) == 25, f"Expected 25 SMILES, got {len(final_25)}"

        # STRICTLY 2 COLUMNS: molecule_id, smiles
        submission_rows.append({"molecule_id": mol_id, "smiles": ";".join(final_25)})

        if (idx + 1) % 50 == 0 or idx + 1 == len(molecules):
            print(f"  Processed {idx+1}/{len(molecules)} molecules ({time.time()-t0:.1f}s)...", flush=True)

    # Output and Verification
    out_path = Path(output_csv)
    sub_df = pd.DataFrame(submission_rows)
    sub_df.to_csv(out_path, index=False)
    print(f"\n✓ Successfully saved submission to: {out_path}")
    print(f"  Row count: {len(sub_df)} | Columns: {list(sub_df.columns)}")
    assert len(sub_df) == len(molecules), "Row count mismatch!"
    assert list(sub_df.columns) == ["molecule_id", "smiles"], f"Columns mismatch: {list(sub_df.columns)}"
    assert sub_df["smiles"].isna().sum() == 0, "Found null SMILES!"
    print("✓ All checks passed! Guaranteed valid submission.")

if __name__ == "__main__":
    out_dir = Path("/kaggle/working")
    if not out_dir.exists():
        out_dir = Path(".")
    run_kaggle_inference(str(out_dir / "submission.csv"))
