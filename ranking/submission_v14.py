"""
ranking/submission_v14.py

Version 14: Multi-Channel In-Silico Fragmentation (MetFrag-Lite) + Residual FPNet Ranker
Leaderboard Target: Breakthrough beyond 0.145 towards SOTA (0.339 - 0.451)

Key Breakthroughs in Version 14:
  1. In-Silico Fragment Peak Matching (MetFrag-Lite):
     Uses RDKit single-bond cleavage and neutral losses (-H2O, -CO, -CO2, -CH3) to generate
     theoretical fragment masses for top candidates. Computes explained MS2 peak intensity
     to break constitutional isomer ambiguity that limits pure neural models.
  2. Full 10-Adduct Exact Neutral Mass Table (100% test set coverage including
     [M+Na]+, [M+CH2O2-H]-, [M+NH4]+, [M+K]+, [M+Cl]-).
  3. Collision Energy Weighting: 1.3x weight for 15-45 eV diagnostic fragmentation spectra.
  4. Calibrated Gaussian ppm Mass Penalty (sigma = 10.0 ppm, 30% weight).
  5. Biological Reference Prior: 1.08x boost for candidate structures in 'train' source.
  6. Guaranteed Valid Kaggle Submission: exactly 400 rows, 2 columns ['molecule_id', 'smiles'],
     25 deduplicated valid SMILES per row. Saved to /kaggle/working/submission.csv.
"""

import os
import sys
import glob
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set

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
from rdkit.Chem import AllChem, MACCSkeys, Descriptors

# 1. Model Definition (Residual FPNet)
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
    def __init__(self, in_dim: int = 4099, hidden_dim: int = 2048, out_dim: int = 2214, dropout: float = 0.0, num_res_blocks: int = 2):
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

# 2. Fast Binned Spectrum Encoding
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

# 3. Cheminformatics & Fingerprinting
def compute_fingerprint_bits(smiles: str) -> np.ndarray:
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        arr_m = np.zeros(2048, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp_morgan, arr_m)
        fp_maccs = MACCSkeys.GenMACCSkeys(mol)
        arr_k = np.zeros(167, dtype=np.float32)
        AllChem.DataStructs.ConvertToNumpyArray(fp_maccs, arr_k)
        return np.concatenate([arr_m, arr_k[1:]])
    except Exception:
        return None

# 4. In-Silico Fragmentation (MetFrag-Lite)
def get_candidate_fragment_peaks(mol) -> np.ndarray:
    if mol is None:
        return np.array([], dtype=np.float32)
    try:
        mw = Descriptors.ExactMolWt(mol)
        raw = [mw, mw - 18.0106, mw - 27.9949, mw - 43.9898, mw - 15.0234]
        for b in mol.GetBonds():
            if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing():
                try:
                    frags = Chem.FragmentOnBonds(mol, [b.GetIdx()], addDummies=False)
                    for f in Chem.GetMolFrags(frags, asMols=True):
                        fm = Descriptors.ExactMolWt(f)
                        raw.extend([
                            fm,
                            fm + 1.0073,           # [M+H]+
                            fm - 1.0073,           # [M-H]-
                            fm + 1.0073 - 18.0106, # [M+H-H2O]+
                            fm + 1.0073 - 15.0234, # [M+H-CH3]+
                            fm + 1.0073 - 27.9949, # [M+H-CO]+
                            fm - 15.0234,          # -CH3
                            fm - 18.0106,          # -H2O
                        ])
                except Exception:
                    pass
        valid = [m for m in raw if 40.0 <= m <= mw + 5.0]
        if not valid:
            return np.array([], dtype=np.float32)
        return np.sort(np.unique(np.round(valid, 3))).astype(np.float32)
    except Exception:
        return np.array([], dtype=np.float32)

def score_explained_peaks(frag_arr: np.ndarray, exp_mzs: np.ndarray, exp_ints: np.ndarray, tol: float = 0.03) -> float:
    if len(frag_arr) == 0 or len(exp_mzs) == 0:
        return 0.0
    matched_int = 0.0
    total_int = float(np.sum(exp_ints)) + 1e-8
    for mz, intensity in zip(exp_mzs, exp_ints):
        idx = np.searchsorted(frag_arr, mz)
        if idx < len(frag_arr) and abs(frag_arr[idx] - mz) <= tol:
            matched_int += intensity
        elif idx > 0 and abs(frag_arr[idx - 1] - mz) <= tol:
            matched_int += intensity
    return float(matched_int / total_int)

# 5. Fast Batch Tanimoto Similarity
def fast_tanimoto_batch(query_fp: np.ndarray, cand_matrix: np.ndarray) -> np.ndarray:
    q_norm = np.dot(query_fp, query_fp)
    c_norms = np.sum(cand_matrix * cand_matrix, axis=1)
    dots = np.dot(cand_matrix, query_fp)
    denom = q_norm + c_norms - dots
    denom = np.maximum(denom, 1e-8)
    return dots / denom

# 6. Calibrated Gaussian PPM Mass Accuracy Penalty
def gaussian_mass_penalty(cand_masses: np.ndarray, neutral_mass: float, sigma_ppm: float = 10.0) -> np.ndarray:
    if neutral_mass <= 0:
        return np.ones(len(cand_masses), dtype=np.float32)
    ppm_diff = np.abs(cand_masses - neutral_mass) / neutral_mass * 1e6
    return np.exp(-0.5 * (ppm_diff / sigma_ppm) ** 2)

# Full 10-Adduct Neutral Mass Lookup Table (100% Test Set Coverage)
ADDUCT_SHIFTS = {
    "[M+H]+": -1.007276,
    "[M-H]-": +1.007276,
    "[M+Na]+": -22.989218,
    "[M+K]+": -38.963158,
    "[M+NH4]+": -18.033823,
    "[M-H2O+H]+": +17.003289,
    "[M+HCOO]-": -44.998201,
    "[M+CH2O2-H]-": -44.998201,
    "[M+CH3COO]-": -59.013851,
    "[M+Cl]-": -34.969402,
}

def extract_float(val, default: float = 30.0) -> float:
    if val is None:
        return default
    try:
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 0:
                return default
            return float(val[0])
        if pd.isna(val):
            return default
        if isinstance(val, (int, float)):
            return float(val)
        val_str = str(val).strip("[]() ")
        if not val_str:
            return default
        parts = val_str.split(",")
        return float(parts[0])
    except Exception:
        return default

def find_file(filename: str) -> str:
    possible_roots = [
        "/kaggle/input",
        "/kaggle/working",
        ".",
        "data",
        "models",
        "kaggle_artifacts",
    ]
    for root in possible_roots:
        matches = glob.glob(f"{root}/**/{filename}", recursive=True)
        if matches:
            return matches[0]
    return filename

# 7. Main Inference Pipeline
def run_kaggle_inference(output_csv: str = "submission.csv"):
    print("=" * 70)
    print("STARTING VERSION 14 ADVANCED INFERENCE PIPELINE")
    print("Features: In-Silico Fragmentation (MetFrag-Lite) + Residual FPNet")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # File paths
    test_path = find_file("test.parquet")
    weights_path = find_file("fpnet_weights.pt")
    cand_path = find_file("candidate_db.parquet")

    print(f"Test data:    {test_path}")
    print(f"Model weights:{weights_path}")
    print(f"Candidate DB: {cand_path}")

    # Load Model
    print("Loading FPNet model...")
    model = FPNet(in_dim=4099, hidden_dim=2048, out_dim=2214, dropout=0.0, num_res_blocks=2).to(device)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("✓ Model loaded successfully (2 residual blocks).")

    # Load Candidate DB
    print("Loading candidate database...")
    cand_tbl = pq.read_table(cand_path)
    all_smiles = cand_tbl["canonical_smiles"].to_pylist()
    all_masses = cand_tbl["exact_mass"].to_numpy()
    all_sources = cand_tbl["source"].to_pylist()

    sort_idx = np.argsort(all_masses)
    all_masses = all_masses[sort_idx]
    all_smiles = [all_smiles[i] for i in sort_idx]
    all_sources = [all_sources[i] for i in sort_idx]
    print(f"✓ Loaded {len(all_smiles):,} candidates ({all_masses[0]:.2f} - {all_masses[-1]:.2f} Da).")

    # Group test spectra by molecule_id
    print("Parsing test spectra...")
    tbl_test = pq.read_table(test_path)
    p_test = tbl_test.to_pydict()

    molecules: Dict[str, Dict] = {}
    for i in range(len(p_test["molecule_id"])):
        mol_id = p_test["molecule_id"][i]
        pm = float(p_test["precursor_mz"][i])
        adduct = str(p_test["adduct"][i])
        ion = str(p_test["ionization_mode"][i]).lower()
        ce = extract_float(p_test.get("collision_energy_ev", [None])[i], default=30.0)

        mzs = np.array(p_test["ms2_mzs"][i], dtype=np.float32)
        ints = np.array(p_test["ms2_normalized_intensities"][i], dtype=np.float32)

        if mol_id not in molecules:
            molecules[mol_id] = {
                "molecule_id": mol_id,
                "precursor_mz": pm,
                "adduct": adduct,
                "ion": ion,
                "spectra": []
            }
        molecules[mol_id]["spectra"].append({"mzs": mzs, "ints": ints, "pm": pm, "ce": ce})

    print(f"✓ Loaded {len(molecules)} test molecules.")

    # Predict and Rank
    t0 = time.time()
    submission_rows = []
    fp_cache: Dict[str, np.ndarray] = {}
    frag_cache: Dict[str, np.ndarray] = {}

    for idx, (mol_id, mol_info) in enumerate(molecules.items()):
        pm = mol_info["precursor_mz"]
        add = mol_info["adduct"]
        shift = ADDUCT_SHIFTS.get(add, -1.007276 if mol_info["ion"].startswith("pos") else +1.007276)
        neutral_mass = pm + shift

        # Window slice (15 ppm primary)
        delta = neutral_mass * 15e-6
        low = np.searchsorted(all_masses, neutral_mass - delta)
        high = np.searchsorted(all_masses, neutral_mass + delta)
        cand_pool = all_smiles[low:high]
        cand_mass_slice = all_masses[low:high]
        cand_source_slice = all_sources[low:high]

        # Expand window if sparse
        if len(cand_pool) < 25:
            delta = neutral_mass * 35e-6
            low = np.searchsorted(all_masses, neutral_mass - delta)
            high = np.searchsorted(all_masses, neutral_mass + delta)
            cand_pool = all_smiles[low:high]
            cand_mass_slice = all_masses[low:high]
            cand_source_slice = all_sources[low:high]

        if len(cand_pool) < 25:
            delta = neutral_mass * 70e-6
            low = np.searchsorted(all_masses, neutral_mass - delta)
            high = np.searchsorted(all_masses, neutral_mass + delta)
            cand_pool = all_smiles[low:high]
            cand_mass_slice = all_masses[low:high]
            cand_source_slice = all_sources[low:high]

        # Collect top diagnostic MS2 peaks across spectra for fragment matching
        all_exp_mzs = []
        all_exp_ints = []
        for s in mol_info["spectra"]:
            top_k = np.argsort(-s["ints"])[:20]
            all_exp_mzs.extend(s["mzs"][top_k])
            all_exp_ints.extend(s["ints"][top_k])
        all_exp_mzs = np.array(all_exp_mzs, dtype=np.float32)
        all_exp_ints = np.array(all_exp_ints, dtype=np.float32)

        # Predict FPNet fingerprint across spectra with Collision Energy Weighting
        pred_fps = []
        ce_weights = []
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

            # Diagnostic collision energy window weighting (15-45 eV)
            w = 1.3 if (15.0 <= s["ce"] <= 45.0) else 0.8
            ce_weights.append(w)

        ce_weights = np.array(ce_weights, dtype=np.float32)
        ce_weights /= np.sum(ce_weights)
        avg_fp = np.sum([p * w for p, w in zip(pred_fps, ce_weights)], axis=0)

        # Get candidate fingerprints
        valid_smis = []
        valid_masses = []
        valid_sources = []
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
                valid_sources.append(cand_source_slice[smi_idx])
                fps_list.append(fp)

        if len(valid_smis) > 0:
            cand_matrix = np.stack(fps_list)
            tan_scores = fast_tanimoto_batch(avg_fp, cand_matrix)

            # Calibrated Gaussian ppm mass accuracy penalty (sigma = 10.0 ppm)
            mass_penalties = gaussian_mass_penalty(np.array(valid_masses), neutral_mass, sigma_ppm=10.0)

            # Biological reference prior (1.08x boost for training library structures)
            train_priors = np.array([1.08 if src == "train" else 1.0 for src in valid_sources])

            # Base neural score
            base_scores = tan_scores * (0.70 + 0.30 * mass_penalties) * train_priors

            # Re-rank top 40 candidates using In-Silico Fragment Peak Matching (MetFrag-Lite)
            top_candidates_idx = np.argsort(-base_scores)[:40]
            frag_boosts = np.zeros(len(valid_smis), dtype=np.float32)
            for top_i in top_candidates_idx:
                smi = valid_smis[top_i]
                if smi not in frag_cache:
                    mol = Chem.MolFromSmiles(smi)
                    frag_cache[smi] = get_candidate_fragment_peaks(mol)
                frag_arr = frag_cache[smi]
                f_score = score_explained_peaks(frag_arr, all_exp_mzs, all_exp_ints, tol=0.03)
                frag_boosts[top_i] = f_score

            # Multi-Channel Composite Score
            combined_scores = base_scores * (1.0 + 0.40 * frag_boosts)
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

        # Backfill if < 25
        if len(final_smiles) < 25:
            for s in cand_pool:
                if s not in seen:
                    seen.add(s)
                    final_smiles.append(s)
                if len(final_smiles) >= 25:
                    break

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
