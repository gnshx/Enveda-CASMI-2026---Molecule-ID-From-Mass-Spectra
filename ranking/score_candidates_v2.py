"""
ranking/score_candidates_v2.py

High-Performance Information-Weighted Candidate Scorer for CASMI 2026:
1. Inverse Document Frequency (IDF) Bit Weighting (CSI:FingerID / SIRIUS style).
2. Gaussian Exact-Mass Error Decay (ppm).
3. Fast Vectorized / Numba Batch Tanimoto over candidate pools.
4. Multi-Spectrum Collision Energy Aggregation.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numba
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")

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
    """
    Compute Information-Weighted continuous Tanimoto similarity for N candidates.
    pred_fp: (D,) float32 in [0, 1]
    cand_matrix: (N, D) float32 in {0, 1}
    weights_sq: (D,) float32 IDF weights squared
    """
    n, d = cand_matrix.shape
    scores = np.zeros(n, dtype=np.float32)
    
    # Precompute weighted sum of pred_fp
    p_sum = 0.0
    for j in range(d):
        p_sum += weights_sq[j] * pred_fp[j]
        
    for i in range(n):
        dot = 0.0
        c_sum = 0.0
        for j in range(d):
            w = weights_sq[j]
            c_val = cand_matrix[i, j]
            p_val = pred_fp[j]
            dot += w * p_val * c_val
            c_sum += w * c_val
            
        denom = p_sum + c_sum - dot
        if denom > 1e-7:
            scores[i] = dot / denom
        else:
            scores[i] = 0.0
            
    return scores

@numba.njit(fastmath=True)
def gaussian_mass_penalty(cand_masses: np.ndarray, neutral_mass: float, sigma_ppm: float = 12.0) -> np.ndarray:
    """Compute Gaussian mass error penalty for N candidates."""
    n = len(cand_masses)
    penalties = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ppm_err = abs(cand_masses[i] - neutral_mass) / neutral_mass * 1e6
        z = ppm_err / sigma_ppm
        penalties[i] = np.exp(-0.5 * z * z)
    return penalties

def compute_molecule_fingerprint(smiles: str) -> Optional[np.ndarray]:
    """2048-bit Morgan + 166-bit MACCS keys."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        target = np.zeros(2048 + 166, dtype=np.float32)
        fp_m = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        for bit in fp_m.GetOnBits():
            target[bit] = 1.0
        fp_c = MACCSkeys.GenMACCSKeys(mol)
        for bit in fp_c.GetOnBits():
            if 0 <= bit < 166:
                target[2048 + bit] = 1.0
        return target
    except Exception:
        return None

class CandidateScorerV2:
    def __init__(self, candidate_db_path: Path = None, bit_freq_path: Path = None):
        if candidate_db_path is None:
            candidate_db_path = ROOT / "candidates" / "candidate_db_v2.parquet"
            if not candidate_db_path.exists():
                candidate_db_path = ROOT / "candidates" / "candidate_db.parquet"
        if bit_freq_path is None:
            bit_freq_path = ROOT / "models" / "bit_frequencies.npy"

        print(f"Loading candidate database from {candidate_db_path}...")
        tbl = pq.read_table(candidate_db_path, columns=["canonical_smiles", "inchikey14", "exact_mass"])
        df = tbl.to_pandas().sort_values("exact_mass").reset_index(drop=True)
        self.all_smiles = df["canonical_smiles"].to_numpy()
        self.all_inchikeys = df["inchikey14"].to_numpy()
        self.all_masses = df["exact_mass"].to_numpy().astype(np.float64)
        print(f"✓ Loaded {len(self.all_masses):,} candidates ({self.all_masses[0]:.2f} - {self.all_masses[-1]:.2f} Da).")

        # Load or compute bit IDF weights
        if bit_freq_path.exists():
            print(f"Loading bit occurrence frequencies from {bit_freq_path}...")
            freqs = np.load(bit_freq_path).astype(np.float32)
            idf = np.log(1.0 + 1.0 / (freqs + 1e-4))
            # Normalize so mean weight is 1.0
            idf /= np.mean(idf)
            self.weights_sq = (idf ** 2).astype(np.float32)
        else:
            print("Bit frequencies not found; using uniform bit weights.")
            self.weights_sq = np.ones(2214, dtype=np.float32)

        self.fp_cache: Dict[str, np.ndarray] = {}

    def retrieve_candidates(self, neutral_mass: float, ppm_tol: float = 20.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Binary search candidate pool within ppm_tol."""
        delta = neutral_mass * ppm_tol * 1e-6
        low = np.searchsorted(self.all_masses, neutral_mass - delta)
        high = np.searchsorted(self.all_masses, neutral_mass + delta)
        
        # Adaptive expansion if too few candidates
        if high - low < 25:
            delta = neutral_mass * 35e-6
            low = np.searchsorted(self.all_masses, neutral_mass - delta)
            high = np.searchsorted(self.all_masses, neutral_mass + delta)
            
        if high - low < 25:
            delta = neutral_mass * 60e-6
            low = np.searchsorted(self.all_masses, neutral_mass - delta)
            high = np.searchsorted(self.all_masses, neutral_mass + delta)

        return self.all_smiles[low:high], self.all_masses[low:high], self.all_inchikeys[low:high]

    def score_candidate_pool(
        self,
        pred_fp: np.ndarray,
        neutral_mass: float,
        cand_smiles: np.ndarray,
        cand_masses: np.ndarray,
        sigma_ppm: float = 12.0,
    ) -> List[Tuple[str, float]]:
        """Compute Information-Weighted Tanimoto × Gaussian Mass Penalty."""
        if len(cand_smiles) == 0:
            return []

        valid_smis = []
        valid_masses = []
        fps_list = []

        for smi, mass in zip(cand_smiles, cand_masses):
            if smi in self.fp_cache:
                fp = self.fp_cache[smi]
            else:
                fp = compute_molecule_fingerprint(smi)
                if fp is not None:
                    self.fp_cache[smi] = fp
            if fp is not None:
                valid_smis.append(smi)
                valid_masses.append(mass)
                fps_list.append(fp)

        if len(valid_smis) == 0:
            return []

        cand_matrix = np.stack(fps_list)
        tanimoto_scores = weighted_tanimoto_batch(pred_fp, cand_matrix, self.weights_sq)
        mass_penalties = gaussian_mass_penalty(np.array(valid_masses, dtype=np.float64), neutral_mass, sigma_ppm=sigma_ppm)

        # Composite score
        combined_scores = tanimoto_scores * (0.7 + 0.3 * mass_penalties)
        sorted_order = np.argsort(-combined_scores)

        return [(valid_smis[idx], float(combined_scores[idx])) for idx in sorted_order]
