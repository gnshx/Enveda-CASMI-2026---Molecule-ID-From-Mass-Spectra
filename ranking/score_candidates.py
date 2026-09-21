"""
ranking/score_candidates.py

Phase 2 & 3: Candidate Scoring Pipeline for Enveda CASMI 2026.
1. Retrieves candidate pool from candidates/candidate_db.parquet (±15 ppm neutral mass window).
2. Predicts 2,214-bit functional fingerprints via FPNet.
3. Scores candidates via Continuous Tanimoto similarity + cosine similarity.
4. Blends with baseline spectral library cosine matches.
5. Formats exact top-25 SMILES for validation and submission.
"""

import os
import sys
import time
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numba
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
sys.path.insert(0, str(ROOT))

from ranking.fpnet import FPNet

# Proton mass (Da)
PROTON_MASS = 1.00727646688

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
            if out[i] > mx:
                mx = out[i]
    if mx > 0:
        for i in range(n_bins):
            out[i] /= mx
    return out


# ── Fingerprint Calculation ──────────────────────────────────────────────────
def compute_fingerprint_bits(smiles: str) -> Optional[np.ndarray]:
    """Compute 2214-bit binary array: 2048 Morgan + 166 MACCS."""
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


# ── Continuous Tanimoto Similarity ───────────────────────────────────────────
def fast_tanimoto_batch(query_fp: np.ndarray, cand_fps: np.ndarray) -> np.ndarray:
    """
    Vectorized Continuous Tanimoto similarity between query vector (D,)
    and candidate binary matrix (N, D).
    """
    dots = cand_fps @ query_fp  # shape (N,)
    q_sq = np.sum(query_fp ** 2)
    c_sq = np.sum(cand_fps ** 2, axis=1)  # shape (N,)
    denom = q_sq + c_sq - dots
    denom = np.maximum(denom, 1e-7)
    return dots / denom


# ── Candidate Database Manager ───────────────────────────────────────────────
class CandidateDB:
    def __init__(self, db_path: Path = ROOT / "candidates" / "candidate_db.parquet"):
        print(f"Loading candidate database from {db_path}...", flush=True)
        tbl = pq.read_table(str(db_path), columns=["canonical_smiles", "inchikey14", "exact_mass"])
        df = tbl.to_pandas().sort_values("exact_mass").reset_index(drop=True)
        self.smiles = df["canonical_smiles"].to_numpy()
        self.inchikey14 = df["inchikey14"].to_numpy()
        self.masses = df["exact_mass"].to_numpy()
        self.n_candidates = len(self.masses)
        self.fp_cache: Dict[str, np.ndarray] = {}
        print(f"Loaded {self.n_candidates:,} candidates (mass range: {self.masses[0]:.2f} - {self.masses[-1]:.2f} Da).", flush=True)

    def query_by_mass(self, target_mass: float, ppm_tol: float = 15.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Binary search for candidates within ±ppm_tol of target neutral mass."""
        delta = target_mass * ppm_tol * 1e-6
        low = np.searchsorted(self.masses, target_mass - delta, side="left")
        high = np.searchsorted(self.masses, target_mass + delta, side="right")
        return self.smiles[low:high], self.inchikey14[low:high], self.masses[low:high]

    def get_candidate_fps(self, smiles_list: np.ndarray) -> Tuple[List[str], List[str], np.ndarray]:
        """Retrieve or compute fingerprints for candidate pool."""
        valid_smiles = []
        fps = []
        for smi in smiles_list:
            if smi in self.fp_cache:
                fp = self.fp_cache[smi]
            else:
                fp = compute_fingerprint_bits(smi)
                if fp is not None:
                    self.fp_cache[smi] = fp
            if fp is not None:
                valid_smiles.append(smi)
                fps.append(fp)
        if len(fps) == 0:
            return [], [], np.empty((0, 2214), dtype=np.float32)
        return valid_smiles, np.stack(fps)


# ── Pipeline Scorer ──────────────────────────────────────────────────────────
class CandidateScorer:
    def __init__(self, model_path: Path = ROOT / "models" / "fpnet_weights.pt", device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        print(f"Loading FPNet from {model_path} onto {self.device}...", flush=True)
        self.model = FPNet(in_dim=4099, hidden_dim=2048, out_dim=2214, dropout=0.0, num_res_blocks=2).to(self.device)
        self.model.load_state_dict(torch.load(str(model_path), map_location=self.device))
        self.model.eval()
        self.cand_db = CandidateDB()

    def predict_spectrum_fp(self, mzs: np.ndarray, ints: np.ndarray, pm: float, ce: float, ion: float) -> np.ndarray:
        """Predict functional fingerprint vector from a single spectrum."""
        b = bin_spectrum(mzs, ints)
        feat = np.zeros((1, 4099), dtype=np.float32)
        feat[0, :4096] = b
        feat[0, 4096] = pm / 1000.0
        feat[0, 4097] = ce / 100.0
        feat[0, 4098] = ion

        with torch.no_grad():
            x = torch.from_numpy(feat).to(self.device)
            logits = self.model(x)
            pred = torch.sigmoid(logits).cpu().numpy()[0]
        return pred

    def rank_candidates(
        self,
        query_spectra: List[Dict],
        precursor_mz: float,
        adduct_type: str = "pos",
        library_hits: Optional[Dict[str, float]] = None,
        top_k: int = 25
    ) -> List[Tuple[str, float]]:
        """
        Full candidate ranking for a query molecule:
        1. Calculate neutral mass from precursor m/z
        2. Query candidate DB (expanding window if needed)
        3. Predict ensemble FPNet fingerprint across collision energies
        4. Compute Tanimoto similarity against candidate pool
        5. Blend with library cosine matches
        6. Return top-K SMILES with scores
        """
        # 1. Neutral mass calculation
        if adduct_type == "pos":
            target_neutral_mass = precursor_mz - PROTON_MASS
        else:
            target_neutral_mass = precursor_mz + PROTON_MASS

        # 2. Query candidate DB (start with 15 ppm, expand to 25 ppm if pool small)
        cand_smiles, cand_inchis, _ = self.cand_db.query_by_mass(target_neutral_mass, ppm_tol=15.0)
        if len(cand_smiles) < 25:
            cand_smiles, cand_inchis, _ = self.cand_db.query_by_mass(target_neutral_mass, ppm_tol=25.0)

        # If still empty, test sodium adduct [M+Na]+ (neutral mass = mz - 22.9892)
        if len(cand_smiles) == 0:
            target_na = precursor_mz - 22.989218
            cand_smiles, cand_inchis, _ = self.cand_db.query_by_mass(target_na, ppm_tol=20.0)

        if len(cand_smiles) == 0:
            # Fallback if no candidates found
            return []

        # 3. Predict average FPNet fingerprint across all spectra for this molecule
        pred_fps = []
        for spec in query_spectra:
            fp_pred = self.predict_spectrum_fp(
                spec["mzs"], spec["ints"],
                spec.get("pm", precursor_mz),
                spec.get("ce", 30.0),
                1.0 if adduct_type == "pos" else 0.0
            )
            pred_fps.append(fp_pred)

        avg_fp = np.mean(pred_fps, axis=0)

        # 4. Retrieve candidate fingerprints and compute Tanimoto
        valid_smiles, cand_matrix = self.cand_db.get_candidate_fps(cand_smiles)
        if len(valid_smiles) == 0:
            return []

        tanimoto_scores = fast_tanimoto_batch(avg_fp, cand_matrix)

        # 5. Blend with library hits if available
        scores = np.copy(tanimoto_scores)
        if library_hits:
            for idx, smi in enumerate(valid_smiles):
                if smi in library_hits:
                    lib_sim = library_hits[smi]
                    # Strong boost for high spectral cosine match
                    scores[idx] += 0.5 * lib_sim

        # 6. Sort and return top_k
        sorted_indices = np.argsort(-scores)[:top_k]
        return [(valid_smiles[i], float(scores[i])) for i in sorted_indices]
