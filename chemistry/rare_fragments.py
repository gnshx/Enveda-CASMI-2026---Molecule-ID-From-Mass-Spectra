"""
chemistry/rare_fragments.py
=============================================================================
CASMI 2026 — Strategy D: Inverse-Frequency Rare-Fragment Weighting

Computes document frequency df(f) of MS/MS fragment peaks (0.02 Da bins)
and neutral losses (prec_mz - mz, 0.02 Da bins) conditioned on ionization mode
from `train_cv` spectra (strictly excluding holdout molecules), yielding IDF:
    w(f) = log((N + 1) / (df(f) + 1))

Provides rare-fragment-weighted spectral similarity and rare-fragment-weighted
explained intensity for candidate ranking in V45/V48.
=============================================================================
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


class RareFragmentTable:
    """Mode-conditioned IDF weights for fragment m/z and neutral loss bins."""

    def __init__(self, bin_width: float = 0.02, max_mz: float = 1000.0):
        self.bin_width = bin_width
        self.max_mz = max_mz
        self.n_bins = int(max_mz / bin_width)
        self.n_docs = {"pos": 0, "neg": 0}
        self.frag_idf: Dict[str, np.ndarray] = {}
        self.nl_idf: Dict[str, np.ndarray] = {}

    def fit_from_train_cv(
        self,
        train_parquet: Path,
        allowed_smiles: set,
        max_spectra: int = 250000,
    ) -> Dict[str, int]:
        frag_df = {"pos": np.zeros(self.n_bins, dtype=np.int64), "neg": np.zeros(self.n_bins, dtype=np.int64)}
        nl_df = {"pos": np.zeros(self.n_bins, dtype=np.int64), "neg": np.zeros(self.n_bins, dtype=np.int64)}
        n_docs = {"pos": 0, "neg": 0}

        pf = pq.ParquetFile(str(train_parquet))
        cols = ["normalized_smiles", "ionization_mode", "precursor_mz", "ms2_mzs", "ms2_normalized_intensities"]
        seen = 0
        for rg in range(pf.num_row_groups):
            if seen >= max_spectra:
                break
            df = pf.read_row_group(rg, columns=cols).to_pandas()
            df = df[df["normalized_smiles"].isin(allowed_smiles)]
            if len(df) == 0:
                continue
            for row in df.itertuples(index=False):
                if seen >= max_spectra:
                    break
                mode = "pos" if "pos" in str(row.ionization_mode).lower() else "neg"
                prec = float(row.precursor_mz)
                mzs = np.asarray(row.ms2_mzs, dtype=np.float64)
                its = np.asarray(row.ms2_normalized_intensities, dtype=np.float64)
                ok = (mzs >= 20.0) & (mzs <= prec - 1.5) & (mzs < self.max_mz) & (its >= 0.01)
                if not ok.any():
                    continue
                m = mzs[ok]
                fb = np.unique(np.clip((m / self.bin_width).astype(np.int32), 0, self.n_bins - 1))
                frag_df[mode][fb] += 1

                nl = prec - m
                nl_ok = (nl >= 12.0) & (nl < self.max_mz)
                if nl_ok.any():
                    nb = np.unique(np.clip((nl[nl_ok] / self.bin_width).astype(np.int32), 0, self.n_bins - 1))
                    nl_df[mode][nb] += 1

                n_docs[mode] += 1
                seen += 1

        self.n_docs = n_docs
        for mode in ("pos", "neg"):
            nd = max(n_docs[mode], 1)
            self.frag_idf[mode] = np.log((nd + 1.0) / (frag_df[mode] + 1.0)).astype(np.float32)
            self.nl_idf[mode] = np.log((nd + 1.0) / (nl_df[mode] + 1.0)).astype(np.float32)
        return n_docs

    def peak_weights(
        self,
        mzs: np.ndarray,
        its: np.ndarray,
        prec_mz: float,
        mode: str,
    ) -> np.ndarray:
        """Return sqrt(it) * (1 + 0.35 * max(frag_idf, nl_idf)) normalized."""
        m = np.asarray(mzs, dtype=np.float64)
        it = np.asarray(its, dtype=np.float64)
        w = np.sqrt(np.maximum(it, 0.0))
        m_key = "pos" if str(mode).lower().startswith("pos") or str(mode) == "1" else "neg"
        f_tab = self.frag_idf.get(m_key)
        n_tab = self.nl_idf.get(m_key)
        if f_tab is not None and n_tab is not None and len(m) > 0:
            fb = np.clip((m / self.bin_width).astype(np.int32), 0, self.n_bins - 1)
            idf_f = f_tab[fb]
            nl = np.clip(prec_mz - m, 0.0, self.max_mz - 1e-3)
            nb = np.clip((nl / self.bin_width).astype(np.int32), 0, self.n_bins - 1)
            idf_n = np.where((prec_mz - m) >= 12.0, n_tab[nb], 0.0)
            idf = np.maximum(idf_f, idf_n)
            # Normalize IDF relative to log(N) ~ 11 so rare fragments stand out smoothly
            w = w * (0.5 + idf / max(math.log(max(self.n_docs[m_key], 2)), 1.0))
        norm = float(np.linalg.norm(w))
        return (w / norm).astype(np.float32) if norm > 1e-12 else w.astype(np.float32)

    def rare_weighted_bin_vector(
        self,
        mzs: np.ndarray,
        its: np.ndarray,
        prec_mz: float,
        mode: str,
        coarse_bin: float = 0.05,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build IDF-weighted sparse 0.05 Da fragment + neutral-loss vector."""
        ok = (mzs >= 20.0) & (mzs <= prec_mz - 1.5) & (mzs < self.max_mz) & (its > 0)
        if not ok.any():
            return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
        m = mzs[ok]
        it = its[ok]
        pw = self.peak_weights(m, it, prec_mz, mode)

        n_c = int(self.max_mz / coarse_bin)
        fb = np.clip((m / coarse_bin).astype(np.int32), 0, n_c - 1)
        nl_m = prec_mz - m
        nl_ok = (nl_m >= 12.0) & (nl_m < self.max_mz)
        nb = n_c + np.clip((nl_m[nl_ok] / coarse_bin).astype(np.int32), 0, n_c - 1)

        idx = np.concatenate([fb, nb])
        vals = np.concatenate([pw, 0.7 * pw[nl_ok]])
        u_idx, inv = np.unique(idx, return_inverse=True)
        agg = np.zeros(len(u_idx), dtype=np.float32)
        np.maximum.at(agg, inv, vals.astype(np.float32))
        norm = float(np.linalg.norm(agg))
        if norm > 1e-12:
            agg /= norm
        return u_idx, agg

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "RareFragmentTable":
        with open(path, "rb") as f:
            return pickle.load(f)
