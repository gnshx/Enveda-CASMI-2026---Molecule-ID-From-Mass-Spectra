"""
chemistry/v47_v48_formula_ce.py
=============================================================================
CASMI 2026 — V47 & V48: Multi-Adduct Formula Consensus, Subformula Peak
Consistency, Rare-Fragment Subformula Weighting & CE-Weighted Spectral Pooling.
=============================================================================
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Optional, Tuple

import numpy as np

from chemistry.ms_engine import ADDUCT_OFFSETS, PROTON, TIMSTOF_OFFSET_PPM, ce_information_weight
from chemistry.rare_fragments import RareFragmentTable

ATOMIC_MASSES: Dict[str, float] = {
    "C": 12.0000000,
    "H": 1.007825032,
    "N": 14.003074005,
    "O": 15.994914620,
    "P": 30.973761998,
    "S": 31.972071174,
    "F": 18.998403163,
    "Cl": 34.968852682,
    "Br": 78.918337600,
    "I": 126.904471900,
}

_FORM_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


def parse_formula(formula: str) -> Optional[Dict[str, int]]:
    if not formula or any(c in formula for c in "+-."):
        return None
    counts: Dict[str, int] = {}
    pos = 0
    for m in _FORM_TOKEN.finditer(formula):
        if m.start() != pos:
            return None
        el, cnt = m.group(1), m.group(2)
        if el not in ATOMIC_MASSES:
            return None
        counts[el] = counts.get(el, 0) + (int(cnt) if cnt else 1)
        pos = m.end()
    return counts if pos == len(formula) and counts else None


def enumerate_subformula_masses(counts: Dict[str, int], max_mass: float = 950.0) -> np.ndarray:
    """
    Enumerate sorted unique neutral monoisotopic masses of all valid elemental
    subformulas <= parent formula counts (with fast heavy-atom branching).
    """
    heavy_elems = [(el, cnt, ATOMIC_MASSES[el]) for el, cnt in counts.items() if el != "H" and cnt > 0]
    h_max = counts.get("H", 0)
    total_heavy = sum(cnt for _, cnt, _ in heavy_elems)
    # Cap combinatorics for giant exotic polymers while covering 99.9% of small NPs
    if total_heavy > 65 or len(heavy_elems) > 7:
        heavy_elems = sorted(heavy_elems, key=lambda x: -x[1])[:6]

    heavy_masses = [0.0]
    for el, max_c, amass in heavy_elems:
        new_m = []
        for c in range(max_c + 1):
            add = c * amass
            if add > max_mass:
                break
            for base in heavy_masses:
                val = base + add
                if val <= max_mass:
                    new_m.append(val)
        # Keep unique to 0.1 mDa resolution to prevent blowup
        if len(new_m) > 25000:
            arr = np.unique(np.round(np.asarray(new_m, dtype=np.float64), 4))
            heavy_masses = arr.tolist()
        else:
            heavy_masses = new_m

    hm_arr = np.unique(np.round(np.asarray(heavy_masses, dtype=np.float64), 5))
    h_shifts = np.arange(h_max + 1, dtype=np.float64) * ATOMIC_MASSES["H"]
    all_m = (hm_arr[:, None] + h_shifts[None, :]).ravel()
    all_m = all_m[(all_m >= 12.0) & (all_m <= max_mass)]
    return np.sort(np.unique(np.round(all_m, 5)))


def score_formula_against_spectrum(
    sub_masses: np.ndarray,
    mzs: np.ndarray,
    its: np.ndarray,
    prec_mz: float,
    mode: int,
    rare_tbl: Optional[RareFragmentTable] = None,
    tol_da: float = 0.008,
) -> Tuple[float, float, float]:
    """
    Match top MS/MS peaks against subformula neutral masses (allowing +/- 2 H shifts
    plus proton carrier). Returns (peak_frac, int_frac, rare_int_frac).
    """
    if len(sub_masses) == 0 or len(mzs) == 0:
        return 0.0, 0.0, 0.0
    ok = (mzs >= 20.0) & (mzs <= prec_mz - 1.5) & (its > 0)
    if not ok.any():
        return 0.0, 0.0, 0.0
    m = mzs[ok]
    it = its[ok]
    if len(m) > 60:
        top = np.argpartition(-it, 60)[:60]
        m, it = m[top], it[top]

    carrier = PROTON if mode >= 0 else -PROTON
    # Query neutral masses under h_shift in (-1, 0, +1, +2)
    neutral_q = np.stack([m - carrier - dh * ATOMIC_MASSES["H"] for dh in (-1, 0, 1, 2)], axis=0)
    lo = np.searchsorted(sub_masses, neutral_q - tol_da, side="left")
    hi = np.searchsorted(sub_masses, neutral_q + tol_da, side="right")
    matched = (hi > lo).any(axis=0)

    peak_frac = float(matched.mean())
    it_sum = float(it.sum())
    int_frac = float(it[matched].sum() / it_sum) if it_sum > 0 else 0.0

    if rare_tbl is not None:
        rw = rare_tbl.peak_weights(m, it, prec_mz, "pos" if mode >= 0 else "neg")
        rw_sum = float(rw.sum())
        rare_int_frac = float(rw[matched].sum() / rw_sum) if rw_sum > 0 else 0.0
    else:
        rare_int_frac = int_frac

    return peak_frac, int_frac, rare_int_frac


def compute_ce_weights_for_spectra(spectra_frag: List[dict], spectra_ces: List[float]) -> np.ndarray:
    """Phase 36 CE-aware * sqrt(n_peaks) normalized spectral weights."""
    raw = []
    for sp, ce in zip(spectra_frag, spectra_ces):
        n_pk = max(len(sp.get("mz", [])), 1)
        raw.append(ce_information_weight(ce) * math.sqrt(n_pk))
    arr = np.asarray(raw, dtype=np.float32)
    s = float(arr.sum())
    return (arr / s) if s > 0 else np.full(len(spectra_frag), 1.0 / max(len(spectra_frag), 1), dtype=np.float32)
