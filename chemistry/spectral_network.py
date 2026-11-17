"""
chemistry/spectral_network.py
=============================================================================
CASMI 2026 Phase 29/30: In-Test Molecular Networking Engine
Builds an empirical spectral similarity graph across test queries using:
  1. Stepped multi-CE pooled MS/MS spectra (chemistry/ms_engine.py).
  2. Modified cosine calculation with neutral mass shift (GNPS standard).
  3. Strict adduct stratification (only queries with identical adduct are paired).
  4. Scaffold-homology certified thresholds (tau >= 0.70, matching peaks >= 6).
=============================================================================
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from chemistry.ms_engine import pool_multice_spectra


def prepare_binned_spectrum(
    specs: List[Dict],
    max_peaks: int = 120,
    bin_decimals: int = 2,
    tol_da: float = 0.010
) -> Optional[Dict]:
    """
    Pools multi-CE spectra and produces a normalized, binned spectrum for fast modified cosine.
    Returns dict with keys: 'u' (unique mz bins), 'w' (normalized sqrt intensity), 'prec', 'adduct'.
    """
    if not specs:
        return None

    pooled = pool_multice_spectra(specs, tol_da=tol_da)
    if not pooled:
        return None

    p0 = pooled[0]
    pm = float(p0["prec"])
    add = str(p0["adduct"])
    mz = np.asarray(p0["mz"], dtype=np.float64)
    it = np.asarray(p0["it"], dtype=np.float64)

    # Exclude precursor and unfragmented ions within 1.5 Da
    mask = (mz > 10.0) & (mz < pm - 1.5) & (it > 0)
    mz = mz[mask]
    it = it[mask]
    if len(mz) == 0:
        return None

    # Bin peaks to avoid sub-peak splits
    bins = np.round(mz, bin_decimals)
    u_bins, inv = np.unique(bins, return_inverse=True)
    w_bins = np.zeros(len(u_bins), dtype=np.float64)
    np.maximum.at(w_bins, inv, it)

    # Keep top N most intense peaks
    top_indices = np.argsort(-w_bins)[:max_peaks]
    u_top = u_bins[top_indices]
    w_top = np.sqrt(w_bins[top_indices])

    # Sort ascending by m/z for binary search matching
    order = np.argsort(u_top)
    u_sorted = u_top[order]
    w_sorted = w_top[order]

    # Unit vector normalization
    norm = np.linalg.norm(w_sorted)
    if norm < 1e-12:
        return None
    w_normalized = w_sorted / norm

    return {
        "u": u_sorted,
        "w": w_normalized,
        "prec": pm,
        "adduct": add
    }


def compute_modified_cosine(
    qa: Dict,
    qb: Dict,
    tol_da: float = 0.0101
) -> Tuple[float, int]:
    """
    Computes modified cosine score between two binned spectra qa and qb.
    Considers both direct fragment match (shift = 0.0) and precursor-shifted match (shift = prec_a - prec_b).
    Uses greedy bipartite matching without peak reuse.
    """
    ma, ia, pa = qa["u"], qa["w"], qa["prec"]
    mb, ib, pb = qb["u"], qb["w"], qb["prec"]
    shift = pa - pb

    candidates = []
    # Identify matching peak pairs within tolerance
    for i in range(len(ma)):
        mz_a = ma[i]
        int_a = ia[i]
        for off in (0.0, shift):
            target = mz_a - off
            lo = np.searchsorted(mb, target - tol_da)
            hi = np.searchsorted(mb, target + tol_da, side="right")
            for j in range(lo, hi):
                candidates.append((int_a * ib[j], i, j))

    if not candidates:
        return 0.0, 0

    # Sort matches descending by product of intensities
    candidates.sort(reverse=True, key=lambda x: x[0])

    used_i = set()
    used_j = set()
    score = 0.0
    n_matched = 0

    for prod, i, j in candidates:
        if i in used_i or j in used_j:
            continue
        used_i.add(i)
        used_j.add(j)
        score += prod
        n_matched += 1

    return float(score), n_matched


def build_test_spectral_network(
    test_parquet_path: str = "data/test.parquet",
    tau_threshold: float = 0.70,
    min_peaks: int = 6,
    max_peaks_per_spec: int = 120,
    tol_da: float = 0.0101
) -> Tuple[Dict[str, List[Tuple[str, float]]], Dict[str, any]]:
    """
    Constructs the in-test molecular network adjacency list for all test molecules.
    
    Returns:
      network_edges: Dict mapping molecule_id -> list of (neighbor_molecule_id, cosine_weight)
      diagnostics: Dict of graph statistics
    """
    t0 = time.time()
    df = pd.read_parquet(test_parquet_path)

    # Group spectra by molecule_id
    mol_spectra: Dict[str, List[Dict]] = {}
    mol_adduct: Dict[str, str] = {}
    for row in df.itertuples(index=False):
        mid = str(row.molecule_id)
        mzs = np.asarray(row.ms2_mzs, dtype=np.float64)
        ints = np.asarray(row.ms2_normalized_intensities, dtype=np.float64)
        prec = float(row.precursor_mz)
        add = str(getattr(row, "adduct", "[M+H]+"))
        ce = getattr(row, "collision_energy_ev", 30.0)

        mol_spectra.setdefault(mid, []).append({
            "mz": mzs,
            "it": ints,
            "prec": prec,
            "adduct": add,
            "ce": ce
        })
        mol_adduct[mid] = add

    # Prepare binned spectra for all molecules
    binned_queries: Dict[str, Dict] = {}
    for mid, specs in mol_spectra.items():
        b = prepare_binned_spectrum(specs, max_peaks=max_peaks_per_spec, tol_da=tol_da)
        if b is not None:
            binned_queries[mid] = b

    mids = sorted(binned_queries.keys())
    network_edges: Dict[str, List[Tuple[str, float]]] = {m: [] for m in mids}
    
    # Pairwise comparison within the same adduct
    adduct_groups: Dict[str, List[str]] = {}
    for m in mids:
        adduct_groups.setdefault(mol_adduct[m], []).append(m)

    total_pairs_evaluated = 0
    total_edges_formed = 0
    cosine_values = []

    for add, group_mids in adduct_groups.items():
        n = len(group_mids)
        for i in range(n):
            mid_a = group_mids[i]
            qa = binned_queries[mid_a]
            for j in range(i + 1, n):
                mid_b = group_mids[j]
                qb = binned_queries[mid_b]
                total_pairs_evaluated += 1

                cos_s, n_peaks = compute_modified_cosine(qa, qb, tol_da=tol_da)
                if n_peaks >= min_peaks and cos_s >= tau_threshold:
                    network_edges[mid_a].append((mid_b, cos_s))
                    network_edges[mid_b].append((mid_a, cos_s))
                    total_edges_formed += 1
                    cosine_values.append(cos_s)

    # Sort neighbor lists by cosine descending
    for m in mids:
        network_edges[m].sort(key=lambda x: x[1], reverse=True)

    elapsed = time.time() - t0
    queries_with_edges = sum(1 for m, edges in network_edges.items() if len(edges) > 0)

    diagnostics = {
        "elapsed_seconds": round(elapsed, 2),
        "total_test_molecules": len(mids),
        "total_pairs_evaluated": total_pairs_evaluated,
        "total_edges_formed": total_edges_formed,
        "queries_with_edges": queries_with_edges,
        "tau_threshold": tau_threshold,
        "min_peaks": min_peaks,
        "mean_edge_cosine": round(float(np.mean(cosine_values)), 4) if cosine_values else 0.0,
        "max_edge_cosine": round(float(np.max(cosine_values)), 4) if cosine_values else 0.0,
    }

    return network_edges, diagnostics
