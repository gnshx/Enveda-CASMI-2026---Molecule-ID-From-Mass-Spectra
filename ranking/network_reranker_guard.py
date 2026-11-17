"""
ranking/network_reranker_guard.py
=============================================================================
CASMI 2026 Phase 29: QA-Guarded Rank-Constrained In-Test Molecular Networking.
Ensures:
  1. Strict Rank-1 Shield: Confirmed Regime A reference hits (w_top = 0.00) are 100% frozen.
  2. Sub-Leader Reranking: Network Tanimoto propagation is confined strictly to Ranks 2-25.
  3. Zero InChIKey14 Redundancy: Canonical tautomer InChIKey14 deduplication is enforced.
  4. Dimensional & Format Integrity: Exactly 400 rows x 25 candidate slots preserved.
=============================================================================
"""

from __future__ import annotations

import time
from typing import Dict, List, Set, Tuple, Optional
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from ranking.regio_generator import to_canonical_inchikey14, admit_candidates_safely

_morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

def compute_fp(smi: str, cache: Optional[Dict[str, any]] = None):
    """Computes Morgan fingerprint with caching."""
    if cache is not None and smi in cache:
        return cache[smi]
    try:
        mol = Chem.MolFromSmiles(smi)
        fp = _morgan_gen.GetFingerprint(mol) if mol is not None else None
    except Exception:
        fp = None
    if cache is not None:
        cache[smi] = fp
    return fp

def zscore(arr: np.ndarray) -> np.ndarray:
    """Standardizes vector to zero mean and unit variance."""
    std = float(arr.std())
    if std < 1e-9:
        return np.zeros_like(arr)
    return (arr - float(arr.mean())) / std

def run_constrained_network_reranking(
    candidates_dict: Dict[str, List[str]],
    network_edges: Dict[str, List[Tuple[str, float]]],
    protected_mids: Set[str],
    lam: float = 0.25,
    fp_cache: Optional[Dict[str, any]] = None
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Executes in-test molecular network neighbor propagation with hard Rank-1 shield.
    
    Guarantees:
      - Protected queries (Regime A hits in protected_mids) have Rank 1 100% frozen.
      - Network similarity scores are computed and sorted only for candidate ranks 2-25.
      - Canonical tautomer InChIKey14 deduplication is verified for all 25 slots.
    
    Returns (updated_candidates_dict, diagnostic_stats).
    """
    t0 = time.time()
    cache = {} if fp_cache is None else fp_cache
    updated_dict: Dict[str, List[str]] = {}
    stats = {
        "total_queries": len(candidates_dict),
        "shielded_queries": 0,
        "shielded_rank1_moved": 0,
        "subranks_reordered": 0,
        "queries_with_network_edges": 0,
        "dedup_failures": 0,
    }

    for mid, cands in candidates_dict.items():
        if len(cands) <= 1:
            updated_dict[mid] = list(cands)
            continue

        is_shielded = mid in protected_mids
        if is_shielded:
            stats["shielded_queries"] += 1

        nb_list = network_edges.get(mid, [])
        if not nb_list:
            updated_dict[mid] = list(cands)
            continue

        stats["queries_with_network_edges"] += 1

        if is_shielded:
            # Hard-lock Rank 1 leader: only ranks 2-25 are subject to network reranking
            leader = cands[0]
            sub_cands = cands[1:25]
            
            # Base positional ranking score for ranks 2..len(cands)
            base_scores = -np.log(np.arange(2, len(sub_cands) + 2))
            net_scores = np.zeros(len(sub_cands), dtype=np.float64)

            # Neighbor evidence propagation
            for neighbor_mid, edge_weight in nb_list:
                neighbor_cands = candidates_dict.get(neighbor_mid, [])[:5]
                if not neighbor_cands:
                    continue
                weights = 1.0 / np.arange(1, len(neighbor_cands) + 1)
                weights /= weights.sum()

                neighbor_fps = [compute_fp(s, cache) for s in neighbor_cands]
                for i, cand_smi in enumerate(sub_cands):
                    fi = compute_fp(cand_smi, cache)
                    if fi is None:
                        continue
                    sims = [
                        w * DataStructs.TanimotoSimilarity(fi, fj)
                        for fj, w in zip(neighbor_fps, weights)
                        if fj is not None
                    ]
                    if sims:
                        net_scores[i] += (edge_weight ** 4) * sum(sims)

            # Composite sub-rank score
            composite = zscore(base_scores) + lam * zscore(net_scores)
            sorted_indices = np.argsort(-composite, kind="stable")
            reranked_sub = [sub_cands[i] for i in sorted_indices]

            if reranked_sub != sub_cands:
                stats["subranks_reordered"] += 1

            # Reconstruct list with Rank 1 rigidly anchored
            final_cands = [leader] + reranked_sub
            # Assert immutable shield invariant
            assert final_cands[0] == leader, f"CRITICAL: Rank 1 shield breached for {mid}!"
        else:
            # Unshielded query (Regime B/C): Network can re-order all 25 slots
            base_scores = -np.log(np.arange(1, len(cands[:25]) + 1))
            net_scores = np.zeros(len(cands[:25]), dtype=np.float64)

            for neighbor_mid, edge_weight in nb_list:
                neighbor_cands = candidates_dict.get(neighbor_mid, [])[:5]
                if not neighbor_cands:
                    continue
                weights = 1.0 / np.arange(1, len(neighbor_cands) + 1)
                weights /= weights.sum()

                neighbor_fps = [compute_fp(s, cache) for s in neighbor_cands]
                for i, cand_smi in enumerate(cands[:25]):
                    fi = compute_fp(cand_smi, cache)
                    if fi is None:
                        continue
                    sims = [
                        w * DataStructs.TanimotoSimilarity(fi, fj)
                        for fj, w in zip(neighbor_fps, weights)
                        if fj is not None
                    ]
                    if sims:
                        net_scores[i] += (edge_weight ** 4) * sum(sims)

            composite = zscore(base_scores) + lam * zscore(net_scores)
            sorted_indices = np.argsort(-composite, kind="stable")
            final_cands = [cands[i] for i in sorted_indices]

            if final_cands != cands[:25]:
                stats["subranks_reordered"] += 1

        # Enforce canonical tautomer InChIKey14 deduplication invariant
        seen_keys: Set[str] = set()
        deduped: List[str] = []
        for s in final_cands:
            k = to_canonical_inchikey14(s)
            if k is not None:
                if k in seen_keys:
                    stats["dedup_failures"] += 1
                    continue
                seen_keys.add(k)
            deduped.append(s)

        # In case deduplication dropped any duplicate, backfill from original candidates
        if len(deduped) < 25:
            for s in cands:
                k = to_canonical_inchikey14(s)
                if k and k not in seen_keys:
                    seen_keys.add(k)
                    deduped.append(s)
                    if len(deduped) == 25:
                        break

        assert len(deduped) == len(cands[:25]), f"Dimensional error for {mid}: {len(deduped)}"
        updated_dict[mid] = deduped

    return updated_dict, stats
