"""
validation/metrics.py
Evaluation metrics for CASMI 2026 (MRR@25, Recall@K).
"""

import re
from functools import lru_cache
import numpy as np
from typing import List, Dict, Optional, Tuple

_INCHIKEY_RE = re.compile(r"^[A-Z]{14}(?:-[A-Z]{10}-[A-Z])?$")


@lru_cache(maxsize=200000)
def normalize_to_inchikey14(candidate: str) -> Optional[str]:
    """
    Normalize either an InChIKey/InChIKey14 or a SMILES string to InChIKey14.
    InChIKeys are sliced to 14 chars; SMILES are canonicalized via RDKit tautomer rules.
    """
    if not isinstance(candidate, str):
        return None
    cand = candidate.strip()
    if not cand:
        return None
    if _INCHIKEY_RE.match(cand):
        return cand[:14]
    return smiles_to_inchikey14(cand)


def reciprocal_rank(candidates: List[str], true_inchikey14: str) -> float:
    """
    Compute reciprocal rank for a single molecule.

    Args:
        candidates: Ordered list of predicted SMILES or InChIKey14s (best first)
        true_inchikey14: Ground-truth InChIKey14 (or full InChIKey / SMILES)

    Returns:
        1/rank if true molecule is in top-25, else 0.0
    """
    true_key = normalize_to_inchikey14(true_inchikey14) or true_inchikey14[:14]
    for rank, cand in enumerate(candidates[:25], start=1):
        cand_key = normalize_to_inchikey14(cand)
        if cand_key == true_key:
            return 1.0 / rank
    return 0.0


def mrr_at_25(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, str],
) -> float:
    """
    Compute MRR@25 over all molecules.

    Args:
        predictions: {molecule_id: [ranked_inchikey14_or_smiles_list]}
        ground_truth: {molecule_id: true_inchikey14}

    Returns:
        Mean Reciprocal Rank @ 25
    """
    rrs = []
    for mol_id, true_key in ground_truth.items():
        candidates = predictions.get(mol_id, [])
        rr = reciprocal_rank(candidates, true_key)
        rrs.append(rr)

    return float(np.mean(rrs)) if rrs else 0.0


def recall_at_k(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, str],
    k: int,
) -> float:
    """
    Compute Recall@K: fraction of molecules where true structure is in top-K predictions.

    Args:
        predictions: {molecule_id: [ranked_inchikey14_or_smiles_list]}
        ground_truth: {molecule_id: true_inchikey14}
        k: Cutoff rank

    Returns:
        Recall@K as a float in [0, 1]
    """
    hits = 0
    total = len(ground_truth)

    for mol_id, true_key in ground_truth.items():
        norm_true = normalize_to_inchikey14(true_key) or true_key[:14]
        candidates = predictions.get(mol_id, [])
        for cand in candidates[:k]:
            cand_key = normalize_to_inchikey14(cand)
            if cand_key == norm_true:
                hits += 1
                break

    return hits / total if total > 0 else 0.0


def paired_bootstrap_ci(
    baseline_preds: Dict[str, List[str]],
    challenger_preds: Dict[str, List[str]],
    ground_truth: Dict[str, str],
    n_bootstrap: int = 2000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute paired bootstrap 95% confidence interval for MRR@25 delta (challenger - baseline)."""
    mol_ids = sorted(ground_truth.keys())
    if not mol_ids:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_win": 0.0}
    base_rr = np.array([reciprocal_rank(baseline_preds.get(m, []), ground_truth[m]) for m in mol_ids], dtype=np.float64)
    chal_rr = np.array([reciprocal_rank(challenger_preds.get(m, []), ground_truth[m]) for m in mol_ids], dtype=np.float64)
    deltas = chal_rr - base_rr
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot[b] = deltas[idx].mean()
    return {
        "baseline_mrr": float(base_rr.mean()),
        "challenger_mrr": float(chal_rr.mean()),
        "mean_delta": float(deltas.mean()),
        "ci_low": float(np.percentile(boot, 2.5)),
        "ci_high": float(np.percentile(boot, 97.5)),
        "p_win": float((boot > 0).mean()),
    }


def evaluate_pipeline(
    predictions: Dict[str, List[str]],
    ground_truth: Dict[str, str],
    ks: List[int] = [1, 5, 10, 25, 50, 100],
) -> Dict[str, float]:
    """
    Full evaluation suite: MRR@25 + Recall@K for multiple K values.

    Returns:
        Dictionary of metric_name -> value
    """
    results = {
        "mrr_at_25": mrr_at_25(predictions, ground_truth),
    }
    for k in ks:
        results[f"recall_at_{k}"] = recall_at_k(predictions, ground_truth, k)

    return results


def print_evaluation_report(metrics: Dict[str, float]) -> None:
    """Pretty-print evaluation metrics."""
    print("=" * 50)
    print("CASMI 2026 Evaluation Report")
    print("=" * 50)
    print(f"  MRR@25:      {metrics.get('mrr_at_25', 0):.4f}")
    print()
    for k in [1, 5, 10, 25, 50, 100]:
        key = f"recall_at_{k}"
        if key in metrics:
            print(f"  Recall@{k:<4}: {metrics[key]:.4f}")
    print("=" * 50)


# ─── InChIKey utilities ───────────────────────────────────────────────────────

def smiles_to_inchikey14(smiles: str) -> Optional[str]:
    """
    Convert SMILES to the first 14 characters of InChIKey.
    Uses RDKit for canonicalization and tautomer standardization.
    Returns None if SMILES is invalid.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem.MolStandardize import rdMolStandardize

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Tautomer canonicalization (matches competition scoring)
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)

        from rdkit.Chem.inchi import MolToInchiKey
        inchikey = MolToInchiKey(mol)
        if inchikey:
            return inchikey[:14]
    except Exception:
        pass
    return None


def batch_smiles_to_inchikey14(smiles_list: List[str]) -> List[Optional[str]]:
    """Batch convert SMILES to InChIKey14."""
    return [smiles_to_inchikey14(s) for s in smiles_list]
