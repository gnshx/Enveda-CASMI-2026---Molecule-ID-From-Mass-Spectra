"""
validation/metrics.py
Evaluation metrics for CASMI 2026 (MRR@25, Recall@K).
"""

import numpy as np
from typing import List, Dict, Optional


def reciprocal_rank(candidates: List[str], true_inchikey14: str) -> float:
    """
    Compute reciprocal rank for a single molecule.

    Args:
        candidates: Ordered list of predicted SMILES or InChIKey14s (best first)
        true_inchikey14: Ground-truth InChIKey14

    Returns:
        1/rank if true molecule is in top-25, else 0.0
    """
    for rank, cand in enumerate(candidates[:25], start=1):
        # Support both InChIKey14 and full InChIKey comparison
        cand_key = cand[:14] if len(cand) >= 14 else cand
        if cand_key == true_inchikey14:
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
        predictions: {molecule_id: [ranked_inchikey14_list]}
        ground_truth: {molecule_id: true_inchikey14}
        k: Cutoff rank

    Returns:
        Recall@K as a float in [0, 1]
    """
    hits = 0
    total = len(ground_truth)

    for mol_id, true_key in ground_truth.items():
        candidates = predictions.get(mol_id, [])
        for cand in candidates[:k]:
            cand_key = cand[:14] if len(cand) >= 14 else cand
            if cand_key == true_key:
                hits += 1
                break

    return hits / total if total > 0 else 0.0


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
