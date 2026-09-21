"""
submission/build_submission.py
Build and validate the final submission.csv for CASMI 2026.

Official submission format (from competition overview page):
    molecule_id,smiles
    m_0014ef,SMILES_1;SMILES_2;SMILES_25

    *** Separator is ';' with NO trailing space — match exactly ***

Rejection criteria (Kaggle will hard-reject if any apply):
    - Missing molecule_id or smiles column
    - Empty submission
    - Nulls in either column
    - Repeated molecule_id
    - More than 25 semicolon-separated guesses for any molecule

Scoring:
    - RDKit tautomer canonicalization pinned at version 2026.03.3
    - Compared by InChIKey14 (first 14 chars of InChIKey)
    - Stereochemistry and tautomers are NOT penalized
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path


def build_submission(
    predictions: Dict[str, List[str]],
    output_path: str = "submission.csv",
    max_candidates: int = 25,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Build competition submission from ranked candidate SMILES.

    Args:
        predictions: {molecule_id: [ranked_smiles_list]}
        output_path: Where to save submission.csv
        max_candidates: Maximum candidates per molecule (25 per rules)
        validate: Whether to validate SMILES with RDKit

    Returns:
        Submission DataFrame
    """
    rows = []

    for mol_id, candidates in predictions.items():
        # Truncate to max candidates
        candidates = candidates[:max_candidates]

        # Optionally validate and filter broken SMILES
        if validate:
            candidates = _filter_valid_smiles(candidates)

        # Canonicalize SMILES
        candidates = _canonicalize_smiles_list(candidates)

        # Deduplicate by InChIKey14 (keep first occurrence = highest rank)
        candidates = _deduplicate_by_inchikey14(candidates)

        # Join with semicolons — official format uses ';' with NO space
        smiles_str = ";".join(candidates[:max_candidates])

        rows.append({"molecule_id": mol_id, "smiles": smiles_str})

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} rows to {output_path}")

    # Sanity checks
    _validate_submission(df, max_candidates)

    return df


def _filter_valid_smiles(smiles_list: List[str]) -> List[str]:
    """Remove SMILES that RDKit cannot parse."""
    try:
        from rdkit import Chem
        valid = []
        for smi in smiles_list:
            if smi and Chem.MolFromSmiles(smi) is not None:
                valid.append(smi)
        return valid
    except ImportError:
        return smiles_list  # If RDKit not available, skip validation


def _canonicalize_smiles_list(smiles_list: List[str]) -> List[str]:
    """Canonicalize all SMILES using RDKit."""
    try:
        from rdkit import Chem
        canonical = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canonical.append(Chem.MolToSmiles(mol, canonical=True))
            else:
                canonical.append(smi)
        return canonical
    except ImportError:
        return smiles_list


def _deduplicate_by_inchikey14(smiles_list: List[str]) -> List[str]:
    """Remove duplicate structures by InChIKey14 (keep first = best ranked)."""
    try:
        from rdkit import Chem
        from rdkit.Chem.inchi import MolToInchiKey
        from rdkit.Chem.MolStandardize import rdMolStandardize

        seen_keys = set()
        unique = []
        enumerator = rdMolStandardize.TautomerEnumerator()

        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            mol = enumerator.Canonicalize(mol)
            key = MolToInchiKey(mol)
            if key:
                key14 = key[:14]
                if key14 not in seen_keys:
                    seen_keys.add(key14)
                    unique.append(smi)
        return unique
    except Exception:
        # Fallback: deduplicate by exact SMILES string
        return list(dict.fromkeys(smiles_list))


def _validate_submission(df: pd.DataFrame, max_candidates: int) -> None:
    """Run sanity checks matching Kaggle's hard-rejection criteria."""
    print("\n── Submission Validation ──")
    all_ok = True

    def check(label: str, bad_count: int, fatal: bool = True) -> None:
        nonlocal all_ok
        status = "✓" if bad_count == 0 else ("✗ FATAL" if fatal else "⚠ WARNING")
        print(f"  {label}: {bad_count} {status}")
        if bad_count > 0 and fatal:
            all_ok = False

    # 1. Null values (Kaggle hard-rejects these)
    null_mol_ids = df["molecule_id"].isna().sum()
    null_smiles   = df["smiles"].isna().sum()
    check("Null molecule_ids", null_mol_ids)
    check("Null smiles", null_smiles)

    # 2. Duplicate molecule IDs (Kaggle hard-rejects)
    dupes = df["molecule_id"].duplicated().sum()
    check("Duplicate molecule_ids", dupes)

    # 3. Over the 25-candidate limit (Kaggle hard-rejects)
    n_candidates = df["smiles"].str.split(";").str.len()
    over_limit = (n_candidates > max_candidates).sum()
    check(f"Molecules over {max_candidates}-candidate limit", over_limit)

    # 4. Empty SMILES (Kaggle hard-rejects)
    empty = (df["smiles"].fillna("").str.strip() == "").sum()
    check("Empty SMILES entries", empty)

    # 5. Informational stats
    print(f"  Total molecules:              {len(df)}")
    print(f"  Avg candidates per molecule:  {n_candidates.mean():.1f}")
    print(f"  Min candidates per molecule:  {n_candidates.min()}")
    print(f"  Max candidates per molecule:  {n_candidates.max()}")

    if all_ok:
        print("  ✓ Submission looks valid — safe to upload")
    else:
        print("  ✗ Fix FATAL errors before submitting!")
    print("──────────────────────────")
