#!/usr/bin/env python3
"""
scripts/verify_regio_generator_qa.py
=============================================================================
CASMI 2026 Phase 28: QA Verification Suite for Regioisomer Generation & Admission Guard.
Audits:
  1. Exact formula conservation across all generated regioisomers.
  2. RDKit canonical tautomer InChIKey14 standardization.
  3. Rejection of tautomer/stereoisomer duplicates (e.g. enamine/imine, keto/enol).
  4. Safe candidate admission gate (admit_candidates_safely) preventing pool contamination.
=============================================================================
"""

import sys
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ranking.regio_generator import (
    to_canonical_inchikey14,
    generate_constitutional_isomers,
    admit_candidates_safely
)

def test_canonical_tautomer_resolution():
    print("Test 1: Canonical Tautomer Standardization...")
    # 1. 2-pyridone vs 2-hydroxypyridine
    smi1 = "O=c1[nH]cccc1"
    smi2 = "Oc1ncccc1"
    k1 = to_canonical_inchikey14(smi1)
    k2 = to_canonical_inchikey14(smi2)
    assert k1 == k2, f"Failed 2-pyridone tautomer equivalence: {k1} vs {k2}"
    print(f"  ✓ 2-pyridone / 2-hydroxypyridine correctly mapped to identical key: {k1}")

    # 2. Enamine vs Imine pair from m_cd6458
    enamine = "CC(C)OC(=O)C1=C(N)OC2=C(OC(CO)=CC2=O)[C@@H]1C1=CC=CC(Cl)=C1Cl"
    imine = "CC(C)OC(=O)C1C(=N)Oc2c(oc(CO)cc2=O)C1c1cccc(Cl)c1Cl"
    k_en = to_canonical_inchikey14(enamine)
    k_im = to_canonical_inchikey14(imine)
    assert k_en == k_im, f"Failed enamine/imine resolution: {k_en} vs {k_im}"
    print(f"  ✓ Enamine / Imine pair correctly resolved to identical canonical InChIKey14: {k_en}")

def test_regio_generator_formula_and_dedup():
    print("\nTest 2: Regioisomer Generation Formula Conservation & Dedup...")
    test_parents = [
        "CC(=O)c1ccccc1C",        # o-methylacetophenone (C9H10O)
        "CN(C)c1ccc(C(=O)O)cc1",   # 4-(dimethylamino)benzoic acid (C9H11NO2)
        "O=C1CCc2ccccc21",         # 1-indanone (C9H8O)
    ]

    for parent in test_parents:
        parent_mol = Chem.MolFromSmiles(parent)
        parent_formula = rdMolDescriptors.CalcMolFormula(parent_mol)
        parent_key = to_canonical_inchikey14(parent_mol)

        isomers = generate_constitutional_isomers(parent, max_variants=25)
        print(f"  Parent: {parent} ({parent_formula}) -> Generated {len(isomers)} regioisomers")
        
        seen_keys = set()
        for iso_smi in isomers:
            iso_mol = Chem.MolFromSmiles(iso_smi)
            assert iso_mol is not None, f"Generated invalid SMILES: {iso_smi}"
            # Check formula
            iso_formula = rdMolDescriptors.CalcMolFormula(iso_mol)
            assert iso_formula == parent_formula, f"Formula mismatch! {iso_formula} != {parent_formula}"
            # Check distinct InChIKey14
            iso_key = to_canonical_inchikey14(iso_mol)
            assert iso_key != parent_key, f"Regioisomer duplicated parent key! {iso_key}"
            assert iso_key not in seen_keys, f"Regioisomer duplicate within pool! {iso_key}"
            seen_keys.add(iso_key)
        
        print(f"    ✓ All {len(isomers)} isomers strictly conserved formula {parent_formula} with 0 duplicate InChIKey14s.")

def test_admit_candidates_safely():
    print("\nTest 3: Admission Gate (admit_candidates_safely)...")
    existing = [
        "CC(=O)c1ccccc1C",  # key 1
        "CC(=O)c1cccc(C)c1",  # key 2
    ]
    # Pool contains: valid isomer, tautomer of existing[0], invalid SMILES, duplicate of another candidate
    pool = [
        "CC(=O)c1ccc(C)cc1",   # valid key 3
        "CC(=O)c1ccccc1C",     # exact duplicate of existing[0] -> reject
        "INVALID_SMILES_XYZ",  # invalid parse -> reject
        "COc1ccccc1",          # formula mismatch -> reject (if required_formula is set)
        "CC(=O)c1ccc(C)cc1",   # duplicate of pool[0] -> reject
    ]

    parent_formula = "C9H10O"
    admitted, stats = admit_candidates_safely(
        existing, pool, target_count=5, required_formula=parent_formula
    )

    print(f"  Admission Stats: {stats}")
    print(f"  Admitted SMILES count: {len(admitted)} (expected 3: existing 2 + pool 1)")
    assert len(admitted) == 3
    assert stats["existing_kept"] == 2
    assert stats["pool_admitted"] == 1
    assert stats["pool_rejected_tautomer_dupe"] >= 2
    assert stats["pool_rejected_invalid"] >= 1
    assert stats["pool_rejected_formula_mismatch"] >= 1

    # Verify all admitted have distinct InChIKey14
    keys = [to_canonical_inchikey14(s) for s in admitted]
    assert len(keys) == len(set(keys)), "Admission gate allowed duplicate InChIKey14!"
    print("  ✓ Admission Gate strictly rejected duplicates, formula mismatches, and syntax errors.")

def main():
    print("=" * 75)
    print("🔬 QA VERIFICATION: REGIOISOMER GENERATOR & ADMISSION GUARD")
    print("=" * 75)
    test_canonical_tautomer_resolution()
    test_regio_generator_formula_and_dedup()
    test_admit_candidates_safely()
    print("=" * 75)
    print("🎉 ALL REGIOISOMER QA CHECKS PASSED (100% COMPLIANT)!")
    print("=" * 75)

if __name__ == "__main__":
    main()
