"""
ranking/regio_generator.py

Multi-Channel Constitutional Isomer Generator & Admission Guard (RDKit).
From the 0.358+ SOTA pipeline by haideptry, upgraded for CASMI 2026 Phase 28.
Generates:
  1. Aromatic 6-ring regioisomers (ortho/meta/para migrations & swaps)
  2. N-Alkyl to Ar-Alkyl shifts (Tertiary N-Me -> Secondary NH + Ar-Me)
  3. Carbonyl / Enol ring unsaturation shifts
Guarantees exact formula conservation and strict RDKit canonical tautomer InChIKey14 deduplication.
"""

from __future__ import annotations

import itertools
from typing import List, Dict, Set, Tuple, Optional
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, AllChem
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

# Thread-safe canonical tautomer enumerator with safety caps
_tautomer_enumerator = rdMolStandardize.TautomerEnumerator()
_tautomer_enumerator.SetMaxTautomers(100)
_tautomer_enumerator.SetMaxTransforms(100)


def to_canonical_inchikey14(mol_or_smi: Chem.Mol | str) -> Optional[str]:
    """
    Computes competition-exact canonical tautomer InChIKey14.
    Matches Kaggle evaluation metric and scripts/verify_submission.py Check 5.
    """
    if mol_or_smi is None:
        return None
    try:
        if isinstance(mol_or_smi, str):
            mol = Chem.MolFromSmiles(mol_or_smi)
            if mol is None:
                return None
        else:
            mol = mol_or_smi
        try:
            canon = _tautomer_enumerator.Canonicalize(mol)
        except Exception:
            canon = mol
        ik = Chem.MolToInchiKey(canon)
        return ik[:14] if ik else None
    except Exception:
        return None


def get_ordered_ring(mol, ring_indices):
    """Orders ring atoms sequentially along the perimeter."""
    ordered = [ring_indices[0]]
    curr = ring_indices[0]
    visited = {curr}
    ring_set = set(ring_indices)
    while len(ordered) < len(ring_indices):
        atom = mol.GetAtomWithIdx(curr)
        found = False
        for nbr in atom.GetNeighbors():
            n_idx = nbr.GetIdx()
            if n_idx in ring_set and n_idx not in visited:
                ordered.append(n_idx)
                visited.add(n_idx)
                curr = n_idx
                found = True
                break
        if not found:
            break
    return ordered if len(ordered) == len(ring_indices) else list(ring_indices)


def generate_constitutional_isomers(
    smi: str,
    max_variants: int = 25,
    existing_keys: Optional[Set[str]] = None
) -> List[str]:
    """
    Given a parent SMILES string:
    Generates aromatic regioisomers, N-alkyl to Ar-alkyl shifts, and ring unsaturation / enol shifts.
    Guarantees exact elemental formula conservation and strict canonical tautomer InChIKey14 deduplication.
    """
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return []
    
    parent_key = to_canonical_inchikey14(mol)
    parent_formula = rdMolDescriptors.CalcMolFormula(mol)
    
    generated: Dict[str, str] = {} # InChIKey14 -> SMILES
    blocked_keys = set(existing_keys) if existing_keys else set()
    if parent_key:
        blocked_keys.add(parent_key)

    def try_add_mol(m):
        if not m:
            return
        try:
            Chem.SanitizeMol(m)
            k = to_canonical_inchikey14(m)
            if k and k not in blocked_keys and k not in generated:
                if rdMolDescriptors.CalcMolFormula(m) == parent_formula:
                    generated[k] = Chem.MolToSmiles(m)
        except Exception:
            pass

    # 1. Aromatic 6-ring Regioisomers (Benzene, Pyridine, etc.)
    ring_info = mol.GetRingInfo()
    aromatic_6rings = [r for r in ring_info.AtomRings() 
                       if len(r) == 6 and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    
    for ring in aromatic_6rings:
        ring_set = set(ring)
        substituents = []
        for r_idx in ring:
            atom = mol.GetAtomWithIdx(r_idx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in ring_set:
                    substituents.append((r_idx, nbr.GetIdx()))
        
        if 2 <= len(substituents) <= 4:
            ordered_ring = get_ordered_ring(mol, ring)
            if len(ordered_ring) == 6:
                pos_map = {r_idx: pos for pos, r_idx in enumerate(ordered_ring)}
                ring_atom_at_pos = {pos: r_idx for pos, r_idx in enumerate(ordered_ring)}
                
                occupied = {pos_map[r] for r, _ in substituents}
                free_c = [
                    p for p in range(6) 
                    if p not in occupied and mol.GetAtomWithIdx(ring_atom_at_pos[p]).GetAtomicNum() == 6
                ]
                
                # Single substituent migrations
                for sub_ring_idx, ext_idx in substituents:
                    for target_pos in free_c:
                        target_atom = ring_atom_at_pos[target_pos]
                        try:
                            em = Chem.EditableMol(mol)
                            em.RemoveBond(sub_ring_idx, ext_idx)
                            em.AddBond(target_atom, ext_idx, Chem.BondType.SINGLE)
                            try_add_mol(em.GetMol())
                        except Exception:
                            continue
                
                # Pairwise substituent swaps
                if len(substituents) >= 2:
                    for (r1, ext1), (r2, ext2) in itertools.combinations(substituents, 2):
                        if mol.GetAtomWithIdx(ext1).GetAtomicNum() != mol.GetAtomWithIdx(ext2).GetAtomicNum():
                            try:
                                em = Chem.EditableMol(mol)
                                em.RemoveBond(r1, ext1)
                                em.RemoveBond(r2, ext2)
                                em.AddBond(r1, ext2, Chem.BondType.SINGLE)
                                em.AddBond(r2, ext1, Chem.BondType.SINGLE)
                                try_add_mol(em.GetMol())
                            except Exception:
                                continue

    # 2. N-Alkyl to Ar-Alkyl Shift (Tertiary N-Me -> Secondary NH + Ar-Me)
    n_me_matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[#7;X3:1]-[CH3:2]'))
    ar_ch_matches = mol.GetSubstructMatches(Chem.MolFromSmarts('[c;H1:1]'))
    
    if n_me_matches and ar_ch_matches:
        for n_idx, me_idx in n_me_matches:
            for (c_ar,) in ar_ch_matches:
                try:
                    em = Chem.EditableMol(mol)
                    em.RemoveBond(n_idx, me_idx)
                    em.AddBond(c_ar, me_idx, Chem.BondType.SINGLE)
                    try_add_mol(em.GetMol())
                except Exception:
                    continue

    # 3. Carbonyl / Enol Ring Unsaturation Shifts
    try:
        rxn_en = AllChem.ReactionFromSmarts('[C:1](=[O:4])[C;H1,H2:2][C;H1,H2:3] >> [C:1](-[O:4][H])[C:2]=[C:3]')
        ps = rxn_en.RunReactants((mol,))
        for p in ps:
            try_add_mol(p[0])
    except Exception:
        pass

    return list(generated.values())[:max_variants]


def admit_candidates_safely(
    existing_candidates: List[str],
    candidate_pool: List[str],
    target_count: int = 25,
    required_formula: Optional[str] = None
) -> Tuple[List[str], Dict[str, int]]:
    """
    Admission Gate: Safely admits candidate structures into the target candidate list.
    Enforces:
      1. RDKit parse validity
      2. Exact formula matching (if required_formula is provided)
      3. Strict InChIKey14 uniqueness via Canonical Tautomer Enumeration
    
    Returns (admitted_smiles_list, diagnostic_stats).
    """
    admitted: List[str] = []
    seen_k14: Set[str] = set()
    stats = {
        "existing_kept": 0,
        "existing_dupes_pruned": 0,
        "pool_admitted": 0,
        "pool_rejected_tautomer_dupe": 0,
        "pool_rejected_invalid": 0,
        "pool_rejected_formula_mismatch": 0,
    }

    # 1. Process existing candidates first
    for smi in existing_candidates:
        if not smi:
            continue
        k = to_canonical_inchikey14(smi)
        if k is None:
            stats["existing_dupes_pruned"] += 1
            continue
        if k in seen_k14:
            stats["existing_dupes_pruned"] += 1
            continue
        seen_k14.add(k)
        admitted.append(smi)
        stats["existing_kept"] += 1
        if len(admitted) >= target_count:
            break

    # 2. Admit from candidate pool up to target_count
    if len(admitted) < target_count:
        for smi in candidate_pool:
            if not smi:
                continue
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                stats["pool_rejected_invalid"] += 1
                continue
            if required_formula is not None:
                if rdMolDescriptors.CalcMolFormula(mol) != required_formula:
                    stats["pool_rejected_formula_mismatch"] += 1
                    continue
            k = to_canonical_inchikey14(mol)
            if k is None:
                stats["pool_rejected_invalid"] += 1
                continue
            if k in seen_k14:
                stats["pool_rejected_tautomer_dupe"] += 1
                continue
            seen_k14.add(k)
            admitted.append(smi)
            stats["pool_admitted"] += 1
            if len(admitted) >= target_count:
                break

    return admitted, stats

