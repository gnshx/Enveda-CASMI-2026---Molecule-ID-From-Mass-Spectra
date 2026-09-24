"""
ranking/regio_generator.py

Multi-Channel Constitutional Isomer Generator (RDKit).
From the 0.358+ SOTA pipeline by haideptry.
Generates:
  1. Aromatic 6-ring regioisomers (ortho/meta/para migrations & swaps)
  2. N-Alkyl to Ar-Alkyl shifts (Tertiary N-Me -> Secondary NH + Ar-Me)
  3. Carbonyl / Enol ring unsaturation shifts
Guarantees exact formula conservation and distinct InChIKey14.
"""

import itertools
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, AllChem

RDLogger.DisableLog("rdApp.*")

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

def generate_constitutional_isomers(smi: str, max_variants: int = 25):
    """
    Given a parent SMILES string:
    Generates aromatic regioisomers, N-alkyl to Ar-alkyl shifts, and ring unsaturation / enol shifts.
    Guarantees exact elemental formula conservation and distinct InChIKey14.
    """
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        return []
    
    parent_key = Chem.MolToInchiKey(mol)[:14]
    parent_formula = rdMolDescriptors.CalcMolFormula(mol)
    
    generated = {} # InChIKey14 -> SMILES

    def try_add_mol(m):
        if not m: return
        try:
            Chem.SanitizeMol(m)
            k = Chem.MolToInchiKey(m)[:14]
            if k != parent_key and k not in generated:
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
