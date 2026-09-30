"""regio_generator.py — Enhanced Multi-Channel Constitutional Isomer Generator."""
import itertools
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors, AllChem

RDLogger.DisableLog("rdApp.*")

def get_ordered_ring(mol, ring_indices):
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

def generate_constitutional_isomers(smi: str, max_variants: int = 30):
    mol = Chem.MolFromSmiles(smi)
    if not mol: return []
    parent_key = Chem.MolToInchiKey(mol)[:14]
    parent_formula = rdMolDescriptors.CalcMolFormula(mol)
    generated = {}

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

    ring_info = mol.GetRingInfo()
    # 1. BOTH 5-ring & 6-ring Aromatic Regioisomers
    aromatic_rings = [r for r in ring_info.AtomRings() 
                      if len(r) in (5, 6) and all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
    
    for ring in aromatic_rings:
        ring_len = len(ring)
        ring_set = set(ring)
        substituents = []
        for r_idx in ring:
            atom = mol.GetAtomWithIdx(r_idx)
            for nbr in atom.GetNeighbors():
                if nbr.GetIdx() not in ring_set:
                    substituents.append((r_idx, nbr.GetIdx()))
        
        if 1 <= len(substituents) <= 4:
            ordered_ring = get_ordered_ring(mol, ring)
            if len(ordered_ring) == ring_len:
                pos_map = {r_idx: pos for pos, r_idx in enumerate(ordered_ring)}
                ring_atom_at_pos = {pos: r_idx for pos, r_idx in enumerate(ordered_ring)}
                occupied = {pos_map[r] for r, _ in substituents}
                free_c = [
                    p for p in range(ring_len) 
                    if p not in occupied and mol.GetAtomWithIdx(ring_atom_at_pos[p]).GetAtomicNum() == 6
                ]
                
                # Single migrations
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
                
                # Pairwise swaps
                if len(substituents) >= 2:
                    for (r1, ext1), (r2, ext2) in itertools.combinations(substituents, 2):
                        if r1 != r2:
                            try:
                                em = Chem.EditableMol(mol)
                                em.RemoveBond(r1, ext1)
                                em.RemoveBond(r2, ext2)
                                em.AddBond(r1, ext2, Chem.BondType.SINGLE)
                                em.AddBond(r2, ext1, Chem.BondType.SINGLE)
                                try_add_mol(em.GetMol())
                            except Exception:
                                continue

    # 2. N-Alkyl to Ar-Alkyl Shift
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

    # 3. Alkyl Chain Branching Isomers (n-propyl <-> isopropyl)
    try:
        rxn_p1 = AllChem.ReactionFromSmarts('[*:1]-[CH2:2]-[CH2:3]-[CH3:4] >> [*:1]-[CH:2](-[CH3:3])-[CH3:4]')
        for p in rxn_p1.RunReactants((mol,)):
            try_add_mol(p[0])
        rxn_p2 = AllChem.ReactionFromSmarts('[*:1]-[CH:2](-[CH3:3])-[CH3:4] >> [*:1]-[CH2:2]-[CH2:3]-[CH3:4]')
        for p in rxn_p2.RunReactants((mol,)):
            try_add_mol(p[0])
    except Exception:
        pass

    # 4. Carbonyl / Enol Shifts
    try:
        rxn_en = AllChem.ReactionFromSmarts('[C:1](=[O:4])[C;H1,H2:2][C;H1,H2:3] >> [C:1](-[O:4][H])[C:2]=[C:3]')
        for p in rxn_en.RunReactants((mol,)):
            try_add_mol(p[0])
    except Exception:
        pass

    return list(generated.values())[:max_variants]
