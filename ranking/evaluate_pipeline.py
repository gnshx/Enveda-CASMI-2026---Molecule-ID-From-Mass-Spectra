"""
ranking/evaluate_pipeline.py

Rigorous Offline Validation of Candidate Generation + FPNet Pipeline.
Evaluates exact competition MRR@25 metric on the 250 timsTOF validation molecules (enveda-np-examples).
Compares baseline score (0.080) vs new pipeline score.
"""

import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pyarrow.parquet as pq
from rdkit import Chem

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
sys.path.insert(0, str(ROOT))

from ranking.score_candidates import CandidateScorer

def extract_float(val, default=30.0) -> float:
    if val is None:
        return default
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return default
        return float(val[0])
    try:
        return float(val)
    except Exception:
        return default

def evaluate():
    print("=" * 70)
    print("EVALUATING PIPELINE ON 250 ENVEDA-NP-EXAMPLES VALIDATION MOLECULES")
    print("=" * 70)

    # 1. Load validation spectra from RG 20
    train_path = ROOT / "data" / "train.parquet"
    pf = pq.ParquetFile(str(train_path))
    COLS = [
        "ingest_lib", "normalized_smiles", "inchikey14", "precursor_mz",
        "ms2_mzs", "ms2_normalized_intensities", "num_peaks", "collision_energy_ev", "ionization_mode"
    ]
    tbl_val = pf.read_row_group(20, columns=COLS)
    p_val = tbl_val.to_pydict()

    molecules: Dict[str, Dict] = {}
    for i in range(len(p_val["ingest_lib"])):
        if p_val["ingest_lib"][i] != "enveda-np-examples":
            continue
        ik14 = p_val["inchikey14"][i]
        smi = p_val["normalized_smiles"][i]
        pm = extract_float(p_val["precursor_mz"][i], 300.0)
        ce = extract_float(p_val["collision_energy_ev"][i], 30.0)
        ion = str(p_val["ionization_mode"][i]).lower()
        adduct = "pos" if ion.startswith("pos") else "neg"

        mzs = np.asarray(p_val["ms2_mzs"][i], dtype=np.float32)
        ints = np.asarray(p_val["ms2_normalized_intensities"][i], dtype=np.float32)

        if ik14 not in molecules:
            molecules[ik14] = {
                "inchikey14": ik14,
                "smiles": smi,
                "precursor_mz": pm,
                "adduct": adduct,
                "spectra": []
            }
        molecules[ik14]["spectra"].append({
            "mzs": mzs, "ints": ints, "pm": pm, "ce": ce
        })

    print(f"Loaded {len(molecules)} unique validation molecules with {sum(len(m['spectra']) for m in molecules.values())} spectra.")

    # 2. Initialize Scorer
    scorer = CandidateScorer()

    # 3. Evaluate each molecule
    ranks = []
    reciprocal_ranks = []
    top1_hits = 0
    top5_hits = 0
    top25_hits = 0
    candidate_pool_sizes = []

    t0 = time.time()
    for idx, (ik14, mol_info) in enumerate(molecules.items()):
        ranked = scorer.rank_candidates(
            query_spectra=mol_info["spectra"],
            precursor_mz=mol_info["precursor_mz"],
            adduct_type=mol_info["adduct"],
            top_k=25
        )

        candidate_pool_sizes.append(len(ranked))
        
        # Check rank of true molecule
        found_rank = None
        for r_idx, (cand_smi, score) in enumerate(ranked):
            try:
                m = Chem.MolFromSmiles(cand_smi)
                cand_ik14 = Chem.MolToInchiKey(m)[:14] if m else ""
            except Exception:
                cand_ik14 = ""

            if cand_ik14 == ik14 or cand_smi == mol_info["smiles"]:
                found_rank = r_idx + 1  # 1-based rank
                break

        if found_rank is not None:
            ranks.append(found_rank)
            reciprocal_ranks.append(1.0 / found_rank)
            if found_rank == 1:
                top1_hits += 1
            if found_rank <= 5:
                top5_hits += 1
            if found_rank <= 25:
                top25_hits += 1
        else:
            ranks.append(None)
            reciprocal_ranks.append(0.0)

        if (idx + 1) % 50 == 0 or idx + 1 == len(molecules):
            curr_mrr = np.mean(reciprocal_ranks)
            print(f"  Processed {idx+1}/{len(molecules)} molecules | Running MRR@25: {curr_mrr:.4f} | Time: {time.time()-t0:.1f}s", flush=True)

    n_total = len(molecules)
    mrr_25 = np.mean(reciprocal_ranks)

    print("\n" + "=" * 70)
    print("FINAL VALIDATION RESULTS (250 TIMSTOF NATURAL PRODUCTS)")
    print("=" * 70)
    print(f"Total Molecules Evaluated: {n_total}")
    print(f"Top-1 Accuracy:            {top1_hits}/{n_total} ({top1_hits/n_total*100:.2f}%)")
    print(f"Top-5 Accuracy:            {top5_hits}/{n_total} ({top5_hits/n_total*100:.2f}%)")
    print(f"Top-25 Recall:             {top25_hits}/{n_total} ({top25_hits/n_total*100:.2f}%)")
    print(f"Official Metric MRR@25:    {mrr_25:.4f}")
    print(f"Baseline Score was:        0.0800")
    print(f"Score Improvement:         {mrr_25 - 0.0800:+.4f} ({(mrr_25 / 0.0800 - 1.0) * 100:+.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    evaluate()
