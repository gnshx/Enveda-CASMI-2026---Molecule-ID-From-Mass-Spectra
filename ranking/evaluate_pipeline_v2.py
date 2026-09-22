"""
ranking/evaluate_pipeline_v2.py

Rigorous Validation of V2 Pipeline with Information-Weighted Tanimoto & Mass Penalty.
Evaluates MRR@25 on the 250 timsTOF natural products (enveda-np-examples).
"""

import sys
import time
from pathlib import Path
from typing import Dict, List
import numpy as np
import pyarrow.parquet as pq
import torch
from rdkit import Chem

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
sys.path.insert(0, str(ROOT))

from ranking.fpnet import FPNet
from ranking.score_candidates_v2 import CandidateScorerV2, ADDUCT_SHIFTS
import numba

@numba.njit(fastmath=True)
def bin_spectrum(mzs: np.ndarray, ints: np.ndarray, n_bins: int = 4096, min_mz: float = 50.0, max_mz: float = 1074.0) -> np.ndarray:
    bin_size = (max_mz - min_mz) / n_bins
    out = np.zeros(n_bins, dtype=np.float32)
    for i in range(len(mzs)):
        mz = mzs[i]
        if min_mz <= mz < max_mz:
            idx = int((mz - min_mz) / bin_size)
            if ints[i] > out[idx]:
                out[idx] = ints[i]
    mx = 0.0
    for i in range(n_bins):
        if out[i] > 0:
            out[i] = np.sqrt(out[i])
            if out[i] > mx: mx = out[i]
    if mx > 0:
        for i in range(n_bins):
            out[i] /= mx
    return out

def extract_float(val, default=30.0) -> float:
    if val is None:
        return default
    if isinstance(val, (list, np.ndarray)):
        if len(val) == 0:
            return default
        return float(val[0])
    try:
        if pd.isna(val):
            return default
        return float(val)
    except Exception:
        return default

def evaluate_v2(weights_path: Path = None, candidate_db_path: Path = None):
    if weights_path is None:
        weights_path = ROOT / "models" / "fpnet_v2_weights.pt"
        if not weights_path.exists():
            weights_path = ROOT / "models" / "fpnet_weights.pt"
    if candidate_db_path is None:
        candidate_db_path = ROOT / "candidates" / "candidate_db_v2.parquet"
        if not candidate_db_path.exists():
            candidate_db_path = ROOT / "candidates" / "candidate_db.parquet"

    print("=" * 70)
    print("EVALUATING V2 PIPELINE ON 250 TIMSTOF NATURAL PRODUCTS")
    print("=" * 70)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load Model
    print(f"Loading weights from {weights_path}...")
    state_dict = torch.load(weights_path, map_location=device)
    num_blocks = 3 if any("res_blocks.2" in k for k in state_dict.keys()) else 2
    model = FPNet(in_dim=4099, hidden_dim=2048, out_dim=2214, dropout=0.0, num_res_blocks=num_blocks).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load Scorer
    scorer = CandidateScorerV2(candidate_db_path=candidate_db_path)

    # Load Validation Data (RG 20)
    train_path = ROOT / "data" / "train.parquet"
    pf = pq.ParquetFile(str(train_path))
    COLS = [
        "ingest_lib", "normalized_smiles", "inchikey14", "precursor_mz",
        "ms2_mzs", "ms2_normalized_intensities", "num_peaks", "collision_energy_ev", "ionization_mode", "adduct"
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
        add = str(p_val["adduct"][i])

        mzs = np.asarray(p_val["ms2_mzs"][i], dtype=np.float32)
        ints = np.asarray(p_val["ms2_normalized_intensities"][i], dtype=np.float32)

        if ik14 not in molecules:
            molecules[ik14] = {
                "inchikey14": ik14,
                "smiles": smi,
                "precursor_mz": pm,
                "adduct": add,
                "ion": ion,
                "spectra": []
            }
        molecules[ik14]["spectra"].append({"mzs": mzs, "ints": ints, "pm": pm, "ce": ce})

    print(f"✓ Loaded {len(molecules)} unique validation molecules.")

    # Evaluate
    ranks = []
    rrs = []
    top1 = 0
    top5 = 0
    top25 = 0

    t0 = time.time()
    for idx, (ik14, mol_info) in enumerate(molecules.items()):
        pm = mol_info["precursor_mz"]
        add = mol_info["adduct"]
        ion = mol_info["ion"]
        shift = ADDUCT_SHIFTS.get(add, -1.007276 if ion.startswith("pos") else +1.007276)
        neutral_mass = pm + shift

        cand_smis, cand_masses, _ = scorer.retrieve_candidates(neutral_mass, ppm_tol=20.0)

        # Predict FP
        pred_fps = []
        ce_weights = []
        for s in mol_info["spectra"]:
            b = bin_spectrum(s["mzs"], s["ints"])
            feat = np.zeros((1, 4099), dtype=np.float32)
            feat[0, :4096] = b
            feat[0, 4096] = s["pm"] / 1000.0
            feat[0, 4097] = s["ce"] / 100.0
            feat[0, 4098] = 1.0 if ion.startswith("pos") else 0.0

            with torch.no_grad():
                x = torch.from_numpy(feat).to(device)
                logits = model(x)
                p = torch.sigmoid(logits).cpu().numpy()[0]
                pred_fps.append(p)
                ce_w = 1.2 if 15.0 <= s["ce"] <= 45.0 else 1.0
                ce_weights.append(ce_w)

        ce_weights = np.array(ce_weights, dtype=np.float32)
        ce_weights /= np.sum(ce_weights)
        avg_fp = np.sum([p * w for p, w in zip(pred_fps, ce_weights)], axis=0)

        scored_cands = scorer.score_candidate_pool(
            pred_fp=avg_fp,
            neutral_mass=neutral_mass,
            cand_smiles=cand_smis,
            cand_masses=cand_masses,
            sigma_ppm=12.0
        )

        # Check rank of true molecule
        found_rank = None
        for r_idx, (cand_smi, score) in enumerate(scored_cands[:25]):
            if cand_smi == mol_info["smiles"]:
                found_rank = r_idx + 1
                break
            try:
                m = Chem.MolFromSmiles(cand_smi)
                c_ik14 = Chem.MolToInchiKey(m)[:14] if m else ""
                if c_ik14 == ik14:
                    found_rank = r_idx + 1
                    break
            except Exception:
                pass

        if found_rank is not None:
            ranks.append(found_rank)
            rr = 1.0 / found_rank
            rrs.append(rr)
            if found_rank == 1: top1 += 1
            if found_rank <= 5: top5 += 1
            if found_rank <= 25: top25 += 1
        else:
            ranks.append(None)
            rrs.append(0.0)

        if (idx + 1) % 50 == 0 or idx + 1 == len(molecules):
            curr_mrr = np.mean(rrs)
            print(f"  Processed {idx+1}/{len(molecules)} molecules | Running MRR@25: {curr_mrr:.4f} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    n_total = len(molecules)
    mrr_25 = np.mean(rrs)
    print("\n" + "=" * 70)
    print("V2 PIPELINE VALIDATION RESULTS (250 TIMSTOF NATURAL PRODUCTS)")
    print("=" * 70)
    print(f"Total Molecules Evaluated: {n_total}")
    print(f"Top-1 Accuracy:            {top1}/{n_total} ({top1/n_total*100:.2f}%)")
    print(f"Top-5 Accuracy:            {top5}/{n_total} ({top5/n_total*100:.2f}%)")
    print(f"Top-25 Recall:             {top25}/{n_total} ({top25/n_total*100:.2f}%)")
    print(f"Official Metric MRR@25:    {mrr_25:.4f}")
    print("=" * 70)

if __name__ == "__main__":
    evaluate_v2()
