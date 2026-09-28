"""
ranking/local_validator.py

Offline Local Validation Harness for CASMI 2026.
Evaluates official competition metric MRR@25 and Recall@K on the 250 ground-truth
timsTOF natural product molecules from data/train.parquet Row Group 20 (enveda-np-examples).

Enables local empirical benchmarking before submitting to Kaggle.
"""

import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, MACCSkeys, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
sys.path.insert(0, str(ROOT))

from ranking.fpnet import FPNet
from ranking.score_candidates_v2 import (
    CandidateScorerV2,
    ADDUCT_SHIFTS,
    bin_spectrum,
    compute_molecule_fingerprint,
    weighted_tanimoto_batch
)
import ranking.regio_generator as rg

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

def load_validation_data(max_mols: Optional[int] = None) -> Dict[str, Dict]:
    """Loads 250 ground-truth natural product molecules from train.parquet Row Group 20."""
    train_path = ROOT / "data" / "train.parquet"
    pf = pq.ParquetFile(str(train_path))
    COLS = [
        "ingest_lib", "normalized_smiles", "inchikey14", "precursor_mz",
        "ms2_mzs", "ms2_normalized_intensities", "collision_energy_ev", "ionization_mode", "adduct"
    ]
    tbl_val = pf.read_row_group(20, columns=COLS).to_pydict()

    molecules: Dict[str, Dict] = {}
    for i in range(len(tbl_val["ingest_lib"])):
        if tbl_val["ingest_lib"][i] != "enveda-np-examples":
            continue
        ik14 = tbl_val["inchikey14"][i]
        if not ik14:
            continue
        if ik14 not in molecules:
            molecules[ik14] = {
                "inchikey14": ik14,
                "smiles": tbl_val["normalized_smiles"][i],
                "precursor_mz": extract_float(tbl_val["precursor_mz"][i], 300.0),
                "adduct": str(tbl_val["adduct"][i]),
                "ion": str(tbl_val["ionization_mode"][i]).lower(),
                "spectra": []
            }
        molecules[ik14]["spectra"].append({
            "mzs": np.asarray(tbl_val["ms2_mzs"][i], dtype=np.float32),
            "ints": np.asarray(tbl_val["ms2_normalized_intensities"][i], dtype=np.float32),
            "pm": extract_float(tbl_val["precursor_mz"][i], 300.0),
            "ce": extract_float(tbl_val["collision_energy_ev"][i], 30.0)
        })

    if max_mols is not None:
        molecules = dict(list(molecules.items())[:max_mols])

    return molecules

def evaluate_predictions(predictions: List[str], true_ik14: str, topn: int = 25) -> Tuple[Optional[int], float]:
    """
    Computes rank (1-indexed) and reciprocal rank for a ranked candidate list.
    Evaluates InChIKey14 match (exact 2D skeletal topology as per CASMI evaluation).
    """
    for r_idx, smi in enumerate(predictions[:topn]):
        try:
            m = Chem.MolFromSmiles(smi)
            if m:
                cand_ik14 = Chem.MolToInchiKey(m)[:14]
                if cand_ik14 == true_ik14:
                    rank = r_idx + 1
                    return rank, 1.0 / rank
        except Exception:
            continue
    return None, 0.0

def run_local_validation(num_molecules: int = 50, verbose: bool = True):
    print("=" * 80)
    print(f"CASMI 2026 LOCAL VALIDATION HARNESS — BENCHMARKING ON {num_molecules} NATURAL PRODUCTS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Device: {device}")

    # Load FPNet model
    weights_path = ROOT / "models" / "fpnet_v2_weights.pt"
    if not weights_path.exists():
        weights_path = ROOT / "models" / "fpnet_weights.pt"
    state_dict = torch.load(weights_path, map_location=device)
    num_blocks = 3 if any("res_blocks.2" in k for k in state_dict.keys()) else 2
    model = FPNet(in_dim=4099, hidden_dim=2048, out_dim=2214, dropout=0.0, num_res_blocks=num_blocks).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # Load Candidate Scorer
    cand_db_path = ROOT / "candidates" / "candidate_db.parquet"
    scorer = CandidateScorerV2(candidate_db_path=cand_db_path)

    # Load Validation Set
    molecules = load_validation_data(max_mols=num_molecules)
    print(f"✓ Loaded {len(molecules)} validation molecules (RG 20 enveda-np-examples).")

    # Strategy accumulators
    strategies = [
        "baseline_db_only",      # Strategy 0: Raw DB candidates, no regioisomers
        "v3_unconditional_slots", # Strategy 1: V3 allocation (Slots 2, 4, 5, 6 regioisomers)
        "v4_strict_gating",       # Strategy 2: V4 strict gating (sc >= parent_sc - 1.5)
        "v5_calibrated_slots"     # Strategy 3: Calibrated multi-channel regioisomers
    ]
    results = {s: {"rrs": [], "top1": 0, "top5": 0, "top25": 0} for s in strategies}

    t0 = time.time()
    for idx, (ik14, mol_info) in enumerate(molecules.items()):
        pm = mol_info["precursor_mz"]
        add = mol_info["adduct"]
        ion = mol_info["ion"]
        shift = ADDUCT_SHIFTS.get(add, -1.007276 if ion.startswith("pos") else +1.007276)
        neutral_mass = pm + shift

        cand_smis, cand_masses, _ = scorer.retrieve_candidates(neutral_mass, ppm_tol=20.0)
        if len(cand_smis) == 0:
            for s in strategies:
                results[s]["rrs"].append(0.0)
            continue

        # Predict FP from spectra
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

        # Base candidate scoring
        scored_cands = scorer.score_candidate_pool(
            pred_fp=avg_fp, neutral_mass=neutral_mass,
            cand_smiles=cand_smis, cand_masses=cand_masses, sigma_ppm=12.0
        )
        base_ranked = [s for s, _ in scored_cands]
        if not base_ranked:
            for s in strategies:
                results[s]["rrs"].append(0.0)
            continue

        parent_smi = base_ranked[0]
        base_tail = base_ranked[1:]

        # Regioisomer Generation & Scoring
        try:
            raw_vars = rg.generate_constitutional_isomers(parent_smi, max_variants=30)
            raw_vars = [v for v in raw_vars if v != parent_smi]
        except Exception:
            raw_vars = []

        # Score variants with continuous Tanimoto
        var_scored = []
        p_fp = compute_molecule_fingerprint(parent_smi)
        p_sc = 0.0
        if p_fp is not None:
            dot = np.dot(avg_fp, p_fp)
            denom = np.sum(avg_fp) + np.sum(p_fp) - dot + 1e-9
            p_sc = dot / denom

        for v in raw_vars:
            v_fp = compute_molecule_fingerprint(v)
            if v_fp is not None:
                dot = np.dot(avg_fp, v_fp)
                denom = np.sum(avg_fp) + np.sum(v_fp) - dot + 1e-9
                var_scored.append((v, dot / denom))
            else:
                var_scored.append((v, 0.0))
        var_scored.sort(key=lambda x: -x[1])
        all_vars = [s for s, _ in var_scored]

        # ── Strategy 0: Baseline DB Only
        pred_s0 = base_ranked[:25]

        # ── Strategy 1: V3 Unconditional Slot Allocation (Slots 2, 4, 5, 6)
        pred_s1 = [parent_smi]
        seen1 = {parent_smi}
        v_idx, t_idx = 0, 0
        while len(pred_s1) < 25:
            r = len(pred_s1)
            if r in (1, 3, 4, 5) and v_idx < len(all_vars):
                c = all_vars[v_idx]; v_idx += 1
            elif t_idx < len(base_tail):
                c = base_tail[t_idx]; t_idx += 1
            elif v_idx < len(all_vars):
                c = all_vars[v_idx]; v_idx += 1
            else:
                c = "CCO"
            if c not in seen1:
                seen1.add(c)
                pred_s1.append(c)

        # ── Strategy 2: V4 Strict Gating (sc >= parent_sc - 1.5, in practice 0.05 tanimoto)
        # In V4, Slot 2 only promoted if high_conf, and Slot 5 was omitted (r in (3, 5))
        high_conf_v4 = [s for s, sc in var_scored if sc >= p_sc - 0.02]
        pred_s2 = [parent_smi]
        seen2 = {parent_smi}
        v_idx, t_idx = 0, 0
        while len(pred_s2) < 25:
            r = len(pred_s2)
            if r == 1:
                if high_conf_v4 and v_idx < len(high_conf_v4):
                    c = high_conf_v4[v_idx]; v_idx += 1
                elif t_idx < len(base_tail):
                    c = base_tail[t_idx]; t_idx += 1
                elif v_idx < len(all_vars):
                    c = all_vars[v_idx]; v_idx += 1
                else:
                    c = "CCO"
            elif r in (3, 5) and v_idx < len(all_vars):
                c = all_vars[v_idx]; v_idx += 1
            elif t_idx < len(base_tail):
                c = base_tail[t_idx]; t_idx += 1
            elif v_idx < len(all_vars):
                c = all_vars[v_idx]; v_idx += 1
            else:
                c = "CCO"
            if c not in seen2:
                seen2.add(c)
                pred_s2.append(c)

        # ── Strategy 3: V5 Calibrated Multi-Slot Injection
        # Promote top variant to Slot 2 unconditionally (proven by V3),
        # plus Slot 4 and Slot 5 for next top variants
        pred_s3 = [parent_smi]
        seen3 = {parent_smi}
        v_idx, t_idx = 0, 0
        while len(pred_s3) < 25:
            r = len(pred_s3)
            # Ranks 2, 4, 5 get top regioisomers
            if r in (1, 3, 4) and v_idx < len(all_vars):
                c = all_vars[v_idx]; v_idx += 1
            elif t_idx < len(base_tail):
                c = base_tail[t_idx]; t_idx += 1
            elif v_idx < len(all_vars):
                c = all_vars[v_idx]; v_idx += 1
            else:
                c = "CCO"
            if c not in seen3:
                seen3.add(c)
                pred_s3.append(c)

        # Evaluate each strategy
        preds_dict = {
            "baseline_db_only": pred_s0,
            "v3_unconditional_slots": pred_s1,
            "v4_strict_gating": pred_s2,
            "v5_calibrated_slots": pred_s3
        }

        for s_name, preds in preds_dict.items():
            rank, rr = evaluate_predictions(preds, ik14)
            results[s_name]["rrs"].append(rr)
            if rank == 1:
                results[s_name]["top1"] += 1
            if rank is not None and rank <= 5:
                results[s_name]["top5"] += 1
            if rank is not None and rank <= 25:
                results[s_name]["top25"] += 1

        if verbose and ((idx + 1) % 10 == 0 or idx + 1 == len(molecules)):
            s0_mrr = np.mean(results["baseline_db_only"]["rrs"])
            s1_mrr = np.mean(results["v3_unconditional_slots"]["rrs"])
            s2_mrr = np.mean(results["v4_strict_gating"]["rrs"])
            s3_mrr = np.mean(results["v5_calibrated_slots"]["rrs"])
            print(f"[{idx+1:3d}/{len(molecules)}] MRR@25 -> Base: {s0_mrr:.4f} | V3: {s1_mrr:.4f} | V4: {s2_mrr:.4f} | V5: {s3_mrr:.4f} ({time.time()-t0:.1f}s)", flush=True)

    print("\n" + "=" * 80)
    print("FINAL BENCHMARK COMPARISON ON VALIDATION SET")
    print("=" * 80)
    n_total = len(molecules)
    for s_name in strategies:
        mrr = np.mean(results[s_name]["rrs"])
        t1 = results[s_name]["top1"]
        t5 = results[s_name]["top5"]
        t25 = results[s_name]["top25"]
        print(f"Strategy: {s_name:<24} | MRR@25: {mrr:.4f} | Top-1: {t1/n_total*100:5.1f}% | Top-5: {t5/n_total*100:5.1f}% | Top-25: {t25/n_total*100:5.1f}%")
    print("=" * 80)

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    run_local_validation(num_molecules=n)
