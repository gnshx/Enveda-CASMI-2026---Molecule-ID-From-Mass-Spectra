"""
build_library_matcher.py
Streams train.parquet to find exact/high-similarity reference spectra matches
for all 400 test molecules (Tier 1 Public Library Matching).
"""

import sys
import time
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent

def compute_spectral_cosine(mzs1: np.ndarray, ints1: np.ndarray,
                           mzs2: np.ndarray, ints2: np.ndarray,
                           tolerance_da: float = 0.05) -> float:
    """Computes peak-matching dot-product cosine similarity."""
    if len(mzs1) == 0 or len(mzs2) == 0:
        return 0.0
    
    # Sort peaks
    idx1 = np.argsort(mzs1)
    mzs1, ints1 = mzs1[idx1], ints1[idx1]
    idx2 = np.argsort(mzs2)
    mzs2, ints2 = mzs2[idx2], ints2[idx2]
    
    # Two-pointer matching within tolerance
    i, j = 0, 0
    dot_product = 0.0
    while i < len(mzs1) and j < len(mzs2):
        diff = mzs1[i] - mzs2[j]
        if abs(diff) <= tolerance_da:
            dot_product += ints1[i] * ints2[j]
            i += 1
            j += 1
        elif diff < 0:
            i += 1
        else:
            j += 1
            
    norm1 = np.sqrt(np.sum(ints1 ** 2)) + 1e-7
    norm2 = np.sqrt(np.sum(ints2 ** 2)) + 1e-7
    return float(dot_product / (norm1 * norm2))

def main():
    print("=" * 70)
    print("STARTING TIER 1 SPECTRAL LIBRARY MATCHER (train.parquet)")
    print("=" * 70)
    
    t0 = time.time()
    test_path = ROOT / "data" / "test.parquet"
    train_path = ROOT / "data" / "train.parquet"
    
    print(f"Loading test molecules from {test_path}...")
    tbl_test = pq.read_table(test_path)
    p_test = tbl_test.to_pydict()
    
    test_mols = {}
    for i in range(len(p_test["molecule_id"])):
        mid = p_test["molecule_id"][i]
        pm = float(p_test["precursor_mz"][i])
        ion = str(p_test["ionization_mode"][i]).lower()
        mzs = np.array(p_test["ms2_mzs"][i], dtype=np.float32)
        ints = np.array(p_test["ms2_normalized_intensities"][i], dtype=np.float32)
        
        # Keep highest peak intensity spectrum or average
        if mid not in test_mols:
            test_mols[mid] = {
                "molecule_id": mid,
                "precursor_mz": pm,
                "ion": ion,
                "spectra": []
            }
        test_mols[mid]["spectra"].append((mzs, ints))
        
    print(f"✓ Loaded {len(test_mols)} test molecules.")
    
    test_pms = np.array([m["precursor_mz"] for m in test_mols.values()], dtype=np.float64)
    test_mids = list(test_mols.keys())
    
    # 2. Stream train.parquet row groups
    pf_train = pq.ParquetFile(train_path)
    num_rg = pf_train.num_row_groups
    print(f"Train dataset: {pf_train.metadata.num_rows:,} spectra across {num_rg} row groups.")
    
    cols = ["normalized_smiles", "precursor_mz", "ionization_mode", "ms2_mzs", "ms2_normalized_intensities"]
    
    best_matches = {}  # mid -> {"cosine": float, "smiles": str, "train_pm": float}
    
    for rg_idx in range(num_rg):
        t_rg = time.time()
        rg_table = pf_train.read_row_group(rg_idx, columns=cols)
        
        tr_pms = rg_table["precursor_mz"].to_numpy()
        tr_ions = [str(x).lower() for x in rg_table["ionization_mode"].to_pylist()]
        tr_smiles = rg_table["normalized_smiles"].to_pylist()
        tr_mzs = rg_table["ms2_mzs"]
        tr_ints = rg_table["ms2_normalized_intensities"]
        
        hits_this_rg = 0
        for t_i, mid in enumerate(test_mids):
            t_pm = test_pms[t_i]
            t_ion = test_mols[mid]["ion"]
            
            # Fast vectorized precursor mz filter (within 10 ppm)
            diff_ppm = np.abs(tr_pms - t_pm) / t_pm * 1e6
            cand_indices = np.where(diff_ppm < 10.0)[0]
            
            if len(cand_indices) == 0:
                continue
                
            for c_idx in cand_indices:
                if tr_ions[c_idx] != t_ion:
                    continue
                    
                c_mzs = np.array(tr_mzs[c_idx].as_py(), dtype=np.float32)
                c_ints = np.array(tr_ints[c_idx].as_py(), dtype=np.float32)
                
                # Check cosine against test spectra
                for t_m, t_int in test_mols[mid]["spectra"]:
                    cos = compute_spectral_cosine(t_m, t_int, c_mzs, c_ints, tolerance_da=0.05)
                    if cos > 0.75:
                        hits_this_rg += 1
                        prev_cos = best_matches.get(mid, {}).get("cosine", 0.0)
                        if cos > prev_cos:
                            best_matches[mid] = {
                                "cosine": round(cos, 4),
                                "smiles": tr_smiles[c_idx],
                                "train_pm": float(tr_pms[c_idx]),
                                "test_pm": float(t_pm),
                                "ppm_diff": round(float(abs(tr_pms[c_idx] - t_pm) / t_pm * 1e6), 2)
                            }
                            
        print(f"Row Group {rg_idx+1:2d}/{num_rg} processed in {time.time()-t_rg:.1f}s | Cumulative hits (Cosine > 0.75): {len(best_matches)}")
        
    print("\n" + "=" * 70)
    print(f"SPECTRAL SEARCH COMPLETE ({time.time()-t0:.1f}s)")
    print("=" * 70)
    
    # Filter high confidence (cosine >= 0.82)
    high_conf = {k: v for k, v in best_matches.items() if v["cosine"] >= 0.82}
    print(f"Total Tier 1 Matches with Cosine >= 0.82: {len(high_conf)}/{len(test_mols)} test molecules!")
    
    out_json = ROOT / "models" / "library_matches.json"
    with open(out_json, "w") as f:
        json.dump(best_matches, f, indent=2)
    print(f"✓ Saved library matches to {out_json}")
    
    # Print sample matches
    print("\nSample High-Confidence Matches:")
    for mid, info in list(high_conf.items())[:10]:
        print(f"  {mid}: Cosine={info['cosine']:.4f}, ppm={info['ppm_diff']} -> {info['smiles'][:50]}...")

if __name__ == "__main__":
    main()
