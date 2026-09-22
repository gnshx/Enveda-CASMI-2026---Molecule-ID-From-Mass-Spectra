"""
candidates/extract_pubchem_test_candidates.py

High-performance extraction of candidates from PubChem (8.99M) + Existing DB (729k).
Filters candidates to match test neutral mass windows (±25 ppm) across 400 test molecules.
Saves batches incrementally to candidates/pubchem_matches/ for zero data loss.
"""

import os
import sys
import time
import bisect
import shutil
from pathlib import Path
from multiprocessing import Pool
from typing import List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
PUBCHEM_PATH = ROOT / "data" / "references" / "pubchem_10m.parquet"
EXISTING_DB_PATH = ROOT / "candidates" / "candidate_db.parquet"
TEST_PATH = ROOT / "data" / "test.parquet"
PARTIAL_DIR = ROOT / "candidates" / "pubchem_matches"
OUT_PATH = ROOT / "candidates" / "candidate_db_v2.parquet"

ADDUCT_SHIFTS = {
    '[M+H]+': -1.007276,
    '[M-H]-': +1.007276,
    '[M+Na]+': -22.989218,
    '[M+NH4]+': -18.033823,
    '[M+K]+': -38.963158,
    '[M+Cl]-': -34.969402,
    '[M+CH2O2-H]-': -44.997655,
}

def get_test_intervals(ppm_tol: float = 25.0) -> List[Tuple[float, float]]:
    tbl = pq.read_table(TEST_PATH)
    df = tbl.to_pandas().drop_duplicates("molecule_id")
    
    raw_intervals = []
    for _, r in df.iterrows():
        pm = float(r["precursor_mz"])
        add = str(r["adduct"])
        ion = str(r["ionization_mode"]).lower()
        shift = ADDUCT_SHIFTS.get(add, -1.007276 if ion.startswith("pos") else +1.007276)
        nm = pm + shift
        delta = nm * ppm_tol * 1e-6
        raw_intervals.append((nm - delta, nm + delta))
        
    raw_intervals.sort()
    merged = []
    for low, high in raw_intervals:
        if not merged:
            merged.append([low, high])
        else:
            if low <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], high)
            else:
                merged.append([low, high])
                
    return [(low, high) for low, high in merged]

INTERVAL_LOWS = []
INTERVAL_HIGHS = []

def init_worker(lows: List[float], highs: List[float]):
    global INTERVAL_LOWS, INTERVAL_HIGHS
    INTERVAL_LOWS = lows
    INTERVAL_HIGHS = highs

def is_mass_in_intervals(mass: float) -> bool:
    idx = bisect.bisect_right(INTERVAL_LOWS, mass) - 1
    if idx >= 0:
        if mass <= INTERVAL_HIGHS[idx]:
            return True
    return False

def process_smiles_batch(smiles_chunk: List[str]) -> List[Tuple[str, str, float]]:
    hits = []
    for smi in smiles_chunk:
        if len(smi) < 12 or len(smi) > 100:
            continue
        try:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            mass = Descriptors.ExactMolWt(m)
            if is_mass_in_intervals(mass):
                ik14 = Chem.MolToInchiKey(m)[:14]
                can_smi = Chem.MolToSmiles(m, canonical=True)
                hits.append((can_smi, ik14, float(mass)))
        except Exception:
            continue
    return hits

def main():
    t0 = time.time()
    print("=" * 70)
    print("EXTRACTING CANDIDATES MATCHING TEST PRECURSOR WINDOWS")
    print("=" * 70)
    
    intervals = get_test_intervals(ppm_tol=25.0)
    lows = [low for low, high in intervals]
    highs = [high for low, high in intervals]
    print(f"Constructed {len(intervals)} merged test mass intervals spanning [{lows[0]:.2f}, {highs[-1]:.2f}] Da.")

    PARTIAL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load existing DB (COCONUT + Train) within intervals
    print("\n[Step 1/3] Filtering existing candidate database (COCONUT + Train)...")
    existing_tbl = pq.read_table(EXISTING_DB_PATH)
    existing_df = existing_tbl.to_pandas()
    
    existing_masses = existing_df["exact_mass"].to_numpy()
    keep_mask = np.zeros(len(existing_masses), dtype=bool)
    for low, high in intervals:
        l_idx = np.searchsorted(existing_masses, low)
        r_idx = np.searchsorted(existing_masses, high)
        keep_mask[l_idx:r_idx] = True
        
    filtered_existing = existing_df[keep_mask].copy()
    print(f"  Existing DB has {len(filtered_existing):,} molecules matching test intervals.")
    
    existing_hits = []
    for _, r in filtered_existing.iterrows():
        existing_hits.append((r["canonical_smiles"], r["inchikey14"], float(r["exact_mass"]), str(r["source"])))

    # 2. Process PubChem 9M using Multiprocessing Pool
    print("\n[Step 2/3] Streaming and filtering PubChem 9M in parallel...")
    pf_pub = pq.ParquetFile(str(PUBCHEM_PATH))
    num_rg = pf_pub.num_row_groups
    total_smiles = pf_pub.metadata.num_rows
    print(f"  PubChem has {num_rg} row groups ({total_smiles:,} SMILES).")
    
    num_workers = min(18, os.cpu_count() or 4)
    print(f"  Using {num_workers} parallel CPU workers.")
    
    CHUNK_SIZE = 50  # 50 row groups = ~500,000 SMILES per batch
    total_pubchem_hits = 0
    
    with Pool(processes=num_workers, initializer=init_worker, initargs=(lows, highs)) as pool:
        for batch_idx, start_rg in enumerate(range(0, num_rg, CHUNK_SIZE)):
            batch_file = PARTIAL_DIR / f"batch_{batch_idx:03d}.parquet"
            if batch_file.exists():
                batch_tbl = pq.read_table(batch_file)
                total_pubchem_hits += batch_tbl.num_rows
                processed_smiles = min((start_rg + CHUNK_SIZE) * 10000, total_smiles)
                pct = processed_smiles / total_smiles * 100
                print(f"  [{pct:5.1f}%] Batch {batch_idx:02d} already exists ({batch_tbl.num_rows:,} hits). Total so far: {total_pubchem_hits:,}", flush=True)
                continue

            end_rg = min(start_rg + CHUNK_SIZE, num_rg)
            batch_smiles = []
            for rg in range(start_rg, end_rg):
                tbl = pf_pub.read_row_group(rg, columns=["smiles"])
                batch_smiles.extend(tbl["smiles"].to_pylist())
                
            worker_chunk_size = max(1, len(batch_smiles) // num_workers)
            chunks = [batch_smiles[i:i + worker_chunk_size] for i in range(0, len(batch_smiles), worker_chunk_size)]
            
            results = pool.map(process_smiles_batch, chunks)
            batch_hits = []
            for res in results:
                for can_smi, ik14, mass in res:
                    batch_hits.append((can_smi, ik14, mass, "pubchem"))
                    
            if batch_hits:
                df_batch = pd.DataFrame(batch_hits, columns=["canonical_smiles", "inchikey14", "exact_mass", "source"])
                df_batch.to_parquet(batch_file, index=False)
                total_pubchem_hits += len(batch_hits)

            processed_smiles = min((start_rg + CHUNK_SIZE) * 10000, total_smiles)
            pct = processed_smiles / total_smiles * 100
            print(f"  [{pct:5.1f}%] Processed {processed_smiles:,} SMILES | Total matching PubChem candidates: {total_pubchem_hits:,} | Elapsed: {time.time()-t0:.1f}s", flush=True)

    # 3. Combine, Deduplicate, and Save
    print("\n[Step 3/3] Combining and deduplicating candidate database...")
    all_dfs = [pd.DataFrame(existing_hits, columns=["canonical_smiles", "inchikey14", "exact_mass", "source"])]
    for bf in sorted(PARTIAL_DIR.glob("batch_*.parquet")):
        all_dfs.append(pq.read_table(bf).to_pandas())
        
    df_all = pd.concat(all_dfs, ignore_index=True)
    print(f"  Total candidate matches before deduplication: {len(df_all):,}")
    
    source_priority = {"coconut": 1, "train": 2, "pubchem": 3}
    df_all["prio"] = df_all["source"].map(source_priority).fillna(4)
    df_all = df_all.sort_values(["exact_mass", "prio"]).drop_duplicates(subset=["inchikey14"]).drop(columns=["prio"])
    df_all = df_all.sort_values("exact_mass").reset_index(drop=True)
    
    print(f"  Final Unique Candidate Database Size: {len(df_all):,} molecules.")
    print(f"  Source breakdown: {df_all['source'].value_counts().to_dict()}")
    
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(OUT_PATH, index=False, compression="zstd")
    file_size_mb = OUT_PATH.stat().st_size / (1024 * 1024)
    print(f"✓ Saved candidate database to {OUT_PATH} ({file_size_mb:.2f} MB).")
    print(f"✓ Total extraction time: {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
