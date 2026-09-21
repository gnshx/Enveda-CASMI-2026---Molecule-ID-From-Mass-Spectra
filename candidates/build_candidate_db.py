"""
candidates/build_candidate_db.py

Builds unified high-precision candidate database from:
1. COCONUT September 2026 (738k natural products)
2. Train.parquet unique structures (277k structures)

Deduplicates by inchikey14 and canonical SMILES, sorted by exact_mass for O(log N) binary search.
"""

import time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
from rdkit import Chem
from rdkit.Chem import Descriptors

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
REF_DIR = ROOT / "data" / "references"
OUT_PATH = ROOT / "candidates" / "candidate_db.parquet"

T0 = time.time()
def log(msg: str):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


def main():
    log("=== Building Unified Natural Products Candidate Database ===")
    
    # 1. Load COCONUT
    coco_path = REF_DIR / "coconut_csv_lite-09-2026.csv"
    log(f"Loading COCONUT from {coco_path}...")
    coco_cols = ["canonical_smiles", "standard_inchi_key", "exact_molecular_weight", "molecular_formula"]
    coco = pd.read_csv(coco_path, usecols=coco_cols).dropna(subset=["canonical_smiles", "exact_molecular_weight"])
    coco["inchikey14"] = coco["standard_inchi_key"].str.slice(0, 14)
    coco = coco.rename(columns={"exact_molecular_weight": "exact_mass"})
    coco["source"] = "coconut"
    log(f"  Loaded {len(coco):,} COCONUT natural products.")

    # 2. Load Train Structures
    train_path = ROOT / "data" / "train.parquet"
    log(f"Extracting unique training structures from {train_path}...")
    pf = pq.ParquetFile(str(train_path))
    train_rows = []
    seen_train_keys = set()
    for rg in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg, columns=["normalized_smiles", "inchikey14", "molecular_formula", "precursor_mz", "adduct"])
        df_rg = tbl.to_pandas()
        for _, r in df_rg.iterrows():
            k = r["inchikey14"]
            if k not in seen_train_keys and pd.notna(r["normalized_smiles"]):
                seen_train_keys.add(k)
                train_rows.append({
                    "canonical_smiles": r["normalized_smiles"],
                    "inchikey14": k,
                    "molecular_formula": r["molecular_formula"],
                    "source": "train",
                })
    train_df = pd.DataFrame(train_rows)
    log(f"  Extracted {len(train_df):,} unique training molecules.")

    # Compute exact masses for train molecules missing in COCONUT
    coco_keys = set(coco["inchikey14"])
    missing_train = train_df[~train_df["inchikey14"].isin(coco_keys)].copy()
    log(f"  {len(missing_train):,} training molecules need exact mass calculation...")
    
    exact_masses = []
    valid_mask = []
    for smi in missing_train["canonical_smiles"]:
        try:
            m = Chem.MolFromSmiles(smi)
            if m is not None:
                mass = Descriptors.ExactMolWt(m)
                exact_masses.append(mass)
                valid_mask.append(True)
            else:
                exact_masses.append(0.0)
                valid_mask.append(False)
        except Exception:
            exact_masses.append(0.0)
            valid_mask.append(False)
            
    missing_train["exact_mass"] = exact_masses
    missing_train = missing_train[valid_mask]

    # 3. Combine and Deduplicate
    log("Combining and deduplicating...")
    cols = ["canonical_smiles", "inchikey14", "exact_mass", "molecular_formula", "source"]
    combined = pd.concat([coco[cols], missing_train[cols]], ignore_index=True)
    
    # Deduplicate by inchikey14 (keeping first)
    combined = combined.drop_duplicates(subset=["inchikey14"]).dropna(subset=["exact_mass", "canonical_smiles"])
    
    # Sort by exact_mass for fast binary search
    combined["exact_mass"] = combined["exact_mass"].astype(np.float64)
    combined = combined.sort_values("exact_mass").reset_index(drop=True)
    
    log(f"Unified Candidate DB size: {len(combined):,} unique molecules.")
    log(f"Mass range: {combined['exact_mass'].min():.2f} to {combined['exact_mass'].max():.2f} Da.")

    # 4. Save to Parquet
    combined.to_parquet(str(OUT_PATH), index=False)
    log(f"✓ Saved unified candidate database to {OUT_PATH} ({len(combined):,} rows).")


if __name__ == "__main__":
    main()
