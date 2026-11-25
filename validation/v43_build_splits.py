#!/usr/bin/env python3
"""
validation/v43_build_splits.py
=============================================================================
CASMI 2026 — V43 Phase 2: Molecule-Grouped, Tautomer-Deduplicated Splits

Builds leak-free validation folds from data/train.parquet:
  1. Canonicalizes every unique normalized_smiles via RDKit TautomerEnumerator
     to obtain canonical tautomer InChIKey14 (`canon_ik14`) and `exact_mass`.
  2. Identifies any `canon_ik14` present in `enveda-np-examples` as
     `np_examples_purged` (noting cross-library overlap so evaluation can purge
     all matching spectra across ALL ingest_libs).
  3. Samples an untouched `timstof_holdout` (1,000 molecules from `enveda-180`
     with TEST_ADDUCTS spectra, stratified by mass & ionization mode, disjoint
     by `canon_ik14` from `np_examples_purged` and all CV folds).
  4. Assigns all remaining `canon_ik14` groups to 5 deterministic molecule-grouped
     CV folds (`fold` 0..4).

Outputs:
  - data/validation_folds/v43_molecules.parquet
  - data/validation_folds/v43_split_summary.json
=============================================================================
"""

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chemistry.ms_engine import TEST_ADDUCTS

_TAUT = None


def _init_worker():
    global _TAUT
    from rdkit import RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize

    RDLogger.DisableLog("rdApp.*")
    _TAUT = rdMolStandardize.TautomerEnumerator()
    _TAUT.SetMaxTautomers(100)
    _TAUT.SetMaxTransforms(100)


def _canon_batch(smiles_list: List[str]) -> List[Tuple[str, str, float]]:
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    out = []
    for smi in smiles_list:
        canon_ik14 = ""
        exact_mass = float("nan")
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                exact_mass = float(Descriptors.ExactMolWt(mol))
                try:
                    cmol = _TAUT.Canonicalize(mol)
                except Exception:
                    cmol = mol
                ik = Chem.MolToInchiKey(cmol)
                if ik and len(ik) >= 14:
                    canon_ik14 = ik[:14]
        except Exception:
            pass
        out.append((smi, canon_ik14, exact_mass))
    return out


def deterministic_fold(canon_ik14: str, n_folds: int = 5) -> int:
    digest = hashlib.blake2b(canon_ik14.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_folds


def main():
    t0 = time.time()
    train_path = ROOT / "data" / "train.parquet"
    out_dir = ROOT / "data" / "validation_folds"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V43 SPLIT BUILDER: Molecule-Grouped & Tautomer-Deduplicated Validation Folds")
    print("=" * 80)

    pf = pq.ParquetFile(str(train_path))
    cols = ["ingest_lib", "normalized_smiles", "inchikey14", "molecular_formula", "adduct", "ionization_mode", "instrument_type"]

    mol_records: Dict[str, dict] = {}
    unique_smiles: set[str] = set()

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        unique_smiles.update(df["normalized_smiles"].dropna().astype(str).unique())

        df["in_test_adduct"] = df["adduct"].isin(TEST_ADDUCTS)
        df["is_timstof"] = df["instrument_type"].fillna("").eq("timsTOF") | df["ingest_lib"].isin(["enveda-180", "enveda-np-examples"])

        for row in df.itertuples(index=False):
            smi = str(row.normalized_smiles)
            rec = mol_records.get(smi)
            if rec is None:
                rec = {
                    "normalized_smiles": smi,
                    "raw_ik14": str(row.inchikey14),
                    "molecular_formula": str(row.molecular_formula),
                    "n_spectra": 0,
                    "n_test_adduct_spectra": 0,
                    "has_timstof": False,
                    "in_enveda_180": False,
                    "in_enveda_np": False,
                    "has_test_adduct_timstof": False,
                    "libs": set(),
                }
                mol_records[smi] = rec
            rec["n_spectra"] += 1
            if row.in_test_adduct:
                rec["n_test_adduct_spectra"] += 1
            if row.is_timstof:
                rec["has_timstof"] = True
                if row.in_test_adduct:
                    rec["has_test_adduct_timstof"] = True
            lib = str(row.ingest_lib)
            rec["libs"].add(lib)
            if lib == "enveda-180":
                rec["in_enveda_180"] = True
            elif lib == "enveda-np-examples":
                rec["in_enveda_np"] = True

    print(f"Scanned {pf.metadata.num_rows:,} spectra across {len(mol_records):,} unique normalized_smiles ({time.time()-t0:.1f}s)")

    # Parallel RDKit canonical tautomer InChIKey14 computation
    smi_list = sorted(unique_smiles)
    n_workers = min(16, os.cpu_count() or 4)
    chunk_size = max(500, len(smi_list) // (n_workers * 4))
    chunks = [smi_list[i : i + chunk_size] for i in range(0, len(smi_list), chunk_size)]

    smi_to_canon: Dict[str, Tuple[str, float]] = {}
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as ex:
        for res in ex.map(_canon_batch, chunks):
            for smi, cik, mass in res:
                smi_to_canon[smi] = (cik, mass)

    print(f"Canonicalized {len(smi_to_canon):,} SMILES via RDKit TautomerEnumerator ({time.time()-t0:.1f}s)")

    rows = []
    tautomer_merges = 0
    for smi, rec in mol_records.items():
        cik, mass = smi_to_canon[smi]
        if not cik:
            cik = rec["raw_ik14"]
        if cik != rec["raw_ik14"]:
            tautomer_merges += 1
        rows.append({
            "canon_ik14": cik,
            "raw_ik14": rec["raw_ik14"],
            "normalized_smiles": smi,
            "molecular_formula": rec["molecular_formula"],
            "exact_mass": mass,
            "n_spectra": rec["n_spectra"],
            "n_test_adduct_spectra": rec["n_test_adduct_spectra"],
            "has_timstof": rec["has_timstof"],
            "has_test_adduct_timstof": rec["has_test_adduct_timstof"],
            "in_enveda_180": rec["in_enveda_180"],
            "in_enveda_np": rec["in_enveda_np"],
            "n_libs": len(rec["libs"]),
            "ingest_libs": ",".join(sorted(rec["libs"])),
        })

    mol_df = pd.DataFrame(rows)
    # Aggregate group-level flags per canon_ik14 so tautomer groups stay atomic
    grp_np = set(mol_df.loc[mol_df["in_enveda_np"], "canon_ik14"])
    grp_e180 = set(mol_df.loc[mol_df["in_enveda_180"] & mol_df["has_test_adduct_timstof"], "canon_ik14"])
    # Untouched timsTOF holdout candidates: in enveda-180 (timsTOF + TEST_ADDUCTS), NOT in enveda-np-examples
    holdout_pool = mol_df[
        mol_df["canon_ik14"].isin(grp_e180 - grp_np)
        & mol_df["exact_mass"].notna()
    ].drop_duplicates("canon_ik14").copy()

    # Stratified sample of 1,000 unique canon_ik14 across 10 mass deciles
    holdout_pool["mass_bin"] = pd.qcut(holdout_pool["exact_mass"], q=10, labels=False, duplicates="drop")
    rng = np.random.default_rng(20261125)
    holdout_iks = set()
    per_bin = 1000 // holdout_pool["mass_bin"].nunique()
    for _, bdf in holdout_pool.groupby("mass_bin"):
        chosen = rng.choice(bdf["canon_ik14"].to_numpy(), size=min(per_bin, len(bdf)), replace=False)
        holdout_iks.update(chosen.tolist())
    # Top up to exact 1,000 if needed
    if len(holdout_iks) < 1000:
        rem = holdout_pool[~holdout_pool["canon_ik14"].isin(holdout_iks)]["canon_ik14"].to_numpy()
        extra = rng.choice(rem, size=1000 - len(holdout_iks), replace=False)
        holdout_iks.update(extra.tolist())

    def assign_role(cik: str) -> str:
        if cik in grp_np:
            return "np_examples_purged"
        if cik in holdout_iks:
            return "timstof_holdout"
        return "train_cv"

    mol_df["split_role"] = mol_df["canon_ik14"].map(assign_role)
    mol_df["fold"] = np.where(
        mol_df["split_role"] == "train_cv",
        mol_df["canon_ik14"].map(deterministic_fold),
        -1,
    )

    # Zero-leakage assertions across canon_ik14
    role_sets = {r: set(g["canon_ik14"]) for r, g in mol_df.groupby("split_role")}
    assert role_sets["train_cv"].isdisjoint(role_sets["timstof_holdout"]), "Leakage: train_cv vs timstof_holdout!"
    assert role_sets["train_cv"].isdisjoint(role_sets["np_examples_purged"]), "Leakage: train_cv vs np_examples_purged!"
    assert role_sets["timstof_holdout"].isdisjoint(role_sets["np_examples_purged"]), "Leakage: holdout vs np!"

    out_parquet = out_dir / "v43_molecules.parquet"
    mol_df.to_parquet(out_parquet, index=False)

    summary = {
        "total_spectra_scanned": int(pf.metadata.num_rows),
        "unique_normalized_smiles": int(len(mol_df)),
        "unique_raw_ik14": int(mol_df["raw_ik14"].nunique()),
        "unique_canon_ik14": int(mol_df["canon_ik14"].nunique()),
        "smiles_canonicalized_differently_from_raw_ik14": int(tautomer_merges),
        "split_counts_smiles": mol_df["split_role"].value_counts().to_dict(),
        "split_counts_canon_ik14": {r: int(g["canon_ik14"].nunique()) for r, g in mol_df.groupby("split_role")},
        "fold_counts_canon_ik14": {
            int(f): int(g["canon_ik14"].nunique())
            for f, g in mol_df[mol_df["split_role"] == "train_cv"].groupby("fold")
        },
        "timstof_holdout_pure_enveda180_only": int(
            mol_df[(mol_df["split_role"] == "timstof_holdout") & (mol_df["n_libs"] == 1)]["canon_ik14"].nunique()
        ),
        "timstof_holdout_also_in_external_train_libs": int(
            mol_df[(mol_df["split_role"] == "timstof_holdout") & (mol_df["n_libs"] > 1)]["canon_ik14"].nunique()
        ),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (out_dir / "v43_split_summary.json").write_text(json.dumps(summary, indent=2))
    print("✓ Saved:", out_parquet)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
