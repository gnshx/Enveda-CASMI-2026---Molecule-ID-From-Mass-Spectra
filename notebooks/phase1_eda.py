"""
Phase 1 — EDA + Data Validation
Run: python notebooks/phase1_eda.py
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR   = Path("/home/dlcv/Enveda-CASMI-2026/data")
TRAIN_PATH = DATA_DIR / "train.parquet"
TEST_PATH  = DATA_DIR / "test.parquet"

print("=" * 60)
print("CASMI 2026 — Phase 1 EDA")
print("=" * 60)

print("\n[1] Loading data...")
train = pd.read_parquet(TRAIN_PATH)
test  = pd.read_parquet(TEST_PATH)
print(f"  train shape : {train.shape}")
print(f"  test  shape : {test.shape}")

print("\n[2] Train columns (dtype + null%):")
for col in train.columns:
    pct = train[col].isna().sum() / len(train) * 100
    print(f"  {col:<35} {str(train[col].dtype):<15} nulls={pct:.1f}%")

print("\n[3] Test columns (null%):")
for col in test.columns:
    pct = test[col].isna().sum() / len(test) * 100
    print(f"  {col:<35} nulls={pct:.1f}%")

print("\n[4] Molecule / spectrum counts:")
print(f"  Train unique molecules : {train['molecule_id'].nunique():,}")
print(f"  Train unique spectra   : {len(train):,}")
print(f"  Test  unique molecules : {test['molecule_id'].nunique():,}")
print(f"  Test  unique spectra   : {len(test):,}")

train_spm = train.groupby("molecule_id").size()
test_spm  = test.groupby("molecule_id").size()
print(f"  Train spectra/mol: min={train_spm.min()} median={train_spm.median():.0f} max={train_spm.max()}")
print(f"  Test  spectra/mol: min={test_spm.min()} median={test_spm.median():.0f} max={test_spm.max()}")

print("\n[5] Test adduct distribution:")
TEST_SET_ADDUCTS = {"[M+H]+","[M+NH4]+","[M-H2O+H]+","[M-2H2O+H]+","[M+Na]+","[M+K]+","[M-H]-","[M-H2O-H]-","[M+CH2O2-H]-","[M+Cl]-"}
for adduct, cnt in test["adduct"].value_counts().items():
    print(f"  {adduct:<25} {cnt:>5}  ({cnt/len(test)*100:.1f}%)")
unexpected = set(test["adduct"].unique()) - TEST_SET_ADDUCTS
print(f"  Unexpected adducts: {unexpected if unexpected else 'none ✓'}")

print("\n[6] Training library breakdown:")
lib = train.groupby("ingest_lib").agg(n_spectra=("spectrum_id","count"), n_mols=("molecule_id","nunique")).sort_values("n_spectra",ascending=False)
print(lib.to_string())

print("\n[7] Precursor m/z ranges:")
for label, df in [("train", train), ("test", test)]:
    pmz = df["precursor_mz"]
    print(f"  {label}: min={pmz.min():.1f}  median={pmz.median():.1f}  max={pmz.max():.1f}")

print("\n[8] Test ionization modes:")
print(test["ionization_mode"].value_counts().to_string())

if "instrument_type" in test.columns:
    print("\n[9] Test instrument_type:")
    print(test["instrument_type"].value_counts().to_string())

if "num_peaks" in train.columns:
    np_col = train["num_peaks"].dropna()
    print(f"\n[10] Train num_peaks: min={np_col.min():.0f}  median={np_col.median():.0f}  max={np_col.max():.0f}  mean={np_col.mean():.1f}")

test_np = test["ms2_mzs"].apply(lambda x: len(x) if x is not None else 0)
print(f"  Test n_peaks: min={test_np.min()}  median={test_np.median():.0f}  max={test_np.max()}  mean={test_np.mean():.1f}")

if "base_peak_intensity" in test.columns:
    bp = test["base_peak_intensity"].dropna()
    if len(bp):
        print(f"\n[11] Test base_peak_intensity: n_valid={len(bp)}  min={bp.min():.0f}  median={bp.median():.0f}  max={bp.max():.0f}")
        print(f"  Below 1000 (noisy): {(bp < 1000).sum()}")

if "normalized_smiles" in train.columns:
    print(f"\n[12] Train normalized_smiles: {train['normalized_smiles'].notna().sum():,} valid, {train['normalized_smiles'].nunique():,} unique")

np_ex = train[train["ingest_lib"] == "enveda-np-examples"]
print(f"\n[13] enveda-np-examples (gold NP subset): {len(np_ex)} spectra, {np_ex['molecule_id'].nunique()} molecules")

# Collision energy
print("\n[14] Test collision_energy_ev samples (first 10 rows):")
print(test["collision_energy_ev"].head(10).to_string())

print("\n" + "=" * 60)
print("EDA COMPLETE — proceeding to cosine retrieval baseline")
print("=" * 60)
