"""
Enveda CASMI 2026 - Official Kaggle Submission Notebook
Phase 1 Baseline: Numba-Accelerated Cosine + Modified Cosine Spectral Retrieval

Instructions for Kaggle:
1. Create a New Notebook on the competition page.
2. In Notebook Settings (right sidebar):
   - Environment: Python 3
   - Internet: Off (as required by competition rules)
   - Accelerator: None (CPU is plenty fast)
3. Paste this entire script into a single code cell.
4. Click 'Save Version' -> 'Save & Run All (Commit)'.
5. Once complete, click 'Submit to Competition'.
"""

import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numba
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.compute as pc

warnings.filterwarnings("ignore")

import glob

# ── Dynamic Path Detection (Works with any Kaggle casing or local run) ────────
train_matches = glob.glob("/kaggle/input/**/train.parquet", recursive=True)
test_matches = glob.glob("/kaggle/input/**/test.parquet", recursive=True)

if train_matches and test_matches:
    TRAIN_PATH = Path(train_matches[0])
    TEST_PATH = Path(test_matches[0])
    SUB_PATH = Path("/kaggle/working/submission.csv")
    print(f"Found Kaggle input paths:\n  Train: {TRAIN_PATH}\n  Test:  {TEST_PATH}")
else:
    # Fallback to local server paths
    DATA_DIR = Path("/home/dlcv/Enveda-CASMI-2026/data")
    TRAIN_PATH = DATA_DIR / "train.parquet"
    TEST_PATH = DATA_DIR / "test.parquet"
    SUB_PATH = Path("/home/dlcv/Enveda-CASMI-2026/submission.csv")

T0 = time.time()
def log(msg: str):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)



# ── 1. Numba Kernels for Fast Spectral Matching ──────────────────────────────

@numba.njit(fastmath=True)
def numba_cosine(qmzs: np.ndarray, qints: np.ndarray, rmzs: np.ndarray, rints: np.ndarray, mz_tol: float = 0.05):
    """Greedy 2-pointer cosine similarity on sorted peak arrays."""
    nq = len(qmzs)
    nr = len(rmzs)
    if nq == 0 or nr == 0:
        return 0.0, 0

    q_idx = 0
    r_idx = 0
    dot = 0.0
    matched = 0

    while q_idx < nq and r_idx < nr:
        diff = qmzs[q_idx] - rmzs[r_idx]
        if abs(diff) <= mz_tol:
            dot += qints[q_idx] * rints[r_idx]
            matched += 1
            q_idx += 1
            r_idx += 1
        elif diff < 0:
            q_idx += 1
        else:
            r_idx += 1

    norm_q = 0.0
    for i in range(nq): norm_q += qints[i] * qints[i]
    norm_r = 0.0
    for i in range(nr): norm_r += rints[i] * rints[i]
    denom = np.sqrt(norm_q * norm_r)
    score = dot / denom if denom > 0 else 0.0
    return score, matched


@numba.njit(fastmath=True)
def numba_modified_cosine(qmzs: np.ndarray, qints: np.ndarray, qpmz: float,
                           rmzs: np.ndarray, rints: np.ndarray, rpmz: float,
                           mz_tol: float = 0.05):
    """Modified cosine accounting for precursor mass shift."""
    nq = len(qmzs)
    nr = len(rmzs)
    if nq == 0 or nr == 0:
        return 0.0, 0

    shift = qpmz - rpmz
    used_r = np.zeros(nr, dtype=np.bool_)
    dot = 0.0
    matched = 0

    # 1. Direct matches
    for i in range(nq):
        q_mz = qmzs[i]
        best_diff = mz_tol + 1.0
        best_j = -1
        for j in range(nr):
            if not used_r[j]:
                d = abs(q_mz - rmzs[j])
                if d <= mz_tol and d < best_diff:
                    best_diff = d
                    best_j = j
        if best_j >= 0:
            dot += qints[i] * rints[best_j]
            used_r[best_j] = True
            matched += 1

    # 2. Shifted matches
    if abs(shift) > mz_tol:
        for i in range(nq):
            q_mz = qmzs[i]
            best_diff = mz_tol + 1.0
            best_j = -1
            for j in range(nr):
                if not used_r[j]:
                    d = abs((q_mz - rmzs[j]) - shift)
                    if d <= mz_tol and d < best_diff:
                        best_diff = d
                        best_j = j
            if best_j >= 0:
                dot += qints[i] * rints[best_j]
                used_r[best_j] = True
                matched += 1

    norm_q = 0.0
    for i in range(nq): norm_q += qints[i] * qints[i]
    norm_r = 0.0
    for i in range(nr): norm_r += rints[i] * rints[i]
    denom = np.sqrt(norm_q * norm_r)
    score = dot / denom if denom > 0 else 0.0
    return score, matched


# Warmup JIT
_d_m = np.array([100.0, 200.0], dtype=np.float32)
_d_i = np.array([0.5, 1.0], dtype=np.float32)
numba_cosine(_d_m, _d_i, _d_m, _d_i)
numba_modified_cosine(_d_m, _d_i, 200.0, _d_m, _d_i, 200.0)


# ── 2. Peak Preprocessing ────────────────────────────────────────────────────

def preprocess_peaks(mzs, ints, precursor_mz: float,
                     precursor_buffer: float = 2.0,
                     int_thresh: float = 0.005,
                     max_peaks: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    mzs = np.asarray(mzs, dtype=np.float32)
    ints = np.asarray(ints, dtype=np.float32)
    if len(mzs) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    # Base peak normalization
    mx = ints.max()
    if mx > 0: ints = ints / mx

    # Filter precursor buffer
    keep = mzs <= (precursor_mz + precursor_buffer)
    mzs, ints = mzs[keep], ints[keep]

    # Filter intensity floor
    keep = ints >= int_thresh
    mzs, ints = mzs[keep], ints[keep]

    if len(mzs) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    # Top-K peaks
    if len(mzs) > max_peaks:
        top_idx = np.argsort(ints)[-max_peaks:]
        mzs, ints = mzs[top_idx], ints[top_idx]

    # Sqrt transform & re-normalize
    ints = np.sqrt(ints)
    mx = ints.max()
    if mx > 0: ints = ints / mx

    # Sort by m/z
    sort_idx = np.argsort(mzs)
    return mzs[sort_idx], ints[sort_idx]


# ── 3. Reference Library Indexer ─────────────────────────────────────────────

def build_reference_index(train_path: Path) -> Dict[str, Any]:
    log("Reading train.parquet...")
    pf = pq.ParquetFile(str(train_path))
    
    COLS = [
        "ingest_lib", "normalized_smiles", "inchikey14", "precursor_mz",
        "ms2_mzs", "ms2_normalized_intensities", "num_peaks"
    ]
    
    INDEX_LIBS = {
        "pluskal_ms2", "riken", "gnps", "massbank", "mona",
        "msdial", "spectraverse", "drug_plus", "masaryk", "enveda-np-examples"
    }
    
    ref_pmzs, ref_mzs, ref_ints, ref_inchikey, ref_smiles = [], [], [], [], []

    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=COLS)
        
        # Filter: natural product libraries, precursor mass 180-520 Da, >= 3 peaks
        ref_mask = pc.and_(
            pc.and_(pc.greater_equal(tbl["precursor_mz"], 180.0), pc.less_equal(tbl["precursor_mz"], 520.0)),
            pc.and_(pc.greater_equal(tbl["num_peaks"], 3), pc.is_in(tbl["ingest_lib"], value_set=pa.array(list(INDEX_LIBS))))
        )
        filtered = tbl.filter(ref_mask)
        if len(filtered) > 0:
            df_ref = filtered.to_pandas()
            for _, r in df_ref.iterrows():
                pm = float(r["precursor_mz"])
                m, i = preprocess_peaks(r["ms2_mzs"], r["ms2_normalized_intensities"], pm)
                if len(m) >= 3:
                    ref_pmzs.append(pm)
                    ref_mzs.append(m)
                    ref_ints.append(i)
                    ref_inchikey.append(r["inchikey14"])
                    ref_smiles.append(r["normalized_smiles"])

        if (rg_idx + 1) % 7 == 0 or rg_idx == pf.num_row_groups - 1:
            log(f"  Processed {rg_idx+1}/{pf.num_row_groups} row groups ({len(ref_pmzs):,} spectra indexed)...")

    # Sort reference library by precursor_mz for binary search
    log("Sorting reference library by precursor_mz...")
    ref_pmzs = np.array(ref_pmzs, dtype=np.float32)
    sort_order = np.argsort(ref_pmzs)
    
    return {
        "pmzs": ref_pmzs[sort_order],
        "mzs": [ref_mzs[idx] for idx in sort_order],
        "ints": [ref_ints[idx] for idx in sort_order],
        "inchikey14": [ref_inchikey[idx] for idx in sort_order],
        "smiles": [ref_smiles[idx] for idx in sort_order],
    }


# ── 4. Candidate Search & Multi-Spectrum Aggregation ──────────────────────────

def query_library_spectrum(qmzs: np.ndarray, qints: np.ndarray, qpmz: float,
                           ref_index: Dict[str, Any], mz_window: float = 2.0,
                           mz_tol: float = 0.05, min_peaks: int = 3, top_k: int = 100):
    pmzs = ref_index["pmzs"]
    low = np.searchsorted(pmzs, qpmz - mz_window, side="left")
    high = np.searchsorted(pmzs, qpmz + mz_window, side="right")

    if low >= high: return []

    hits = []
    for idx in range(low, high):
        rmzs = ref_index["mzs"][idx]
        rints = ref_index["ints"][idx]
        rpmz = float(pmzs[idx])

        cos, nm = numba_cosine(qmzs, qints, rmzs, rints, mz_tol)
        if cos > 0 and nm >= min_peaks:
            mcos, _ = numba_modified_cosine(qmzs, qints, qpmz, rmzs, rints, rpmz, mz_tol)
            mass_err = abs(qpmz - rpmz) / qpmz * 1e6
            hits.append({
                "inchikey14": ref_index["inchikey14"][idx],
                "smiles": ref_index["smiles"][idx],
                "cosine": cos,
                "mod_cosine": mcos,
                "mass_error_ppm": mass_err,
            })

    if not hits: return []
    hits.sort(key=lambda x: -x["cosine"])
    return hits[:top_k]


def predict_molecule(mol_specs: List[Dict[str, Any]], ref_index: Dict[str, Any], top_n: int = 25) -> List[str]:
    candidate_map = {}
    for spec in mol_specs:
        hits = query_library_spectrum(spec["mzs"], spec["ints"], spec["precursor_mz"], ref_index)
        for h in hits:
            k = h["inchikey14"]
            if k not in candidate_map:
                candidate_map[k] = {
                    "smiles": h["smiles"],
                    "cosines": [],
                    "mod_cosines": [],
                }
            candidate_map[k]["cosines"].append(h["cosine"])
            candidate_map[k]["mod_cosines"].append(h["mod_cosine"])

    if not candidate_map:
        return ["CCO"] * top_n

    # Multi-spectrum composite scoring
    scored = []
    for k, data in candidate_map.items():
        cos_max = max(data["cosines"])
        cos_mean = float(np.mean(data["cosines"]))
        mod_max = max(data["mod_cosines"])
        n_supp = len(data["cosines"])

        score = cos_max + 0.1 * cos_mean + 0.05 * mod_max + 0.02 * min(n_supp, 5)
        scored.append((data["smiles"], score))

    scored.sort(key=lambda x: -x[1])
    slist = [smi for smi, _ in scored if smi]
    if len(slist) < top_n:
        slist.extend(["CCO"] * (top_n - len(slist)))
    return slist[:top_n]


# ── 5. Main Execution on Test Set ────────────────────────────────────────────

def main():
    log("=== Enveda CASMI 2026 Submission Pipeline ===")
    
    ref_index = build_reference_index(TRAIN_PATH)
    log(f"Reference library indexed: {len(ref_index['pmzs']):,} spectra.")

    log(f"Loading test.parquet from {TEST_PATH}...")
    test_df = pd.read_parquet(str(TEST_PATH))
    log(f"Test data: {len(test_df)} spectra, {test_df['molecule_id'].nunique()} unique molecules.")


    predictions = {}
    grouped = test_df.groupby("molecule_id")

    for i, (mol_id, grp) in enumerate(grouped):
        mol_specs = []
        for _, r in grp.iterrows():
            pm = float(r["precursor_mz"])
            m, it = preprocess_peaks(r["ms2_mzs"], r["ms2_normalized_intensities"], pm)
            if len(m) >= 3:
                mol_specs.append({"precursor_mz": pm, "mzs": m, "ints": it})

        slist = predict_molecule(mol_specs, ref_index, top_n=25)
        predictions[mol_id] = slist

        if (i + 1) % 100 == 0 or (i + 1) == len(grouped):
            log(f"  Predicted {i+1}/{len(grouped)} test molecules...")

    # Write submission CSV
    sub_rows = [
        {"molecule_id": mid, "smiles": ";".join(smiles_list)}
        for mid, smiles_list in predictions.items()
    ]
    sub_df = pd.DataFrame(sub_rows)
    sub_df.to_csv(str(SUB_PATH), index=False)
    log(f"Saved submission to {SUB_PATH} ({len(sub_df)} rows).")

    # Kaggle Submission Checks
    assert list(sub_df.columns) == ["molecule_id", "smiles"], "Invalid columns"
    assert len(sub_df) == test_df["molecule_id"].nunique(), "Incorrect row count"
    assert not sub_df["smiles"].isna().any(), "Contains NaN SMILES"
    for _, row in sub_df.iterrows():
        assert len(row["smiles"].split(";")) == 25, f"Row {row['molecule_id']} must have exactly 25 SMILES"

    log("✓ Submission successfully verified and ready for scoring!")


if __name__ == "__main__":
    main()
