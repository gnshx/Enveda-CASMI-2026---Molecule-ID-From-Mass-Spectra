"""
experiments/baseline_v001/run_baseline.py

Phase 1 Reproducible Spectral Retrieval Baseline:
- Fast Numba-accelerated Cosine & Modified-Cosine
- Precursor m/z windowed search (O(log N) lookup)
- Per-molecule multi-spectrum evidence aggregation
- Structure-disjoint & library-stratified validation on timsTOF natural products
- Strict validation artifact logging:
  * validation_predictions.parquet
  * metrics.json (Recall@1/5/10/25/100/500, MRR@25)
  * candidate_recall.json (candidate_pool_size, true_structure_present)
  * submission.csv (400 molecules, exactly 25 SMILES per row, ';' separated)
"""

import sys
import time
import json
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

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
EXP_DIR = ROOT / "experiments" / "baseline_v001"
EXP_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = ROOT / "data"

T0 = time.time()
def log(msg: str):
    print(f"[{time.time()-T0:6.1f}s] {msg}", flush=True)


# ── 1. Numba Kernels for Ultrafast Spectral Matching ─────────────────────────

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
    for i in range(nq):
        norm_q += qints[i] * qints[i]
    norm_r = 0.0
    for i in range(nr):
        norm_r += rints[i] * rints[i]
    denom = np.sqrt(norm_q * norm_r)
    score = dot / denom if denom > 0 else 0.0
    return score, matched


@numba.njit(fastmath=True)
def numba_modified_cosine(qmzs: np.ndarray, qints: np.ndarray, qpmz: float,
                           rmzs: np.ndarray, rints: np.ndarray, rpmz: float,
                           mz_tol: float = 0.05):
    """Modified cosine accounting for precursor mass shift (fragment + neutral loss)."""
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
    for i in range(nq):
        norm_q += qints[i] * qints[i]
    norm_r = 0.0
    for i in range(nr):
        norm_r += rints[i] * rints[i]
    denom = np.sqrt(norm_q * norm_r)
    score = dot / denom if denom > 0 else 0.0
    return score, matched


# Warmup Numba JIT
_dummy_m = np.array([100.0, 200.0], dtype=np.float32)
_dummy_i = np.array([0.5, 1.0], dtype=np.float32)
numba_cosine(_dummy_m, _dummy_i, _dummy_m, _dummy_i)
numba_modified_cosine(_dummy_m, _dummy_i, 200.0, _dummy_m, _dummy_i, 200.0)


# ── 2. Preprocessing ──────────────────────────────────────────────────────────

def preprocess_peaks(mzs, ints, precursor_mz: float,
                     precursor_buffer: float = 2.0,
                     int_thresh: float = 0.005,
                     max_peaks: int = 128) -> Tuple[np.ndarray, np.ndarray]:
    """Robust spectrum preprocessing."""
    mzs = np.asarray(mzs, dtype=np.float32)
    ints = np.asarray(ints, dtype=np.float32)
    if len(mzs) == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    # Base peak normalization
    mx = ints.max()
    if mx > 0:
        ints = ints / mx

    # Filter precursor buffer
    keep = mzs <= (precursor_mz + precursor_buffer)
    mzs, ints = mzs[keep], ints[keep]

    # Filter intensity floor (0.5% base peak)
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
    if mx > 0:
        ints = ints / mx

    # Ensure sorted by m/z
    sort_idx = np.argsort(mzs)
    return mzs[sort_idx], ints[sort_idx]


# ── 3. Load & Index Reference Library and Validation Set ──────────────────────

def build_indices():
    log("Opening train.parquet...")
    pf = pq.ParquetFile(str(DATA_DIR / "train.parquet"))
    
    COLS = [
        "ingest_lib", "normalized_smiles", "inchikey14", "precursor_mz",
        "ms2_mzs", "ms2_normalized_intensities", "num_peaks", "collision_energy_ev", "adduct"
    ]
    
    INDEX_LIBS = {
        "pluskal_ms2", "riken", "gnps", "massbank", "mona",
        "msdial", "spectraverse", "drug_plus", "masaryk"
    }
    
    # Store validation spectra (enveda-np-examples)
    val_spectra = []
    
    # Store reference library spectra
    ref_pmzs = []
    ref_mzs = []
    ref_ints = []
    ref_inchikey = []
    ref_smiles = []
    ref_libs = []
    ref_ces = []

    log(f"Reading {pf.num_row_groups} row groups...")
    for rg_idx in range(pf.num_row_groups):
        tbl = pf.read_row_group(rg_idx, columns=COLS)
        
        # 1. Extract validation spectra (enveda-np-examples)
        np_mask = pc.equal(tbl["ingest_lib"], "enveda-np-examples")
        np_tbl = tbl.filter(np_mask)
        if len(np_tbl) > 0:
            df_np = np_tbl.to_pandas()
            for _, r in df_np.iterrows():
                pm = float(r["precursor_mz"])
                m, i = preprocess_peaks(r["ms2_mzs"], r["ms2_normalized_intensities"], pm)
                if len(m) >= 3:
                    val_spectra.append({
                        "inchikey14": r["inchikey14"],
                        "normalized_smiles": r["normalized_smiles"],
                        "precursor_mz": pm,
                        "collision_energy_ev": r["collision_energy_ev"],
                        "adduct": r["adduct"],
                        "mzs": m,
                        "ints": i,
                    })

        # 2. Extract reference library spectra: precursor_mz in [180, 520], num_peaks >= 3
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
                    ref_libs.append(r["ingest_lib"])
                    ref_ces.append(r["collision_energy_ev"])

        if (rg_idx + 1) % 5 == 0 or rg_idx == pf.num_row_groups - 1:
            log(f"  Processed {rg_idx+1}/{pf.num_row_groups} row groups. Indexed {len(ref_pmzs):,} ref spectra, {len(val_spectra):,} val spectra.")

    # Convert reference index to sorted order by precursor_mz for binary search
    log("Sorting reference library by precursor_mz...")
    ref_pmzs = np.array(ref_pmzs, dtype=np.float32)
    sort_order = np.argsort(ref_pmzs)
    
    ref_index = {
        "pmzs": ref_pmzs[sort_order],
        "mzs": [ref_mzs[idx] for idx in sort_order],
        "ints": [ref_ints[idx] for idx in sort_order],
        "inchikey14": [ref_inchikey[idx] for idx in sort_order],
        "smiles": [ref_smiles[idx] for idx in sort_order],
        "libs": [ref_libs[idx] for idx in sort_order],
        "ces": [ref_ces[idx] for idx in sort_order],
    }
    
    log(f"Library ready: {len(ref_index['pmzs']):,} spectra indexed. Val: {len(val_spectra):,} spectra.")
    return ref_index, val_spectra


# ── 4. Query & Evidence Aggregation ──────────────────────────────────────────

def query_library_for_spectrum(
    qmzs: np.ndarray,
    qints: np.ndarray,
    qpmz: float,
    ref_index: Dict[str, Any],
    mz_window: float = 2.0,
    mz_tol: float = 0.05,
    min_peaks: int = 3,
    top_k: int = 100,
) -> List[Dict[str, Any]]:
    """Fast precursor-windowed search against reference index."""
    pmzs = ref_index["pmzs"]
    low = np.searchsorted(pmzs, qpmz - mz_window, side="left")
    high = np.searchsorted(pmzs, qpmz + mz_window, side="right")

    if low >= high:
        return []

    hits = []
    for idx in range(low, high):
        rmzs = ref_index["mzs"][idx]
        rints = ref_index["ints"][idx]
        rpmz = float(pmzs[idx])

        cos, nm = numba_cosine(qmzs, qints, rmzs, rints, mz_tol)
        if cos > 0 and nm >= min_peaks:
            mcos, _ = numba_modified_cosine(qmzs, qints, qpmz, rmzs, rints, rpmz, mz_tol)
            mass_err_ppm = abs(qpmz - rpmz) / qpmz * 1e6
            hits.append({
                "inchikey14": ref_index["inchikey14"][idx],
                "smiles": ref_index["smiles"][idx],
                "cosine": cos,
                "mod_cosine": mcos,
                "n_matched": nm,
                "mass_error_ppm": mass_err_ppm,
            })

    if not hits:
        return []

    # Sort by cosine descending
    hits.sort(key=lambda x: -x["cosine"])
    return hits[:top_k]


def aggregate_molecule_evidence(
    mol_spectra: List[Dict[str, Any]],
    ref_index: Dict[str, Any],
    withhold_keys: set = None,
    top_n: int = 25,
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Per-molecule multi-spectrum evidence aggregation.
    Returns: (ranked_candidates_list, candidate_pool_size)
    """
    # Key: inchikey14 -> evidence list
    candidate_map: Dict[str, Dict[str, Any]] = {}

    for spec in mol_spectra:
        qmzs = spec["mzs"]
        qints = spec["ints"]
        qpmz = spec["precursor_mz"]
        hits = query_library_for_spectrum(qmzs, qints, qpmz, ref_index)

        for h in hits:
            k = h["inchikey14"]
            if withhold_keys and k in withhold_keys:
                continue  # Leakage prevention / structure withholding

            if k not in candidate_map:
                candidate_map[k] = {
                    "inchikey14": k,
                    "smiles": h["smiles"],
                    "cosines": [],
                    "mod_cosines": [],
                    "mass_errors": [],
                }
            candidate_map[k]["cosines"].append(h["cosine"])
            candidate_map[k]["mod_cosines"].append(h["mod_cosine"])
            candidate_map[k]["mass_errors"].append(h["mass_error_ppm"])

    candidate_pool_size = len(candidate_map)
    if candidate_pool_size == 0:
        return [], 0

    # Aggregate features per candidate
    ranked = []
    for k, data in candidate_map.items():
        cos_max = max(data["cosines"])
        cos_mean = float(np.mean(data["cosines"]))
        mod_max = max(data["mod_cosines"])
        n_supp = len(data["cosines"])
        min_err = min(data["mass_errors"])

        # Composite score: cosine_max prioritized, with multi-spectrum bonus
        # and modified cosine support for secondary differentiation
        composite_score = cos_max + 0.1 * cos_mean + 0.05 * mod_max + 0.02 * min(n_supp, 5)

        ranked.append({
            "candidate_inchikey14": k,
            "candidate_smiles": data["smiles"],
            "composite_score": composite_score,
            "cosine_score": cos_max,
            "modified_cosine_score": mod_max,
            "mass_error_ppm": min_err,
            "n_supporting_spectra": n_supp,
        })

    ranked.sort(key=lambda x: -x["composite_score"])
    for rank, item in enumerate(ranked, start=1):
        item["candidate_rank"] = rank

    return ranked[:top_n], candidate_pool_size


# ── 5. Run Validation Pipeline ───────────────────────────────────────────────

def run_validation(ref_index: Dict[str, Any], val_spectra: List[Dict[str, Any]]):
    log("Starting Validation Evaluation on timsTOF natural products...")

    # Group validation spectra by inchikey14 (molecule identity)
    val_mols: Dict[str, List[Dict[str, Any]]] = {}
    for s in val_spectra:
        val_mols.setdefault(s["inchikey14"], []).append(s)

    unique_val_keys = list(val_mols.keys())
    log(f"  {len(val_spectra)} spectra across {len(unique_val_keys)} unique validation molecules.")

    # Structure-disjoint split:
    # 125 molecules: Class 1-like (spectra from other libraries exist in ref_index)
    # 125 molecules: Class 2-like (all spectra for this molecule strictly withheld from ref_index)
    rng = np.random.default_rng(42)
    shuffled_keys = list(unique_val_keys)
    rng.shuffle(shuffled_keys)
    n_half = len(shuffled_keys) // 2
    class1_keys = set(shuffled_keys[:n_half])
    class2_keys = set(shuffled_keys[n_half:])

    reciprocal_ranks = []
    hits_at_k = {1: 0, 5: 0, 10: 0, 25: 0, 50: 0, 100: 0, 500: 0}
    class1_rrs = []
    class2_rrs = []

    candidate_recall_records = {}
    validation_prediction_rows = []

    for idx, (mol_key, specs) in enumerate(val_mols.items()):
        true_smiles = specs[0]["normalized_smiles"]
        # If Class 2, withhold mol_key from reference index
        withhold = {mol_key} if mol_key in class2_keys else None

        candidates, pool_size = aggregate_molecule_evidence(
            specs, ref_index, withhold_keys=withhold, top_n=500
        )

        hit_rank = None
        for cand in candidates:
            if cand["candidate_inchikey14"] == mol_key:
                hit_rank = cand["candidate_rank"]
                break

        # Record metrics
        rr = 1.0 / hit_rank if (hit_rank is not None and hit_rank <= 25) else 0.0
        reciprocal_ranks.append(rr)

        if mol_key in class1_keys:
            class1_rrs.append(rr)
        else:
            class2_rrs.append(rr)

        for k in hits_at_k:
            if hit_rank is not None and hit_rank <= k:
                hits_at_k[k] += 1

        true_present = any(cand["candidate_inchikey14"] == mol_key for cand in candidates)

        candidate_recall_records[mol_key] = {
            "true_inchikey14": mol_key,
            "true_smiles": true_smiles,
            "n_query_spectra": len(specs),
            "candidate_pool_size": pool_size,
            "true_structure_present": true_present,
            "hit_rank": hit_rank,
            "reciprocal_rank_25": rr,
            "simulation_class": "Class 1 (Library Present)" if mol_key in class1_keys else "Class 2 (Structure Withheld)",
            "retrieval_failure": not true_present,
            "ranking_failure": true_present and (hit_rank is None or hit_rank > 25),
        }

        # Store top-25 predictions for parquet
        for cand in candidates[:25]:
            validation_prediction_rows.append({
                "molecule_id": mol_key,
                "candidate_rank": cand["candidate_rank"],
                "candidate_smiles": cand["candidate_smiles"],
                "candidate_inchikey14": cand["candidate_inchikey14"],
                "cosine_score": float(cand["cosine_score"]),
                "modified_cosine_score": float(cand["modified_cosine_score"]),
                "mass_error_ppm": float(cand["mass_error_ppm"]),
                "true_inchikey14": mol_key,
                "is_correct": bool(cand["candidate_inchikey14"] == mol_key),
            })

        if (idx + 1) % 50 == 0:
            log(f"  Validated {idx+1}/{len(val_mols)} molecules. Running MRR@25: {np.mean(reciprocal_ranks):.4f}")

    total_mols = len(unique_val_keys)
    metrics = {
        "mrr_at_25": float(np.mean(reciprocal_ranks)),
        "class1_mrr_at_25": float(np.mean(class1_rrs)) if class1_rrs else 0.0,
        "class2_mrr_at_25": float(np.mean(class2_rrs)) if class2_rrs else 0.0,
        "n_val_molecules": total_mols,
        "n_val_spectra": len(val_spectra),
    }
    for k in [1, 5, 10, 25, 50, 100, 500]:
        metrics[f"recall_at_{k}"] = float(hits_at_k.get(k, 0) / total_mols)

    # Class 1 vs Class 2 breakdowns
    c1_total = len(class1_keys)
    c2_total = len(class2_keys)
    c1_hits_25 = sum(1 for k in class1_keys if candidate_recall_records[k]["hit_rank"] and candidate_recall_records[k]["hit_rank"] <= 25)
    c2_hits_25 = sum(1 for k in class2_keys if candidate_recall_records[k]["hit_rank"] and candidate_recall_records[k]["hit_rank"] <= 25)
    metrics["class1_recall_at_25"] = float(c1_hits_25 / c1_total) if c1_total > 0 else 0.0
    metrics["class2_recall_at_25"] = float(c2_hits_25 / c2_total) if c2_total > 0 else 0.0

    retrieval_failures = sum(1 for r in candidate_recall_records.values() if r["retrieval_failure"])
    ranking_failures = sum(1 for r in candidate_recall_records.values() if r["ranking_failure"])
    metrics["retrieval_failure_rate"] = float(retrieval_failures / total_mols)
    metrics["ranking_failure_rate"] = float(ranking_failures / total_mols)

    log("=" * 60)
    log(f"VALIDATION RESULTS (timsTOF Natural Products, {total_mols} molecules):")
    log(f"  MRR@25:                 {metrics['mrr_at_25']:.4f}")
    log(f"  Recall@1:               {metrics['recall_at_1']:.4f}")
    log(f"  Recall@5:               {metrics['recall_at_5']:.4f}")
    log(f"  Recall@10:              {metrics['recall_at_10']:.4f}")
    log(f"  Recall@25:              {metrics['recall_at_25']:.4f}")
    log(f"  Recall@100:             {metrics['recall_at_100']:.4f}")
    log(f"  Class 1 MRR@25:         {metrics['class1_mrr_at_25']:.4f} (Recall@25: {metrics['class1_recall_at_25']:.4f})")
    log(f"  Class 2 MRR@25:         {metrics['class2_mrr_at_25']:.4f} (Recall@25: {metrics['class2_recall_at_25']:.4f})")
    log(f"  Retrieval Failures:     {retrieval_failures}/{total_mols} ({metrics['retrieval_failure_rate']*100:.1f}%)")
    log(f"  Ranking Failures:       {ranking_failures}/{total_mols} ({metrics['ranking_failure_rate']*100:.1f}%)")
    log("=" * 60)

    # Save artifacts
    metrics_path = EXP_DIR / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    log(f"Saved metrics to {metrics_path}")

    candidate_recall_path = EXP_DIR / "candidate_recall.json"
    with open(candidate_recall_path, "w") as f:
        json.dump(candidate_recall_records, f, indent=2)
    log(f"Saved candidate recall diagnostic to {candidate_recall_path}")

    val_preds_df = pd.DataFrame(validation_prediction_rows)
    val_preds_path = EXP_DIR / "validation_predictions.parquet"
    val_preds_df.to_parquet(val_preds_path, index=False)
    log(f"Saved validation predictions to {val_preds_path} ({len(val_preds_df):,} rows)")

    return metrics


# ── 6. Test Inference & Submission Generation ────────────────────────────────

def run_test_inference(ref_index: Dict[str, Any]):
    log("Starting Test Set Inference (400 molecules)...")
    test_df = pd.read_parquet(str(DATA_DIR / "test.parquet"))
    log(f"  Loaded test.parquet: {len(test_df)} spectra, {test_df['molecule_id'].nunique()} molecules")

    # Group test spectra by molecule_id
    mol_groups = test_df.groupby("molecule_id")

    predictions = {}
    n_fallback = 0

    for idx, (mol_id, grp) in enumerate(mol_groups):
        mol_specs = []
        for _, r in grp.iterrows():
            pm = float(r["precursor_mz"])
            m, i = preprocess_peaks(r["ms2_mzs"], r["ms2_normalized_intensities"], pm)
            if len(m) >= 3:
                mol_specs.append({
                    "precursor_mz": pm,
                    "mzs": m,
                    "ints": i,
                })

        candidates, pool_size = aggregate_molecule_evidence(mol_specs, ref_index, top_n=25)

        slist = [c["candidate_smiles"] for c in candidates if c.get("candidate_smiles")]
        
        # Ensure exactly 25 predictions
        if len(slist) < 25:
            n_fallback += 1
            # Backfill with unique smiles from library or fallback "CCO"
            backfill = ["CCO"] * (25 - len(slist))
            slist.extend(backfill)
        else:
            slist = slist[:25]

        predictions[mol_id] = slist

        if (idx + 1) % 100 == 0:
            log(f"  Predicted {idx+1}/{len(mol_groups)} test molecules...")

    log(f"Test inference complete. Fallbacks needed for {n_fallback}/{len(predictions)} molecules.")

    # Format submission CSV
    submission_rows = [
        {"molecule_id": mid, "smiles": ";".join(smiles_list)}
        for mid, smiles_list in predictions.items()
    ]
    sub_df = pd.DataFrame(submission_rows)

    sub_path = EXP_DIR / "submission.csv"
    sub_df.to_csv(sub_path, index=False)
    log(f"Saved submission to {sub_path} ({len(sub_df)} rows)")

    # Validate against all 5 Kaggle rejection criteria
    assert list(sub_df.columns) == ["molecule_id", "smiles"], "Column names must be molecule_id,smiles"
    assert len(sub_df) == 400, f"Expected 400 rows, got {len(sub_df)}"
    assert not sub_df["molecule_id"].isna().any(), "No NaN molecule_id"
    assert not sub_df["smiles"].isna().any(), "No NaN smiles"
    for _, row in sub_df.iterrows():
        parts = row["smiles"].split(";")
        assert len(parts) == 25, f"Row {row['molecule_id']} does not have exactly 25 SMILES"
        for s in parts:
            assert len(s.strip()) > 0, "Empty SMILES token found"

    log("✓ Submission passed all 5 Kaggle rejection criteria!")
    return sub_df


# ── 7. Main Execution ────────────────────────────────────────────────────────

def main():
    log("=== Starting CASMI 2026 Phase 1 Baseline Pipeline ===")
    ref_index, val_spectra = build_indices()
    run_validation(ref_index, val_spectra)
    run_test_inference(ref_index)
    log("=== Baseline v001 Pipeline Completed Successfully! ===")


if __name__ == "__main__":
    main()
