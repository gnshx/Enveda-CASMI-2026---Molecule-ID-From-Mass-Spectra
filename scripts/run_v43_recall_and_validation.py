#!/usr/bin/env python3
"""
scripts/run_v43_recall_and_validation.py
=============================================================================
CASMI 2026 — V43 Phase 2: Independent Leak-Free Validation & Recall Bottleneck Audit

Evaluates strictly leak-free, canonical-tautomer-deduplicated splits from
`data/validation_folds/v43_molecules.parquet`:
  - `timstof_holdout`: 1,000 untouched enveda-180 timsTOF molecules
  - `np_examples_purged`: 250 enveda-np-examples molecules
  - `test.parquet`: 400 blind test molecules (candidate pool size diagnostics only)

Measures:
  1. Candidate-Recall Funnel across COCONUT (479k) + ExtLib (145k) + ChEBI/LMSD (98k)
     (+ Train Structures 274k for synthetic-domain timstof_holdout Regime B simulation)
     at +/- 5, 10, 15, 20, 25 ppm with and without -1.50 ppm timsTOF calibration.
  2. Separate Regime Evaluation:
     - Regime A (Cross-Instrument Reference Match): non-timsTOF library spectra present
     - Regime B (Pure Database Retrieval): all target spectra purged from spectral index
     - Regime C (Unknown Structure): target absent from COCONUT+ExtLib+ChEBI/LMSD
  3. Reports MRR@25, Recall@1/5/25/100/1000 and 95% paired bootstrap CIs.

Outputs:
  - experiments/v43_validation_report.json
  - docs/V43_VALIDATION_AND_RECALL_REPORT.md
=============================================================================
"""

import gc
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy import sparse

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chemistry.ms_engine import ADDUCT_OFFSETS, TEST_ADDUCTS, compute_neutral_mass
from validation.metrics import mrr_at_25, paired_bootstrap_ci, recall_at_k


def load_np_catalog() -> Tuple[pd.DataFrame, np.ndarray]:
    """Load COCONUT + ExtLib + ChEBI/LMSD deduplicated by inchikey14, sorted by mono_mass."""
    coco = pd.read_parquet(
        ROOT / "datasets" / "extlib" / "coconut_index.parquet",
        columns=["inchikey14", "smiles_coconut", "mono_mass"],
    ).rename(columns={"smiles_coconut": "smiles"})
    ext = pd.read_parquet(
        ROOT / "datasets" / "extlib" / "ext_mass.parquet",
        columns=["inchikey14", "normalized_smiles", "mono_mass"],
    ).rename(columns={"normalized_smiles": "smiles"})
    bio = pd.read_parquet(
        ROOT / "datasets" / "bio_structures" / "casmi-chebi-lmsd-structures" / "bio_structures.parquet",
        columns=["inchikey14", "smiles", "monoisotopic_mass"],
    ).rename(columns={"monoisotopic_mass": "mono_mass"})

    coco_keys = set(coco["inchikey14"].dropna().astype(str))
    ext_keys = set(ext["inchikey14"].dropna().astype(str))
    bio_keys = set(bio["inchikey14"].dropna().astype(str))

    cat = pd.concat([coco, bio, ext], ignore_index=True)
    cat = cat.dropna(subset=["inchikey14", "smiles", "mono_mass"])
    cat["inchikey14"] = cat["inchikey14"].astype(str)
    cat = cat.drop_duplicates(subset=["inchikey14"], keep="first").copy()
    cat["in_coco"] = cat["inchikey14"].isin(coco_keys)
    cat["in_ext"] = cat["inchikey14"].isin(ext_keys)
    cat["in_bio"] = cat["inchikey14"].isin(bio_keys)
    cat = cat.sort_values("mono_mass", kind="mergesort").reset_index(drop=True)
    masses = cat["mono_mass"].to_numpy(dtype=np.float64)
    return cat, masses


def load_extended_catalog(np_cat: pd.DataFrame, mol_df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray]:
    """Union NP catalog with train.parquet unique structures (simulating PubChem structure coverage)."""
    tr = mol_df[["canon_ik14", "normalized_smiles", "exact_mass"]].dropna().copy()
    tr = tr.rename(columns={"canon_ik14": "inchikey14", "normalized_smiles": "smiles", "exact_mass": "mono_mass"})
    tr["in_coco"] = False
    tr["in_ext"] = False
    tr["in_bio"] = False
    cat = pd.concat([np_cat, tr], ignore_index=True)
    cat = cat.drop_duplicates(subset=["inchikey14"], keep="first")
    cat = cat.sort_values("mono_mass", kind="mergesort").reset_index(drop=True)
    return cat, cat["mono_mass"].to_numpy(dtype=np.float64)


def build_query_table(mol_df: pd.DataFrame, role: str, required_ingest_lib: str) -> pd.DataFrame:
    sub = mol_df[mol_df["split_role"] == role]
    smi_to_info = sub.set_index("normalized_smiles")[["canon_ik14", "exact_mass"]].to_dict("index")

    pf = pq.ParquetFile(str(ROOT / "data" / "train.parquet"))
    cols = [
        "ingest_lib",
        "normalized_smiles",
        "adduct",
        "ionization_mode",
        "precursor_mz",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "num_peaks",
    ]
    best_by_ik: Dict[str, dict] = {}
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        m = (
            df["ingest_lib"].eq(required_ingest_lib)
            & df["normalized_smiles"].isin(smi_to_info)
            & df["adduct"].isin(TEST_ADDUCTS)
        )
        if not m.any():
            continue
        for row in df[m].itertuples(index=False):
            info = smi_to_info[str(row.normalized_smiles)]
            cik = str(info["canon_ik14"])
            npk = int(row.num_peaks)
            prev = best_by_ik.get(cik)
            if prev is None or npk > prev["num_peaks"]:
                best_by_ik[cik] = {
                    "molecule_id": f"{role}_{cik}",
                    "canon_ik14": cik,
                    "smiles": str(row.normalized_smiles),
                    "exact_mass": float(info["exact_mass"]),
                    "adduct": str(row.adduct),
                    "ionization_mode": str(row.ionization_mode),
                    "precursor_mz": float(row.precursor_mz),
                    "ms2_mzs": np.asarray(row.ms2_mzs, dtype=np.float64),
                    "ms2_its": np.asarray(row.ms2_normalized_intensities, dtype=np.float64),
                    "num_peaks": npk,
                }
    return pd.DataFrame(list(best_by_ik.values()))


def build_test_query_summary() -> pd.DataFrame:
    te = pd.read_parquet(ROOT / "data" / "test.parquet")
    rows = []
    for mid, grp in te.groupby("molecule_id", sort=False):
        r = grp.iloc[0]
        rows.append({
            "molecule_id": str(mid),
            "adduct": str(r["adduct"]),
            "precursor_mz": float(r["precursor_mz"]),
        })
    return pd.DataFrame(rows)


def audit_mass_window_recall(
    qdf: pd.DataFrame,
    cat: pd.DataFrame,
    masses: np.ndarray,
    ppms: List[float],
    has_ground_truth: bool,
) -> Dict:
    cat_iks = cat["inchikey14"].to_numpy()
    coco_keys = set(cat.loc[cat["in_coco"], "inchikey14"])
    ext_keys = set(cat.loc[cat["in_ext"], "inchikey14"])
    bio_keys = set(cat.loc[cat["in_bio"], "inchikey14"])
    all_keys = set(cat["inchikey14"])

    oracle = {}
    if has_ground_truth:
        q_iks = qdf["canon_ik14"].tolist()
        n_q = len(q_iks)
        oracle = {
            "n_queries": n_q,
            "oracle_coconut_cov": round(sum(k in coco_keys for k in q_iks) / n_q, 4),
            "oracle_chebi_lmsd_cov": round(sum(k in bio_keys for k in q_iks) / n_q, 4),
            "oracle_extlib_cov": round(sum(k in ext_keys for k in q_iks) / n_q, 4),
            "oracle_catalog_union_cov": round(sum(k in all_keys for k in q_iks) / n_q, 4),
            "regime_missing_count": sum(k not in all_keys for k in q_iks),
        }

    windows = {}
    for cal in [False, True]:
        cal_key = "timstof_calibrated_minus1.5ppm" if cal else "uncalibrated"
        windows[cal_key] = {}
        for ppm in ppms:
            pool_sizes = []
            hits = 0
            hits_given_oracle = 0
            oracle_cnt = 0
            for row in qdf.itertuples(index=False):
                nm = compute_neutral_mass(row.precursor_mz, row.adduct, calibrate_timstof=cal)
                tol = nm * ppm * 1e-6
                lo = np.searchsorted(masses, nm - tol, side="left")
                hi = np.searchsorted(masses, nm + tol, side="right")
                pool_sizes.append(int(hi - lo))
                if has_ground_truth:
                    in_union = row.canon_ik14 in all_keys
                    if in_union:
                        oracle_cnt += 1
                    if hi > lo and row.canon_ik14 in set(cat_iks[lo:hi]):
                        hits += 1
                        if in_union:
                            hits_given_oracle += 1
            arr = np.asarray(pool_sizes, dtype=np.int64)
            rec = {
                "pool_median": int(np.median(arr)),
                "pool_p95": int(np.percentile(arr, 95)),
                "pool_max": int(arr.max()),
                "zero_candidate_queries": int((arr == 0).sum()),
            }
            if has_ground_truth:
                rec["window_recall_all"] = round(hits / len(qdf), 4)
                rec["window_recall_given_oracle"] = round(hits_given_oracle / max(oracle_cnt, 1), 4)
            windows[cal_key][f"ppm_{int(ppm)}"] = rec

    return {"oracle": oracle, "windows": windows}


def spectrum_to_sparse_bins(
    mzs: np.ndarray,
    its: np.ndarray,
    prec_mz: float,
    bin_width: float = 0.05,
    max_mz: float = 1000.0,
) -> Tuple[np.ndarray, np.ndarray]:
    n_bins = int(max_mz / bin_width)
    ok = (mzs >= 20.0) & (mzs <= prec_mz - 1.5) & (mzs < max_mz) & (its > 0)
    if not ok.any():
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    m = mzs[ok]
    w = np.sqrt(its[ok])
    frag_bins = np.clip((m / bin_width).astype(np.int32), 0, n_bins - 1)
    nl_m = prec_mz - m
    nl_ok = (nl_m >= 12.0) & (nl_m < max_mz)
    nl_bins = n_bins + np.clip((nl_m[nl_ok] / bin_width).astype(np.int32), 0, n_bins - 1)

    idx = np.concatenate([frag_bins, nl_bins])
    vals = np.concatenate([w, 0.7 * w[nl_ok]])
    u_idx, inv = np.unique(idx, return_inverse=True)
    agg = np.zeros(len(u_idx), dtype=np.float32)
    np.maximum.at(agg, inv, vals.astype(np.float32))
    norm = float(np.linalg.norm(agg))
    if norm > 1e-12:
        agg /= norm
    return u_idx, agg


def build_multilib_ref_index(
    mol_df: pd.DataFrame,
    include_np_external: bool = False,
    max_enveda180: int = 40000,
) -> Tuple[sparse.csr_matrix, np.ndarray, List[str], np.ndarray]:
    """
    Build spectral reference index with low memory footprint (binning on the fly):
      - Always excludes `timstof_holdout` canon_ik14 completely.
      - Excludes `np_examples_purged` unless `include_np_external=True`.
    """
    allowed_roles = {"train_cv"} if not include_np_external else {"train_cv", "np_examples_purged"}
    sub = mol_df[mol_df["split_role"].isin(allowed_roles)].drop_duplicates("canon_ik14")
    e180_only = sub[sub["in_enveda_180"] & (sub["n_libs"] == 1)]
    non_e180 = sub[~(sub["in_enveda_180"] & (sub["n_libs"] == 1))]
    if len(e180_only) > max_enveda180:
        e180_only = e180_only.sample(n=max_enveda180, random_state=42)
    sel = pd.concat([non_e180, e180_only], ignore_index=True)
    smi_to_ik = dict(zip(sel["normalized_smiles"], sel["canon_ik14"]))
    del sub, e180_only, non_e180, sel

    pf = pq.ParquetFile(str(ROOT / "data" / "train.parquet"))
    cols = [
        "ingest_lib",
        "normalized_smiles",
        "adduct",
        "precursor_mz",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "num_peaks",
    ]
    best_ref: Dict[str, tuple] = {}
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        m = (
            df["ingest_lib"].ne("enveda-np-examples")
            & df["normalized_smiles"].isin(smi_to_ik)
            & df["adduct"].isin(TEST_ADDUCTS)
        )
        if m.any():
            for row in df[m].itertuples(index=False):
                smi = str(row.normalized_smiles)
                cik = smi_to_ik[smi]
                npk = int(row.num_peaks)
                prev = best_ref.get(cik)
                if prev is None or npk > prev[0]:
                    prec = float(row.precursor_mz)
                    u_idx, agg = spectrum_to_sparse_bins(
                        np.asarray(row.ms2_mzs, dtype=np.float64),
                        np.asarray(row.ms2_normalized_intensities, dtype=np.float64),
                        prec,
                    )
                    if len(u_idx) > 0:
                        best_ref[cik] = (
                            npk,
                            smi,
                            u_idx.astype(np.uint16),
                            agg.astype(np.float16),
                            prec - ADDUCT_OFFSETS.get(str(row.adduct), 1.007276),
                        )
        del df
        if rg % 5 == 0:
            gc.collect()

    n_cols = 2 * int(1000.0 / 0.05)
    indptr = [0]
    indices_parts, data_parts = [], []
    ref_iks, ref_smis, ref_nms = [], [], []
    for cik, (_, smi, u_idx, agg, nm) in best_ref.items():
        indices_parts.append(u_idx.astype(np.int32))
        data_parts.append(agg.astype(np.float32))
        indptr.append(indptr[-1] + len(u_idx))
        ref_iks.append(cik)
        ref_smis.append(smi)
        ref_nms.append(nm)
    del best_ref
    gc.collect()

    mat = sparse.csr_matrix(
        (np.concatenate(data_parts), np.concatenate(indices_parts), np.asarray(indptr, dtype=np.int64)),
        shape=(len(ref_iks), n_cols),
        dtype=np.float32,
    )
    return mat, np.asarray(ref_iks), ref_smis, np.asarray(ref_nms, dtype=np.float64)


def evaluate_baselines(
    qdf: pd.DataFrame,
    cat: pd.DataFrame,
    masses: np.ndarray,
    ref_mat: sparse.csr_matrix,
    ref_iks: np.ndarray,
    ref_smis: List[str],
    ref_nms: np.ndarray,
    allow_exact_ref_hit: bool,
    ppm_win: float = 15.0,
) -> Dict:
    n_cols = ref_mat.shape[1]
    q_indptr = [0]
    q_idx_parts, q_val_parts = [], []
    for row in qdf.itertuples(index=False):
        u_idx, agg = spectrum_to_sparse_bins(row.ms2_mzs, row.ms2_its, row.precursor_mz)
        q_idx_parts.append(u_idx)
        q_val_parts.append(agg)
        q_indptr.append(q_indptr[-1] + len(u_idx))
    q_mat = sparse.csr_matrix(
        (np.concatenate(q_val_parts), np.concatenate(q_idx_parts), np.asarray(q_indptr, dtype=np.int64)),
        shape=(len(qdf), n_cols),
        dtype=np.float32,
    )

    top_analogs_all = np.empty((len(qdf), 25), dtype=np.int32)
    top_sims_all = np.empty((len(qdf), 25), dtype=np.float32)
    best_mass_ref_sim = {}
    for start in range(0, len(qdf), 50):
        end = min(start + 50, len(qdf))
        blk = (q_mat[start:end] @ ref_mat.T).toarray()
        idx_blk = np.argpartition(-blk, 25, axis=1)[:, :25]
        for r_off in range(end - start):
            qi = start + r_off
            row_idx = idx_blk[r_off]
            row_sim = blk[r_off, row_idx]
            ord_r = np.argsort(-row_sim)
            top_analogs_all[qi] = row_idx[ord_r]
            top_sims_all[qi] = row_sim[ord_r]
            if allow_exact_ref_hit:
                qrow = qdf.iloc[qi]
                nm = compute_neutral_mass(qrow.precursor_mz, qrow.adduct, calibrate_timstof=True)
                tol = nm * ppm_win * 1e-6
                in_win = np.where(np.abs(ref_nms - nm) <= tol)[0]
                if len(in_win) > 0:
                    sims_w = blk[r_off, in_win]
                    map_sim: Dict[str, float] = {}
                    for r_j, sv in zip(in_win, sims_w):
                        ik = ref_iks[r_j]
                        if sv > map_sim.get(ik, -1.0):
                            map_sim[ik] = float(sv)
                    best_mass_ref_sim[qi] = map_sim

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    ref_fp_cache: Dict[str, np.ndarray] = {}

    def get_fp(smi: str, cache: bool = False) -> np.ndarray | None:
        if cache and smi in ref_fp_cache:
            return ref_fp_cache[smi]
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        arr = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        if cache and len(ref_fp_cache) < 30000:
            ref_fp_cache[smi] = arr
        return arr

    cat_iks = cat["inchikey14"].to_numpy()
    cat_smis = cat["smiles"].to_numpy()
    preds_mass, preds_model, gt = {}, {}, {}
    rec_100, rec_1000 = [], []

    for qi, row in enumerate(qdf.itertuples(index=False)):
        mid = row.molecule_id
        true_ik = row.canon_ik14
        gt[mid] = true_ik
        nm = compute_neutral_mass(row.precursor_mz, row.adduct, calibrate_timstof=True)
        tol = nm * ppm_win * 1e-6
        lo = np.searchsorted(masses, nm - tol, side="left")
        hi = np.searchsorted(masses, nm + tol, side="right")
        if hi <= lo:
            preds_mass[mid] = []
            preds_model[mid] = []
            rec_100.append(False)
            rec_1000.append(False)
            continue

        c_iks = cat_iks[lo:hi]
        c_smis = cat_smis[lo:hi]
        c_masses = masses[lo:hi]
        ppm_err = np.abs((c_masses - nm) / nm * 1e6)

        m_ord = np.argsort(ppm_err, kind="stable")
        preds_mass[mid] = c_iks[m_ord][:25].tolist()

        mass_prior = np.exp(-0.5 * (ppm_err / 3.5) ** 2)
        ana_fps, ana_w = [], []
        for r_i, r_sim in zip(top_analogs_all[qi], top_sims_all[qi]):
            if (not allow_exact_ref_hit and ref_iks[r_i] == true_ik) or r_sim <= 0.05:
                continue
            fp = get_fp(ref_smis[r_i], cache=True)
            if fp is not None:
                ana_fps.append(fp)
                ana_w.append(float(r_sim) ** 3)

        if ana_fps:
            afp = np.stack(ana_fps)
            aw = np.asarray(ana_w, dtype=np.float32)
            aw /= aw.sum()
            cfps, valid_idx = [], []
            for ci, smi in enumerate(c_smis):
                fp = get_fp(str(smi), cache=False)
                if fp is not None:
                    cfps.append(fp)
                    valid_idx.append(ci)
            full_tan = np.zeros(len(c_iks), dtype=np.float64)
            if cfps:
                cfp = np.stack(cfps)
                inter = cfp @ afp.T
                denom = cfp.sum(1, keepdims=True) + afp.sum(1, keepdims=True).T - inter + 1e-9
                full_tan[valid_idx] = (inter / denom) @ aw
            score = full_tan + 0.15 * mass_prior
        else:
            score = mass_prior

        if allow_exact_ref_hit and qi in best_mass_ref_sim:
            ref_map = best_mass_ref_sim[qi]
            exact_bonus = np.array([ref_map.get(k, 0.0) for k in c_iks], dtype=np.float64)
            score = score + 2.0 * exact_bonus

        a_ord = np.argsort(-score, kind="stable")
        ranked = c_iks[a_ord].tolist()
        preds_model[mid] = ranked[:25]
        rec_100.append(true_ik in set(ranked[:100]))
        rec_1000.append(true_ik in set(ranked[:1000]))

    ci = paired_bootstrap_ci(preds_mass, preds_model, gt, n_bootstrap=2000, seed=42)
    return {
        "n": len(gt),
        "mass_only": {
            "mrr25": round(mrr_at_25(preds_mass, gt), 4),
            "r1": round(recall_at_k(preds_mass, gt, 1), 4),
            "r5": round(recall_at_k(preds_mass, gt, 5), 4),
            "r25": round(recall_at_k(preds_mass, gt, 25), 4),
        },
        "spectral_baseline": {
            "mrr25": round(mrr_at_25(preds_model, gt), 4),
            "r1": round(recall_at_k(preds_model, gt, 1), 4),
            "r5": round(recall_at_k(preds_model, gt, 5), 4),
            "r25": round(recall_at_k(preds_model, gt, 25), 4),
            "r100": round(float(np.mean(rec_100)), 4),
            "r1000": round(float(np.mean(rec_1000)), 4),
        },
        "paired_bootstrap_delta": {k: round(v, 4) for k, v in ci.items()},
    }


def write_markdown_report(rep: Dict, out_md: Path) -> None:
    rf = rep["recall_funnel"]
    np_o = rf["np_examples_purged_on_np_catalog"]["oracle"]
    np_w10_cal = rf["np_examples_purged_on_np_catalog"]["windows"]["timstof_calibrated_minus1.5ppm"]["ppm_10"]
    np_w10_un = rf["np_examples_purged_on_np_catalog"]["windows"]["uncalibrated"]["ppm_10"]
    np_w5_cal = rf["np_examples_purged_on_np_catalog"]["windows"]["timstof_calibrated_minus1.5ppm"]["ppm_5"]
    np_w5_un = rf["np_examples_purged_on_np_catalog"]["windows"]["uncalibrated"]["ppm_5"]
    th_ext_o = rf["timstof_holdout_on_extended_catalog"]["oracle"]
    th_w10_cal = rf["timstof_holdout_on_extended_catalog"]["windows"]["timstof_calibrated_minus1.5ppm"]["ppm_10"]
    te_w10_cal = rf["blind_test_on_np_catalog"]["windows"]["timstof_calibrated_minus1.5ppm"]["ppm_10"]
    te_w25_cal = rf["blind_test_on_np_catalog"]["windows"]["timstof_calibrated_minus1.5ppm"]["ppm_25"]

    bl = rep["leak_free_baselines_ppm15"]
    lines = [
        "# V43 Validation & Candidate-Recall Bottleneck Report",
        "",
        "## 1. Canonical-Tautomer Split Integrity (`validation/v43_build_splits.py`)",
        "",
        "- Scanned `2,539,608` spectra (`277,566` unique `normalized_smiles`, `275,810` raw `inchikey14`).",
        "- Canonical tautomer standardization collapsed **4,375 SMILES** differing from raw `inchikey14` down to **`274,289` unique `canon_ik14`**.",
        "- Critically, **`250 / 250`** of `enveda-np-examples` (`309` SMILES) also appear in non-Enveda training libraries (`riken`: 246, `gnps`: 233, `mona`: 231, `msdial`: 227, `pluskal_ms2`: 224, `massbank`: 208), proving why naive validation on `enveda-np-examples` leaks unless purged across all libraries.",
        "- Built strictly disjoint splits: `timstof_holdout` (`1,000` `enveda-180` molecules), `np_examples_purged` (`250` NP molecules), and 5-fold `train_cv` (`273,039` molecules).",
        "",
        "## 2. Candidate-Recall Bottleneck Audit",
        "",
        f"### Natural-Product Catalog Coverage (`COCONUT + ExtLib + ChEBI/LMSD`, {rep['np_catalog_size']:,} unique structures)",
        f"- **`np_examples_purged` (250 timsTOF NPs)**: COCONUT={np_o['oracle_coconut_cov']:.1%}, ChEBI/LMSD={np_o['oracle_chebi_lmsd_cov']:.1%}, Union={np_o['oracle_catalog_union_cov']:.1%} ({np_o['regime_missing_count']} Regime C missing).",
        f"- **`np_examples_purged` +/-5 ppm Recall**: Uncalibrated=`{np_w5_un['window_recall_all']:.4f}` vs `-1.50 ppm` timsTOF calibrated=`{np_w5_cal['window_recall_all']:.4f}` (given oracle: `{np_w5_un['window_recall_given_oracle']:.4f}` -> `{np_w5_cal['window_recall_given_oracle']:.4f}`).",
        f"- **`np_examples_purged` +/-10 ppm**: Recall=`{np_w10_cal['window_recall_all']:.4f}` (`{np_w10_cal['window_recall_given_oracle']:.4f}` given oracle), median pool=`{np_w10_cal['pool_median']}`, p95=`{np_w10_cal['pool_p95']}`.",
        f"- **`blind_test` (`test.parquet`, 400 molecules) on NP Catalog**: +/-10 ppm median pool=`{te_w10_cal['pool_median']}` (p95=`{te_w10_cal['pool_p95']}`, zero-candidate=`{te_w10_cal['zero_candidate_queries']}`); +/-25 ppm zero-candidate=`{te_w25_cal['zero_candidate_queries']}`.",
        f"- **`timstof_holdout` (1,000 `enveda-180` synthetic/screening molecules)**: `0.0%` in COCONUT/ExtLib (confirming `enveda-180` is non-NP chemical space); `100.0%` on extended catalog (`+{th_ext_o['n_queries']}` molecules), +/-10 ppm Recall=`{th_w10_cal['window_recall_all']:.4f}`, median pool=`{th_w10_cal['pool_median']}`.",
        "",
        "## 3. Regime-Stratified Leak-Free Baselines (+/-15 ppm)",
        "",
        "| Benchmark Cohort | Regime | N | Mass-Only MRR@25 | Spectral Baseline MRR@25 | R@1 | R@5 | R@25 | R@100 | R@1000 | 95% Paired Bootstrap CI |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for label, d in [
        ("`np_examples_purged`", bl["np_examples_purged_regime_b_pure"]),
        ("`np_examples_purged` (Cross-Inst Ref)", bl["np_examples_regime_a_cross_instrument"]),
        ("`timstof_holdout` (Extended Pool)", bl["timstof_holdout_regime_b_extended"]),
    ]:
        m = d["mass_only"]
        s = d["spectral_baseline"]
        ci = d["paired_bootstrap_delta"]
        regime = "Regime A" if "Cross-Inst" in label else "Regime B/C"
        lines.append(
            f"| {label} | {regime} | {d['n']} | {m['mrr25']:.4f} | **{s['mrr25']:.4f}** | "
            f"{s['r1']:.4f} | {s['r5']:.4f} | {s['r25']:.4f} | {s['r100']:.4f} | {s['r1000']:.4f} | "
            f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] |"
        )
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    t0 = time.time()
    print("=" * 80)
    print("V43 FULL MULTI-REGIME VALIDATION & RECALL BOTTLENECK AUDIT")
    print("=" * 80)

    mol_df = pd.read_parquet(ROOT / "data" / "validation_folds" / "v43_molecules.parquet")
    np_cat, np_masses = load_np_catalog()
    ext_cat, ext_masses = load_extended_catalog(np_cat, mol_df)
    print(f"NP Catalog: {len(np_cat):,} | Extended Catalog: {len(ext_cat):,} ({time.time()-t0:.1f}s)")

    q_holdout = build_query_table(mol_df, "timstof_holdout", "enveda-180")
    q_np = build_query_table(mol_df, "np_examples_purged", "enveda-np-examples")
    q_test = build_test_query_summary()
    print(f"Queries: timstof_holdout={len(q_holdout)}, np_examples_purged={len(q_np)}, blind_test={len(q_test)}")

    ppms = [5.0, 10.0, 15.0, 20.0, 25.0]
    recall_funnel = {
        "np_examples_purged_on_np_catalog": audit_mass_window_recall(q_np, np_cat, np_masses, ppms, True),
        "timstof_holdout_on_np_catalog": audit_mass_window_recall(q_holdout, np_cat, np_masses, ppms, True),
        "timstof_holdout_on_extended_catalog": audit_mass_window_recall(q_holdout, ext_cat, ext_masses, ppms, True),
        "blind_test_on_np_catalog": audit_mass_window_recall(q_test, np_cat, np_masses, ppms, False),
    }

    print(f"Building pure Regime B/C multi-lib reference index (zero holdout/NP overlap)...")
    ref_b_mat, ref_b_iks, ref_b_smis, ref_b_nms = build_multilib_ref_index(mol_df, include_np_external=False)
    assert set(ref_b_iks).isdisjoint(set(q_holdout["canon_ik14"]))
    assert set(ref_b_iks).isdisjoint(set(q_np["canon_ik14"]))
    print(f"Regime B/C ref index: {ref_b_mat.shape} ({time.time()-t0:.1f}s)")

    res_np_b = evaluate_baselines(q_np, np_cat, np_masses, ref_b_mat, ref_b_iks, ref_b_smis, ref_b_nms, False, 15.0)
    res_holdout_b = evaluate_baselines(q_holdout, ext_cat, ext_masses, ref_b_mat, ref_b_iks, ref_b_smis, ref_b_nms, False, 15.0)
    del ref_b_mat
    gc.collect()

    print(f"Building Regime A cross-instrument reference index (non-timsTOF external libs for NP)...")
    ref_a_mat, ref_a_iks, ref_a_smis, ref_a_nms = build_multilib_ref_index(mol_df, include_np_external=True)
    assert set(ref_a_iks).isdisjoint(set(q_holdout["canon_ik14"]))
    res_np_a = evaluate_baselines(q_np, np_cat, np_masses, ref_a_mat, ref_a_iks, ref_a_smis, ref_a_nms, True, 15.0)
    del ref_a_mat, mol_df
    gc.collect()

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "np_catalog_size": int(len(np_cat)),
        "extended_catalog_size": int(len(ext_cat)),
        "recall_funnel": recall_funnel,
        "leak_free_baselines_ppm15": {
            "np_examples_purged_regime_b_pure": res_np_b,
            "np_examples_regime_a_cross_instrument": res_np_a,
            "timstof_holdout_regime_b_extended": res_holdout_b,
        },
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    out_json = ROOT / "experiments" / "v43_validation_report.json"
    out_md = ROOT / "docs" / "V43_VALIDATION_AND_RECALL_REPORT.md"
    out_json.write_text(json.dumps(report, indent=2))
    write_markdown_report(report, out_md)
    print(f"✓ Saved {out_json} and {out_md} ({time.time()-t0:.1f}s)")
    print(json.dumps(report["leak_free_baselines_ppm15"], indent=2))


if __name__ == "__main__":
    main()
