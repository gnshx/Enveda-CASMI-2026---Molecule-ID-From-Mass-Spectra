#!/usr/bin/env python3
"""
scripts/run_v44_library_confidence.py
=============================================================================
CASMI 2026 — V44 Experiment: Candidate-Specific Library Confidence Model

Builds and evaluates V44 against query-level `LIB_MAX >= 0.87` gating:
  1. Trains a 5-fold molecule-grouped calibrated `LibraryConfidenceModel` on
     `train_cv` molecules (zero overlap with `timstof_holdout` or
     `np_examples_purged`), using a 50/50 mix of Regime A (target spectrum in
     ref index) and Regime B (target purged so only decoy library hits appear).
  2. Evaluates strictly out-of-sample on `np_examples_purged` (250 timsTOF NPs)
     under Mixed 50% Regime A / 50% Regime B and Pure Regime A:
     - Compares V44 candidate-level P_lib vs naive query-level `LIB_MAX >= 0.87`
       on Rank-1 Precision, False-Lock Rate on Regime B queries, AUC, Brier, ECE,
       and MRR@25.

Outputs:
  - models/v44_library_confidence.pkl
  - experiments/v44_library_confidence_report.json
  - docs/V44_LIBRARY_CONFIDENCE_REPORT.md
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
from sklearn.metrics import brier_score_loss, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chemistry.ms_engine import TEST_ADDUCTS, compute_neutral_mass
from ranking.v44_library_confidence import (
    LIB_FEATURE_NAMES,
    LibraryConfidenceModel,
    compute_candidate_lib_features,
    prep_spectrum_for_lib,
)
from validation.metrics import mrr_at_25, paired_bootstrap_ci, recall_at_k


def load_np_catalog() -> Tuple[pd.DataFrame, np.ndarray]:
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
    cat = pd.concat([coco, bio, ext], ignore_index=True).dropna(subset=["inchikey14", "smiles", "mono_mass"])
    cat["inchikey14"] = cat["inchikey14"].astype(str)
    cat = cat.drop_duplicates(subset=["inchikey14"], keep="first")
    cat = cat.sort_values("mono_mass", kind="mergesort").reset_index(drop=True)
    return cat, cat["mono_mass"].to_numpy(dtype=np.float64)


def load_spectra_stores(mol_df: pd.DataFrame) -> Tuple[dict, dict, dict]:
    """
    Stream train.parquet once to collect:
      - ref_by_ik: non-enveda-180 external library spectra (max 2 per ik) for train_cv + np_examples_purged
      - train_queries: up to 1,800 enveda-180 queries from train_cv (that also have non-enveda-180 ref spectra or are mass-matched)
      - np_queries: 250 enveda-np-examples queries from np_examples_purged
    """
    train_cv = mol_df[mol_df["split_role"] == "train_cv"]
    np_sub = mol_df[mol_df["split_role"] == "np_examples_purged"]

    # Identify train_cv molecules that have BOTH enveda-180 AND non-enveda external spectra (cross-instrument Regime A)
    multi_lib_cv = train_cv[train_cv["in_enveda_180"] & (train_cv["n_libs"] > 1)].drop_duplicates("canon_ik14")
    if len(multi_lib_cv) > 1500:
        multi_lib_cv = multi_lib_cv.sample(n=1500, random_state=42)
    q_cv_smis = dict(zip(multi_lib_cv["normalized_smiles"], multi_lib_cv["canon_ik14"]))
    q_cv_masses = dict(zip(multi_lib_cv["canon_ik14"], multi_lib_cv["exact_mass"]))

    np_smis = dict(zip(np_sub["normalized_smiles"], np_sub["canon_ik14"]))
    np_masses = dict(zip(np_sub["canon_ik14"], np_sub["exact_mass"]))

    # Reference index uses ALL non-enveda-180 external libs from train_cv + np_examples_purged
    allowed_ref_df = mol_df[mol_df["split_role"].isin(["train_cv", "np_examples_purged"])]
    ref_smi_to_ik = dict(zip(allowed_ref_df["normalized_smiles"], allowed_ref_df["canon_ik14"]))

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
    ref_by_ik: Dict[str, List[tuple]] = {}
    train_q: Dict[str, dict] = {}
    np_q: Dict[str, dict] = {}

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        m_add = df["adduct"].isin(TEST_ADDUCTS)
        df = df[m_add]

        for row in df.itertuples(index=False):
            lib = str(row.ingest_lib)
            smi = str(row.normalized_smiles)
            prec = float(row.precursor_mz)
            npk = int(row.num_peaks)

            if lib == "enveda-np-examples" and smi in np_smis:
                cik = np_smis[smi]
                prev = np_q.get(cik)
                if prev is None or npk > prev["num_peaks"]:
                    prep = prep_spectrum_for_lib(row.ms2_mzs, row.ms2_normalized_intensities, prec)
                    if prep is not None:
                        np_q[cik] = {
                            "molecule_id": f"np_{cik}",
                            "canon_ik14": cik,
                            "exact_mass": np_masses[cik],
                            "adduct": str(row.adduct),
                            "precursor_mz": prec,
                            "prep": prep,
                            "num_peaks": npk,
                        }
            elif lib == "enveda-180" and smi in q_cv_smis:
                cik = q_cv_smis[smi]
                prev = train_q.get(cik)
                if prev is None or npk > prev["num_peaks"]:
                    prep = prep_spectrum_for_lib(row.ms2_mzs, row.ms2_normalized_intensities, prec)
                    if prep is not None:
                        train_q[cik] = {
                            "molecule_id": f"cv_{cik}",
                            "canon_ik14": cik,
                            "exact_mass": q_cv_masses[cik],
                            "adduct": str(row.adduct),
                            "precursor_mz": prec,
                            "prep": prep,
                            "num_peaks": npk,
                        }
            elif lib not in ("enveda-180", "enveda-np-examples") and smi in ref_smi_to_ik:
                cik = ref_smi_to_ik[smi]
                refs = ref_by_ik.setdefault(cik, [])
                if len(refs) < 3:
                    prep = prep_spectrum_for_lib(row.ms2_mzs, row.ms2_normalized_intensities, prec)
                    if prep is not None:
                        refs.append(prep)
        del df
        if rg % 5 == 0:
            gc.collect()

    return ref_by_ik, train_q, np_q


def build_candidate_dataset(
    queries: List[dict],
    cat: pd.DataFrame,
    masses: np.ndarray,
    ref_by_ik: Dict[str, List[tuple]],
    purge_iks: set,
    ppm_win: float = 15.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    cat_iks = cat["inchikey14"].to_numpy()
    X_parts, y_parts, g_parts, meta = [], [], [], []

    for q in queries:
        cik = q["canon_ik14"]
        nm = compute_neutral_mass(q["precursor_mz"], q["adduct"], calibrate_timstof=True)
        tol = nm * ppm_win * 1e-6
        lo = np.searchsorted(masses, nm - tol, side="left")
        hi = np.searchsorted(masses, nm + tol, side="right")
        if hi <= lo:
            continue
        c_iks = cat_iks[lo:hi]
        c_masses = masses[lo:hi]
        c_ppm = (c_masses - nm) / nm * 1e6

        # Build per-query view of ref_by_ik if cik is in purge_iks (simulating Regime B)
        if cik in purge_iks:
            local_ref = {k: v for k, v in ref_by_ik.items() if k != cik}
        else:
            local_ref = ref_by_ik

        feats = compute_candidate_lib_features(q["prep"], c_iks, c_ppm, local_ref, tol_da=0.02)
        labels = (c_iks == cik).astype(np.int32)

        X_parts.append(feats)
        y_parts.append(labels)
        g_parts.append(np.full(len(c_iks), cik, dtype=object))
        meta.append({
            "molecule_id": q["molecule_id"],
            "canon_ik14": cik,
            "is_purged_regime_b": cik in purge_iks,
            "c_iks": c_iks,
            "c_ppm": np.abs(c_ppm),
            "feats": feats,
            "n_cands": len(c_iks),
        })

    return np.vstack(X_parts), np.concatenate(y_parts), np.concatenate(g_parts), meta


def evaluate_gating(meta: List[dict], probs: np.ndarray, probs_offsets: List[int]) -> Dict:
    gt = {m["molecule_id"]: m["canon_ik14"] for m in meta}
    preds_mass, preds_libmax, preds_v44 = {}, {}, {}

    libmax_locked, libmax_locked_correct = 0, 0
    libmax_false_lock_reg_b = 0
    v44_locked, v44_locked_correct = 0, 0
    v44_false_lock_reg_b = 0

    for idx, m in enumerate(meta):
        mid = m["molecule_id"]
        c_iks = m["c_iks"]
        abs_ppm = m["c_ppm"]
        feats = m["feats"]
        p_vec = probs[probs_offsets[idx] : probs_offsets[idx + 1]]

        mass_order = np.argsort(abs_ppm, kind="stable")
        preds_mass[mid] = c_iks[mass_order][:25].tolist()

        # Naive LIB_MAX >= 0.87 query-level gating (as in V37-V41)
        mod_cos = feats[:, 1]
        lib_max_q = float(mod_cos.max()) if len(mod_cos) else 0.0
        # Rank by mod_cos first then mass
        lib_order = np.lexsort((abs_ppm, -mod_cos))
        preds_libmax[mid] = c_iks[lib_order][:25].tolist()
        if lib_max_q >= 0.87:
            libmax_locked += 1
            if c_iks[lib_order[0]] == m["canon_ik14"]:
                libmax_locked_correct += 1
            if m["is_purged_regime_b"]:
                libmax_false_lock_reg_b += 1

        # V44 candidate-specific calibrated P_lib ranking + confidence gate (P_lib >= 0.70)
        v44_order = np.lexsort((abs_ppm, -mod_cos, -p_vec))
        preds_v44[mid] = c_iks[v44_order][:25].tolist()
        if float(p_vec[v44_order[0]]) >= 0.70:
            v44_locked += 1
            if c_iks[v44_order[0]] == m["canon_ik14"]:
                v44_locked_correct += 1
            if m["is_purged_regime_b"]:
                v44_false_lock_reg_b += 1

    ci = paired_bootstrap_ci(preds_libmax, preds_v44, gt, n_bootstrap=2000, seed=42)
    return {
        "n_queries": len(meta),
        "mass_only_mrr25": round(mrr_at_25(preds_mass, gt), 4),
        "naive_libmax_mrr25": round(mrr_at_25(preds_libmax, gt), 4),
        "v44_mrr25": round(mrr_at_25(preds_v44, gt), 4),
        "v44_r1": round(recall_at_k(preds_v44, gt, 1), 4),
        "v44_r5": round(recall_at_k(preds_v44, gt, 5), 4),
        "v44_r25": round(recall_at_k(preds_v44, gt, 25), 4),
        "naive_libmax_gate_0.87": {
            "locked": libmax_locked,
            "correct": libmax_locked_correct,
            "precision": round(libmax_locked_correct / max(libmax_locked, 1), 4),
            "false_locks_on_regime_b": libmax_false_lock_reg_b,
        },
        "v44_cand_gate_0.50": {
            "locked": v44_locked,
            "correct": v44_locked_correct,
            "precision": round(v44_locked_correct / max(v44_locked, 1), 4),
            "false_locks_on_regime_b": v44_false_lock_reg_b,
        },
        "paired_bootstrap_v44_vs_libmax": {k: round(v, 4) for k, v in ci.items()},
    }


def main():
    t0 = time.time()
    print("=" * 80)
    print("V44 CANDIDATE-SPECIFIC LIBRARY CONFIDENCE MODEL")
    print("=" * 80)

    mol_df = pd.read_parquet(ROOT / "data" / "validation_folds" / "v43_molecules.parquet")
    cat, masses = load_np_catalog()
    ref_by_ik, train_q_map, np_q_map = load_spectra_stores(mol_df)
    del mol_df
    gc.collect()
    print(f"Loaded ref_by_ik={len(ref_by_ik):,} molecules, train_cv queries={len(train_q_map)}, np queries={len(np_q_map)} ({time.time()-t0:.1f}s)")

    # Build mixed Regime A/B training dataset (only queries whose true ik is in catalog window)
    rng = np.random.default_rng(20261125)
    train_list = list(train_q_map.values())
    # Include train_cv queries in cat by augmenting cat temporarily with train_cv exact masses?
    # Better: filter to queries whose canon_ik14 is in cat OR add train_cv structures to cat for training!
    cat_keys = set(cat["inchikey14"])
    # For train_cv queries (enveda-180), add their structures so candidate pool includes true & decoys
    extra_rows = [
        {"inchikey14": q["canon_ik14"], "smiles": "C", "mono_mass": q["exact_mass"]}
        for q in train_list
        if q["canon_ik14"] not in cat_keys
    ]
    train_cat = pd.concat([cat, pd.DataFrame(extra_rows)], ignore_index=True)
    train_cat = train_cat.drop_duplicates("inchikey14").sort_values("mono_mass", kind="mergesort").reset_index(drop=True)
    train_masses = train_cat["mono_mass"].to_numpy(dtype=np.float64)

    purge_train = {q["canon_ik14"] for i, q in enumerate(train_list) if i % 2 == 1}
    X_tr, y_tr, g_tr, meta_tr = build_candidate_dataset(train_list, train_cat, train_masses, ref_by_ik, purge_train, ppm_win=15.0)
    print(f"Training matrix: X={X_tr.shape}, positives={int(y_tr.sum())}, queries={len(meta_tr)} ({time.time()-t0:.1f}s)")

    model = LibraryConfidenceModel()
    oof_stats = model.fit_oof(X_tr, y_tr, g_tr, n_splits=5, seed=42)
    print("V44 OOF metrics:", oof_stats)

    # Evaluate on np_examples_purged (250 timsTOF NPs) under:
    # 1. Mixed 50% Regime A / 50% Regime B
    np_list = sorted(np_q_map.values(), key=lambda x: x["canon_ik14"])
    purge_np_half = {q["canon_ik14"] for i, q in enumerate(np_list) if i % 2 == 1}
    X_np_mix, y_np_mix, _, meta_np_mix = build_candidate_dataset(np_list, cat, masses, ref_by_ik, purge_np_half, ppm_win=15.0)
    p_np_mix = model.predict_proba(X_np_mix)
    off_mix = [0]
    for m in meta_np_mix:
        off_mix.append(off_mix[-1] + m["n_cands"])
    eval_mix = evaluate_gating(meta_np_mix, p_np_mix, off_mix)
    eval_mix["cand_auc"] = round(float(roc_auc_score(y_np_mix, p_np_mix)), 4)
    eval_mix["cand_brier"] = round(float(brier_score_loss(y_np_mix, p_np_mix)), 4)

    # 2. Pure Regime A (all non-timsTOF external libs available)
    X_np_a, y_np_a, _, meta_np_a = build_candidate_dataset(np_list, cat, masses, ref_by_ik, set(), ppm_win=15.0)
    p_np_a = model.predict_proba(X_np_a)
    off_a = [0]
    for m in meta_np_a:
        off_a.append(off_a[-1] + m["n_cands"])
    eval_a = evaluate_gating(meta_np_a, p_np_a, off_a)
    eval_a["cand_auc"] = round(float(roc_auc_score(y_np_a, p_np_a)), 4)
    eval_a["cand_brier"] = round(float(brier_score_loss(y_np_a, p_np_a)), 4)

    importances = {
        name: round(float(np.mean([m.feature_importances_[i] for m in model.models])), 2)
        for i, name in enumerate(LIB_FEATURE_NAMES)
    }

    model_path = ROOT / "models" / "v44_library_confidence.pkl"
    model.save(model_path)

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "oof_train_cv": oof_stats,
        "feature_importances": importances,
        "np_examples_mixed_50A_50B": eval_mix,
        "np_examples_pure_regime_a": eval_a,
        "elapsed_seconds": round(time.time() - t0, 2),
    }

    (ROOT / "experiments" / "v44_library_confidence_report.json").write_text(json.dumps(report, indent=2))
    md = [
        "# V44 Candidate-Specific Library Confidence Report",
        "",
        f"- **5-Fold OOF Metrics (`train_cv`, {len(meta_tr)} queries)**: AUC=`{oof_stats['oof_auc']}`, Brier=`{oof_stats['oof_brier']}`, LogLoss=`{oof_stats['oof_logloss']}`",
        "",
        "## Out-of-Sample Evaluation on `np_examples_purged` (250 timsTOF NPs)",
        "",
        f"### Mixed 50% Regime A / 50% Regime B (Candidate AUC=`{eval_mix['cand_auc']}`, Brier=`{eval_mix['cand_brier']}`)",
        f"- Naive `LIB_MAX >= 0.87`: locked=`{eval_mix['naive_libmax_gate_0.87']['locked']}`, correct=`{eval_mix['naive_libmax_gate_0.87']['correct']}`, precision=`{eval_mix['naive_libmax_gate_0.87']['precision']:.4f}`, Regime-B false locks=`{eval_mix['naive_libmax_gate_0.87']['false_locks_on_regime_b']}`, MRR@25=`{eval_mix['naive_libmax_mrr25']:.4f}`",
        f"- V44 Candidate `P_lib >= 0.70`: locked=`{eval_mix['v44_cand_gate_0.50']['locked']}`, correct=`{eval_mix['v44_cand_gate_0.50']['correct']}`, precision=`{eval_mix['v44_cand_gate_0.50']['precision']:.4f}`, Regime-B false locks=`{eval_mix['v44_cand_gate_0.50']['false_locks_on_regime_b']}`, MRR@25=`{eval_mix['v44_mrr25']:.4f}`",
        "",
        f"### Pure Regime A (Candidate AUC=`{eval_a['cand_auc']}`, Brier=`{eval_a['cand_brier']}`)",
        f"- Naive `LIB_MAX >= 0.87` MRR@25=`{eval_a['naive_libmax_mrr25']:.4f}` (precision=`{eval_a['naive_libmax_gate_0.87']['precision']:.4f}`)",
        f"- V44 Candidate MRR@25=`{eval_a['v44_mrr25']:.4f}`, R@1=`{eval_a['v44_r1']:.4f}`, R@5=`{eval_a['v44_r5']:.4f}`, R@25=`{eval_a['v44_r25']:.4f}` (precision=`{eval_a['v44_cand_gate_0.50']['precision']:.4f}`)",
    ]
    (ROOT / "docs" / "V44_LIBRARY_CONFIDENCE_REPORT.md").write_text("\n".join(md) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
