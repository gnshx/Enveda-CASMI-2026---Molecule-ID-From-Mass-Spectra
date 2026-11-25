#!/usr/bin/env python3
"""
scripts/run_v45_rank1_challenger.py
=============================================================================
CASMI 2026 — V45 Experiment: Learned Rank-1 Challenger on Hard Negatives

1. Extracts candidate-specific multi-family evidence (V44 Library, Fold-Safe
   Neural Scorer, Analog & Rare-Fragment IDF Retrieval, MetFrag-lite
   Fragmentation, Chemistry/Mass/Provenance) across:
     - `train_cv` (1,500 queries with mixed Regime A/B simulation)
     - `np_examples_purged` (250 untouched timsTOF NPs — evaluated in Pure Regime B/C & Mixed 50A/50B)
     - `timstof_holdout` (1,000 untouched enveda-180 molecules — Pure Regime B)
   Caches extracted candidate tables to `data/v45_cache/`.
2. Trains LightGBM LambdaMART + CatBoost Ranker via 5-fold molecule-grouped
   OOF CV on `train_cv`, tuning the Rank-1 challenger gate strictly on OOF.
3. Evaluates strictly out-of-sample on `np_examples_purged` and
   `timstof_holdout` and writes `experiments/v45_rank1_challenger_report.json`
   and `docs/V45_RANK1_CHALLENGER_REPORT.md`.
=============================================================================
"""

import gc
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from scipy import sparse

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets" / "saner_model" / "casmi_spectrum_model_v1"))

from chemistry import frag_rescore
from chemistry.ms_engine import TEST_ADDUCTS, compute_neutral_mass
from chemistry.rare_fragments import RareFragmentTable
from features import metadata as feat_metadata
from features import peaks as feat_peaks
from model import SpectrumCandidateModel
from ranking.v44_library_confidence import (
    LibraryConfidenceModel,
    compute_candidate_lib_features,
    prep_spectrum_for_lib,
)
from ranking.v45_rank1_challenger import Rank1ChallengerEnsemble, V45_FEATURE_COLS


def load_catalogs(mol_df: pd.DataFrame):
    coco = pd.read_parquet(
        ROOT / "datasets" / "extlib" / "coconut_index.parquet",
        columns=["inchikey14", "smiles_coconut", "mono_mass"],
    ).rename(columns={"smiles_coconut": "smiles"})
    bio = pd.read_parquet(
        ROOT / "datasets" / "bio_structures" / "casmi-chebi-lmsd-structures" / "bio_structures.parquet",
        columns=["inchikey14", "smiles", "monoisotopic_mass"],
    ).rename(columns={"monoisotopic_mass": "mono_mass"})
    ext = pd.read_parquet(
        ROOT / "datasets" / "extlib" / "ext_mass.parquet",
        columns=["inchikey14", "normalized_smiles", "mono_mass"],
    ).rename(columns={"normalized_smiles": "smiles"})

    coco_keys = set(coco["inchikey14"].dropna().astype(str))
    bio_keys = set(bio["inchikey14"].dropna().astype(str))

    np_cat = pd.concat([coco, bio, ext], ignore_index=True).dropna(subset=["inchikey14", "smiles", "mono_mass"])
    np_cat["inchikey14"] = np_cat["inchikey14"].astype(str)
    np_cat = np_cat.drop_duplicates(subset=["inchikey14"], keep="first").copy()
    np_cat["in_coco"] = np_cat["inchikey14"].isin(coco_keys).astype(np.float32)
    np_cat["in_chebi_lmsd"] = np_cat["inchikey14"].isin(bio_keys).astype(np.float32)
    np_cat = np_cat.sort_values("mono_mass", kind="mergesort").reset_index(drop=True)

    tr = mol_df[["canon_ik14", "normalized_smiles", "exact_mass"]].dropna().copy()
    tr = tr.rename(columns={"canon_ik14": "inchikey14", "normalized_smiles": "smiles", "exact_mass": "mono_mass"})
    tr["in_coco"] = 0.0
    tr["in_chebi_lmsd"] = 0.0
    ext_cat = pd.concat([np_cat, tr], ignore_index=True).drop_duplicates("inchikey14", keep="first")
    ext_cat = ext_cat.sort_values("mono_mass", kind="mergesort").reset_index(drop=True)
    return np_cat, ext_cat


def bin_vec(mzs: np.ndarray, its: np.ndarray, prec: float, bw: float = 0.05, max_mz: float = 1000.0):
    nc = int(max_mz / bw)
    ok = (mzs >= 20.0) & (mzs <= prec - 1.5) & (mzs < max_mz) & (its > 0)
    if not ok.any():
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)
    m, it = mzs[ok], its[ok]
    w = np.sqrt(it)
    fb = np.clip((m / bw).astype(np.int32), 0, nc - 1)
    nl = prec - m
    n_ok = (nl >= 12.0) & (nl < max_mz)
    nb = nc + np.clip((nl[n_ok] / bw).astype(np.int32), 0, nc - 1)
    idx = np.concatenate([fb, nb])
    val = np.concatenate([w, 0.7 * w[n_ok]])
    ui, inv = np.unique(idx, return_inverse=True)
    agg = np.zeros(len(ui), dtype=np.float32)
    np.maximum.at(agg, inv, val.astype(np.float32))
    norm = float(np.linalg.norm(agg))
    if norm > 1e-12:
        agg /= norm
    return ui, agg


def load_data_and_refs(mol_df: pd.DataFrame, rare_tbl: RareFragmentTable):
    """Stream train.parquet once to build ref index + queries for train_cv, np_purged, timstof_holdout."""
    train_cv = mol_df[mol_df["split_role"] == "train_cv"].drop_duplicates("canon_ik14")
    # Pick 1,500 train_cv queries (mix of multi-lib + pure enveda-180 timsTOF)
    multi = train_cv[train_cv["in_enveda_180"] & (train_cv["n_libs"] > 1)]
    pure_e = train_cv[train_cv["in_enveda_180"] & (train_cv["n_libs"] == 1)]
    sel_multi = multi.sample(n=min(750, len(multi)), random_state=42)
    sel_pure = pure_e.sample(n=min(750, len(pure_e)), random_state=43)
    cv_q_df = pd.concat([sel_multi, sel_pure], ignore_index=True)

    np_df = mol_df[mol_df["split_role"] == "np_examples_purged"]
    th_df = mol_df[mol_df["split_role"] == "timstof_holdout"]

    cv_smis = dict(zip(cv_q_df["normalized_smiles"], cv_q_df["canon_ik14"]))
    np_smis = dict(zip(np_df["normalized_smiles"], np_df["canon_ik14"]))
    th_smis = dict(zip(th_df["normalized_smiles"], th_df["canon_ik14"]))

    # Ref index excludes timstof_holdout; external non-timsTOF refs cover train_cv + np_purged
    ref_pool_df = mol_df[mol_df["split_role"].isin(["train_cv", "np_examples_purged"])].drop_duplicates("canon_ik14")
    e180_ref = ref_pool_df[ref_pool_df["in_enveda_180"] & (ref_pool_df["n_libs"] == 1)]
    if len(e180_ref) > 40000:
        e180_ref = e180_ref.sample(n=40000, random_state=99)
    non_e180_ref = ref_pool_df[~(ref_pool_df["in_enveda_180"] & (ref_pool_df["n_libs"] == 1))]
    ref_sel = pd.concat([non_e180_ref, e180_ref], ignore_index=True)
    ref_smi_to_ik = dict(zip(ref_sel["normalized_smiles"], ref_sel["canon_ik14"]))

    pf = pq.ParquetFile(str(ROOT / "data" / "train.parquet"))
    cols = [
        "ingest_lib",
        "normalized_smiles",
        "adduct",
        "ionization_mode",
        "precursor_mz",
        "collision_energy_ev",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "num_peaks",
    ]
    q_stores = {"cv": {}, "np": {}, "th": {}}
    best_ref = {}
    ref_lib_by_ik: Dict[str, List[tuple]] = {}

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        df = df[df["adduct"].isin(TEST_ADDUCTS)]
        for row in df.itertuples(index=False):
            lib = str(row.ingest_lib)
            smi = str(row.normalized_smiles)
            prec = float(row.precursor_mz)
            add = str(row.adduct)
            ion = str(row.ionization_mode)
            npk = int(row.num_peaks)
            mzs = np.asarray(row.ms2_mzs, dtype=np.float64)
            its = np.asarray(row.ms2_normalized_intensities, dtype=np.float64)

            target_bucket = None
            cik = None
            if lib == "enveda-180" and smi in cv_smis:
                target_bucket, cik = "cv", cv_smis[smi]
            elif lib == "enveda-np-examples" and smi in np_smis:
                target_bucket, cik = "np", np_smis[smi]
            elif lib == "enveda-180" and smi in th_smis:
                target_bucket, cik = "th", th_smis[smi]

            if target_bucket is not None:
                try:
                    pk, ln = feat_peaks(mzs, its, prec, max_peaks=96)
                    if ln >= 3:
                        mt, ad_id = feat_metadata(prec, add, ion, row.collision_energy_ev)
                        rec = q_stores[target_bucket].setdefault(
                            cik,
                            {
                                "molecule_id": f"{target_bucket}_{cik}",
                                "canon_ik14": cik,
                                "adduct": add,
                                "ionization_mode": ion,
                                "precursor_mz": prec,
                                "spectra_nn": [],
                                "spectra_frag": [],
                                "best_npk": -1,
                                "best_prep": None,
                                "best_std_vec": None,
                                "best_rare_vec": None,
                            },
                        )
                        if len(rec["spectra_nn"]) < 4:
                            rec["spectra_nn"].append((pk, ln, mt, ad_id))
                            mode_i = 1 if "pos" in ion.lower() else -1
                            ok = (mzs > 0) & (its > 0) & (mzs <= prec + 0.5)
                            rec["spectra_frag"].append({"mz": mzs[ok], "it": its[ok], "adduct": add, "prec": prec, "mode": mode_i})
                        if npk > rec["best_npk"]:
                            rec["best_npk"] = npk
                            rec["precursor_mz"] = prec
                            rec["adduct"] = add
                            rec["best_prep"] = prep_spectrum_for_lib(mzs, its, prec)
                            rec["best_std_vec"] = bin_vec(mzs, its, prec)
                            rec["best_rare_vec"] = rare_tbl.rare_weighted_bin_vector(mzs, its, prec, ion)
                except Exception:
                    pass

            if lib != "enveda-np-examples" and smi in ref_smi_to_ik:
                r_ik = ref_smi_to_ik[smi]
                # Library exact-match store (non-enveda-180 only, max 3 per ik)
                if lib != "enveda-180":
                    refs = ref_lib_by_ik.setdefault(r_ik, [])
                    if len(refs) < 3:
                        pr = prep_spectrum_for_lib(mzs, its, prec)
                        if pr is not None:
                            refs.append(pr)
                # Sparse analog index (1 best spec per ik)
                prev = best_ref.get(r_ik)
                if prev is None or npk > prev[0]:
                    ui, agg = bin_vec(mzs, its, prec)
                    rui, ragg = rare_tbl.rare_weighted_bin_vector(mzs, its, prec, ion)
                    if len(ui) > 0 and len(rui) > 0:
                        best_ref[r_ik] = (npk, smi, ui.astype(np.uint16), agg.astype(np.float16), rui.astype(np.uint16), ragg.astype(np.float16))
        del df
        if rg % 5 == 0:
            gc.collect()

    n_cols = 2 * int(1000.0 / 0.05)
    indptr, idx_p, val_p, r_idx_p, r_val_p = [0], [], [], [], []
    r_indptr = [0]
    ref_iks, ref_smis = [], []
    for cik, (_, smi, ui, agg, rui, ragg) in best_ref.items():
        idx_p.append(ui.astype(np.int32)); val_p.append(agg.astype(np.float32))
        indptr.append(indptr[-1] + len(ui))
        r_idx_p.append(rui.astype(np.int32)); r_val_p.append(ragg.astype(np.float32))
        r_indptr.append(r_indptr[-1] + len(rui))
        ref_iks.append(cik); ref_smis.append(smi)
    del best_ref
    gc.collect()

    std_mat = sparse.csr_matrix((np.concatenate(val_p), np.concatenate(idx_p), np.asarray(indptr, dtype=np.int64)), shape=(len(ref_iks), n_cols), dtype=np.float32)
    rare_mat = sparse.csr_matrix((np.concatenate(r_val_p), np.concatenate(r_idx_p), np.asarray(r_indptr, dtype=np.int64)), shape=(len(ref_iks), n_cols), dtype=np.float32)
    return q_stores, ref_lib_by_ik, std_mat, rare_mat, np.asarray(ref_iks), ref_smis


@torch.inference_mode()
def extract_features_for_cohort(
    queries: List[dict],
    cat: pd.DataFrame,
    neural_model: SpectrumCandidateModel,
    neural_cfg: dict,
    device: torch.device,
    v44_model: LibraryConfidenceModel,
    ref_lib_by_ik: dict,
    std_mat: sparse.csr_matrix,
    rare_mat: sparse.csr_matrix,
    ref_iks: np.ndarray,
    ref_smis: List[str],
    bit_freq_w: np.ndarray,
    purge_target_in_lib: set,
    ppm_win: float = 15.0,
    frag_shortlist_k: int = 35,
) -> pd.DataFrame:
    masses = cat["mono_mass"].to_numpy(dtype=np.float64)
    cat_iks = cat["inchikey14"].to_numpy()
    cat_smis = cat["smiles"].to_numpy()
    cat_coco = cat["in_coco"].to_numpy()
    cat_bio = cat["in_chebi_lmsd"].to_numpy()

    # Batch sparse analog queries
    n_cols = std_mat.shape[1]
    def _q_sparse(key):
        ip, ix, va = [0], [], []
        for q in queries:
            u, a = q[key]
            ix.append(u); va.append(a); ip.append(ip[-1] + len(u))
        return sparse.csr_matrix((np.concatenate(va), np.concatenate(ix), np.asarray(ip, dtype=np.int64)), shape=(len(queries), n_cols), dtype=np.float32)

    q_std_m = _q_sparse("best_std_vec")
    q_rare_m = _q_sparse("best_rare_vec")

    def _top25(qm, rm):
        top_i = np.empty((len(queries), 25), dtype=np.int32)
        top_s = np.empty((len(queries), 25), dtype=np.float32)
        for st in range(0, len(queries), 50):
            en = min(st + 50, len(queries))
            blk = (qm[st:en] @ rm.T).toarray()
            ib = np.argpartition(-blk, 25, axis=1)[:, :25]
            for ro in range(en - st):
                ri = ib[ro]; rs = blk[ro, ri]; o = np.argsort(-rs)
                top_i[st + ro] = ri[o]; top_s[st + ro] = rs[o]
        return top_i, top_s

    std_top_i, std_top_s = _top25(q_std_m, std_mat)
    rare_top_i, rare_top_s = _top25(q_rare_m, rare_mat)

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    mol_feat_cache: Dict[str, tuple] = {}

    def get_mol_info(smi: str):
        v = mol_feat_cache.get(smi)
        if v is not None:
            return v
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        fp = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        nh = float(m.GetNumHeavyAtoms())
        ri = m.GetRingInfo()
        nr = float(ri.NumRings())
        nar = float(rdMolDescriptors.CalcNumAromaticRings(m))
        form = rdMolDescriptors.CalcMolFormula(m)
        res = (fp, nh, nr, nar, form)
        if len(mol_feat_cache) < 80000:
            mol_feat_cache[smi] = res
        return res

    frag_cfg = {
        "max_breaks": 2, "max_bonds_2": 40, "max_bonds_3": 0,
        "w1": 1.0, "w2": 0.6, "w3": 0.0, "tol": 0.015,
        "mz_power": 0.25, "intensity": "sqrt", "h_shifts": (-2, -1, 0, 1, 2),
    }

    rows_dfs = []
    for qi, q in enumerate(queries):
        cik = q["canon_ik14"]
        mid = q["molecule_id"]
        nm = compute_neutral_mass(q["precursor_mz"], q["adduct"], calibrate_timstof=True)
        tol = nm * ppm_win * 1e-6
        lo = np.searchsorted(masses, nm - tol, side="left")
        hi = np.searchsorted(masses, nm + tol, side="right")
        if hi <= lo:
            continue

        c_iks = cat_iks[lo:hi]
        c_smis = cat_smis[lo:hi]
        c_masses = masses[lo:hi]
        c_coco = cat_coco[lo:hi]
        c_bio = cat_bio[lo:hi]
        n_c = len(c_iks)
        ppm_err = (c_masses - nm) / nm * 1e6
        abs_ppm = np.abs(ppm_err)
        abs_ppm_cal = np.abs(ppm_err + 1.5)
        mass_prior = np.exp(-0.5 * (abs_ppm_cal / 3.5) ** 2)

        # 1. V44 Library Evidence (purge target if cik in purge_target_in_lib)
        local_ref = {k: v for k, v in ref_lib_by_ik.items() if k != cik} if cik in purge_target_in_lib else ref_lib_by_ik
        lib_feats = compute_candidate_lib_features(q["best_prep"], c_iks, ppm_err, local_ref, tol_da=0.02)
        v44_p = v44_model.predict_proba(lib_feats)

        # 2. Candidate fingerprints & chemistry
        cfps = np.zeros((n_c, 2048), dtype=np.float32)
        n_heavy = np.zeros(n_c, dtype=np.float32)
        n_rings = np.zeros(n_c, dtype=np.float32)
        n_arom = np.zeros(n_c, dtype=np.float32)
        c_forms = [""] * n_c
        valid_c = np.zeros(n_c, dtype=bool)
        for ci, smi in enumerate(c_smis):
            info = get_mol_info(str(smi))
            if info is not None:
                cfps[ci], n_heavy[ci], n_rings[ci], n_arom[ci], c_forms[ci] = info
                valid_c[ci] = True

        # 3. Fold-Safe Neural Scores (all query spectra)
        sp_list = q["spectra_nn"]
        x_t = torch.tensor(np.stack([s[0] for s in sp_list]), device=device, dtype=torch.float32)
        l_t = torch.tensor([s[1] for s in sp_list], device=device, dtype=torch.long)
        m_t = torch.tensor(np.stack([s[2] for s in sp_list]), device=device, dtype=torch.float32)
        a_t = torch.tensor([s[3] for s in sp_list], device=device, dtype=torch.long)
        Z, P = neural_model.encode_spectra(x_t, l_t, m_t, a_t)
        F_t = torch.tensor(cfps, device=device, dtype=torch.float32)
        M_t = neural_model.encode_molecules(F_t)
        cos_m = (Z @ M_t.T).cpu().numpy()
        fp_m = (P @ F_t.T / neural_cfg["fp_bits"]).cpu().numpy()
        comb_m = cos_m / neural_cfg["temperature"] + neural_cfg["fp_score_weight"] * fp_m

        n_cos = cos_m.mean(0).astype(np.float32)
        n_fp = fp_m.mean(0).astype(np.float32)
        n_sc = comb_m.mean(0).astype(np.float32)
        n_std = comb_m.std(0).astype(np.float32)
        n_sc[~valid_c] = -1e4

        n_ord = np.argsort(-n_sc, kind="stable")
        n_ranks = np.empty(n_c, dtype=np.int32)
        n_ranks[n_ord] = np.arange(1, n_c + 1)
        n_z = (n_sc - n_sc.mean()) / max(float(n_sc.std()), 1e-6)
        top1_n = float(n_sc[n_ord[0]])
        top2_n = float(n_sc[n_ord[1]]) if n_c > 1 else top1_n
        n_gap = np.where(np.arange(n_c) == n_ord[0], top1_n - top2_n, n_sc - top1_n).astype(np.float32)
        n_rr = (1.0 / (3.0 + n_ranks)).astype(np.float32)

        # 4. Standard & Rare-Fragment Analog Tanimoto (skip same canon_ik14!)
        def _analog_tan(t_idx, t_sim):
            afps, aw = [], []
            max_s = 0.0
            for ri, rs in zip(t_idx, t_sim):
                if ref_iks[ri] == cik or rs <= 0.05:
                    continue
                info = get_mol_info(ref_smis[ri])
                if info is not None:
                    afps.append(info[0])
                    aw.append(float(rs) ** 3)
                    if rs > max_s:
                        max_s = float(rs)
            if not afps:
                return np.zeros(n_c, dtype=np.float32), np.zeros(n_c, dtype=np.float32), 0.0
            af = np.stack(afps)
            w = np.asarray(aw, dtype=np.float32); w /= w.sum()
            inter = cfps @ af.T
            tan = inter / (cfps.sum(1, keepdims=True) + af.sum(1, keepdims=True).T - inter + 1e-9)
            w_tan = (tan @ w).astype(np.float32)
            # Frequency-weighted Tanimoto
            cfw = cfps * bit_freq_w[None, :]
            afw = af * bit_freq_w[None, :]
            iw = cfw @ afw.T
            fw_tan = ((iw / (cfw.sum(1, keepdims=True) + afw.sum(1, keepdims=True).T - iw + 1e-9)) @ w).astype(np.float32)
            return w_tan, fw_tan, max_s

        ana_tan, freq_tan, max_ana_s = _analog_tan(std_top_i[qi], std_top_s[qi])
        rare_tan, _, _ = _analog_tan(rare_top_i[qi], rare_top_s[qi])

        a_ord = np.argsort(-ana_tan, kind="stable")
        a_ranks = np.empty(n_c, dtype=np.int32)
        a_ranks[a_ord] = np.arange(1, n_c + 1)
        a_z = (ana_tan - ana_tan.mean()) / max(float(ana_tan.std()), 1e-6)
        r_z = (rare_tan - rare_tan.mean()) / max(float(rare_tan.std()), 1e-6)
        top1_a = float(ana_tan[a_ord[0]])
        top2_a = float(ana_tan[a_ord[1]]) if n_c > 1 else top1_a
        a_gap = np.where(np.arange(n_c) == a_ord[0], top1_a - top2_a, ana_tan - top1_a).astype(np.float32)

        # 5. MetFrag-lite fragmentation on shortlist union (neural + analog + mass)
        m_ord = np.argsort(abs_ppm_cal, kind="stable")
        short_idx = sorted(set(n_ord[:20].tolist() + a_ord[:15].tolist() + m_ord[:10].tolist()))[:frag_shortlist_k]
        short_smis = [str(c_smis[i]) for i in short_idx]
        f_vals = frag_rescore.score_molecule(q["spectra_frag"], short_smis, cfg=frag_cfg)
        frag_sc = np.zeros(n_c, dtype=np.float32)
        frag_comp = np.zeros(n_c, dtype=np.float32)
        for si, fv in zip(short_idx, f_vals):
            if fv is not None:
                frag_sc[si] = float(fv)
                frag_comp[si] = 1.0
        f_ord = np.argsort(-frag_sc, kind="stable")
        f_ranks = np.empty(n_c, dtype=np.int32)
        f_ranks[f_ord] = np.arange(1, n_c + 1)
        f_z = (frag_sc - frag_sc.mean()) / max(float(frag_sc.std()), 1e-6)
        top1_f = float(frag_sc[f_ord[0]])
        top2_f = float(frag_sc[f_ord[1]]) if n_c > 1 else top1_f
        f_gap = np.where(np.arange(n_c) == f_ord[0], top1_f - top2_f, frag_sc - top1_f).astype(np.float32)

        top1_formula = c_forms[n_ord[0]]
        same_form = np.array([1.0 if (cf and cf == top1_formula) else 0.0 for cf in c_forms], dtype=np.float32)

        df_q = pd.DataFrame({
            "molecule_id": mid,
            "true_canon_ik14": cik,
            "canon_ik14": c_iks,
            "label": (c_iks == cik).astype(np.int32),
            "cand_lib_present": lib_feats[:, 0],
            "cand_lib_mod_cos": lib_feats[:, 1],
            "cand_lib_entropy_sim": lib_feats[:, 2],
            "cand_lib_shared_peaks": lib_feats[:, 3],
            "cand_lib_explained_int": lib_feats[:, 4],
            "cand_lib_log_n_refs": lib_feats[:, 5],
            "cand_lib_mass_err_ppm": lib_feats[:, 6],
            "cand_lib_margin_cos": lib_feats[:, 7],
            "cand_lib_margin_entropy": lib_feats[:, 8],
            "cand_lib_max_other_cos": lib_feats[:, 9],
            "v44_lib_prob": v44_p.astype(np.float32),
            "neural_cos": n_cos,
            "neural_fp": n_fp,
            "neural_score": n_sc,
            "neural_std": n_std,
            "neural_z": n_z.astype(np.float32),
            "neural_top_gap": n_gap,
            "neural_rr": n_rr,
            "analog_tan": ana_tan,
            "analog_tan_z": a_z.astype(np.float32),
            "analog_top_gap": a_gap,
            "analog_rr": (1.0 / (3.0 + a_ranks)).astype(np.float32),
            "rare_analog_tan": rare_tan,
            "rare_analog_tan_z": r_z.astype(np.float32),
            "freq_tan": freq_tan,
            "max_analog_sim": np.full(n_c, max_ana_s, dtype=np.float32),
            "neural_agrees_analog_top1": np.full(n_c, float(n_ord[0] == a_ord[0]), dtype=np.float32),
            "abs_rank_diff_neural_analog": np.abs(n_ranks - a_ranks).astype(np.float32),
            "frag_computed": frag_comp,
            "frag_score": frag_sc,
            "frag_z": f_z.astype(np.float32),
            "frag_top_gap": f_gap,
            "frag_rr": (1.0 / (3.0 + f_ranks)).astype(np.float32),
            "ppm_err": ppm_err.astype(np.float32),
            "abs_ppm_err": abs_ppm.astype(np.float32),
            "abs_ppm_cal_minus1_5": abs_ppm_cal.astype(np.float32),
            "timstof_mass_prior": mass_prior.astype(np.float32),
            "n_heavy": n_heavy,
            "n_rings": n_rings,
            "n_arom_rings": n_arom,
            "log1p_n_cands": np.full(n_c, math.log1p(n_c), dtype=np.float32),
            "in_coconut": c_coco.astype(np.float32),
            "in_chebi_lmsd": c_bio.astype(np.float32),
            "same_formula_as_neural_top1": same_form,
        })
        rows_dfs.append(df_q)

    out = pd.concat(rows_dfs, ignore_index=True)
    # Keep true_canon_ik14 as `canon_ik14` grouping key for GroupKFold; rename candidate key
    out = out.rename(columns={"canon_ik14": "cand_ik14", "true_canon_ik14": "canon_ik14"})
    return out


def main():
    t0 = time.time()
    print("=" * 80)
    print("V45 LEARNED RANK-1 CHALLENGER (LGBM LAMBDARANK + CATBOOST + GATE)")
    print("=" * 80)

    cache_dir = ROOT / "data" / "v45_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cv_p = cache_dir / "train_cv_feats.parquet"
    np_b_p = cache_dir / "np_purged_b_feats.parquet"
    np_mix_p = cache_dir / "np_purged_mix_feats.parquet"
    th_p = cache_dir / "timstof_holdout_feats.parquet"

    if not (cv_p.is_file() and np_b_p.is_file() and np_mix_p.is_file() and th_p.is_file()):
        mol_df = pd.read_parquet(ROOT / "data" / "validation_folds" / "v43_molecules.parquet")
        np_cat, ext_cat = load_catalogs(mol_df)
        rare_tbl = RareFragmentTable.load(ROOT / "models" / "v45_rare_fragments.pkl")
        v44_model = LibraryConfidenceModel.load(ROOT / "models" / "v44_library_confidence.pkl")
        bit_freq = np.load(ROOT / "models" / "bit_frequencies.npy").astype(np.float32)[:2048]
        bit_freq_w = 1.0 / np.sqrt(np.maximum(bit_freq, 1e-4))
        bit_freq_w /= bit_freq_w.mean()

        ck = torch.load(ROOT / "models" / "v45_fold_safe_neural.pt", map_location="cpu", weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        neural_model = SpectrumCandidateModel(ck["cfg"]).to(device)
        neural_model.load_state_dict(ck["state_dict"])
        neural_model.eval()
        frag_rescore.warmup()

        print(f"Loading spectra & disjoint reference indices ({time.time()-t0:.1f}s)...")
        q_stores, ref_lib_by_ik, std_mat, rare_mat, ref_iks, ref_smis = load_data_and_refs(mol_df, rare_tbl)
        del mol_df
        gc.collect()
        print(f"Queries loaded: cv={len(q_stores['cv'])}, np={len(q_stores['np'])}, th={len(q_stores['th'])} ({time.time()-t0:.1f}s)")

        cv_qs = list(q_stores["cv"].values())
        purge_cv_half = {q["canon_ik14"] for i, q in enumerate(cv_qs) if i % 2 == 1}
        df_cv = extract_features_for_cohort(
            cv_qs, ext_cat, neural_model, ck["cfg"], device, v44_model,
            ref_lib_by_ik, std_mat, rare_mat, ref_iks, ref_smis, bit_freq_w, purge_cv_half,
        )
        df_cv.to_parquet(cv_p, index=False)

        np_qs = sorted(q_stores["np"].values(), key=lambda x: x["canon_ik14"])
        df_np_b = extract_features_for_cohort(
            np_qs, np_cat, neural_model, ck["cfg"], device, v44_model,
            ref_lib_by_ik, std_mat, rare_mat, ref_iks, ref_smis, bit_freq_w, {q["canon_ik14"] for q in np_qs},
        )
        df_np_b.to_parquet(np_b_p, index=False)

        purge_np_half = {q["canon_ik14"] for i, q in enumerate(np_qs) if i % 2 == 1}
        df_np_mix = extract_features_for_cohort(
            np_qs, np_cat, neural_model, ck["cfg"], device, v44_model,
            ref_lib_by_ik, std_mat, rare_mat, ref_iks, ref_smis, bit_freq_w, purge_np_half,
        )
        df_np_mix.to_parquet(np_mix_p, index=False)

        th_qs = list(q_stores["th"].values())
        df_th = extract_features_for_cohort(
            th_qs, ext_cat, neural_model, ck["cfg"], device, v44_model,
            ref_lib_by_ik, std_mat, rare_mat, ref_iks, ref_smis, bit_freq_w, {q["canon_ik14"] for q in th_qs},
        )
        df_th.to_parquet(th_p, index=False)
        print(f"✓ Extracted and cached all V45 feature tables ({time.time()-t0:.1f}s)")
    else:
        print("Loaded cached V45 feature tables from data/v45_cache/")
        df_cv = pd.read_parquet(cv_p)
        df_np_b = pd.read_parquet(np_b_p)
        df_np_mix = pd.read_parquet(np_mix_p)
        df_th = pd.read_parquet(th_p)

    # Train only on queries where the gold candidate exists in the candidate pool
    has_gold = df_cv.groupby("molecule_id")["label"].transform("max") == 1
    df_cv_train = df_cv[has_gold].copy()
    print(f"Training V45 Challenger on {df_cv_train['molecule_id'].nunique()} train_cv queries ({len(df_cv_train):,} candidate rows)...")

    ensemble = Rank1ChallengerEnsemble()
    oof_info = ensemble.fit_oof(df_cv_train, n_splits=5, seed=42)
    ensemble.save(ROOT / "models" / "v45_rank1_challenger.pkl")
    print("OOF Gate Calibration:", oof_info)

    results = {"train_cv_oof": oof_info}
    for name, dfe in [
        ("np_examples_purged_regime_b", df_np_b),
        ("np_examples_mixed_50A_50B", df_np_mix),
        ("timstof_holdout_regime_b", df_th),
    ]:
        dfe = dfe.copy()
        dfe["v45_score"] = ensemble.predict_scores(dfe)
        results[name] = ensemble.evaluate_gated_df(dfe, score_col="v45_score")

    # Feature importances from LightGBM rankers
    imp = np.mean([m.feature_importances_ for m in ensemble.lgb_models], axis=0)
    top_feats = sorted(zip(V45_FEATURE_COLS, imp), key=lambda x: -x[1])[:20]

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "results": results,
        "top_20_lgb_features": {k: round(float(v), 2) for k, v in top_feats},
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (ROOT / "experiments" / "v45_rank1_challenger_report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# V45 Learned Rank-1 Challenger Report",
        "",
        f"- **OOF Gate Threshold**: `delta_min = {oof_info['oof_gate_threshold']}` (lib lock `>= {ensemble.lib_lock_threshold}`)",
        "",
        "## Out-of-Sample Holdout Results (Zero Neural/Feature/Ref Leakage)",
        "",
        "| Holdout Cohort | N | Incumbent Neural MRR@25 | Full Rerank MRR@25 | Gated Challenger MRR@25 | Correct R1 Promotions | Incorrect R1 Demotions | Gated 95% CI vs Incumbent |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for name, label in [
        ("np_examples_purged_regime_b", "`np_examples_purged` (Regime B/C)"),
        ("np_examples_mixed_50A_50B", "`np_examples_purged` (Mixed 50A/50B)"),
        ("timstof_holdout_regime_b", "`timstof_holdout` (Regime B)"),
    ]:
        r = results[name]
        ci = r["ci_gated_vs_incumbent"]
        md.append(
            f"| {label} | {r['n_queries']} | {r['incumbent_mrr25']:.4f} | **{r['full_rerank_mrr25']:.4f}** | "
            f"**{r['gated_mrr25']:.4f}** | {r['promotions_correct']} | {r['demotions_incorrect']} | "
            f"[{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] (p_win={ci['p_win']:.3f}) |"
        )
    (ROOT / "docs" / "V45_RANK1_CHALLENGER_REPORT.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
