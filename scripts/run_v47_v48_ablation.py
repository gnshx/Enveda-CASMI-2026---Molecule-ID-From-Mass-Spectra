#!/usr/bin/env python3
"""
scripts/run_v47_v48_ablation.py
=============================================================================
CASMI 2026 — V47 & V48 Experiment:
  - V47: Multi-Adduct Mass Consensus + Elemental Subformula Consistency
  - V48: Rare-Fragment Subformula Weighting + CE-Aware Neural Pooling

Enriches `data/v45_cache/v46_*.parquet` -> `data/v45_cache/v48_*.parquet`
and evaluates incremental MRR@25 + 95% paired bootstrap CIs across:
  - V45 (baseline challenger)
  - V45 + V47 (multi-adduct & subformula consistency)
  - V45 + V47 + V48 (+ rare-fragment subformula & CE-weighted neural)
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
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator, rdMolDescriptors

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets" / "saner_model" / "casmi_spectrum_model_v1"))

from chemistry.ms_engine import ADDUCT_OFFSETS, TEST_ADDUCTS, compute_neutral_mass
from chemistry.rare_fragments import RareFragmentTable
from chemistry.v47_v48_formula_ce import (
    compute_ce_weights_for_spectra,
    enumerate_subformula_masses,
    parse_formula,
    score_formula_against_spectrum,
)
from features import metadata as feat_metadata
from features import peaks as feat_peaks
from model import SpectrumCandidateModel
from ranking.v45_rank1_challenger import Rank1ChallengerEnsemble, V45_FEATURE_COLS

V47_COLS = [
    "multi_adduct_mass_prior",
    "n_unique_adducts",
    "subform_peak_frac",
    "subform_int_frac",
    "subform_peak_z",
    "subform_int_z",
]

V48_COLS = [
    "subform_rare_int_frac",
    "subform_rare_int_z",
    "ce_neural_score",
    "ce_neural_z",
    "ce_neural_top_gap",
    "ce_neural_rr",
]


def _extract_ce(ce_raw) -> float:
    if isinstance(ce_raw, (list, np.ndarray)):
        return float(ce_raw[0]) if len(ce_raw) > 0 else 30.0
    try:
        v = float(ce_raw)
        return 30.0 if np.isnan(v) else v
    except Exception:
        return 30.0


def load_query_multispec(mol_df: pd.DataFrame, needed_by_role: Dict[str, set]) -> Dict[str, dict]:
    """Stream train.parquet once to collect up to 6 multi-CE/multi-adduct spectra per query."""
    cv_smis = mol_df[mol_df["canon_ik14"].isin(needed_by_role["cv"])].set_index("normalized_smiles")["canon_ik14"].to_dict()
    np_smis = mol_df[mol_df["canon_ik14"].isin(needed_by_role["np"])].set_index("normalized_smiles")["canon_ik14"].to_dict()
    th_smis = mol_df[mol_df["canon_ik14"].isin(needed_by_role["th"])].set_index("normalized_smiles")["canon_ik14"].to_dict()

    pf = pq.ParquetFile(str(ROOT / "data" / "train.parquet"))
    cols = [
        "ingest_lib", "normalized_smiles", "adduct", "ionization_mode",
        "precursor_mz", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities", "num_peaks",
    ]
    stores = {"cv": {}, "np": {}, "th": {}}
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        df = df[df["adduct"].isin(TEST_ADDUCTS)]
        for row in df.itertuples(index=False):
            lib = str(row.ingest_lib)
            smi = str(row.normalized_smiles)
            bucket, cik = None, None
            if lib == "enveda-180" and smi in cv_smis:
                bucket, cik = "cv", cv_smis[smi]
            elif lib == "enveda-np-examples" and smi in np_smis:
                bucket, cik = "np", np_smis[smi]
            elif lib == "enveda-180" and smi in th_smis:
                bucket, cik = "th", th_smis[smi]
            if bucket is None:
                continue

            prec = float(row.precursor_mz)
            add = str(row.adduct)
            ion = str(row.ionization_mode)
            ce = _extract_ce(row.collision_energy_ev)
            mzs = np.asarray(row.ms2_mzs, dtype=np.float64)
            its = np.asarray(row.ms2_normalized_intensities, dtype=np.float64)
            try:
                pk, ln = feat_peaks(mzs, its, prec, max_peaks=96)
                if ln < 3:
                    continue
                mt, ad_id = feat_metadata(prec, add, ion, ce)
                rec = stores[bucket].setdefault(
                    cik,
                    {
                        "spectra_nn": [],
                        "spectra_raw": [],
                        "spectra_ces": [],
                        "adduct_nms": set(),
                        "best_npk": -1,
                        "best_raw": None,
                    },
                )
                if len(rec["spectra_nn"]) < 6:
                    rec["spectra_nn"].append((pk, ln, mt, ad_id))
                    mode_i = 1 if "pos" in ion.lower() else -1
                    rec["spectra_raw"].append((mzs, its, prec, mode_i))
                    rec["spectra_ces"].append(ce)
                    nm_obs = compute_neutral_mass(prec, add, calibrate_timstof=True)
                    if nm_obs is not None:
                        rec["adduct_nms"].add((add, float(nm_obs)))
                npk = int(row.num_peaks)
                if npk > rec["best_npk"]:
                    rec["best_npk"] = npk
                    mode_i = 1 if "pos" in ion.lower() else -1
                    rec["best_raw"] = (mzs, its, prec, mode_i)
            except Exception:
                continue
        del df
        if rg % 5 == 0:
            gc.collect()
    return stores


@torch.inference_mode()
def enrich_cohort(
    df_c: pd.DataFrame,
    q_store: dict,
    ik_to_smi: Dict[str, str],
    ik_to_mass: Dict[str, float],
    neural_model: SpectrumCandidateModel,
    neural_cfg: dict,
    device: torch.device,
    rare_tbl: RareFragmentTable,
) -> pd.DataFrame:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    cand_cache: Dict[str, Tuple[np.ndarray, str]] = {}
    submass_cache: Dict[str, np.ndarray] = {}

    def get_cand_fp_form(cand_ik: str):
        v = cand_cache.get(cand_ik)
        if v is not None:
            return v
        smi = ik_to_smi.get(cand_ik)
        if not smi:
            return None
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return None
        fp = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        form = rdMolDescriptors.CalcMolFormula(m)
        res = (fp, form)
        if len(cand_cache) < 100000:
            cand_cache[cand_ik] = res
        return res

    def get_submasses(form: str) -> np.ndarray:
        arr = submass_cache.get(form)
        if arr is not None:
            return arr
        cnts = parse_formula(form)
        arr = enumerate_subformula_masses(cnts) if cnts else np.zeros(0, dtype=np.float64)
        if len(submass_cache) < 40000:
            submass_cache[form] = arr
        return arr

    n_rows = len(df_c)
    multi_add_prior = np.zeros(n_rows, dtype=np.float32)
    n_uniq_add = np.zeros(n_rows, dtype=np.float32)
    sf_peak = np.zeros(n_rows, dtype=np.float32)
    sf_int = np.zeros(n_rows, dtype=np.float32)
    sf_peak_z = np.zeros(n_rows, dtype=np.float32)
    sf_int_z = np.zeros(n_rows, dtype=np.float32)
    sf_rare = np.zeros(n_rows, dtype=np.float32)
    sf_rare_z = np.zeros(n_rows, dtype=np.float32)
    ce_n_sc = np.zeros(n_rows, dtype=np.float32)
    ce_n_z = np.zeros(n_rows, dtype=np.float32)
    ce_n_gap = np.zeros(n_rows, dtype=np.float32)
    ce_n_rr = np.zeros(n_rows, dtype=np.float32)

    for _, g in df_c.groupby("molecule_id", sort=False):
        idx = g.index.to_numpy()
        cik = str(g["canon_ik14"].iloc[0])
        q = q_store.get(cik)
        if q is None:
            continue

        c_iks = g["cand_ik14"].astype(str).tolist()
        n_c = len(c_iks)
        c_masses = np.array([ik_to_mass.get(k, 0.0) for k in c_iks], dtype=np.float64)

        # V47: Multi-adduct Gaussian mass consistency
        adduct_list = list(q["adduct_nms"])
        distinct_adds = {a for a, _ in adduct_list}
        n_uniq_add[idx] = float(len(distinct_adds))
        if adduct_list:
            priors = np.zeros(n_c, dtype=np.float64)
            for _, obs_nm in adduct_list:
                if obs_nm > 0:
                    ppm = np.abs((c_masses - obs_nm) / obs_nm * 1e6)
                    priors += np.exp(-0.5 * (ppm / 3.5) ** 2)
            multi_add_prior[idx] = (priors / len(adduct_list)).astype(np.float32)
        else:
            multi_add_prior[idx] = g["timstof_mass_prior"].to_numpy(dtype=np.float32)

        # Candidate FPs + molecular formulas
        cfps = np.zeros((n_c, 2048), dtype=np.float32)
        c_forms = [""] * n_c
        valid_c = np.zeros(n_c, dtype=bool)
        for ci, ck in enumerate(c_iks):
            res = get_cand_fp_form(ck)
            if res is not None:
                cfps[ci], c_forms[ci] = res
                valid_c[ci] = True

        # V47 + V48: Subformula peak consistency + rare-fragment subformula intensity (per unique formula!)
        best_raw = q["best_raw"]
        if best_raw is not None:
            mzs, its, prec, mode_i = best_raw
            uniq_forms = {f for f in c_forms if f}
            form_scores: Dict[str, Tuple[float, float, float]] = {}
            for f in uniq_forms:
                sm = get_submasses(f)
                form_scores[f] = score_formula_against_spectrum(sm, mzs, its, prec, mode_i, rare_tbl=rare_tbl, tol_da=0.008)
            pk_arr = np.array([form_scores[f][0] if f in form_scores else 0.0 for f in c_forms], dtype=np.float32)
            in_arr = np.array([form_scores[f][1] if f in form_scores else 0.0 for f in c_forms], dtype=np.float32)
            ra_arr = np.array([form_scores[f][2] if f in form_scores else 0.0 for f in c_forms], dtype=np.float32)
            sf_peak[idx] = pk_arr
            sf_int[idx] = in_arr
            sf_rare[idx] = ra_arr
            sf_peak_z[idx] = (pk_arr - pk_arr.mean()) / max(float(pk_arr.std()), 1e-6)
            sf_int_z[idx] = (in_arr - in_arr.mean()) / max(float(in_arr.std()), 1e-6)
            sf_rare_z[idx] = (ra_arr - ra_arr.mean()) / max(float(ra_arr.std()), 1e-6)

        # V48: CE * sqrt(n_peaks) weighted multi-spectrum neural pooling
        sp_nn = q["spectra_nn"]
        sp_raw = [{"mz": r[0]} for r in q["spectra_raw"]]
        w_ce = compute_ce_weights_for_spectra(sp_raw, q["spectra_ces"])
        x_t = torch.tensor(np.stack([s[0] for s in sp_nn]), device=device, dtype=torch.float32)
        l_t = torch.tensor([s[1] for s in sp_nn], device=device, dtype=torch.long)
        m_t = torch.tensor(np.stack([s[2] for s in sp_nn]), device=device, dtype=torch.float32)
        a_t = torch.tensor([s[3] for s in sp_nn], device=device, dtype=torch.long)
        Z, P = neural_model.encode_spectra(x_t, l_t, m_t, a_t)
        F_t = torch.tensor(cfps, device=device, dtype=torch.float32)
        M_t = neural_model.encode_molecules(F_t)
        cos_m = (Z @ M_t.T).cpu().numpy()
        fp_m = (P @ F_t.T / neural_cfg["fp_bits"]).cpu().numpy()
        comb_m = cos_m / neural_cfg["temperature"] + neural_cfg["fp_score_weight"] * fp_m
        ce_sc = (w_ce[:, None] * comb_m).sum(axis=0).astype(np.float32)
        ce_sc[~valid_c] = -1e4
        ord_c = np.argsort(-ce_sc, kind="stable")
        ranks_c = np.empty(n_c, dtype=np.int32)
        ranks_c[ord_c] = np.arange(1, n_c + 1)
        top1_c = float(ce_sc[ord_c[0]])
        top2_c = float(ce_sc[ord_c[1]]) if n_c > 1 else top1_c

        ce_n_sc[idx] = ce_sc
        ce_n_z[idx] = (ce_sc - ce_sc.mean()) / max(float(ce_sc.std()), 1e-6)
        ce_n_gap[idx] = np.where(np.arange(n_c) == ord_c[0], top1_c - top2_c, ce_sc - top1_c).astype(np.float32)
        ce_n_rr[idx] = (1.0 / (3.0 + ranks_c)).astype(np.float32)

    out = df_c.copy()
    out["multi_adduct_mass_prior"] = multi_add_prior
    out["n_unique_adducts"] = n_uniq_add
    out["subform_peak_frac"] = sf_peak
    out["subform_int_frac"] = sf_int
    out["subform_peak_z"] = sf_peak_z
    out["subform_int_z"] = sf_int_z
    out["subform_rare_int_frac"] = sf_rare
    out["subform_rare_int_z"] = sf_rare_z
    out["ce_neural_score"] = ce_n_sc
    out["ce_neural_z"] = ce_n_z
    out["ce_neural_top_gap"] = ce_n_gap
    out["ce_neural_rr"] = ce_n_rr
    return out


def run_ablation_suite(cohorts: Dict[str, pd.DataFrame], feature_sets: Dict[str, List[str]]) -> Dict:
    results = {}
    has_gold = cohorts["cv"].groupby("molecule_id")["label"].transform("max") == 1
    df_cv_tr = cohorts["cv"][has_gold].copy()

    for label, cols in feature_sets.items():
        ens = Rank1ChallengerEnsemble()
        ens.feature_cols = list(cols)
        oof = ens.fit_oof(df_cv_tr, n_splits=5, seed=42)
        res = {"oof_mrr25": oof["oof_stats"]["gated_mrr25"], "gate_thr": oof["oof_gate_threshold"]}
        for cname in ("np_b", "np_mix", "th"):
            dfe = cohorts[cname].copy()
            dfe["v_score"] = ens.predict_scores(dfe)
            st = ens.evaluate_gated_df(dfe, score_col="v_score")
            res[cname] = st
        results[label] = res
    return results


def main():
    t0 = time.time()
    print("=" * 80)
    print("V47 & V48: MULTI-ADDUCT FORMULA CONSENSUS, RARE FRAGMENTS & CE WEIGHTING")
    print("=" * 80)

    cache_dir = ROOT / "data" / "v45_cache"
    names = ("cv", "np_b", "np_mix", "th")
    v48_paths = {k: cache_dir / f"v48_{k}_feats.parquet" for k in names}

    if not all(p.is_file() for p in v48_paths.values()):
        mol_df = pd.read_parquet(ROOT / "data" / "validation_folds" / "v43_molecules.parquet")
        cohorts = {k: pd.read_parquet(cache_dir / f"v46_{k}_feats.parquet") for k in names}
        needed_by_role = {
            "cv": set(cohorts["cv"]["canon_ik14"].unique()),
            "np": set(cohorts["np_b"]["canon_ik14"].unique()),
            "th": set(cohorts["th"]["canon_ik14"].unique()),
        }

        coco = pd.read_parquet(ROOT / "datasets" / "extlib" / "coconut_index.parquet", columns=["inchikey14", "smiles_coconut", "mono_mass"]).rename(columns={"smiles_coconut": "smiles"})
        bio = pd.read_parquet(ROOT / "datasets" / "bio_structures" / "casmi-chebi-lmsd-structures" / "bio_structures.parquet", columns=["inchikey14", "smiles", "monoisotopic_mass"]).rename(columns={"monoisotopic_mass": "mono_mass"})
        ext = pd.read_parquet(ROOT / "datasets" / "extlib" / "ext_mass.parquet", columns=["inchikey14", "normalized_smiles", "mono_mass"]).rename(columns={"normalized_smiles": "smiles"})
        tr = mol_df[["canon_ik14", "normalized_smiles", "exact_mass"]].rename(columns={"canon_ik14": "inchikey14", "normalized_smiles": "smiles", "exact_mass": "mono_mass"})
        cat_all = pd.concat([coco, bio, ext, tr], ignore_index=True).dropna().drop_duplicates("inchikey14")
        ik_to_smi = dict(zip(cat_all["inchikey14"].astype(str), cat_all["smiles"].astype(str)))
        ik_to_mass = dict(zip(cat_all["inchikey14"].astype(str), cat_all["mono_mass"].astype(float)))
        del coco, bio, ext, tr, cat_all

        rare_tbl = RareFragmentTable.load(ROOT / "models" / "v45_rare_fragments.pkl")
        ck = torch.load(ROOT / "models" / "v45_fold_safe_neural.pt", map_location="cpu", weights_only=False)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        neural_model = SpectrumCandidateModel(ck["cfg"]).to(device)
        neural_model.load_state_dict(ck["state_dict"])
        neural_model.eval()

        print(f"Loading multi-CE/multi-adduct query spectra ({time.time()-t0:.1f}s)...")
        stores = load_query_multispec(mol_df, needed_by_role)
        del mol_df
        gc.collect()

        for cname, role_key in [("cv", "cv"), ("np_b", "np"), ("np_mix", "np"), ("th", "th")]:
            enr = enrich_cohort(cohorts[cname], stores[role_key], ik_to_smi, ik_to_mass, neural_model, ck["cfg"], device, rare_tbl)
            enr.to_parquet(v48_paths[cname], index=False)
            cohorts[cname] = enr
            print(f"  ✓ Enriched {cname}: {enr.shape} ({time.time()-t0:.1f}s)", flush=True)
    else:
        print("Loaded cached v48 enriched feature parquets.")
        cohorts = {k: pd.read_parquet(v48_paths[k]) for k in names}

    base_cols = [c for c in V45_FEATURE_COLS if c not in ("in_coconut", "in_chebi_lmsd")]
    feature_sets = {
        "V45_control": base_cols,
        "V47_plus_formula_adduct": base_cols + V47_COLS,
        "V48_plus_rare_ce": base_cols + V47_COLS + V48_COLS,
        "V48_plus_V46_emb": base_cols + V47_COLS + V48_COLS + ["v46_emb_tan", "v46_emb_tan_z", "v46_max_sim"],
    }

    suite = run_ablation_suite(cohorts, feature_sets)
    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "ablations": suite,
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (ROOT / "experiments" / "v47_v48_ablation_report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# V47 & V48 Ablation Report: Formula Consensus, Rare Fragments & CE Weighting",
        "",
        "| Variant | OOF MRR@25 | `np_purged` (Regime B/C) | `np_purged` (Mixed 50A/50B) | `timstof_holdout` (Regime B) |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, r in suite.items():
        md.append(
            f"| `{name}` | {r['oof_mrr25']:.4f} | **{r['np_b']['gated_mrr25']:.4f}** (R1={r['np_b']['gated_r1']:.3f}) | "
            f"**{r['np_mix']['gated_mrr25']:.4f}** (R1={r['np_mix']['gated_r1']:.3f}) | **{r['th']['gated_mrr25']:.4f}** (R1={r['th']['gated_r1']:.3f}) |"
        )
    (ROOT / "docs" / "V47_V48_ABLATION_REPORT.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
