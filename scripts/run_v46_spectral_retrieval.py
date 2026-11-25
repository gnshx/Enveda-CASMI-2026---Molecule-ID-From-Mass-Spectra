#!/usr/bin/env python3
"""
scripts/run_v46_spectral_retrieval.py
=============================================================================
CASMI 2026 — V46 Experiment: timsTOF-Adapted Spectral Similarity Retrieval

1. Trains `SpectralSimilarityEncoder` strictly on `train_cv` (zero overlap
   with `timstof_holdout` or `np_examples_purged`), using:
     - Pairwise Tanimoto similarity regression: MSE(0.5*(1 + z_i @ z_j), tanimoto_ij)
       focused on hard mass-adjacent & cross-instrument spectrum pairs
     - Auxiliary Morgan-2048 fingerprint prediction BCE
     - Emphasis on timsTOF spectra (upweighting enveda-180 & high-peak spectra)
2. Indexes disjoint `train_cv` reference spectra into dense L2-normalized
   embeddings and evaluates V46 analog retrieval against V43 binned-cosine
   analog retrieval on `np_examples_purged` (250 NPs) and `timstof_holdout` (1,000).
3. Extracts and caches `v46_emb_tan` + `v46_max_sim` features for V50 ensemble!

Outputs:
  - models/v46_spectral_encoder.pt
  - data/v45_cache/v46_np_b_feats.parquet, v46_np_mix_feats.parquet, v46_th_feats.parquet, v46_cv_feats.parquet
  - experiments/v46_spectral_retrieval_report.json
  - docs/V46_SPECTRAL_RETRIEVAL_REPORT.md
=============================================================================
"""

import collections
import gc
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
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from torch.nn import functional as F

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets" / "saner_model" / "casmi_spectrum_model_v1"))

from chemistry.ms_engine import TEST_ADDUCTS, compute_neutral_mass
from features import metadata as feat_metadata
from features import peaks as feat_peaks
from retrieval.v46_spectral_encoder import SpectralSimilarityEncoder
from validation.metrics import mrr_at_25, paired_bootstrap_ci, recall_at_k


def _init_w():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def _fps_batch(smis: List[str]) -> List[Tuple[str, np.ndarray | None]]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s in smis:
        try:
            m = Chem.MolFromSmiles(s)
            out.append((s, gen.GetFingerprintAsNumPy(m).astype(np.uint8) if m is not None else None))
        except Exception:
            out.append((s, None))
    return out


def train_v46_encoder(mol_df: pd.DataFrame, out_pt: Path, device: torch.device) -> Tuple[SpectralSimilarityEncoder, dict]:
    t0 = time.time()
    train_cv = mol_df[mol_df["split_role"] == "train_cv"].drop_duplicates("canon_ik14").copy()
    non_e180 = train_cv[~(train_cv["in_enveda_180"] & (train_cv["n_libs"] == 1))]
    e180_only = train_cv[train_cv["in_enveda_180"] & (train_cv["n_libs"] == 1)]
    if len(e180_only) > 45000:
        e180_only = e180_only.sample(n=45000, random_state=77)
    sel_mol = pd.concat([non_e180, e180_only], ignore_index=True).dropna(subset=["exact_mass"]).reset_index(drop=True)

    smi_to_idx = dict(zip(sel_mol["normalized_smiles"], range(len(sel_mol))))
    n_mols = len(sel_mol)
    smis = sel_mol["normalized_smiles"].tolist()
    n_workers = min(16, os.cpu_count() or 4)
    chunk = max(500, len(smis) // (n_workers * 4))
    fp_arr = np.zeros((n_mols, 2048), dtype=np.uint8)
    valid = np.zeros(n_mols, dtype=bool)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_w) as ex:
        for res in ex.map(_fps_batch, [smis[i : i + chunk] for i in range(0, len(smis), chunk)]):
            for s, fp in res:
                if fp is not None:
                    i = smi_to_idx[s]
                    fp_arr[i] = fp
                    valid[i] = True

    pf = pq.ParquetFile(str(ROOT / "data" / "train.parquet"))
    cols = ["ingest_lib", "normalized_smiles", "adduct", "ionization_mode", "precursor_mz", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities"]
    sx, sl, sm, sa, smol = [], [], [], [], []
    cnt = np.zeros(n_mols, dtype=np.int32)
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        m = df["normalized_smiles"].isin(smi_to_idx) & df["adduct"].isin(TEST_ADDUCTS)
        for row in df[m].itertuples(index=False):
            mi = smi_to_idx[str(row.normalized_smiles)]
            if not valid[mi] or cnt[mi] >= 2:
                continue
            try:
                pk, ln = feat_peaks(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz, max_peaks=96)
                if ln < 5:
                    continue
                mt, ad = feat_metadata(row.precursor_mz, row.adduct, row.ionization_mode, row.collision_energy_ev)
                sx.append(pk.astype(np.float16))
                sl.append(ln); sm.append(mt); sa.append(ad); smol.append(mi)
                cnt[mi] += 1
            except Exception:
                continue
        del df
        if rg % 5 == 0:
            gc.collect()

    X_pk = np.stack(sx); L_arr = np.asarray(sl, dtype=np.int64)
    M_arr = np.stack(sm).astype(np.float32); A_arr = np.asarray(sa, dtype=np.int64); S_mol = np.asarray(smol, dtype=np.int64)
    del sx, sl, sm, sa, smol
    gc.collect()

    by_mol = collections.defaultdict(list)
    for si, mi in enumerate(S_mol):
        by_mol[int(mi)].append(si)
    by_mol = {k: np.asarray(v, dtype=np.int64) for k, v in by_mol.items()}
    usable = np.array([i for i in range(n_mols) if i in by_mol], dtype=np.int64)
    folds = sel_mol["fold"].to_numpy()
    tr_ids = usable[folds[usable] != 4]
    va_ids = usable[folds[usable] == 4]

    # Sort train ids by mass so half of each batch can be mass-neighborhood hard pairs
    tr_mass_order = tr_ids[np.argsort(sel_mol["exact_mass"].to_numpy()[tr_ids])]

    cfg = {"width": 256, "heads": 8, "layers": 4, "emb_dim": 256, "dropout": 0.1}
    model = SpectralSimilarityEncoder(**cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(20261125)

    counts = fp_arr[tr_ids].sum(axis=0).astype(np.float64)
    pos_w = torch.tensor(np.clip((len(tr_ids) - counts) / (counts + 1.0), 1.0, 20.0), device=device, dtype=torch.float32)

    best_loss = 1e9
    batch_size = 128
    for ep in range(1, 13):
        model.train()
        ep_losses = []
        for _ in range(400):
            # Mix 64 random molecules + 64 mass-neighbors so high-similarity & isomer pairs appear in every batch!
            anchors = rng.integers(0, len(tr_mass_order), size=64)
            offsets = rng.integers(-40, 41, size=64)
            mates = np.clip(anchors + offsets, 0, len(tr_mass_order) - 1)
            b_ids = np.concatenate([tr_mass_order[anchors], tr_mass_order[mates]])
            sr = np.array([rng.choice(by_mol[int(i)]) for i in b_ids])

            x_t = torch.tensor(X_pk[sr], device=device, dtype=torch.float32)
            l_t = torch.tensor(L_arr[sr], device=device, dtype=torch.long)
            m_t = torch.tensor(M_arr[sr], device=device, dtype=torch.float32)
            a_t = torch.tensor(A_arr[sr], device=device, dtype=torch.long)
            fp_t = torch.tensor(fp_arr[b_ids], device=device, dtype=torch.float32)

            with torch.no_grad():
                inter = fp_t @ fp_t.T
                sums = fp_t.sum(1, keepdim=True)
                tan_target = inter / (sums + sums.T - inter + 1e-8)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                z, fp_logits = model(x_t, l_t, m_t, a_t)
                cos_pred = (z @ z.T).clamp(-1.0, 1.0)
                # Map cosine [-1,1] or [0,1] directly to Tanimoto [0,1] via 0.5*(cos+1) or direct cos on unit sphere
                # Using direct z@z.T -> tan_target works best when positive structural sims stay in [0, 1]
                loss_tan = F.smooth_l1_loss(cos_pred, tan_target)
                loss_fp = F.binary_cross_entropy_with_logits(fp_logits, fp_t, pos_weight=pos_w)
                loss = 5.0 * loss_tan + 0.5 * loss_fp
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ep_losses.append(float(loss.detach()))

        # Val pairwise Tanimoto RMSE on fold 4
        model.eval()
        val_rmses = []
        with torch.inference_mode():
            for st in range(0, min(1024, len(va_ids)), 128):
                b = va_ids[st : st + 128]
                sr = np.array([by_mol[int(i)][0] for i in b])
                x_t = torch.tensor(X_pk[sr], device=device, dtype=torch.float32)
                l_t = torch.tensor(L_arr[sr], device=device, dtype=torch.long)
                m_t = torch.tensor(M_arr[sr], device=device, dtype=torch.float32)
                a_t = torch.tensor(A_arr[sr], device=device, dtype=torch.long)
                fp_t = torch.tensor(fp_arr[b], device=device, dtype=torch.float32)
                z, _ = model(x_t, l_t, m_t, a_t)
                inter = fp_t @ fp_t.T
                sums = fp_t.sum(1, keepdim=True)
                tan_t = inter / (sums + sums.T - inter + 1e-8)
                rmse = torch.sqrt(F.mse_loss(z @ z.T, tan_t)).item()
                val_rmses.append(rmse)
        v_rmse = float(np.mean(val_rmses))
        print(f"V46 Epoch {ep:02d} | train_loss={np.mean(ep_losses):.4f} | fold4_tanimoto_rmse={v_rmse:.4f} | {time.time()-t0:.1f}s", flush=True)
        if v_rmse < best_loss:
            best_loss = v_rmse
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "cfg": cfg, "best_rmse": best_loss}, out_pt)

    ck = torch.load(out_pt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, {"best_fold4_tanimoto_rmse": round(best_loss, 4)}


def main():
    t0 = time.time()
    print("=" * 80)
    print("V46 INDEPENDENT timsTOF-ADAPTED SPECTRAL SIMILARITY RETRIEVAL")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mol_df = pd.read_parquet(ROOT / "data" / "validation_folds" / "v43_molecules.parquet")
    out_pt = ROOT / "models" / "v46_spectral_encoder.pt"

    if not out_pt.is_file():
        model, train_info = train_v46_encoder(mol_df, out_pt, device)
    else:
        ck = torch.load(out_pt, map_location="cpu", weights_only=False)
        model = SpectralSimilarityEncoder(**ck["cfg"]).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        train_info = {"best_fold4_tanimoto_rmse": round(float(ck["best_rmse"]), 4)}
        print(f"Loaded existing {out_pt} ({train_info})")

    # Build reference embeddings across train_cv (~125k disjoint molecules) + query embeddings
    train_cv = mol_df[mol_df["split_role"] == "train_cv"].drop_duplicates("canon_ik14")
    e180_ref = train_cv[train_cv["in_enveda_180"] & (train_cv["n_libs"] == 1)]
    if len(e180_ref) > 45000:
        e180_ref = e180_ref.sample(n=45000, random_state=99)
    non_e180_ref = train_cv[~(train_cv["in_enveda_180"] & (train_cv["n_libs"] == 1))]
    ref_sel = pd.concat([non_e180_ref, e180_ref], ignore_index=True)
    ref_smi_to_ik = dict(zip(ref_sel["normalized_smiles"], ref_sel["canon_ik14"]))

    # Load molecule IDs from cached V45 feature parquets so we match exact evaluation cohorts!
    cache_dir = ROOT / "data" / "v45_cache"
    cohorts = {
        "cv": pd.read_parquet(cache_dir / "train_cv_feats.parquet"),
        "np_b": pd.read_parquet(cache_dir / "np_purged_b_feats.parquet"),
        "np_mix": pd.read_parquet(cache_dir / "np_purged_mix_feats.parquet"),
        "th": pd.read_parquet(cache_dir / "timstof_holdout_feats.parquet"),
    }
    needed_iks = {
        "cv": set(cohorts["cv"]["canon_ik14"].unique()),
        "np": set(cohorts["np_b"]["canon_ik14"].unique()),
        "th": set(cohorts["th"]["canon_ik14"].unique()),
    }

    ik_to_smis_cv = mol_df[mol_df["canon_ik14"].isin(needed_iks["cv"])].set_index("normalized_smiles")["canon_ik14"].to_dict()
    ik_to_smis_np = mol_df[mol_df["canon_ik14"].isin(needed_iks["np"])].set_index("normalized_smiles")["canon_ik14"].to_dict()
    ik_to_smis_th = mol_df[mol_df["canon_ik14"].isin(needed_iks["th"])].set_index("normalized_smiles")["canon_ik14"].to_dict()

    pf = pq.ParquetFile(str(ROOT / "data" / "train.parquet"))
    cols = ["ingest_lib", "normalized_smiles", "adduct", "ionization_mode", "precursor_mz", "collision_energy_ev", "ms2_mzs", "ms2_normalized_intensities", "num_peaks"]
    q_specs = {"cv": {}, "np": {}, "th": {}}
    ref_best = {}

    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        df = df[df["adduct"].isin(TEST_ADDUCTS)]
        for row in df.itertuples(index=False):
            lib = str(row.ingest_lib)
            smi = str(row.normalized_smiles)
            prec = float(row.precursor_mz)
            npk = int(row.num_peaks)
            try:
                if lib == "enveda-180" and smi in ik_to_smis_cv:
                    cik = ik_to_smis_cv[smi]
                    prev = q_specs["cv"].get(cik)
                    if prev is None or npk > prev[0]:
                        pk, ln = feat_peaks(row.ms2_mzs, row.ms2_normalized_intensities, prec, 96)
                        if ln >= 3:
                            mt, ad = feat_metadata(prec, row.adduct, row.ionization_mode, row.collision_energy_ev)
                            q_specs["cv"][cik] = (npk, pk, ln, mt, ad)
                elif lib == "enveda-np-examples" and smi in ik_to_smis_np:
                    cik = ik_to_smis_np[smi]
                    prev = q_specs["np"].get(cik)
                    if prev is None or npk > prev[0]:
                        pk, ln = feat_peaks(row.ms2_mzs, row.ms2_normalized_intensities, prec, 96)
                        if ln >= 3:
                            mt, ad = feat_metadata(prec, row.adduct, row.ionization_mode, row.collision_energy_ev)
                            q_specs["np"][cik] = (npk, pk, ln, mt, ad)
                elif lib == "enveda-180" and smi in ik_to_smis_th:
                    cik = ik_to_smis_th[smi]
                    prev = q_specs["th"].get(cik)
                    if prev is None or npk > prev[0]:
                        pk, ln = feat_peaks(row.ms2_mzs, row.ms2_normalized_intensities, prec, 96)
                        if ln >= 3:
                            mt, ad = feat_metadata(prec, row.adduct, row.ionization_mode, row.collision_energy_ev)
                            q_specs["th"][cik] = (npk, pk, ln, mt, ad)

                if lib != "enveda-np-examples" and smi in ref_smi_to_ik:
                    rik = ref_smi_to_ik[smi]
                    prev = ref_best.get(rik)
                    if prev is None or npk > prev[0]:
                        pk, ln = feat_peaks(row.ms2_mzs, row.ms2_normalized_intensities, prec, 96)
                        if ln >= 5:
                            mt, ad = feat_metadata(prec, row.adduct, row.ionization_mode, row.collision_energy_ev)
                            ref_best[rik] = (npk, smi, pk.astype(np.float16), ln, mt, ad)
            except Exception:
                continue
        del df
        if rg % 5 == 0:
            gc.collect()

    print(f"Encoded prep: refs={len(ref_best):,}, cv={len(q_specs['cv'])}, np={len(q_specs['np'])}, th={len(q_specs['th'])} ({time.time()-t0:.1f}s)")

    ref_iks = np.array(list(ref_best.keys()))
    ref_smis = [v[1] for v in ref_best.values()]
    ref_vals = list(ref_best.values())
    del ref_best
    gc.collect()

    @torch.inference_mode()
    def encode_batch(items, idx_offset=2):
        embs = []
        for st in range(0, len(items), 256):
            b = items[st : st + 256]
            x_t = torch.tensor(np.stack([v[idx_offset] for v in b]), device=device, dtype=torch.float32)
            l_t = torch.tensor([v[idx_offset + 1] for v in b], device=device, dtype=torch.long)
            m_t = torch.tensor(np.stack([v[idx_offset + 2] for v in b]), device=device, dtype=torch.float32)
            a_t = torch.tensor([v[idx_offset + 3] for v in b], device=device, dtype=torch.long)
            z, _ = model(x_t, l_t, m_t, a_t)
            embs.append(z.cpu().numpy())
        return np.vstack(embs)

    Z_ref = encode_batch(ref_vals, idx_offset=2)
    del ref_vals
    gc.collect()

    # Load candidate SMILES lookup from catalogs
    coco = pd.read_parquet(ROOT / "datasets" / "extlib" / "coconut_index.parquet", columns=["inchikey14", "smiles_coconut"]).rename(columns={"smiles_coconut": "smiles"})
    bio = pd.read_parquet(ROOT / "datasets" / "bio_structures" / "casmi-chebi-lmsd-structures" / "bio_structures.parquet", columns=["inchikey14", "smiles"])
    ext = pd.read_parquet(ROOT / "datasets" / "extlib" / "ext_mass.parquet", columns=["inchikey14", "normalized_smiles"]).rename(columns={"normalized_smiles": "smiles"})
    tr_s = mol_df[["canon_ik14", "normalized_smiles"]].rename(columns={"canon_ik14": "inchikey14", "normalized_smiles": "smiles"})
    ik_to_smi = pd.concat([coco, bio, ext, tr_s], ignore_index=True).dropna().drop_duplicates("inchikey14").set_index("inchikey14")["smiles"].to_dict()

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp_cache: Dict[str, np.ndarray] = {}

    def get_fp(smi: str):
        v = fp_cache.get(smi)
        if v is not None:
            return v
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        arr = gen.GetFingerprintAsNumPy(m).astype(np.float32)
        if len(fp_cache) < 80000:
            fp_cache[smi] = arr
        return arr

    Z_ref_t = torch.tensor(Z_ref, device=device, dtype=torch.float32)

    def add_v46_features(df_cohort: pd.DataFrame, q_dict: dict) -> pd.DataFrame:
        ciks = sorted(df_cohort["canon_ik14"].unique())
        valid_ciks = [c for c in ciks if c in q_dict]
        q_items = [q_dict[c] for c in valid_ciks]
        Z_q = encode_batch(q_items, idx_offset=1)
        Z_q_t = torch.tensor(Z_q, device=device, dtype=torch.float32)
        sims = (Z_q_t @ Z_ref_t.T).cpu().numpy()  # (n_q, n_ref)
        top_idx = np.argpartition(-sims, 25, axis=1)[:, :25]
        cik_to_top = {}
        for qi, c in enumerate(valid_ciks):
            ri = top_idx[qi]
            rs = sims[qi, ri]
            o = np.argsort(-rs)
            cik_to_top[c] = (ri[o], rs[o])

        v46_tan = np.zeros(len(df_cohort), dtype=np.float32)
        v46_max = np.zeros(len(df_cohort), dtype=np.float32)
        v46_z = np.zeros(len(df_cohort), dtype=np.float32)

        for mid, g in df_cohort.groupby("molecule_id", sort=False):
            c = str(g["canon_ik14"].iloc[0])
            idx_rows = g.index.to_numpy()
            if c not in cik_to_top:
                continue
            r_idx, r_sim = cik_to_top[c]
            afps, aw = [], []
            ms = 0.0
            for ri, rs in zip(r_idx, r_sim):
                if ref_iks[ri] == c or rs <= 0.05:
                    continue
                fp = get_fp(ref_smis[ri])
                if fp is not None:
                    afps.append(fp)
                    aw.append(float(rs) ** 3)
                    if rs > ms:
                        ms = float(rs)
            if not afps:
                continue
            af = np.stack(afps)
            w = np.asarray(aw, dtype=np.float32); w /= w.sum()
            cfps, v_pos = [], []
            for pos, cand_ik in enumerate(g["cand_ik14"]):
                smi = ik_to_smi.get(str(cand_ik))
                fp = get_fp(str(smi)) if smi else None
                if fp is not None:
                    cfps.append(fp)
                    v_pos.append(pos)
            if cfps:
                cf = np.stack(cfps)
                inter = cf @ af.T
                tan = inter / (cf.sum(1, keepdims=True) + af.sum(1, keepdims=True).T - inter + 1e-9)
                scores = (tan @ w).astype(np.float32)
                full_s = np.zeros(len(idx_rows), dtype=np.float32)
                full_s[v_pos] = scores
                v46_tan[idx_rows] = full_s
                v46_max[idx_rows] = ms
                v46_z[idx_rows] = (full_s - full_s.mean()) / max(float(full_s.std()), 1e-6)

        out = df_cohort.copy()
        out["v46_emb_tan"] = v46_tan
        out["v46_emb_tan_z"] = v46_z
        out["v46_max_sim"] = v46_max
        return out

    # Evaluate V46 retrieval standalone vs V45 analog_tan on np_b and timstof_holdout
    eval_summary = {"train_info": train_info}
    for name, df_c, q_key in [
        ("cv", cohorts["cv"], "cv"),
        ("np_b", cohorts["np_b"], "np"),
        ("np_mix", cohorts["np_mix"], "np"),
        ("th", cohorts["th"], "th"),
    ]:
        enriched = add_v46_features(df_c, q_specs[q_key])
        enriched.to_parquet(cache_dir / f"v46_{name}_feats.parquet", index=False)

        if name in ("np_b", "th"):
            gt, p_ana, p_v46, p_blend = {}, {}, {}, {}
            for mid, g in enriched.groupby("molecule_id", sort=False):
                gt[mid] = str(g["canon_ik14"].iloc[0])
                ciks = g["cand_ik14"].to_numpy()
                ppm = g["abs_ppm_cal_minus1_5"].to_numpy()
                o_ana = np.lexsort((ppm, -g["analog_tan"].to_numpy()))
                o_v46 = np.lexsort((ppm, -g["v46_emb_tan"].to_numpy()))
                o_bld = np.lexsort((ppm, -(0.5 * g["analog_tan_z"].to_numpy() + 0.5 * g["v46_emb_tan_z"].to_numpy())))
                p_ana[mid] = ciks[o_ana][:25].tolist()
                p_v46[mid] = ciks[o_v46][:25].tolist()
                p_blend[mid] = ciks[o_bld][:25].tolist()
            ci_v46 = paired_bootstrap_ci(p_ana, p_v46, gt, n_bootstrap=2000, seed=42)
            ci_bld = paired_bootstrap_ci(p_ana, p_blend, gt, n_bootstrap=2000, seed=42)
            eval_summary[name] = {
                "n": len(gt),
                "binned_analog_mrr25": round(mrr_at_25(p_ana, gt), 4),
                "v46_emb_analog_mrr25": round(mrr_at_25(p_v46, gt), 4),
                "v46_emb_r1": round(recall_at_k(p_v46, gt, 1), 4),
                "v46_emb_r5": round(recall_at_k(p_v46, gt, 5), 4),
                "v46_emb_r25": round(recall_at_k(p_v46, gt, 25), 4),
                "blend_analog_plus_v46_mrr25": round(mrr_at_25(p_blend, gt), 4),
                "ci_v46_vs_binned": {k: round(v, 4) for k, v in ci_v46.items()},
                "ci_blend_vs_binned": {k: round(v, 4) for k, v in ci_bld.items()},
            }

    (ROOT / "experiments" / "v46_spectral_retrieval_report.json").write_text(json.dumps(eval_summary, indent=2))
    md = [
        "# V46 Independent timsTOF-Adapted Spectral Similarity Retrieval Report",
        "",
        f"- **Fold-4 Pairwise Tanimoto RMSE**: `{train_info['best_fold4_tanimoto_rmse']}`",
        "",
        "| Cohort | N | Binned Cosine Analog MRR@25 | V46 Spectral Embedding MRR@25 | Blend (Binned + V46) MRR@25 | Blend 95% CI vs Binned |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for name, label in [("np_b", "`np_examples_purged` (Regime B/C)"), ("th", "`timstof_holdout` (Regime B)")]:
        r = eval_summary[name]
        ci = r["ci_blend_vs_binned"]
        md.append(
            f"| {label} | {r['n']} | {r['binned_analog_mrr25']:.4f} | **{r['v46_emb_analog_mrr25']:.4f}** | "
            f"**{r['blend_analog_plus_v46_mrr25']:.4f}** | [{ci['ci_low']:+.4f}, {ci['ci_high']:+.4f}] (p_win={ci['p_win']:.3f}) |"
        )
    (ROOT / "docs" / "V46_SPECTRAL_RETRIEVAL_REPORT.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
