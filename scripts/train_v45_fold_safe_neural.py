#!/usr/bin/env python3
"""
scripts/train_v45_fold_safe_neural.py
=============================================================================
CASMI 2026 — V45 Step A: Leak-Free Fold-Safe Neural Scorer & Rare-Fragment IDF

Trains:
  1. `models/v45_rare_fragments.pkl`: Mode-conditioned inverse-frequency
     rare-fragment weights strictly from `train_cv` spectra.
  2. `models/v45_fold_safe_neural.pt`: `SpectrumCandidateModel` trained strictly
     on `train_cv` molecules (0% overlap with `timstof_holdout` or
     `np_examples_purged`) on RTX 3060 with FP16 AMP and hard-negative
     same-formula contrastive pools.
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
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "datasets" / "saner_model" / "casmi_spectrum_model_v1"))

from chemistry.ms_engine import TEST_ADDUCTS
from chemistry.rare_fragments import RareFragmentTable
from features import metadata as feat_metadata
from features import peaks as feat_peaks
from model import SpectrumCandidateModel


def _init_fp_worker():
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")


def _compute_fps(smis: List[str]) -> List[Tuple[str, np.ndarray | None]]:
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator

    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    out = []
    for s in smis:
        try:
            m = Chem.MolFromSmiles(s)
            if m is None:
                out.append((s, None))
            else:
                out.append((s, gen.GetFingerprintAsNumPy(m).astype(np.uint8)))
        except Exception:
            out.append((s, None))
    return out


def hard_pools(formulas: np.ndarray, masses: np.ndarray, ids: np.ndarray, count: int = 96) -> Dict[int, np.ndarray]:
    groups = collections.defaultdict(list)
    for i in ids:
        groups[formulas[i]].append(int(i))
    order = ids[np.argsort(masses[ids], kind="stable")]
    m_ord = masses[order]
    pools = {}
    for i in ids:
        same = [j for j in groups[formulas[i]] if j != i]
        pos = int(np.searchsorted(m_ord, masses[i]))
        nearby = order[max(0, pos - count) : min(len(order), pos + count + 1)].tolist()
        nearby.sort(key=lambda j: (abs(masses[j] - masses[i]), j))
        seen = {int(i)}
        out = []
        for j in same + nearby:
            if j not in seen:
                seen.add(j)
                out.append(j)
            if len(out) >= count:
                break
        pools[int(i)] = np.asarray(out, dtype=np.int64)
    return pools


def main():
    t0 = time.time()
    print("=" * 80)
    print("V45 STEP A: LEAK-FREE RARE FRAGMENT TABLE & FOLD-SAFE NEURAL SCORER")
    print("=" * 80)

    mol_df = pd.read_parquet(ROOT / "data" / "validation_folds" / "v43_molecules.parquet")
    train_cv = mol_df[mol_df["split_role"] == "train_cv"].drop_duplicates("canon_ik14").copy()
    assert set(train_cv["canon_ik14"]).isdisjoint(set(mol_df.loc[mol_df["split_role"] != "train_cv", "canon_ik14"]))

    # 1. Fit RareFragmentTable strictly on train_cv
    rare_path = ROOT / "models" / "v45_rare_fragments.pkl"
    if not rare_path.is_file():
        print("\n[1/2] Fitting RareFragmentTable on train_cv...")
        allowed_smis = set(mol_df.loc[mol_df["split_role"] == "train_cv", "normalized_smiles"])
        rf = RareFragmentTable(bin_width=0.02, max_mz=1000.0)
        n_docs = rf.fit_from_train_cv(ROOT / "data" / "train.parquet", allowed_smis, max_spectra=250000)
        rf.save(rare_path)
        print(f"✓ Saved {rare_path} (docs={n_docs}, elapsed={time.time()-t0:.1f}s)")
    else:
        print(f"\n[1/2] Found existing {rare_path}")

    # 2. Sample up to 65,000 train_cv molecules (all non-enveda180 NPs + 45k enveda-180 timsTOF)
    print("\n[2/2] Preparing leak-free train_cv neural dataset...")
    non_e180 = train_cv[~(train_cv["in_enveda_180"] & (train_cv["n_libs"] == 1))]
    e180_only = train_cv[train_cv["in_enveda_180"] & (train_cv["n_libs"] == 1)]
    if len(e180_only) > 45000:
        e180_only = e180_only.sample(n=45000, random_state=20261125)
    sel_mol = pd.concat([non_e180, e180_only], ignore_index=True).dropna(subset=["exact_mass"])
    sel_mol = sel_mol.drop_duplicates("canon_ik14").reset_index(drop=True)

    smi_to_idx = dict(zip(sel_mol["normalized_smiles"], range(len(sel_mol))))
    n_mols = len(sel_mol)
    print(f"Selected {n_mols:,} train_cv molecules (fold 4 held out for early stopping).")

    # Compute 2048-bit Morgan fingerprints in parallel
    smis = sel_mol["normalized_smiles"].tolist()
    n_workers = min(16, os.cpu_count() or 4)
    chunk = max(500, len(smis) // (n_workers * 4))
    chunks = [smis[i : i + chunk] for i in range(0, len(smis), chunk)]
    fp_arr = np.zeros((n_mols, 2048), dtype=np.uint8)
    valid_mol = np.zeros(n_mols, dtype=bool)
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_fp_worker) as ex:
        for res in ex.map(_compute_fps, chunks):
            for s, fp in res:
                if fp is not None:
                    idx = smi_to_idx[s]
                    fp_arr[idx] = fp
                    valid_mol[idx] = True

    # Stream up to 2 spectra per molecule from train.parquet
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
    ]
    spec_x, spec_len, spec_meta, spec_add, spec_mol = [], [], [], [], []
    mol_spec_cnt = np.zeros(n_mols, dtype=np.int32)
    for rg in range(pf.num_row_groups):
        df = pf.read_row_group(rg, columns=cols).to_pandas()
        m = df["normalized_smiles"].isin(smi_to_idx) & df["adduct"].isin(TEST_ADDUCTS)
        for row in df[m].itertuples(index=False):
            mi = smi_to_idx[str(row.normalized_smiles)]
            if not valid_mol[mi] or mol_spec_cnt[mi] >= 2:
                continue
            try:
                pk, ln = feat_peaks(row.ms2_mzs, row.ms2_normalized_intensities, row.precursor_mz, max_peaks=96)
                if ln < 3:
                    continue
                mt, ad = feat_metadata(row.precursor_mz, row.adduct, row.ionization_mode, row.collision_energy_ev)
                spec_x.append(pk.astype(np.float16))
                spec_len.append(ln)
                spec_meta.append(mt)
                spec_add.append(ad)
                spec_mol.append(mi)
                mol_spec_cnt[mi] += 1
            except Exception:
                continue
        del df
        if rg % 5 == 0:
            gc.collect()

    X_pk = np.stack(spec_x)
    L_arr = np.asarray(spec_len, dtype=np.int64)
    M_arr = np.stack(spec_meta).astype(np.float32)
    A_arr = np.asarray(spec_add, dtype=np.int64)
    S_mol = np.asarray(spec_mol, dtype=np.int64)
    del spec_x, spec_len, spec_meta, spec_add, spec_mol
    gc.collect()

    by_mol: Dict[int, np.ndarray] = collections.defaultdict(list)
    for si, mi in enumerate(S_mol):
        by_mol[int(mi)].append(si)
    by_mol = {k: np.asarray(v, dtype=np.int64) for k, v in by_mol.items()}

    usable = np.array([i for i in range(n_mols) if i in by_mol], dtype=np.int64)
    folds = sel_mol["fold"].to_numpy()
    tr_ids = usable[folds[usable] != 4]
    va_ids = usable[folds[usable] == 4]
    print(f"Prepared {len(S_mol):,} spectra across {len(usable):,} molecules (train={len(tr_ids):,}, val={len(va_ids):,}, {time.time()-t0:.1f}s)")

    formulas = sel_mol["molecular_formula"].astype(str).to_numpy()
    masses = sel_mol["exact_mass"].to_numpy(dtype=np.float64)
    tr_pools = hard_pools(formulas, masses, tr_ids, count=64)
    va_pools = hard_pools(formulas, masses, usable, count=127)

    cfg = {
        "seed": 20261125,
        "fp_bits": 2048,
        "max_peaks": 96,
        "min_peaks": 3,
        "width": 192,
        "heads": 6,
        "layers": 3,
        "embedding": 192,
        "dropout": 0.1,
        "temperature": 0.1,
        "aux_weight": 0.5,
        "fp_score_weight": 2.0,
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])

    model = SpectrumCandidateModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    counts = fp_arr[tr_ids].sum(axis=0).astype(np.float64)
    pos_weight = torch.tensor(np.clip((len(tr_ids) - counts) / (counts + 1.0), 1.0, 20.0), device=device, dtype=torch.float32)

    best_mrr = -1.0
    out_pt = ROOT / "models" / "v45_fold_safe_neural.pt"
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    steps_per_epoch = 450
    batch_size = 128
    hard_neg = 16

    for ep in range(1, 16):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            b_ids = rng.choice(tr_ids, size=batch_size, replace=False)
            sr = np.array([rng.choice(by_mol[int(i)]) for i in b_ids])
            neg = np.stack([rng.choice(tr_pools[int(i)], size=hard_neg, replace=False) for i in b_ids])

            x_t = torch.tensor(X_pk[sr], device=device, dtype=torch.float32)
            l_t = torch.tensor(L_arr[sr], device=device, dtype=torch.long)
            m_t = torch.tensor(M_arr[sr], device=device, dtype=torch.float32)
            a_t = torch.tensor(A_arr[sr], device=device, dtype=torch.long)
            fp_t = torch.tensor(fp_arr[b_ids], device=device, dtype=torch.float32)
            nfp_t = torch.tensor(fp_arr[neg.reshape(-1)], device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                z, pred = model.encode_spectra(x_t, l_t, m_t, a_t)
                mol_e = model.encode_molecules(fp_t)
                neg_e = model.encode_molecules(nfp_t).reshape(batch_size, hard_neg, -1)
                pos_l = z @ mol_e.T / cfg["temperature"]
                neg_l = torch.einsum("bd,bkd->bk", z, neg_e) / cfg["temperature"]
                dup = np.isin(neg, b_ids)
                neg_l = neg_l.masked_fill(torch.tensor(dup, device=device), float("-inf"))
                loss_ret = F.cross_entropy(torch.cat([pos_l, neg_l], dim=1), torch.arange(batch_size, device=device))
                loss_aux = F.binary_cross_entropy_with_logits(pred, fp_t, pos_weight=pos_weight)
                loss = loss_ret + cfg["aux_weight"] * loss_aux
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach()))

        # Validate on 1,000 molecules from fold 4 (128-candidate pools)
        model.eval()
        mrrs = []
        with torch.inference_mode():
            for vi in va_ids[:1000]:
                sr = by_mol[int(vi)]
                cids = np.r_[int(vi), va_pools[int(vi)]]
                x_t = torch.tensor(X_pk[sr], device=device, dtype=torch.float32)
                l_t = torch.tensor(L_arr[sr], device=device, dtype=torch.long)
                m_t = torch.tensor(M_arr[sr], device=device, dtype=torch.float32)
                a_t = torch.tensor(A_arr[sr], device=device, dtype=torch.long)
                z, p = model.encode_spectra(x_t, l_t, m_t, a_t)
                zm = z.mean(0)
                pm = p.mean(0)
                cfp = torch.tensor(fp_arr[cids], device=device, dtype=torch.float32)
                sc = (model.encode_molecules(cfp) @ zm) / cfg["temperature"] + cfg["fp_score_weight"] * (cfp @ pm) / cfg["fp_bits"]
                sc = sc.cpu().numpy()
                rk = 1 + int(np.sum(sc[1:] >= sc[0]))
                mrrs.append(1.0 / rk if rk <= 25 else 0.0)
        val_mrr = float(np.mean(mrrs))
        print(f"Epoch {ep:02d} | loss={np.mean(losses):.4f} | fold4_diag_mrr25={val_mrr:.4f} | elapsed={time.time()-t0:.1f}s", flush=True)
        if val_mrr > best_mrr:
            best_mrr = val_mrr
            torch.save({"state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "cfg": cfg, "best_mrr": best_mrr}, out_pt)

    print(f"✓ Saved leak-free fold-safe neural checkpoint: {out_pt} (best_mrr={best_mrr:.4f})")


if __name__ == "__main__":
    main()
