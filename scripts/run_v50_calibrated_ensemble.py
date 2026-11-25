#!/usr/bin/env python3
"""
scripts/run_v50_calibrated_ensemble.py
=============================================================================
CASMI 2026 — V50: Calibrated Multi-Engine Ensemble & Gate Optimization

Evaluates clean feature subsets and 3-family model diversity:
  - LightGBM LambdaMART (lambdarank)
  - CatBoost Ranker (YetiRank)
  - Calibrated HistGradientBoostingClassifier (pairwise/pointwise logloss)
Strict rule: admit only feature groups & ensemble blends that improve or
preserve OOF AND held-out metrics across `np_examples_purged` and `timstof_holdout`.

Produces:
  - models/v50_calibrated_ensemble.pkl (full ensemble)
  - models/v50_compact_deploy.json.zlib (lightweight compact LightGBM JSON for Kaggle notebook embedding)
  - experiments/v50_ensemble_report.json
  - docs/V50_FINAL_ENSEMBLE_REPORT.md
=============================================================================
"""

import json
import pickle
import time
import zlib
from pathlib import Path
from typing import Dict, List

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(ROOT))

from ranking.v45_rank1_challenger import V45_FEATURE_COLS, _group_sizes, _zscore_by_group

BASE_V45_COLS = [c for c in V45_FEATURE_COLS if c not in ("in_coconut", "in_chebi_lmsd")]

# Selective V47+V48 additions avoiding collinear bloat:
V50_SELECTED_COLS = BASE_V45_COLS + [
    "multi_adduct_mass_prior",
    "ce_neural_score",
    "ce_neural_z",
    "ce_neural_top_gap",
]


def evaluate_scores(df: pd.DataFrame, score_col: str, delta_thr: float = 0.0, lib_lock_thr: float = 0.85) -> Dict:
    inc_mrrs, full_mrrs, gated_mrrs = [], [], []
    inc_r1, full_r1, gated_r1 = [], [], []
    full_r5, full_r25, gated_r5, gated_r25 = [], [], [], []
    prom, dem, swaps, locked = 0, 0, 0, 0

    for _, g in df.groupby("molecule_id", sort=False):
        inc_ord = np.lexsort((g["abs_ppm_cal_minus1_5"].to_numpy(), -g["neural_score"].to_numpy()))
        inc_lbl = g["label"].to_numpy()[inc_ord]

        sc = g[score_col].to_numpy()
        full_ord = np.argsort(-sc, kind="stable")
        full_lbl = g["label"].to_numpy()[full_ord]

        inc_top = int(inc_ord[0])
        chal_top = int(full_ord[0])
        inc_lib_p = float(g["v44_lib_prob"].to_numpy()[inc_top])
        margin = float(sc[chal_top] - sc[inc_top])

        if inc_lib_p >= lib_lock_thr:
            locked += 1
            gated_ord = inc_ord
        elif chal_top != inc_top and margin >= delta_thr:
            swaps += 1
            rest = [i for i in full_ord if i != chal_top]
            gated_ord = np.array([chal_top] + rest, dtype=np.int64)
            if g["label"].to_numpy()[chal_top] == 1 and inc_lbl[0] == 0:
                prom += 1
            elif g["label"].to_numpy()[inc_top] == 1 and g["label"].to_numpy()[chal_top] == 0:
                dem += 1
        else:
            rest = [i for i in full_ord if i != inc_top]
            gated_ord = np.array([inc_top] + rest, dtype=np.int64)
        gated_lbl = g["label"].to_numpy()[gated_ord]

        def _m(lbl):
            pos = np.where(lbl[:25] == 1)[0]
            return (1.0 / (pos[0] + 1) if len(pos) > 0 else 0.0), int(lbl[0] == 1), int(lbl[:5].sum() > 0), int(len(pos) > 0)

        im, ir1, _, _ = _m(inc_lbl)
        fm, fr1, fr5, fr25 = _m(full_lbl)
        gm, gr1, gr5, gr25 = _m(gated_lbl)
        inc_mrrs.append(im); inc_r1.append(ir1)
        full_mrrs.append(fm); full_r1.append(fr1); full_r5.append(fr5); full_r25.append(fr25)
        gated_mrrs.append(gm); gated_r1.append(gr1); gated_r5.append(gr5); gated_r25.append(gr25)

    d = np.asarray(gated_mrrs) - np.asarray(inc_mrrs)
    rng = np.random.default_rng(42)
    boot = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
    return {
        "n": len(inc_mrrs),
        "incumbent_mrr25": round(float(np.mean(inc_mrrs)), 4),
        "full_mrr25": round(float(np.mean(full_mrrs)), 4),
        "gated_mrr25": round(float(np.mean(gated_mrrs)), 4),
        "gated_r1": round(float(np.mean(gated_r1)), 4),
        "gated_r5": round(float(np.mean(gated_r5)), 4),
        "gated_r25": round(float(np.mean(gated_r25)), 4),
        "promotions_correct": prom,
        "demotions_incorrect": dem,
        "rank1_swaps": swaps,
        "lib_locked": locked,
        "mean_delta_vs_inc": round(float(d.mean()), 4),
        "ci95_vs_inc": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "p_win_vs_inc": round(float((boot > 0).mean()), 4),
    }


def train_and_eval(df_cv: pd.DataFrame, holdouts: Dict[str, pd.DataFrame], cols: List[str], tag: str):
    df_cv = df_cv.sort_values(["molecule_id", "neural_score"], ascending=[True, False], kind="stable").reset_index(drop=True)
    X = df_cv[cols].to_numpy(dtype=np.float32)
    y = df_cv["label"].to_numpy(dtype=np.int32)
    groups = df_cv["canon_ik14"].to_numpy()
    mids = df_cv["molecule_id"].to_numpy()

    # Balanced per-molecule sample weights for HGB pointwise logloss
    mol_cnt = df_cv.groupby("molecule_id")["label"].transform("count").to_numpy()
    sw = np.where(y == 1, 0.5, 0.5 / np.maximum(mol_cnt - 1, 1))
    sw = sw / sw.mean()

    gkf = GroupKFold(n_splits=5)
    oof_lgb = np.zeros(len(df_cv), dtype=np.float64)
    oof_cb = np.zeros(len(df_cv), dtype=np.float64)
    oof_hgb = np.zeros(len(df_cv), dtype=np.float64)
    lgbs, cbs, hgbs = [], [], []

    for fold, (tr_i, va_i) in enumerate(gkf.split(X, y, groups)):
        tr_m, va_m = mids[tr_i], mids[va_i]
        ranker = lgb.LGBMRanker(
            objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 5, 25],
            n_estimators=320, learning_rate=0.03, num_leaves=23, min_child_samples=40,
            subsample=0.85, colsample_bytree=0.80, reg_alpha=1.0, reg_lambda=5.0,
            random_state=42 + fold, n_jobs=8, verbose=-1,
        )
        ranker.fit(X[tr_i], y[tr_i], group=_group_sizes(tr_m), eval_set=[(X[va_i], y[va_i])], eval_group=[_group_sizes(va_m)])
        oof_lgb[va_i] = ranker.predict(X[va_i])
        lgbs.append(ranker)

        cb = CatBoostRanker(
            iterations=400, learning_rate=0.04, depth=6, loss_function="YetiRank",
            l2_leaf_reg=5.0, random_seed=42 + fold, thread_count=8, verbose=0,
        )
        cb.fit(Pool(X[tr_i], y[tr_i].astype(np.float32), group_id=tr_m),
               eval_set=Pool(X[va_i], y[va_i].astype(np.float32), group_id=va_m),
               use_best_model=True)
        oof_cb[va_i] = cb.predict(X[va_i])
        cbs.append(cb)

        hgb = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.04, max_leaf_nodes=23, min_samples_leaf=40,
            l2_regularization=5.0, random_state=42 + fold,
        )
        hgb.fit(X[tr_i], y[tr_i], sample_weight=sw[tr_i])
        oof_hgb[va_i] = hgb.predict_proba(X[va_i])[:, 1]
        hgbs.append(hgb)

    z_lgb = _zscore_by_group(oof_lgb, mids)
    z_cb = _zscore_by_group(oof_cb, mids)
    z_hgb = _zscore_by_group(oof_hgb, mids)

    # Search blend weights (0.45 LGB + 0.45 CB + 0.10 HGB vs 0.5/0.5/0.0) & delta_thr strictly on OOF
    best_cfg = None
    best_oof = -1.0
    for w_hgb in [0.0, 0.15, 0.25]:
        w_rc = 0.5 * (1.0 - w_hgb)
        sc = w_rc * z_lgb + w_rc * z_cb + w_hgb * z_hgb
        df_tmp = df_cv[["molecule_id", "label", "neural_score", "abs_ppm_cal_minus1_5", "v44_lib_prob"]].copy()
        df_tmp["sc"] = sc
        for thr in [0.0, 0.25, 0.50]:
            st = evaluate_scores(df_tmp, "sc", delta_thr=thr, lib_lock_thr=0.85)
            if st["gated_mrr25"] > best_oof:
                best_oof = st["gated_mrr25"]
                best_cfg = (w_rc, w_rc, w_hgb, thr, st)

    w_l, w_c, w_h, thr, oof_st = best_cfg
    out = {"tag": tag, "n_features": len(cols), "weights": {"lgb": w_l, "cb": w_c, "hgb": w_h, "delta_thr": thr}, "oof": oof_st}
    for hname, hdf in holdouts.items():
        Xh = hdf[cols].to_numpy(dtype=np.float32)
        mh = hdf["molecule_id"].to_numpy()
        pl = _zscore_by_group(np.mean([m.predict(Xh) for m in lgbs], axis=0), mh)
        pc = _zscore_by_group(np.mean([m.predict(Xh) for m in cbs], axis=0), mh)
        ph = _zscore_by_group(np.mean([m.predict_proba(Xh)[:, 1] for m in hgbs], axis=0), mh)
        hdf_e = hdf[["molecule_id", "label", "neural_score", "abs_ppm_cal_minus1_5", "v44_lib_prob"]].copy()
        hdf_e["sc"] = w_l * pl + w_c * pc + w_h * ph
        out[hname] = evaluate_scores(hdf_e, "sc", delta_thr=thr, lib_lock_thr=0.85)

    bundle = {"cols": cols, "lgbs": lgbs, "cbs": cbs, "hgbs": hgbs, "weights": out["weights"]}
    return out, bundle


def main():
    t0 = time.time()
    print("=" * 80)
    print("V50 CALIBRATED MULTI-ENGINE ENSEMBLE & SELECTION")
    print("=" * 80)

    cache_dir = ROOT / "data" / "v45_cache"
    df_cv = pd.read_parquet(cache_dir / "v48_cv_feats.parquet")
    holdouts = {
        "np_b": pd.read_parquet(cache_dir / "v48_np_b_feats.parquet"),
        "np_mix": pd.read_parquet(cache_dir / "v48_np_mix_feats.parquet"),
        "th": pd.read_parquet(cache_dir / "v48_th_feats.parquet"),
    }
    has_gold = df_cv.groupby("molecule_id")["label"].transform("max") == 1
    df_cv_tr = df_cv[has_gold].copy()

    res_v45, bun_v45 = train_and_eval(df_cv_tr, holdouts, BASE_V45_COLS, "V50_3Model_Base42")
    res_v50, bun_v50 = train_and_eval(df_cv_tr, holdouts, V50_SELECTED_COLS, "V50_3Model_Plus_CE_MultiAdd")

    # Select winner strictly by consistent generalization across OOF & all held-out cohorts
    score_v45 = (res_v45["np_b"]["gated_mrr25"] + res_v45["np_mix"]["gated_mrr25"] + res_v45["th"]["gated_mrr25"]) / 3.0
    score_v50 = (res_v50["np_b"]["gated_mrr25"] + res_v50["np_mix"]["gated_mrr25"] + res_v50["th"]["gated_mrr25"]) / 3.0
    winner_res, winner_bun = (res_v50, bun_v50) if (res_v50["oof"]["gated_mrr25"] > res_v45["oof"]["gated_mrr25"] and score_v50 > score_v45) else (res_v45, bun_v45)

    with open(ROOT / "models" / "v50_calibrated_ensemble.pkl", "wb") as f:
        pickle.dump(winner_bun, f)

    # Save compact zlib-compressed LightGBM model strings for zero-dependency notebook embedding
    compact_payload = {
        "cols": winner_bun["cols"],
        "weights": winner_bun["weights"],
        "lgb_boosters": [m.booster_.model_to_string() for m in winner_bun["lgbs"]],
    }
    comp_bytes = zlib.compress(json.dumps(compact_payload).encode("utf-8"), level=9)
    (ROOT / "models" / "v50_compact_lgb.json.zlib").write_bytes(comp_bytes)

    report = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "candidates": [res_v45, res_v50],
        "selected": winner_res["tag"],
        "compact_lgb_zlib_bytes": len(comp_bytes),
        "elapsed_seconds": round(time.time() - t0, 2),
    }
    (ROOT / "experiments" / "v50_ensemble_report.json").write_text(json.dumps(report, indent=2))

    md = [
        "# V50 Calibrated Ensemble Report",
        "",
        f"- **Selected Model**: `{winner_res['tag']}` (`weights={winner_res['weights']}`)",
        f"- **Compact Deploy Payload Size**: `{len(comp_bytes):,}` bytes (`models/v50_compact_lgb.json.zlib`)",
        "",
        "| Variant | OOF MRR@25 | `np_purged` (Regime B/C) | `np_purged` (Mixed 50A/50B) | `timstof_holdout` (Regime B) |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in (res_v45, res_v50):
        md.append(
            f"| `{r['tag']}` | {r['oof']['gated_mrr25']:.4f} | "
            f"**{r['np_b']['gated_mrr25']:.4f}** (R1={r['np_b']['gated_r1']:.3f}, +{r['np_b']['promotions_correct']}/-{r['np_b']['demotions_incorrect']}) | "
            f"**{r['np_mix']['gated_mrr25']:.4f}** (R1={r['np_mix']['gated_r1']:.3f}, +{r['np_mix']['promotions_correct']}/-{r['np_mix']['demotions_incorrect']}) | "
            f"**{r['th']['gated_mrr25']:.4f}** (R1={r['th']['gated_r1']:.3f}, +{r['th']['promotions_correct']}/-{r['th']['demotions_incorrect']}) |"
        )
    (ROOT / "docs" / "V50_FINAL_ENSEMBLE_REPORT.md").write_text("\n".join(md) + "\n")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
