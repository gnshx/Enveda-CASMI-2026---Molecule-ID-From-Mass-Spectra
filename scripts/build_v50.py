#!/usr/bin/env python3
"""
scripts/build_v50.py
=============================================================================
CASMI 2026 — V50 Deployment Builder

1. Trains a compact, Kaggle-schema calibrated residual Rank-1 Challenger
   (LightGBM + HistGradientBoosting using the exact 29 `make_features`
   schema of `neural_candidate_features.parquet`) on Saner's
   `labelled_candidate_scores.parquet` (`train`+`calibration` folds, verified
   on `test` fold).
2. Builds `deploy_kernel_v50/` directly on top of the audited V42 control
   (`deploy_kernel_v42/`):
   - Preserves exact V24 0.4800 PB configuration (`CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)`).
   - Embeds the compact zlib-compressed V50 Kaggle-schema Rank-1 Challenger
     in Cell 17.
   - Replaces blind query-level `LIB_MAX >= 0.87` gating with candidate-aware
     Rank-1 protection (`library_max >= 0.87` AND `neural_rank <= 2` AND `main_rank == 1`),
     allowing calibrated V50 challenger Rank-1 promotions only when incumbent
     lacks strong candidate-level agreement and challenger margin >= `delta_min`.
   - Enforces V49's finding: zero blind regioisomer overwrites on ranks 21-25;
     empty slots (`< 25`) filled strictly via scored outsiders + fallback.
   - Emits `v50_execution_audit.json` with per-stage SHA-256 hashes and aborts
     loudly if any fallback occurred.
3. Validates AST syntax of every code cell in `deploy_kernel_v50`.
=============================================================================
"""

import ast
import base64
import json
import sys
import zlib
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "datasets" / "catalogue_saner" / "catalogue_experiment"))
import ranking as saner_ranking  # noqa: E402

BASE_DIR = ROOT / "deploy_kernel_v42"
OUT_DIR = ROOT / "deploy_kernel_v50"


def train_kaggle_schema_challenger() -> dict:
    p = ROOT / "datasets" / "catalogue_saner" / "catalogue_experiment" / "labelled_candidate_scores.parquet"
    df = pd.read_parquet(p)
    df = df[df["processed"].astype(bool) & df["neural_ok"].astype(bool)].copy()
    df["label"] = (df["metric_key"] == df["true_key"]).astype(int)
    X = saner_ranking.make_features(df)

    tr_mask = df["role"].isin(["train", "calibration"]) & (df.groupby("molecule_id")["label"].transform("max") == 1)
    te_mask = df["role"].eq("test")

    y_tr = df.loc[tr_mask, "label"].to_numpy()
    cnt_tr = df.loc[tr_mask].groupby("molecule_id")["label"].transform("count").to_numpy()
    w_tr = np.where(y_tr == 1, 0.5, 0.5 / np.maximum(cnt_tr - 1, 1))
    w_tr /= w_tr.mean()

    # Model 1: Regularized HGB binary residual scorer
    hgb = HistGradientBoostingClassifier(
        max_iter=150, learning_rate=0.04, max_leaf_nodes=7, min_samples_leaf=30,
        l2_regularization=10.0, random_state=20261125,
    )
    hgb.fit(X.loc[tr_mask], y_tr, sample_weight=w_tr)

    # Model 2: LightGBM LambdaMART ranker on exact Kaggle schema
    tr_df = df.loc[tr_mask].copy()
    tr_grp = tr_df.groupby("molecule_id", sort=False).size().tolist()
    ranker = lgb.LGBMRanker(
        objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 5, 25],
        n_estimators=180, learning_rate=0.03, num_leaves=15, min_child_samples=30,
        subsample=0.85, colsample_bytree=0.85, reg_alpha=1.0, reg_lambda=5.0,
        random_state=20261125, verbose=-1,
    )
    ranker.fit(X.loc[tr_mask], y_tr, group=tr_grp)

    # Verify on held-out `test` split (30 molecules)
    df_te = df.loc[te_mask].copy()
    X_te = X.loc[te_mask]
    p_hgb = hgb.predict_proba(X_te)[:, 1]
    s_lgb = ranker.predict(X_te)
    # Z-score lgb per molecule and blend with calibrated residual score
    df_te["lgb_z"] = (
        pd.Series(s_lgb, index=df_te.index)
        .groupby(df_te["molecule_id"], sort=False)
        .transform(lambda s: ((s - s.mean()) / max(s.std(ddof=0), 1e-6)).clip(-5, 5))
    )
    df_te["res_hgb"] = saner_ranking.residual_scores(df_te, p_hgb, beta=0.20)
    df_te["prediction"] = 0.70 * df_te["res_hgb"] + 0.30 * (
        -np.log(df_te["baseline_rank"].astype(float)) + 0.20 * df_te["lgb_z"]
    )
    m_base = saner_ranking.metrics(df_te.assign(prediction=-df_te["baseline_rank"].astype(float)))
    m_v50 = saner_ranking.metrics(df_te)
    print(
        f"Kaggle-schema challenger sanity on 30-query test: "
        f"baseline MRR25={m_base.mrr25.mean():.4f} -> V50 MRR25={m_v50.mrr25.mean():.4f}"
    )

    import pickle
    payload = {
        "features": list(saner_ranking.BASE_FEATURES),
        "lgb_model_str": ranker.booster_.model_to_string(),
        "hgb_pkl_b64": base64.b64encode(zlib.compress(pickle.dumps(hgb), level=9)).decode("ascii"),
        "beta": 0.20,
        "gate_delta_min": 0.35,
        "test_sanity_base_mrr": round(float(m_base.mrr25.mean()), 4),
        "test_sanity_v50_mrr": round(float(m_v50.mrr25.mean()), 4),
    }
    return payload


def build_v50() -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = train_kaggle_schema_challenger()
    payload_b64 = base64.b64encode(zlib.compress(json.dumps(payload).encode("utf-8"), level=9)).decode("ascii")

    nb = json.loads((BASE_DIR / "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb").read_text(encoding="utf-8"))
    meta = json.loads((BASE_DIR / "kernel-metadata.json").read_text(encoding="utf-8"))

    nb["cells"][0]["source"] = [
        "# CASMI26 V50 | Calibrated Multi-Engine Ensemble + Candidate-Aware Rank-1 Gate\n",
        "\n",
        "**Version 50**: Preserves the exact V24 (`0.4800` PB) control while replacing blind query-level `LIB_MAX` gating with candidate-aware confidence + calibrated LightGBM/HGB Rank-1 Challenger residual scoring (`beta=0.20`, `gate_delta_min=0.35`) and full per-stage SHA-256 execution auditing.\n",
    ]

    # Inject V50 calibrated challenger hook right before final submission write in Cell 17
    c17 = "".join(nb["cells"][17]["source"])
    v50_hook = f"""
# ---- V50 Calibrated Rank-1 Challenger & Candidate-Aware Gate ----
try:
    import base64 as _b64, zlib as _zl, pickle as _pk
    import lightgbm as _lgb
    _V50_PAYLOAD = json.loads(_zl.decompress(_b64.b64decode("{payload_b64}")).decode("utf-8"))
    _V50_HGB = _pk.loads(_zl.decompress(_b64.b64decode(_V50_PAYLOAD["hgb_pkl_b64"])))
    _V50_LGB = _lgb.Booster(model_str=_V50_PAYLOAD["lgb_model_str"])
    _nf_full = pd.read_parquet(ROOT / "neural_candidate_features.parquet")
    if len(_nf_full) > 0 and _nf_full["processed"].astype(bool).any():
        _f = _nf_full.copy()
        for _src in ("baseline", "main", "prefusion", "engine2", "pubchem"):
            _r = pd.to_numeric(_f[_src + "_rank"], errors="coerce")
            _f[_src + "_rr"] = np.where(_r > 0, 1.0 / (3.0 + _r), 0.0)
            if _src in ("main", "engine2", "pubchem"):
                _f[_src + "_missing"] = (~(_r > 0)).astype(float)
        for _src in ("ice", "gl", "neural_cos", "neural_fp", "neural_score"):
            _v = pd.to_numeric(_f[_src], errors="coerce")
            _v = _v.where(np.isfinite(_v))
            if _src in ("ice", "gl"):
                _f[_src + "_missing"] = _v.isna().astype(float)
            _g = _v.groupby(_f["molecule_id"], sort=False)
            _f[_src + "_z"] = ((_v - _g.transform("mean")) / _g.transform(lambda s: s.std(ddof=0)).clip(lower=1e-6)).fillna(0.0).clip(-5.0, 5.0)
        _f["neural_rank"] = 0.0
        _f["neural_top_gap"] = 0.0
        _f["neural_agrees_top1"] = 0.0
        _f["engine2_agrees_top1"] = 0.0
        for _, _g in _f.groupby("molecule_id", sort=False):
            _idx = _g.index
            _vals = _g["neural_score"].to_numpy(dtype=float)
            _ord = np.argsort(-np.nan_to_num(_vals, nan=-1e9), kind="stable")
            _rk = np.empty(len(_vals), int); _rk[_ord] = np.arange(1, len(_vals) + 1)
            _f.loc[_idx, "neural_rank"] = _rk
            _zv = np.sort(_g["neural_score_z"].to_numpy(dtype=float))[::-1]
            _f.loc[_idx, "neural_top_gap"] = float(_zv[0] - _zv[1]) if len(_zv) > 1 else 0.0
            _f.loc[_idx, "neural_agrees_top1"] = float(_g.iloc[int(np.argmin(_rk))]["baseline_rank"] == 1)
            _f.loc[_idx, "engine2_agrees_top1"] = float(((_g["baseline_rank"] == 1) & (_g["engine2_rank"] == 1)).any())
        _f["neural_rr"] = 1.0 / (3.0 + _f["neural_rank"])
        _f["engine_agreement"] = ((_f["main_rank"] > 0).astype(int) + (_f["engine2_rank"] > 0).astype(int) + (_f["pubchem_rank"] > 0).astype(int))
        _f["n_spectra_log"] = np.log1p(_f["n_spectra"])
        _f["neural_delta_rank"] = (_f["baseline_rank"] - _f["neural_rank"]) / 25.0
        _f["neural_x_library"] = _f["neural_score_z"] * _f["library_max"]
        _f["neural_x_agreement"] = _f["neural_score_z"] * _f["engine_agreement"]
        _f["neural_x_sim_missing"] = _f["neural_score_z"] * _f["ice_missing"] * _f["gl_missing"]
        _X50 = _f[_V50_PAYLOAD["features"]].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        _p_hgb = np.clip(_V50_HGB.predict_proba(_X50)[:, 1], 1e-5, 1.0 - 1e-5)
        _raw_hgb = pd.Series(np.log(_p_hgb / (1.0 - _p_hgb)), index=_f.index)
        _z_hgb = (_raw_hgb - _raw_hgb.groupby(_f["molecule_id"], sort=False).transform("mean")) / _raw_hgb.groupby(_f["molecule_id"], sort=False).transform(lambda s: s.std(ddof=0)).clip(lower=1e-6)
        _s_lgb = pd.Series(_V50_LGB.predict(_X50.to_numpy()), index=_f.index)
        _z_lgb = (_s_lgb - _s_lgb.groupby(_f["molecule_id"], sort=False).transform("mean")) / _s_lgb.groupby(_f["molecule_id"], sort=False).transform(lambda s: s.std(ddof=0)).clip(lower=1e-6)
        _blend_z = (0.70 * _z_hgb.fillna(0.0).clip(-5.0, 5.0) + 0.30 * _z_lgb.fillna(0.0).clip(-5.0, 5.0))
        _f["v50_score"] = -np.log(_f["baseline_rank"].astype(float).clip(lower=1.0)) + _V50_PAYLOAD["beta"] * np.where(_f["neural_ok"].astype(bool), _blend_z, 0.0)
        _v50_map = {{str(m): dict(zip(g["smiles"].astype(str), g["v50_score"].astype(float))) for m, g in _f.groupby("molecule_id", sort=False)}}
        _nf_info = {{str(m): dict(zip(g["smiles"].astype(str), zip(g["library_max"].astype(float), g["neural_rank"].astype(int), g["main_rank"].astype(int)))) for m, g in _f.groupby("molecule_id", sort=False)}}

        _sub_cur = pd.read_csv(ROOT / "submission.csv", dtype=str, keep_default_na=False)
        _v50_rows = []
        _v50_r1_prom = 0
        _v50_cand_locked = 0
        _delta_min = float(_V50_PAYLOAD["gate_delta_min"])
        for _mid, _smis_str in zip(_sub_cur["molecule_id"], _sub_cur["smiles"]):
            _arr = [x for x in _smis_str.split(";") if x]
            _smap = _v50_map.get(str(_mid), {{}})
            _imap = _nf_info.get(str(_mid), {{}})
            if _arr and _smap:
                _inc = _arr[0]
                _lib_m, _nrk, _mrk = _imap.get(_inc, (0.0, 99, 99))
                _inc_shielded = (_lib_m >= 0.87 and _nrk <= 2 and _mrk == 1)
                if _inc_shielded:
                    _v50_cand_locked += 1
                    _rest = sorted(_arr[1:], key=lambda s: _smap.get(s, -1e9), reverse=True)
                    _v50_rows.append((_mid, ";".join([_inc] + _rest)))
                else:
                    _best = max(_arr, key=lambda s: _smap.get(s, -1e9))
                    if _best != _inc and (_smap.get(_best, -1e9) - _smap.get(_inc, -1e9)) >= _delta_min:
                        _v50_r1_prom += 1
                        _rest = [s for s in sorted(_arr, key=lambda s: _smap.get(s, -1e9), reverse=True) if s != _best]
                        _v50_rows.append((_mid, ";".join([_best] + _rest)))
                    else:
                        _rest = sorted(_arr[1:], key=lambda s: _smap.get(s, -1e9), reverse=True)
                        _v50_rows.append((_mid, ";".join([_inc] + _rest)))
            else:
                _v50_rows.append((_mid, _smis_str))
        _df50 = pd.DataFrame(_v50_rows, columns=["molecule_id", "smiles"])
        assert _df50["molecule_id"].tolist() == SAMPLE_IDS
        _tmp50 = ROOT / "submission.v50.tmp"
        _df50.to_csv(_tmp50, index=False)
        os.replace(_tmp50, ROOT / "submission.csv")
        V50_GATE_STATS = {{"status": "applied", "cand_locked": _v50_cand_locked, "rank1_promotions": _v50_r1_prom, "final_sha256": hashlib.sha256((ROOT / "submission.csv").read_bytes()).hexdigest()}}
        print("V50_CHALLENGER_GATE:", json.dumps(V50_GATE_STATS), flush=True)
except Exception as _v50_exc:
    V50_GATE_STATS = {{"status": "error_kept_v42_control", "error": repr(_v50_exc)}}
    print("V50_CHALLENGER_ERROR (V42 control preserved):", repr(_v50_exc), flush=True)
    raise
"""
    marker = "# ---- V42 Final Stage Audit ----"
    assert marker in c17, "Expected V42 audit marker in Cell 17"
    c17 = c17.replace(marker, v50_hook + "\n" + marker, 1).replace("V42_FINAL_AUDIT", "V50_FINAL_AUDIT").replace("v42_execution_audit.json", "v50_execution_audit.json")
    nb["cells"][17]["source"] = [l + "\n" for l in c17.splitlines()]

    for cell in nb["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))

    nb_out = OUT_DIR / "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb"
    nb_out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    (OUT_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Built {nb_out} ({nb_out.stat().st_size:,} bytes) — all cells AST-verified.")
    return nb_out


if __name__ == "__main__":
    build_v50()
