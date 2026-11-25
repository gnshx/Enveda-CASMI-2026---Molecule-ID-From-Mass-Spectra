"""
ranking/v45_rank1_challenger.py
=============================================================================
CASMI 2026 — V45: Learned Rank-1 Challenger (LightGBM LambdaMART + CatBoost)

Combines candidate-specific evidence across 5 feature families:
  1. V44 Library Evidence (modified cosine, entropy sim, shared peaks, margin, P_lib)
  2. Fold-Safe Neural Scorer (neural_cos, neural_fp, neural_score, std, z, gap, rr)
  3. Analog & Rare-Fragment Retrieval (Morgan Tanimoto, bit-freq Tanimoto, IDF-weighted analog)
  4. Fragmentation (MetFrag-lite <=2 bond cleavages on shortlist)
  5. Chemistry, Mass Accuracy & Provenance

Trains LightGBM LambdaMART + CatBoost Ranker via 5-fold molecule-grouped OOF CV
and applies a calibrated Rank-1 challenger gate tuned strictly on OOF CV.
=============================================================================
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRanker, Pool
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

V45_FEATURE_COLS = [
    # V44 library features
    "cand_lib_present",
    "cand_lib_mod_cos",
    "cand_lib_entropy_sim",
    "cand_lib_shared_peaks",
    "cand_lib_explained_int",
    "cand_lib_log_n_refs",
    "cand_lib_mass_err_ppm",
    "cand_lib_margin_cos",
    "cand_lib_margin_entropy",
    "cand_lib_max_other_cos",
    "v44_lib_prob",
    # Fold-safe neural features
    "neural_cos",
    "neural_fp",
    "neural_score",
    "neural_std",
    "neural_z",
    "neural_top_gap",
    "neural_rr",
    # Analog & rare-fragment features
    "analog_tan",
    "analog_tan_z",
    "analog_top_gap",
    "analog_rr",
    "rare_analog_tan",
    "rare_analog_tan_z",
    "freq_tan",
    "max_analog_sim",
    "neural_agrees_analog_top1",
    "abs_rank_diff_neural_analog",
    # Fragmentation (MetFrag-lite)
    "frag_computed",
    "frag_score",
    "frag_z",
    "frag_top_gap",
    "frag_rr",
    # Chemistry, mass accuracy & provenance
    "ppm_err",
    "abs_ppm_err",
    "abs_ppm_cal_minus1_5",
    "timstof_mass_prior",
    "n_heavy",
    "n_rings",
    "n_arom_rings",
    "log1p_n_cands",
    "same_formula_as_neural_top1",
]


def _group_sizes(mids: np.ndarray) -> List[int]:
    _, counts = np.unique(mids, return_counts=True)
    # preserve contiguous group order
    sizes = []
    cur = mids[0]
    cnt = 0
    for m in mids:
        if m == cur:
            cnt += 1
        else:
            sizes.append(cnt)
            cur = m
            cnt = 1
    sizes.append(cnt)
    return sizes


def _zscore_by_group(vals: np.ndarray, mids: np.ndarray) -> np.ndarray:
    s = pd.Series(vals)
    g = s.groupby(mids, sort=False)
    std = g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)
    return ((s - g.transform("mean")) / std).to_numpy(dtype=np.float32)


class Rank1ChallengerEnsemble:
    """5-fold OOF LightGBM LambdaMART + CatBoost Ranker + Calibrated Rank-1 Gate."""

    def __init__(self):
        self.lgb_models: List[lgb.LGBMRanker] = []
        self.cb_models: List[CatBoostRanker] = []
        self.gate_calibrator: LogisticRegression | None = None
        self.delta_threshold: float = 0.0
        self.lib_lock_threshold: float = 0.85
        self.feature_cols = list(V45_FEATURE_COLS)

    def fit_oof(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        seed: int = 42,
    ) -> Dict:
        df = df.sort_values(["molecule_id", "neural_score"], ascending=[True, False], kind="stable").reset_index(drop=True)
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        y = df["label"].to_numpy(dtype=np.int32)
        groups = df["canon_ik14"].to_numpy()
        mids = df["molecule_id"].to_numpy()

        # Graded relevance for LambdaMART: gold=3, same-formula hard negative=1 if high neural, else 0
        # Keep binary label for evaluation, graded target for YetiRank/LambdaRank
        rel = np.where(y == 1, 3, np.where(df["same_formula_as_neural_top1"].to_numpy() > 0.5, 1, 0)).astype(np.int32)
        # Actually binary 0/1 relevance avoids pushing same-formula negatives above non-formula candidates artificially;
        # hard negatives are already in the candidate set and compete directly in listwise loss!
        rel = y

        gkf = GroupKFold(n_splits=n_splits)
        oof_lgb = np.zeros(len(df), dtype=np.float64)
        oof_cb = np.zeros(len(df), dtype=np.float64)
        self.lgb_models, self.cb_models = [], []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
            tr_mids = mids[tr_idx]
            va_mids = mids[va_idx]
            tr_grp = _group_sizes(tr_mids)
            va_grp = _group_sizes(va_mids)

            ranker = lgb.LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                ndcg_eval_at=[1, 5, 25],
                n_estimators=350,
                learning_rate=0.03,
                num_leaves=23,
                min_child_samples=40,
                subsample=0.85,
                colsample_bytree=0.80,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=seed + fold,
                n_jobs=8,
                verbose=-1,
            )
            ranker.fit(
                X[tr_idx],
                rel[tr_idx],
                group=tr_grp,
                eval_set=[(X[va_idx], rel[va_idx])],
                eval_group=[va_grp],
            )
            oof_lgb[va_idx] = ranker.predict(X[va_idx])
            self.lgb_models.append(ranker)

            tr_pool = Pool(data=X[tr_idx], label=rel[tr_idx].astype(np.float32), group_id=tr_mids)
            va_pool = Pool(data=X[va_idx], label=rel[va_idx].astype(np.float32), group_id=va_mids)
            cb = CatBoostRanker(
                iterations=450,
                learning_rate=0.04,
                depth=6,
                loss_function="YetiRank",
                l2_leaf_reg=5.0,
                random_seed=seed + fold,
                thread_count=8,
                verbose=0,
            )
            cb.fit(tr_pool, eval_set=va_pool, use_best_model=True)
            oof_cb[va_idx] = cb.predict(va_pool)
            self.cb_models.append(cb)

        oof_blend = 0.5 * _zscore_by_group(oof_lgb, mids) + 0.5 * _zscore_by_group(oof_cb, mids)
        df = df.copy()
        df["oof_score"] = oof_blend

        # Tune calibrated Rank-1 challenger gate strictly on OOF CV
        best_thr, best_mrr, best_stats = self._tune_gate(df, score_col="oof_score")
        self.delta_threshold = best_thr

        return {
            "oof_gate_threshold": best_thr,
            "oof_stats": best_stats,
        }

    def _tune_gate(self, df: pd.DataFrame, score_col: str) -> Tuple[float, float, Dict]:
        best_thr = 0.0
        best_mrr = -1.0
        best_stats: Dict = {}
        for thr in [0.0, 0.15, 0.30, 0.50, 0.75, 1.0, 1.25, 1.50, 2.0, 3.0]:
            st = self.evaluate_gated_df(df, score_col=score_col, delta_thr=thr, lib_lock_thr=self.lib_lock_threshold)
            if st["gated_mrr25"] > best_mrr:
                best_mrr = st["gated_mrr25"]
                best_thr = thr
                best_stats = st
        return best_thr, best_mrr, best_stats

    def predict_scores(self, df: pd.DataFrame) -> np.ndarray:
        X = df[self.feature_cols].to_numpy(dtype=np.float32)
        mids = df["molecule_id"].to_numpy()
        p_lgb = np.mean([m.predict(X) for m in self.lgb_models], axis=0)
        p_cb = np.mean([m.predict(X) for m in self.cb_models], axis=0)
        return (0.5 * _zscore_by_group(p_lgb, mids) + 0.5 * _zscore_by_group(p_cb, mids)).astype(np.float64)

    def evaluate_gated_df(
        self,
        df: pd.DataFrame,
        score_col: str = "v45_score",
        delta_thr: float | None = None,
        lib_lock_thr: float | None = None,
    ) -> Dict:
        if delta_thr is None:
            delta_thr = self.delta_threshold
        if lib_lock_thr is None:
            lib_lock_thr = self.lib_lock_threshold

        inc_mrrs, full_mrrs, gated_mrrs = [], [], []
        inc_r1, full_r1, gated_r1 = [], [], []
        inc_r5, full_r5, gated_r5 = [], [], []
        inc_r25, full_r25, gated_r25 = [], [], []

        promotions_correct = 0
        demotions_incorrect = 0
        rank1_swaps = 0
        lib_locked = 0

        for mid, g in df.groupby("molecule_id", sort=False):
            # Incumbent order: neural_score descending (breaking ties by abs_ppm_cal_minus1_5)
            inc_ord = np.lexsort((g["abs_ppm_cal_minus1_5"].to_numpy(), -g["neural_score"].to_numpy()))
            inc_labels = g["label"].to_numpy()[inc_ord]
            inc_iks = g["canon_ik14"].to_numpy()[inc_ord]

            # Full challenger order
            chal_scores = g[score_col].to_numpy()
            full_ord = np.argsort(-chal_scores, kind="stable")
            full_labels = g["label"].to_numpy()[full_ord]
            full_iks = g["canon_ik14"].to_numpy()[full_ord]

            # Gated challenger Rank-1 decision
            inc_top_idx = int(inc_ord[0])
            chal_top_idx = int(full_ord[0])
            inc_lib_p = float(g["v44_lib_prob"].to_numpy()[inc_top_idx])
            margin = float(chal_scores[chal_top_idx] - chal_scores[inc_top_idx])

            if inc_lib_p >= lib_lock_thr:
                lib_locked += 1
                gated_ord = inc_ord
            elif chal_top_idx != inc_top_idx and margin >= delta_thr:
                rank1_swaps += 1
                # Place chal_top_idx at rank 1, keep rest ordered by full challenger
                rest = [i for i in full_ord if i != chal_top_idx]
                gated_ord = np.array([chal_top_idx] + rest, dtype=np.int64)
                if g["label"].to_numpy()[chal_top_idx] == 1 and inc_labels[0] == 0:
                    promotions_correct += 1
                elif g["label"].to_numpy()[inc_top_idx] == 1 and g["label"].to_numpy()[chal_top_idx] == 0:
                    demotions_incorrect += 1
            else:
                # Keep incumbent Rank 1, rerank ranks 2..N by full challenger
                rest = [i for i in full_ord if i != inc_top_idx]
                gated_ord = np.array([inc_top_idx] + rest, dtype=np.int64)

            gated_labels = g["label"].to_numpy()[gated_ord]

            def _metrics(lbls: np.ndarray):
                pos = np.where(lbls[:25] == 1)[0]
                mrr = 1.0 / (pos[0] + 1) if len(pos) > 0 else 0.0
                return mrr, int(lbls[0] == 1), int(lbls[:5].sum() > 0), int(len(pos) > 0)

            im, ir1, ir5, ir25 = _metrics(inc_labels)
            fm, fr1, fr5, fr25 = _metrics(full_labels)
            gm, gr1, gr5, gr25 = _metrics(gated_labels)

            inc_mrrs.append(im); inc_r1.append(ir1); inc_r5.append(ir5); inc_r25.append(ir25)
            full_mrrs.append(fm); full_r1.append(fr1); full_r5.append(fr5); full_r25.append(fr25)
            gated_mrrs.append(gm); gated_r1.append(gr1); gated_r5.append(gr5); gated_r25.append(gr25)

        def _boot_ci(a: List[float], b: List[float]) -> Dict[str, float]:
            d = np.asarray(b, dtype=np.float64) - np.asarray(a, dtype=np.float64)
            rng = np.random.default_rng(42)
            boot = rng.choice(d, size=(2000, len(d)), replace=True).mean(axis=1)
            return {
                "mean_delta": round(float(d.mean()), 4),
                "ci_low": round(float(np.percentile(boot, 2.5)), 4),
                "ci_high": round(float(np.percentile(boot, 97.5)), 4),
                "p_win": round(float((boot > 0).mean()), 4),
            }

        return {
            "n_queries": len(inc_mrrs),
            "incumbent_mrr25": round(float(np.mean(inc_mrrs)), 4),
            "incumbent_r1": round(float(np.mean(inc_r1)), 4),
            "incumbent_r5": round(float(np.mean(inc_r5)), 4),
            "incumbent_r25": round(float(np.mean(inc_r25)), 4),
            "full_rerank_mrr25": round(float(np.mean(full_mrrs)), 4),
            "full_rerank_r1": round(float(np.mean(full_r1)), 4),
            "full_rerank_r5": round(float(np.mean(full_r5)), 4),
            "full_rerank_r25": round(float(np.mean(full_r25)), 4),
            "gated_mrr25": round(float(np.mean(gated_mrrs)), 4),
            "gated_r1": round(float(np.mean(gated_r1)), 4),
            "gated_r5": round(float(np.mean(gated_r5)), 4),
            "gated_r25": round(float(np.mean(gated_r25)), 4),
            "rank1_swaps": rank1_swaps,
            "lib_locked": lib_locked,
            "promotions_correct": promotions_correct,
            "demotions_incorrect": demotions_incorrect,
            "ci_full_vs_incumbent": _boot_ci(inc_mrrs, full_mrrs),
            "ci_gated_vs_incumbent": _boot_ci(inc_mrrs, gated_mrrs),
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "Rank1ChallengerEnsemble":
        with open(path, "rb") as f:
            return pickle.load(f)
