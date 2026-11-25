"""
ranking/v44_library_confidence.py
=============================================================================
CASMI 2026 — V44: Candidate-Specific Library Confidence Model

Replaces query-level `LIB_MAX` (Bug 4) with candidate-specific measured
spectral evidence:
  - Modified cosine similarity (direct + precursor-shifted fragments, 0.02 Da)
  - Spectral entropy similarity (Shannon entropy overlap, 0.02 Da)
  - Matched peak count & fraction of query intensity explained
  - Candidate mass accuracy (ppm error under -1.50 ppm timsTOF calibration)
  - Per-query margin vs best competing different-molecule reference hit

Trains a molecule-grouped calibrated LightGBM classifier predicting
P(candidate is true structure | candidate spectral library evidence).
=============================================================================
"""

from __future__ import annotations

import math
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.calibration import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

LIB_FEATURE_NAMES = [
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
]


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if len(p) == 0:
        return 0.0
    return float(-np.sum(p * np.log(p)))


def spectral_entropy_similarity(
    q_mz: np.ndarray,
    q_p: np.ndarray,
    r_mz: np.ndarray,
    r_p: np.ndarray,
    tol_da: float = 0.02,
) -> float:
    """Normalized spectral entropy similarity in [0, 1] (Li et al., 2021)."""
    if len(q_mz) == 0 or len(r_mz) == 0:
        return 0.0
    # Match peaks within tol_da by merging sorted peak lists
    sa = _entropy(q_p)
    sb = _entropy(r_p)
    # Two-pointer merge on sorted q_mz, r_mz
    i, j = 0, 0
    nq, nr = len(q_mz), len(r_mz)
    sab = 0.0
    while i < nq and j < nr:
        diff = q_mz[i] - r_mz[j]
        if abs(diff) <= tol_da:
            p_mix = 0.5 * (q_p[i] + r_p[j])
            if p_mix > 0:
                sab -= p_mix * math.log(p_mix)
            i += 1
            j += 1
        elif diff < 0:
            p_mix = 0.5 * q_p[i]
            if p_mix > 0:
                sab -= p_mix * math.log(p_mix)
            i += 1
        else:
            p_mix = 0.5 * r_p[j]
            if p_mix > 0:
                sab -= p_mix * math.log(p_mix)
            j += 1
    while i < nq:
        p_mix = 0.5 * q_p[i]
        if p_mix > 0:
            sab -= p_mix * math.log(p_mix)
        i += 1
    while j < nr:
        p_mix = 0.5 * r_p[j]
        if p_mix > 0:
            sab -= p_mix * math.log(p_mix)
        j += 1

    sim = 1.0 - (2.0 * sab - sa - sb) / (2.0 * math.log(2.0))
    return float(max(0.0, min(1.0, sim)))


def modified_cosine_metrics(
    q_mz: np.ndarray,
    q_sqrt: np.ndarray,
    q_it_norm: np.ndarray,
    q_prec: float,
    r_mz: np.ndarray,
    r_sqrt: np.ndarray,
    r_prec: float,
    tol_da: float = 0.02,
) -> Tuple[float, int, float]:
    """
    Fast greedy modified cosine (direct + precursor-shifted peaks).
    Assumes q_sqrt and r_sqrt are L2-normalized, and q_it_norm sums to 1.0.
    Returns (mod_cos, shared_peaks, explained_query_intensity).
    """
    if len(q_mz) == 0 or len(r_mz) == 0:
        return 0.0, 0, 0.0

    shift = q_prec - r_prec
    # Direct peak matches via binary search
    cand_pairs = []
    idx_lo = np.searchsorted(r_mz, q_mz - tol_da, side="left")
    idx_hi = np.searchsorted(r_mz, q_mz + tol_da, side="right")
    for qi in range(len(q_mz)):
        for rj in range(idx_lo[qi], idx_hi[qi]):
            cand_pairs.append((float(q_sqrt[qi] * r_sqrt[rj]), qi, rj))

    if abs(shift) > tol_da:
        s_lo = np.searchsorted(r_mz, q_mz - shift - tol_da, side="left")
        s_hi = np.searchsorted(r_mz, q_mz - shift + tol_da, side="right")
        for qi in range(len(q_mz)):
            for rj in range(s_lo[qi], s_hi[qi]):
                cand_pairs.append((float(q_sqrt[qi] * r_sqrt[rj]), qi, rj))

    if not cand_pairs:
        return 0.0, 0, 0.0

    cand_pairs.sort(key=lambda x: -x[0])
    used_q, used_r = set(), set()
    cos_sum = 0.0
    exp_int = 0.0
    for prod, qi, rj in cand_pairs:
        if qi not in used_q and rj not in used_r:
            used_q.add(qi)
            used_r.add(rj)
            cos_sum += prod
            exp_int += float(q_it_norm[qi])

    return float(min(1.0, cos_sum)), len(used_q), float(min(1.0, exp_int))


def prep_spectrum_for_lib(
    mzs: np.ndarray,
    its: np.ndarray,
    prec_mz: float,
    max_peaks: int = 128,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]]:
    """Precompute sorted m/z, L2-norm sqrt intensities, L1-norm intensities, and precursor m/z."""
    m = np.asarray(mzs, dtype=np.float64)
    it = np.asarray(its, dtype=np.float64)
    ok = np.isfinite(m) & np.isfinite(it) & (m > 10.0) & (m <= prec_mz - 1.5) & (it > 0)
    if not ok.any():
        return None
    m, it = m[ok], it[ok]
    if len(m) > max_peaks:
        top = np.argpartition(-it, max_peaks)[:max_peaks]
        m, it = m[top], it[top]
    ord_i = np.argsort(m, kind="mergesort")
    m, it = m[ord_i], it[ord_i]
    sq = np.sqrt(it)
    sq_norm = float(np.linalg.norm(sq))
    if sq_norm <= 1e-12:
        return None
    sq = (sq / sq_norm).astype(np.float32)
    p = (it / (it.sum() + 1e-12)).astype(np.float32)
    return m.astype(np.float32), sq, p, float(prec_mz)


def compute_candidate_lib_features(
    q_prep: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float]],
    cand_iks: np.ndarray,
    cand_ppm_errs: np.ndarray,
    ref_by_ik: Dict[str, List[Tuple[np.ndarray, np.ndarray, np.ndarray, float]]],
    tol_da: float = 0.02,
) -> np.ndarray:
    """
    Compute (n_cands, len(LIB_FEATURE_NAMES)) candidate-specific reference features.
    """
    n = len(cand_iks)
    feats = np.zeros((n, len(LIB_FEATURE_NAMES)), dtype=np.float32)
    feats[:, 6] = np.abs(cand_ppm_errs).astype(np.float32)

    if q_prep is None:
        return feats
    q_mz, q_sq, q_p, q_prec = q_prep

    best_cos = np.zeros(n, dtype=np.float32)
    best_ent = np.zeros(n, dtype=np.float32)
    for ci, ik in enumerate(cand_iks):
        refs = ref_by_ik.get(str(ik))
        if not refs:
            continue
        feats[ci, 0] = 1.0
        feats[ci, 5] = math.log1p(len(refs))
        max_c, max_e, max_pk, max_exp = 0.0, 0.0, 0, 0.0
        for r_mz, r_sq, r_p, r_prec in refs:
            mc, pk, exp_i = modified_cosine_metrics(q_mz, q_sq, q_p, q_prec, r_mz, r_sq, r_prec, tol_da=tol_da)
            es = spectral_entropy_similarity(q_mz, q_p, r_mz, r_p, tol_da=tol_da)
            if mc > max_c:
                max_c = mc
                max_pk = pk
                max_exp = exp_i
            if es > max_e:
                max_e = es
        best_cos[ci] = max_c
        best_ent[ci] = max_e
        feats[ci, 1] = max_c
        feats[ci, 2] = max_e
        feats[ci, 3] = float(max_pk)
        feats[ci, 4] = float(max_exp)

    # Compute per-candidate margin against the best DIFFERENT candidate
    if n >= 2:
        top1_c_idx = int(np.argmax(best_cos))
        top1_c = float(best_cos[top1_c_idx])
        second_c = float(np.partition(best_cos, -2)[-2])

        top1_e_idx = int(np.argmax(best_ent))
        top1_e = float(best_ent[top1_e_idx])
        second_e = float(np.partition(best_ent, -2)[-2])

        for ci in range(n):
            other_c = second_c if ci == top1_c_idx else top1_c
            other_e = second_e if ci == top1_e_idx else top1_e
            feats[ci, 7] = best_cos[ci] - other_c
            feats[ci, 8] = best_ent[ci] - other_e
            feats[ci, 9] = other_c
    else:
        feats[:, 7] = best_cos
        feats[:, 8] = best_ent
        feats[:, 9] = 0.0

    return feats


class LibraryConfidenceModel:
    """Calibrated LightGBM model predicting candidate-specific library hit probability."""

    def __init__(self):
        self.models: List[lgb.LGBMClassifier] = []
        self.calibrator: Optional[IsotonicRegression] = None
        self.feature_names = list(LIB_FEATURE_NAMES)

    def fit_oof(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
        n_splits: int = 5,
        seed: int = 42,
    ) -> Dict[str, float]:
        from sklearn.model_selection import GroupKFold

        gkf = GroupKFold(n_splits=n_splits)
        oof_raw = np.zeros(len(y), dtype=np.float64)
        self.models = []

        for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups)):
            clf = lgb.LGBMClassifier(
                n_estimators=200,
                learning_rate=0.04,
                num_leaves=15,
                min_child_samples=40,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=1.0,
                reg_lambda=5.0,
                random_state=seed + fold,
                n_jobs=4,
                verbose=-1,
            )
            clf.fit(X[tr_idx], y[tr_idx])
            oof_raw[va_idx] = clf.predict_proba(X[va_idx])[:, 1]
            self.models.append(clf)

        self.calibrator = IsotonicRegression(out_of_bounds="clip")
        self.calibrator.fit(oof_raw, y)
        oof_cal = self.calibrator.transform(oof_raw)

        return {
            "oof_auc": round(float(roc_auc_score(y, oof_cal)), 4),
            "oof_brier": round(float(brier_score_loss(y, oof_cal)), 4),
            "oof_logloss": round(float(log_loss(y, np.clip(oof_cal, 1e-6, 1 - 1e-6))), 4),
        }

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self.models:
            raise RuntimeError("Model not trained")
        raw = np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)
        if self.calibrator is not None:
            return np.asarray(self.calibrator.transform(raw), dtype=np.float64)
        return raw

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> "LibraryConfidenceModel":
        with open(path, "rb") as f:
            return pickle.load(f)
