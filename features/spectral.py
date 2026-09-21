"""
features/spectral.py
Spectral similarity features for the learning-to-rank model.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict


# ─── Cosine Similarity ────────────────────────────────────────────────────────

def cosine_similarity(
    mzs1: np.ndarray,
    intensities1: np.ndarray,
    mzs2: np.ndarray,
    intensities2: np.ndarray,
    mz_tolerance: float = 0.05,
) -> float:
    """
    Standard cosine similarity between two spectra.

    Peaks are matched within mz_tolerance (Da). Unmatched peaks contribute 0.
    Scores in [0, 1].
    """
    if len(mzs1) == 0 or len(mzs2) == 0:
        return 0.0

    # Build matched intensity vectors
    matched1, matched2 = [], []
    unmatched1, unmatched2 = list(intensities1), list(intensities2)
    used2 = [False] * len(mzs2)

    for i, mz in enumerate(mzs1):
        # Find best match in spectrum 2 within tolerance
        diffs = np.abs(mzs2 - mz)
        valid = (diffs <= mz_tolerance) & (~np.array(used2))
        if valid.any():
            j = np.argmin(np.where(valid, diffs, np.inf))
            matched1.append(intensities1[i])
            matched2.append(intensities2[j])
            used2[j] = True

    if not matched1:
        return 0.0

    m1 = np.array(matched1)
    m2 = np.array(matched2)

    dot = np.dot(m1, m2)
    norm = np.linalg.norm(intensities1) * np.linalg.norm(intensities2)
    return float(dot / norm) if norm > 0 else 0.0


def modified_cosine_similarity(
    mzs1: np.ndarray,
    intensities1: np.ndarray,
    precursor_mz1: float,
    mzs2: np.ndarray,
    intensities2: np.ndarray,
    precursor_mz2: float,
    mz_tolerance: float = 0.05,
) -> float:
    """
    Modified cosine similarity (Watrous et al.).

    Also considers shifted matches: mz1 - mz2 ≈ precursor_mz1 - precursor_mz2
    This handles cases where the same fragmentation occurs at different precursor masses.
    """
    shift = precursor_mz1 - precursor_mz2

    # Standard matches
    standard_score = cosine_similarity(mzs1, intensities1, mzs2, intensities2, mz_tolerance)

    # Shifted matches: try matching mz1[i] with mzs2[j] ± shift
    shifted_mzs2 = mzs2 + shift
    shifted_score = cosine_similarity(mzs1, intensities1, shifted_mzs2, intensities2, mz_tolerance)

    return max(standard_score, shifted_score)


# ─── Fragment Matching ────────────────────────────────────────────────────────

def fragment_match_features(
    query_mzs: np.ndarray,
    query_intensities: np.ndarray,
    candidate_fragment_mzs: List[float],
    mz_tolerance: float = 0.05,
) -> Dict[str, float]:
    """
    Score how many predicted fragments of a candidate match the observed spectrum.

    Args:
        query_mzs: Observed spectrum m/z values
        query_intensities: Observed spectrum intensities (normalized)
        candidate_fragment_mzs: Theoretically predicted fragment m/z values for candidate
        mz_tolerance: Matching tolerance in Da

    Returns:
        Dictionary of fragment match features
    """
    if len(candidate_fragment_mzs) == 0 or len(query_mzs) == 0:
        return {
            "n_matched_fragments": 0,
            "fraction_fragments_matched": 0.0,
            "fraction_intensity_explained": 0.0,
            "max_matched_intensity": 0.0,
        }

    cand_mzs = np.array(candidate_fragment_mzs)
    n_matched = 0
    total_explained_intensity = 0.0
    max_matched_intensity = 0.0

    for cand_mz in cand_mzs:
        diffs = np.abs(query_mzs - cand_mz)
        if diffs.min() <= mz_tolerance:
            best_idx = np.argmin(diffs)
            n_matched += 1
            total_explained_intensity += query_intensities[best_idx]
            max_matched_intensity = max(max_matched_intensity, query_intensities[best_idx])

    return {
        "n_matched_fragments": n_matched,
        "fraction_fragments_matched": n_matched / len(cand_mzs),
        "fraction_intensity_explained": float(total_explained_intensity / query_intensities.sum())
            if query_intensities.sum() > 0 else 0.0,
        "max_matched_intensity": max_matched_intensity,
    }


# ─── Neutral Loss Features ────────────────────────────────────────────────────

COMMON_NEUTRAL_LOSSES = {
    "H2O":          18.010565,
    "NH3":          17.026549,
    "CO":           27.994915,
    "CO2":          43.989830,
    "CH2O":         30.010565,
    "HF":           19.998840,
    "CH4":          16.031300,
    "C2H2O":        42.010565,
    "hexose":       162.052824,
    "deoxyhexose":  146.057909,
    "glucuronide":  176.032088,
    "sulfate":      79.956816,
    "phosphate":    97.976895,
    "acetyl":       42.010565,
    "methyl":       14.015650,
}


def neutral_loss_features(
    query_mzs: np.ndarray,
    precursor_mz: float,
    mz_tolerance: float = 0.02,
) -> Dict[str, float]:
    """
    Extract neutral loss features from a spectrum.

    Computes which common neutral losses are present: precursor_mz - fragment_mz.

    Returns:
        Dict of {loss_name: bool_as_float, ...} + aggregate features
    """
    features = {}
    observed_losses = precursor_mz - query_mzs
    observed_losses = observed_losses[observed_losses > 0]

    for loss_name, loss_mass in COMMON_NEUTRAL_LOSSES.items():
        matches = np.abs(observed_losses - loss_mass) <= mz_tolerance
        features[f"has_loss_{loss_name}"] = float(matches.any())

    features["n_neutral_losses_matched"] = sum(
        features[f"has_loss_{k}"] for k in COMMON_NEUTRAL_LOSSES
    )
    features["has_sugar_loss"] = max(
        features.get("has_loss_hexose", 0),
        features.get("has_loss_deoxyhexose", 0),
        features.get("has_loss_glucuronide", 0),
    )

    return features
