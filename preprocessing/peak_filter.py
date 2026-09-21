"""
preprocessing/peak_filter.py
Peak filtering and normalization for MS/MS spectra.

Test-set preprocessing applied by the competition (use as defaults):
    - Spectra with base_peak_intensity < 1,000 raw counts → dropped (noise)
    - Peaks above precursor_mz + 2 Da → removed  ← NOT +5 Da
    - Spectra with precursor mass inconsistent with structure/adduct → dropped
    No other cleaning applied by the organizers; everything else is our choice.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class PeakFilterConfig:
    """Configuration for peak filtering pipeline."""
    min_intensity_fraction: float = 0.005   # 0.5% of base peak
    max_peaks: int = 128
    min_peaks: int = 3
    # Competition test set removes peaks above precursor + 2 Da.
    # Use 2.0 to match; allow slight extra headroom for isotopes (+2 = M+1 + M+2 safe).
    precursor_mz_buffer: float = 2.0        # Da above precursor to keep (test set: +2 Da)
    remove_isotopes: bool = True
    isotope_tolerance: float = 0.02         # Da tolerance for isotope removal
    intensity_transform: str = "sqrt"       # "raw", "sqrt", "log1p"
    min_base_peak_raw_counts: float = 0.0   # Drop spectrum if base_peak_intensity < this.
                                             # Competition used 1000; train data is often pre-
                                             # normalised (null base_peak_intensity) so default=0.


def filter_spectrum(
    mzs: np.ndarray,
    intensities: np.ndarray,
    precursor_mz: float,
    config: PeakFilterConfig = PeakFilterConfig(),
    base_peak_raw_intensity: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply the full peak filtering pipeline to a single spectrum.

    Args:
        mzs: Array of m/z values
        intensities: Array of intensity values (pre-normalized so max=1.0)
        precursor_mz: Precursor ion m/z
        config: Filtering configuration
        base_peak_raw_intensity: Raw count of the base peak before normalisation
            (= base_peak_intensity column in the parquet files). When provided and
            below config.min_base_peak_raw_counts the spectrum is rejected entirely.
            The test set dropped spectra with base_peak_intensity < 1,000.

    Returns:
        Filtered (mzs, intensities) — both empty arrays if spectrum is rejected.
    """
    if len(mzs) == 0:
        return np.array([]), np.array([])

    # Pre-filter: low-quality spectrum guard (mirrors competition's own cleaning)
    if (
        base_peak_raw_intensity is not None
        and config.min_base_peak_raw_counts > 0
        and base_peak_raw_intensity < config.min_base_peak_raw_counts
    ):
        return np.array([]), np.array([])

    mzs = np.asarray(mzs, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)

    # Step 1: Sort by m/z
    order = np.argsort(mzs)
    mzs, intensities = mzs[order], intensities[order]

    # Step 2: Normalize to max = 1.0
    max_intensity = intensities.max()
    if max_intensity > 0:
        intensities = intensities / max_intensity

    # Step 3: Remove peaks near/above precursor
    keep = mzs <= (precursor_mz + config.precursor_mz_buffer)
    mzs, intensities = mzs[keep], intensities[keep]

    # Step 4: Relative intensity threshold
    keep = intensities >= config.min_intensity_fraction
    mzs, intensities = mzs[keep], intensities[keep]

    # Step 5: Remove isotope satellites
    if config.remove_isotopes:
        mzs, intensities = _remove_isotopes(mzs, intensities, config.isotope_tolerance)

    # Step 6: Keep top-K peaks by intensity
    if len(mzs) > config.max_peaks:
        top_k = np.argsort(intensities)[-config.max_peaks:]
        top_k = np.sort(top_k)  # re-sort by m/z
        mzs, intensities = mzs[top_k], intensities[top_k]

    # Step 7: Check minimum peaks
    if len(mzs) < config.min_peaks:
        return np.array([]), np.array([])

    # Step 8: Intensity transform
    intensities = _apply_transform(intensities, config.intensity_transform)

    # Step 9: Re-normalize after transform
    max_i = intensities.max()
    if max_i > 0:
        intensities = intensities / max_i

    return mzs, intensities


def _remove_isotopes(
    mzs: np.ndarray,
    intensities: np.ndarray,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Remove 13C isotope satellite peaks."""
    ISOTOPE_DELTAS = [1.003355, 2.006710]  # 13C, 2x13C
    keep = np.ones(len(mzs), dtype=bool)

    for i in range(len(mzs)):
        if not keep[i]:
            continue
        for delta in ISOTOPE_DELTAS:
            target = mzs[i] + delta
            close = np.abs(mzs - target) < tol
            # Only remove the satellite if it's weaker than the monoisotopic peak
            for j in np.where(close)[0]:
                if j != i and intensities[j] < intensities[i]:
                    keep[j] = False

    return mzs[keep], intensities[keep]


def _apply_transform(intensities: np.ndarray, transform: str) -> np.ndarray:
    """Apply intensity transformation."""
    if transform == "sqrt":
        return np.sqrt(intensities)
    elif transform == "log1p":
        return np.log1p(intensities)
    elif transform == "raw":
        return intensities
    else:
        raise ValueError(f"Unknown intensity transform: {transform}")


def batch_filter_spectra(
    mzs_list: list,
    intensities_list: list,
    precursor_mzs: list,
    config: PeakFilterConfig = PeakFilterConfig(),
    base_peak_raw_intensities: Optional[list] = None,
) -> list:
    """Batch process multiple spectra."""
    results = []
    for i, (mzs, intensities, precursor_mz) in enumerate(
        zip(mzs_list, intensities_list, precursor_mzs)
    ):
        raw_bp = base_peak_raw_intensities[i] if base_peak_raw_intensities else None
        filtered_mzs, filtered_intensities = filter_spectrum(
            np.asarray(mzs), np.asarray(intensities), precursor_mz, config, raw_bp
        )
        results.append((filtered_mzs, filtered_intensities))
    return results
