"""
chemistry/adducts.py
Adduct mass offset table and neutral mass calculation.

Test set uses exactly these 10 adducts (from official data page):
    [M+H]+, [M+NH4]+, [M-H2O+H]+, [M-2H2O+H]+, [M+Na]+, [M+K]+,
    [M-H]-, [M-H2O-H]-, [M+CH2O2-H]-, [M+Cl]-

The training set contains many more adducts — only the 10 above will appear
in the hidden test set.

Collision energy note (from official data page):
    NCE → eV conversion: eV ≈ NCE × precursor_mz / 500 × charge_factor
    This is approximate; true conversion depends on instrument tuning.
    Use collision_energy_ev from the parquet — it is already converted.
"""

from typing import Optional


# Adduct mass offsets (to subtract from precursor_mz * |charge| to get neutral mass)
# neutral_mass = precursor_mz * |charge| - offset
ADDUCT_OFFSETS: dict[str, float] = {
    # ── Positive mode ─────────────────────────────────────────────────────────
    "[M+H]+":           1.007276,       # TEST SET ✓
    "[M+NH4]+":         18.034164,      # TEST SET ✓
    "[M+Na]+":          22.989218,      # TEST SET ✓
    "[M+K]+":           38.963158,      # TEST SET ✓
    "[M-H2O+H]+":       1.007276 - 18.010565,    # TEST SET ✓
    "[M-2H2O+H]+":      1.007276 - 2 * 18.010565, # TEST SET ✓
    # Train-only positive adducts (not in test set):
    "[M+2H]2+":         2 * 1.007276,
    "[M+H+Na]2+":       1.007276 + 22.989218,
    "[M-NH3+H]+":       1.007276 - 17.026549,
    "[M+Li]+":          7.015460 - 1.007276,      # Li mass - H mass + H offset
    # ── Negative mode ─────────────────────────────────────────────────────────
    "[M-H]-":          -1.007276,       # TEST SET ✓
    "[M+Cl]-":          34.969402,      # TEST SET ✓
    "[M+CH2O2-H]-":     44.997655,      # TEST SET ✓  (formate adduct)
    "[M-H2O-H]-":      -1.007276 - 18.010565,    # TEST SET ✓
    # Train-only negative adducts (not in test set):
    "[M+HCOO]-":        44.997655,      # synonym for [M+CH2O2-H]-
    "[M+CH3COO]-":      59.013305,
    "[M+Na-2H]-":       22.989218 - 2 * 1.007276,
    "[M+K-2H]-":        38.963158 - 2 * 1.007276,
    "[M-2H]2-":         2 * (-1.007276),
}

# The exact 10 adducts that appear in the hidden test set (from official data page).
# Use this to validate test.parquet rows and flag unexpected adducts.
TEST_SET_ADDUCTS: frozenset[str] = frozenset({
    "[M+H]+", "[M+NH4]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M+Na]+", "[M+K]+",
    "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+Cl]-",
})

ADDUCT_CHARGES: dict[str, int] = {
    "[M+H]+":       1,
    "[M+NH4]+":     1,
    "[M+Na]+":      1,
    "[M+K]+":       1,
    "[M+2H]2+":     2,
    "[M+H+Na]2+":   2,
    "[M-H2O+H]+":   1,
    "[M-2H2O+H]+":  1,
    "[M-NH3+H]+":   1,
    "[M+Li]+":      1,
    "[M-H]-":       1,
    "[M+Cl]-":      1,
    "[M+HCOO]-":    1,
    "[M+CH2O2-H]-": 1,
    "[M+CH3COO]-":  1,
    "[M-H2O-H]-":   1,
    "[M+Na-2H]-":   1,
    "[M+K-2H]-":    1,
    "[M-2H]2-":     2,
}


def neutral_mass(precursor_mz: float, adduct: str) -> Optional[float]:
    """
    Compute the neutral monoisotopic mass from a precursor m/z and adduct.

    neutral_mass = precursor_mz * charge - adduct_offset

    Args:
        precursor_mz: Observed precursor m/z
        adduct: Adduct string, e.g. "[M+H]+"

    Returns:
        Neutral mass in Da, or None if adduct is unknown
    """
    if adduct not in ADDUCT_OFFSETS:
        return None
    charge = ADDUCT_CHARGES.get(adduct, 1)
    offset = ADDUCT_OFFSETS[adduct]
    return precursor_mz * charge - offset


def mass_to_mz(neutral_mass_da: float, adduct: str) -> Optional[float]:
    """
    Convert neutral mass back to expected precursor m/z for a given adduct.
    Inverse of neutral_mass().
    """
    if adduct not in ADDUCT_OFFSETS:
        return None
    charge = ADDUCT_CHARGES.get(adduct, 1)
    offset = ADDUCT_OFFSETS[adduct]
    return (neutral_mass_da + offset) / charge


def mass_error_ppm(observed_mz: float, theoretical_mz: float) -> float:
    """Compute mass error in parts per million."""
    return abs(observed_mz - theoretical_mz) / theoretical_mz * 1e6


def filter_candidates_by_mass(
    candidate_masses: list[float],
    query_neutral_mass: float,
    tolerance_ppm: float = 10.0,
) -> list[bool]:
    """
    Return boolean mask for candidates within mass tolerance.

    Args:
        candidate_masses: List of candidate exact masses (Da)
        query_neutral_mass: Target neutral mass (Da)
        tolerance_ppm: Mass tolerance in ppm

    Returns:
        List of booleans, True = within tolerance
    """
    tol_da = query_neutral_mass * tolerance_ppm / 1e6
    return [abs(m - query_neutral_mass) <= tol_da for m in candidate_masses]


def nce_to_ev(nce: float, precursor_mz: float, charge: int = 1) -> float:
    """
    Convert Normalised Collision Energy (NCE) to electron volts (eV).

    Formula from official data page:
        eV ≈ NCE × precursor_mz / 500 × charge_factor

    This is approximate — the true conversion depends on instrument tuning.
    The MSnLib/pluskal_ms2 library uses NCE; the test set uses eV.
    Use collision_energy_ev from the parquet (already converted) where possible.

    Args:
        nce: Normalised collision energy (Thermo percentage scale)
        precursor_mz: Precursor m/z of the spectrum
        charge: Charge state (usually 1)

    Returns:
        Estimated collision energy in eV
    """
    charge_factor = charge ** 0.5  # Thermo convention approximation
    return nce * precursor_mz / 500.0 * charge_factor


def validate_test_adduct(adduct: str) -> bool:
    """
    Return True if adduct is one of the 10 valid test-set adducts.
    Log a warning if an unexpected adduct appears in test.parquet.
    """
    return adduct in TEST_SET_ADDUCTS


# Sanity test
if __name__ == "__main__":
    # Caffeine: [M+H]+ = 195.0877, neutral = 194.0804
    mz = 195.0877
    adduct = "[M+H]+"
    nm = neutral_mass(mz, adduct)
    print(f"Caffeine precursor: {mz}, neutral mass: {nm:.4f} (expected ~194.0804)")

    # Check all 10 test adducts are in the offset table
    missing = TEST_SET_ADDUCTS - set(ADDUCT_OFFSETS.keys())
    if missing:
        print(f"WARNING: Missing test adducts from ADDUCT_OFFSETS: {missing}")
    else:
        print(f"All {len(TEST_SET_ADDUCTS)} test-set adducts have offsets ✓")
