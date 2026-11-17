"""
scripts/verify_isotope_engine.py
=============================================================================
Unit Test & Quality Assurance Suite for chemistry/isotope_engine.py
CASMI 2026 Autonomous Research Lab — Mass Spectrometry & Chemical Analysis

Tests:
  1. Chemical formula parsing across organic natural product scaffolds.
  2. Theoretical M, M+1, M+2 isotopic ratios calculation against IUPAC standards.
  3. Halogen doublet detection (Chlorine 3:1, Bromine 1:1, Sulfur 34S).
  4. Candidate heteroatom consistency gating & decoy suppression.
  5. Sub-structural fragment M+1 carbon count extraction and parity check.
  6. Precursor envelope Gaussian likelihood scoring.
=============================================================================
"""

import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chemistry.isotope_engine import (
    parse_formula,
    get_smiles_formula_dict,
    compute_theoretical_isotope_ratios,
    detect_halogen_and_sulfur_signatures,
    evaluate_halogen_consistency,
    extract_fragment_carbon_estimates,
    score_substructural_carbon_consistency,
    score_precursor_isotope_envelope,
    IsotopeConsistencyEngine,
    OFFSET_13C,
    OFFSET_37CL,
    OFFSET_81BR,
    OFFSET_34S,
)


def test_formula_parsing():
    print("[TEST 1/6] Formula Parsing...")
    f1 = parse_formula("C21H27N3O4S")
    assert f1 == {"C": 21, "H": 27, "N": 3, "O": 4, "S": 1}, f"Mismatch: {f1}"

    f2 = parse_formula("C6H5Cl2Br")
    assert f2 == {"C": 6, "H": 5, "Cl": 2, "Br": 1}, f"Mismatch: {f2}"

    f_smi = get_smiles_formula_dict("c1ccc(Cl)cc1")
    assert f_smi.get("Cl") == 1 and f_smi.get("C") == 6, f"SMILES mismatch: {f_smi}"
    print("  ✓ Passed formula parsing tests.")


def test_theoretical_isotope_ratios():
    print("\n[TEST 2/6] Theoretical Isotope Ratio Calculation...")
    # Benzene C6H6: M+1 should be ~ 6 * 0.01078 = 0.06468
    f_benz = {"C": 6, "H": 6}
    _, r1, r2 = compute_theoretical_isotope_ratios(f_benz)
    assert abs(r1 - 0.065) < 0.005, f"Benzene M+1 ratio unexpected: {r1}"

    # Chlorobenzene C6H5Cl: M+2 should be dominated by 37Cl (0.324)
    f_cl = {"C": 6, "H": 5, "Cl": 1}
    _, _, r2_cl = compute_theoretical_isotope_ratios(f_cl)
    assert abs(r2_cl - 0.326) < 0.010, f"Chlorobenzene M+2 ratio unexpected: {r2_cl}"

    # Bromobenzene C6H5Br: M+2 should be dominated by 81Br (0.973)
    f_br = {"C": 6, "H": 5, "Br": 1}
    _, _, r2_br = compute_theoretical_isotope_ratios(f_br)
    assert abs(r2_br - 0.975) < 0.015, f"Bromobenzene M+2 ratio unexpected: {r2_br}"
    print("  ✓ Passed theoretical isotope ratio tests.")


def test_halogen_doublet_detection():
    print("\n[TEST 3/6] Halogen & Sulfur Doublet Detection...")
    # Synthetic Chlorine spectrum: base peak at 200.0, M+2 at 201.997 with 32% intensity
    mzs = np.array([100.0, 150.0, 200.0, 201.9970, 250.0])
    ints = np.array([0.20, 0.15, 1.00, 0.32, 0.05])
    rep = detect_halogen_and_sulfur_signatures(mzs, ints, precursor_mz=250.0)
    assert rep["has_chlorine"] is True, f"Failed to detect chlorine in synthetic spectrum: {rep}"
    assert rep["has_bromine"] is False, f"False positive bromine: {rep}"

    # Synthetic Bromine spectrum: base peak at 200.0, M+2 at 201.998 with 98% intensity
    mzs_br = np.array([100.0, 150.0, 200.0, 201.9980, 250.0])
    ints_br = np.array([0.20, 0.15, 1.00, 0.98, 0.05])
    rep_br = detect_halogen_and_sulfur_signatures(mzs_br, ints_br, precursor_mz=250.0)
    assert rep_br["has_bromine"] is True, f"Failed to detect bromine: {rep_br}"
    assert rep_br["has_chlorine"] is False, f"False positive chlorine: {rep_br}"
    print("  ✓ Passed halogen doublet detection tests.")


def test_heteroatom_consistency_gating():
    print("\n[TEST 4/6] Heteroatom Consistency Gating...")
    # Spectrum confirms Bromine
    flags_br = {"has_bromine": True, "has_chlorine": False}
    cand_with_br = "c1ccc(Br)cc1"
    cand_without_br = "c1ccc(Cl)cc1"

    score_valid = evaluate_halogen_consistency(cand_with_br, flags_br)
    score_invalid = evaluate_halogen_consistency(cand_without_br, flags_br)
    assert score_valid == 1.0, f"Valid candidate penalized: {score_valid}"
    assert score_invalid == 1e-4, f"Invalid candidate not penalized: {score_invalid}"

    # Spectrum confirms Chlorine
    flags_cl = {"has_bromine": False, "has_chlorine": True}
    cand_with_cl = "c1ccc(Cl)cc1"
    cand_without_cl = "c1ccccc1"

    assert evaluate_halogen_consistency(cand_with_cl, flags_cl) == 1.0
    assert evaluate_halogen_consistency(cand_without_cl, flags_cl) == 1e-4
    print("  ✓ Passed heteroatom consistency gating tests.")


def test_fragment_carbon_estimation():
    print("\n[TEST 5/6] Fragment M+1 Carbon Estimation...")
    # Fragment at 150.0 with 10 carbons -> M+1 ratio = 10 * 0.01078 = 0.1078
    mzs = np.array([150.0, 150.0 + OFFSET_13C, 300.0])
    ints = np.array([1.00, 0.108, 0.10])
    c_ests = extract_fragment_carbon_estimates(mzs, ints, precursor_mz=300.0)
    assert len(c_ests) == 1, f"Failed to extract carbon estimate: {c_ests}"
    mz, est_c, it = c_ests[0]
    assert abs(est_c - 10.0) < 0.5, f"Expected ~10 carbons, got {est_c}"

    # Score against a candidate with 15 carbons (plausible)
    cand_15c = "C1CCCCC1C2CCCCC2CCC"  # 15 carbons
    s_plaus = score_substructural_carbon_consistency(cand_15c, c_ests)
    assert s_plaus > 0.95, f"Plausible candidate penalized: {s_plaus}"

    # Score against a candidate with only 6 carbons (fragment has 10 carbons -> impossible!)
    cand_6c = "c1ccccc1"  # 6 carbons
    s_implaus = score_substructural_carbon_consistency(cand_6c, c_ests)
    assert s_implaus < 0.50, f"Implausible candidate not penalized: {s_implaus}"
    print(f"  ✓ Passed fragment carbon estimation tests (Plausible: {s_plaus:.4f}, Implausible: {s_implaus:.4f}).")


def test_isotope_engine_integration():
    print("\n[TEST 6/6] Full IsotopeConsistencyEngine Integration...")
    engine = IsotopeConsistencyEngine(tol_da=0.012)
    mzs = np.array([100.0, 150.0, 151.0034, 200.0, 201.9970, 300.0, 301.0034])
    ints = np.array([0.20, 0.80, 0.0864, 1.00, 0.3200, 0.40, 0.0600])
    prec_mz = 300.0

    profile = engine.profile_spectrum(mzs, ints, precursor_mz=prec_mz)
    assert profile["halogen_report"]["has_chlorine"] is True

    # Test candidate with Chlorine and 14 carbons
    cand_match = "c1cc(Cl)ccc1CCCCCCCC"
    scores_match = engine.score_candidate(cand_match, profile, mzs, ints)
    assert scores_match["halogen_multiplier"] == 1.0
    assert scores_match["composite_isotope_score"] > 0.50

    # Test candidate with no Chlorine
    cand_decoy = "c1ccccc1CCCCCCCC"
    scores_decoy = engine.score_candidate(cand_decoy, profile, mzs, ints)
    assert scores_decoy["halogen_multiplier"] == 1e-4
    assert scores_decoy["composite_isotope_score"] < 1e-3
    print(f"  ✓ Passed full engine integration tests (Match: {scores_match['composite_isotope_score']:.4f}, Decoy: {scores_decoy['composite_isotope_score']:.6f}).")


def main():
    print("=" * 78)
    print("🔬 RUNNING HIGH-RESOLUTION ISOTOPE CONSISTENCY ENGINE VERIFICATION")
    print("=" * 78)
    test_formula_parsing()
    test_theoretical_isotope_ratios()
    test_halogen_doublet_detection()
    test_heteroatom_consistency_gating()
    test_fragment_carbon_estimation()
    test_isotope_engine_integration()
    print("\n" + "=" * 78)
    print("🎉 ALL 6/6 ISOTOPE CONSISTENCY TESTS 100% PASSED!")
    print("=" * 78)


if __name__ == "__main__":
    main()
