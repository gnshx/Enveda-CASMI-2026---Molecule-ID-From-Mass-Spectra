"""
scripts/verify_chemistry_engine.py
==================================
Unit and Integration Tests for CASMI 2026 Chemistry & Mass Spectrometry Engine.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chemistry.ms_engine import (
    ADDUCT_OFFSETS,
    TEST_ADDUCTS,
    TIMSTOF_OFFSET_PPM,
    compute_neutral_mass,
    timstof_gaussian_mass_prior,
    pool_multice_spectra,
    diagnostic_fragment_weights,
    detect_neutral_losses,
    evaluate_chemical_plausibility,
    RegimeAReferenceEngine
)

def test_adduct_chemistry():
    print("Test 1: Adduct Chemistry & Official Test Adducts...")
    # Verify all 10 official test adducts are covered
    for add in TEST_ADDUCTS:
        assert add in ADDUCT_OFFSETS, f"Missing test adduct: {add}"
    
    # Check Caffeine [M+H]+: mz = 195.08765, true neutral = 194.08038
    nm_uncal = compute_neutral_mass(195.08765, "[M+H]+", calibrate_timstof=False)
    assert abs(nm_uncal - 194.08037) < 0.001, f"Caffeine neutral mass error: {nm_uncal}"
    
    # Check Formate adduct: [M+CH2O2-H]- and [M+HCOO]- equivalence
    nm_form1 = compute_neutral_mass(300.0, "[M+CH2O2-H]-", calibrate_timstof=False)
    nm_form2 = compute_neutral_mass(300.0, "[M+HCOO]-", calibrate_timstof=False)
    assert abs(nm_form1 - nm_form2) < 1e-6, "Formate synonym mismatch"
    assert abs(nm_form1 - (300.0 - 44.998201)) < 1e-6
    
    # Check [M+Na]+
    nm_na = compute_neutral_mass(300.0, "[M+Na]+", calibrate_timstof=False)
    assert abs(nm_na - (300.0 - 22.989218)) < 1e-6
    
    # Check timsTOF -1.5 ppm calibration
    nm_cal = compute_neutral_mass(195.08765, "[M+H]+", calibrate_timstof=True)
    ppm_diff = (nm_cal - nm_uncal) / nm_uncal * 1e6
    assert abs(ppm_diff - TIMSTOF_OFFSET_PPM) < 1e-4, f"Calibration offset error: {ppm_diff}"
    print("  ✓ All 10 test adducts and timsTOF -1.5 ppm calibration verified.")

def test_timstof_prior():
    print("Test 2: timsTOF Bayesian Gaussian Prior...")
    nm_obs = 300.000000
    # Candidate sitting at exact -1.5 ppm should receive maximum score (1.0)
    cand_sweet = nm_obs * (1.0 - 1.5e-6)
    cand_decoy = nm_obs * (1.0 + 8.0e-6) # sitting at +8 ppm
    
    priors = timstof_gaussian_mass_prior(np.array([cand_sweet, cand_decoy]), nm_obs)
    assert abs(priors[0] - 1.0) < 1e-4, f"Sweet spot prior != 1.0: {priors[0]}"
    assert priors[1] < 0.05, f"Decoy (+8 ppm) not sufficiently penalized: {priors[1]}"
    print(f"  ✓ timsTOF sweet spot score: {priors[0]:.4f}, decoy (+8 ppm) score: {priors[1]:.4f}")

def test_multice_pooling():
    print("Test 3: Multi-Collision Energy CID Pooling...")
    s1 = {"mz": np.array([100.001, 200.000]), "it": np.array([0.2, 1.0]), "prec": 250.0, "mode": 1, "adduct": "[M+H]+", "ce": 20.0}
    s2 = {"mz": np.array([80.000, 100.002, 150.000]), "it": np.array([0.8, 0.5, 0.4]), "prec": 250.0, "mode": 1, "adduct": "[M+H]+", "ce": 40.0}
    
    pooled = pool_multice_spectra([s1, s2], tol_da=0.01)
    assert len(pooled) == 1
    p = pooled[0]
    # Expect 4 merged peaks: 80, 100, 150, 200
    assert len(p["mz"]) == 4, f"Expected 4 peaks, got {len(p['mz'])}"
    # 100.001 and 100.002 should be clustered into one centroid
    assert any(abs(mz - 100.0015) < 0.001 for mz in p["mz"])
    print(f"  ✓ Multi-CE pooling merged 5 input peaks into {len(p['mz'])} composite peaks with accurate centroids.")

def test_neutral_loss_and_plausibility():
    print("Test 4: Neutral Loss Detection & Plausibility Gating...")
    prec_mz = 300.000
    # Simulate spectrum with prominent H2O loss: 300.0 - 18.0106 = 281.9894
    mzs = np.array([100.0, 200.0, 281.989, 300.0])
    ints = np.array([0.1, 0.3, 0.8, 1.0])
    
    losses = detect_neutral_losses(mzs, ints, prec_mz)
    assert "h2o" in losses, "Failed to detect prominent H2O neutral loss"
    print(f"  ✓ Detected neutral losses: {losses}")
    
    # Phenol/Alcohol (has OH): should receive boost (>= 1.0)
    smi_alcohol = "OCC1CCCC1"
    score_alc = evaluate_chemical_plausibility(smi_alcohol, losses)
    assert score_alc >= 1.0, f"Alcohol should be boosted, got {score_alc}"
    
    # Ether (no OH): should be penalized (< 0.8)
    smi_ether = "COCC1CCCC1"
    score_eth = evaluate_chemical_plausibility(smi_ether, losses)
    assert score_eth < 0.8, f"Ether without OH should be penalized, got {score_eth}"
    print(f"  ✓ Plausibility score alcohol: {score_alc:.3f} | ether (lacking OH): {score_eth:.3f}")

def main():
    print("=" * 70)
    print("VERIFYING CASMI 2026 CHEMISTRY & MASS SPECTROMETRY ENGINE")
    print("=" * 70)
    test_adduct_chemistry()
    test_timstof_prior()
    test_multice_pooling()
    test_neutral_loss_and_plausibility()
    print("=" * 70)
    print("ALL CHEMISTRY ENGINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
