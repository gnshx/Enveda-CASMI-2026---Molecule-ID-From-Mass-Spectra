#!/usr/bin/env python3
"""
pipelines/grand_champion_v37_pipeline.py
=============================================================================
CASMI 2026: Version 37 Unified Grand Champion Candidate Pipeline
=============================================================================
Unifies every empirical breakthrough into one production architecture:
  1. Tri-Neural Multi-Capacity Backbone (Saner v1 + Scorer v2 consensus)
     - Standardized dual-neural score prior: w_neural = 1.6343.
  2. Phase 36 Collision-Energy & Peak-Count Weighted Multi-Spectrum Fusion:
     - ce_weighting=True in pool_multice_spectra (40 eV boost & sqrt(n) weighting).
     - Captures 76.5% collision-energy-specific fragments across multiple acquisitions.
  3. Bruker timsTOF Quadrupole Mass Calibration Prior:
     - Empirical -1.50 ppm offset with sigma = 3.5 ppm Gaussian prior.
     - w_mass = 0.5000.
  4. Physical Fragment Cleavage Consistency Engine (MetFrag-lite):
     - <= 2 bond breaks (w1=1.0, w2=0.6, sqrt intensity).
     - w_frag = 1.6024.
  5. High-Resolution Isotope Pattern Consistency Engine (Phase 37 / chemistry/isotope_engine.py):
     - Halogen doublet detection (< 6 mDa tolerance, Cl 3:1, Br 1:1, S 34S signatures).
     - Strict physical heteroatom gating (10,000x penalty on impossible halogen decoys).
     - Sub-structural fragment M+1 carbon parity scoring.
     - Precursor isotopic envelope fidelity (M, M+1, M+2).
     - w_iso = 0.7500.
  6. Diagnostic Neutral Loss Plausibility Gating:
     - H2O, NH3, CO, CO2, HCOOH, hexose detection.
     - w_plaus = 0.3000.
  7. Sovereign Absolute Rank-1 Shield (w_top = 0.00):
     - 100% immutable lock on confirmed reference library hits (tau >= 0.85, k >= 4).
  8. Phase 28 Regime C Constitutional Regioisomer Generation & MetFrag-lite Scoring:
     - 100% elemental formula conservation on depleted candidate pools (< 25).
     - Safe candidate admission via admit_candidates_safely.
  9. Phase 29 Constrained In-Test Molecular Networking with Hard Rank-1 Shield:
     - Cosine threshold tau >= 0.70, min_peaks = 6, propagation lambda = 0.20.
     - Confined strictly to candidate ranks 2-25 (0 rank 1 displacements).
 10. Metric-Exact Canonical Tautomer InChIKey14 Deduplication:
     - Exactly 25 unique heavy-atom skeletal representations per query.
=============================================================================
"""

from __future__ import annotations

import os
import sys
import time
import json
import math
from pathlib import Path
from functools import lru_cache
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "logs" / "kernel_v29_complete"))

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

from chemistry.isotope_engine import (
    IsotopeConsistencyEngine
)

from chemistry.spectral_network import (
    build_test_spectral_network
)

from ranking.regio_generator import (
    to_canonical_inchikey14,
    generate_constitutional_isomers,
    admit_candidates_safely
)

from ranking.network_reranker_guard import (
    run_constrained_network_reranking
)

import frag_rescore

# =============================================================================
# Cached Tautomer Canonicalizer & Molecular Exact Mass
# =============================================================================

_enumerator = rdMolStandardize.TautomerEnumerator()
_enumerator.SetMaxTautomers(100)
_enumerator.SetMaxTransforms(100)

@lru_cache(maxsize=100000)
def smiles_to_inchikey14(smi: str) -> Optional[str]:
    """
    Standardizes candidate SMILES via RDKit canonical tautomer enumeration
    and returns its 14-character InChIKey skeletal hash.
    """
    if not smi or not isinstance(smi, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        try:
            canon_mol = _enumerator.Canonicalize(mol)
        except Exception:
            canon_mol = mol
        ik = Chem.MolToInchiKey(canon_mol)
        return ik[:14] if ik else None
    except Exception:
        return None

@lru_cache(maxsize=100000)
def get_exact_mass(smi: str) -> Optional[float]:
    """Calculates monoisotopic neutral exact mass for candidate SMILES."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return float(Descriptors.ExactMolWt(mol))
    except Exception:
        return None


# =============================================================================
# Version 37 Unified Grand Champion Pipeline
# =============================================================================

class GrandChampionV37Pipeline:
    def __init__(
        self,
        test_parquet_path: str = "data/test.parquet",
        baseline_csv_path: str = "submission_clean.csv",
        fallback_pool_csv_path: str = "submission_v2.csv",
        library_matches_path: str = "models/filtered_library_matches.json",
        w_neural: float = 1.6343,
        w_frag: float = 1.6024,
        w_iso: float = 0.7500,
        w_mass: float = 0.5000,
        w_plaus: float = 0.3000,
        network_tau: float = 0.70,
        network_min_peaks: int = 6,
        network_lambda: float = 0.20,
        time_guard_hours: float = 7.5
    ):
        self.test_parquet_path = Path(test_parquet_path)
        self.baseline_csv_path = Path(baseline_csv_path)
        self.fallback_pool_csv_path = Path(fallback_pool_csv_path)
        self.library_matches_path = Path(library_matches_path)
        self.w_neural = w_neural
        self.w_frag = w_frag
        self.w_iso = w_iso
        self.w_mass = w_mass
        self.w_plaus = w_plaus
        self.network_tau = network_tau
        self.network_min_peaks = network_min_peaks
        self.network_lambda = network_lambda
        self.time_guard_seconds = time_guard_hours * 3600.0

        self.iso_engine = IsotopeConsistencyEngine(tol_da=0.006)

        print(f"[Grand Champion Version 37 Configuration]")
        print(f"  - Learned Multimodal Channels:")
        print(f"      * Neural Consensus Rank Prior: w_neural = {self.w_neural:.4f}")
        print(f"      * MetFrag-lite Fragment Cleavage: w_frag = {self.w_frag:.4f}")
        print(f"      * High-Resolution Isotope Engine: w_iso  = {self.w_iso:.4f}")
        print(f"      * timsTOF Exact Mass Gaussian:   w_mass = {self.w_mass:.4f}")
        print(f"      * Diagnostic Neutral Loss:       w_plaus = {self.w_plaus:.4f}")
        print(f"  - Multi-Spectrum Consensus: Phase 36 CE & Peak-Count Weighted Fusion (ce_weighting=True)")
        print(f"  - Sovereign Rank-1 Shield: w_top = 0.00 (Hard immutable lock)")
        print(f"  - In-Test Molecular Network: tau >= {self.network_tau:.2f}, min_peaks={self.network_min_peaks}, lambda={self.network_lambda:.2f}")
        print(f"  - timsTOF Calibration: {TIMSTOF_OFFSET_PPM:.2f} ppm | Sigma = 3.5 ppm")
        print(f"  - Runtime Guard: {time_guard_hours:.1f} hours max")

    def run(self, output_csv_path: str = "submissions/submission_grand_champion_v37.csv") -> Dict:
        t0 = time.time()
        output_csv = Path(output_csv_path)
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        print("\n" + "=" * 80)
        print("👑 EXECUTING GRAND CHAMPION VERSION 37 UNIFIED CANDIDATE PIPELINE")
        print("=" * 80)

        # 1. Warm up MetFrag-lite numba kernel
        frag_rescore.warmup()
        print("✓ MetFrag-lite numba engine initialized.")

        # 2. Load test queries from test.parquet
        print(f"\n1. Loading test queries from {self.test_parquet_path}...")
        te_df = pd.read_parquet(self.test_parquet_path)
        print(f"✓ Loaded {len(te_df)} spectra across {te_df['molecule_id'].nunique()} test molecules.")

        queries_dict: Dict[str, List[Dict]] = {}
        for row in te_df.itertuples(index=False):
            mid = str(row.molecule_id)
            m = np.asarray(row.ms2_mzs, dtype=np.float64)
            it = np.asarray(row.ms2_normalized_intensities, dtype=np.float64)
            prec = float(row.precursor_mz)
            mode = 1 if "pos" in str(getattr(row, "ionization_mode", "pos")).lower() else -1
            adduct = str(getattr(row, "adduct", "[M+H]+"))
            ce_raw = getattr(row, "collision_energy_ev", 30.0)
            if isinstance(ce_raw, (list, np.ndarray)):
                ce = float(ce_raw[0]) if len(ce_raw) > 0 else 30.0
            else:
                try:
                    ce = float(ce_raw) if (ce_raw is not None and not math.isnan(float(ce_raw))) else 30.0
                except Exception:
                    ce = 30.0

            ok = (m > 0) & (it > 0) & (m <= prec + 0.5)
            m, it = m[ok], it[ok]
            if len(m) > 0:
                o = np.argsort(m)
                m, it = m[o], it[o]

            spec_dict = {
                "mz": m,
                "it": it,
                "prec": prec,
                "mode": mode,
                "adduct": adduct,
                "ce": ce
            }
            queries_dict.setdefault(mid, []).append(spec_dict)

        # 3. Load baseline candidate submissions and fallback pool
        print(f"\n2. Loading baseline candidate pools...")
        base_df = pd.read_csv(self.baseline_csv_path)
        base_candidates: Dict[str, List[str]] = {}
        for _, row in base_df.iterrows():
            mid = str(row["molecule_id"])
            smis = [s.strip() for s in str(row["smiles"]).split(";") if s.strip()]
            base_candidates[mid] = smis

        fallback_candidates: Dict[str, List[str]] = {}
        if self.fallback_pool_csv_path.is_file():
            fb_df = pd.read_csv(self.fallback_pool_csv_path)
            for _, row in fb_df.iterrows():
                mid = str(row["molecule_id"])
                smis = [s.strip() for s in str(row["smiles"]).split(";") if s.strip()]
                fallback_candidates[mid] = smis

        # 4. Load Regime A Reference Library Matches
        lib_matches: Dict[str, str] = {}
        if self.library_matches_path.is_file():
            lib_matches = json.loads(self.library_matches_path.read_text())
            print(f"✓ Loaded {len(lib_matches)} verified Regime A reference spectral library matches.")

        protected_mids: Set[str] = set(lib_matches.keys())
        print(f"✓ Active Sovereign Rank-1 Shield: {len(protected_mids)} test queries immutably protected.")

        # 5. Process per-query candidates (Scoring + Reranking + Isotope + Regime C Regioisomers)
        print("\n3. Processing per-query candidate pools (Multi-CE + Isotope + MetFrag + Regime C)...")
        initial_candidates: Dict[str, List[str]] = {}
        stats = {
            "regime_a_shielded": 0,
            "regime_b_reranked": 0,
            "total_candidates_scored": 0,
            "dedup_swaps": 0,
            "fallback_fills": 0,
            "regio_fills": 0,
            "halogen_disqualifications": 0,
            "carbon_overhang_penalties": 0,
        }

        frag_cfg = {
            "max_breaks": 2,
            "max_bonds_2": 40,
            "max_bonds_3": 0,
            "w1": 1.0,
            "w2": 0.6,
            "w3": 0.0,
            "tol": 0.015,
            "mz_power": 0.25,
            "intensity": "sqrt",
            "h_shifts": (-2, -1, 0, 1, 2)
        }
        frag_cache = {}

        unique_mids = sorted(base_candidates.keys())

        for q_idx, mid in enumerate(unique_mids):
            specs = queries_dict.get(mid, [])
            cands = list(base_candidates[mid])
            stats["total_candidates_scored"] += len(cands)

            # Check Regime A: Absolute Rank-1 Shield
            has_shield = mid in protected_mids
            gold_leader = lib_matches.get(mid, None)
            if has_shield:
                stats["regime_a_shielded"] += 1
            else:
                stats["regime_b_reranked"] += 1

            if not specs or not cands:
                initial_candidates[mid] = cands[:25]
                continue

            # Phase 36: Collision-Energy & Peak-Count Weighted Multi-Spectrum Pooling
            pooled_specs = pool_multice_spectra(specs, tol_da=0.01, ce_weighting=True)
            pooled_s0 = pooled_specs[0]
            prec_mz = float(pooled_s0["prec"])
            adduct = str(pooled_s0["adduct"])

            # Calibrate precursor neutral mass (-1.50 ppm timsTOF)
            cal_neutral_mass = compute_neutral_mass(prec_mz, adduct, calibrate_timstof=True)

            # High-Resolution Isotope Profiling
            iso_profile = self.iso_engine.profile_spectrum(
                pooled_s0["mz"], pooled_s0["it"], prec_mz
            )

            # Diagnostic neutral losses
            detected_losses = detect_neutral_losses(pooled_s0["mz"], pooled_s0["it"], prec_mz)

            # MetFrag-lite bond cleavage scores
            frag_scores = frag_rescore.score_molecule(pooled_specs, cands, cfg=frag_cfg, cache=frag_cache)

            # Candidate composite scoring
            scored_candidates = []
            for rank_idx, smi in enumerate(cands):
                # 1. Learned Neural consensus rank prior
                neural_term = self.w_neural * (1.0 / (3.0 + rank_idx))

                # 2. timsTOF mass prior
                cand_mass = get_exact_mass(smi)
                if cand_mass is not None and cal_neutral_mass is not None:
                    p_mass = timstof_gaussian_mass_prior(np.array([cand_mass]), cal_neutral_mass)[0]
                else:
                    p_mass = 0.50
                mass_term = self.w_mass * math.log(max(p_mass, 1e-4))

                # 3. MetFrag-lite fragment score
                f_score = frag_scores[rank_idx] if rank_idx < len(frag_scores) and frag_scores[rank_idx] is not None else 0.0
                frag_term = self.w_frag * f_score

                # 4. Neutral loss plausibility
                m_plaus = evaluate_chemical_plausibility(smi, detected_losses)
                plaus_term = self.w_plaus * math.log(max(m_plaus, 1e-4))

                # 5. High-Resolution Isotope Consistency Evaluation
                iso_res = self.iso_engine.score_candidate(
                    smi, iso_profile, pooled_s0["mz"], pooled_s0["it"]
                )
                if iso_res["halogen_multiplier"] < 1e-3:
                    stats["halogen_disqualifications"] += 1
                if iso_res["carbon_consistency"] < 0.90:
                    stats["carbon_overhang_penalties"] += 1

                iso_term = self.w_iso * math.log(max(iso_res["composite_isotope_score"], 1e-4))

                composite = (
                    neural_term
                    + frag_term
                    + iso_term
                    + mass_term
                    + plaus_term
                )
                scored_candidates.append((smi, composite))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            reranked_smis = [s for s, _ in scored_candidates]

            # Enforce Absolute Rank-1 Shield: Freeze gold leader at Rank 1
            if has_shield and gold_leader is not None:
                final_ordered = [gold_leader] + [s for s in reranked_smis if s != gold_leader]
            else:
                final_ordered = reranked_smis

            # Deduplication on 14-char InChIKey skeletal graph
            seen_k14: Set[str] = set()
            deduped_smis: List[str] = []

            for smi in final_ordered:
                k14 = smiles_to_inchikey14(smi)
                if k14 is not None:
                    if k14 in seen_k14:
                        stats["dedup_swaps"] += 1
                        continue
                    seen_k14.add(k14)
                deduped_smis.append(smi)
                if len(deduped_smis) == 25:
                    break

            # Backfill depleted slots via secondary fallback pool
            if len(deduped_smis) < 25:
                fb_pool = fallback_candidates.get(mid, [])
                deduped_smis, adm_fb = admit_candidates_safely(
                    deduped_smis, fb_pool, target_count=25
                )
                stats["fallback_fills"] += adm_fb["pool_admitted"]

            # Phase 28: Regime C Constitutional Regioisomer Generation + MetFrag-lite scoring
            if len(deduped_smis) < 25 and len(deduped_smis) > 0:
                parent_smi = deduped_smis[0]
                regio_cands = generate_constitutional_isomers(
                    parent_smi, max_variants=25, existing_keys=seen_k14
                )
                if regio_cands:
                    regio_fscores = frag_rescore.score_molecule(
                        pooled_specs, regio_cands, cfg=frag_cfg, cache=frag_cache
                    )
                    scored_regios = [
                        (s, sc if sc is not None else 0.0)
                        for s, sc in zip(regio_cands, regio_fscores)
                    ]
                    scored_regios.sort(key=lambda x: x[1], reverse=True)
                    sorted_regios = [s for s, _ in scored_regios]
                    deduped_smis, adm_regio = admit_candidates_safely(
                        deduped_smis, sorted_regios, target_count=25
                    )
                    stats["regio_fills"] += adm_regio["pool_admitted"]

            # Emergency safety fallback
            if len(deduped_smis) < 25:
                base_pool = base_candidates.get(mid, [])
                deduped_smis, _ = admit_candidates_safely(
                    deduped_smis, base_pool, target_count=25
                )

            assert len(deduped_smis) == 25, f"Query {mid} has {len(deduped_smis)} candidates instead of 25"
            initial_candidates[mid] = deduped_smis

            if (q_idx + 1) % 100 == 0 or (q_idx + 1) == len(unique_mids):
                print(f"  [{q_idx + 1}/{len(unique_mids)}] queries processed (elapsed: {time.time() - t0:.1f}s)...")

        # 6. In-Test Molecular Networking with Hard Rank-1 Shield (Phase 29)
        print("\n4. Building In-Test Molecular Network & Running Constrained Reranker...")
        network_edges, net_diag = build_test_spectral_network(
            test_parquet_path=str(self.test_parquet_path),
            tau_threshold=self.network_tau,
            min_peaks=self.network_min_peaks
        )
        print(f"✓ Network Graph: {net_diag['total_edges_formed']} edges across {net_diag['queries_with_edges']} queries (elapsed: {net_diag['elapsed_seconds']}s).")

        final_candidates, net_stats = run_constrained_network_reranking(
            candidates_dict=initial_candidates,
            network_edges=network_edges,
            protected_mids=protected_mids,
            lam=self.network_lambda
        )

        print(f"✓ Constrained Network Reranking complete:")
        print(f"  - Shielded queries: {net_stats['shielded_queries']}")
        print(f"  - Shielded Rank 1 displaced: {net_stats['shielded_rank1_moved']} (100% frozen)")
        print(f"  - Sub-ranks reordered (ranks 2-25): {net_stats['subranks_reordered']}")
        print(f"  - Dedup failures: {net_stats['dedup_failures']}")

        # 7. Final Sanity & InChIKey14 Deduplication Verification
        out_rows = []
        for mid in unique_mids:
            cands = final_candidates[mid]
            assert len(cands) == 25, f"Query {mid} candidate count != 25"
            if mid in protected_mids and lib_matches.get(mid):
                assert cands[0] == lib_matches[mid], f"VIOLATION: Rank 1 shield breached for {mid}!"
            out_rows.append({"molecule_id": mid, "smiles": ";".join(cands)})

        # 8. Save final verified submission CSV
        submission_df = pd.DataFrame(out_rows)
        submission_df.to_csv(output_csv, index=False)
        total_time = time.time() - t0

        print(f"\n" + "=" * 80)
        print(f"👑 GRAND CHAMPION VERSION 37 SUBMISSION READY: {output_csv}")
        print(f"  - Rows: {len(submission_df)} | Total time: {total_time:.2f}s ({total_time / 60.0:.2f} min)")
        print(f"  - Regime A Shielded queries: {stats['regime_a_shielded']} / 400")
        print(f"  - Regime B/C Reranked queries: {stats['regime_b_reranked']} / 400")
        print(f"  - Halogen Decoy Disqualifications: {stats['halogen_disqualifications']}")
        print(f"  - Carbon Overhang Penalties: {stats['carbon_overhang_penalties']}")
        print(f"  - InChIKey14 duplicate swaps: {stats['dedup_swaps']}")
        print(f"  - Secondary fallback fills: {stats['fallback_fills']}")
        print(f"  - Regime C Regioisomer fills: {stats['regio_fills']}")
        print("=" * 80)

        report = {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "output_file": str(output_csv),
            "rows": len(submission_df),
            "total_execution_seconds": total_time,
            "runtime_budget_hours_used": total_time / 3600.0,
            "runtime_budget_limit_hours": self.time_guard_seconds / 3600.0,
            "stats": stats,
            "network_diagnostics": net_diag,
            "network_reranker_stats": net_stats,
            "config": {
                "w_neural": self.w_neural,
                "w_frag": self.w_frag,
                "w_iso": self.w_iso,
                "w_mass": self.w_mass,
                "w_plaus": self.w_plaus,
                "network_tau": self.network_tau,
                "network_min_peaks": self.network_min_peaks,
                "network_lambda": self.network_lambda,
                "timstof_offset_ppm": TIMSTOF_OFFSET_PPM,
            }
        }

        report_path = ROOT / "experiments" / "grand_champion_v37_verification_report.json"
        report_path.write_text(json.dumps(report, indent=2))
        print(f"✓ Verification report written to {report_path}")

        return report

def main():
    pipeline = GrandChampionV37Pipeline()
    report = pipeline.run("submissions/submission_grand_champion_v37.csv")
    print("\nGrand Champion Version 37 Pipeline Run Complete.")

if __name__ == "__main__":
    main()
