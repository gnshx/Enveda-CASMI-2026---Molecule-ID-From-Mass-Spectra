"""
aggregation/multi_spectrum.py
Aggregate evidence from multiple spectra for the same molecule.

Key insight: A molecule with 3 spectra at 20/40/60 eV provides more information
than a single spectrum. Different collision energies reveal different structural features.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field


@dataclass
class SpectrumEvidence:
    """Evidence for a single candidate from one spectrum."""
    spectrum_id: str
    collision_energy_ev: float
    cosine_score: float
    dreams_score: float
    fragment_match_score: float
    mass_error_ppm: float


@dataclass
class MoleculeCandidate:
    """A candidate structure with aggregated evidence across multiple spectra."""
    smiles: str
    inchikey14: Optional[str]
    evidence: List[SpectrumEvidence] = field(default_factory=list)

    def add_evidence(self, ev: SpectrumEvidence) -> None:
        self.evidence.append(ev)

    def aggregate_features(self) -> Dict[str, float]:
        """
        Compute aggregated features across all supporting spectra.
        These become inputs to the learning-to-rank model.
        """
        if not self.evidence:
            return self._empty_features()

        cosines = [e.cosine_score for e in self.evidence]
        dreams = [e.dreams_score for e in self.evidence]
        fragments = [e.fragment_match_score for e in self.evidence]
        mass_errors = [e.mass_error_ppm for e in self.evidence]
        energies = [e.collision_energy_ev for e in self.evidence]

        return {
            # Cosine aggregation
            "cosine_max":     max(cosines),
            "cosine_mean":    np.mean(cosines),
            "cosine_min":     min(cosines),
            "cosine_std":     float(np.std(cosines)),

            # DreaMS aggregation
            "dreams_max":     max(dreams),
            "dreams_mean":    np.mean(dreams),
            "dreams_std":     float(np.std(dreams)),

            # Fragment match aggregation
            "fragment_max":   max(fragments),
            "fragment_mean":  np.mean(fragments),

            # Mass error
            "mass_error_min": min(mass_errors),
            "mass_error_mean": np.mean(mass_errors),

            # Multi-spectrum consistency
            "n_supporting_spectra": len(self.evidence),
            "n_spectra_cosine_gt_0.3": sum(1 for c in cosines if c > 0.3),
            "n_spectra_cosine_gt_0.5": sum(1 for c in cosines if c > 0.5),
            "fraction_spectra_supporting": sum(1 for c in cosines if c > 0.1) / len(cosines),

            # Collision energy coverage
            "has_low_ce":  float(any(e < 30 for e in energies)),
            "has_mid_ce":  float(any(30 <= e <= 60 for e in energies)),
            "has_high_ce": float(any(e > 60 for e in energies)),
            "ce_range":    max(energies) - min(energies) if len(energies) > 1 else 0.0,
            "n_ce_levels": len(set(round(e / 10) * 10 for e in energies)),
        }

    def _empty_features(self) -> Dict[str, float]:
        return {k: 0.0 for k in [
            "cosine_max", "cosine_mean", "cosine_min", "cosine_std",
            "dreams_max", "dreams_mean", "dreams_std",
            "fragment_max", "fragment_mean",
            "mass_error_min", "mass_error_mean",
            "n_supporting_spectra", "n_spectra_cosine_gt_0.3",
            "n_spectra_cosine_gt_0.5", "fraction_spectra_supporting",
            "has_low_ce", "has_mid_ce", "has_high_ce", "ce_range", "n_ce_levels",
        ]}


class MoleculeEvidenceAggregator:
    """
    Aggregate per-spectrum candidate scores into per-molecule candidate rankings.

    Usage:
        agg = MoleculeEvidenceAggregator()
        for spectrum in molecule_spectra:
            candidates = retrieval_system.search(spectrum)
            agg.add_spectrum_results(spectrum.id, spectrum.ce, candidates)

        top25 = agg.get_top_k(k=25)
    """

    def __init__(self):
        self.candidates: Dict[str, MoleculeCandidate] = {}  # smiles -> candidate

    def add_spectrum_results(
        self,
        spectrum_id: str,
        collision_energy_ev: float,
        candidate_scores: Dict[str, Dict[str, float]],
        # candidate_scores: {smiles: {cosine, dreams, fragment, mass_error_ppm}}
    ) -> None:
        """Add retrieval results from one spectrum."""
        for smiles, scores in candidate_scores.items():
            if smiles not in self.candidates:
                self.candidates[smiles] = MoleculeCandidate(smiles=smiles, inchikey14=None)

            ev = SpectrumEvidence(
                spectrum_id=spectrum_id,
                collision_energy_ev=collision_energy_ev,
                cosine_score=scores.get("cosine", 0.0),
                dreams_score=scores.get("dreams", 0.0),
                fragment_match_score=scores.get("fragment", 0.0),
                mass_error_ppm=scores.get("mass_error_ppm", 999.0),
            )
            self.candidates[smiles].add_evidence(ev)

    def get_feature_matrix(self) -> Tuple[np.ndarray, List[str]]:
        """
        Return feature matrix for all candidates.

        Returns:
            (feature_matrix [n_candidates, n_features], smiles_list)
        """
        smiles_list = list(self.candidates.keys())
        if not smiles_list:
            return np.array([]), []

        feature_dicts = [
            self.candidates[smi].aggregate_features()
            for smi in smiles_list
        ]
        feature_names = list(feature_dicts[0].keys())
        matrix = np.array([[d[k] for k in feature_names] for d in feature_dicts])
        return matrix, smiles_list

    def get_top_k_by_cosine_max(self, k: int = 25) -> List[str]:
        """Simple baseline: rank by max cosine score across spectra."""
        scored = [
            (smi, cand.aggregate_features()["cosine_max"])
            for smi, cand in self.candidates.items()
        ]
        scored.sort(key=lambda x: -x[1])
        return [smi for smi, _ in scored[:k]]
