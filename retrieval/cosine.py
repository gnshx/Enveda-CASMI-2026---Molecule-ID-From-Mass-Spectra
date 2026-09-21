"""
retrieval/cosine.py
Fast cosine similarity spectral retrieval against the training library.

Strategy:
  - Index NP-relevant training libraries (riken, gnps, mona, massbank,
    enveda-np-examples, spectraverse, msdial, masaryk) — skipping enveda-180
    (synthetic drug-like, different chemical space from test)
  - For each test spectrum: cosine search → top-K candidates by InChIKey14
  - Aggregate candidates across spectra for same molecule_id → top-25 SMILES
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.peak_filter import PeakFilterConfig, filter_spectrum
from chemistry.adducts import neutral_mass, TEST_SET_ADDUCTS

# ── Libraries to index (NP-relevant only) ────────────────────────────────────
# Excluding enveda-180: ~1.15M synthetic drug-like spectra, wrong chemical space
NP_LIBRARIES = {
    "riken", "gnps", "mona", "massbank",
    "enveda-np-examples", "spectraverse", "msdial", "masaryk",
    "pluskal_ms2",   # include — bioactive compounds, multiple CE levels
}


@dataclass
class CosineConfig:
    mz_tolerance: float = 0.05      # Da — peak matching tolerance
    top_k_per_spectrum: int = 50    # candidates to retrieve per spectrum
    min_matched_peaks: int = 3      # discard matches with fewer matched peaks
    filter_cfg: PeakFilterConfig = field(default_factory=PeakFilterConfig)


def _preprocess(mzs, intensities, precursor_mz, cfg: PeakFilterConfig):
    """Preprocess a single spectrum, return (mzs, intensities) arrays."""
    mzs = np.asarray(mzs, dtype=np.float32)
    ints = np.asarray(intensities, dtype=np.float32)
    return filter_spectrum(mzs, ints, float(precursor_mz), cfg)


def cosine_score(
    mzs1: np.ndarray, ints1: np.ndarray,
    mzs2: np.ndarray, ints2: np.ndarray,
    tol: float = 0.05,
) -> Tuple[float, int]:
    """
    Compute cosine similarity between two preprocessed spectra.
    Returns (score, n_matched_peaks).
    """
    if len(mzs1) == 0 or len(mzs2) == 0:
        return 0.0, 0

    matched1, matched2 = [], []
    used2 = np.zeros(len(mzs2), dtype=bool)

    for i in range(len(mzs1)):
        diffs = np.abs(mzs2 - mzs1[i])
        mask = (diffs <= tol) & (~used2)
        if mask.any():
            j = np.argmin(np.where(mask, diffs, np.inf))
            matched1.append(ints1[i])
            matched2.append(ints2[j])
            used2[j] = True

    if not matched1:
        return 0.0, 0

    m1, m2 = np.array(matched1), np.array(matched2)
    dot = float(np.dot(m1, m2))
    norm = float(np.linalg.norm(ints1) * np.linalg.norm(ints2))
    score = dot / norm if norm > 0 else 0.0
    return score, len(matched1)


class SpectralLibrary:
    """
    In-memory spectral library index.
    Stores preprocessed spectra + metadata for fast cosine search.
    """

    def __init__(self, cfg: CosineConfig = CosineConfig()):
        self.cfg = cfg
        self.spectra: List[Tuple[np.ndarray, np.ndarray]] = []  # (mzs, ints)
        self.metadata: List[Dict] = []   # inchikey14, normalized_smiles, precursor_mz

    def build_from_train(
        self,
        train_df: pd.DataFrame,
        libraries: Optional[set] = None,
        max_per_lib: Optional[int] = None,
    ) -> None:
        """
        Index training spectra from selected libraries.

        Args:
            train_df: Full train.parquet DataFrame
            libraries: Set of ingest_lib values to include (None = all)
            max_per_lib: Cap spectra per library (for speed testing)
        """
        if libraries is not None:
            df = train_df[train_df["ingest_lib"].isin(libraries)].copy()
        else:
            df = train_df.copy()

        if max_per_lib is not None:
            df = df.groupby("ingest_lib").head(max_per_lib).reset_index(drop=True)

        print(f"  Indexing {len(df):,} spectra from {df['ingest_lib'].nunique()} libraries...")

        n_valid = 0
        for _, row in df.iterrows():
            mzs, ints = _preprocess(
                row["ms2_mzs"],
                row["ms2_normalized_intensities"],
                row["precursor_mz"],
                self.cfg.filter_cfg,
            )
            if len(mzs) < self.cfg.filter_cfg.min_peaks:
                continue

            self.spectra.append((mzs, ints))
            self.metadata.append({
                "inchikey14":       row.get("inchikey14", ""),
                "normalized_smiles": row.get("normalized_smiles", ""),
                "precursor_mz":     float(row["precursor_mz"]),
                "ingest_lib":       row.get("ingest_lib", ""),
            })
            n_valid += 1

        print(f"  Indexed {n_valid:,} valid spectra.")

    def search(
        self,
        query_mzs: np.ndarray,
        query_ints: np.ndarray,
        query_precursor_mz: float,
        top_k: int = 50,
    ) -> List[Dict]:
        """
        Search library for top-K most similar spectra.

        Returns list of dicts sorted by cosine score descending:
            {inchikey14, normalized_smiles, cosine_score, n_matched, precursor_mz}
        """
        results = []
        for i, (lib_mzs, lib_ints) in enumerate(self.spectra):
            score, n_matched = cosine_score(
                query_mzs, query_ints,
                lib_mzs, lib_ints,
                tol=self.cfg.mz_tolerance,
            )
            if score > 0 and n_matched >= self.cfg.min_matched_peaks:
                results.append({
                    **self.metadata[i],
                    "cosine_score": score,
                    "n_matched_peaks": n_matched,
                })

        results.sort(key=lambda x: -x["cosine_score"])
        return results[:top_k]


def predict_molecule(
    spectra_rows: pd.DataFrame,
    library: SpectralLibrary,
    cfg: CosineConfig = CosineConfig(),
    top_n: int = 25,
) -> List[str]:
    """
    Predict top-N SMILES for one molecule from all its spectra.

    Aggregation: max cosine score across spectra per candidate.
    Returns: ranked list of SMILES strings (best first).
    """
    candidate_scores: Dict[str, float] = {}   # inchikey14 → best score
    candidate_smiles: Dict[str, str] = {}      # inchikey14 → smiles

    for _, row in spectra_rows.iterrows():
        mzs, ints = _preprocess(
            row["ms2_mzs"],
            row["ms2_normalized_intensities"],
            row["precursor_mz"],
            cfg.filter_cfg,
        )
        if len(mzs) < cfg.filter_cfg.min_peaks:
            continue

        hits = library.search(mzs, ints, float(row["precursor_mz"]), top_k=cfg.top_k_per_spectrum)

        for hit in hits:
            key = hit["inchikey14"]
            if not key:
                continue
            prev = candidate_scores.get(key, 0.0)
            if hit["cosine_score"] > prev:
                candidate_scores[key] = hit["cosine_score"]
                candidate_smiles[key] = hit["normalized_smiles"]

    # Sort by best cosine score, return SMILES
    ranked = sorted(candidate_scores.items(), key=lambda x: -x[1])
    return [candidate_smiles[k] for k, _ in ranked[:top_n] if candidate_smiles.get(k)]


def run_baseline(
    train_path: str,
    test_path: str,
    output_path: str = "submission_cosine_baseline.csv",
    libraries: Optional[set] = NP_LIBRARIES,
    max_per_lib: Optional[int] = None,
) -> pd.DataFrame:
    """End-to-end cosine baseline pipeline."""
    print("Loading data...")
    train = pd.read_parquet(train_path)
    test  = pd.read_parquet(test_path)
    print(f"  Train: {len(train):,} spectra | Test: {test['molecule_id'].nunique()} molecules")

    cfg = CosineConfig()

    print("\nBuilding spectral library index...")
    lib = SpectralLibrary(cfg)
    lib.build_from_train(train, libraries=libraries, max_per_lib=max_per_lib)

    print("\nPredicting test molecules...")
    predictions = {}
    mol_ids = test["molecule_id"].unique()

    for i, mol_id in enumerate(mol_ids):
        mol_spectra = test[test["molecule_id"] == mol_id]
        smiles_list = predict_molecule(mol_spectra, lib, cfg, top_n=25)

        # Fallback: if no hits, submit ethanol (like sample submission)
        if not smiles_list:
            smiles_list = ["CCO"] * 25

        predictions[mol_id] = smiles_list

        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(mol_ids)} molecules done...")

    # Build submission CSV
    rows = [{"molecule_id": mid, "smiles": ";".join(smi[:25])}
            for mid, smi in predictions.items()]
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"\nSaved → {output_path}  ({len(df)} rows)")
    return df


if __name__ == "__main__":
    DATA = Path("/home/dlcv/Enveda-CASMI-2026/data")
    run_baseline(
        train_path=str(DATA / "train.parquet"),
        test_path=str(DATA / "test.parquet"),
        output_path="/home/dlcv/Enveda-CASMI-2026/submission_cosine_baseline.csv",
        libraries=NP_LIBRARIES,
        max_per_lib=None,
    )
