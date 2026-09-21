"""
validation/splitter.py
Three validation split strategies for CASMI 2026.

Critical: NEVER use random train/test_split on spectra — this leaks molecule identity.

Actual train.parquet column names (from official data page):
    molecule_id, spectrum_id, ms2_mzs, ms2_normalized_intensities,
    base_peak_intensity, adduct, ionization_mode, instrument_type,
    precursor_mz, collision_energy_ev, collision_energy_orig,
    collision_energy_orig_units, num_peaks,
    normalized_smiles  ← label (NOT 'smiles')
    inchikey, inchikey14, molecular_formula,
    ingest_lib  ← source library (NOT 'source')
    adduct_orig, precursor_error_ppm
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from dataclasses import dataclass


@dataclass
class SplitResult:
    train_ids: List[str]   # molecule_ids in train fold
    val_ids: List[str]     # molecule_ids in val fold
    split_name: str
    description: str


def structure_split(
    df: pd.DataFrame,
    molecule_col: str = "molecule_id",
    smiles_col: str = "normalized_smiles",   # train.parquet label column
    val_fraction: float = 0.15,
    seed: int = 42,
) -> SplitResult:
    """
    Split A: No molecule overlap between train and val.

    Each unique molecule_id appears in ONLY one split.
    This is the minimum correct way to validate.

    Args:
        df: DataFrame with molecule_id and SMILES columns
        val_fraction: Fraction of unique molecules to put in val
        seed: Random seed for reproducibility

    Returns:
        SplitResult with molecule_id lists
    """
    rng = np.random.default_rng(seed)
    unique_mols = df[molecule_col].unique()
    rng.shuffle(unique_mols)

    n_val = int(len(unique_mols) * val_fraction)
    val_ids = unique_mols[:n_val].tolist()
    train_ids = unique_mols[n_val:].tolist()

    return SplitResult(
        train_ids=train_ids,
        val_ids=val_ids,
        split_name="structure_split",
        description=f"No molecule overlap. Val: {n_val}/{len(unique_mols)} unique molecules.",
    )


def library_split(
    df: pd.DataFrame,
    molecule_col: str = "molecule_id",
    source_col: str = "ingest_lib",          # actual column name in train.parquet
    val_libraries: Optional[List[str]] = None,
) -> SplitResult:
    """
    Split B: Train on some spectral libraries, validate on others.

    Tests domain generalization across different instrument/library combinations.
    Recommended val_libraries: ["riken", "mona"] — different instruments from train.

    Note: ingest_lib values in train.parquet are lowercase:
        enveda-180, pluskal_ms2, riken, gnps, massbank, mona,
        spectraverse, msdial, drug_plus, enveda-np-examples, masaryk

    Args:
        df: DataFrame with molecule_id, ingest_lib columns
        val_libraries: List of ingest_lib values to use as validation

    Returns:
        SplitResult with molecule_id lists
    """
    if val_libraries is None:
        val_libraries = ["riken", "mona"]

    val_mask = df[source_col].isin(val_libraries)

    # A molecule is in val only if ALL its spectra come from val libraries
    mol_sources = df.groupby(molecule_col)[source_col].apply(set)
    val_mols = mol_sources[mol_sources.apply(lambda s: s.issubset(set(val_libraries)))].index.tolist()
    train_mols = mol_sources[~mol_sources.index.isin(val_mols)].index.tolist()

    return SplitResult(
        train_ids=train_mols,
        val_ids=val_mols,
        split_name="library_split",
        description=f"Val libraries: {val_libraries}. Val mols: {len(val_mols)}",
    )


def natural_product_split(
    df: pd.DataFrame,
    molecule_col: str = "molecule_id",
    source_col: str = "ingest_lib",          # actual column name in train.parquet
    np_libraries: Optional[List[str]] = None,
) -> SplitResult:
    """
    Split C: Train on general chemistry, validate on natural product spectra.

    This is the most competition-realistic split, since:
    - The test set is NP-like chemistry (plants, mammals, microbes)
    - The dominant training source (enveda-180) is synthetic drug-like chemistry
    - Only enveda-180 and enveda-np-examples are timsTOF (matching test instrument)

    Recommended NP val libraries: enveda-np-examples, riken
    enveda-np-examples is 250 common NPs on the SAME instrument as the test set —
    treat this as your gold-standard validation subset.
    """
    if np_libraries is None:
        np_libraries = ["enveda-np-examples", "riken"]

    np_mask = df[source_col].isin(np_libraries)
    val_mol_ids = df[np_mask][molecule_col].unique().tolist()
    train_mol_ids = df[~df[molecule_col].isin(val_mol_ids)][molecule_col].unique().tolist()

    return SplitResult(
        train_ids=train_mol_ids,
        val_ids=val_mol_ids,
        split_name="np_split",
        description=(
            f"NP val libraries: {np_libraries}. "
            f"Val mols: {len(val_mol_ids)}, Train mols: {len(train_mol_ids)}"
        ),
    )


def simulate_novelty_classes(
    val_df: pd.DataFrame,
    train_smiles_set: set,
    database_smiles_set: set,
    molecule_col: str = "molecule_id",
    smiles_col: str = "normalized_smiles",   # actual label column in train.parquet
) -> dict:
    """
    Assign simulated novelty class to each val molecule:

    Class 1: molecule has known spectra in train (spectral library hit)
    Class 2: molecule in database (COCONUT/PubChem) but not in train spectra
    Class 3: molecule NOT in any database

    Args:
        val_df: Validation DataFrame
        train_smiles_set: Set of SMILES/InChIKey14 in training spectral library
        database_smiles_set: Set of SMILES/InChIKey14 in candidate databases

    Returns:
        {molecule_id: 1|2|3}
    """
    mol_classes = {}
    for mol_id, group in val_df.groupby(molecule_col):
        smiles = group[smiles_col].iloc[0]

        if smiles in train_smiles_set:
            mol_classes[mol_id] = 1
        elif smiles in database_smiles_set:
            mol_classes[mol_id] = 2
        else:
            mol_classes[mol_id] = 3

    class_counts = {1: 0, 2: 0, 3: 0}
    for c in mol_classes.values():
        class_counts[c] += 1

    print("Simulated novelty class distribution:")
    for c in [1, 2, 3]:
        print(f"  Class {c}: {class_counts[c]} molecules ({class_counts[c]/len(mol_classes)*100:.1f}%)")

    return mol_classes
