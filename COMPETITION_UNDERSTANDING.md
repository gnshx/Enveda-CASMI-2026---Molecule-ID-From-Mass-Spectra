# 🔬 COMPETITION_UNDERSTANDING.md
# Enveda CASMI 2026: Molecule Identification From Mass Spectra
**Official Competition Link**: [kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra](https://www.kaggle.com/competitions/enveda-CASMI26-molecule-id-mass-spectra)  
**Timeline**: Closes **December 14, 2026 at 23:59 UTC** (Entry & Team Merger Deadline: December 7, 2026).  
**Mission**: Predict 2D chemical structures (SMILES strings) of small molecules detected in complex biological extracts from LC-MS/MS tandem mass spectra.

---

## 1. Executive Scientific Overview

The Enveda CASMI 2026 challenge resurrects the renowned Critical Assessment of Small Molecule Identification (CASMI) as a blind, competitive machine-learning benchmark. Unlike synthetic benchmark datasets, this competition is drawn from real-world, high-complexity biological extracts (plants, fungi, bacteria, mammalian metabolomes).

### The Scientific Task
Given one or multiple tandem mass spectra (MS/MS) belonging to an unknown biological molecule, predict an ordered candidate list of up to 25 molecular structures formatted as valid SMILES strings.

### The Official Evaluation Metric: MRR@25
The competition is evaluated on **Mean Reciprocal Rank @ 25 (MRR@25)**:
$$\text{MRR@25} = \frac{1}{N} \sum_{i=1}^{N} \text{RR}_i$$
where:
$$\text{RR}_i = \begin{cases} \frac{1}{\text{rank}_i} & \text{if true structure is found at } \text{rank}_i \in [1, 25] \\ 0 & \text{otherwise} \end{cases}$$

- **Rank 1**: $\text{RR} = 1.000$ (Full credit)
- **Rank 2**: $\text{RR} = 0.500$
- **Rank 3**: $\text{RR} = 0.333$
- **Rank 5**: $\text{RR} = 0.200$
- **Rank 10**: $\text{RR} = 0.100$
- **Rank 25**: $\text{RR} = 0.040$
- **Rank > 25**: $\text{RR} = 0.000$

### Molecular Canonicalization & Identity Definition
The competition measures identity on **Tautomer-Canonical InChIKey14**:
1. Candidate SMILES is parsed with RDKit: `mol = Chem.MolFromSmiles(smiles)`
2. Tautomer standardization: `mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)`
3. First 14 characters of InChIKey: `inchikey14 = Chem.MolToInchiKey(mol)[:14]`

**Critical Cheminformatic Implication**:
- The first 14 characters of the InChIKey encode the **molecular connectivity / heavy-atom skeleton and elemental composition**, excluding stereochemistry (R/S, cis/trans) and tautomeric hydrogen placement.
- Therefore, optimizing stereochemical assignments (e.g. chiral centers `@` / `@@`) does NOT yield any metric gain.
- Conversely, duplicate stereoisomers within a molecule's 25 candidate predictions are redundant and waste valuable ranking slots. Candidates MUST be strictly deduplicated at the InChIKey14 level before output.

---

## 2. The Three Discovery Regimes

The unknown test molecules fall into three distinct scientific regimes, each requiring fundamentally different algorithmic approaches:

```
┌─────────────────────────────────────────────────────────────────────────────────────────────┐
│                                  THE THREE DISCOVERY REGIMES                                │
├──────────────────────────────┬──────────────────────────────┬───────────────────────────────┤
│ Regime A: Library Match      │ Regime B: Database Retrieval │ Regime C: De Novo Generation  │
│ (~10 - 15% of test set)      │ (~45 - 55% of test set)      │ (~30 - 40% of test set)       │
├──────────────────────────────┼──────────────────────────────┼───────────────────────────────┤
│ The exact reference spectrum │ The spectrum is novel, but   │ The molecule is a completely  │
│ exists in public reference   │ the structure exists in      │ novel natural product absent  │
│ libraries (MassBank, GNPS,   │ chemical databases (PubChem, │ from all databases.           │
│ Takumu Hata, RIKEN, etc.).   │ COCONUT, LOTUS, NPAtlas).    │                               │
├──────────────────────────────┼──────────────────────────────┼───────────────────────────────┤
│ Strategy:                    │ Strategy:                    │ Strategy:                     │
│ Exact precursor matching     │ Narrow by precursor formula, │ Substructure reasoning,       │
│ (±10 ppm), peak alignment,   │ predict molecular fingerprint│ neutral loss cleavage trees,  │
│ high-threshold modified      │ (FPNet/MS2DeepScore), score  │ graph-rewiring de novo        │
│ cosine (≥ 0.80), and direct  │ with neural rerankers        │ generation, GFN2-xTB          │
│ promotion to Rank 1.         │ (Saner v1, custom scorer).   │ quantum stability filtering.  │
└──────────────────────────────┴──────────────────────────────┴───────────────────────────────┘
```

---

## 3. Dataset Schemas & Acquisition Conditions

### Training Set (`train.parquet`: 2.89 GB)
Contains ~2.43 million spectra covering ~275,000 unique molecular entities across diverse instruments:
- `molecule_id`: Unique identifier for the chemical compound.
- `spectrum_id`: Unique identifier for the individual acquisition.
- `ms2_mzs`: Array of fragment mass-to-charge ratios ($m/z$).
- `ms2_normalized_intensities`: Array of peak intensities (normalized to base peak = 1.0).
- `base_peak_intensity`: Absolute raw intensity of the tallest peak.
- `precursor_mz`: Mass-to-charge of the isolated parent ion.
- `adduct`: Ionization adduct (e.g., `[M+H]+`, `[M-H]-`, `[M+Na]+`, `[M+NH4]+`, `[M+HCOO]-`).
- `ionization_mode`: `positive` or `negative`.
- `instrument_type`: `timsTOF` (Bruker), `Orbitrap`, `Q-Exactive`, `Q-TOF`.
- `collision_energy_ev`: Calibrated collision energy in electron-volts.
- `normalized_smiles`: Ground-truth 2D chemical structure.
- `inchikey` / `inchikey14`: Standard and skeletal InChIKey hashes.
- `molecular_formula`: Elemental composition (e.g., `C15H10O4`).
- `ingest_lib`: Source repository (`enveda-180`, `riken`, `gnps`, `massbank`, `mona`, `enveda-np-examples`).

### Test Set (`test.parquet`: 4.62 MB)
Contains MS/MS spectra for unknown molecules.
- Crucially, the test set is acquired primarily on a **Bruker timsTOF** instrument.
- The chemical domain is enriched for complex natural products from biological extracts.
- Ground truth `normalized_smiles` is hidden and evaluated on Kaggle's private scoring worker.

---

## 4. Kaggle Platform Constraints & Compliance Rules

1. **Execution Environment**:
   - Notebook code competition: execution is triggered by `kaggle competitions submit`.
   - **Internet Access**: Strictly **DISABLED** during scoring rerun. All models, wheels, and libraries must be packaged in Kaggle Datasets.
   - **Execution Time Guard**: Maximum **9.0 hours (32,400 seconds)** hard limit. Exceeding 9.0 hours results in an abrupt `SIGKILL` and permanent `SubmissionStatus.ERROR`.
   - **Compute Hardware**: 1x NVIDIA Tesla T4 or P100 GPU (16 GB VRAM), 4 CPU cores, 30 GB RAM.

2. **Submission Constraints**:
   - Output format: `submission.csv` containing two columns: `molecule_id` and `smiles`.
   - Format: Up to 25 ranked candidate SMILES separated by semicolons (`;`).
   - Every molecule in `test.parquet` must appear exactly once.
   - No missing or null values.

3. **External Data Rules**:
   - Permitted: Open-source scientific databases and pretrained checkpoints released before the deadline with open, reproducible licenses (CC BY, CC0, Apache 2.0).
   - Forbidden: Proprietary commercial databases that restrict redistribution of model derivatives (e.g. NIST commercial library).
