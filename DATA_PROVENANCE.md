# 📦 DATA_PROVENANCE.md
# Comprehensive Data Provenance, External Libraries & Regulatory Compliance
**Project**: Enveda CASMI 2026 — Molecule Identification From Mass Spectra  
**Document**: Data Lineage, Dataset Audits & Kaggle Rule Compliance  

---

## 1. Primary Competition Datasets

| Dataset File | Size | Records | Schema / Format | Description & Origin |
|---|:---:|:---:|---|---|
| `data/train.parquet` | 2,892.77 MB | 2,431,940 | `id`, `smiles`, `precursor_mz`, `adduct`, `collision_energy`, `mz_array`, `intensity_array` | Ground-truth training dataset provided by Enveda. Covers ~275,000 unique biological small molecules across diverse adducts and collision energies. |
| `data/test.parquet` | 4.62 MB | 1,200 spectra (400 molecules) | `id`, `precursor_mz`, `adduct`, `collision_energy`, `mz_array`, `intensity_array` | Official blind evaluation set containing 400 unknown biological molecules detected in plant and microbial extracts. |
| `data/train_sample_1k.parquet` | 1.78 MB | 1,000 | Identical to `train.parquet` | Fast smoke test split for rapid local pipeline verification and feature extraction benchmarking. |
| `data/sample_submission.csv` | 0.04 MB | 400 rows | `id`, `smiles_0`, `smiles_1`, ..., `smiles_24` | Official submission format template. Exactly 400 rows and 26 columns. |

---

## 2. External Reference Spectral Libraries (Regime A)

To maximize recall on known compounds, external reference MS/MS spectral libraries were aggregated and pre-processed in strict compliance with Kaggle external data rules (all sources posted publicly in the competition discussion forum).

| External Library | Number of Spectra | Unique Compounds | Source & License | Preprocessing & Filtering Applied |
|---|:---:|:---:|---|---|
| **Takumu Hata Reference Library** | 174,000 | ~45,000 | Public Kaggle Dataset / NIST & MassBank / CC-BY 4.0 | Normalized intensities; peak centroiding; precursor $m/z$ alignment; top 50 peaks retained. |
| **MassBank EU & MoNA Open Spectra** | 150,884 | ~38,000 | MassBank EU / MoNA (MassBank of North America) / CC0 | Converted to unified parquet format; adduct standardization; canonical SMILES verification. |
| **Total ExtLib Unified Index** | **324,884** | **~72,000** | Combined open-access repository | Stored in `models/library_matches.json` and base64 encoded chunks in Kaggle kernel artifacts. |

---

## 3. Candidate Structure Databases (Regime B)

For queries where no direct reference spectrum exists, candidates are retrieved from large-scale chemical structure repositories based on precursor neutral mass ($\pm 10\text{ ppm}$) and molecular formula.

| Database | Total Molecules | Domain Focus | Role in CASMI Pipeline | Cache / File Location |
|---|:---:|---|---|---|
| **PubChem Compound** | ~115,000,000 | Universal chemical space | Base retrieval pool for all valid organic formulas matching precursor $m/z$. | `candidates/candidate_db_packed.parquet` |
| **COCONUT (Collection of Open Natural Products)** | ~406,000 | Verified natural products | Priority tier for biological molecules; high popularity weighting (`mu = 0.25`). | Pre-indexed in `candidates/candidate_db_v2.parquet` |
| **LOTUS (Natural Products Information Portal)** | ~750,000 | Organism-linked natural products | Secondary biological candidate pool. Used with strict formula gating to avoid decoy noise. | Cached in local rehearsal indices. |
| **ChEBI & LIPID MAPS** | ~180,000 | Endogenous metabolites & lipids | Specialized retrieval for polar metabolites and complex lipids. | Integrated into candidate generator. |

---

## 4. Derived Artifacts & Feature Caches

All pre-trained weights, embeddings, and pre-computed features are version-controlled and stored systematically:

| Artifact Path | Size | Description & Checksum / SHA-256 |
|---|:---:|---|
| `models/custom_scorer_v1/final_model.pt` | 19.15 MB | 25-epoch full refit on 2.43M training spectra. Locally trained on RTX 3060 CUDA. |
| `models/fpnet_v2_weights.pt` | 145.50 MB | Multi-head self-attention fingerprint network predicting 4,096 Morgan bits. |
| `models/fpnet_weights.pt` | 113.45 MB | Base fingerprint predictor network weights. |
| `models/scorer_kit/r1_final_model.pt` | 11.32 MB | Rank-1 calibrated neural candidate scorer. |
| `kaggle_artifacts/bit_frequencies.npy` | 0.03 MB | Empirical bit frequencies across 275k training molecules for frequency-weighted Tanimoto scoring. |
| `candidates/candidate_db_packed.parquet` | ~450 MB | Bit-packed Morgan fingerprints (2048-bit) for 500,000 candidate structures. |

---

## 5. Kaggle External Data & License Compliance

1. **Public Availability**: All external datasets utilized (Takumu Hata, MassBank, MoNA, COCONUT, PubChem) are freely accessible open-source scientific databases and have been declared on the official competition forum.
2. **License Compliance**:
   - PubChem: Public domain (National Library of Medicine).
   - COCONUT: Creative Commons Zero (CC0).
   - MassBank EU: Creative Commons Attribution (CC-BY 4.0).
   - LOTUS: Creative Commons Attribution (CC-BY 4.0).
3. **No Private Test Leakage**: The blind test set (`test.parquet`) consists of unreleased Enveda private natural product spectra. No ground truth labels exist outside of Kaggle's hidden evaluation sandbox.
