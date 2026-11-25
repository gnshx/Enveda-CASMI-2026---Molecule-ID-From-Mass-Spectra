# V43 Validation & Candidate-Recall Bottleneck Report

## 1. Canonical-Tautomer Split Integrity (`validation/v43_build_splits.py`)

- Scanned `2,539,608` spectra (`277,566` unique `normalized_smiles`, `275,810` raw `inchikey14`).
- Canonical tautomer standardization collapsed **4,375 SMILES** differing from raw `inchikey14` down to **`274,289` unique `canon_ik14`**.
- Critically, **`250 / 250`** of `enveda-np-examples` (`309` SMILES) also appear in non-Enveda training libraries (`riken`: 246, `gnps`: 233, `mona`: 231, `msdial`: 227, `pluskal_ms2`: 224, `massbank`: 208), proving why naive validation on `enveda-np-examples` leaks unless purged across all libraries.
- Built strictly disjoint splits: `timstof_holdout` (`1,000` `enveda-180` molecules), `np_examples_purged` (`250` NP molecules), and 5-fold `train_cv` (`273,039` molecules).

## 2. Candidate-Recall Bottleneck Audit

### Natural-Product Catalog Coverage (`COCONUT + ExtLib + ChEBI/LMSD`, 701,694 unique structures)
- **`np_examples_purged` (250 timsTOF NPs)**: COCONUT=94.4%, ChEBI/LMSD=90.8%, Union=95.2% (12 Regime C missing).
- **`np_examples_purged` +/-5 ppm Recall**: Uncalibrated=`0.9000` vs `-1.50 ppm` timsTOF calibrated=`0.9160` (given oracle: `0.9454` -> `0.9622`).
- **`np_examples_purged` +/-10 ppm**: Recall=`0.9400` (`0.9874` given oracle), median pool=`53`, p95=`242`.
- **`blind_test` (`test.parquet`, 400 molecules) on NP Catalog**: +/-10 ppm median pool=`33` (p95=`135`, zero-candidate=`2`); +/-25 ppm zero-candidate=`0`.
- **`timstof_holdout` (1,000 `enveda-180` synthetic/screening molecules)**: `0.0%` in COCONUT/ExtLib (confirming `enveda-180` is non-NP chemical space); `100.0%` on extended catalog (`+1000` molecules), +/-10 ppm Recall=`1.0000`, median pool=`129`.

## 3. Regime-Stratified Leak-Free Baselines (+/-15 ppm)

| Benchmark Cohort | Regime | N | Mass-Only MRR@25 | Spectral Baseline MRR@25 | R@1 | R@5 | R@25 | R@100 | R@1000 | 95% Paired Bootstrap CI |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `np_examples_purged` | Regime B/C | 250 | 0.2077 | **0.4893** | 0.3520 | 0.6600 | 0.8440 | 0.9320 | 0.9400 | [+0.2248, +0.3354] |
| `np_examples_purged` (Cross-Inst Ref) | Regime A | 250 | 0.2077 | **0.6706** | 0.5520 | 0.8200 | 0.9240 | 0.9400 | 0.9400 | [+0.4118, +0.5116] |
| `timstof_holdout` (Extended Pool) | Regime B/C | 1000 | 0.1233 | **0.6295** | 0.5050 | 0.7770 | 0.9110 | 0.9780 | 1.0000 | [+0.4818, +0.5312] |
