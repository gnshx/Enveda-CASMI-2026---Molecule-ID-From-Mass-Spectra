# Phase 1 Forensic Audit (V24–V41 -> V42 Control)

## 1. Notebook Lineage & Configuration Findings

Direct cell-by-cell comparison against the `0.4800` Personal Best (`deploy_kernel_midx`, V24):

| Kernel Dir | Cells Differing vs V24 | `CUSTOM` Config | `POP_W, SPEC_W, NP_W` | Verified Observation |
|---|---|---|---|---|
| `deploy_kernel_midx` (V24) | Baseline | `w_top=0.25, w_mid=0.75, rescue=0` | `0.10, 0.05, 0.02` | Comment in Cell 3 claimed `w_top=0.00`, actual code used `w_top=0.25`. |
| `deploy_kernel_v34` | `[0, 3]` | `w_top=0.25, w_mid=0.75, rescue=0` | `0.10, 0.05, 0.02` | Executable cells identical to V24 (`0.4800` LB). |
| `deploy_kernel_v35` | `[0, 3, 17]` | `w_top=0.25, w_mid=0.75, rescue=0` | `0.10, 0.05, 0.02` | Added Phase 29 molecular networking under `try/except` (`0.4800` LB). |
| `deploy_kernel_v36` | `[0, 3, 17]` | `w_top=0.25, w_mid=0.75, rescue=0` | `0.10, 0.05, 0.02` | Identical Cell 0/3 config to V35 (`0.4800` LB). |
| `deploy_kernel_v37` | `[0, 3, 17]` | `w_top=0.75, w_mid=0.90, rescue=0` | `0.20, 0.08, 0.05` | Unlocked Rank-1 promotion in Cell 17 when `LIB_MAX < 0.87`. |
| `deploy_kernel_v40` | `[0, 3, 17]` | `w_top=0.80, w_mid=0.90, rescue=0` | `0.35, 0.10, 0.15` | Cell 0 header still labeled `Version 39`. |
| `deploy_kernel_v41` | `[0, 3, 17]` | `w_top=0.90, w_mid=0.90, rescue=0` | `0.40, 0.15, 0.15` | Comment claimed `w_top=0.00`; Cell 17 silently restores `submission_CUSTOM.csv` on exception. |

## 2. Code Bugs Fixed in V42

1. **Bug 1 (V41 config vs shield comment & V24 control restoration)**: Built `deploy_kernel_v42/` via `scripts/build_v42.py` directly from `deploy_kernel_midx/` (exact V24 `0.4800` PB) with accurate comments.
2. **Bug 2 (Oracle ground-truth injection in Regime C benchmark)**: Updated `scripts/benchmark_regime_c_mrr_impact.py` to remove `candidate_pool = [true_smi] + isomers` and removed isomer generation from `true_smi`; isomers are now generated strictly from retrieved same-formula analog seeds (`analog_seeds`).
3. **Bug 3 (Raw SMILES sliced as `cand[:14]` in evaluator)**: Updated `validation/metrics.py` (`normalize_to_inchikey14`, `reciprocal_rank`, `recall_at_k`) to distinguish InChIKeys from SMILES and canonicalize SMILES via `smiles_to_inchikey14` instead of slicing `cand[:14]`. Added `paired_bootstrap_ci`.
4. **Bug 4 (`LIB_MAX` query-level vs candidate-level confidence)**: Documented in V42 audit; `LIB_MAX` in Cell 17 aggregates `library_max` per `molecule_id` rather than per candidate. Candidate-specific library evidence will replace this in V44.
5. **Bug 5 (Silent fallbacks & unverified execution)**: Added `v42_execution_audit.json` in `deploy_kernel_v42` (Cells 13 & 17) logging `_swap_to_ours` status, model checkpoint SHA-256 hashes, stage output hashes, and aborting if `MERGE_REPORT['status']` falls back after an exception.
6. **Candidate SMILES length filter**: Removed `len(smi) < 12 or len(smi) > 100` in `candidates/extract_pubchem_test_candidates.py` and fixed hardcoded `/home/dlcv/Enveda-CASMI-2026` paths in both `extract_pubchem_test_candidates.py` and `benchmark_regime_c_mrr_impact.py`.

## 3. External Data & License Audit Item (Open)

- `DATA_PROVENANCE.md` lists `Takumu Hata Reference Library` (`Public Kaggle Dataset / NIST & MassBank / CC-BY 4.0`). Any NIST-derived subset requires prize-eligibility verification before final submission selection.
