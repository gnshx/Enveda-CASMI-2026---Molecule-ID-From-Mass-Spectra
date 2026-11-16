# 🛑 FAILURE_ANALYSIS.md
# Comprehensive Forensic Autopsy of Past Regressions & System Errors
**Project**: Enveda CASMI 2026 — Molecule Identification From Mass Spectra  
**Document**: Post-Mortem Analysis & Defensive Engineering Protocols  

---

## 1. Executive Summary of Failure Modes

Across 31 versions, our research lab encountered four critical classes of failures:
1. **Decoy Contamination** (Version 25: $0.480 \to 0.472$)
2. **Rank-1 Demotion / Premature Rescoring** (Versions 26 & 27: $0.480 \to 0.470 \to 0.467$)
3. **Execution Timeout & Container Crash** (Versions 28 & 29: System Error / 7-Day Queue Lock)
4. **Pure Neural Resolution Ceiling** (Phase 1: Plateau at $0.145$)

Below is the scientific autopsy for each failure mode and the architectural safeguards instituted to permanently prevent recurrence.

---

## 2. Autopsy 1: Version 25 LOTUS Decoy Contamination ($0.480 \to 0.472$)

### The Incident
In Version 25, we integrated 150,000 additional natural product structures from the LOTUS database to expand candidate pool coverage. Instead of increasing MRR, the public score dropped by **$-0.008$** (from `0.4800` to `0.4720`).

### Root-Cause Forensic Diagnosis
1. **Unfiltered Isomer Injection**: LOTUS contains many rare, poorly-characterized natural products that share identical molecular formulas with common metabolites.
2. **Decoy Displacement**: Because our neural scorer had a high false-positive rate on macrocycles and glycosides with similar fingerprint bits, the uncurated LOTUS molecules received inflated neural scores.
3. **Slot Stealing**: These false-positive "decoys" displaced the true common metabolites from ranks 1–5 into ranks 10–25 or out of the top 25 entirely. Under MRR, moving a true molecule from rank 1 to rank 3 drops its score contribution by $66.7\%$ ($1.000 \to 0.333$).

### Institutional Safeguard
- **Curated Multi-Tier Database Gating**:
  - Never add candidate structures without an associated citation/occurrence frequency prior (`POP_MU`).
  - Candidates from secondary databases (LOTUS, unverified natural products) are restricted to **backfill only** (ranks 15–25) unless supported by an exact precursor mass match within $\pm 2.0\text{ ppm}$ and at least one high-confidence fragment explanation.

---

## 3. Autopsy 2: Versions 26 & 27 Rank-1 Demotion ($0.480 \to 0.470 \to 0.467$)

### The Incident
In Version 26 and 27, we implemented a dynamic neural consensus blend using a linear weighting parameter $w_{\text{top}} = 0.25$ applied uniformly across all candidates, including top reference library matches. The score degraded from `0.4800` to `0.4700` (V26) and `0.4670` (V27).

### Root-Cause Forensic Diagnosis
1. **The Rank-1 Shield Violation**: In Version 24 (PB `0.4800`), if an external library match achieved cosine similarity $\ge 0.85$, its rank was **locked at Rank 1** ($w_{\text{top}} = 0.00$).
2. **Neural Overconfidence**: When $w_{\text{top}} = 0.25$ was introduced, the neural network's subtle biases on difficult spectra slightly penalized the exact library match, demoting it to rank 2 or 3 in favor of a structurally similar homolog.
3. **Metric Math**: Flipping just 4 correct rank-1 predictions to rank 2 across 400 test queries causes an immediate metric loss of:
   $$\Delta \text{MRR} = \frac{4 \times (0.500 - 1.000)}{400} = \frac{-2.000}{400} = -0.0050$$

### Institutional Safeguard
- **The Absolute Rank-1 Shield**:
  ```python
  # Permanent Architecture Rule
  if extlib_cosine >= 0.85 and shared_peaks >= 4:
      w_top = 0.00  # ZERO neural perturbation allowed
      candidate_ranks[0] = extlib_match  # Permanently locked at Rank 1
  ```

---

## 4. Autopsy 3: Versions 28 & 29 Container Timeout (9.0-Hour Crash)

### The Incident
Versions 28 and 29 were submitted on November 9, 2026. Both submissions remained locked in `SubmissionStatus.PENDING` on Kaggle for 7 consecutive days, blocking further evaluations until Version 31 flushed the queue. Both failed runs ultimately reported `SubmissionStatus.ERROR`.

### Root-Cause Forensic Diagnosis
1. **The 9.0-Hour Hard Ceiling**: Kaggle enforces a strict 9.0-hour wall-clock limit on rerun execution for code competitions. If a notebook runs for 9 hours and 1 second, the Kubernetes pod is abruptly terminated via `SIGKILL` without flushing error traces or partial output.
2. **Double-Pass Outsider Neural Scoring**:
   - The public 0.465 baseline requires ~7.5 hours on Kaggle's single Tesla T4 GPU.
   - In V28/V29, we added a secondary neural inference pass for "outside candidates" (candidates not in the pre-indexed database) without a time budget check.
   - This secondary pass added ~2.0 hours of inference time:
     $$\text{Total Runtime} = 7.5\text{ h} + 2.0\text{ h} = 9.5\text{ hours} > 9.0\text{ hours}$$
   - The container crashed at the 9.0-hour mark, stranding the worker process in Kaggle's backend queue.

### Institutional Safeguard
- **Strict Time-Guarded Single-Pass Inference**:
  ```python
  TIME_GUARD_HOURS = 7.5  # Hard safety margin: 1.5 hours below Kaggle limit
  start_time = time.time()
  
  # Single pass only:
  OUTSIDERS_OK = score_outside_candidates(needed=True, max_time_seconds=3600)
  
  if (time.time() - start_time) / 3600.0 > TIME_GUARD_HOURS:
      logger.warning("Approaching safety threshold! Aborting optional passes.")
      finalize_submission_and_exit()
  ```

---

## 5. Autopsy 4: Phase 1 Pure Neural Resolution Ceiling (Plateau at 0.145)

### The Incident
During Phase 1 (Versions 1–10), we spent 10 iterations scaling convolutional and transformer neural networks predicting 2,048-bit Morgan fingerprints. Despite increasing parameter count from 2M to 35M, public MRR plateaued at `0.1450`.

### Root-Cause Forensic Diagnosis
1. **The Fingerprint Resolution Limit**:
   - Mass spectra do not encode exact Cartesian coordinates; they encode fragmentation pathways.
   - Two positional regio-isomers (e.g., ortho- vs para-substituted aromatics) produce nearly identical fragment mass spectra and have Tanimoto similarity $> 0.90$.
   - A neural network predicting continuous bit probabilities cannot decide which isomer is correct; it predicts intermediate probabilities for both, leading to random ranking between the two.
2. **The Paradigm Shift**:
   - Pure neural inverse modeling is insufficient. Winning CASMI requires combining neural retrieval with **physical forward bond cleavage validation (MetFrag)** and **exact spectral library matching (ExtLib)**.

---

## 6. Summary of Architectural Iron Rules

| Rule # | Name | Formulation | Enforcement |
|:---:|---|---|---|
| **Rule 1** | **Rank-1 Shield** | $w_{\text{top}} = 0.00$ whenever ExtLib cosine $\ge 0.85$ and $k \ge 4$. | Hardcoded in ranking pipeline |
| **Rule 2** | **7.5-Hour Safety Guard** | Total notebook execution aborts all optional passes at $t \ge 7.5\text{ h}$. | Checked at every query batch |
| **Rule 3** | **Skeletal Deduplication** | All candidate lists deduplicated on `Chem.MolToInchiKey(tautomer)[:14]`. | Applied in post-processing |
| **Rule 4** | **Decoy Formula Gating** | External candidates from secondary DBs must match precursor mass within $\pm 2.0\text{ ppm}$. | Pre-filter stage |
| **Rule 5** | **Precursor Mass Prior** | Apply $-1.5\text{ ppm}$ instrument calibration offset to all query $m/z$. | Initial query normalization |
