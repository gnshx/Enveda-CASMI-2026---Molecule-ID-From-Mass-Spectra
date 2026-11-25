# V49 Conditional Candidate Rescue Report (Zero Oracle Leakage)

- **Evaluated Cohort**: `np_examples_purged` (`250` timsTOF NPs; `15` missing from `±15 ppm` DB pool)
- **Low-Confidence Queries Triggered**: `67` (`751` blind regioisomers generated from top-3 candidates)
- **Blind Recovery of Missing-from-DB Targets**: `0 / 15`
- **Baseline MRR@25 / R@25**: `0.4495` / `0.8680`
- **Empty-Slot Only Rescue MRR@25 / R@25**: `0.4495` / `0.8680` (delta `+0.0000`)
- **Tail (Ranks 21-25) Rescue MRR@25 / R@25**: `0.4493` / `0.8640` (delta `-0.0002`, CI `[-0.0005, +0.0000]`)
