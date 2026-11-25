# V44 Candidate-Specific Library Confidence Report

- **5-Fold OOF Metrics (`train_cv`, 1477 queries)**: AUC=`0.9594`, Brier=`0.0076`, LogLoss=`0.0324`

## Out-of-Sample Evaluation on `np_examples_purged` (250 timsTOF NPs)

### Mixed 50% Regime A / 50% Regime B (Candidate AUC=`0.8339`, Brier=`0.0096`)
- Naive `LIB_MAX >= 0.87`: locked=`36`, correct=`23`, precision=`0.6389`, Regime-B false locks=`8`, MRR@25=`0.4394`
- V44 Candidate `P_lib >= 0.70`: locked=`121`, correct=`59`, precision=`0.4876`, Regime-B false locks=`43`, MRR@25=`0.4579`

### Pure Regime A (Candidate AUC=`0.9648`, Brier=`0.0052`)
- Naive `LIB_MAX >= 0.87` MRR@25=`0.7694` (precision=`0.8429`)
- V44 Candidate MRR@25=`0.7760`, R@1=`0.6880`, R@5=`0.8880`, R@25=`0.9400` (precision=`0.7733`)
