"""
scripts/local_benchmark_rerank.py
Local benchmark evaluation simulating re-ranking strategies on the 250 Natural Products benchmark panel.
Tests:
  1. Pure Baseline
  2. Popularity & Natural Product weighting (POP_W, NP_W)
  3. MSI Level-1 Library Leader Promotion (LIB_HIT_GATE)
  4. Natural Product Promotion Margin (NP_PROMO_MARGIN)
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
import numpy as np
from validation.metrics import mrr_at_25

def evaluate_strategy(cands_df, pop_w=0.0, np_w=0.0, spec_w=0.0, lib_gate=0.80, np_promo_margin=0.0):
    scenarios = cands_df['scenario'].unique()
    results = {}
    
    for sc in scenarios:
        df_sc = cands_df[cands_df['scenario'] == sc].copy()
        gt = {}
        preds = {}
        
        for mol, g in df_sc.groupby('molecule', observed=True):
            gt[str(mol)] = str(mol)
            
            # Compute composite score
            # Base rank reciprocal
            base_term = 1.0 / (3.0 + g['rank'])
            neural_term = 0.75 * (g['ranker_score'] - g['ranker_score'].mean()) / (g['ranker_score'].std() + 1e-6)
            pop_term = pop_w * (g['popularity'] / 20.0)
            is_np = (g['pool_source'] > 0).astype(float)
            np_term = np_w * is_np
            lib_max = g['lib_max'].iloc[0] if len(g) > 0 else 0.0
            spec_term = (spec_w if lib_max >= lib_gate else 0.0) * (g['lib_max'] >= 0.70).astype(float)
            
            composite = base_term + neural_term + pop_term + np_term + spec_term
            g = g.assign(comp_score=composite)
            
            # Sort by composite score
            ordered = g.sort_values('comp_score', ascending=False)
            ranked_keys = ordered['candidate_key'].tolist()
            
            # MSI Level-1 Promotion
            if lib_max >= lib_gate and len(ranked_keys) > 1:
                top_cand = ranked_keys[0]
                runner_up = ranked_keys[1]
                # If runner up is a verified library hit but leader is not
                cand_is_lib = bool(ordered.iloc[1]['lib_max'] >= 0.80)
                lead_is_lib = bool(ordered.iloc[0]['lib_max'] >= 0.80)
                if cand_is_lib and not lead_is_lib:
                    ranked_keys[0], ranked_keys[1] = ranked_keys[1], ranked_keys[0]
            
            # Calibrated Natural Product Promotion
            elif np_promo_margin > 0.0 and len(ranked_keys) > 1 and lib_max < 0.80:
                lead_np = bool(is_np.loc[ordered.index[0]])
                cand_np = bool(is_np.loc[ordered.index[1]])
                sc_diff = ordered.iloc[1]['comp_score'] - ordered.iloc[0]['comp_score']
                if cand_np and not lead_np and sc_diff > -np_promo_margin:
                    ranked_keys[0], ranked_keys[1] = ranked_keys[1], ranked_keys[0]
            
            preds[str(mol)] = ranked_keys[:25]
            
        results[str(sc)] = mrr_at_25(preds, gt)
        
    return results

if __name__ == '__main__':
    cands = pd.read_parquet('data/benchmark/main_candidates.parquet')
    print('=== Running Local Benchmark Evaluation on 250 Natural Products ===\n')
    
    # 1. Baseline
    base_res = evaluate_strategy(cands, pop_w=0.0, np_w=0.0, spec_w=0.0)
    print('1. Pure Baseline MRR@25:')
    for sc, score in base_res.items():
        print(f'   - {sc:32s}: {score:.4f}')
    print(f'   => Mean MRR: {np.mean(list(base_res.values())):.4f}\n')
    
    # 2. Balanced Tri-Force Configuration (POP_W=0.12, NP_W=0.12, SPEC_W=0.12, LIB_GATE=0.75, NP_PROMO=0.048)
    opt_res = evaluate_strategy(cands, pop_w=0.12, np_w=0.12, spec_w=0.12, lib_gate=0.75, np_promo_margin=0.048)
    print('2. Balanced Tri-Force + MSI-Level 1 + NP Promotion MRR@25:')
    for sc, score in opt_res.items():
        diff = score - base_res[sc]
        print(f'   - {sc:32s}: {score:.4f} (Δ {diff:+.4f})')
    print(f'   => Mean MRR: {np.mean(list(opt_res.values())):.4f} (Δ {np.mean(list(opt_res.values())) - np.mean(list(base_res.values())):+.4f})')
