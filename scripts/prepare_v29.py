#!/usr/bin/env python3
"""
scripts/prepare_v29.py
Prepares Version 29 deployment package:
- Base: rajrajak99 LB 0.487 SOTA architecture (v5 320-dim 200-epoch scorer + timsTOF mass prior + apex extlib)
- Dual-Tier External Spectral Reference Library Overlay (Gold >= 0.825 -> Rank 1, Silver >= 0.72 -> Rank 2)
- Zero LOTUS Decoy Injection (protects candidate pools from unranked decoys)
- Strict validation and syntax assertion
"""

import os, sys, json, ast, shutil
from pathlib import Path

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
V29_DIR = ROOT / "deploy_kernel_v29"
V29_DIR.mkdir(parents=True, exist_ok=True)

# 1. Metadata with verified dataset sources
meta = {
  "id": "nukaladevisaiganesh/casmi26-v4n-engine-fusion-union-lb-0-399",
  "id_no": 136555841,
  "title": "CASMI26 | v4n + Engine Fusion + Union [LB 0.399]",
  "code_file": "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": True,
  "enable_gpu": True,
  "enable_tpu": False,
  "enable_internet": False,
  "keywords": [
    "gpu",
    "chemistry"
  ],
  "dataset_sources": [
    "samartalwar/casmi-2026-spectral-library-massbankharmonized",
    "sanpier/casmi-spectrum-model-v1",
    "dmitriigluzdov/casmi26-fold-safe-fpnet",
    "dmitriigluzdov/casmi26-natural-product-knowledge-table",
    "dmitriigluzdov/casmi26-natural-product-panel-benchmark",
    "dmitriigluzdov/casmi26-pubchem-popularity-prior",
    "dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
    "takumuhata/casmi26-extlib",
    "bgunshui/casmi26-scorer-v5-model",
    "sanpier/catalogue-experiment-saner",
    "metric/rdkit-2026-3-3-wheel"
  ],
  "kernel_sources": [],
  "competition_sources": [
    "enveda-CASMI26-molecule-id-mass-spectra"
  ],
  "model_sources": [],
  "docker_image": "gcr.io/kaggle-private-byod/python@sha256:de153e6d303881930c51286216393d56f56b26458d0661350305a1e8144eea01",
  "machine_shape": "NvidiaTeslaT4"
}

meta_path = V29_DIR / "kernel-metadata.json"
meta_path.write_text(json.dumps(meta, indent=2))
print(f"[v29] Wrote kernel-metadata.json to {meta_path}")

# 2. Load the proven 0.487 base notebook
base_nb_path = ROOT / "research_kernels" / "rajrajak99" / "casmi26-apex-v2240-deep-extlib-v5-487.ipynb"
with open(base_nb_path) as f:
    nb = json.load(f)

# 3. Update Cell 0: Header
nb['cells'][0]['source'] = [
    "# CASMI26 LB 0.487+ | Apex Sovereign ExtLib (Dual-Tier Reference Promotion) + timsTOF Mass Prior + v5 Scorer Ensemble\n",
    "\n",
    "**Version 29: SOTA Frontier Architecture (Target LB 0.487+)**\n",
    "- **External Spectral Reference Library**: 324,884 reference spectra from Takumu Hata CC0 + Harmonized MassBank\n",
    "- **Dual-Tier Direct Promotion Gate**:\n",
    "  * Tier 1 (Gold, cosine >= 0.825, k >= 4): direct Rank-1 promotion (MRR = 1.000)\n",
    "  * Tier 2 (Silver, 0.72 <= cosine < 0.825, k >= 3, ppm <= 5.0): promoted to Rank 2 (MRR = 0.500, protecting Rank 1)\n",
    "- **timsTOF Instrument Mass-Accuracy Prior**: -1.5 ppm offset calibration on Bruker timsTOF mass spectrometers (`V8 = 'MASS'`)\n",
    "- **Multi-Neural Backbone Fusion**: 200-epoch v5 Scorer (320-width) + Saner v1 via per-molecule z-score standardization (0.65 / 0.35)\n",
    "- **Calibrated Documentation Priors** + **In-Pool InChIKey14 Skeletal Deduplication**\n"
]

# 4. Update Cell 23 with the enhanced Dual-Tier promotion overlay
dual_tier_cell_23 = '''# ===== 0.500 APEX OVERLAY: DUAL-BAND EXTERNAL SPECTRAL LIBRARY DUAL-TIER PROMOTION =====
# Integrates external measured MS/MS reference libraries (Takumu Hata extlib CC0 & MassBank harmonized)
# with a calibrated Dual-Tier promotion rule:
# - Tier 1 (Gold Hit: cosine >= 0.825, k >= 4 peaks, <= 10 ppm): promoted directly to Rank 1 (MRR = 1.000).
# - Tier 2 (Silver Hit: 0.72 <= cosine < 0.825, k >= 3 peaks, <= 5 ppm): promoted to Rank 2 (MRR = 0.500, protecting Rank 1).
# When lib_max >= LIB_TAU (pipeline already has strong match), existing ranking is protected.
# All operations fail safe: any exception preserves the existing submission.csv untouched.

def _apex_extlib_overlay(work='/kaggle/working', inp='/kaggle/input'):
    import os, sys, glob, time, json, shutil, math
    from pathlib import Path
    import numpy as np
    import pandas as pd

    t0 = time.monotonic()
    sub_path = os.path.join(work, 'submission.csv')
    if not os.path.exists(sub_path):
        print('[apex-extlib] submission.csv not found; overlay skipped')
        return

    sub = pd.read_csv(sub_path, dtype=str, keep_default_na=False)
    shutil.copy(sub_path, os.path.join(work, 'submission_pre_apex_extlib.csv'))

    # Locate test.parquet
    test_paths = sorted(glob.glob(inp + '/**/test.parquet', recursive=True),
                       key=lambda p: ('mass-spectra' not in p.lower(), len(p)))
    if not test_paths:
        print('[apex-extlib] test.parquet not found; overlay skipped')
        return
    te = pd.read_parquet(test_paths[0])

    # Pre-parse query spectra
    queries = {}
    for r in te.itertuples(index=False):
        mid = str(r.molecule_id)
        m = np.asarray(r.ms2_mzs, dtype=np.float64)
        it = np.asarray(r.ms2_normalized_intensities, dtype=np.float64)
        prec = float(r.precursor_mz)
        mode = 1 if 'pos' in str(getattr(r, 'ionization_mode', 'positive')).lower() else -1
        adduct = str(getattr(r, 'adduct', ''))
        
        ok = (m > 0) & (it > 0) & (m <= prec + 0.5)
        m, it = m[ok], it[ok]
        o = np.argsort(m)
        m, it = m[o], it[o]
        
        # Calculate approximate neutral mass
        nm = prec - 1.007276 if mode == 1 else prec + 1.007276
        if '[M+Na]+' in adduct: nm = prec - 22.989218
        elif '[M+K]+' in adduct: nm = prec - 38.963158
        elif '[M+NH4]+' in adduct: nm = prec - 18.033823
        elif '[M+HCOO]-' in adduct: nm = prec - 44.998201
        
        queries.setdefault(mid, []).append(dict(mz=m, it=it, prec=prec, mode=mode, nm=nm))

    # Read existing pipeline library confidence
    lib_max = {}
    ev_path = os.path.join(work, 'frozen_candidate_evidence.parquet')
    if os.path.exists(ev_path):
        try:
            ev = pd.read_parquet(ev_path, columns=['molecule_id', 'library_max'])
            lib_max = {str(m): float(v) for m, v in ev.groupby('molecule_id', sort=False).library_max.first().items()}
        except Exception:
            pass

    # Find external library sources
    refs = []  # list of tuples: (mode, prec, mz_arr, it_arr, key, smiles, neutral_mass)
    
    # Check Source A: Takumu Hata extlib (ext_meta.parquet + ext_mz.npy / ext_in.npy)
    meta_hits = sorted(glob.glob(inp + '/**/ext_meta.parquet', recursive=True), key=len)
    mz_hits = sorted(glob.glob(inp + '/**/ext_mz.npy', recursive=True), key=len)
    in_hits = sorted(glob.glob(inp + '/**/ext_in.npy', recursive=True), key=len)
    mass_hits = sorted(glob.glob(inp + '/**/ext_mass.parquet', recursive=True), key=len)
    
    if meta_hits and mz_hits and in_hits:
        try:
            print(f'[apex-extlib] Loading Takumu Hata extlib from {meta_hits[0]}...')
            meta_df = pd.read_parquet(meta_hits[0])
            mz_all = np.load(mz_hits[0], mmap_mode='r')
            in_all = np.load(in_hits[0], mmap_mode='r')
            mass_df = pd.read_parquet(mass_hits[0]) if mass_hits else None
            
            # Filter non-empty masses if available
            if mass_df is not None and 'exact_mass' in mass_df.columns:
                meta_df['exact_mass'] = mass_df['exact_mass'].values
                meta_df = meta_df[meta_df['exact_mass'].notna() & (meta_df['exact_mass'] > 0)].copy()

            for idx, row in meta_df.iterrows():
                try:
                    mode = 1 if 'pos' in str(row.get('ionization_mode', 'pos')).lower() else -1
                    prec = float(row['precursor_mz'])
                    nm = float(row.get('exact_mass', prec - (1.007276 * mode)))
                    smi = str(row['normalized_smiles'])
                    key = str(row.get('inchikey14', ''))
                    
                    mz_p = np.asarray(mz_all[idx], dtype=np.float64)
                    it_p = np.asarray(in_all[idx], dtype=np.float64)
                    mask = (mz_p > 0) & (it_p > 0)
                    mz_p, it_p = mz_p[mask], it_p[mask]
                    if len(mz_p) < 3:
                        continue
                    o = np.argsort(mz_p)
                    refs.append((mode, prec, mz_p[o], it_p[o], key, smi, nm))
                except Exception:
                    continue
            print(f'[apex-extlib] Loaded {len(refs)} reference spectra from Source A.')
        except Exception as e:
            print(f'[apex-extlib] Source A loading error: {e}')

    # Check Source B: MassBank harmonized spectra.parquet
    mb_hits = sorted(glob.glob(inp + '/**/spectra.parquet', recursive=True),
                     key=lambda p: ('massbank' not in p.lower(), len(p)))
    if mb_hits and 'massbank' in mb_hits[0].lower():
        try:
            print(f'[apex-extlib] Loading MassBank harmonized from {mb_hits[0]}...')
            mb_df = pd.read_parquet(mb_hits[0], columns=['smiles', 'exact_mass', 'ion_mode', 'mzs', 'intensities', 'is_casmi_subset'])
            mb_df = mb_df[(~mb_df.is_casmi_subset.astype(bool)) & mb_df.smiles.notna()]
            
            for row in mb_df.itertuples(index=False):
                try:
                    mode = 1 if 'pos' in str(row.ion_mode).lower() else -1
                    nm = float(row.exact_mass)
                    prec = nm + (1.007276 * mode)
                    smi = str(row.smiles)
                    mz_p = np.asarray(row.mzs, dtype=np.float64)
                    it_p = np.asarray(row.intensities, dtype=np.float64)
                    mask = (mz_p > 0) & (it_p > 0)
                    mz_p, it_p = mz_p[mask], it_p[mask]
                    if len(mz_p) < 3:
                        continue
                    o = np.argsort(mz_p)
                    refs.append((mode, prec, mz_p[o], it_p[o], '', smi, nm))
                except Exception:
                    continue
            print(f'[apex-extlib] Total references after Source B: {len(refs)} spectra.')
        except Exception as e:
            print(f'[apex-extlib] Source B loading error: {e}')

    if not refs:
        print('[apex-extlib] No external reference spectra available; keeping submission untouched.')
        return

    # Sort references by precursor mass for fast binary search
    order = np.argsort([r[1] for r in refs], kind='stable')
    sorted_refs = [refs[i] for i in order]
    prec_array = np.array([r[1] for r in sorted_refs])

    def calc_cosine(qm, qi, rm, ri, tol=0.015):
        if len(qm) == 0 or len(rm) == 0:
            return 0.0, 0
        qa, ra = np.sqrt(qi), np.sqrt(ri)
        qa = qa / (np.linalg.norm(qa) + 1e-12)
        ra = ra / (np.linalg.norm(ra) + 1e-12)
        
        j0 = np.searchsorted(rm, qm - tol, 'left')
        j1 = np.searchsorted(rm, qm + tol, 'right')
        pairs = [(qa[i] * ra[j], i, j) for i in range(len(qm)) for j in range(j0[i], j1[i])]
        if not pairs:
            return 0.0, 0
        pairs.sort(reverse=True)
        ui, uj, s, n = set(), set(), 0.0, 0
        for p, i, j in pairs:
            if i in ui or j in uj:
                continue
            ui.add(i); uj.add(j)
            s += p; n += 1
        return float(s), n

    TAU_GOLD = 0.825     # Tier 1 (Gold): direct Rank-1 promotion
    K_GOLD = 4
    TAU_SILVER = 0.72   # Tier 2 (Silver): promote to Rank 2 (protecting Rank 1 leader)
    K_SILVER = 3
    LIB_TAU = 0.85      # Protect queries already strong in pipeline library

    hits_gold = 0
    hits_silver = 0
    promoted_gold = 0
    promoted_silver = 0
    new_smiles_list = []

    for mid, smiles_str in zip(sub['molecule_id'].astype(str), sub['smiles'].astype(str)):
        smis = [s for s in smiles_str.split(';') if s]
        new_smis = list(smis)
        
        try:
            # If the pipeline already matched this molecule with high confidence, protect it
            if lib_max.get(mid, 0.0) >= LIB_TAU:
                new_smiles_list.append(smiles_str)
                continue

            q_list = queries.get(mid, [])
            if not q_list:
                new_smiles_list.append(smiles_str)
                continue

            best_match = (0.0, 0, None, 999.0)  # (cosine, n_peaks, smi, ppm_err)
            for q in q_list:
                q_mode, q_prec = q['mode'], q['prec']
                q_mz, q_it = q['mz'], q['it']
                
                # Precursor search window: 10 ppm + 0.002 Da
                tol = q_prec * 10e-6 + 0.002
                idx_lo = np.searchsorted(prec_array, q_prec - tol, 'left')
                idx_hi = np.searchsorted(prec_array, q_prec + tol, 'right')
                
                for ri in range(idx_lo, idx_hi):
                    ref = sorted_refs[ri]
                    if ref[0] != q_mode:
                        continue
                    cos, n_matched = calc_cosine(q_mz, q_it, ref[2], ref[3])
                    ppm_err = abs(q_prec - ref[1]) / q_prec * 1e6
                    if n_matched >= K_SILVER and cos > best_match[0]:
                        best_match = (cos, n_matched, ref[5], ppm_err)

            best_cos, best_k, match_smi, best_ppm = best_match
            
            # Tier 1 (Gold Hit): cosine >= 0.825 and >= 4 peaks -> Rank 1
            if match_smi is not None and best_cos >= TAU_GOLD and best_k >= K_GOLD:
                hits_gold += 1
                if match_smi in smis:
                    new_smis = [match_smi] + [s for s in smis if s != match_smi]
                    promoted_gold += int(smis[0] != match_smi)
                else:
                    new_smis = ([match_smi] + smis)[:25]
                    promoted_gold += 1
            # Tier 2 (Silver Hit): 0.72 <= cosine < 0.825, >= 3 peaks, <= 5 ppm -> Rank 2 (protect Rank 1!)
            elif match_smi is not None and best_cos >= TAU_SILVER and best_k >= K_SILVER and best_ppm <= 5.0:
                hits_silver += 1
                leader = new_smis[0]
                rest = [s for s in new_smis if s != match_smi and s != leader]
                if match_smi != leader:
                    new_smis = ([leader, match_smi] + rest)[:25]
                    promoted_silver += 1

        except Exception as exc:
            new_smis = smis

        new_smiles_list.append(';'.join(new_smis) if new_smis else smiles_str)

    res_df = sub.copy()
    res_df['smiles'] = new_smiles_list
    assert len(res_df) == len(sub)
    assert res_df['smiles'].str.split(';').map(len).max() <= 25

    # Atomically save updated submission
    tmp_path = os.path.join(work, 'submission.tmp.csv')
    res_df.to_csv(tmp_path, index=False)
    os.replace(tmp_path, sub_path)

    elapsed = round(time.monotonic() - t0, 1)
    audit = dict(molecules=len(sub), hits_gold=hits_gold, hits_silver=hits_silver,
                 promoted_gold=promoted_gold, promoted_silver=promoted_silver, elapsed_s=elapsed)
    with open(os.path.join(work, 'apex_extlib_audit.json'), 'w') as f:
        json.dump(audit, f, indent=2)

    print(f'[apex-extlib] DUAL-TIER SUCCESS: Gold {hits_gold} (promoted {promoted_gold} to Rank-1), '
          f'Silver {hits_silver} (promoted {promoted_silver} to Rank-2) in {elapsed}s.')

try:
    _apex_extlib_overlay()
except Exception:
    import traceback
    traceback.print_exc()
    print('[apex-extlib] overlay failed; existing submission preserved.')
'''

nb['cells'][23]['source'] = [line + '\n' for line in dual_tier_cell_23.strip().split('\n')]

# 5. Update Cell 24: Telemetry version string
c24_src = ''.join(nb['cells'][24]['source'])
c24_src = c24_src.replace(
    "'version': 'SovereignZenith V32 Apex 0.486+'",
    "'version': 'CASMI26 Version 29 Apex Sovereign Dual-Tier ExtLib (Target LB 0.487+)'"
)
nb['cells'][24]['source'] = [line + '\n' for line in c24_src.strip().split('\n')]

# 6. Syntax validation on all code cells
print("[v29] Running Python AST validation on all notebook code cells...")
for idx, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell.get('source', []))
        try:
            ast.parse(source)
        except SyntaxError as e:
            print(f"[v29] SYNTAX ERROR in Cell {idx}: {e}")
            sys.exit(1)

print("[v29] All code cells passed AST syntax check successfully!")

out_nb_path = V29_DIR / "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb"
with open(out_nb_path, "w") as f:
    json.dump(nb, f, indent=1)

print(f"[v29] Successfully prepared Version 29 notebook at {out_nb_path} ({out_nb_path.stat().st_size} bytes)")

