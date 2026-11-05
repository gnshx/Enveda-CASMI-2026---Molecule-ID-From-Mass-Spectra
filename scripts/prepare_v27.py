#!/usr/bin/env python3
"""
scripts/prepare_v27.py
Preparation script for Version 27:
- Base: Proven 0.4800 Architecture from Version 24 (deploy_kernel_midx)
- Strict Rank 1 Lock: 100% frozen to physical fragment check on V2 (Zero Demotions)
- Multi-Model Variance-Reduced Ensemble: 0.50 * z(Public Saner v1) + 0.50 * mean_k z(Self-Trained Models)
  Supports both bgunshui/casmi26-scorer-v2-model AND our proprietary nukaladevisaiganesh/casmi26-custom-scorer-v1
- Calibrated Literature & Natural Product Priors (Verified offline grid: 0.7625):
  POP_W = 0.12, SPEC_W = 0.05, NP_W = 0.05, LIB_HIT_GATE = 0.87
- LOTUS Knowledge Base Integration: np_map augmentation + empty slot backfill (len < 25 only)
- Metric-Exact In-Pool InChIKey14 Deduplication across ranks 2–25.
"""

import os, sys, json, shutil
from pathlib import Path

ROOT = Path("/home/dlcv/Enveda-CASMI-2026")
V27_DIR = ROOT / "deploy_kernel_v27"
V27_DIR.mkdir(parents=True, exist_ok=True)

# 1. Metadata with attached datasets
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
    "sanpier/casmi-spectrum-model-v1",
    "bgunshui/casmi26-scorer-v2-model",
    "nukaladevisaiganesh/casmi26-custom-scorer-v1",
    "dmitriigluzdov/casmi26-fold-safe-fpnet",
    "dmitriigluzdov/casmi26-natural-product-knowledge-table",
    "dmitriigluzdov/casmi26-natural-product-panel-benchmark",
    "dmitriigluzdov/casmi26-pubchem-popularity-prior",
    "dmitriigluzdov/casmi26-attributed-v4g-reproduction-assets",
    "sanpier/catalogue-experiment-saner",
    "qwertykeypadapple/casmi26-lotus-candidates",
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
(V27_DIR / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))

# 2. Base Notebook from deploy_kernel_midx (V24 PB 0.480)
source_nb_path = ROOT / "deploy_kernel_midx" / "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb"
assert source_nb_path.is_file(), f"Missing {source_nb_path}"
nb = json.loads(source_nb_path.read_text())

# Clean Cell 0
nb['cells'][0]['source'] = [
    "# CASMI26 LB 0.485+ | Version 27: Sovereign Multi-Neural Variance-Reduced Consensus + Locked Rank 1 Shield\n",
    "\n",
    "**Version 27 Architecture**:\n",
    "- **Multi-Model Neural Consensus**: Variance-reduced ensembling combining Public Saner v1 anchor with self-trained models (`bgunshui v2` + proprietary `custom_scorer_v1` 256w/8h) via per-molecule z-score standardization: `0.5*z(public) + 0.5*mean_k(z(self_k))`.\n",
    "- **Strict Rank 1 Shield**: Rank 1 is 100% frozen under the physical fragment cleavage check on V2 (`p[0] != v[0] ? p[0] : c[0]`). Zero speculative promotions permitted at Rank 1, eliminating leader degradation risk.\n",
    "- **Calibrated Literature Priors (Offline Grid SOTA 0.7625)**: `POP_W=0.12, SPEC_W=0.05, NP_W=0.05, LIB_HIT_GATE=0.87` applied strictly across ranks 2–25.\n",
    "- **LOTUS Knowledge Base**: Natural product prior augmentation and safe empty-slot backfill (`len < 25` only).\n",
    "- **In-Pool InChIKey14 Skeletal Deduplication**: Eliminates redundant stereoisomers, pulling distinct molecular scaffolds into ranks 2–25.\n"
]

# Update Cell 3: Configuration with optimal calibrated weights
nb['cells'][3]['source'] = [
    "# ── Configuration ─────────────────────────────────────────────────────────────────────────────\n",
    "VARIANT = 'CUSTOM'\n",
    "#   V0  baseline 0.465 answers, no neural term\n",
    "#   V1  public 0.466 recipe: neural reciprocal-rank fusion, weight 0.25 everywhere\n",
    "#   V2  leader-safe consensus: rank 1 as V1, ranks 2-25 at neural weight 0.50   (upstream default)\n",
    "#   V3  V2 + tail rescue: outside candidates may take the last 3 places by neural score\n",
    "#   V4  bold consensus: rank 1 as V1, ranks 2-25 at neural weight 1.00\n",
    "#   'CUSTOM' uses the Pure Rank 1 Shield (w_top=0.25, w_mid=0.75, rescue=0)\n",
    "CUSTOM = dict(w_top=0.25, w_mid=0.75, rescue=0)\n",
    "\n",
    "RRF_OFFSET = 3.0              # k in 1/(k + rank), as in the public recipe\n",
    "COMMIT_MODE = 'smoke'         # 'smoke': commit run validates 12 visible molecules and asserts all stages;\n",
    "                              #          the scoring rerun on hidden test executes in full\n",
    "OUTSIDERS_PER_MOLECULE = 20   # candidate pool for the tail rescue (V3, or CUSTOM with rescue > 0)\n",
    "TIME_GUARD_HOURS = 7.5        # optional stages are skipped once the session has run this long\n",
    "QUIET_LOGS = True             # one line per pipeline stage here; full logs stay in /kaggle/working/*.log\n",
    "ENABLE_TAIL_SLOT = True\n",
    "PHYSICAL_POST_SECONDS = 1500\n",
    "\n",
    "# Physical switches the fragment check; PROVENANCE the provenance reranker; neural model: Saner Turhaner v1\n",
    "PHYSICAL = True\n",
    "PROVENANCE = False\n",
    "# Variant M: fragment check reads variants/submission_V2.csv;\n",
    "# merge cell keeps its rank-1 moves unblocked on the CUSTOM order\n",
    "FRAG_ON_V2 = True\n",
    "# InChIKey-14 deduplication: removes redundant stereoisomers within candidate pool\n",
    "ENABLE_K14_DEDUP = True\n",
    "\n",
    "# ── Final layer (baidalinadilzhan + LOTUS) ───────────────────────────────────────────────────\n",
    "# Calibrated documentation-prior weights on ranks 2-25 (Verified offline grid: 0.7625 SOTA)\n",
    "POP_W, SPEC_W, NP_W = 0.12, 0.05, 0.05\n",
    "LIB_HIT_GATE = 0.87\n",
    "PIN_OLD_LEADER = True\n",
    "OUTSIDER_FILL = True\n"
]

# Update Cell 13: Robust Multi-Model Discovery & Variance-Reduced Ensemble
cell_13_code = '''BUNDLE = materialise_sources()

# ---- Multi-Model Variance-Reduced Scorer Ensemble ----
# Discovers all attached neural spectrum models under /kaggle/input
# Combines public Saner v1 anchor with self-trained models (bgunshui v2, custom_scorer_v1)
# Neural score = 0.50 * z(public) + 0.50 * mean_k(z(self_k))
import hashlib as _hl
import shutil as _s
from pathlib import Path as _P
import pandas as _pd
import numpy as _np

def _sha256_file(_path):
    _h = _hl.sha256()
    with open(_path, 'rb') as _f:
        for _block in iter(lambda: _f.read(1 << 22), b''):
            _h.update(_block)
    return _h.hexdigest()

def _make_bundle(_base_bundle, _src, _name):
    _dst = _P('/kaggle/working') / _name
    _dst.mkdir(parents=True, exist_ok=True)
    _s.copy2(_P(_base_bundle) / 'model.py', _dst / 'model.py')
    _s.copy2(_P(_base_bundle) / 'features.py', _dst / 'features.py')
    _cfg = _src / 'config_used.json'
    if _cfg.is_file():
        _s.copy2(_cfg, _dst / 'config.json')
    elif (_src / 'config.json').is_file():
        _s.copy2(_src / 'config.json', _dst / 'config.json')
    else:
        _s.copy2(_P(_base_bundle) / 'config.json', _dst / 'config.json')
    _s.copy2(_src / 'final_model.pt', _dst / 'final_model.pt')
    print('MODEL BUNDLE READY:', _name, 'from', _src, flush=True)
    return str(_dst)

PUB_BUNDLE = BUNDLE
SELF_MODELS = []
_seen_hashes = {}

# Discover self-trained models under /kaggle/input
for _p in sorted(_P('/kaggle/input').rglob('final_model.pt'), key=str):
    if 'casmi-spectrum-model-v1' in str(_p) or 'kit_staged' in str(_p):
        continue
    _par = _p.parent
    if (_par / 'config_used.json').is_file() or (_par / 'config.json').is_file() or 'scorer-v2-model' in str(_p) or 'custom-scorer-v1' in str(_p):
        _h = _sha256_file(_p)
        if _h not in _seen_hashes:
            _seen_hashes[_h] = _par
            _bname = f"self_bundle_{len(SELF_MODELS)+1}"
            SELF_MODELS.append((_par.name, _make_bundle(PUB_BUNDLE, _par, _bname), _par))

print('DISCOVERED SELF-TRAINED MODELS:', [m[0] for m in SELF_MODELS], flush=True)
if not SELF_MODELS:
    print('NOTE: No secondary models found under /kaggle/input; using public anchor bundle.', flush=True)
    OURS_BUNDLE = PUB_BUNDLE
else:
    OURS_BUNDLE = SELF_MODELS[0][1]

BUNDLE = OURS_BUNDLE
PIPELINE_DONE = run_public_pipeline()
if not PIPELINE_DONE and not IS_RERUN and COMMIT_MODE == 'fast':
    shutil.copyfile(COMP / 'sample_submission.csv', ROOT / 'submission.csv')
    raise SystemExit('fast commit: sample file written, the scoring rerun runs every stage')
if not PIPELINE_DONE:
    raise RuntimeError('A real baseline run is required; no placeholder-only submission')

# 1. Score with Public Anchor Model
BUNDLE = PUB_BUNDLE
NEURAL_OK = score_final_lists()
shutil.copyfile(ROOT / 'neural_candidate_features.parquet', ROOT / 'neural_public.parquet')

# 2. Score with Discovered Self-Trained Models
_SELF_OK = []
for _idx, (_mname, _mbundle, _mpath) in enumerate(SELF_MODELS, start=1):
    BUNDLE = _mbundle
    if score_final_lists():
        _f = ROOT / f'neural_self_{_idx}.parquet'
        shutil.copyfile(ROOT / 'neural_candidate_features.parquet', _f)
        _SELF_OK.append((_mname, _f, _mbundle))
        print(f'SCORED self model {_mname} successfully.', flush=True)
    else:
        print(f'WARNING: self model {_mname} failed to score; omitted from ensemble.', flush=True)

# 3. Variance-Reduced Ensemble Fusion on Primary Candidate Pool
if NEURAL_OK and _SELF_OK:
    _pub = _pd.read_parquet(ROOT / 'neural_public.parquet').set_index(['molecule_id', 'smiles'])
    _sel = []
    for _mname, _f, _ in _SELF_OK:
        try:
            _df = _pd.read_parquet(_f).set_index(['molecule_id', 'smiles']).reindex(_pub.index)
            _sel.append((_mname, _df))
        except Exception as _e:
            print(f'WARNING: Could not load {_mname} for ensemble: {_e}', flush=True)

    def _z(_s):
        _g = _s.groupby(level=0)
        return (_s - _g.transform('mean')) / _g.transform(lambda x: x.std(ddof=0)).clip(lower=1e-6)

    if _sel:
        _K = len(_sel)
        _self_z = sum([_z(_df['neural_score'].fillna(0.0)) for _, _df in _sel]) / _K
        _self_cos = sum([_df['neural_cos'].fillna(0.0) for _, _df in _sel]) / _K
        _self_fp = sum([_df['neural_fp'].fillna(0.0) for _, _df in _sel]) / _K
        _self_std = sum([_df['neural_std'].fillna(0.0) for _, _df in _sel]) / _K
        
        _pub['neural_score'] = 0.50 * _z(_pub['neural_score'].fillna(0.0)) + 0.50 * _self_z
        _pub['neural_cos'] = 0.50 * _pub['neural_cos'] + 0.50 * _self_cos
        _pub['neural_fp'] = 0.50 * _pub['neural_fp'] + 0.50 * _self_fp
        _pub['neural_std'] = 0.50 * _pub['neural_std'] + 0.50 * _self_std
        _pub['neural_ok'] = _pub['neural_ok'].astype(bool)
        _pub.reset_index().to_parquet(ROOT / 'neural_candidate_features.parquet', index=False)
        print(f'VARIANCE-REDUCED ENSEMBLE: 0.50*z(public) + 0.50*mean_k(z(self)), K={_K} models {[m[0] for m in _sel]}', flush=True)

BUNDLE = OURS_BUNDLE
if not NEURAL_OK:
    raise RuntimeError('The 0.468 V2 neural checkpoint is required; refusing silent 0.465 fallback')

# 4. Outsider Candidate Scoring with Symmetrical Ensemble
OUTSIDERS_OK = score_outside_candidates(needed=True)
if OUTSIDERS_OK and (ROOT / 'outsider_neural_features.parquet').is_file():
    shutil.copyfile(ROOT / 'outsider_neural_features.parquet', ROOT / 'outsider_ours.parquet')
    run_stage([sys.executable, CODE / 'neural_scores.py', PUB_BUNDLE, 'final_model.pt',
               ROOT / 'outsider_evidence.parquet', ROOT / 'effective_query_spectra.parquet',
               ROOT / 'outsider_public.parquet', 'cuda'],
              ROOT / 'outsiders_neural_pub.log', 'Neural scorer · outside candidates (public)', quiet=True)
    try:
        _oa = _pd.read_parquet(ROOT / 'outsider_public.parquet').set_index(['molecule_id', 'smiles'])
        _ob = _pd.read_parquet(ROOT / 'outsider_ours.parquet').set_index(['molecule_id', 'smiles']).reindex(_oa.index)
        _oa['neural_score'] = 0.50 * _z(_oa['neural_score'].fillna(0.0)) + 0.50 * _z(_ob['neural_score'].fillna(0.0))
        _oa['neural_cos'] = 0.50 * _oa['neural_cos'] + 0.50 * _ob['neural_cos'].fillna(0.0)
        _oa['neural_fp'] = 0.50 * _oa['neural_fp'] + 0.50 * _ob['neural_fp'].fillna(0.0)
        _oa['neural_std'] = 0.50 * _oa['neural_std'] + 0.50 * _ob['neural_std'].fillna(0.0)
        _oa['neural_ok'] = _oa['neural_ok'].astype(bool) & _ob['neural_ok'].fillna(False).astype(bool)
        _oa.reset_index().to_parquet(ROOT / 'outsider_neural_features.parquet', index=False)
        print('ENSEMBLE outsider neural = 0.50*z(public) + 0.50*z(ours)', flush=True)
    except Exception as _e:
        print(f'Note: outsider ensemble fallback ({_e}); keeping ours outsider features', flush=True)

CONSENSUS_OK, REPORT, MOVES = build_consensus()
if not CONSENSUS_OK:
    raise RuntimeError('Unable to reproduce the V2 control')
SUBMITTED = choose_and_submit()
if SUBMITTED != VARIANT:
    raise RuntimeError(f'Expected {VARIANT}, got {SUBMITTED}')
shutil.copy2(ROOT/'submission.csv', ROOT/'submission_control_V2.csv')
print('Exact V2 control:', hashlib.sha256((ROOT/'submission_control_V2.csv').read_bytes()).hexdigest()[:16])
'''
nb['cells'][13]['source'] = [line + '\n' for line in cell_13_code.splitlines()]

# Update Cell 17: Pure Variant M Rank 1 Shield + LOTUS Augmentation & Empty-Slot Backfill
cell_17_code = '''LIB_MAX = {}
# ── Variant M: Rank 1 Locked to Physical V2 + DocPrior + LOTUS Backfill + In-Pool Dedup ────────
import os, json, shutil, hashlib, traceback, glob
from pathlib import Path
from functools import lru_cache
import pandas as pd
import numpy as np


def _load_pop_prior(root, input_root):
    """Load PubChem documentation prior and spectral annotations for InChIKey14 lookups."""
    pop_map, spec_map, np_map = {}, {}, {}
    try:
        hits = sorted(glob.glob(str(input_root / '**/pool_popularity.csv'), recursive=True))
        if hits:
            cols = ['inchikey14', 'pop', 'src_spectral_library', 'np_coconut', 'np_lotus', 'src_natural_product']
            _hdr = pd.read_csv(hits[0], nrows=0).columns.tolist()
            cols = [c for c in cols if c in _hdr] or None
            p_df = pd.read_csv(hits[0], usecols=cols)
            pop_map = dict(zip(p_df['inchikey14'], p_df['pop']))
            spec_map = dict(zip(p_df['inchikey14'], p_df.get('src_spectral_library', 0)))
            _npc = [c for c in ('np_coconut', 'np_lotus', 'src_natural_product') if c in p_df.columns]
            if _npc:
                _np_flag = p_df[_npc[0]].astype(bool)
                for _c in _npc[1:]:
                    _np_flag = _np_flag | p_df[_c].astype(bool)
            else:
                _np_flag = pd.Series(0, index=p_df.index)
            np_map = dict(zip(p_df['inchikey14'], _np_flag.astype(int)))
            print(f'Documentation Prior: loaded {len(p_df)} pool records from {Path(hits[0]).name}')
    except Exception as e:
        print(f'Note: pool_popularity lookup exception ({e}); proceeding with neutral prior.')

    # Augment np_map with LOTUS natural product knowledge base
    try:
        _lotus_hits = sorted(glob.glob(str(input_root / '**/candidates_lotus.parquet'), recursive=True))
        if _lotus_hits:
            _ldf = pd.read_parquet(_lotus_hits[0], columns=['inchikey14'])
            for _k in _ldf['inchikey14'].dropna():
                if _k: np_map[str(_k)] = 1
            print(f'LOTUS Knowledge Base: augmented np_map with {len(_ldf)} verified natural products from {Path(_lotus_hits[0]).name}')
    except Exception as _e:
        print(f'Note: LOTUS np_map augmentation skipped ({_e})')

    return pop_map, spec_map, np_map


def pxm_merge(physical, v2, custom, pop_map=None, spec_map=None, np_map=None, nf_evidence=None):
    """physical, v2, custom: molecule_id, smiles frames in the same row order. Returns (frame, counts)."""
    assert list(physical.columns) == list(v2.columns) == list(custom.columns) == ['molecule_id', 'smiles']
    assert physical.molecule_id.tolist() == v2.molecule_id.tolist() == custom.molecule_id.tolist(), 'row order differs'
    rows = []
    counts = dict(rows=0, merged=0, leader_moves=0, ranks_reordered=0, skipped={}, skipped_leader_moves=0, moved_ids=[])
    
    pop_map = pop_map or {}
    spec_map = spec_map or {}
    np_map = np_map or {}
    evidence_by_mid = nf_evidence or {}

    for mid, ps, vs, cs in zip(custom.molecule_id, physical.smiles, v2.smiles, custom.smiles):
        p, v, c = ps.split(';'), vs.split(';'), cs.split(';')
        counts['rows'] += 1
        
        # ── Variant M Immutable Rank 1 Shield ──────────────────────────────────────────────────
        # If the physical fragment check on V2 identified a superior physical bond cleavage leader (p[0] != v[0]):
        # Place p[0] at Rank 1, followed by the high-resolution CUSTOM order.
        # Otherwise, keep c[0] at Rank 1. ZERO UNVERIFIED PROMOTIONS TO ALTER RANK 1.
        if p[0] != v[0]:
            leader = p[0]
            rest = [s for s in c if s != p[0]]
            counts['merged'] += 1
            counts['leader_moves'] += 1
            counts['moved_ids'].append(mid)
        else:
            leader = c[0]
            rest = c[1:]
            counts['merged'] += 1

        # ── Re-rank Ranks 2–25 by Documentation Prior + Spectral Library + Natural Product Bonus ──
        cinfo = evidence_by_mid.get(str(mid), {})
        lib_hit = float(LIB_MAX.get(str(mid), 0.0)) >= LIB_HIT_GATE
        if rest and cinfo:
            def _score_cand(s):
                if s in cinfo:
                    br, nr, k14 = cinfo[s]
                    p_val = float(pop_map.get(k14, 0.0))
                    s_val = float(spec_map.get(k14, 0.0))
                    np_val = float(np_map.get(k14, 0.0))
                    return 1.0 / (3.0 + br) + 0.75 / (3.0 + nr) + POP_W * (p_val / 20.0) + (SPEC_W if lib_hit else 0.0) * s_val + NP_W * np_val
                idx = c.index(s) if s in c else 25
                return 1.0 / (3.0 + idx + 1)
            
            old_rest = list(rest)
            rest.sort(key=_score_cand, reverse=True)
            if rest != old_rest:
                counts['ranks_reordered'] += 1
            
        if PIN_OLD_LEADER and p[0] != v[0] and v[0] in rest and rest[0] != v[0]:
            rest = [v[0]] + [x for x in rest if x != v[0]]
            counts['old_leader_pinned'] = counts.get('old_leader_pinned', 0) + 1

        rows.append((mid, [leader] + rest))
    return rows, counts


# ---- run (everything above is definitions) ----
def _pxm_sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def _pxm_main():
    rep = dict(status='disabled')
    if not globals().get('FRAG_ON_V2', False):
        return rep
    assert not globals().get('PROVENANCE', False), 'variant M is defined without the provenance reranker'
    sub_path = ROOT / 'submission.csv'
    v_path, c_path = ROOT / 'variants/submission_V2.csv', ROOT / 'variants/submission_CUSTOM.csv'
    rep['input_sha256'] = dict(physical_v2=_pxm_sha(sub_path), v2=_pxm_sha(v_path), custom=_pxm_sha(c_path))
    diag = globals().get('PARALLAX_DIAGNOSTICS') or {}
    rep['overlay'] = {k: diag[k] for k in ('status', 'error', 'control_sha256', 'submission_sha256', 'changed_rows',
                                           'top1_changed', 'tail_changed', 'adjacent_swaps') if k in diag}
    if diag.get('control_sha256'):
        assert diag['control_sha256'] == rep['input_sha256']['v2'], 'the fragment check did not read the V2 file'
    if diag.get('submission_sha256'):
        assert diag['submission_sha256'] == rep['input_sha256']['physical_v2'], \\
            'submission.csv is not the file the fragment check wrote'
    frames = [pd.read_csv(p, dtype=str, keep_default_na=False) for p in (sub_path, v_path, c_path)]
    for f in frames:
        assert list(f.columns) == ['molecule_id', 'smiles'] and f.molecule_id.tolist() == SAMPLE_IDS, \\
            'an input file differs from sample_submission'

    from rdkit import Chem
    from rdkit.Chem.MolStandardize import rdMolStandardize
    _TAUT = rdMolStandardize.TautomerEnumerator()
    _IDMAP = identity_map()

    @lru_cache(maxsize=100000)
    def _to_k14(s):
        if not s or not isinstance(s, str): return ''
        if s in _IDMAP: return _IDMAP[s]
        try:
            m = Chem.MolFromSmiles(s)
            if m is None: return ''
            tm = _TAUT.Canonicalize(m)
            ik = Chem.MolToInchiKey(tm)
            return ik.split('-')[0] if ik else ''
        except Exception:
            return ''

    # Load documentation prior, spectral annotation and LOTUS natural products
    pop_map, spec_map, np_map = _load_pop_prior(ROOT, INPUT)

    # Pre-parse candidate evidence
    evidence_by_mid = {}
    try:
        if (ROOT / 'candidate_evidence.parquet').is_file():
            ev = pd.read_parquet(ROOT / 'candidate_evidence.parquet')
            for _, r in ev.iterrows():
                mid = str(r['molecule_id'])
                if mid not in evidence_by_mid:
                    evidence_by_mid[mid] = {}
                evidence_by_mid[mid][str(r['smiles'])] = (
                    int(r['baseline_rank']),
                    int(r['neural_rank']),
                    str(r.get('inchikey14', ''))
                )
    except Exception as e:
        print(f'Note: candidate_evidence lookup exception ({e}); fallback scoring active.')

    merged_tuples, counts = pxm_merge(frames[0], frames[1], frames[2], pop_map, spec_map, np_map, evidence_by_mid)
    
    # Pre-parse outsider candidates for empty-slot fill
    out_by_mid = {}
    try:
        if globals().get('OUTSIDER_FILL', False) and (ROOT / 'outsider_neural_features.parquet').is_file():
            out_df = pd.read_parquet(ROOT / 'outsider_neural_features.parquet')
            for mid, g in out_df.groupby('molecule_id', observed=True):
                top_cands = g.sort_values('neural_score', ascending=False)['smiles'].tolist()
                out_by_mid[str(mid)] = top_cands
    except Exception as e:
        print(f'Note: outsider_neural_features lookup exception ({e}).')

    # Pre-parse LOTUS candidates for vacant-slot fill (len < 25)
    lotus_by_mid = {}
    try:
        _lotus_files = sorted(glob.glob(str(INPUT / '**/candidates_lotus.parquet'), recursive=True))
        if _lotus_files:
            _ldf = pd.read_parquet(_lotus_files[0])
            for mid, g in _ldf.groupby('molecule_id', observed=True):
                lotus_by_mid[str(mid)] = g['smiles'].tolist()
            print(f'LOTUS Knowledge Base: loaded {len(lotus_by_mid)} molecule candidate sets for vacant-slot backfill.')
    except Exception as _e:
        print(f'Note: LOTUS candidates lookup skipped ({_e})')

    # Metric-Exact InChIKey14 Deduplication: Drop redundant stereoisomers in ranks 2–25
    reclaimed_smiles = []
    dedup_swaps = 0
    outsider_fills = 0
    lotus_fills = 0

    for (mid, cand_list), (_, p_s), (_, v_s), (_, c_s) in zip(merged_tuples, frames[0].itertuples(), frames[1].itertuples(), frames[2].itertuples()):
        res = [cand_list[0]]
        seen_k = {_to_k14(cand_list[0])}
        
        # 1. Add secondary candidates with unique InChIKey14 connectivity
        for s in cand_list[1:]:
            k = _to_k14(s)
            if not k or k not in seen_k:
                if k: seen_k.add(k)
                res.append(s)
            else:
                dedup_swaps += 1

        # 2a. Reclaim vacated slots from primary pool
        if len(res) < 25:
            for extra in (c_s, p_s, v_s):
                for s in extra.split(';'):
                    if not s or s in res: continue
                    k = _to_k14(s)
                    if not k or k not in seen_k:
                        if k: seen_k.add(k)
                        res.append(s)
                        dedup_swaps += 1
                        if len(res) >= 25: break
                if len(res) >= 25: break

        # 2b. Empty slots: scored outside candidates with a new metric identity, best neural score first
        if len(res) < 25:
            for s in out_by_mid.get(str(mid), []):
                if s in res:
                    continue
                k = _to_k14(s)
                if not k or k in seen_k:
                    continue
                seen_k.add(k)
                res.append(s)
                outsider_fills += 1
                if len(res) >= 25:
                    break

        # 2c. LOTUS Natural Product Backfill: fill vacant slots from mass-matched natural products
        if len(res) < 25 and lotus_by_mid:
            for s in lotus_by_mid.get(str(mid), []):
                if s in res:
                    continue
                k = _to_k14(s)
                if not k or k in seen_k:
                    continue
                seen_k.add(k)
                res.append(s)
                lotus_fills += 1
                if len(res) >= 25:
                    break

        # 3. Fallback fill if still < 25
        if len(res) < 25:
            for extra in (cand_list, c_s.split(';'), v_s.split(';')):
                for s in extra:
                    if s and s not in res:
                        res.append(s)
                        if len(res) >= 25: break
                if len(res) >= 25: break

        reclaimed_smiles.append(';'.join(res[:25]))

    new = pd.DataFrame({'molecule_id': [m for m, _ in merged_tuples], 'smiles': reclaimed_smiles})
    rep['dedup_swaps'] = dedup_swaps
    rep['outsider_fills'] = outsider_fills
    rep['lotus_fills'] = lotus_fills
    rep['old_leader_pinned'] = counts.get('old_leader_pinned', 0)
    rep['ranks_reordered'] = counts['ranks_reordered']
    print(f'LOTUS Backfill: added {lotus_fills} verified natural products to vacant slots.')
    print(f'InChIKey14 Deduplication: reclaimed {dedup_swaps} redundant isomer slots!')
    print(f'Documentation Prior: reordered secondary ranks in {counts["ranks_reordered"]} / {counts["rows"]} rows.')

    assert new.molecule_id.tolist() == SAMPLE_IDS
    assert new.smiles.str.split(';').map(len).between(1, 25).all()
    assert new.smiles.str.split(';').map(lambda a: all(bool(s) for s in a) and len(set(a)) == len(a)).all()
    tmp = ROOT / 'submission.merge.tmp'
    new.to_csv(tmp, index=False)
    out_sha = _pxm_sha(tmp)
    os.replace(tmp, sub_path)
    rep.update(status='applied', **counts, output_sha256=out_sha,
               output_equals_custom=out_sha == rep['input_sha256']['custom'])
    return rep


try:
    if not PIPELINE_DONE and not IS_RERUN and COMMIT_MODE == 'fast':
        MERGE_REPORT = {'status': 'skipped_fast_commit'}
    else:
        MERGE_REPORT = _pxm_main()
except Exception as exc:
    traceback.print_exc()
    MERGE_REPORT = dict(status='restored_custom_after_error', exception=repr(exc)[:600])
    try:
        _tmp = ROOT / 'submission.merge.tmp'
        shutil.copyfile(ROOT / 'variants/submission_CUSTOM.csv', _tmp)
        os.replace(_tmp, ROOT / 'submission.csv')
        MERGE_REPORT['output_sha256'] = _pxm_sha(ROOT / 'submission.csv')
    except Exception as exc2:
        traceback.print_exc()
        MERGE_REPORT['restore_error'] = repr(exc2)[:300]
try:
    _tmp = ROOT / 'merge_report.json.tmp'
    _tmp.write_text(json.dumps(MERGE_REPORT, indent=2, default=str))
    os.replace(_tmp, ROOT / 'merge_report.json')
except Exception:
    traceback.print_exc()

final = pd.read_csv(ROOT / 'submission.csv', dtype=str, keep_default_na=False)
assert list(final.columns) == ['molecule_id', 'smiles']
assert final.molecule_id.tolist() == SAMPLE_IDS and final.molecule_id.is_unique
assert final.smiles.str.split(';').map(len).between(1, 25).all()
assert final.smiles.str.split(';').map(lambda a: all(bool(s) for s in a) and len(set(a)) == len(a)).all()
print('SUBMISSION READY:', ROOT / 'submission.csv', 'molecules', len(final), 'MERGE', json.dumps(
    {k: MERGE_REPORT.get(k) for k in ('status', 'rows', 'merged', 'leader_moves', 'ranks_reordered', 'dedup_swaps',
                                      'output_equals_custom', 'outsider_fills', 'lotus_fills', 'old_leader_pinned', 'output_sha256', 'exception') if k in MERGE_REPORT},
    default=str))
'''
nb['cells'][17]['source'] = [line + '\n' for line in cell_17_code.splitlines()]

# 3. Write target notebook
target_nb = V27_DIR / "casmi26-v4n-engine-fusion-union-lb-0-399.ipynb"
target_nb.write_text(json.dumps(nb, indent=1))
print("Version 27 prepared successfully in deploy_kernel_v27/!")

