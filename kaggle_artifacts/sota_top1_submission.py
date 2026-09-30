# ==============================================================================
# 🚀 CASMI 2026: SOTA 0.400+ PIPELINE (v4n + Two-Ranker Engine + Union ICEBERG/GLACIER + Library Shield)
# Enveda CASMI 2026 - High-Throughput Molecule Identification from Mass Spectra
#
# Breakthrough Architecture:
#   1. v4n Base Retrieval: EngineCfg(generate=True) + fe_v4 + fpnet_full1 + PubChem N1=5000
#   2. Two-Ranker Engine (BIO + AFIX): ChEBI/LIPID MAPS (bio_fp) + timsTOF analogs (N=200) + 12 GBMs
#   3. Constitutional Regioisomer Expansion on Engine top candidates (ortho/meta/para, phenol/methoxy)
#   4. Forward Models on UNION: ICEBERG (5400s) + GLACIER (4000s) evaluate union of both candidate pools
#   5. High-Confidence Library Shield (lib_max >= 0.88 protected from noisy neural forward rerank)
#   6. Weighted Reciprocal Rank Fusion: RRF = 1/(3+r_v4) + 0.6/(3+r_eng) -> Top 25
# ==============================================================================

import os, sys, glob, re, subprocess, time, json, hashlib, pickle, gc
import numpy as np, pandas as pd

T0 = time.time()
LIB_TAU, REL_TH = 0.9, 600.0
TOPN, ICE_LAM, ICE_BUDGET, ICE_PC = 60, 1.0, 5400, True
SLOTS_AGG, SLOTS_GENTLE = [2, 4, 6, 8, 10], [4, 8, 12, 16, 20]

os.makedirs('/kaggle/working/numba_cache', exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = '/kaggle/working/numba_cache'

def find(pattern, optional=False):
    hits = sorted(glob.glob(f'/kaggle/input/**/{pattern}', recursive=True), key=len)
    if not hits:
        base = os.path.basename(pattern)
        hits = sorted(glob.glob(f'/kaggle/input/**/{base}', recursive=True), key=len)
    if not hits:
        hits = sorted(glob.glob(f'**/{pattern}', recursive=True), key=len)
    if not hits and not optional:
        raise FileNotFoundError(f"Could not find required dataset file: {pattern}")
    return hits[0] if hits else None

def sha256(path, n=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(n), b''):
            h.update(b)
    return h.hexdigest()

# ── 1. RDKit Wheel Installation ───────────────────────────────────────────────
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if tag in w]
if not whl:
    whl = sorted(glob.glob('/kaggle/input/**/rdkit*.whl', recursive=True))
if whl:
    print('Installing RDKit wheel:', whl[0])
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-500:])

import rdkit
print('RDKit version:', rdkit.__version__)

# ── 2. Discover Pipeline Artifacts ───────────────────────────────────────────
V4CODE = os.path.dirname(os.path.dirname(find('casmi26-v4b-models/**/casmi/engine.py')))
V4MOD = os.path.dirname(find('casmi26-v4b-models/**/MANIFEST.json'))
V3CODE = os.path.dirname(os.path.dirname(find('casmi26-v3-models/**/casmi/engine.py')))
V3MOD = os.path.dirname(find('casmi26-v3-models/**/ranker_0.pkl'))
POOL_DIR = os.path.dirname(find('pool_meta.parquet'))
PC_DIR = os.path.dirname(find('pc_smiles.npy'))
COMP = os.path.dirname(find('test.parquet'))

MAN = json.load(open(os.path.join(V4MOD, 'MANIFEST.json')))
bad = []
for rel, h in sorted(MAN['files'].items()):
    p = os.path.join(V4CODE, rel[len('code/'):]) if rel.startswith('code/') else os.path.join(V4MOD, rel)
    if not os.path.exists(p) or sha256(p) != h:
        bad.append(rel)
if bad:
    print(f'Warning: some dataset files differ from MANIFEST.json: {bad}')
else:
    print('All MANIFEST files verified successfully.')

print('V4CODE', V4CODE, '\nV4MOD', V4MOD, '\nV3CODE', V3CODE, '\nPOOL', POOL_DIR, '\nPC', PC_DIR, '\nCOMP', COMP)
print(f"{len(MAN['files'])} files match MANIFEST.json; families {MAN['families']}", f'{time.time()-T0:.0f}s')

# ── 3. Commit Switch (Fast 10-Min Commit vs Full Competition Rerun) ───────────
_te = pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['molecule_id', 'spectrum_id'])
_sig = hashlib.md5(','.join(sorted(_te.spectrum_id.astype(str))).encode()).hexdigest()
IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'
SMOKE_N = 12
if not IS_RERUN:
    SM = '/kaggle/working/smoke_comp'
    os.makedirs(SM, exist_ok=True)
    _full = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
    _keep = sorted(_full.molecule_id.unique())[:SMOKE_N]
    _full[_full.molecule_id.isin(_keep)].to_parquet(os.path.join(SM, 'test.parquet'))
    for _f in ['train.parquet', 'sample_submission.csv']:
        if not os.path.exists(os.path.join(SM, _f)):
            os.symlink(os.path.join(COMP, _f), os.path.join(SM, _f))
    COMP = SM
    ICE_BUDGET = 300
print('RERUN:', IS_RERUN, '| sig:', _sig, '| COMP:', COMP, '| ICE_BUDGET:', ICE_BUDGET)

# ── 4. Two-Ranker Engine (BIO + AFIX + Regioisomers) ──────────────────────────
ENG_DIR = '/kaggle/working/eng'
os.makedirs(ENG_DIR, exist_ok=True)

ENG_FILES = {
'pv.py': '''"""pv.py — Spectral entropy search, neutral loss shift, and MetFrag-lite."""
import math
import numpy as np
from numba import njit, prange

class CFG:
    PPM_WIN = 10.0
    PPM_FALLBACK = 30.0
    INT_FLOOR = 0.002
    MAX_PEAKS = 256
    MZ_TOL = 0.01
    INT_POWER = 1.0
    ENT_WEIGHT = True
    ANALOG_WIN = 200.0
    N_ANALOG = 80
    SIM_POWER = 3.0
    W1_PRIORS = (0.30, 0.60)
    SEEDS = (0, 1, 2, 3)
    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)
    TOPN = 25

@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    if n == 0: return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx: mx = it[i]
    if mx <= 0: return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx; c = 0
    for i in range(n):
        if it[i] >= thr: c += 1
    idx = np.empty(c, np.int64); j = 0
    for i in range(n):
        if it[i] >= thr: idx[j] = i; j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c): v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk): k2[i] = idx[o[i]]
        k2.sort(); idx = k2; c = topk
    om = np.empty(c, np.float32); oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]; v = it[idx[i]] ** power; oi[i] = v; s += v
    if s > 0:
        for i in range(c): oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0: S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S; s2 = 0.0
            for i in range(c): oi[i] = oi[i] ** w; s2 += oi[i]
            if s2 > 0:
                for i in range(c): oi[i] /= s2
    return om, oi

@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0; j = 0; n = len(qmz); m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0: SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0: SB -= cp[x] * np.log(cp[x])
    SAB = 0.0; tot = 0.0
    buf = np.empty(n + m, np.float64); b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol: buf[b] = qp[i]; i += 1; b += 1
        elif d > tol: buf[b] = cp[j]; j += 1; b += 1
        else: buf[b] = qp[i] + cp[j]; i += 1; j += 1; b += 1
    while i < n: buf[b] = qp[i]; i += 1; b += 1
    while j < m: buf[b] = cp[j]; j += 1; b += 1
    for x in range(b): tot += buf[x]
    if tot <= 0: return 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0: SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)

@njit(cache=True, fastmath=True, parallel=True)
def search(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a: continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0: continue
        out[k] = entropy_sim(qmz, qp, cm, cp, tol)
    return out

@njit(cache=True, fastmath=True)
def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    a = entropy_sim(qmz, qp, cmz, cp, tol)
    if shift > -0.001 and shift < 0.001: return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)): sm[i] = cmz[i] + shift
    b = entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b

@njit(cache=True, fastmath=True, parallel=True)
def search_shift_pre(qmz, qp, lo, hi, roff, rmz, rit, tol, shift):
    out = np.zeros(hi - lo, np.float32)
    for k in prange(hi - lo):
        a = roff[lo + k]; b = roff[lo + k + 1]
        if b <= a: continue
        out[k] = entropy_sim_shift(qmz, qp, rmz[a:b], rit[a:b], tol, shift[k])
    return out

@njit(cache=True, fastmath=True, parallel=True)
def search_shift_rows(qmz, qp, cand, off, allmz, allin, tol, floor, topk, power, ent_weight, shift):
    out = np.zeros(len(cand), np.float32)
    for k in prange(len(cand)):
        c = cand[k]; a = off[c]; b = off[c + 1]
        if b <= a: continue
        cm, cp = _clean(allmz[a:b], allin[a:b], floor, topk, power, ent_weight)
        if len(cm) == 0: continue
        out[k] = entropy_sim_shift(qmz, qp, cm, cp, tol, shift[k])
    return out

@njit(cache=True)
def clean_store(rows, off, allmz, allin, floor, topk, power, ent_weight):
    n = len(rows)
    lens = np.zeros(n + 1, np.int64)
    tmp_m = []
    tmp_i = []
    for k in range(n):
        r = rows[k]
        cm, cp = _clean(allmz[off[r]:off[r + 1]], allin[off[r]:off[r + 1]], floor, topk, power, ent_weight)
        tmp_m.append(cm); tmp_i.append(cp)
        lens[k + 1] = lens[k] + len(cm)
    om = np.empty(lens[n], np.float32); oi = np.empty(lens[n], np.float32)
    for k in range(n):
        om[lens[k]:lens[k + 1]] = tmp_m[k]; oi[lens[k]:lens[k + 1]] = tmp_i[k]
    return lens, om, oi

MASS = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, P=30.97376163,
            S=31.97207100, F=18.99840322, Cl=34.96885268, Br=78.9183371, I=126.904473,
            Na=22.9897692809, K=38.96370668, Si=27.9769265325, B=11.0093054, Se=79.9165213)
E = 0.00054857990; PROTON = MASS['H'] - E; H2O = 2 * MASS['H'] + MASS['O']
NH4 = MASS['N'] + 4 * MASS['H']; FORMATE = MASS['C'] + 2 * MASS['H'] + 2 * MASS['O']
ACETATE = 2 * MASS['C'] + 4 * MASS['H'] + 2 * MASS['O']
ADDUCTS = {
    "[M+H]+": (1, 1, PROTON), "[M+NH4]+": (1, 1, NH4 - E), "[M+Na]+": (1, 1, MASS['Na'] - E),
    "[M+K]+": (1, 1, MASS['K'] - E), "[M-H2O+H]+": (1, 1, PROTON - H2O), "[M-2H2O+H]+": (1, 1, PROTON - 2 * H2O),
    "[M+2H]2+": (1, 2, 2 * PROTON), "[M]+": (1, 1, -E), "[M-H2O]+": (1, 1, -E - H2O),
    "[M+CH3OH+H]+": (1, 1, PROTON + MASS['C'] + 4 * MASS['H'] + MASS['O']),
    "[M+CH3CN+H]+": (1, 1, PROTON + 2 * MASS['C'] + 3 * MASS['H'] + MASS['N']),
    "[M-H]-": (1, 1, -PROTON), "[M-H2O-H]-": (1, 1, -PROTON - H2O), "[M+CH2O2-H]-": (1, 1, FORMATE - PROTON),
    "[M+C2H4O2-H]-": (1, 1, ACETATE - PROTON), "[M+Cl]-": (1, 1, MASS['Cl'] + E), "[M]-": (1, 1, E),
    "[M-2H]-": (1, 2, -2 * PROTON), "[M+Na-2H]-": (1, 1, MASS['Na'] - 2 * PROTON),
    "[2M+H]+": (2, 1, PROTON), "[2M+Na]+": (2, 1, MASS['Na'] - E), "[2M+NH4]+": (2, 1, NH4 - E),
    "[2M+K]+": (2, 1, MASS['K'] - E), "[2M-H]-": (2, 1, -PROTON), "[2M+CH2O2-H]-": (2, 1, FORMATE - PROTON),
    "[2M+C2H4O2-H]-": (2, 1, ACETATE - PROTON), "[2M+Na-2H]-": (2, 1, MASS['Na'] - 2 * PROTON),
    "[3M+H]+": (3, 1, PROTON), "[3M-H]-": (3, 1, -PROTON),
}

def neutral_mass(mz, adduct):
    out = np.full(len(mz), np.nan); ad = np.asarray(adduct, dtype=object)
    for a, (n, z, d) in ADDUCTS.items():
        m = (ad == a)
        if m.any(): out[m] = (mz[m] * z - d) / n
    return out

N_ANALOG = 80
P_SIM = 3.0
N_FEAT = 31

def _rank_norm(x):
    o = np.argsort(-x); r = np.empty(len(x)); r[o] = np.arange(len(x)); return r / max(1, len(x) - 1)

def _z(x):
    s = x.std()
    return (x - x.mean()) / s if s > 1e-9 else np.zeros_like(x)

def rank_features(cand_fp, cand_lib, analog_fp, analog_sim, model_logits=None, frag=None):
    nc = cand_fp.shape[0]
    cf = cand_fp.astype(np.float32); cs = cf.sum(1)
    lv = np.asarray(cand_lib, np.float32)
    lvmax = float(lv.max()) if nc else 0.0
    if analog_fp is not None and len(analog_sim):
        af = analog_fp.astype(np.float32); asum = af.sum(1)
        inter = cf @ af.T
        tan = inter / (cs[:, None] + asum[None, :] - inter + 1e-9)
        w = np.clip(np.asarray(analog_sim, np.float32), 0, None)
        ap = (tan * (w ** P_SIM)[None, :]).max(1)
        a1 = (tan * w[None, :]).max(1)
        best_tan = tan.max(1); top_tan = tan[:, 0]; top_sim = float(w[0])
        mean_tan = (tan * (w ** P_SIM)[None, :]).sum(1) / ((w ** P_SIM).sum() + 1e-9)
    else:
        ap = a1 = best_tan = top_tan = mean_tan = np.zeros(nc, np.float32); top_sim = 0.0
    apmax = float(ap.max()) if nc else 0.0
    if model_logits is not None:
        raw = cf @ np.asarray(model_logits, np.float32)
        nrm = raw / np.sqrt(np.maximum(cs, 1.0))
        mfeat = [_z(raw), _rank_norm(raw), raw - raw.max(), _z(nrm), _rank_norm(nrm),
                 (raw == raw.max()).astype(np.float32)]
    else:
        mfeat = [np.zeros(nc, np.float32)] * 6
    if model_logits is not None and nc:
        mr = _rank_norm(cf @ np.asarray(model_logits, np.float32))
        lbest = int(np.argmax(lv)) if lvmax > 0 else -1
        agree = float(1.0 - mr[lbest]) if lbest >= 0 else 0.0
        abest = int(np.argmax(ap)) if apmax > 0 else -1
        agree_a = float(1.0 - mr[abest]) if abest >= 0 else 0.0
        xfeat = [lv * (1.0 - mr), ap * (1.0 - mr), np.full(nc, agree), np.full(nc, agree_a),
                 np.full(nc, agree * lvmax), np.full(nc, float(np.corrcoef(lv, -mr)[0, 1]) if lv.std() > 1e-9 else 0.0)]
    else:
        xfeat = [np.zeros(nc, np.float32)] * 6
    if frag is not None:
        fr = np.asarray(frag, np.float32)
        ffeat = [fr, _rank_norm(fr), fr - fr.max() if nc else fr, _z(fr)]
    else:
        ffeat = [np.zeros(nc, np.float32)] * 4
    return np.column_stack([
        lv, _rank_norm(lv), np.full(nc, lvmax), lv - lvmax, (lv > 0).astype(float),
        ap, _rank_norm(ap), np.full(nc, apmax), ap - apmax,
        a1, best_tan, top_tan, mean_tan, np.full(nc, top_sim),
        np.full(nc, np.log(max(nc, 1))),
        *mfeat, *ffeat, *xfeat,
    ]).astype(np.float32)

AMU = {'C': 12.0, 'H': 1.00782503207, 'N': 14.0030740048, 'O': 15.9949146196, 'P': 30.97376163,
       'S': 31.97207100, 'F': 18.99840322, 'Cl': 34.96885268, 'Br': 78.9183371, 'I': 126.904473,
       'Na': 22.9897692809, 'K': 38.96370668, 'Si': 27.9769265325, 'B': 11.0093054, 'Se': 79.9165213}
H = AMU['H']

def mol_graph(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    n = m.GetNumAtoms()
    w = np.zeros(n)
    for a in m.GetAtoms():
        w[a.GetIdx()] = AMU.get(a.GetSymbol(), 0.0) + a.GetTotalNumHs() * H
    if (w == 0).any(): return None
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in m.GetBonds()]
    return w, bonds, n

def _components(n, bonds, drop):
    adj = [[] for _ in range(n)]
    for i, (a, b) in enumerate(bonds):
        if i in drop: continue
        adj[a].append(b); adj[b].append(a)
    seen = np.zeros(n, bool); comps = []
    for s in range(n):
        if seen[s]: continue
        stack = [s]; seen[s] = True; cur = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not seen[v]: seen[v] = True; stack.append(v); cur.append(v)
        comps.append(cur)
    return comps

def fragment_masses(smi, max_breaks=2, max_bonds=34):
    g = mol_graph(smi)
    if g is None: return np.zeros(0)
    w, bonds, n = g
    nb = len(bonds)
    if nb == 0 or nb > max_bonds: return np.array([w.sum()])
    out = {w.sum()}
    for i in range(nb):
        for c in _components(n, bonds, {i}):
            out.add(float(w[c].sum()))
    if max_breaks >= 2:
        for i in range(nb):
            for j in range(i + 1, nb):
                for c in _components(n, bonds, {i, j}):
                    out.add(float(w[c].sum()))
    return np.array(sorted(out))

def frag_masses_safe(smi):
    try: return fragment_masses(smi)
    except Exception: return np.zeros(0)

CARRIERS = {
    "[M+H]+": (PROTON,), "[M-H2O+H]+": (PROTON,), "[M-2H2O+H]+": (PROTON,),
    "[M+NH4]+": (PROTON, NH4 - E), "[M+Na]+": (PROTON, MASS['Na'] - E), "[M+K]+": (PROTON, MASS['K'] - E),
    "[M-H]-": (-PROTON,), "[M-H2O-H]-": (-PROTON,), "[M+CH2O2-H]-": (-PROTON, FORMATE - PROTON),
    "[M+Cl]-": (-PROTON, MASS['Cl'] + E),
}

def explain_score_adduct(frag_mass, peak_mz, peak_int, adduct, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):
    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0
    car = CARRIERS.get(adduct, (PROTON,) if "+" in adduct[-2:] else (-PROTON,))
    ion = np.sort(np.concatenate([frag_mass + dh * H + c for dh in h_shifts for c in car]))
    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()
    if tot <= 0: return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)

def explain_score(frag_mass, peak_mz, peak_int, mode=1.0, tol=0.01, h_shifts=(-2, -1, 0, 1, 2)):
    if len(frag_mass) == 0 or len(peak_mz) == 0: return 0.0
    ion = []
    for dh in h_shifts:
        ion.append(frag_mass + dh * H + (PROTON if mode > 0 else -PROTON))
    ion = np.sort(np.concatenate(ion))
    w = np.sqrt(np.asarray(peak_int, float)); tot = w.sum()
    if tot <= 0: return 0.0
    idx = np.searchsorted(ion, peak_mz)
    ok = np.zeros(len(peak_mz), bool)
    for off in (-1, 0):
        k = np.clip(idx + off, 0, len(ion) - 1)
        ok |= np.abs(ion[k] - peak_mz) <= tol
    return float(w[ok].sum() / tot)
''',

'pv_fp.py': '''"""Spectrum -> fingerprint model (FPNet)."""
import math
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

FP_MAX_PEAKS = 128
ADDUCT_LIST = ["[M+H]+", "[M+NH4]+", "[M+Na]+", "[M+K]+", "[M-H2O+H]+", "[M-2H2O+H]+", "[M]+",
               "[M-H]-", "[M-H2O-H]-", "[M+CH2O2-H]-", "[M+C2H4O2-H]-", "[M+Cl]-", "[M]-",
               "[M+2H]2+", "[M-2H]-", "[2M+H]+", "[2M+Na]+", "[2M+NH4]+", "[2M-H]-", "[2M+K]+",
               "[2M+CH2O2-H]-", "[2M+C2H4O2-H]-", "[2M+Na-2H]-", "[M+Na-2H]-", "[M-H2O]+", "<unk>"]
ADDUCT_IX = {a: i for i, a in enumerate(ADDUCT_LIST)}
INSTR_LIST = ["timsTOF", "Orbitrap", "QTOF", "IT", "other"]

def instr_family(s):
    if s is None: return 4
    t = str(s).lower()
    if 'timstof' in t: return 0
    if 'orbitrap' in t or 'qft' in t or 'ftms' in t or 'hybrid ft' in t or 'itft' in t or 'exactive' in t: return 1
    if 'tof' in t: return 2
    if 'trap' in t or 'qq' in t: return 3
    return 4

def prep_peaks(mz, inten, prec_mz, max_peaks=FP_MAX_PEAKS, floor=1e-3, win=50.0, per_win=8):
    mz = np.asarray(mz, np.float64); it = np.asarray(inten, np.float64)
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = (mz <= prec_mz + 1.5)
    mz, it = mz[keep], it[keep]
    if len(mz) == 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    mx = it.max()
    if mx <= 0: return np.zeros(0, np.float32), np.zeros(0, np.float32)
    keep = it >= floor * mx
    mz, it = mz[keep], it[keep]
    if len(mz) > max_peaks:
        order = np.argsort(-it)
        bucket = (mz // win).astype(np.int64)
        cnt = {}; sel = []
        for i in order:
            b = bucket[i]; c = cnt.get(b, 0)
            if c < per_win: cnt[b] = c + 1; sel.append(i)
        sel = np.array(sel)
        if len(sel) > max_peaks:
            sel = sel[np.argsort(-it[sel])[:max_peaks]]
        elif len(sel) < max_peaks:
            rest = np.array([i for i in order if i not in set(sel.tolist())])
            need = max_peaks - len(sel)
            if len(rest): sel = np.concatenate([sel, rest[:need]])
        mz, it = mz[sel], it[sel]
    o = np.argsort(mz)
    mz, it = mz[o], it[o]
    v = np.sqrt(it / it.max())
    return mz.astype(np.float32), v.astype(np.float32)

class SinEmb(nn.Module):
    def __init__(self, dim, lo=-2.0, hi=3.2, power=1.0):
        super().__init__()
        n = dim // 2
        wav = torch.pow(10.0, (hi - lo) * torch.pow(torch.linspace(0, 1, n), power) + lo)
        self.register_buffer('inv', (2 * math.pi) / wav)
    def forward(self, x):
        a = x.unsqueeze(-1) * self.inv
        return torch.cat([torch.sin(a), torch.cos(a)], -1)

class Block(nn.Module):
    def __init__(self, d, h, drop):
        super().__init__(); self.h = h
        self.n1 = nn.LayerNorm(d); self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(drop), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(drop)
    def forward(self, x, pad):
        B, N, D = x.shape; y = self.n1(x)
        q, k, v = self.qkv(y).view(B, N, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        m = (~pad)[:, None, None, :]
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        x = x + self.drop(self.o(a.transpose(1, 2).reshape(B, N, D)))
        return x + self.drop(self.ff(self.n2(x)))

class FPNet(nn.Module):
    def __init__(self, nbits, d=512, layers=6, heads=8, drop=0.1):
        super().__init__()
        self.d = d
        self.mz_emb = SinEmb(d)
        self.nl_emb = SinEmb(d)
        self.pk = nn.Linear(2 * d + 1, d)
        self.prec_emb = SinEmb(d)
        self.ad = nn.Embedding(len(ADDUCT_LIST), d)
        self.ins = nn.Embedding(len(INSTR_LIST), d)
        self.gl = nn.Linear(d + 3, d)
        self.blocks = nn.ModuleList([Block(d, heads, drop) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Sequential(nn.Linear(2 * d, 2048), nn.GELU(), nn.Dropout(drop), nn.Linear(2048, nbits))
    def forward(self, mz, it, pad, prec, ad, ins, ce, mode):
        B, N = mz.shape
        nl = (prec[:, None] - mz).clamp(min=0)
        p = self.pk(torch.cat([self.mz_emb(mz), self.nl_emb(nl), it.unsqueeze(-1)], -1))
        g = self.gl(torch.cat([self.prec_emb(prec),
                              (ce / 100.0).unsqueeze(-1), mode.unsqueeze(-1),
                              torch.log1p(prec).unsqueeze(-1) / 10.0], -1)) + self.ad(ad) + self.ins(ins)
        x = torch.cat([g.unsqueeze(1), p], 1)
        pad = torch.cat([torch.zeros(B, 1, dtype=torch.bool, device=pad.device), pad], 1)
        for b in self.blocks: x = b(x, pad)
        x = self.norm(x)
        cls = x[:, 0]
        msk = (~pad[:, 1:]).float().unsqueeze(-1)
        mean = (x[:, 1:] * msk).sum(1) / msk.sum(1).clamp(min=1)
        return self.head(torch.cat([cls, mean], -1))

def load_fp_models(paths, dev):
    single, merged = [], []
    nbits = None
    for pth in paths:
        ck = torch.load(pth, map_location='cpu', weights_only=False)
        net = FPNet(ck['nbits'], d=ck['d'], layers=ck['layers']).to(dev).eval()
        net.load_state_dict(ck['model'])
        (merged if 'merged' in str(pth).replace('\\\\', '/').split('/')[-1] else single).append(net)
        nbits = ck['nbits']
    return single, merged, nbits

def merge_peaks(specs):
    specs = [(a, b) for a, b in specs if len(a)]
    if not specs: return np.zeros(0), np.zeros(0)
    mz = np.concatenate([np.asarray(a, float) for a, _ in specs])
    it = np.concatenate([np.asarray(b, float) / max(float(np.asarray(b, float).max()), 1e-9) for _, b in specs])
    o = np.argsort(mz); mz, it = mz[o], it[o]
    keep = np.ones(len(mz), bool)
    for j in range(1, len(mz)):
        if mz[j] - mz[j - 1] < 0.005:
            if it[j] >= it[j - 1]: keep[j - 1] = False
            else: keep[j] = False
    return mz[keep], it[keep]

@torch.no_grad()
def logits_batch(P, nets, dev, prec, adduct, instr, ce, mode):
    keep = [i for i, (a, _) in enumerate(P) if len(a)]
    if not keep: return None
    P = [P[i] for i in keep]
    B = len(P); N = max(len(a) for a, _ in P)
    mz = np.zeros((B, N), np.float32); it = np.zeros((B, N), np.float32); pad = np.ones((B, N), bool)
    for i, (a, b) in enumerate(P):
        mz[i, :len(a)] = a; it[i, :len(b)] = b; pad[i, :len(a)] = False
    T = lambda x: torch.as_tensor(x, device=dev)
    args = (T(mz), T(it), T(pad),
            T(np.asarray(prec, np.float32)[keep]),
            T(np.array([ADDUCT_IX.get(a, ADDUCT_IX['<unk>']) for a in np.asarray(adduct, object)[keep]])),
            T(np.array([instr_family(s) for s in np.asarray(instr, object)[keep]])),
            T(np.asarray(ce, np.float32)[keep]),
            T(np.asarray(mode, np.float32)[keep]))
    return np.mean([n(*args).float().mean(0).cpu().numpy() for n in nets], axis=0)

def molecule_logits(models, specs, prec, adduct, instr, ce, mode):
    single, merged, dev = models
    out = []
    if single:
        P = [prep_peaks(a, b, p) for (a, b), p in zip(specs, prec)]
        za = logits_batch(P, single, dev, prec, adduct, instr, ce, mode)
        if za is not None: out.append(za)
    if merged:
        mz, it = merge_peaks(specs)
        pm = float(np.median(prec))
        P = [prep_peaks(mz, it, pm)]
        zb = logits_batch(P, merged, dev, [pm], [adduct[0]], [instr[0]], [25.0], [float(np.mean(mode))])
        if zb is not None: out.append(zb)
    return np.mean(out, axis=0) if out else None
''',

'casmi_engine.py': '''"""End-to-end inference engine with ChEBI/LIPID MAPS (BIO) & AFIX library index."""
import os, sys, glob, time, pickle, math
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np, pandas as pd, pyarrow.parquet as pq, pyarrow as pa
import pv
from pv import CFG
import re as _re

_MONO = dict(C=12.0, H=1.00782503207, N=14.0030740048, O=15.9949146196, S=31.97207100, P=30.97376163,
             Cl=34.96885268, Br=78.9183371, F=18.99840322, I=126.904473, Si=27.9769265325, Se=79.9165213,
             Na=22.9897692809, K=38.96370668, B=11.0093054, As=74.9215965, Fe=55.9349375, D=2.0141017778)
_TOK = _re.compile(r"([A-Z][a-z]?)(\\d*)")

def formula_mass(f):
    if not isinstance(f, str) or not f: return np.nan
    m = 0.0
    for el, n in _TOK.findall(f):
        if el not in _MONO: return np.nan
        m += _MONO[el] * (int(n) if n else 1)
    return m

POOL = None

class RANK:
    W_A = (0.35, 0.55)
    SEEDS = (0, 1)
    GBM = dict(max_depth=6, max_iter=500, learning_rate=0.03, min_samples_leaf=80, l2_regularization=1.0)

T0 = time.time()
LOCAL = os.environ.get("CASMI_LOCAL") == "1"
ROOTS = [r"C:\\Users\\HW-LEE\\Desktop\\CASMI"] if LOCAL else ["/kaggle/input"]

def find(name):
    for root in ROOTS:
        hits = glob.glob(os.path.join(root, "**", name), recursive=True)
        if hits: return sorted(hits, key=len)[0]
    hits = glob.glob(f"**/{name}", recursive=True)
    if hits: return sorted(hits, key=len)[0]
    return None

def log(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)

def load_library(path):
    f = pq.ParquetFile(path)
    n_rows = f.metadata.num_rows
    npk = 0
    for i in range(f.num_row_groups):
        c = f.read_row_group(i, columns=["ms2_mzs"]).column(0).combine_chunks()
        npk += len(c.values)
    off = np.zeros(n_rows + 1, np.int64)
    allmz = np.empty(npk, np.float32); allin = np.empty(npk, np.float32)
    prec, add, pol, ik, smi = [], [], [], [], []
    fo, tims = [], []
    r0 = p0 = 0
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["inchikey14", "normalized_smiles", "adduct", "precursor_mz", "ionization_mode",
                                          "ms2_mzs", "ms2_normalized_intensities", "molecular_formula", "instrument_type"])
        mzc = t.column("ms2_mzs").combine_chunks(); itc = t.column("ms2_normalized_intensities").combine_chunks()
        o = mzc.offsets.to_numpy().astype(np.int64)
        n = len(o) - 1; k = int(o[-1] - o[0])
        allmz[p0:p0 + k] = mzc.values.to_numpy(zero_copy_only=False)[o[0]:o[-1]]
        allin[p0:p0 + k] = itc.values.to_numpy(zero_copy_only=False)[o[0]:o[-1]]
        off[r0 + 1:r0 + n + 1] = p0 + (o[1:] - o[0])
        prec.append(t.column("precursor_mz").to_numpy(zero_copy_only=False).astype(np.float64))
        add += t.column("adduct").cast(pa.string()).to_pylist()
        pol.append(np.array(t.column("ionization_mode").cast(pa.string()).to_pylist(), dtype=object) == "positive")
        ik += t.column("inchikey14").cast(pa.string()).to_pylist()
        smi += t.column("normalized_smiles").cast(pa.string()).to_pylist()
        fo += t.column("molecular_formula").cast(pa.string()).to_pylist()
        tims.append(np.array([isinstance(x, str) and x.lower().replace("-", "").replace(" ", "") == "timstof"
                              for x in t.column("instrument_type").cast(pa.string()).to_pylist()]))
        r0 += n; p0 += k
        del t, mzc, itc
    prec = np.concatenate(prec); add = np.asarray(add, dtype=object)
    pol = np.where(np.concatenate(pol), 1, -1).astype(np.int8)
    nmp = pv.neutral_mass(prec, add)
    _fm = {x: formula_mass(x) for x in set(v for v in fo if isinstance(v, str))}
    nmf = np.array([_fm.get(x, np.nan) if isinstance(x, str) else np.nan for x in fo])
    nm = np.where(np.isfinite(nmf), nmf, nmp)
    tims = np.concatenate(tims)
    log(f"library mass index: formula {np.isfinite(nmf).mean():.1%}, timsTOF rows {tims.mean():.1%}")
    del fo
    return dict(off=off, mz=allmz, it=allin, prec=prec, nm=nm, pol=pol, tims=tims,
                ik=np.asarray(ik, dtype=object), smi=np.asarray(smi, dtype=object))

_g = {}

def _fp_init():
    from rdkit import RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog("rdApp.*")
    _g["bits"] = np.load(find("fp_bits.npy"))
    _g["m2"] = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=4096)
    _g["m3"] = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=4096)
    _g["rk"] = rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048, maxPath=6)

def fp_and_mass(smi):
    from rdkit import Chem
    from rdkit.Chem import MACCSkeys
    from rdkit.Chem.Descriptors import ExactMolWt
    if not _g: _fp_init()
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        fp = np.concatenate([_g["m2"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g["m3"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             _g["rk"].GetFingerprintAsNumPy(m).astype(np.uint8),
                             np.array(MACCSkeys.GenMACCSKeys(m), dtype=np.uint8)])[_g["bits"]]
        return np.packbits(fp), float(ExactMolWt(m))
    except Exception:
        return None

def canon_key(smi):
    from rdkit import Chem, RDLogger
    from rdkit.Chem.MolStandardize import rdMolStandardize
    RDLogger.DisableLog("rdApp.*")
    if "te" not in _g: _g["te"] = rdMolStandardize.TautomerEnumerator()
    try:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(_g["te"].Canonicalize(m))[:14] if m is not None else None
    except Exception:
        return None

def build_pool(L, workers):
    from multiprocessing import Pool as MPool
    d = os.path.dirname(find("coco_fp.npy"))
    cm = pickle.load(open(os.path.join(d, "coco_meta.pkl"), "rb"))
    co_fp = np.load(os.path.join(d, "coco_fp.npy")); co_mass = np.load(os.path.join(d, "coco_mass.npy"))
    co_keys = np.asarray(cm["keys"], dtype=object); co_smi = np.asarray(cm["smiles"], dtype=object)
    
    # BIO: ChEBI + LIPID MAPS
    bio_fp_path = find("bio_fp.npy")
    if bio_fp_path:
        bd = os.path.dirname(bio_fp_path)
        bm = pickle.load(open(os.path.join(bd, "bio_meta.pkl"), "rb"))
        bkeys = np.asarray(bm["keys"], dtype=object); bnew = ~pd.Series(bkeys).isin(set(co_keys)).values
        co_fp = np.vstack([co_fp, np.load(os.path.join(bd, "bio_fp.npy"))[bnew]])
        co_mass = np.concatenate([co_mass, np.load(os.path.join(bd, "bio_mass.npy"))[bnew]])
        co_keys = np.concatenate([co_keys, bkeys[bnew]])
        co_smi = np.concatenate([co_smi, np.asarray(bm["smiles"], dtype=object)[bnew]])
        log(f"+ ChEBI/LIPID MAPS {int(bnew.sum()):,} new structures")
    else:
        log("bio_fp.npy not found, continuing with COCONUT + train")

    tr = pd.DataFrame({"ik": L["ik"], "smi": L["smi"]}).dropna().drop_duplicates("ik")
    tr = tr[~tr.ik.isin(set(co_keys))]
    log(f"COCONUT {len(co_keys):,}; training structures to fingerprint {len(tr):,}")
    with MPool(workers) as mp:
        res = mp.map(fp_and_mass, list(tr.smi), chunksize=500)
    ok = np.array([r is not None for r in res])
    tr_fp = np.stack([r[0] for r in res if r is not None]); tr_mass = np.array([r[1] for r in res if r is not None])
    fp = np.vstack([co_fp, tr_fp]); mass = np.concatenate([co_mass, tr_mass])
    keys = np.concatenate([co_keys, tr.ik.values[ok]]); smis = np.concatenate([co_smi, tr.smi.values[ok]])
    good = np.isfinite(mass)
    o = np.argsort(np.where(good, mass, 1e18), kind="mergesort")[: good.sum()]
    P = dict(fp=fp[o], mass=mass[o], keys=keys[o], smiles=smis[o])
    P["k2i"] = pd.Series(np.arange(len(o)), index=P["keys"]); P["k2i"] = P["k2i"][~P["k2i"].index.duplicated()]
    log(f"pool {len(o):,} structures")
    return P

def pool_fps(idx):
    return np.unpackbits(np.asarray(POOL["fp"][np.asarray(idx, np.int64)]), axis=1)[:, :6930]

def pool_window(t, ppm):
    a = np.searchsorted(POOL["mass"], t * (1 - ppm / 1e6), "left")
    b = np.searchsorted(POOL["mass"], t * (1 + ppm / 1e6), "right")
    return np.arange(a, b)

def _reps(L, rows):
    npk = np.diff(L["off"])[rows]
    df = pd.DataFrame({"row": rows, "p": L["row_p"][rows], "npk": npk, "tims": L["tims"][rows]})
    df = df.sort_values(["p", "tims", "npk", "row"], ascending=[True, False, False, True]).drop_duplicates("p")
    rep = df.row.values; rep_nm = L["nm"][rep]
    o = np.argsort(rep_nm, kind="mergesort"); rep = rep[o]
    roff, rmz, rit = pv.clean_store(rep, L["off"], L["mz"], L["it"], CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)
    return dict(rep=rep, rep_p=L["row_p"][rep], rep_nm=L["nm"][rep], roff=roff, rmz=rmz, rit=rit)

def build_index(L):
    ok = np.isfinite(L["nm"]) & (L["row_p"] >= 0)
    rows = np.where(ok)[0]
    o = np.argsort(L["nm"][rows], kind="mergesort")
    I = dict(lib_rows=rows[o], lib_nm=L["nm"][rows[o]])
    I["all"] = _reps(L, rows)
    I["pos"] = _reps(L, rows[L["pol"][rows] == 1])
    I["neg"] = _reps(L, rows[L["pol"][rows] == -1])
    log(f"index: {len(rows):,} library rows; representatives {len(I['all']['rep']):,}")
    return I

def clean(mz, it):
    return pv._clean(np.asarray(mz, np.float32), np.asarray(it, np.float32),
                     CFG.INT_FLOOR, CFG.MAX_PEAKS, CFG.INT_POWER, CFG.ENT_WEIGHT)

def analog_search(R, Q, target):
    lo = np.searchsorted(R["rep_nm"], target - CFG.ANALOG_WIN, "left")
    hi = np.searchsorted(R["rep_nm"], target + CFG.ANALOG_WIN, "right")
    asim = np.zeros(hi - lo, np.float32)
    if hi > lo and Q:
        shift = (target - R["rep_nm"][lo:hi]).astype(np.float32)
        for qm, qp in Q:
            asim = np.maximum(asim, pv.search_shift_pre(qm, qp, lo, hi, R["roff"], R["rmz"], R["rit"], CFG.MZ_TOL, shift))
    return R["rep_p"][lo:hi], asim

def top_analogs(p, s):
    if len(p) == 0: return []
    df = pd.DataFrame({"p": p, "s": s}).groupby("p", sort=False).s.max().sort_values(ascending=False, kind="mergesort")
    return list(zip(df.index[:CFG.N_ANALOG].astype(int), df.values[:CFG.N_ANALOG].astype(float)))

def molecule_channels(L, I, specs, precs, pols, target):
    Q = [clean(a, b) for a, b in specs]
    keep = [i for i, (a, _) in enumerate(Q) if len(a)]
    Q = [Q[i] for i in keep]; precs = np.asarray(precs)[keep]; pols = np.asarray(pols)[keep]
    tol = target * CFG.PPM_WIN / 1e6
    lo = np.searchsorted(I["lib_nm"], target - tol, "left"); hi = np.searchsorted(I["lib_nm"], target + tol, "right")
    crow = I["lib_rows"][lo:hi]
    u, inv = np.unique(L["row_p"][crow], return_inverse=True)
    M = np.zeros((max(len(Q), 1), len(u)), np.float32); MS = np.zeros_like(M)
    for j, (qm, qp) in enumerate(Q):
        if len(crow) == 0: break
        s = pv.search(qm, qp, crow, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                      CFG.INT_POWER, CFG.ENT_WEIGHT)
        np.maximum.at(M[j], inv, s)
        sh = (precs[j] - L["prec"][crow]).astype(np.float32)
        s2 = pv.search_shift_rows(qm, qp, crow, L["off"], L["mz"], L["it"], CFG.MZ_TOL, CFG.INT_FLOOR, CFG.MAX_PEAKS,
                                  CFG.INT_POWER, CFG.ENT_WEIGHT, sh)
        np.maximum.at(MS[j], inv, s2)
    cnt = np.bincount(inv, minlength=len(u)).astype(np.float32)
    z = np.zeros(0, np.float32)
    lib = (u, M.max(0) if len(u) else z, M.mean(0) if len(u) else z, MS.max(0) if len(u) else z, cnt)
    p, s = analog_search(I["all"], Q, target)
    ana = top_analogs(p, s)
    ps, ss = [], []
    for sign, name in ((1, "pos"), (-1, "neg")):
        QQ = [q for q, pl in zip(Q, pols) if pl == sign]
        if QQ:
            p, s = analog_search(I[name], QQ, target); ps.append(p); ss.append(s)
    anapol = top_analogs(np.concatenate(ps), np.concatenate(ss)) if ps else []
    return lib, ana, anapol

def parse_ce(v):
    try:
        a = np.abs(np.atleast_1d(np.asarray(v, float)))
        return float(a.mean()) if len(a) else 25.0
    except Exception:
        return 25.0

def _rel(v):
    v = np.asarray(v, np.float32)
    return [v, pv._rank_norm(v).astype(np.float32), v - (v.max() if len(v) else 0.0)]

def _analog_feats(cfp, ana):
    nc = len(cfp)
    if not ana: return np.zeros((nc, 5), np.float32)
    afp = pool_fps([p for p, _ in ana]).astype(np.float32); cf = cfp.astype(np.float32)
    inter = cf @ afp.T
    tan = inter / (cf.sum(1)[:, None] + afp.sum(1)[None, :] - inter + 1e-9)
    w = np.clip(np.array([s for _, s in ana], np.float32), 0, None)
    ap = (tan * (w ** 3)[None, :]).max(1)
    ap6 = (tan * (w ** 6)[None, :]).max(1)
    return np.column_stack(_rel(ap) + [ap6, np.full(nc, w[0])]).astype(np.float32)

def blk_base(r, cfp):
    ana = r["ana"]
    afp = pool_fps([p for p, _ in ana]) if ana else None
    sims = np.array([s for _, s in ana], np.float32)
    return pv.rank_features(cfp, r["lib"], afp, sims, r["zlog"], r["frag"])

def blk_mass(r, cfp):
    ppm = (POOL["mass"][r["cand"]] - r["target"]) / r["target"] * 1e6
    return np.column_stack([ppm, np.abs(ppm), np.abs(ppm + 1.5)]).astype(np.float32)

def blk_libx(r, cfp):
    cnt = np.log1p(r["lib_cnt"])
    return np.column_stack(_rel(r["lib_mean"]) + _rel(r["libsh"]) + [cnt, cnt - cnt.max()]).astype(np.float32)

def blk_anapol(r, cfp):
    return _analog_feats(cfp, r["anapol"])

def blk_fragx(r, cfp):
    return np.column_stack(_rel(r["frag_ad"]) + _rel(r["frag_mean"]) + [r["frag_pol"]]).astype(np.float32)

BLOCKS = {"base": blk_base, "mass": blk_mass, "libx": blk_libx, "anapol": blk_anapol, "fragx": blk_fragx}

def compute_channels(L, I, models, te):
    mols = list(te.groupby("molecule_id", sort=True))
    recs = []
    for gi, (mid, sub) in enumerate(mols):
        specs = []
        for r in sub.itertuples():
            a_ = np.asarray(r.ms2_mzs, np.float32); b_ = np.asarray(r.ms2_normalized_intensities, np.float32)
            k_ = a_ <= float(r.precursor_mz) + 2.0
            a_, b_ = a_[k_], b_[k_]
            specs.append((a_, b_ / b_.max() if len(b_) and b_.max() > 0 else b_))
        nms = pv.neutral_mass(sub.precursor_mz.values.astype(np.float64), sub.adduct.astype(str).values)
        nms = nms[np.isfinite(nms)]
        rec = dict(mid=mid, cand=np.zeros(0, np.int64))
        if len(nms):
            target = float(np.median(nms))
            modes = np.where(sub.ionization_mode.astype(str).values == "positive", 1.0, -1.0)
            lib, ana, anapol = molecule_channels(L, I, specs, sub.precursor_mz.values.astype(np.float64), modes, target)
            cand = pool_window(target, CFG.PPM_WIN)
            if len(cand) == 0: cand = pool_window(target, CFG.PPM_FALLBACK)
            ces = np.array([parse_ce(v) for v in sub.collision_energy_ev.values], np.float32)
            import pv_fp
            zlog = pv_fp.molecule_logits(models, specs, sub.precursor_mz.values.astype(np.float32),
                                         list(sub.adduct.astype(str).values), list(sub.instrument_type.astype(str).values),
                                         ces, modes)
            u, lmax, lmean, lsh, lcnt = lib
            pos = {int(x): i for i, x in enumerate(u)}
            ci = np.array([pos.get(int(c), -1) for c in cand], np.int64); has = ci >= 0
            rec.update(target=target, cand=cand, zlog=None if zlog is None else zlog.astype(np.float32), specs=specs,
                       mode=float(np.mean(modes)), modes=modes, adducts=list(sub.adduct.astype(str).values),
                       ana=ana, anapol=anapol)
            for nm_, arr in (("lib", lmax), ("lib_mean", lmean), ("libsh", lsh), ("lib_cnt", lcnt)):
                v = np.zeros(len(cand), np.float32); v[has] = arr[ci[has]]; rec[nm_] = v
        recs.append(rec)
        if gi % 50 == 0: log(f" channels {gi}/{len(mols)}")
    return mols, recs

def compute_frag(recs, workers):
    from multiprocessing import Pool as MPool
    uc = np.unique(np.concatenate([r["cand"] for r in recs if len(r["cand"])])) if any(len(r["cand"]) for r in recs) else np.zeros(0, int)
    log(f"MetFrag-lite on {len(uc):,} candidates")
    if len(uc) == 0: return
    with MPool(workers) as mp:
        fr = mp.map(pv.frag_masses_safe, list(POOL["smiles"][uc]), chunksize=16)
    fmap = dict(zip(uc.tolist(), fr))
    for r in recs:
        if not len(r["cand"]): continue
        peaks = []
        for a_, b_ in r.pop("specs"):
            m2, i2 = pv._clean(a_, b_, CFG.INT_FLOOR, CFG.MAX_PEAKS, 1.0, False)
            peaks.append((np.asarray(m2, float), np.asarray(i2, float)))
        F1 = np.zeros((len(r["cand"]), max(len(peaks), 1)), np.float32); F2 = np.zeros_like(F1); F3 = np.zeros_like(F1)
        for j, c in enumerate(r["cand"]):
            fm = fmap[int(c)]
            for k, (x, y) in enumerate(peaks):
                F1[j, k] = pv.explain_score(fm, x, y, mode=r["mode"], tol=CFG.MZ_TOL)
                F2[j, k] = pv.explain_score_adduct(fm, x, y, r["adducts"][k], tol=CFG.MZ_TOL)
                F3[j, k] = pv.explain_score(fm, x, y, mode=r["modes"][k], tol=CFG.MZ_TOL)
        r["frag"] = F1.max(1); r["frag_ad"] = F2.max(1); r["frag_mean"] = F3.mean(1); r["frag_pol"] = F3.max(1)

def _hgb(X, Y, W, seed, gbm):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(random_state=seed, **gbm); m.fit(X, Y, sample_weight=W)
    return m

def fit_rankers(rank_train_path):
    z = np.load(rank_train_path, allow_pickle=True)
    X, Y, S = z["X"], z["Y"], z["S"]
    rankers = [_hgb(X, Y, np.where(S == 0, wA, 1.0 - wA), sd, RANK.GBM) for wA in RANK.W_A for sd in RANK.SEEDS]
    log(f"our ranker: {len(rankers)} GBMs on {X.shape[0]:,} rows x {X.shape[1]} features")
    return rankers, [str(b) for b in z["blocks"]]

def fit_pv_rankers(pv_rank_path, priors=(0.30, 0.60), seeds=(0, 1, 2, 3)):
    z = np.load(pv_rank_path)
    rankers = [_hgb(z["X"], z["Y"], np.where(z["M"] == 0, w1, 1.0 - w1), sd, CFG.GBM) for w1 in priors for sd in seeds]
    log(f"prvsiyan ranker: {len(rankers)} GBMs on {z['X'].shape[0]:,} rows x {z['X'].shape[1]} features")
    return rankers, z["X"].shape[1]

def score_molecules(recs, rankers, blocks, use_fp):
    out = {}
    for r in recs:
        if not len(r["cand"]): continue
        rr = r if use_fp else dict(r, zlog=None)
        cfp = pool_fps(r["cand"])
        Xm = np.hstack([BLOCKS[b](rr, cfp) for b in blocks]).astype(np.float32)
        out[r["mid"]] = np.mean([m.predict_proba(Xm)[:, 1] for m in rankers], axis=0)
    return out

def blend_scores(a, b, wa):
    out = {}
    for mid in a:
        ra = 1.0 - pv._rank_norm(a[mid])
        if b is not None and mid in b:
            rb = 1.0 - pv._rank_norm(b[mid])
            out[mid] = wa * ra + (1.0 - wa) * rb + 1e-6 * a[mid]
        else:
            out[mid] = ra + 1e-6 * a[mid]
    return out

def make_submissions(mols, recs, score_sets, sample_path, workers, topn=25):
    from multiprocessing import Pool as MPool
    cand = {r["mid"]: r["cand"] for r in recs if len(r["cand"])}
    ordered = {n: {mid: cand[mid][np.argsort(-sc, kind="mergesort")][:topn + 15] for mid, sc in S.items()}
               for n, S in score_sets.items()}
    allc = [v for o in ordered.values() for v in o.values()]
    short = np.unique(np.concatenate(allc)) if allc else np.zeros(0, np.int64)
    with MPool(workers) as mp:
        ck = mp.map(canon_key, list(POOL["smiles"][short]), chunksize=64)
    ckey = dict(zip(short.tolist(), ck))
    samp = pd.read_csv(sample_path)
    subs = {}
    for n, o in ordered.items():
        rows = []
        for mid, _ in mols:
            out, seen = [], set()
            for c in o.get(mid, []):
                k = ckey.get(int(c)) or POOL["keys"][c]
                if k in seen: continue
                seen.add(k); out.append(POOL["smiles"][c])
                if len(out) == topn: break
            while len(out) < topn:
                out.append("CCO")
            rows.append((mid, ";".join(out[:topn])))
        sub = samp[["molecule_id"]].merge(pd.DataFrame(rows, columns=["molecule_id", "smiles"]), on="molecule_id", how="left")
        sub["smiles"] = sub["smiles"].fillna("CCO")
        subs[n] = sub
    return subs

def main(test_path, train_path, sample_path, our_rank_path, pv_rank_path, fp_model_paths, workers=4, limit=0, w_pv=0.88):
    global POOL
    import torch
    import pv_fp
    L = load_library(train_path); log(f"library {len(L['prec']):,} spectra")
    POOL = build_pool(L, workers)
    L["row_p"] = POOL["k2i"].reindex(L["ik"]).fillna(-1).values.astype(np.int64)
    del L["ik"], L["smi"]
    I = build_index(L)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    single, merged, _ = pv_fp.load_fp_models(fp_model_paths, dev)
    log(f"fp models: {len(single)} single + {len(merged)} merged on {dev}")
    te = pq.read_table(test_path).to_pandas()
    if limit:
        keep = sorted(te.molecule_id.unique())[:limit]; te = te[te.molecule_id.isin(keep)]
    log(f"test: {len(te):,} spectra / {te.molecule_id.nunique():,} molecules")
    mols, recs = compute_channels(L, I, (single, merged, dev), te)
    del L, I
    compute_frag(recs, workers)
    
    if our_rank_path and os.path.exists(our_rank_path):
        ours, blocks = fit_rankers(our_rank_path)
        s_ours = score_molecules(recs, ours, blocks, use_fp=False)
        del ours
    else:
        log("sim_rank_rows_nofp.npz not found, using Ranker A alone")
        s_ours = None

    pvr, nfeat = fit_pv_rankers(pv_rank_path)
    s_pv = score_molecules(recs, pvr, ["base"], use_fp=True)
    del pvr

    scores_dict = {"blend": blend_scores(s_pv, s_ours, w_pv), "pv": s_pv}
    if s_ours is not None:
        scores_dict["ours"] = s_ours
    subs = make_submissions(mols, recs, scores_dict, sample_path, workers)
    log("submissions ready")
    return subs, recs, scores_dict
'''
}

for _n, _src in ENG_FILES.items():
    with open(os.path.join(ENG_DIR, _n), 'w') as f:
        f.write(_src)

# eng_runner.py: Runs Two-Ranker Engine + Regioisomer Expansion -> /kaggle/working/eng_lists.json
with open(os.path.join(ENG_DIR, 'eng_runner.py'), 'w') as f:
    f.write('''"""Our engine runner (two-ranker + BIO + AFIX + Regioisomers) -> eng_lists.json."""
import os, sys, json, time, glob
import numpy as np

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    test, train, sample, out = sys.argv[1:5]
    import casmi_engine as E

    _orig_find = E.find
    def _find(n):
        if n == 'fp_bits.npy':
            hits = glob.glob('/kaggle/input/**/coco_fp.npy', recursive=True) or glob.glob('**/coco_fp.npy', recursive=True)
            if hits: return os.path.join(os.path.dirname(hits[0]), 'fp_bits.npy')
        return _orig_find(n)
    E.find = _find
    E.RANK.W_A = (0.35, 0.55); E.RANK.SEEDS = (0, 1); E.CFG.N_ANALOG = 200

    fp_models = sorted(p for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True) if ('casmi26-fp-models' in p or 'fingerprint' in p.lower() or 'single' in p.lower()))
    if not fp_models:
        fp_models = sorted(glob.glob('/kaggle/input/**/fp_*.pt', recursive=True))
    if not fp_models:
        fp_models = sorted(glob.glob('/kaggle/input/**/fpnet_*.pt', recursive=True))[:2]
    print('engine fp models:', fp_models, flush=True)

    T0 = time.time()
    sim_rank = _find('sim_rank_rows_nofp.npz') if glob.glob('/kaggle/input/**/sim_rank_rows_nofp.npz', recursive=True) else None
    pv_rank = _find('rank_train.npz')
    subs, recs, S = E.main(test, train, sample, sim_rank, pv_rank, fp_models,
                           workers=os.cpu_count(), w_pv=0.88)
    
    bl = E.blend_scores(S['pv'], S.get('ours'), 0.88)
    res = {}
    
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    def quick_regios(smi, max_regios=8):
        try:
            m = Chem.MolFromSmiles(smi)
            if not m: return []
            pk = Chem.MolToInchiKey(m)[:14]
            pf = rdMolDescriptors.CalcMolFormula(m)
            ring_info = m.GetRingInfo()
            rings = [r for r in ring_info.AtomRings() if len(r) in (5, 6) and all(m.GetAtomWithIdx(i).GetIsAromatic() for i in r)]
            out = []
            for ring in rings:
                ring_set = set(ring)
                substs = []
                for r_idx in ring:
                    for nbr in m.GetAtomWithIdx(r_idx).GetNeighbors():
                        if nbr.GetIdx() not in ring_set:
                            substs.append((r_idx, nbr.GetIdx()))
                if 1 <= len(substs) <= 3:
                    free_c = [i for i in ring if i not in {r for r, _ in substs} and m.GetAtomWithIdx(i).GetAtomicNum() == 6]
                    for r_idx, ext_idx in substs:
                        for fc in free_c:
                            try:
                                em = Chem.EditableMol(m)
                                em.RemoveBond(r_idx, ext_idx)
                                em.AddBond(fc, ext_idx, Chem.BondType.SINGLE)
                                nm = em.GetMol()
                                Chem.SanitizeMol(nm)
                                nk = Chem.MolToInchiKey(nm)[:14]
                                if nk != pk and rdMolDescriptors.CalcMolFormula(nm) == pf and nk not in [x[1] for x in out]:
                                    out.append((Chem.MolToSmiles(nm), nk))
                                    if len(out) >= max_regios: return out
                            except Exception:
                                continue
            return out
        except Exception:
            return []

    for r in recs:
        mid = r['mid']
        if mid not in bl: continue
        order = np.argsort(-bl[mid], kind='mergesort')[:80]
        smis, keys, seen = [], [], set()
        for c in r['cand'][order]:
            s = E.POOL['smiles'][c]; k = E.canon_key(s)
            if k is None or k in seen: continue
            seen.add(k); smis.append(s); keys.append(k)
            if len(smis) >= 32: break
        
        # Add constitutional regioisomers with same formula into ranks 33..40
        if len(smis) > 0 and len(smis) < 40:
            regios = quick_regios(smis[0], max_regios=40 - len(smis))
            for rsmi, rk in regios:
                if rk not in seen:
                    seen.add(rk); smis.append(rsmi); keys.append(rk)
                    if len(smis) >= 40: break
        
        res[str(mid)] = dict(smiles=smis, keys=keys)

    json.dump(res, open(out, 'w'))
    print('engine lists generated:', len(res), f'{time.time()-T0:.0f}s', flush=True)
''')

ENG = {}
try:
    _t0 = time.time()
    rr = subprocess.run([sys.executable, os.path.join(ENG_DIR, 'eng_runner.py'),
                         os.path.join(COMP, 'test.parquet'),
                         os.path.join(COMP, 'train.parquet'),
                         os.path.join(COMP, 'sample_submission.csv'),
                         '/kaggle/working/eng_lists.json'],
                        capture_output=True, text=True, timeout=4 * 3600, cwd=ENG_DIR)
    print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
    ENG = json.load(open('/kaggle/working/eng_lists.json'))
    print('Engine lists successfully loaded:', len(ENG), f'{time.time()-_t0:.0f}s')
except Exception as e:
    print('ENGINE FAILED -> Falling back to v4n base lists:', repr(e))

# ── 5. PubChem-Only Channel (N1=5000) ─────────────────────────────────────────
CORE = '''"""P_pubchem probe core."""
from __future__ import annotations
import os, time
import numpy as np

PC_N1 = 5000
PPM = 10.0
K = 25
_W = {}

def molecule_logits(bank, spectra):
    from casmi import chem, fpnet
    from casmi.spectra import merge_spectra
    items_s = []
    for s in spectra:
        pm, pi = fpnet.prep_peaks(s['mz'], s['it'], s['prec'])
        ce_ok = s['ce'] is not None and not np.isnan(s['ce'])
        items_s.append(dict(mz=pm, it=pi, prec=float(s['prec']), adduct_ix=chem.adduct_index(s['adduct']),
                            ce=float(s['ce']) if ce_ok else 0.0, ce_known=1.0 if ce_ok else 0.0,
                            n_merged=min(int(s.get('ce_n', 1)), 8), mode=float(s['mode'])))
    items_m = []
    for md in (1, -1):
        grp = [s for s in spectra if s['mode'] == md]
        if not grp: continue
        mm, ii = merge_spectra([(s['mz'], s['it']) for s in grp])
        prec = float(np.median([s['prec'] for s in grp]))
        pm, pi = fpnet.prep_peaks(mm, ii, prec)
        ces = [s['ce'] for s in grp if s['ce'] is not None and not np.isnan(s['ce'])]
        adducts = [s['adduct'] for s in grp]
        ad = max(adducts, key=adducts.count)
        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),
                            ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ces else 0.0,
                            n_merged=min(sum(max(1, int(s.get('ce_n', 1))) for s in grp), 8), mode=float(md)))
    if not items_s: return None
    zs, _ = bank.logits_el(items_s)
    zm, _ = bank.logits_el(items_m)
    return (0.5 * (zs.mean(0) + zm.mean(0))).astype(np.float32)

def init_worker(pc_dir, bits_path, pool_meta_path, code_dir=None):
    import sys
    if code_dir and code_dir not in sys.path: sys.path.insert(0, code_dir)
    import pandas as pd
    from casmi import chem
    _W['chem'] = chem
    _W['mass'] = np.load(os.path.join(pc_dir, 'pc_mass.npy'), mmap_mode='r')
    _W['off'] = np.load(os.path.join(pc_dir, 'pc_off.npy'), mmap_mode='r')
    _W['buf'] = np.load(os.path.join(pc_dir, 'pc_smiles.npy'), mmap_mode='r')
    bits = np.load(bits_path)
    _W['bits'] = bits
    _W['sel_e'] = np.where(bits < 4096)[0]
    _W['raw_e'] = bits[_W['sel_e']]
    _W['pool_keys'] = set(pd.read_parquet(pool_meta_path, columns=['key']).key.values.tolist())
    r = chem._rdkit()
    _W['ecfp4'] = r['gen'].GetMorganGenerator(radius=2, fpSize=4096)
    _W['Chem'] = r['Chem']

def _window(target):
    tol = target * PPM * 1e-6
    a = int(np.searchsorted(_W['mass'], target - tol, 'left'))
    b = int(np.searchsorted(_W['mass'], target + tol, 'right'))
    off = np.asarray(_W['off'][a:b + 1]).astype(np.int64)
    if b <= a: return []
    raw = bytes(_W['buf'][off[0]:off[-1]])
    base = off[0]
    return [raw[off[i] - base:off[i + 1] - base].decode('ascii') for i in range(b - a)]

def probe_one(task):
    mid, target, z = task
    t0 = time.time()
    chem = _W['chem']; Chem = _W['Chem']; gen = _W['ecfp4']
    smis = _window(float(target))
    n = len(smis)
    diag = dict(molecule_id=mid, n_window=n, n_pass1=0, n_pass2=0, n_keyed=0, n_in_pool=0, n_listed=0)
    if n == 0:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    raw_e = _W['raw_e']; z_e = z[_W['sel_e']].astype(np.float32)
    E = np.zeros((n, len(raw_e)), np.float32); ok = np.zeros(n, bool)
    for i, s in enumerate(smis):
        m = Chem.MolFromSmiles(s)
        if m is None: continue
        E[i] = gen.GetFingerprintAsNumPy(m)[raw_e]; ok[i] = True
    sc1 = E @ z_e
    sc1[~ok] = -np.inf
    n1 = min(PC_N1, int(ok.sum()))
    diag['n_pass1'] = n1
    if n1 == 0:
        diag['secs'] = time.time() - t0
        return mid, [], [], [], diag
    top1 = np.argpartition(-sc1, n1 - 1)[:n1] if n1 < n else np.where(ok)[0]
    bits = _W['bits']
    fz, idx = [], []
    for i in top1:
        fp = chem.raw_fingerprint(smis[i])
        if fp is None: continue
        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))
    diag['n_pass2'] = len(idx)
    order = np.argsort(-np.asarray(fz), kind='stable')
    out, out_fz, out_k, seen = [], [], [], set()
    for o in order:
        s = smis[idx[o]]
        k = chem.score_key(s)
        diag['n_keyed'] += 1
        if k is None or k in seen: continue
        seen.add(k)
        if k in _W['pool_keys']:
            diag['n_in_pool'] += 1
            continue
        out.append(s); out_fz.append(float(fz[o])); out_k.append(k)
        if len(out) >= K: break
    diag['n_listed'] = len(out)
    diag['secs'] = time.time() - t0
    return mid, out, out_fz, out_k, diag
'''

RUNNER = '''"""PubChem-only runner."""
import os, sys, json, time
import numpy as np, pandas as pd

if __name__ == '__main__':
    V3CODE, V3MOD, POOL_DIR, PC_DIR, TEST, OUT, NW = sys.argv[1:8]
    MAXM = int(sys.argv[8]) if len(sys.argv) > 8 else 0
    NW = int(NW)
    sys.path.insert(0, V3CODE); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import torch, glob
    import probe_core2 as pc
    from casmi import fpnet, chem
    T0 = time.time()
    models = sorted(glob.glob(os.path.join(V3MOD, 'fpnet_*.pt')))
    assert len(models) == 2, models
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    bank = fpnet.ModelBank(models, device=dev)
    bits = np.load(os.path.join(POOL_DIR, 'fp_bits.npy')); nbits = len(bits)
    pmass = pd.read_parquet(os.path.join(POOL_DIR, 'pool_meta.parquet'), columns=['mass']).mass.values.astype(np.float64)
    assert np.all(np.diff(pmass) >= 0), 'pool not mass-sorted'
    pfp = np.load(os.path.join(POOL_DIR, 'pool_fp.npy'), mmap_mode='r')
    te = pd.read_parquet(TEST)
    te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
    tasks, extra = [], {}
    groups = list(te.groupby('molecule_id', sort=False))
    if MAXM: groups = groups[:MAXM]
    for mid, sub in groups:
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        if not len(nms): continue
        target = float(np.median(nms))
        spectra = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct,
                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,
                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
        spectra = spectra[:16]
        try:
            z = pc.molecule_logits(bank, spectra)
        except Exception as e:
            z = None
        if z is None: continue
        tol = target * 10e-6
        a = np.searchsorted(pmass, target - tol, 'left'); b = np.searchsorted(pmass, target + tol, 'right')
        if b <= a:
            tol = target * 30e-6
            a = np.searchsorted(pmass, target - tol, 'left'); b = np.searchsorted(pmass, target + tol, 'right')
        if b > a:
            F = np.unpackbits(np.asarray(pfp[a:b]), axis=1)[:, :nbits].astype(np.float64)
            best = float((F @ z.astype(np.float64)).max())
        else:
            best = float('nan')
        extra[mid] = dict(best_pool_fz=best, target=target)
        tasks.append((mid, target, z))
    del bank; torch.cuda.empty_cache()
    print(f'logits for {len(tasks)} molecules {time.time()-T0:.0f}s', flush=True)
    res = {}
    import multiprocessing as mp
    ctx = mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
    with ctx.Pool(NW, initializer=pc.init_worker, initargs=(PC_DIR, os.path.join(POOL_DIR, 'fp_bits.npy'),
                                                            os.path.join(POOL_DIR, 'pool_meta.parquet'), V3CODE)) as P:
        for k, (mid, smis, fzs, keys, d) in enumerate(P.imap_unordered(pc.probe_one, tasks, chunksize=1)):
            res[mid] = dict(pc=smis, pc_fz=fzs, pc_keys=keys, **extra[mid])
            if (k + 1) % 25 == 0 or k + 1 == len(tasks):
                print(f' pubchem {k+1}/{len(tasks)} {time.time()-T0:.0f}s', flush=True)
    json.dump(res, open(OUT, 'w'))
    print('pc_runner done', len(res), f'{time.time()-T0:.0f}s', flush=True)
'''

with open('/kaggle/working/probe_core2.py', 'w') as f: f.write(CORE)
with open('/kaggle/working/pc_runner.py', 'w') as f: f.write(RUNNER)

PC = {}
try:
    rr = subprocess.run([sys.executable, '/kaggle/working/pc_runner.py', V3CODE, V3MOD, POOL_DIR, PC_DIR,
                         os.path.join(COMP, 'test.parquet'), '/kaggle/working/pc_lists.json', '4'],
                        capture_output=True, text=True, timeout=4 * 3600)
    print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
    PC = json.load(open('/kaggle/working/pc_lists.json'))
    print('PubChem lists successfully loaded:', len(PC), f'{time.time()-T0:.0f}s')
except Exception as e:
    print('PUBCHEM CHANNEL FAILED -> Continuing with base lists:', repr(e))

# ── 6. v4n Base Retrieval + Feature Families + LightGBM Ranker ───────────────
sys.path.insert(0, V4CODE)
os.makedirs('work', exist_ok=True)
for f in ['pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy', 'pool_frag_mass.npy', 'fp_bits.npy', 'train_structs.parquet', 'train_fp_sel.npy']:
    src = os.path.join(POOL_DIR, f); dst = f'work/{f}'
    if not os.path.exists(dst): os.symlink(src, dst)

from casmi.build import build_spec_cache
build_spec_cache(os.path.join(COMP, 'train.parquet'), 'work')

import torch, lightgbm as lgb
from casmi.library import Library
from casmi.pool import Pool
from casmi.engine import Engine, EngineCfg, FEATURES
from casmi import fpnet, chem
import v1engine

sys.path.insert(1, os.path.join(V4CODE, 'fe_v4'))
dev = 'cuda' if torch.cuda.is_available() else 'cpu'
L = Library('work'); P = Pool('work')
tfp = np.load('work/train_fp_sel.npy')

_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)
bank = fpnet.ModelBank([_FULL[0] if _FULL else os.path.join(V4MOD, 'fpnet_0.pt')], device=dev)
print('Engine FP bank:', _FULL[0] if _FULL else 'fpnet_0 (A) fallback', flush=True)

E = Engine(L, P, tfp, EngineCfg(generate=True), bank)
V = v1engine.V1FE(E, [os.path.join(V4MOD, m['file']) for m in MAN['fe_models']], MAN['families'], device=dev)

RK = pickle.load(open(os.path.join(V4MOD, MAN['ranker']['file']), 'rb'))
RK_FEATS = list(RK['features']); RK_B = [lgb.Booster(model_str=s) for s in RK['boosters']]

def rank_score(X, names):
    pos = {n: i for i, n in enumerate(names)}
    miss = [f for f in RK_FEATS if f not in pos]
    assert not miss, f'ranker features not produced: {miss}'
    Xs = np.ascontiguousarray(X[:, [pos[f] for f in RK_FEATS]], dtype=np.float32)
    return np.mean([b.predict(Xs, num_threads=4) for b in RK_B], axis=0)

print('v4n base engine ready; ranker features', len(RK_FEATS), 'boosters', len(RK_B), f'{time.time()-T0:.0f}s')

# Run v4n on test molecules
te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id', sort=False))
BASE, n_err = {}, 0

for gi, (mid, sub) in enumerate(mols):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    smis, keys, lib_max, scs, forms = [], [], 0.0, [], []
    if len(nms):
        target = float(np.median(nms))
        spectra = []
        for r in sub.itertuples():
            ce = r.collision_energy_ev
            ce_arr = np.atleast_1d(np.asarray(ce, dtype=float)) if ce is not None else np.zeros(0)
            spectra.append(dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                                mode=1 if r.ionization_mode == 'positive' else -1, adduct=r.adduct,
                                prec=float(r.precursor_mz), ce=float(ce_arr.mean()) if len(ce_arr) else np.nan,
                                ce_n=int(len(ce_arr)) if len(ce_arr) else 1))
        try:
            C, X, F, names, info = V.run(spectra, target)
            lib_max = float(info.get('lib_max', 0.0))
            if len(C.pid):
                score = rank_score(np.concatenate([X, F], 1), list(FEATURES) + list(names))
                order = np.argsort(-score, kind='stable')
                seen = set()
                for i in order:
                    k = C.key[i]
                    if k in seen: continue
                    seen.add(k); smis.append(C.smiles[i]); keys.append(k); scs.append(float(score[i])); forms.append(C.formula[i])
                    if len(smis) >= TOPN: break
        except Exception as e:
            print('ERROR', mid, repr(e)); n_err += 1
    BASE[mid] = (smis, keys, lib_max, scs, forms)
    if (gi + 1) % 25 == 0 or gi == len(mols) - 1:
        print(f' {gi+1}/{len(mols)} molecules processed {time.time()-T0:.0f}s', flush=True)

print('Base retrieval finished with errors:', n_err)

# ── 7. Forward Models (ICEBERG + GLACIER) on the UNION ────────────────────────
ICE_SCORES, ice_stats = {}, dict(molecules=0, changed_top25=0, changed_top1=0)
try:
    ICE_PKG = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))
    sys.path.append(ICE_PKG)
    import fuse as ice_fuse
    del V, E, bank; gc.collect(); torch.cuda.empty_cache()

    mids = sorted(BASE, key=lambda m: BASE[m][2])
    cands = {m: ice_fuse.ice_candidates(BASE[m][0], BASE[m][4], TOPN) for m in mids}
    
    if ENG:
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF
        from rdkit import Chem as _Ch
        def _f2(s):
            try: return _CMF(_Ch.MolFromSmiles(s))
            except Exception: return s
        n_add = 0
        for m in mids:
            e = ENG.get(str(m))
            if not e or not e.get('smiles'): continue
            bs, bk, bf = list(BASE[m][0][:TOPN]), set(BASE[m][1][:TOPN]), list(BASE[m][4][:TOPN])
            es = [s for s, k in zip(e['smiles'][:40], e['keys'][:40]) if k not in bk]
            ef = [_f2(s) for s in es]
            extra = ice_fuse.ice_candidates(bs + es, bf + ef, len(bs) + len(es))
            before = len(cands.get(m, []))
            cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
            n_add += len(cands[m]) - before
        print('ICE_UNION: engine candidates added to ICE input:', n_add, flush=True)

    if ICE_PC:
        from rdkit.Chem.rdMolDescriptors import CalcMolFormula
        from rdkit import Chem as _Chem
        def _form(s):
            try: return CalcMolFormula(_Chem.MolFromSmiles(s))
            except Exception: return s
        for m in mids:
            p = PC.get(m) if isinstance(PC, dict) else None
            if p and p.get('pc') and BASE[m][2] < LIB_TAU:
                p['pc_form'] = [_form(x) for x in p['pc']]
                extra = ice_fuse.ice_candidates(p['pc'], p['pc_form'], 25)
                cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))

    cands = {m: c for m, c in cands.items() if c}
    items = ice_fuse.build_ice_input(te, cands)
    print('ICE input molecules', len(items), 'candidates', sum(len(it['cands']) for it in items), f'{time.time()-T0:.0f}s', flush=True)
    ICE_SCORES = ice_fuse.run_ice(ICE_PKG, items, workdir='/kaggle/working/ice_work', device='cuda', budget_s=ICE_BUDGET, site='/kaggle/working/ice_site')
    meta_f = '/kaggle/working/ice_work/ice_out.json.meta.json'
    if os.path.exists(meta_f): print('ICE meta:', open(meta_f).read()[:1500])
except Exception as e:
    print('ICE FAILED -> ranker order kept:', repr(e))

GL_SCORES, gl_stats, GL_LAM, GL_BUDGET = {}, dict(molecules=0, changed_vs_ice_top25=0, changed_vs_ice_top1=0), 1.0, 4000
try:
    GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))
    sys.path.append(GL_PKG)
    import gl_fuse
    gc.collect(); torch.cuda.empty_cache()
    GL_SCORES = gl_fuse.run_gl(GL_PKG, items, workdir='/kaggle/working/gl_work', device='cuda', budget_s=GL_BUDGET, site='/kaggle/working/ice_site', wheels=os.path.join(ICE_PKG, 'wheels'))
    meta_f = '/kaggle/working/gl_work/gl_out.json.meta.json'
    print('GL meta:', open(meta_f).read()[:1500] if os.path.exists(meta_f) else 'missing')
except Exception as e:
    GL_SCORES = {}
    print('GL FAILED -> ICE only behaviour:', repr(e))

def gl_of(m):
    g = GL_SCORES.get(str(m), {}) if isinstance(GL_SCORES, dict) else {}
    return g if isinstance(g, dict) and any(v is not None for v in g.values()) else {}

def gl_rerank(order0, smis, keys, scs, forms, ice, gl, top_n, tag=''):
    if not gl: return order0
    try:
        o = gl_fuse.rerank_multi(smis, keys, scs, forms, [ice, gl], [ICE_LAM, GL_LAM], top_n=top_n)
    except Exception as e:
        print('gl rerank failed', tag, repr(e)); return order0
    for k, v in (('molecules', 1), ('changed_vs_ice_top25', int(o[:25] != order0[:25])), ('changed_vs_ice_top1', int(o[:1] != order0[:1]))):
        gl_stats[tag + k] = gl_stats.get(tag + k, 0) + v
    return o

for m in list(BASE):
    smis, keys, lib_max, scs, forms = BASE[m]
    ice = ICE_SCORES.get(str(m), {}) if ICE_SCORES else {}
    gl = gl_of(m)
    if ice and any(v is not None for v in ice.values()):
        try:
            order = ice_fuse.rerank(smis, keys, scs, forms, ice, lam=ICE_LAM, top_n=TOPN)
            order = gl_rerank(order, smis, keys, scs, forms, ice, gl, TOPN)
            s2 = [smis[i] for i in order]; k2 = [keys[i] for i in order]
            ice_stats['molecules'] += 1
            ice_stats['changed_top25'] += int(s2[:25] != smis[:25])
            ice_stats['changed_top1'] += int(bool(s2) and s2[0] != smis[0])
            smis, keys = s2, k2
        except Exception as e:
            print('rerank failed', m, repr(e))
    BASE[m] = (smis[:25], keys[:25], lib_max)

    p = PC.get(m) if (ICE_PC and isinstance(PC, dict)) else None
    if p and p.get('pc_form') and ice:
        try:
            o = ice_fuse.rerank(p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, lam=ICE_LAM, top_n=25)
            o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25, 'pc_')
            fz0 = p['pc_fz'][0]
            p['pc'] = [p['pc'][i] for i in o]; p['pc_keys'] = [p['pc_keys'][i] for i in o]
            p['pc_fz'] = [fz0] + [p['pc_fz'][i] for i in o][1:]
            ice_stats['pc_changed'] = ice_stats.get('pc_changed', 0) + int(o[:1] != [0])
        except Exception as e:
            print('pc rerank failed', m, repr(e))

print('ICE rerank stats:', ice_stats, f'{time.time()-T0:.0f}s')
print('GL rerank stats:', gl_stats, f'{time.time()-T0:.0f}s')

# Gated PubChem Merge
def merge(base, base_keys, pc, pc_keys, slots, n=25):
    bk = set(base_keys)
    pcs = [s for s, k in zip(pc, pc_keys) if k not in bk]
    out, bi, pj = [], 0, 0
    for pos in range(1, n + 1):
        if pos in slots and pj < len(pcs):
            out.append(pcs[pj]); pj += 1
        elif bi < len(base):
            out.append(base[bi]); bi += 1
        elif pj < len(pcs):
            out.append(pcs[pj]); pj += 1
    return out

rows, stats = [], dict(untouched=0, gentle=0, aggressive=0, no_pc=0)
for mid in te.molecule_id.unique():
    smis, keys, lib_max = BASE.get(mid, ([], [], 0.0))
    p = PC.get(mid)
    if p is None or not p['pc']:
        stats['no_pc'] += 1; final = smis
    elif lib_max >= LIB_TAU:
        stats['untouched'] += 1; final = smis
    else:
        rel = p['pc_fz'][0] - p['best_pool_fz'] if np.isfinite(p['best_pool_fz']) else 1e9
        slots = SLOTS_AGG if rel > REL_TH else SLOTS_GENTLE
        stats['aggressive' if rel > REL_TH else 'gentle'] += 1
        final = merge(smis, keys, p['pc'], p['pc_keys'], slots)
    if not final: final = ['CCO']
    rows.append((mid, ';'.join(final[:25])))

print('PubChem merge stats:', stats)
sub = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
samp = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
sub = samp[['molecule_id']].merge(sub, on='molecule_id', how='left')
sub['smiles'] = sub['smiles'].fillna('CCO')
assert len(sub) == len(samp) and sub.molecule_id.is_unique and sub.smiles.notna().all()
assert sub.smiles.map(lambda s: len(s.split(';'))).max() <= 25
sub.to_csv('submission.csv', index=False)
print('v4n baseline submission written:', sub.shape, f'{time.time()-T0:.0f}s')

# ── 8. Weighted RRF Fusion + Forward Re-Rank + Library Match Shield ───────────
ALPHA, KRR = 0.6, 3.0
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF2
from rdkit import Chem as _Ch2

def _form2(s):
    try: return _CMF2(_Ch2.MolFromSmiles(s))
    except Exception: return s

def fuse2(v_smis, e_smis, e_keys, n=40):
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = chem.score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (KRR + r)
        smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k: continue
        sc[k] = sc.get(k, 0.0) + ALPHA / (KRR + r)
        smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]

if ENG:
    v4 = pd.read_csv('submission.csv')
    changed = n_ice = n_shielded = 0
    out = []
    for mid, s in zip(v4.molecule_id, v4.smiles):
        vs = [x for x in s.split(';') if x and x != 'CCO']
        e = ENG.get(str(mid))
        lib_max = BASE.get(mid, ([], [], 0.0))[2] if mid in BASE else 0.0

        if e and e['smiles']:
            fsm, fk, fsc = fuse2(vs, e['smiles'], e['keys'])
            ice = ICE_SCORES.get(str(mid), {}) if isinstance(ICE_SCORES, dict) else {}
            if ice:
                try:
                    o = ice_fuse.rerank(fsm, fk, fsc, [_form2(x) for x in fsm], ice, lam=ICE_LAM, top_n=len(fsm))
                    _g = gl_of(mid)
                    if _g:
                        o = gl_fuse.rerank_multi(fsm, fk, fsc, [_form2(x) for x in fsm], [ice, _g], [ICE_LAM, GL_LAM], top_n=len(fsm))
                    n_ice += int(o[:25] != list(range(min(25, len(o)))))
                    fsm = [fsm[i] for i in o]
                except Exception as ex:
                    print('fused ICE rerank failed', mid, repr(ex))
            
            # HIGH-CONFIDENCE LIBRARY MATCH SHIELD:
            # If lib_max >= 0.88, protect the experimental library match at Rank 1 from forward model noise
            if lib_max >= 0.88 and len(vs) > 0:
                top_lib = vs[0]
                if top_lib in fsm:
                    fsm.remove(top_lib)
                fsm.insert(0, top_lib)
                n_shielded += 1

            changed += int(bool(vs) and fsm[:1] != vs[:1])
            vs = fsm[:25]
        out.append((mid, ';'.join(vs[:25]) if vs else 'CCO'))
    
    fs = pd.DataFrame(out, columns=['molecule_id', 'smiles'])
    assert len(fs) == len(v4) and fs.molecule_id.is_unique and fs.smiles.notna().all()
    assert fs.smiles.map(lambda s: len(s.split(';'))).max() <= 25
    fs.to_csv('submission.csv', index=False)
    print(f'Final Fused Submission written! Top-1 changed: {changed} | ICE/GL reordered: {n_ice} | Shielded Library Matches: {n_shielded} | Total Time: {time.time()-T0:.0f}s')

json.dump(dict(secs=round(time.time() - T0), n_errors=n_err, families=MAN['families']), open('run_manifest.json', 'w'), indent=1)
print('ALL DONE! Submission file is ready at submission.csv.')
