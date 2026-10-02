# ==============================================================================
# 🧪 CASMI 2026: SOTA 0.409 MASTER PIPELINE
# CASMI26 | v4n Fusion + Popularity Prior + Library Gate [LB 0.409]
#
# Ablation Progression:
#   Step 0: Reference (v4n + Engine Fusion + Union)             -> LB 0.399
#   Step 1: Popularity prior (mu=0.15) + Frag re-score (gap)   -> LB 0.404
#   Step 2: Library gate for forward models (ICE/GL_LAM_LIB=0) -> LB 0.404 (safety)
#   Step 3: Stronger popularity prior (POP_MU=0.25)             -> LB 0.409
# ==============================================================================

import os, sys, glob, re, subprocess, time, json, hashlib, pickle, gc
import numpy as np, pandas as pd

T0 = time.time()
WORK = '/kaggle/working'
STAGE = os.path.join(WORK, 'stage_cache')
OURS = os.path.join(WORK, 'ours')

os.makedirs(os.path.join(WORK, 'numba_cache'), exist_ok=True)
os.environ['NUMBA_CACHE_DIR'] = os.path.join(WORK, 'numba_cache')
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

# ── Robust Artifact Discovery ──────────────────────────────────────────────────
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

def log(*a):
    print(f'[{time.time()-T0:6.0f}s]', *a, flush=True)

# ── 1. Embedded Sources ────────────────────────────────────────────────────────
EMBED = {
'eng/casmi_engine.py': '''"""End-to-end inference (Kaggle notebook body). Mirrors e02_channels.py + e05_rank.py feature blocks exactly,
but reads train.parquet / test.parquet directly. Runs locally too (set CASMI_LOCAL=1).

Stages: library -> candidate pool -> index -> per-molecule channels -> MetFrag-lite -> features -> ranker -> submission
"""
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
    raise FileNotFoundError(name)


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
    bd = os.path.dirname(find("bio_fp.npy")); bm = pickle.load(open(os.path.join(bd, "bio_meta.pkl"), "rb"))
    bkeys = np.asarray(bm["keys"], dtype=object); bnew = ~pd.Series(bkeys).isin(set(co_keys)).values
    co_fp = np.vstack([co_fp, np.load(os.path.join(bd, "bio_fp.npy"))[bnew]])
    co_mass = np.concatenate([co_mass, np.load(os.path.join(bd, "bio_mass.npy"))[bnew]])
    co_keys = np.concatenate([co_keys, bkeys[bnew]])
    co_smi = np.concatenate([co_smi, np.asarray(bm["smiles"], dtype=object)[bnew]])
    log(f"+ ChEBI/LIPID MAPS {int(bnew.sum()):,} new structures")
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
    uc = np.unique(np.concatenate([r["cand"] for r in recs]))
    log(f"MetFrag-lite on {len(uc):,} candidates")
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
        ra = 1.0 - pv._rank_norm(a[mid]); rb = 1.0 - pv._rank_norm(b[mid])
        out[mid] = wa * ra + (1.0 - wa) * rb + 1e-6 * a[mid]
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
        assert len(sub) == len(samp) and sub.molecule_id.duplicated().sum() == 0 and sub.smiles.isnull().sum() == 0
        assert (sub.smiles.str.split(";").map(len) <= topn).all()
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
    ours, blocks = fit_rankers(our_rank_path)
    s_ours = score_molecules(recs, ours, blocks, use_fp=False)
    del ours
    pvr, nfeat = fit_pv_rankers(pv_rank_path)
    s_pv = score_molecules(recs, pvr, ["base"], use_fp=True)
    del pvr
    subs = make_submissions(mols, recs, {"blend": blend_scores(s_pv, s_ours, w_pv), "pv": s_pv, "ours": s_ours},
     sample_path, workers)
    log("submissions ready")
    return subs, recs, {"pv": s_pv, "ours": s_ours}
''',

'eng/eng_runner.py': '''"""Our engine (two-ranker, BIO + AFIX) -> {molecule_id: {smiles: [...], keys: [...]}} (blend order, top 40)."""
import os, sys, json, time
import numpy as np

if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    test, train, sample, out = sys.argv[1:5]
    import casmi_engine as E
    C = dict(W_A=(0.35, 0.55), SEEDS=(0, 1), N_ANALOG=200, W_PV=0.88, ORDER_K=80, TOPK=40)
    C.update(json.loads(os.environ.get('CASMI_ENG_CFG', '{}')))
    _orig_find = E.find
    def _find(n):
        if n == 'fp_bits.npy':
            return os.path.join(os.path.dirname(_orig_find('coco_fp.npy')), 'fp_bits.npy')
        return _orig_find(n)
    E.find = _find
    E.RANK.W_A = tuple(C['W_A']); E.RANK.SEEDS = tuple(C['SEEDS']); E.CFG.N_ANALOG = int(C['N_ANALOG'])
    import glob
    fp_models = sorted(p for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True) if 'casmi26-fp-models-v2' in p)
    if not fp_models:
        fp_models = sorted(p for p in glob.glob('/kaggle/input/**/fp_*.pt', recursive=True))
    print('engine fp models', fp_models, flush=True)
    T0 = time.time()
    subs, recs, S = E.main(test, train, sample, E.find('sim_rank_rows_nofp.npz'), E.find('rank_train.npz'), fp_models,
     workers=os.cpu_count(), w_pv=C['W_PV'])
    bl = E.blend_scores(S['pv'], S['ours'], C['W_PV'])
    res = {}
    for r in recs:
        mid = r['mid']
        if mid not in bl:
            continue
        order = np.argsort(-bl[mid], kind='mergesort')[:int(C['ORDER_K'])]
        smis, keys, seen = [], [], set()
        for c in r['cand'][order]:
            s = E.POOL['smiles'][c]; k = E.canon_key(s)
            if k is None or k in seen:
                continue
            seen.add(k); smis.append(s); keys.append(k)
            if len(smis) >= int(C['TOPK']):
                break
        res[str(mid)] = dict(smiles=smis, keys=keys)
    json.dump(res, open(out, 'w'))
    print('engine lists', len(res), f'{time.time()-T0:.0f}s', flush=True)
''',

'eng/pv.py': '''"""prvsiyan analog-propagation pipeline components (public notebook, version of 2026-09-16), copied
verbatim where possible so local simulations and the Kaggle notebook share one source of truth.
Additions are marked `# ADDED`.
"""
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

'eng/pv_fp.py': '''"""Spectrum -> fingerprint model (prvsiyan FPNet), kept apart so CPU worker processes never import torch."""
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
        (merged if 'merged' in str(pth).replace('\\', '/').split('/')[-1] else single).append(net)
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

'frag_rescore.py': '''"""Untrained MetFrag-lite re-scorer: how much MS/MS intensity a candidate's <=2-bond-cleavage fragments explain.

Independent of pv.py (same idea as pv.fragment_masses / explain_score_adduct, new code): enumerate the neutral masses
of the connected pieces left after removing 1 or 2 bonds of the heavy-atom graph (implicit H stay on their atom),
remember the fewest cleavages that give each mass, then match every peak against
`neutral fragment + dh*H + charge carrier` for dh in `h_shifts` and the carriers the precursor adduct allows.
A peak's credit is the weight of its cheapest explanation (1 for 0/1 cleavages, `w2` for 2, `w3` for 3),
and a spectrum's score is the credited share of its (weighted) intensity. Defaults follow the forum ablation on a
241-structure same-formula isomer panel (+0.065 MRR over the common baseline): w2=0.6, linear intensity,
tol 0.005 Da, H shifts -2..+3, at most 2 cleavages. `BASELINE_CFG` is the common/pv.py-style setting.

 score_molecule(spectra, cand_smiles, cfg=None, cache=None) -> list[float | None]
"""
from __future__ import annotations
import numpy as np

AMU = {'C': 12.0, 'H': 1.00782503207, 'N': 14.0030740048, 'O': 15.9949146196, 'P': 30.97376163,
 'S': 31.97207100, 'F': 18.99840322, 'Cl': 34.96885268, 'Br': 78.9183371, 'I': 126.904473,
 'Na': 22.9897692809, 'K': 38.96370668, 'Si': 27.9769265325, 'B': 11.0093054, 'Se': 79.9165213}
H = AMU['H']
E = 0.00054857990
PROTON = H - E
H2O = 2 * H + AMU['O']
NH4 = AMU['N'] + 4 * H
FORMATE = AMU['C'] + 2 * H + 2 * AMU['O']

CARRIERS = {
 '[M+H]+': (PROTON,), '[M-H2O+H]+': (PROTON,), '[M-2H2O+H]+': (PROTON,),
 '[M+NH4]+': (PROTON, NH4 - E), '[M+Na]+': (PROTON, AMU['Na'] - E), '[M+K]+': (PROTON, AMU['K'] - E),
 '[M-H]-': (-PROTON,), '[M-H2O-H]-': (-PROTON,), '[M+CH2O2-H]-': (-PROTON, FORMATE - PROTON),
 '[M+Cl]-': (-PROTON, AMU['Cl'] + E),
}

DEFAULT_CFG = dict(
 w1=1.0,
 w2=0.6,
 w3=0.3,
 intensity='linear',
 mz_power=0.0,
 tol=0.005,
 h_shifts=tuple(range(-2, 4)),
 max_breaks=2,
 max_bonds_2=100,
 max_bonds_3=40,
)
BASELINE_CFG = dict(DEFAULT_CFG, w2=1.0, intensity='sqrt', tol=0.01, h_shifts=tuple(range(-2, 3)))


def resolve_cfg(cfg=None):
    out = dict(DEFAULT_CFG)
    for k, v in (cfg or {}).items():
        if k not in out:
            raise KeyError(f'frag_rescore: unknown cfg key {k!r}')
        out[k] = v
    out['h_shifts'] = tuple(int(d) for d in out['h_shifts'])
    if not 1 <= int(out['max_breaks']) <= 3:
        raise ValueError('frag_rescore: max_breaks must be 1, 2 or 3')
    return out


def _find(parent, a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, k):
    for a in range(n):
        parent[a] = a
        acc[a] = 0.0
    ncomp = n
    for e in range(ea.shape[0]):
        if removed[e]:
            continue
        ra = _find(parent, ea[e])
        rb = _find(parent, eb[e])
        if ra != rb:
            parent[ra] = rb
            ncomp -= 1
    if ncomp < 2:
        return cnt
    for a in range(n):
        r = _find(parent, a)
        acc[r] += w[a]
    for a in range(n):
        if parent[a] == a:
            out_m[cnt] = acc[a]
            out_k[cnt] = k
            cnt += 1
    return cnt


def _enumerate(n, w, ea, eb, max_breaks, base_components):
    nb = ea.shape[0]
    n2 = nb * (nb - 1) // 2 if max_breaks >= 2 else 0
    n3 = nb * (nb - 1) * (nb - 2) // 6 if max_breaks >= 3 else 0
    cap = base_components + (base_components + 1) * nb + (base_components + 2) * n2 + (base_components + 3) * n3
    out_m = np.empty(cap, np.float64)
    out_k = np.empty(cap, np.int8)
    parent = np.empty(n, np.int64)
    acc = np.empty(n, np.float64)
    removed = np.zeros(nb, np.bool_)
    cnt = 0
    for i in range(nb):
        removed[i] = True
        cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 1)
        if max_breaks >= 2:
            for j in range(i + 1, nb):
                removed[j] = True
                cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 2)
                if max_breaks >= 3:
                    for l in range(j + 1, nb):
                        removed[l] = True
                        cnt = _emit_components(n, w, ea, eb, removed, parent, acc, out_m, out_k, cnt, 3)
                        removed[l] = False
                removed[j] = False
        removed[i] = False
    return out_m[:cnt], out_k[:cnt]


try:
    from numba import njit
    _find = njit(cache=True, nogil=True)(_find)
    _emit_components = njit(cache=True, nogil=True)(_emit_components)
    _enumerate = njit(cache=True, nogil=True)(_enumerate)
    HAVE_NUMBA = True
except Exception:
    HAVE_NUMBA = False


def mol_graph(smi):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    try:
        m = Chem.MolFromSmiles(smi) if isinstance(smi, str) and smi else None
    except Exception:
        m = None
    if m is None or m.GetNumAtoms() == 0:
        return None
    pt = Chem.GetPeriodicTable()
    w = np.empty(m.GetNumAtoms(), np.float64)
    for a in m.GetAtoms():
        s = a.GetSymbol()
        am = AMU.get(s)
        if am is None:
            am = pt.GetMostCommonIsotopeMass(s)
        w[a.GetIdx()] = am + a.GetTotalNumHs() * H
    ea = np.array([b.GetBeginAtomIdx() for b in m.GetBonds()], np.int64)
    eb = np.array([b.GetEndAtomIdx() for b in m.GetBonds()], np.int64)
    return w, ea, eb, len(Chem.GetMolFrags(m))


def fragment_levels(smi, max_breaks=2, max_bonds_2=100, max_bonds_3=40):
    g = mol_graph(smi)
    if g is None:
        return None
    w, ea, eb, npieces = g
    nb = len(ea)
    mb = int(max_breaks)
    if mb >= 3 and nb > max_bonds_3:
        mb = 2
    if mb >= 2 and nb > max_bonds_2:
        mb = 1
    levels = [np.array([w.sum()])]
    if npieces > 1:
        from rdkit import Chem
        m = Chem.MolFromSmiles(smi)
        levels = [np.unique(np.concatenate([[w.sum()], [w[list(ix)].sum() for ix in Chem.GetMolFrags(m)]]))]
    if nb == 0:
        return levels + [np.zeros(0) for _ in range(int(max_breaks))]
    fm, fk = _enumerate(len(w), w, ea, eb, mb, npieces)
    fm = np.concatenate([levels[0], fm])
    fk = np.concatenate([np.zeros(len(levels[0]), np.int8), fk])
    key = np.round(fm, 5)
    order = np.lexsort((fk, key))
    key, fm, fk = key[order], fm[order], fk[order]
    first = np.ones(len(key), bool)
    first[1:] = key[1:] != key[:-1]
    fm, fk = fm[first], fk[first]
    return [np.sort(fm[fk == k]) for k in range(int(max_breaks) + 1)]


def _carriers(adduct, mode):
    c = CARRIERS.get(adduct)
    if c is not None:
        return c
    if isinstance(adduct, str) and adduct.endswith('-'):
        return (-PROTON,)
    if isinstance(adduct, str) and adduct.endswith('+'):
        return (PROTON,)
    return (PROTON,) if (mode or 1) > 0 else (-PROTON,)


def _prep_spectrum(s, c):
    mz = np.asarray(s.get('mz', ()), np.float64).ravel()
    it = np.asarray(s.get('it', ()), np.float64).ravel()
    n = min(len(mz), len(it))
    mz, it = mz[:n], it[:n]
    ok = np.isfinite(mz) & np.isfinite(it) & (it > 0)
    mz, it = mz[ok], it[ok]
    if len(mz) == 0:
        return None
    p = c['intensity']
    iw = it if p == 'linear' else np.sqrt(it) if p == 'sqrt' else it ** float(p)
    if c['mz_power']:
        prec = float(s.get('prec') or mz.max())
        iw = iw * (mz / prec) ** float(c['mz_power'])
    tot = iw.sum()
    if not tot > 0:
        return None
    shifts = np.array([car + dh * H for car in _carriers(s.get('adduct'), s.get('mode')) for dh in c['h_shifts']])
    q = (mz[None, :] - shifts[:, None]).ravel()
    return iw / tot, q, len(shifts), len(mz)


def _score_spectrum(levels, weights, prep, tol):
    iw, q, ns, npk = prep
    best = np.zeros(npk)
    for arr, wt in zip(levels, weights):
        if len(arr) == 0 or wt <= 0:
            continue
        a = np.searchsorted(arr, q - tol, 'left')
        b = np.searchsorted(arr, q + tol, 'right')
        hit = (b > a).reshape(ns, npk).any(axis=0)
        best = np.where(hit & (wt > best), wt, best)
    return float(iw @ best)


def _levels_cached(smi, c, cache):
    key = (smi, int(c['max_breaks']), int(c['max_bonds_2']), int(c['max_bonds_3']))
    if cache is not None and key in cache:
        return cache[key]
    try:
        lv = fragment_levels(smi, c['max_breaks'], c['max_bonds_2'], c['max_bonds_3'])
    except Exception:
        lv = None
    if cache is not None:
        cache[key] = lv
    return lv


def score_molecule(spectra, cand_smiles, cfg=None, cache=None):
    c = resolve_cfg(cfg)
    cache = {} if cache is None else cache
    weights = [1.0, float(c['w1']), float(c['w2']), float(c['w3'])][: int(c['max_breaks']) + 1]
    preps = [p for p in (_prep_spectrum(s, c) for s in (spectra or [])) if p is not None]
    tol = float(c['tol'])
    out = []
    for smi in cand_smiles:
        lv = _levels_cached(smi, c, cache)
        if lv is None:
            out.append(None)
            continue
        if not preps:
            out.append(0.0)
            continue
        out.append(float(np.mean([_score_spectrum(lv, weights, p, tol) for p in preps])))
    return out


def warmup():
    fragment_levels('CCOC(=O)c1ccccc1O')
    return HAVE_NUMBA
''',

'fusion_core.py': '''"""Final-stage fusion for casmi26-ours: ICE/GLACIER re-ranking -> gated PubChem merge -> engine-2 RRF fusion."""
import copy
import numpy as np
import pandas as pd

DEFAULTS = dict(LIB_TAU=0.9, REL_TH=600.0, SLOTS_AGG=[2, 4, 6, 8, 10], SLOTS_GENTLE=[4, 8, 12, 16, 20], TOPN=60,
 ICE_LAM=1.0, GL_LAM=1.0, ICE_PC=True, FUSE_ALPHA=0.6, FUSE_KRR=3.0, FUSE_N=40, FALLBACK='CCO',
 POP_MU=0.0, FUSE_ENG_K=40, FRAG_LAM=0.0, FRAG_MODE='all',
 POST_ICE_LAM=None, POST_GL_LAM=None,
 FUSE_SKIP_LIB=None,
 ICE_LAM_LIB=None, GL_LAM_LIB=None,
 FILL_25=False,
 POP_LIB_OFF=False, FRAG_LIB_OFF=False)


def popularity_reorder(smis, keys, scs, forms, pids, pool_pop, mu):
    s = np.asarray(scs, np.float64); z = (s - s.mean()) / (s.std() + 1e-9)
    pid = np.asarray(pids); isg = pid < 0
    f = z + mu * np.where(isg, 0.0, pool_pop[np.maximum(pid, 0)])
    pool_i = np.where(~isg)[0]; o = np.empty(len(pid), np.int64)
    o[isg] = np.where(isg)[0]; o[~isg] = pool_i[np.argsort(-f[pool_i], kind='stable')]
    return ([smis[i] for i in o], [keys[i] for i in o], [float(f[i]) for i in o],
     [forms[i] + ('|gen' if isg[i] else '') for i in o], [int(pid[i]) for i in o])


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


def fuse_rrf(v_smis, e_smis, e_keys, score_key, alpha, krr, n):
    sc, smi_of = {}, {}
    for r, s in enumerate(v_smis, 1):
        k = score_key(s) or s
        sc[k] = sc.get(k, 0.0) + 1.0 / (krr + r); smi_of.setdefault(k, s)
    for r, (s, k) in enumerate(zip(e_smis, e_keys), 1):
        if not k:
            continue
        sc[k] = sc.get(k, 0.0) + alpha / (krr + r); smi_of.setdefault(k, s)
    order = sorted(sc, key=lambda k: -sc[k])[:n]
    return [smi_of[k] for k in order], order, [sc[k] for k in order]


def build_submission(base, pc, eng, ice_scores, gl_scores, cfg, mol_order, sample_ids,
 ice_fuse=None, gl_fuse=None, score_key=None, formula=None, log=print, pool_pop=None,
 frag_scores=None):
    c = dict(DEFAULTS); c.update({k: v for k, v in cfg.items() if k in DEFAULTS})
    base = copy.deepcopy(base); pc = copy.deepcopy(pc) if isinstance(pc, dict) else {}
    eng = eng or {}; ice_scores = ice_scores or {}; gl_scores = gl_scores or {}
    ICE_LAM, GL_LAM, TOPN, LIB_TAU = c['ICE_LAM'], c['GL_LAM'], c['TOPN'], c['LIB_TAU']
    stats = dict(ice=dict(molecules=0, changed_top25=0, changed_top1=0), gl=dict(molecules=0),
     merge=dict(untouched=0, gentle=0, aggressive=0, no_pc=0), fuse=dict(top1_changed=0, ice_reordered=0))

    def gl_of(m):
        g = gl_scores.get(str(m), {}) if isinstance(gl_scores, dict) else {}
        return g if isinstance(g, dict) and any(v is not None for v in g.values()) else {}

    def gl_rerank(order0, smis, keys, scs, forms, ice, gl, top_n, tag=''):
        if not gl or gl_fuse is None:
            return order0
        try:
            o = gl_fuse.rerank_multi(smis, keys, scs, forms, [ice, gl], [ICE_LAM, GL_LAM], top_n=top_n)
        except Exception as e:
            log('gl rerank failed', tag, repr(e)); return order0
        g = stats['gl']
        for k, v in (('molecules', 1), ('changed_vs_ice_top25', int(o[:25] != order0[:25])),
         ('changed_vs_ice_top1', int(o[:1] != order0[:1]))):
            g[tag + k] = g.get(tag + k, 0) + v
        return o

    def covered(d):
        return bool(d) and any(v is not None for v in d.values())

    def frag_rerank(m, smis, keys, scs, forms, ice, gl, top_n):
        fr = (frag_scores or {}).get(str(m), {})
        if c['FRAG_LAM'] <= 0 or not covered(fr) or gl_fuse is None:
            return None
        if c['FRAG_LIB_OFF'] and lib_of_m.get(m, lib_of_m.get(str(m), 0.0)) >= LIB_TAU:
            return None
        if c['FRAG_MODE'] == 'gap':
            if covered(ice):
                return None
            terms, lams = [fr], [c['FRAG_LAM']]
        else:
            terms = [d for d in (ice, gl) if covered(d)] + [fr]
            lams = [l for d, l in ((ice, ICE_LAM), (gl, GL_LAM)) if covered(d)] + [c['FRAG_LAM']]
        stats.setdefault('frag', {'molecules': 0})['molecules'] += 1
        return gl_fuse.rerank_multi(smis, keys, scs, forms, terms, lams, top_n=top_n)

    def lam_pair(lib_max):
        lib = lib_max >= LIB_TAU
        il = c['ICE_LAM_LIB'] if (lib and c['ICE_LAM_LIB'] is not None) else c['ICE_LAM']
        gll = c['GL_LAM_LIB'] if (lib and c['GL_LAM_LIB'] is not None) else c['GL_LAM']
        return il, gll

    extra = {}
    lib_of_m = {m: v[2] for m, v in base.items()}

    # ---- stage A: ICE (+GL) re-rank of base and PubChem
    for m in list(base):
        v = base[m]
        smis, keys, lib_max, scs, forms = list(v[0][:TOPN]), list(v[1][:TOPN]), v[2], list(v[3][:TOPN]), list(v[4][:TOPN])
        pids = list(v[5][:TOPN]) if len(v) > 5 else None
        ICE_LAM, GL_LAM = lam_pair(lib_max)
        pop_on = c['POP_MU'] > 0 and not (c['POP_LIB_OFF'] and lib_max >= LIB_TAU)
        if pop_on and pool_pop is not None and pids is not None and len(scs) > 1:
            try:
                smis, keys, scs, forms, pids = popularity_reorder(smis, keys, scs, forms, pids, pool_pop, c['POP_MU'])
                stats.setdefault('pop', {'molecules': 0})['molecules'] += 1
            except Exception as e:
                log('popularity prior failed', m, repr(e))
        ice = ice_scores.get(str(m), {}) if ice_scores else {}
        gl = gl_of(m)
        smis0, keys0, scs0, forms0 = smis, keys, scs, forms
        if ice_fuse is not None and ice and any(v is not None for v in ice.values()):
            try:
                order = ice_fuse.rerank(smis, keys, scs, forms, ice, lam=ICE_LAM, top_n=TOPN)
                order = gl_rerank(order, smis, keys, scs, forms, ice, gl, TOPN)
                s2 = [smis[i] for i in order]; k2 = [keys[i] for i in order]
                st = stats['ice']; st['molecules'] += 1
                st['changed_top25'] += int(s2[:25] != smis[:25]); st['changed_top1'] += int(bool(s2) and s2[0] != smis[0])
                smis, keys = s2, k2
            except Exception as e:
                log('rerank failed', m, repr(e))
        try:
            o = frag_rerank(m, smis0, keys0, scs0, forms0, ice, gl, TOPN)
            if o is not None:
                smis, keys = [smis0[i] for i in o], [keys0[i] for i in o]
        except Exception as e:
            log('frag rerank failed', m, repr(e))
        base[m] = (smis[:25], keys[:25], lib_max)
        extra[m] = smis[25:]
        p = pc.get(m) if c['ICE_PC'] else None
        if ice_fuse is not None and p and p.get('pc') and lib_max < LIB_TAU and ice:
            try:
                if not p.get('pc_form'):
                    p['pc_form'] = [formula(x) for x in p['pc']]
                o = ice_fuse.rerank(p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, lam=ICE_LAM, top_n=25)
                o = gl_rerank(o, p['pc'], p['pc_keys'], p['pc_fz'], p['pc_form'], ice, gl, 25, 'pc_')
                fz0 = p['pc_fz'][0]
                p['pc'] = [p['pc'][i] for i in o]; p['pc_keys'] = [p['pc_keys'][i] for i in o]
                p['pc_fz'] = [fz0] + [p['pc_fz'][i] for i in o][1:]
                stats['ice']['pc_changed'] = stats['ice'].get('pc_changed', 0) + int(o[:1] != [0])
            except Exception as e:
                log('pc rerank failed', m, repr(e))

    # ---- stage B: gated PubChem merge
    rows = {}
    for mid in mol_order:
        smis, keys, lib_max = base.get(mid, ([], [], 0.0))
        p = pc.get(mid)
        ms = stats['merge']
        if p is None or not p.get('pc'):
            ms['no_pc'] += 1; final = smis
        elif lib_max >= LIB_TAU:
            ms['untouched'] += 1; final = smis
        else:
            bp = p.get('best_pool_fz')
            rel = p['pc_fz'][0] - bp if bp is not None and np.isfinite(bp) else 1e9
            aggressive = rel > c['REL_TH']
            ms['aggressive' if aggressive else 'gentle'] += 1
            final = merge(smis, keys, p['pc'], p['pc_keys'], c['SLOTS_AGG'] if aggressive else c['SLOTS_GENTLE'])
        if not final:
            final = [c['FALLBACK']]
        rows[mid] = ';'.join(final[:25])
    v4 = [(mid, rows.get(mid, c['FALLBACK'])) for mid in sample_ids]
    lib_of = {m: v[2] for m, v in base.items()}

    def fill(mid, lst):
        if not c['FILL_25'] or len(lst) >= 25:
            return lst
        seen = {score_key(x) or x for x in lst}; out = list(lst)
        pool = list((eng.get(str(mid)) or {}).get('smiles', [])) + list((pc.get(mid) or {}).get('pc', [])) + list(extra.get(mid, []))
        for x in pool:
            k = score_key(x) or x
            if k not in seen:
                seen.add(k); out.append(x)
            if len(out) >= 25:
                break
        stats.setdefault('fill', {'molecules': 0, 'added': 0})
        stats['fill']['molecules'] += int(len(out) > len(lst)); stats['fill']['added'] += len(out) - len(lst)
        return out

    # ---- stage C: engine-2 fusion (+ ICE/GL on the fused list)
    if not eng:
        if c['FILL_25']:
            v4 = [(mid, ';'.join(fill(mid, [x for x in s.split(';') if x and x != c['FALLBACK']])) or c['FALLBACK'])
             for mid, s in v4]
        return pd.DataFrame(v4, columns=['molecule_id', 'smiles']), stats
    out = []
    for mid, s in v4:
        vs = [x for x in s.split(';') if x and x != c['FALLBACK']]
        ICE_LAM, GL_LAM = lam_pair(lib_of.get(mid, 0.0))
        P_ICE = ICE_LAM if c['POST_ICE_LAM'] is None else c['POST_ICE_LAM']
        P_GL = GL_LAM if c['POST_GL_LAM'] is None else c['POST_GL_LAM']
        e = eng.get(str(mid))
        if c['FUSE_SKIP_LIB'] is not None and lib_of.get(mid, 0.0) >= c['FUSE_SKIP_LIB']:
            e = None
            stats['fuse']['skipped_lib'] = stats['fuse'].get('skipped_lib', 0) + 1
        if e and e.get('smiles'):
            K = int(c['FUSE_ENG_K'])
            fsm, fk, fsc = fuse_rrf(vs, e['smiles'][:K], e['keys'][:K], score_key, c['FUSE_ALPHA'], c['FUSE_KRR'], c['FUSE_N'])
            ice = ice_scores.get(str(mid), {}) if isinstance(ice_scores, dict) else {}
            fsm0, fk0, fsc0 = fsm, fk, fsc
            if ice and ice_fuse is not None:
                try:
                    forms = [formula(x) for x in fsm]
                    o = ice_fuse.rerank(fsm, fk, fsc, forms, ice, lam=P_ICE, top_n=len(fsm))
                    g = gl_of(mid)
                    if g and gl_fuse is not None:
                        o = gl_fuse.rerank_multi(fsm, fk, fsc, forms, [ice, g], [P_ICE, P_GL], top_n=len(fsm))
                    stats['fuse']['ice_reordered'] += int(o[:25] != list(range(min(25, len(o)))))
                    fsm = [fsm[i] for i in o]
                except Exception as ex:
                    log('fused ICE rerank failed', mid, repr(ex))
            if c['FRAG_LAM'] > 0:
                try:
                    o = frag_rerank(mid, fsm0, fk0, fsc0, [formula(x) for x in fsm0], ice, gl_of(mid), len(fsm0))
                    if o is not None:
                        fsm = [fsm0[i] for i in o]
                except Exception as ex:
                    log('fused frag rerank failed', mid, repr(ex))
            stats['fuse']['top1_changed'] += int(bool(vs) and fsm[:1] != vs[:1])
            vs = fsm[:25]
        vs = fill(mid, vs[:25])
        out.append((mid, ';'.join(vs[:25]) if vs else c['FALLBACK']))
    return pd.DataFrame(out, columns=['molecule_id', 'smiles']), stats


def validate(sub, sample_ids, max_rank=25):
    assert len(sub) == len(sample_ids) and sub.molecule_id.is_unique and sub.smiles.notna().all()
    assert set(sub.molecule_id) == set(sample_ids)
    assert sub.smiles.map(lambda s: len([x for x in s.split(';') if x])).min() >= 1
    assert sub.smiles.map(lambda s: len(s.split(';'))).max() <= max_rank
''',

'pc/pc_runner.py': '''"""PubChem-only channel for the final submission."""
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
    if MAXM:
        groups = groups[:MAXM]
    for mid, sub in groups:
        nms = sub.nm.values[np.isfinite(sub.nm.values)]
        if not len(nms):
            continue
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
            print('ERROR logits', mid, repr(e), flush=True); z = None
        if z is None:
            continue
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
''',

'pc/probe_core2.py': '''"""P_pubchem probe core."""
from __future__ import annotations
import os, time
import numpy as np

PC_N1 = int(os.environ.get('CASMI_PC_N1', 5000))
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
        if not grp:
            continue
        mm, ii = merge_spectra([(s['mz'], s['it']) for s in grp])
        prec = float(np.median([s['prec'] for s in grp]))
        pm, pi = fpnet.prep_peaks(mm, ii, prec)
        ces = [s['ce'] for s in grp if s['ce'] is not None and not np.isnan(s['ce'])]
        adducts = [s['adduct'] for s in grp]
        ad = max(adducts, key=adducts.count)
        items_m.append(dict(mz=pm, it=pi, prec=prec, adduct_ix=chem.adduct_index(ad),
         ce=float(np.mean(ces)) if ces else 0.0, ce_known=1.0 if ce_ok else 0.0,
         n_merged=min(sum(max(1, int(s.get('ce_n', 1))) for s in grp), 8), mode=float(md)))
    if not items_s:
        return None
    zs, _ = bank.logits_el(items_s)
    zm, _ = bank.logits_el(items_m)
    return (0.5 * (zs.mean(0) + zm.mean(0))).astype(np.float32)


def init_worker(pc_dir, bits_path, pool_meta_path, code_dir=None):
    import sys
    if code_dir and code_dir not in sys.path:
        sys.path.insert(0, code_dir)
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
    a = int(np.searchsorted(_W['mass'], target - tol, 'left')); b = int(np.searchsorted(_W['mass'], target + tol, 'right'))
    off = np.asarray(_W['off'][a:b + 1]).astype(np.int64)
    if b <= a:
        return []
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
        if m is None:
            continue
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
        if fp is None:
            continue
        fz.append(float(fp[bits].astype(np.float32) @ z)); idx.append(int(i))
    diag['n_pass2'] = len(idx)
    order = np.argsort(-np.asarray(fz), kind='stable')
    out, out_fz, out_k, seen = [], [], [], set()
    for o in order:
        s = smis[idx[o]]
        k = chem.score_key(s)
        diag['n_keyed'] += 1
        if k is None or k in seen:
            continue
        seen.add(k)
        if k in _W['pool_keys']:
            diag['n_in_pool'] += 1
            continue
        out.append(s); out_fz.append(float(fz[o])); out_k.append(k)
        if len(out) >= K:
            break
    diag['n_listed'] = len(out)
    diag['secs'] = time.time() - t0
    return mid, out, out_fz, out_k, diag
'''
}

# ── 2. Unified Master Config (0.409 Breakthrough Knobs) ───────────────────────
CFG = dict(
    VERSION='v4b-libgate-pop025',
    TOPN=60,
    FP_BANK='full1',
    LIB_TAU=0.9,
    REL_TH=600.0,
    SLOTS_AGG=[2, 4, 6, 8, 10],
    SLOTS_GENTLE=[4, 8, 12, 16, 20],
    USE_PC=True,
    PC_N1=5000,
    PC_WORKERS=4,
    USE_ICE=True,
    ICE_LAM=1.0,
    ICE_BUDGET=5400,
    ICE_PC=True,
    ICE_UNION=True,
    ICE_UNION_K=40,
    USE_GL=True,
    GL_LAM=1.0,
    GL_BUDGET=4000,
    USE_ENG=True,
    ENG_W_PV=0.88,
    ENG_W_A=[0.35, 0.55],
    ENG_SEEDS=[0, 1],
    ENG_N_ANALOG=200,
    ENG_ORDER_K=80,
    ENG_TOPK=40,
    FUSE_ALPHA=0.6,
    FUSE_KRR=3.0,
    FUSE_N=40,
    FUSE_ENG_K=40,
    POP_MU=0.25,
    FRAG_LAM=0.5,
    FRAG_MODE='gap',
    POST_ICE_LAM=None,
    POST_GL_LAM=None,
    ICE_LAM_LIB=0.0,
    GL_LAM_LIB=0.0,
    FUSE_SKIP_LIB=None,
    FILL_25=False,
    POP_LIB_OFF=False,
    FRAG_LIB_OFF=False,
    SMOKE_N=12,
    SMOKE_ICE_BUDGET=300,
    FULL_ON_COMMIT=False,
    DUMP_STAGES=True,
    STAGE_TIMEOUT_S=4 * 3600,
    VALIDATION=False,
    VAL_LIB='enveda-np-examples',
    VAL_N=250,
    VAL_SEED=0,
    VAL_MAX_SPEC=16,
    VAL_PURGE='all',
)

def dump(name, obj):
    if CFG['DUMP_STAGES']:
        os.makedirs(STAGE, exist_ok=True)
        json.dump(obj, open(os.path.join(STAGE, f'{name}.json'), 'w'))

# ── 3. Setup: RDKit Wheel, Path Discovery, Write Embedded Sources ───────────────
tag = f'cp{sys.version_info.major}{sys.version_info.minor}'
whl = [w for w in glob.glob('/kaggle/input/**/rdkit-*.whl', recursive=True) if tag in w]
if not whl:
    whl = sorted(glob.glob('/kaggle/input/**/rdkit*.whl', recursive=True))
if whl:
    log('rdkit wheel:', whl[0])
    r = subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-index', '--no-deps', whl[0]], capture_output=True, text=True)
    print(r.stdout[-500:], r.stderr[-500:])

import rdkit
log('rdkit', rdkit.__version__)

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
assert not bad, f'dataset files differ from MANIFEST.json: {bad}'

log('V4CODE', V4CODE, '| V4MOD', V4MOD, '| V3CODE', V3CODE, '| POOL', POOL_DIR, '| PC', PC_DIR, '| COMP', COMP)
log(f"{len(MAN['files'])} files match MANIFEST.json; families {MAN['families']}")

for _rel, _src in EMBED.items():
    _p = os.path.join(OURS, _rel)
    os.makedirs(os.path.dirname(_p), exist_ok=True)
    open(_p, 'w').write(_src)

sys.path.insert(0, OURS)
import fusion_core
log('embedded sources written to disk:', sorted(EMBED))
json.dump(CFG, open(os.path.join(WORK, 'cfg.json'), 'w'), indent=1)

# ── 4. Commit Run vs. Scoring Rerun Detection ──────────────────────────────────
_te = pd.read_parquet(os.path.join(COMP, 'test.parquet'), columns=['molecule_id', 'spectrum_id'])
_sig = hashlib.md5(','.join(sorted(_te.spectrum_id.astype(str))).encode()).hexdigest()
IS_RERUN = bool(os.getenv('KAGGLE_IS_COMPETITION_RERUN')) or _sig != '654e030e4173780e945252b4a5adf70b'
ICE_BUDGET = CFG['ICE_BUDGET']

if not IS_RERUN and not CFG['FULL_ON_COMMIT']:
    SM = os.path.join(WORK, 'smoke_comp')
    os.makedirs(SM, exist_ok=True)
    _full = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
    _keep = sorted(_full.molecule_id.unique())[:CFG['SMOKE_N']]
    _full[_full.molecule_id.isin(_keep)].to_parquet(os.path.join(SM, 'test.parquet'))
    for _f in ['train.parquet', 'sample_submission.csv']:
        if not os.path.exists(os.path.join(SM, _f)):
            os.symlink(os.path.join(COMP, _f), os.path.join(SM, _f))
    COMP = SM
    ICE_BUDGET = CFG['SMOKE_ICE_BUDGET']

log('RERUN', IS_RERUN, '| sig', _sig, '| COMP', COMP, '| ICE_BUDGET', ICE_BUDGET, '| FULL_ON_COMMIT', CFG['FULL_ON_COMMIT'])

# ── 5. Run Engine 2 Subprocess ────────────────────────────────────────────────
ENG = {}
if CFG['USE_ENG']:
    try:
        _t0 = time.time()
        _env = dict(os.environ, CASMI_ENG_CFG=json.dumps(dict(
            W_A=CFG['ENG_W_A'], SEEDS=CFG['ENG_SEEDS'], N_ANALOG=CFG['ENG_N_ANALOG'],
            W_PV=CFG['ENG_W_PV'], ORDER_K=CFG['ENG_ORDER_K'], TOPK=CFG['ENG_TOPK'])))
        rr = subprocess.run([sys.executable, os.path.join(OURS, 'eng', 'eng_runner.py'),
                             os.path.join(COMP, 'test.parquet'), os.path.join(COMP, 'train.parquet'),
                             os.path.join(COMP, 'sample_submission.csv'), os.path.join(WORK, 'eng_lists.json')],
                            capture_output=True, text=True, timeout=CFG['STAGE_TIMEOUT_S'], cwd=os.path.join(OURS, 'eng'), env=_env)
        print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
        ENG = json.load(open(os.path.join(WORK, 'eng_lists.json')))
        log('engine-2 lists', len(ENG), f'{time.time()-_t0:.0f}s')
    except Exception as e:
        print('ENGINE-2 FAILED -> v4 lists kept:', repr(e))
dump('eng', ENG)

# ── 6. Run PubChem-Only Channel Subprocess ─────────────────────────────────────
PC = {}
if CFG['USE_PC']:
    try:
        rr = subprocess.run([sys.executable, os.path.join(OURS, 'pc', 'pc_runner.py'),
                             V3CODE, V3MOD, POOL_DIR, PC_DIR, os.path.join(COMP, 'test.parquet'),
                             os.path.join(WORK, 'pc_lists.json'), str(CFG['PC_WORKERS'])],
                            capture_output=True, text=True, timeout=CFG['STAGE_TIMEOUT_S'],
                            env=dict(os.environ, CASMI_PC_N1=str(CFG['PC_N1'])))
        print(rr.stdout[-3000:]); print(rr.stderr[-3000:])
        PC = json.load(open(os.path.join(WORK, 'pc_lists.json')))
    except Exception as e:
        print('PUBCHEM CHANNEL FAILED -> base lists only:', repr(e))
    log('pubchem lists', len(PC))
dump('pc', PC)

# ── 7. Main Engine Setup ──────────────────────────────────────────────────────
sys.path.insert(0, V4CODE)
os.makedirs('work', exist_ok=True)
for f in ['pool_meta.parquet', 'pool_fp.npy', 'pool_frag_off.npy', 'pool_frag_mass.npy', 'fp_bits.npy', 'train_structs.parquet', 'train_fp_sel.npy']:
    if not os.path.exists(f'work/{f}'):
        os.symlink(os.path.join(POOL_DIR, f), f'work/{f}')

from casmi.build import build_spec_cache
build_spec_cache(os.path.join(COMP, 'train.parquet'), 'work')
import casmi; log('casmi from', casmi.__file__)
sys.path.insert(1, os.path.join(V4CODE, 'fe_v4'))

import torch, lightgbm as lgb
from casmi.library import Library
from casmi.pool import Pool
from casmi.engine import Engine, EngineCfg, FEATURES
from casmi import fpnet, chem
import v1engine

dev = 'cuda' if torch.cuda.is_available() else 'cpu'
L = Library('work'); P = Pool('work')
tfp = np.load('work/train_fp_sel.npy')
_FULL = glob.glob('/kaggle/input/**/casmi26-fpnet-full1/**/fpnet_full1.pt', recursive=True)
_A = os.path.join(V4MOD, 'fpnet_0.pt')

if CFG['FP_BANK'] == 'ens' and _FULL:
    _fp = [_FULL[0], _A]
elif CFG['FP_BANK'] == 'full1' and _FULL:
    _fp = [_FULL[0]]
else:
    _fp = [_A]

bank = fpnet.ModelBank(_fp, device=dev)
log('engine FP bank:', _fp)
E = Engine(L, P, tfp, EngineCfg(generate=True), bank)
V = v1engine.V1FE(E, [os.path.join(V4MOD, m['file']) for m in MAN['fe_models']], MAN['families'], device=dev)

if 'derivation' in MAN['families']:
    n_bad = V.mods['derivation'].self_test()
    log('derivation prior self-test mismatches:', n_bad)
    assert n_bad == 0, 'RDKit build differs from derivation prior'

RK = pickle.load(open(os.path.join(V4MOD, MAN['ranker']['file']), 'rb'))
RK_FEATS = list(RK['features'])
RK_B = [lgb.Booster(model_str=s) for s in RK['boosters']]

def rank_score(X, names):
    pos = {n: i for i, n in enumerate(names)}
    miss = [f for f in RK_FEATS if f not in pos]
    assert not miss, f'ranker features not produced: {miss}'
    Xs = np.ascontiguousarray(X[:, [pos[f] for f in RK_FEATS]], dtype=np.float32)
    return np.mean([b.predict(Xs, num_threads=4) for b in RK_B], axis=0)

log('engine ready; ranker features', len(RK_FEATS), 'boosters', len(RK_B))

# ── 8. Compute Base Candidate Lists ───────────────────────────────────────────
te = pd.read_parquet(os.path.join(COMP, 'test.parquet'))
te['nm'] = chem.neutral_mass(te.precursor_mz.values.astype(np.float64), te.adduct.values)
mols = list(te.groupby('molecule_id', sort=False))
BASE, n_err = {}, 0

for gi, (mid, sub) in enumerate(mols):
    nms = sub.nm.values[np.isfinite(sub.nm.values)]
    smis, keys, lib_max, scs, forms, pids = [], [], 0.0, [], [], []
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
                    if k in seen:
                        continue
                    seen.add(k); smis.append(C.smiles[i]); keys.append(k); scs.append(float(score[i])); forms.append(C.formula[i]); pids.append(int(C.pid[i]))
                    if len(smis) >= CFG['TOPN']:
                        break
        except Exception as e:
            print('ERROR', mid, repr(e)); n_err += 1
    BASE[mid] = [smis, keys, lib_max, scs, forms, pids]
    if (gi + 1) % 25 == 0 or gi == len(mols) - 1:
        log(f' {gi+1}/{len(mols)} molecules')

log('base lists done, errors', n_err)
dump('base', BASE)

# ── 9. Forward Models (ICEBERG + GLACIER) & MetFrag-lite Rescoring ─────────────
from rdkit import Chem as _Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula as _CMF

def formula_of(s):
    try:
        return _CMF(_Chem.MolFromSmiles(s))
    except Exception:
        return s

ICE_SCORES, GL_SCORES, items, ice_fuse, gl_fuse = {}, {}, None, None, None
TOPN, LIB_TAU = CFG['TOPN'], CFG['LIB_TAU']

if CFG['USE_ICE']:
    try:
        ICE_PKG = os.path.dirname(find('casmi26-iceberg/**/ice_runner.py'))
        sys.path.append(ICE_PKG)
        import fuse as ice_fuse
        del V, E, bank; gc.collect(); torch.cuda.empty_cache()
        mids = sorted(BASE, key=lambda m: BASE[m][2])
        cands = {m: ice_fuse.ice_candidates(BASE[m][0], BASE[m][4], TOPN) for m in mids}
        if ENG and CFG['ICE_UNION']:
            K, n_add = CFG['ICE_UNION_K'], 0
            for m in mids:
                e = ENG.get(str(m))
                if not e or not e.get('smiles'):
                    continue
                bs, bk, bf = list(BASE[m][0][:TOPN]), set(BASE[m][1][:TOPN]), list(BASE[m][4][:TOPN])
                es = [s for s, k in zip(e['smiles'][:K], e['keys'][:K]) if k not in bk]
                extra = ice_fuse.ice_candidates(bs + es, bf + [formula_of(s) for s in es], len(bs) + len(es))
                before = len(cands.get(m, []))
                cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
                n_add += len(cands[m]) - before
            log('ICE_UNION: engine-2 candidates added:', n_add)
        if CFG['ICE_PC']:
            for m in mids:
                p = PC.get(m) if isinstance(PC, dict) else None
                if p and p.get('pc') and BASE[m][2] < LIB_TAU:
                    p['pc_form'] = [formula_of(x) for x in p['pc']]
                    extra = ice_fuse.ice_candidates(p['pc'], p['pc_form'], 25)
                    cands[m] = list(dict.fromkeys(list(cands.get(m, [])) + extra))
        cands = {m: c for m, c in cands.items() if c}
        items = ice_fuse.build_ice_input(te, cands)
        log('ICE input molecules', len(items), 'candidates', sum(len(it['cands']) for it in items))
        ICE_SCORES = ice_fuse.run_ice(ICE_PKG, items, workdir=os.path.join(WORK, 'ice_work'), device='cuda',
                                      budget_s=ICE_BUDGET, site=os.path.join(WORK, 'ice_site'))
        meta_f = os.path.join(WORK, 'ice_work', 'ice_out.json.meta.json')
        if os.path.exists(meta_f):
            print('ICE meta', open(meta_f).read()[:1500])
    except Exception as e:
        ICE_SCORES = {}
        print('ICE FAILED -> ranker order kept:', repr(e))

if CFG['USE_GL'] and items:
    try:
        GL_PKG = os.path.dirname(find('casmi26-glacier/**/gl_runner.py'))
        sys.path.append(GL_PKG)
        import gl_fuse
        gc.collect(); torch.cuda.empty_cache()
        GL_SCORES = gl_fuse.run_gl(GL_PKG, items, workdir=os.path.join(WORK, 'gl_work'), device='cuda',
                                   budget_s=CFG['GL_BUDGET'], site=os.path.join(WORK, 'ice_site'),
                                   wheels=os.path.join(ICE_PKG, 'wheels'))
        meta_f = os.path.join(WORK, 'gl_work', 'gl_out.json.meta.json')
        print('GL meta', open(meta_f).read()[:1500] if os.path.exists(meta_f) else 'missing')
    except Exception as e:
        GL_SCORES = {}
        print('GL FAILED -> ICE only:', repr(e))

FRAG = {}
if CFG['FRAG_LAM'] > 0:
    try:
        import frag_rescore
        _t0 = time.time()
        for mid, _sub in te.groupby('molecule_id', sort=False):
            if mid not in BASE:
                continue
            _spec = [dict(mz=np.asarray(r.ms2_mzs, np.float64), it=np.asarray(r.ms2_normalized_intensities, np.float64),
                          adduct=r.adduct, prec=float(r.precursor_mz), mode=1 if r.ionization_mode == 'positive' else -1)
                     for r in _sub.itertuples()][:16]
            _c = list(dict.fromkeys(list(BASE[mid][0]) + list((ENG.get(str(mid)) or {}).get('smiles', [])) + list((PC.get(mid) or {}).get('pc', []))))
            FRAG[mid] = dict(zip(_c, frag_rescore.score_molecule(_spec, _c)))
        log('frag scores', len(FRAG), 'molecules', f'{time.time()-_t0:.0f}s')
    except Exception as e:
        FRAG = {}
        print('FRAG FAILED -> off:', repr(e))

log('ICE molecules', len(ICE_SCORES), '| GL molecules', len(GL_SCORES), '| FRAG molecules', len(FRAG))
dump('ice', ICE_SCORES); dump('gl', GL_SCORES); dump('frag', FRAG); dump('pc', PC)

# ── 10. Popularity Prior & Multi-Stage Final Fusion ────────────────────────────
samp = pd.read_csv(os.path.join(COMP, 'sample_submission.csv'))
MOL_ORDER = list(te.molecule_id.unique())
dump('meta', dict(mol_order=MOL_ORDER, sample_ids=list(samp.molecule_id), is_rerun=IS_RERUN, cfg=CFG))

POOL_POP = None
if CFG['POP_MU'] > 0:
    try:
        _pp = os.path.dirname(find('pool_lsid.npy'))
        POOL_POP = np.load(os.path.join(_pp, 'pool_lsid.npy')).astype(np.float64) + \
                   np.load(os.path.join(_pp, 'pool_lpmid.npy')).astype(np.float64)
        log('popularity prior loaded', _pp, POOL_POP.shape)
    except Exception as e:
        print('POPULARITY PRIOR UNAVAILABLE -> off:', repr(e))

sub, FSTATS = fusion_core.build_submission(
    BASE, PC, ENG, ICE_SCORES, GL_SCORES, CFG, MOL_ORDER, list(samp.molecule_id),
    ice_fuse=ice_fuse, gl_fuse=gl_fuse, score_key=chem.score_key, formula=formula_of,
    log=print, pool_pop=POOL_POP, frag_scores=FRAG)

fusion_core.validate(sub, list(samp.molecule_id))
sub.to_csv(os.path.join(WORK, 'submission.csv'), index=False)
json.dump(dict(version=CFG['VERSION'], secs=round(time.time() - T0), stats=FSTATS, n_errors=n_err,
               families=MAN['families'], gpu=torch.cuda.get_device_name(0) if dev == 'cuda' else None, cfg=CFG),
          open(os.path.join(WORK, 'run_manifest.json'), 'w'), indent=1)

log('fusion stats', FSTATS)
log('done submission generated:', sub.shape)

import shutil
shutil.rmtree(os.path.join(WORK, 'ice_site'), ignore_errors=True)

print("\n--- Top 5 Sample Predictions ---")
print(sub.head())
