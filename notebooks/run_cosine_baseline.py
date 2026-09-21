import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from pathlib import Path
import time

DATA = Path("/home/dlcv/Enveda-CASMI-2026/data")
OUT  = Path("/home/dlcv/Enveda-CASMI-2026")
t0   = time.time()

MZ_TOL, MIN_PEAKS, INT_THRESH, PRECURSOR_BUF = 0.05, 3, 0.005, 2.0
INDEX_LIBS = {"enveda-np-examples","riken","gnps","mona","massbank","spectraverse","msdial","masaryk","pluskal_ms2"}
MASS_MIN, MASS_MAX = 150.0, 600.0

def log(msg): print(f"[{time.time()-t0:.0f}s] {msg}", flush=True)

def preprocess(mzs, ints, pmz):
    mzs  = np.asarray(mzs,  dtype=np.float32)
    ints = np.asarray(ints, dtype=np.float32)
    if len(mzs)==0: return mzs, ints
    mx=ints.max();
    if mx>0: ints=ints/mx
    keep=mzs<=(pmz+PRECURSOR_BUF); mzs,ints=mzs[keep],ints[keep]
    keep=ints>=INT_THRESH; mzs,ints=mzs[keep],ints[keep]
    if len(mzs)>128:
        top=np.argsort(ints)[-128:]; mzs,ints=mzs[top],ints[top]
    ints=np.sqrt(ints); mx=ints.max()
    if mx>0: ints=ints/mx
    return mzs, ints

def cosine(mzs1, ints1, mzs2, ints2):
    used2=np.zeros(len(mzs2),bool); m1,m2=[],[]
    for i in range(len(mzs1)):
        diffs=np.abs(mzs2-mzs1[i]); mask=(diffs<=MZ_TOL)&(~used2)
        if mask.any():
            j=int(np.argmin(np.where(mask,diffs,np.inf)))
            m1.append(ints1[i]); m2.append(ints2[j]); used2[j]=True
    if not m1: return 0.0,0
    m1,m2=np.array(m1),np.array(m2)
    norm=np.linalg.norm(ints1)*np.linalg.norm(ints2)
    return (float(np.dot(m1,m2)/norm) if norm>0 else 0.0), len(m1)

log("Loading test...")
test = pd.read_parquet(str(DATA/"test.parquet"))
log(f"  {len(test)} spectra, {test['molecule_id'].nunique()} molecules")

log("Loading train spectra (NP libs + mass range)...")
COLS=["ingest_lib","inchikey14","normalized_smiles","precursor_mz","ms2_mzs","ms2_normalized_intensities","num_peaks"]
train = pd.read_parquet(str(DATA/"train.parquet"), columns=COLS)
mask=(train["ingest_lib"].isin(INDEX_LIBS) & train["precursor_mz"].between(MASS_MIN,MASS_MAX) & (train["num_peaks"]>=MIN_PEAKS))
idx_df = train[mask].reset_index(drop=True)
del train
log(f"  {len(idx_df):,} spectra to index from {idx_df['ingest_lib'].nunique()} libs")
log(f"  Library counts:\n{idx_df['ingest_lib'].value_counts().to_string()}")

log("Building index...")
LIB_M, LIB_I, LIB_META = [], [], []
for _, row in idx_df.iterrows():
    m,i=preprocess(row["ms2_mzs"],row["ms2_normalized_intensities"],row["precursor_mz"])
    if len(m)<MIN_PEAKS: continue
    LIB_M.append(m); LIB_I.append(i)
    LIB_META.append({"k":row["inchikey14"],"s":row["normalized_smiles"],"p":float(row["precursor_mz"])})
N=len(LIB_META)
log(f"  {N:,} valid library spectra indexed")

log(f"Predicting {test['molecule_id'].nunique()} molecules...")
predictions={}
mol_ids=test["molecule_id"].unique()

for mi, mol_id in enumerate(mol_ids):
    rows=test[test["molecule_id"]==mol_id]
    best={}; sm={}
    for _,row in rows.iterrows():
        qm,qi=preprocess(row["ms2_mzs"],row["ms2_normalized_intensities"],row["precursor_mz"])
        if len(qm)<MIN_PEAKS: continue
        for li in range(N):
            sc,nm=cosine(qm,qi,LIB_M[li],LIB_I[li])
            if sc>0 and nm>=MIN_PEAKS:
                k=LIB_META[li]["k"]
                if sc>best.get(k,0.0): best[k]=sc; sm[k]=LIB_META[li]["s"]
    ranked=sorted(best.items(),key=lambda x:-x[1])
    slist=[sm[k] for k,_ in ranked[:25]]
    predictions[mol_id]=slist if slist else ["CCO"]*25
    if (mi+1)%50==0: log(f"  {mi+1}/{len(mol_ids)} done")

out=[{"molecule_id":mid,"smiles":";".join(s[:25])} for mid,s in predictions.items()]
df_out=pd.DataFrame(out)
p=str(OUT/"submission_cosine_baseline.csv")
df_out.to_csv(p,index=False)
log(f"DONE → {p}  ({len(df_out)} rows)")
n_hits=sum(1 for s in predictions.values() if s[0]!="CCO")
log(f"  Molecules with >=1 library hit: {n_hits}/{len(predictions)} ({n_hits/len(predictions)*100:.1f}%)")
