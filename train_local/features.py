"""Shared training/inference features. No notebook globals required."""
import ast, json
import numpy as np
ADDUCTS = ['[M+H]+','[M-H]-','[M+Na]+','[M+K]+','[M+NH4]+','[M+Cl]-','[M+FA-H]-','[M+H-H2O]+','[M+2H]2+','[M-2H]2-']
def array_value(x):
    if isinstance(x,str):
        try: x=json.loads(x)
        except (ValueError,TypeError):
            try:x=ast.literal_eval(x)
            except (ValueError,SyntaxError):x=x.replace(',',' ').split()
    return np.asarray(x,dtype=np.float32).reshape(-1)
def peaks(mzs,intensities,precursor,max_peaks=96):
    mz,it=array_value(mzs),array_value(intensities)
    if len(mz)!=len(it): raise ValueError('Peak and intensity lengths differ')
    keep=np.isfinite(mz)&np.isfinite(it)&(mz>0)&(it>0)
    mz,it=mz[keep],it[keep]
    if not len(mz):raise ValueError('No valid peaks')
    ix=np.argsort(-it,kind='stable')[:max_peaks];ix=ix[np.argsort(mz[ix],kind='stable')]
    mz,it=mz[ix],it[ix];it=it/it.max()
    out=np.zeros((max_peaks,4),np.float32)
    out[:len(mz)]=np.stack([mz/1000,np.sqrt(it),np.log1p(100*it)/np.log(101), (float(precursor)-mz)/1000],1)
    return out,len(mz)
def metadata(precursor,adduct,mode,ce):
    try:
        a=array_value(ce);a=a[np.isfinite(a)];ev=float(a.mean()) if len(a) else 0.;known=float(len(a)>0)
    except (ValueError,TypeError):ev,known=0.,0.
    mode=str(mode).lower();pol=1. if mode in ('positive','pos','1','+') else -1. if mode in ('negative','neg','-1','-') else (1. if str(adduct).endswith('+') else -1. if str(adduct).endswith('-') else 0.)
    return np.array([float(precursor)/1000,np.clip(ev,0,500)/100,known,pol],np.float32),ADDUCTS.index(adduct)+1 if adduct in ADDUCTS else 0

def structure(smiles,nbits=2048,pre_key=None,pre_formula=None):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator,rdMolDescriptors
    mol=Chem.MolFromSmiles(str(smiles))
    if mol is None:raise ValueError('Invalid structure')
    Chem.RemoveStereochemistry(mol)
    if pre_key and len(str(pre_key)) >= 14:
        key = str(pre_key)[:14]
    else:
        from rdkit.Chem.MolStandardize import rdMolStandardize
        tmol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
        key = Chem.MolToInchiKey(tmol)[:14]
    if not key:raise ValueError('No InChIKey')
    formula = str(pre_formula) if pre_formula else rdMolDescriptors.CalcMolFormula(mol)
    mass = rdMolDescriptors.CalcExactMolWt(mol)
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=nbits).GetFingerprintAsNumPy(mol)
    return key,Chem.MolToSmiles(mol),formula,mass,fp
