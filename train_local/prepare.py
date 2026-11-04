"""Prepare competition parquet plus explicitly configured external parquet/TSV files."""
import hashlib,json,collections
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from features import peaks,metadata,structure
ALIASES={'smiles':['smiles','canonical_smiles','normalized_smiles'],'mzs':['ms2_mzs','mzs'],
'intensities':['ms2_normalized_intensities','intensities'], 'precursor':['precursor_mz'],
'adduct':['adduct'],'mode':['ionization_mode','ion_mode'],'ce':['collision_energy_ev','collision_energy']}
def batches(path):
    if str(path).endswith('.parquet'):
        for b in pq.ParquetFile(path).iter_batches(batch_size=8192):yield b.to_pandas()
    elif str(path).endswith(('.tsv','.tsv.gz')):
        yield from pd.read_csv(path,sep='\t',chunksize=8192)
    else:raise ValueError('Supported inputs: .parquet, .tsv, .tsv.gz; convert raw MSP/MGF first: '+str(path))
def prepare(cfg):
    root=Path(cfg['work']);root.mkdir(parents=True,exist_ok=True)
    cache={};molecules={};rows=[];reject=collections.Counter();stats=[];seen=set()
    # Binary spools avoid holding all peak tensors in RAM.
    px=open(root/'peaks.bin','wb');mx=open(root/'metadata.bin','wb')
    try:
        for source_index,path in enumerate([cfg['train_path']]+cfg['external_paths']):
            path=Path(path);assert path.is_file(),path
            accepted=0;total=0;columns=None
            for frame in batches(path):
                mapping={k:next((c for c in names if c in frame.columns),None) for k,names in ALIASES.items()}
                assert all(mapping[k] for k in ['smiles','mzs','intensities','precursor']), ('Missing columns',path,list(frame.columns))
                columns=list(frame.columns)
                for row in frame.to_dict('records'):
                    total+=1;s=str(row[mapping['smiles']])
                    try:
                        if s not in cache:
                            try:cache[s]=structure(s,cfg['fp_bits'],pre_key=row.get('inchikey14'),pre_formula=row.get('molecular_formula'))
                            except (ValueError,RuntimeError):cache[s]=None
                        st=cache[s]
                        if st is None:reject['invalid_structure']+=1;continue
                        key,canonical,formula,mass,fp=st
                        precursor=float(row[mapping['precursor']])
                        if not np.isfinite(precursor) or precursor<=0:reject['invalid_precursor']+=1;continue
                        pk,length=peaks(row[mapping['mzs']],row[mapping['intensities']],precursor,cfg['max_peaks'])
                        if length<cfg['min_peaks']:reject['too_few_peaks']+=1;continue
                        ad=row.get(mapping['adduct'],'');mo=row.get(mapping['mode'],'');ce=row.get(mapping['ce'],None)
                        mt,ai=metadata(precursor,ad,mo,ce)
                        # Deduplication after model preprocessing, including acquisition metadata.
                        digest=hashlib.sha256(key.encode()+pk.astype('<f2').tobytes()+mt.astype('<f4').tobytes()+bytes([ai])).digest()
                        if digest in seen:reject['duplicate_model_input']+=1;continue
                        seen.add(digest)
                        if key not in molecules:molecules[key]=(canonical,formula,mass,fp)
                        pk.astype('<f2').tofile(px);mt.astype('<f4').tofile(mx)
                        rows.append((key,length,ai,source_index,str(row.get('molecule_id','')),str(row.get('spectrum_id',row.get('identifier','')))))
                        accepted+=1
                        if accepted % 50000 == 0:
                            print(f'Progress: accepted {accepted:,} spectra ({len(molecules):,} unique molecules)...', flush=True)
                    except (ValueError,TypeError,OverflowError):reject['invalid_spectrum']+=1
            digest=hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda:handle.read(8*1024*1024),b''):digest.update(chunk)
            stats.append(dict(path=str(path),sha256=digest.hexdigest(),input_rows=total,accepted=accepted,columns=columns))
            print('PREPARED',stats[-1]['path'],accepted,'/',total,flush=True)
    finally:px.close();mx.close()
    assert rows,'No valid paired spectra found'
    keys=sorted(molecules);lookup={k:i for i,k in enumerate(keys)}
    mol=pd.DataFrame([(k,*molecules[k][:3]) for k in keys],columns=['key','smiles','formula','mass'])
    fps=np.stack([molecules[k][3] for k in keys]).astype(np.uint8);np.save(root/'fingerprints.npy',fps)
    spec=pd.DataFrame(rows,columns=['key','length','adduct_id','source','original_molecule_id','original_spectrum_id'])
    spec['mol_idx']=spec.key.map(lookup)
    # Choose validation identities from competition only; exclude ALL their external spectra from training.
    comp=sorted(spec.loc[spec.source.eq(0),'key'].unique())
    assert len(comp)>=20,'Need at least 20 competition identities for meaningful holdout'
    order=sorted(comp,key=lambda k:hashlib.sha256((str(cfg['seed'])+k).encode()).hexdigest())
    nv=max(2,int(len(comp)*cfg['val_fraction']));holdout=set(order[:nv])
    mol['split']=['validation' if k in holdout else 'train' for k in keys]
    spec['split']=spec.key.map(dict(zip(mol.key,mol.split)))
    mol.to_parquet(root/'molecules.parquet',index=False);spec.to_parquet(root/'spectra.parquet',index=False)
    report=dict(sources=stats,rejected=dict(reject),spectra=len(spec),molecules=len(mol),
        competition_identities=len(comp),validation_identities=len(holdout),
        excluded_external_validation_spectra=int(((spec.source>0)&spec.split.eq('validation')).sum()),
        unique_external_only_identities=len(set(keys)-set(comp)),
        identity='stereo-stripped tautomer-canonical InChIKey14; no salt stripping or uncharging',
        limitation='Random molecular holdout, not scaffold-disjoint; no hidden test data read.')
    (root/'data_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    return report
