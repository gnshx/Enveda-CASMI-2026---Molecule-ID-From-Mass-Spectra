import os,time,json,random,collections
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F
from model import SpectrumCandidateModel

def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def atomic_save(obj,path):
    path=Path(path);tmp=path.with_suffix('.tmp');torch.save(obj,tmp);os.replace(tmp,path)

def hard_pools(mol,allowed,count):
    """Only allowed training identities appear in each training negative pool."""
    allowed=np.asarray(allowed);groups=collections.defaultdict(list)
    formulas=mol.formula.values;mass_values=mol.mass.values
    for i in allowed:groups[formulas[i]].append(int(i))
    order=allowed[np.argsort(mass_values[allowed],kind='stable')];masses=mass_values[order];pools={}
    for i in allowed:
        same=[j for j in groups[formulas[i]] if j!=i]
        pos=int(np.searchsorted(masses,mass_values[i]))
        nearby=order[max(0,pos-count):min(len(order),pos+count+1)].tolist()
        # closest mass first, then same-formula candidates prioritized.
        nearby.sort(key=lambda j:(abs(mass_values[j]-mass_values[i]),j))
        seen={int(i)};out=[]
        for j in same+nearby:
            if j not in seen:seen.add(j);out.append(j)
            if len(out)>=count:break
        assert out,'Need at least two distinct training molecules'
        pools[int(i)]=np.asarray(out,np.int64)
    return pools

class Data:
    def __init__(self,cfg):
        root=Path(cfg['work']);self.cfg=cfg
        self.mol=pd.read_parquet(root/'molecules.parquet');self.spec=pd.read_parquet(root/'spectra.parquet')
        n=len(self.spec)
        self.x=np.memmap(root/'peaks.bin',mode='r',dtype='<f2',shape=(n,cfg['max_peaks'],4))
        self.meta=np.memmap(root/'metadata.bin',mode='r',dtype='<f4',shape=(n,4))
        self.fp=np.load(root/'fingerprints.npy',mmap_mode='r')
        self.lengths=self.spec.length.values;self.adduct=self.spec.adduct_id.values
        self.by_mol={int(i):g.index.to_numpy() for i,g in self.spec.groupby('mol_idx',sort=False)}
        self.train=np.where(self.mol.split.eq('train'))[0]
        self.val=np.where(self.mol.split.eq('validation'))[0]
        assert not set(self.train)&set(self.val)
    def batch(self,rows,device,augment=False):
        x=torch.tensor(np.asarray(self.x[rows]),device=device,dtype=torch.float32)
        if augment:
            # Intensity jitter only; retain peak positions and padding masks.
            it=(x[:,:,1].square()*torch.exp(.08*torch.randn_like(x[:,:,1]))).clamp_min(0)
            it=it/it.max(dim=1,keepdim=True).values.clamp_min(1e-8)
            x[:,:,1]=it.sqrt();x[:,:,2]=torch.log1p(100*it)/np.log(101)
        return (x,torch.tensor(self.lengths[rows],device=device,dtype=torch.long),
            torch.tensor(np.asarray(self.meta[rows]),device=device),torch.tensor(self.adduct[rows],device=device,dtype=torch.long))

def candidate_sets(data,cfg):
    # Fixed, deliberately artificial retrieval diagnostic; never advertised as competition MRR.
    # True structure + same-formula decoys first, then nearby-mass known structures.
    all_ids=np.arange(len(data.mol));pools=hard_pools(data.mol,all_ids,cfg['val_candidates']-1)
    return {int(i):np.r_[i,pools[int(i)]] for i in data.val[:cfg['max_validation_molecules']]}

@torch.inference_mode()
def validate(model,data,candidates,device,cfg):
    model.eval();rows=[]
    for i,ids in candidates.items():
        # Only held-out COMPETITION spectra score validation; external replicas never improve the measured query.
        sr=data.by_mol[i];sr=sr[data.spec.source.values[sr]==0]
        if not len(sr):continue
        zsum=0;psum=0;n=0
        for start in range(0,len(sr),cfg['batch_size']):
            batch=sr[start:start+cfg['batch_size']]
            z,p=model.encode_spectra(*data.batch(batch,device));zsum=zsum+z.sum(0);psum=psum+p.sum(0);n+=len(batch)
        fp=torch.tensor(np.asarray(data.fp[ids]),device=device,dtype=torch.float32)
        scores=((model.encode_molecules(fp)@(zsum/n))/cfg['temperature']+cfg['fp_score_weight']*(fp@(psum/n))/cfg['fp_bits']).cpu().numpy()
        # Pessimistic ties: target must strictly beat other candidates (gold is at index zero).
        rank=1+int(np.sum(scores[1:]>=scores[0]))
        rows.append(dict(key=data.mol.key.iloc[i],rank=rank,mrr25=1/rank if rank<=25 else 0.,top1=int(rank==1),
            recall25=int(rank<=25),candidate_count=len(ids),spectra=n,
            same_formula_decoys=int(np.sum(data.mol.formula.values[ids[1:]]==data.mol.formula.values[i]))))
    frame=pd.DataFrame(rows);assert len(frame)
    return float(frame.mrr25.mean()),frame

def train_stage(data,cfg,out,device,*,refit=False,epochs=None,deadline=None):
    seed_all(cfg['seed']+(1 if refit else 0));rng=np.random.default_rng(cfg['seed']+(1 if refit else 0))
    ids=np.arange(len(data.mol)) if refit else data.train
    assert len(ids)>=2
    pools=hard_pools(data.mol,ids,cfg['negative_pool'])
    model=SpectrumCandidateModel(cfg).to(device)
    opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'])
    use_amp=device.type=='cuda';scaler=torch.amp.GradScaler('cuda',enabled=use_amp)
    # Training-only statistics, so holdout fingerprints do not influence the auxiliary loss.
    counts=np.zeros(cfg['fp_bits'],np.float64)
    for st in range(0,len(ids),4096):counts+=np.asarray(data.fp[ids[st:st+4096]]).sum(0)
    pos_weight=torch.tensor(np.clip((len(ids)-counts)/(counts+1),1,20),device=device,dtype=torch.float32)
    candidates=None if refit else candidate_sets(data,cfg)
    if not refit:
        np.savez_compressed(out/'validation_candidates.npz',**{str(i):v for i,v in candidates.items()})
    best=-1.;best_epoch=0;bad=0;history=[];max_epochs=epochs or cfg['epochs'];stopped=False
    for ep in range(1,max_epochs+1):
        model.train();losses=[];start=time.monotonic()
        # Balanced by molecule, not by number of spectra; one randomly selected spectrum per molecule.
        for step in range(cfg['steps_per_epoch']):
            if deadline is not None and time.monotonic()>deadline:stopped=True;break
            batch_ids=rng.choice(ids,min(cfg['batch_size'],len(ids)),replace=False)
            sr=np.array([rng.choice(data.by_mol[int(i)]) for i in batch_ids])
            neg=np.stack([rng.choice(pools[int(i)],cfg['hard_negatives'],replace=len(pools[int(i)])<cfg['hard_negatives']) for i in batch_ids])
            # Candidate columns are all in-batch positives plus per-query hard negatives.
            fp=torch.tensor(np.asarray(data.fp[batch_ids]),device=device,dtype=torch.float32)
            negfp=torch.tensor(np.asarray(data.fp[neg.reshape(-1)]),device=device,dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp):
                z,pred=model.encode_spectra(*data.batch(sr,device,augment=True))
                m=model.encode_molecules(fp);nm=model.encode_molecules(negfp).reshape(len(batch_ids),cfg['hard_negatives'],-1)
                positive=z@m.T/cfg['temperature'];negative=torch.einsum('bd,bkd->bk',z,nm)/cfg['temperature']
                # Mask repeated candidates: each identity gets one vote in the softmax denominator.
                dup=np.isin(neg,batch_ids)
                for r in range(len(neg)):
                    seen=set()
                    for j,k in enumerate(neg[r]):
                        if int(k) in seen:dup[r,j]=True
                        seen.add(int(k))
                negative=negative.masked_fill(torch.tensor(dup,device=device),float('-inf'))
                retrieval=F.cross_entropy(torch.cat([positive,negative],1),torch.arange(len(batch_ids),device=device))
                auxiliary=F.binary_cross_entropy_with_logits(pred,fp,pos_weight=pos_weight)
                loss=retrieval+cfg['aux_weight']*auxiliary
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            scaler.step(opt);scaler.update();losses.append(float(loss.detach()))
        if not losses:break
        checkpoint=dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},epoch=ep,stage='full_refit' if refit else 'holdout')
        record=dict(epoch=ep,loss=float(np.mean(losses)),steps=len(losses),seconds=time.monotonic()-start,partial_epoch=stopped)
        if refit:
            atomic_save(checkpoint,out/'final_model.pt')
        else:
            score,frame=validate(model,data,candidates,device,cfg);record.update(diagnostic_mrr25=score,top1=float(frame.top1.mean()))
            atomic_save(checkpoint,out/'last_validation_model.pt')
            if score>best:
                best=score;best_epoch=ep;bad=0
                atomic_save(checkpoint,out/'best_validation_model.pt');frame.to_csv(out/'validation_predictions.csv',index=False)
            else:bad+=1
        history.append(record)
        (out/('refit_history.json' if refit else 'training_history.json')).write_text(json.dumps(history,indent=2))
        print('REFIT' if refit else 'HOLDOUT',record,flush=True)
        if stopped or (not refit and bad>=cfg['patience']):break
    del model,opt,scaler
    if device.type=='cuda':torch.cuda.empty_cache()
    return dict(best_epoch=best_epoch,best_diagnostic_mrr25=best if not refit else None,completed_epochs=len(history),time_budget_exhausted=stopped)

def run(cfg):
    root=Path(cfg['output']);root.mkdir(parents=True,exist_ok=True)
    assert not list(root.glob('*.pt')), 'Existing checkpoints found; choose a new OUTPUT directory to avoid mixing runs'
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if cfg['require_gpu'] and device.type!='cuda':raise RuntimeError('Enable a Kaggle GPU accelerator before training')
    torch.set_num_threads(min(4,os.cpu_count() or 1));data=Data(cfg)
    # Training budgets separate holdout selection from full-data refit. Final epoch may be partial if budget expires.
    result=train_stage(data,cfg,root,device,deadline=time.monotonic()+cfg['holdout_hours']*3600)
    assert (root/'best_validation_model.pt').exists(),'No completed validation; increase time budget'
    if cfg['refit_full']:
        result['refit']=train_stage(data,cfg,root,device,refit=True,epochs=result['best_epoch'],deadline=time.monotonic()+cfg['refit_hours']*3600)
    if not (root/'final_model.pt').exists():
        import shutil;shutil.copy2(root/'best_validation_model.pt',root/'final_model.pt')
        result['final_model_training']='holdout training only (no successful full refit)'
    else:result['final_model_training']='full data, including former validation identities; DO NOT evaluate this model on that holdout'
    (root/'training_result.json').write_text(json.dumps(result,indent=2));return result
