"""Example: scorer=Scorer(bundle); scorer.score_candidates(spectra, smiles).
Spectra keys: mzs, intensities, precursor_mz, adduct, ionization_mode, collision_energy.
Returns candidate-aligned scores (larger is better); invalid candidates receive -inf.
This scores supplied candidates; it does not retrieve a database or create a submission.
"""
import json
from pathlib import Path
import numpy as np
import torch
from features import peaks,metadata,structure
from model import SpectrumCandidateModel
class Scorer:
    def __init__(self,bundle,checkpoint='final_model.pt',device=None):
        root=Path(bundle);self.cfg=json.loads((root/'config.json').read_text())
        self.device=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model=SpectrumCandidateModel(self.cfg).to(self.device)
        obj=torch.load(root/checkpoint,map_location=self.device,weights_only=True)
        self.model.load_state_dict(obj['state_dict']);self.model.eval();self.cache={}
    @torch.inference_mode()
    def score_candidates(self,spectra,candidates):
        if not spectra:raise ValueError('At least one spectrum required')
        records=[]
        for s in spectra:
            p,n=peaks(s['mzs'],s['intensities'],s['precursor_mz'],self.cfg['max_peaks'])
            m,a=metadata(s['precursor_mz'],s.get('adduct',''),s.get('ionization_mode',''),s.get('collision_energy'))
            records.append((p,n,m,a))
        zs=[];ps=[]
        for start in range(0,len(records),64):
            b=records[start:start+64]
            tensors=[torch.as_tensor(np.array([r[j] for r in b]),device=self.device) for j in range(4)]
            z,p=self.model.encode_spectra(tensors[0].float(),tensors[1].long(),tensors[2].float(),tensors[3].long());zs.append(z);ps.append(p)
        z=torch.cat(zs).mean(0);p=torch.cat(ps).mean(0)
        out=np.full(len(candidates),-np.inf,np.float32)
        for start in range(0,len(candidates),512):
            valid=[];fps=[]
            for i,s in enumerate(candidates[start:start+512],start):
                if s not in self.cache:
                    try:self.cache[s]=structure(s,self.cfg['fp_bits'])[-1]
                    except (ValueError,RuntimeError):self.cache[s]=None
                if self.cache[s] is not None:valid.append(i);fps.append(self.cache[s])
            if not valid:continue
            fp=torch.as_tensor(np.array(fps),device=self.device,dtype=torch.float32)
            # Equal to averaged per-spectrum score, as in validation.
            score=(self.model.encode_molecules(fp)@z)/self.cfg['temperature']+self.cfg['fp_score_weight']*(fp@p)/self.cfg['fp_bits']
            out[valid]=score.cpu().numpy()
        return out
