"""Fast runtime integration check before the expensive training job."""
import tempfile,json
from pathlib import Path
import numpy as np
import torch
from features import structure,peaks,metadata
from model import SpectrumCandidateModel
from inference import Scorer

def self_test(cfg):
    tiny={**cfg,'width':48,'heads':4,'layers':1,'embedding':32,'dropout':0.}
    smis=['CCO','CCCO','CC(=O)O']
    fps=np.stack([structure(s,tiny['fp_bits'])[-1] for s in smis])
    s=dict(mzs=[15.,29.,43.,45.,60.],intensities=[.1,.2,1.,.4,.1],precursor_mz=61.,adduct='[M+H]+',ionization_mode='positive',collision_energy=20.)
    p,l=peaks(s['mzs'],s['intensities'],61.,tiny['max_peaks']);m,a=metadata(61.,s['adduct'],'positive',20.)
    model=SpectrumCandidateModel(tiny);model.train()
    x=torch.tensor(np.stack([p,p]));length=torch.tensor([l,l]);meta=torch.tensor(np.stack([m,m]));adduct=torch.tensor([a,a])
    z,logits=model.encode_spectra(x,length,meta,adduct)
    loss=-(z*model.encode_molecules(torch.tensor(fps[:2],dtype=torch.float32))).sum()+logits.square().mean()
    loss.backward();assert torch.isfinite(loss)
    opt=torch.optim.AdamW(model.parameters(),lr=1e-4);opt.step();model.eval()
    with torch.no_grad():
        z,logits=model.encode_spectra(x[:1],length[:1],meta[:1],adduct[:1]);fp=torch.tensor(fps,dtype=torch.float32)
        expected=(model.encode_molecules(fp)@z[0]/tiny['temperature']+tiny['fp_score_weight']*(fp@logits[0])/tiny['fp_bits']).numpy()
    with tempfile.TemporaryDirectory() as d:
        root=Path(d);(root/'config.json').write_text(json.dumps(tiny));torch.save({'state_dict':model.state_dict()},root/'final_model.pt')
        scorer=Scorer(root,device='cpu');actual=scorer.score_candidates([s],smis)
        np.testing.assert_allclose(actual,expected,atol=2e-5,rtol=2e-5)
        assert np.isneginf(scorer.score_candidates([s],['this is invalid'])[0])
    print('PASS: features, forward/backward, optimizer step, checkpoint reload, inference score parity')
