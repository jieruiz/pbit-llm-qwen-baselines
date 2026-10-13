"""Fixed power-of-two amplitude ranges from first 8192 WT2 train tokens only."""
import json, os
import torch
from torch.nn import functional as F
from common import ROOT,baseline

def main():
    assert os.environ.get('SLURM_JOB_ID'),'GPU work requires Slurm'
    out=ROOT/'calibration.json'
    if out.exists():print('CALIBRATION_EXISTS',out,flush=True);return
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    tok,model,_=baseline.load();model.requires_grad_(False)
    data=baseline.DATA/'wikitext-2/wiki.train.raw'
    ids=tok(data.read_text(),add_special_tokens=False,return_tensors='pt',verbose=False).input_ids
    total=ids.shape[-1];ids=ids[:,:8192];assert total>8192+65536
    maxima={};handles=[]
    def record(i,signal,x):
        mx=x.detach().float().abs().reshape(-1,x.shape[-1]//64,64).amax((0,2)).cpu()
        key=(i,signal);maxima[key]=mx if key not in maxima else torch.maximum(maxima[key],mx)
    for i,layer in enumerate(model.model.language_model.layers):
        def hook(module,args,index=i):
            x=args[0].float();record(index,'x',x)
            record(index,'g',F.linear(x,module.gate_proj.weight.float()))
            record(index,'u',F.linear(x,module.up_proj.weight.float()))
        handles.append(layer.mlp.register_forward_pre_hook(hook))
    with torch.inference_mode():
        for first in range(0,8192,2048):
            model.model(input_ids=ids[:,first:first+2048].cuda(),use_cache=False)
            print('CALIBRATION_TOKENS',first+2048,flush=True)
    scales={str(i):{s:(2.**maxima[i,s].clamp_min(2.**-24).log2().ceil()).tolist()
        for s in ['x','g','u']} for i in range(24)}
    result={'source':'first 8192 tokens WT2 train, no test/validation/task data, no weight updates',
        'source_file_sha256':baseline.sha(data),'ids_sha256':__import__('hashlib').sha256(ids.numpy().tobytes()).hexdigest(),
        'calibration_tokens':8192,'context':2048,'group':64,'policies':{'group_max':scales},
        'observed_maxima':{str(i):{s:maxima[i,s].tolist() for s in ['x','g','u']} for i in range(24)},
        'model_revision':json.loads((baseline.ROOT/'deployment_manifest.json').read_text())['revision']}
    baseline.save(out,result);print('CALIBRATION_COMPLETE',out,flush=True)

if __name__=='__main__':main()
