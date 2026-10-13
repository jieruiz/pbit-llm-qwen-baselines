"""Read original gates on WT2 train, for state persistence analysis; no weight updates."""
import json,os,sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'qwen35_pbit_20261013'))
from common import baseline,protected_hash

def main():
    assert os.environ.get('SLURM_JOB_ID'),'GPU work requires Slurm'
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
    tok,model,_=baseline.load();model.requires_grad_(False);before=protected_hash(model)
    stats={};handles=[]
    def record(i,name,x):
        x=x.detach().float().reshape(-1,16).double()
        v=torch.stack([torch.full((16,),x.shape[0],device=x.device),x.sum(0),
            (x>.9).sum(0),(x>.99).sum(0),(x>.999).sum(0),x.square().sum(0)])
        key=(i,name);stats[key]=stats.get(key,0)+v.cpu()
    for i,l in enumerate(model.model.language_model.layers):
        if l.block_type!='linear_attention':continue
        d=l.linear_attn
        def a_hook(module,args,output,index=i,delta=d):
            alpha=torch.exp(-delta.A_log.float().exp()*F.softplus(output.float()+delta.dt_bias.float()))
            record(index,'alpha',alpha)
        def b_hook(module,args,output,index=i):record(index,'beta',output.sigmoid())
        handles.append(d.in_proj_a.register_forward_hook(a_hook));handles.append(d.in_proj_b.register_forward_hook(b_hook))
    source=baseline.DATA/'wikitext-2/wiki.train.raw'
    ids=tok(source.read_text(),add_special_tokens=False,return_tensors='pt',verbose=False).input_ids[:,:8192]
    with torch.inference_mode():
        for first in range(0,8192,2048):model.model(input_ids=ids[:,first:first+2048].cuda(),use_cache=False)
    assert protected_hash(model)==before
    per_layer={}
    for (i,name),v in stats.items():
        count,total,p9,p99,p999,sq=v
        per_layer.setdefault(str(i),{})[name]={'mean_per_head':(total/count).tolist(),
            'fraction_gt_0p9':(p9/count).tolist(),'fraction_gt_0p99':(p99/count).tolist(),
            'fraction_gt_0p999':(p999/count).tolist()}
    aggregate={}
    for name in ['alpha','beta']:
        v=sum(v for (i,n),v in stats.items() if n==name).sum(-1)
        count,total,p9,p99,p999,sq=v.tolist()
        aggregate[name]={'observations':int(count),'mean':total/count,'fraction_gt_0p9':p9/count,
            'fraction_gt_0p99':p99/count,'fraction_gt_0p999':p999/count}
    result={'source':'WT2 train first8192 tokens, original model, no calibration of replacement',
        'source_sha256':baseline.sha(source),'slurm_job':os.environ['SLURM_JOB_ID'],
        'aggregate':aggregate,'per_layer':per_layer}
    baseline.save(ROOT/'results/deltanet_gates.json',result)
    print(json.dumps(aggregate,indent=2),flush=True)

if __name__=='__main__':main()
