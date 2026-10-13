"""Prior successful dynamic-scale streaming FFN; PPL only, DeltaNet unchanged."""
import argparse,hashlib,importlib.metadata,json,os,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'qwen35_pbit_20261013'))
from common import baseline,protected_hash
from pv_sampling import install
from streaming_ffn import StreamingFFN,StreamConfig

class CountedFFN(StreamingFFN):
    def __init__(self,*args):super().__init__(*args);self.calls=0
    def forward(self,x):self.calls+=1;return super().forward(x)

def load_variant(variant,seed):
    tok,model,_=baseline.load();model.requires_grad_(False)
    before=protected_hash(model);delta=[id(l.linear_attn) for l in model.model.language_model.layers if l.block_type=='linear_attention']
    ffns={}
    for i,l in enumerate(model.model.language_model.layers):
        l.mlp=CountedFFN(l.mlp,StreamConfig(mode='dense' if variant=='int8_dense' else 'streaming',
            cycles=4096,interval=256,group=64,burn_fraction=.25,weight_bits=8,seed=seed,layer=i))
        ffns[i]=l.mlp
    controls=install(model,'iid',seed,512) if variant=='dynamic_joint' else []
    assert delta==[id(l.linear_attn) for l in model.model.language_model.layers if l.block_type=='linear_attention']
    assert before==protected_hash(model)
    return tok,model,ffns,controls,before

def main():
    assert os.environ.get('SLURM_JOB_ID'),'GPU work requires Slurm'
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['int8_dense','dynamic_ffn','dynamic_joint'],required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--smoke',action='store_true');a=p.parse_args()
    out=ROOT/'results'/('smoke' if a.smoke else 'formal')/f'{a.variant}_seed{a.seed}.json'
    if out.exists():print('ALREADY_COMPLETE',out,flush=True);return
    torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False
    tok,model,ffns,controls,before=load_variant(a.variant,a.seed);start=time.perf_counter()
    with torch.inference_mode():
        if a.smoke:
            x=tok('A test of dynamic-scale stochastic arithmetic. '*20,return_tensors='pt').input_ids[:,:128].cuda()
            y=model(input_ids=x,use_cache=True,logits_to_keep=1)
            z=model(input_ids=y.logits[:,-1].argmax(-1,keepdim=True),past_key_values=y.past_key_values,use_cache=True,logits_to_keep=1)
            assert torch.isfinite(y.logits).all() and torch.isfinite(z.logits).all()
            result={'finite_prefill_and_decode':True}
        else:result=baseline.wt2(tok,model);assert result['scored_tokens']==297192
    assert before==protected_hash(model)
    assert all(m.calls>0 for m in ffns.values()) and all(c.calls>0 for c in controls)
    sources=[ROOT/'experiment.py',ROOT/'streaming_ffn.py',ROOT.parent/'qwen35_pbit_20261013/pv_sampling.py',baseline.ROOT/'baseline.py']
    meta={'args':vars(a),'seconds':time.perf_counter()-start,'slurm_job':os.environ['SLURM_JOB_ID'],
        'ffn_layers':list(ffns),'attention_layers':[c.layer for c in controls],'DeltaNet_operators':'all original',
        'spec':'INT8 tile64; dynamic per-token/group64 maxima for x/g/u; continuous probabilities; input4096/output3072; update256/burn25%; exact count-law and FP32 reference accumulation; BF16 residual',
        'hardware_cost_note':'Online max/scaling is additional hardware work; no claim of GPU acceleration or free dynamic range selection.',
        'model_revision':json.loads((baseline.ROOT/'deployment_manifest.json').read_text())['revision'],
        'source_sha256':{str(p):baseline.sha(p) for p in sources},'protected_sha256':before,
        'attention_diagnostics':[c.summary() for c in controls]}
    baseline.save(out,{'metadata':meta,'result':result});print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
