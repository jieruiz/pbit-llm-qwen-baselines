"""Stepwise DeltaNet projection changes; original recurrent state/gates retained."""
import argparse,hashlib,importlib.metadata,json,math,os,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'qwen35_pbit_20261013'))
from common import baseline
from pv_sampling import install as install_pv
from streaming_ffn import StreamingFFN,StreamConfig
from projection import SCOPES,install

class CountedFFN(StreamingFFN):
    def __init__(self,*args):super().__init__(*args);self.calls=0
    def forward(self,x):self.calls+=1;return super().forward(x)

def protected_hash(model,scope,scenario):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if scenario=='combined' and '.mlp.' in n:continue
        if scope!='baseline' and any('.linear_attn.'+v+'.' in n for v in SCOPES[scope]):continue
        h.update(n.encode());h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

def load(scope,mode,scenario,cycles,seed):
    tok,model,_=baseline.load();model.requires_grad_(False)
    before=protected_hash(model,scope,scenario)
    identities={i:(id(l.linear_attn),id(l.linear_attn.conv1d),id(l.linear_attn.norm),
        id(l.linear_attn.in_proj_a),id(l.linear_attn.in_proj_b))
        for i,l in enumerate(model.model.language_model.layers) if l.block_type=='linear_attention'}
    ffns={};controls=[]
    if scenario=='combined':
        for i,l in enumerate(model.model.language_model.layers):
            l.mlp=CountedFFN(l.mlp,StreamConfig(mode='streaming',cycles=4096,interval=256,group=64,
                burn_fraction=.25,weight_bits=8,seed=seed,layer=i));ffns[i]=l.mlp
        controls=install_pv(model,'iid',seed,512)
    projections=install(model,scope,mode,cycles,seed) if scope!='baseline' else {}
    for i,ids in identities.items():
        d=model.model.language_model.layers[i].linear_attn
        assert ids==(id(d),id(d.conv1d),id(d.norm),id(d.in_proj_a),id(d.in_proj_b))
    assert protected_hash(model,scope,scenario)==before
    return tok,model,projections,ffns,controls,before

def wt2(tok,model,context):
    if context==2048:return baseline.wt2(tok,model)
    source=baseline.DATA/'wikitext-2/wiki.test.raw'
    ids=tok(source.read_text(),add_special_tokens=False,return_tensors='pt',verbose=False).input_ids
    total=0.;count=0;previous_end=1;windows=[];start=time.perf_counter();stride=context//2
    for begin in range(0,ids.shape[1],stride):
        end=min(begin+context,ids.shape[1]);x=ids[:,begin:end].cuda()
        hidden=model.model(input_ids=x,use_cache=False).last_hidden_state
        first=max(previous_end,begin+1);nll=0.
        for target in range(first,end,64):
            stop=min(target+64,end);logits=model.lm_head(hidden[:,target-begin-1:stop-begin-1]).float()
            assert torch.isfinite(logits).all()
            nll+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                x[:,target-begin:stop-begin].reshape(-1),reduction='sum').item()
        count+=end-first;total+=nll;previous_end=end
        windows.append({'begin':begin,'end':end,'scored':end-first,'nll_sum':nll})
        print(json.dumps({'window':len(windows),'scored':count,'ppl':math.exp(total/count),
            'seconds':time.perf_counter()-start}),flush=True)
        if end==ids.shape[1]:break
    assert count==ids.shape[1]-1
    return {'perplexity':math.exp(total/count),'scored_tokens':count,'corpus_tokens':ids.shape[1],
        'nll_sum':total,'corpus_sha256':baseline.sha(source),'windows':windows}

def main():
    assert os.environ.get('SLURM_JOB_ID'),'GPU execution requires Slurm'
    p=argparse.ArgumentParser();p.add_argument('--scope',choices=list(SCOPES)+['baseline'],required=True)
    p.add_argument('--mode',choices=['weights','iid'],default='iid')
    p.add_argument('--scenario',choices=['solo','combined'],default='solo')
    p.add_argument('--cycles',type=int,default=4096);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--context',type=int,choices=[2048,8192],default=2048);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();name=f'{a.scenario}_{a.scope}_{a.mode}_T{a.cycles}_ctx{a.context}_s{a.seed}'
    out=ROOT/'results'/('smoke' if a.smoke else 'formal')/(name+'.json')
    if out.exists():print('ALREADY_COMPLETE',out,flush=True);return
    torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False
    tok,model,proj,ffns,controls,before=load(a.scope,a.mode,a.scenario,a.cycles,a.seed)
    weights=[(m,m.weight_decoded.clone()) for m in proj.values()]
    start=time.perf_counter()
    with torch.inference_mode():
        if a.smoke:
            x=tok('A test of stochastic DeltaNet projections and preserved recurrent updates. '*12,
                return_tensors='pt').input_ids[:,:128].cuda()
            y=model(input_ids=x,use_cache=True,logits_to_keep=1)
            z=model(input_ids=y.logits[:,-1].argmax(-1,keepdim=True),past_key_values=y.past_key_values,use_cache=True,logits_to_keep=1)
            assert torch.isfinite(y.logits).all() and torch.isfinite(z.logits).all()
            result={'finite_prefill_and_decode':True}
        else:result=wt2(tok,model,a.context);assert result['scored_tokens']==297192
    assert protected_hash(model,a.scope,a.scenario)==before
    for m,w in weights:assert torch.equal(m.weight_decoded,w)
    assert all(m.calls>0 for m in proj.values()) and all(m.calls>0 for m in ffns.values())
    if a.scope=='out_z_qkv' and a.mode=='iid':
        assert all(m.entry.hits>0 for m in proj.values() if m.entry is not None)
    files=[ROOT/'experiment.py',ROOT/'projection.py',ROOT/'streaming_ffn.py',baseline.ROOT/'baseline.py',
        ROOT.parent/'qwen35_pbit_20261013/pv_sampling.py']
    metadata={'args':vars(a),'seconds':time.perf_counter()-start,'job':os.environ['SLURM_JOB_ID'],
        'context':a.context,'stride':a.context//2,'batch_size':1,'source_sha256':{str(f):baseline.sha(f) for f in files},
        'model_revision':json.loads((baseline.ROOT/'deployment_manifest.json').read_text())['revision'],
        'projection_spec':'All18 DeltaNet layers; fixed INT8 tile64; dynamic group64 input probability; T4096; exact Binomial count compression; shared input stream for QKV/z in combined scope; BF16 I/O, FP32 reference accumulation.',
        'state_spec':'Original alpha/beta, Conv1d+SiLU, Q/K L2 norm, recurrent/chunked update, RMSNormGated, original state precision; one model update per token, no update per sampling bit.',
        'combined_spec':'If combined: all24 dynamic INT8 FFN input4096/output3072 update256 burn25%, plus six full-attention PV512; otherwise all FFNs and full attention original.',
        'protected_parameters_sha256':before,'projection_diagnostics':{n:m.diagnostics() for n,m in proj.items()},
        'ffn_layers':list(ffns),'attention_diagnostics':[c.summary() for c in controls],
        'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','flash-linear-attention']}}
    baseline.save(out,{'metadata':metadata,'result':result});print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
