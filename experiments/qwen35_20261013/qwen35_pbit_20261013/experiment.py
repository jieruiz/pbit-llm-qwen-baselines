"""Full corpus and task comparisons, original DeltaNet operators in every variant."""
import argparse, importlib.metadata, json, os, time
import torch
from torch import nn
from torch.nn import functional as F
from common import ROOT,BASE,baseline,protected_hash,audit_sources
from exact_ffn import StochasticMatrixFFN,encode_fixed
from pv_sampling import install

VARIANTS=['ffn_mean','attention_dense_control','ffn_exact','attention_pv','joint_pv']
TASKS=['wikitext2','hellaswag','piqa','winogrande','arc_easy','arc_challenge']

class MeanFFN(StochasticMatrixFFN):
    def forward(self,x):
        self.calls+=1;dtype=x.dtype
        _,xq=encode_fixed(x.float(),self.scale_x,self.bits)
        gu=F.linear(xq,self.gu_decoded);g,u=gu[...,:self.width],gu[...,self.width:]
        _,gq=encode_fixed(g,self.scale_g,self.bits);_,uq=encode_fixed(u,self.scale_u,self.bits)
        return F.linear(F.silu(gq)*uq,self.down_decoded).to(dtype)

def load_variant(variant,seed):
    tok,model,info=baseline.load();model.requires_grad_(False)
    layers=model.model.language_model.layers
    delta=[(i,id(l.linear_attn)) for i,l in enumerate(layers) if l.block_type=='linear_attention']
    before=protected_hash(model);ffns={};controls=[]
    if variant in ['ffn_mean','ffn_exact','joint_pv']:
        scales=json.loads((ROOT/'calibration.json').read_text())['policies']['group_max']
        factory=MeanFFN if variant=='ffn_mean' else StochasticMatrixFFN
        for i,l in enumerate(layers):
            l.mlp=factory(l.mlp,scales[str(i)],'fixed_exact',bits=12,cycles=4096,seed=seed,layer=i)
            ffns[i]=l.mlp
    if variant in ['attention_dense_control','attention_pv','joint_pv']:
        controls=install(model,'dense' if variant=='attention_dense_control' else 'iid',seed,512)
    assert delta==[(i,id(l.linear_attn)) for i,l in enumerate(layers) if l.block_type=='linear_attention']
    assert before==protected_hash(model)
    return tok,model,ffns,controls,before,info

def main():
    assert os.environ.get('SLURM_JOB_ID'),'GPU work requires Slurm'
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=VARIANTS,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--task',choices=TASKS,required=True);a=p.parse_args()
    out=ROOT/'results/formal'/a.variant/f'{a.task}_seed{a.seed}.json'
    if out.exists():print('ALREADY_COMPLETE',out,flush=True);return
    torch.set_num_threads(4);torch.manual_seed(a.seed);torch.backends.cuda.matmul.allow_tf32=False
    tok,model,ffns,controls,before,loading=load_variant(a.variant,a.seed)
    scales=[(m,n,b.clone()) for m in ffns.values() for n,b in m.named_buffers() if n.startswith('scale_')]
    start=time.perf_counter()
    with torch.inference_mode():
        result=baseline.wt2(tok,model) if a.task=='wikitext2' else baseline.task_eval(a.task,tok,model)
    assert protected_hash(model)==before
    for m,n,b in scales:assert torch.equal(dict(m.named_buffers())[n],b)
    assert all(m.calls>0 for m in ffns.values())
    assert all(c.calls>0 for c in controls)
    if a.task=='wikitext2':assert result['scored_tokens']==297192
    meta={'args':vars(a),'seconds':time.perf_counter()-start,'gpu':torch.cuda.get_device_name(),
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'context':2048,'stride':1024,'batch_size':1,
        'model_revision':json.loads((BASE/'deployment_manifest.json').read_text())['revision'],
        'versions':{k:importlib.metadata.version(k) for k in ['torch','transformers','lm_eval','flash-linear-attention']},
        'ffn_layers':list(ffns),'attention_layers':[c.layer for c in controls],
        'delta_layers_unchanged':[i for i,l in enumerate(model.model.language_model.layers) if l.block_type=='linear_attention'],
        'ffn_spec':'fixed INT8 per64x64 tile; N12 signed magnitude; immutable group_max scales; T4096 input/gate-up then T4096 product/down; exact spatial/temporal sum; binomial count-law compression; FP32 reference accumulators',
        'sigmoid_spec':'numerical sigmoid(gq); not a physical sigmoid circuit/device model',
        'attention_spec':'accurate FP32 QK/Softmax; categorical PV512; original Q/K norm, RoPE, KV cache, sigmoid output gate and o_proj retained; prefill AND decode',
        'source_sha256':audit_sources(),'calibration_sha256':baseline.sha(ROOT/'calibration.json'),
        'protected_parameters_sha256':before,'loading_info':loading,
        'ffn_diagnostics':{str(i):m.diagnostics() for i,m in ffns.items()},
        'attention_diagnostics':[c.summary() for c in controls]}
    baseline.save(out,{'metadata':meta,'result':result});print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
