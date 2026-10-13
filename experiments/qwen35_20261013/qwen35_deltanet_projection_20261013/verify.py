"""Projection estimator statistics, shared-input semantics, and model integration."""
import json,os,time
import torch
from experiment import load,protected_hash,ROOT
from common import baseline
from projection import Projection,EntryCache
from streaming_ffn import grouped_scale

def main():
    assert os.environ.get('SLURM_JOB_ID'),'GPU work requires Slurm'
    torch.set_num_threads(4);torch.manual_seed(401);torch.backends.cuda.matmul.allow_tf32=False
    x=torch.randn((1,64),device='cuda').expand(8192,-1).contiguous()
    linear=torch.nn.Linear(64,64,bias=False).cuda()
    cache=EntryCache(3991)
    a=Projection(linear,'iid',4096,17,cache);b=Projection(linear,'iid',4096,18,cache)
    with torch.inference_mode():
        ya=a(x);yb=b(x)
        assert torch.equal(ya,yb) and cache.hits==1
        scale=grouped_scale(x[:1],64);p=x[:1].abs()/scale
        mu=torch.nn.functional.linear(x[:1],a.weight_decoded)[0]
        variance=torch.nn.functional.linear(scale.square()*p*(1-p)/4096,a.weight_decoded.square())[0]
        se=(variance/8192).sqrt()
        assert bool(((ya.mean(0)-mu).abs()<=6*se+1e-5).all())
        cache.clear();zero=a(torch.zeros_like(x[:1]));assert torch.equal(zero,torch.zeros_like(zero))
    checks=[]
    for scope,scenario in [('out','solo'),('out_z','solo'),('out_z_qkv','solo'),('out_z_qkv','combined')]:
        start=time.perf_counter();tok,model,proj,ffns,controls,before=load(scope,'iid',scenario,4096,0)
        x=tok('Testing preserved DeltaNet state transitions with stochastic fixed projections. '*15,
            return_tensors='pt').input_ids[:,:128].cuda()
        with torch.inference_mode():
            y=model(input_ids=x,use_cache=True,logits_to_keep=1)
            z=model(input_ids=y.logits[:,-1].argmax(-1,keepdim=True),past_key_values=y.past_key_values,use_cache=True,logits_to_keep=1)
            assert torch.isfinite(y.logits).all() and torch.isfinite(z.logits).all()
        assert all(p.calls>=2 for p in proj.values())
        assert before==protected_hash(model,scope,scenario)
        if scope=='out_z_qkv':assert all(p.entry.hits>=2 for p in proj.values() if p.entry is not None)
        checks.append({'scope':scope,'scenario':scenario,'projection_count':len(proj),
            'prefill_decode_finite':True,'protected_parameters_unchanged':True,'seconds':time.perf_counter()-start})
        del model,proj,ffns,controls,y,z;torch.cuda.empty_cache()
    baseline.save(ROOT/'results/verification.json',{'mean_test_passed':True,'shared_stream_identity':True,
        'zero_input_exact':True,'checks':checks})
    print('ALL_CHECKS_PASSED',json.dumps(checks),flush=True)

if __name__=='__main__':main()
