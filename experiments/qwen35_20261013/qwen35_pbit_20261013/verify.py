"""Counting identity, unbiased PV estimator, and full-model integration checks."""
import json,os,time
import torch
from common import ROOT,baseline,protected_hash,audit_sources
from experiment import load_variant,VARIANTS
from pv_sampling import probabilities,sampled_weights

def main():
    assert os.environ.get('SLURM_JOB_ID'),'GPU work requires Slurm'
    torch.set_num_threads(4);torch.manual_seed(13);torch.backends.cuda.matmul.allow_tf32=False
    # Exactly reconstruct a signed dot product by per-plane AND popcounts.
    streams=torch.randint(0,2,(64,3),dtype=torch.int64)
    sign=torch.tensor([1,-1,1]);codes=torch.tensor([[3,-5,7],[-2,1,0]])
    expected=(streams.sum(0)*sign)@codes.T
    plane_sum=torch.zeros(2,dtype=torch.int64)
    for plane in range(8):
        events=streams[:,None,:]&((codes.abs()>>plane)&1)[None,:,:]
        plane_sum+=(events*sign[None,None,:]*codes.sign()[None,:,:]).sum((0,2))*(1<<plane)
    assert torch.equal(expected,plane_sum)
    # Statistical test of IID selection-add: average of many 512-sample rows.
    p=torch.tensor([[.2,.3,.5]],device='cuda').expand(4096,-1)
    rng=torch.Generator(device='cuda').manual_seed(291)
    v=torch.tensor([-2.,1.,4.],device='cuda')
    y=sampled_weights(p,512,rng)@v
    target=(p[0]*v).sum();se=(((p[0]*(v-target).square()).sum()/512/4096).sqrt()).item()
    error=abs(y.mean().item()-target.item());assert error<6*se,(error,se)
    valid=torch.tensor([[True,False,False],[False,False,False]],device='cuda')
    probs=probabilities(torch.ones((2,3),device='cuda'),valid)
    selected=sampled_weights(probs,512,rng)
    assert selected[0,0]==1 and selected[0,1:].sum()==0 and selected[1].sum()==0
    checks=[]
    for variant in VARIANTS:
        start=time.perf_counter();tok,model,ffns,controls,before,loading=load_variant(variant,0)
        x=tok('This test checks fixed-weight stochastic computation and gated attention. '*12,
            return_tensors='pt',add_special_tokens=False).input_ids[:,:128].cuda()
        with torch.inference_mode():
            out=model(input_ids=x,use_cache=True,logits_to_keep=1)
            assert torch.isfinite(out.logits).all()
            nxt=out.logits[:,-1].argmax(-1,keepdim=True)
            decode=model(input_ids=nxt,past_key_values=out.past_key_values,use_cache=True,logits_to_keep=1)
            assert torch.isfinite(decode.logits).all()
        assert all(m.calls>=2 for m in ffns.values())
        assert all(c.prefill_calls>=1 and c.decode_calls>=1 for c in controls)
        assert protected_hash(model)==before
        checks.append({'variant':variant,'seconds':time.perf_counter()-start,
            'ffn_layers':len(ffns),'attention_layers':[c.layer for c in controls],
            'prefill_decode_finite':True,'protected_parameters_unchanged':True})
        del model,out,decode,ffns,controls;torch.cuda.empty_cache()
    result={'literal_signed_bitplane_counts_exact':True,'pv_mean_error':error,'pv_mean_standard_error':se,
        'masked_rows_correct':True,'variant_checks':checks,'source_sha256':audit_sources()}
    baseline.save(ROOT/'results/verification.json',result)
    print('ALL_VERIFICATION_PASSED',json.dumps(checks),flush=True)

if __name__=='__main__':main()
