"""Finite Ising enumeration, literal Gibbs and compressed CTMC validation."""
import itertools
import json
import math
from pathlib import Path
import numpy as np
from scipy.linalg import expm
import torch
from sampling import ising_counts
from attention import summaries


def finite_check():
    rng=np.random.default_rng(73)
    z=np.array([2.,1.,.7,0.,-.2,-1.])
    delta=z[1:]-z[0]
    states=np.array(list(itertools.product([0,1],repeat=len(delta))),dtype=np.int64)
    result=[]
    for penalty in (0.,4.,8.,16.,24.):
        active=states.sum(-1)
        energy=-states@delta+penalty*active*(active-1)/2
        p=np.exp(-energy-np.max(-energy)); p/=p.sum()
        valid=active<=1
        categorical=np.r_[p[active==0],*[p[(active==1)&(states[:,i]==1)].item() for i in range(len(delta))]]
        categorical/=categorical.sum()
        teacher=np.exp(z-z.max());teacher/=teacher.sum()
        assert np.max(np.abs(categorical-teacher))<1e-12
        # Independently update one randomly selected spin of every replica.
        b=np.zeros((16000,len(delta)),dtype=np.int64)
        row=np.arange(len(b))
        for step in range(600*len(delta)):
            i=rng.integers(len(delta),size=len(b))
            field=delta[i]-penalty*(b.sum(-1)-b[row,i])
            prob=1/(1+np.exp(-np.clip(field,-700,700)))
            b[row,i]=rng.random(len(b))<prob
        ids=(b*(2**np.arange(len(delta)-1,-1,-1))).sum(-1)
        empirical=np.bincount(ids,minlength=len(states))/len(b)
        tv=np.abs(empirical-p).sum()/2
        assert tv<.025,(penalty,tv)
        result.append({'penalty':penalty,'exact_invalid_probability':float(p[~valid].sum()),
            'literal_random_scan_final_ensemble_TV':float(tv),'replicas':len(b),'sweeps':600,
            'conditional_valid_softmax_max_error':float(np.max(np.abs(categorical-teacher)))})
    return result


def star_check():
    z=np.array([2.,1.,.7,0.,-.2,-1.])
    a=1/(1+np.exp(-(z[1:]-z[0])))
    Q=np.zeros((len(z),len(z)))
    Q[0,1:]=a
    Q[np.arange(1,len(z)),0]=1-a
    np.fill_diagonal(Q,-Q.sum(-1))
    target=np.exp(z-z.max());target/=target.sum()
    assert np.max(np.abs(target@Q))<1e-12
    out=[]
    gen=torch.Generator(device='cuda').manual_seed(81)
    for t in (.1,.5,1.,4.,16.):
        reference=expm(Q*t)[0]
        scores=torch.tensor(z,dtype=torch.float32,device='cuda').expand(100000,-1)
        count,_=ising_counts(scores,1,burn=0,spacing=t,generator=gen)
        empirical=count.mean(0).cpu().numpy()
        err=float(np.abs(empirical-reference).max())
        assert err<.006,(t,err)
        out.append({'time_bit_clocks':t,'replicas':100000,'empirical_vs_exact_transient_max_error':err,
            'exact_transient_TV_to_softmax':float(np.abs(reference-target).sum()/2)})
    # Probability noise and output noise versus sample spacing at same sample count.
    sample_results=[]
    for n in (16,64,256,512):
        for gap in (.25,1.,4.):
            zt=torch.tensor(z,dtype=torch.float32,device='cuda').expand(4096,-1)
            counts,extra=ising_counts(zt,n,burn=4,spacing=gap,generator=gen)
            weights=counts/n
            p=torch.tensor(target,device='cuda')
            mse=(weights-p).square().mean(0).sum().item()
            iid_mse=(1-(p*p).sum()).item()/n
            sample_results.append({'samples':n,'spacing':gap,'replicas':4096,'probability_mse':mse,
                'iid_expected_mse':iid_mse,'ESS_from_variance_ratio':n*iid_mse/mse,
                'adjacent_repeat_fraction':extra['adjacent_repeat'].item()})
    return out,sample_results


def summary_check():
    from types import SimpleNamespace
    attn=SimpleNamespace()
    torch.manual_seed(42)
    k=torch.randn(2,2,149,64,device='cuda')
    maxerr=0.
    for n in range(61,149):
        got=summaries(attn,k[:,:,:n],64)
        fresh=summaries(SimpleNamespace(),k[:,:,:n],64)
        maxerr=max(maxerr,max((a-b).abs().max().item() for a,b in zip(got,fresh)))
    assert maxerr<2e-6,maxerr
    return {'incremental_summary_max_error':maxerr,'crosses_block_boundaries':True}


def main():
    torch.set_num_threads(4)
    finite=finite_check()
    transient,ess=star_check()
    result={'status':'passed','finite_penalty':finite,'hard_limit_transient':transient,
        'sample_count_and_spacing':ess,'summary_cache':summary_check(),
        'scope':'Finite-penalty literal random-scan check plus exact-generator transient validation of hard-limit continuous-time simulation.'}
    path=Path('results/pbit_softmax_router_20261006/diagnostics.json');path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__': main()
