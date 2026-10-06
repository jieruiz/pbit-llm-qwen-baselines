"""Monte Carlo moment and literal-bit equivalence tests on CUDA."""
import json
import torch
from sampling import encode, counts


def main():
    torch.manual_seed(147)
    trials, B = 60000, 4
    q = torch.tensor([1., -.37, .08, 0., -.73],device='cuda')
    k = torch.tensor([[1.,2.,3.,4.,-1.],[-2.,1.,-1.,3.,.5]],device='cuda')
    p = q.abs()
    results = {}
    for mode in ('iid','stratified'):
        qhat,c = encode(q.expand(trials,-1),B,mode)
        assert torch.allclose(qhat.mean(0),q,atol=.008,rtol=0)
        assert (c[:,0]==B).all() and (c[:,3]==0).all()
        assert ((c>=0)&(c<=B)&(c==c.round())).all()
        frac = B*p-(B*p).floor()
        variance = p*(1-p)/B if mode=='iid' else frac*(1-frac)/(B*B)
        assert torch.allclose(qhat.var(0),variance,atol=.002,rtol=.04)
        z=qhat@k.T
        assert torch.allclose(z.mean(0),q@k.T,atol=.012,rtol=0)
        assert torch.allclose(z.var(0),variance@k.square().T,atol=.008,rtol=.04)
        literal=torch.zeros_like(c)
        for n in range(B):
            u=torch.rand(c.shape,device='cuda')
            if mode=='stratified': u=(n+u)/B
            literal+=(u<p).float()
        # Compare each count's marginal PMF, not individual random draws.
        for i in range(q.numel()):
            a=torch.bincount(c[:,i].long(),minlength=B+1).float()/trials
            b=torch.bincount(literal[:,i].long(),minlength=B+1).float()/trials
            assert (a-b).abs().max()<.015
        zeros,_=encode(torch.zeros(2,5,device='cuda'),B,mode)
        assert (zeros==0).all() and torch.isfinite(zeros).all()
        scaled,_=encode(torch.tensor([[0.,-2.,2.]],device='cuda'),B,mode)
        assert torch.equal(scaled,torch.tensor([[0.,-2.,2.]],device='cuda'))
        results[mode]={'query_mean':qhat.mean(0).tolist(),'score_variance':z.var(0).tolist(),
                       'expected_score_variance':(variance@k.square().T).tolist()}
    print(json.dumps({'status':'passed','trials':trials,'B':B,'results':results,
                      'coverage':['signed mean','analytic query/score variance','literal bit count law',
                                  'zero query','sign and scale','p=0/1 boundaries']},indent=2))


if __name__=='__main__': main()
