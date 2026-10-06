import json
import torch
from sampling import tree_counts,independent_counts,explicit_independent_counts


def main():
    torch.manual_seed(842)
    trials=40000
    p=torch.tensor([.6,.3,.1,0.,0.],device="cuda").expand(trials,-1)
    value=torch.tensor([[1.,2.],[-2.,1.],[3.,-1.],[100.,100.],[100.,100.]],device="cuda")
    samples=8
    mu=p[0]@value
    expected_cat=((p[0,:,None]*value.square()).sum(0)-mu.square())/samples
    expected_bern=((p[0]*(1-p[0]))[:,None]*value.square()).sum(0)/samples
    results={}
    for name,func in (("tree",tree_counts),("binomial",independent_counts),("literal_bits",explicit_independent_counts)):
        c=func(p,samples)
        assert (c[:,3:]==0).all() and (c>=0).all() and (c==c.round()).all()
        if name=="tree":
            assert (c.sum(-1)==samples).all()
        y=(c/samples)@value
        theory=expected_cat if name=="tree" else expected_bern
        assert torch.allclose(y.mean(0),mu,atol=.015,rtol=0)
        assert torch.allclose(y.var(0),theory,atol=.018,rtol=.04)
        results[name]={"mean":y.mean(0).tolist(),"variance":y.var(0).tolist(),"theoretical_variance":theory.tolist()}
    onehot=torch.tensor([[0.,0.,1.]],device="cuda")
    assert torch.equal(tree_counts(onehot,7),7*onehot)
    assert torch.equal(independent_counts(onehot,7),7*onehot)
    same=torch.ones(5,1,device="cuda")
    c=tree_counts(p,1)
    assert ((c@same)==1).all()
    c=independent_counts(p,1)
    assert ((c.sum(-1)==0).any() and (c.sum(-1)>1).any())
    print(json.dumps({"status":"passed","trials":trials,"samples":samples,"results":results,
        "covered":["means and analytic variances","non-power-of-two padded tree","zero/one probabilities",
        "fixed tree count","independent zero/multiple selections","binomial versus literal p-bit law","constant value preservation"]},indent=2))


if __name__=="__main__":
    main()
