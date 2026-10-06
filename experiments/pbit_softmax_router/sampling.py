"""Probability-law references, NOT hardware timings or sparse CUDA kernels.

Ising: exact continuous-time heat-bath dynamics in the hard one-hot-penalty
limit. We compress alternating empty/active holding times, not the target
categorical distribution. Each physical bit has a unit-rate update clock.
Finite penalties are separately tested in diagnostics.py.
"""
import math
import torch


def ising_counts(scores, samples, burn=4., spacing=1., generator=None):
    shape = scores.shape
    z = scores.reshape(-1, shape[-1]).float()
    ref = z.argmax(-1, keepdim=True)
    delta = z - z.gather(1, ref)
    # No softmax/exponentiated categorical probabilities in this sampler.
    birth = torch.sigmoid(delta).scatter(1, ref, 0.)
    total = birth.sum(-1, keepdim=True)
    degenerate = total == 0
    proposal = birth + torch.zeros_like(birth).scatter(1, ref, degenerate.float())
    horizon = burn + spacing*samples
    # The number of completed active dwells is dominated by Poisson(horizon),
    # since their departure rates <=1; check coverage explicitly below.
    cycles = max(16, math.ceil(horizon + 12*math.sqrt(horizon+1) + 32))
    while True:
        chosen = torch.multinomial(proposal, cycles, replacement=True, generator=generator)
        u = torch.rand((z.shape[0], cycles, 2), device=z.device, generator=generator)
        waits = -u.clamp_min(1e-30).log()
        empty = waits[:,:,0] / total.clamp_min(1e-30)
        active = waits[:,:,1] / (1-birth.gather(1, chosen)).clamp_min(.5)
        end = (empty+active).cumsum(-1)
        if bool((end[:,-1] >= horizon).all()):
            break
        cycles *= 2
    start = end-active
    t = (burn + spacing*torch.arange(1,samples+1,device=z.device)).expand(z.shape[0],-1).contiguous()
    at = torch.searchsorted(end.contiguous(), t).clamp_max(cycles-1)
    index = torch.where(t >= start.gather(1,at), chosen.gather(1,at), ref)
    counts = torch.zeros_like(z).scatter_add_(1,index,torch.ones_like(index,dtype=z.dtype))
    repeated = (index[:,1:] == index[:,:-1]).float().mean() if samples>1 else z.new_tensor(0.)
    # If all relative weights <=1, total invalid equilibrium mass at finite
    # penalty lambda is bounded by sum C(n,m)e^-lambda*m(m-1)/2 /(1+sum w).
    # A tighter input-specific elementary-symmetric bound uses (sum w)^m/m!.
    wsum = delta.exp().sum(-1)-1
    leak_bound24 = .5*wsum.square()/(1+wsum)*math.exp(-24)
    return counts.reshape(shape), {'adjacent_repeat':repeated,
        'finite_lambda24_leak_leading_bound':leak_bound24.mean()}


def iid_counts(scores, samples, generator=None):
    z=scores.reshape(-1,scores.shape[-1])
    ids=torch.multinomial(z.softmax(-1),samples,True,generator=generator)
    return torch.zeros_like(z).scatter_add_(1,ids,torch.ones_like(ids,dtype=z.dtype)).reshape_as(scores)


def choose_blocks(proxy, ratio, generator, deterministic=False, forced_recent=2):
    n=proxy.shape[-1]
    budget=max(1,math.ceil(n*ratio))
    probability=proxy.softmax(-1)
    if deterministic:
        selected=torch.zeros_like(proxy,dtype=torch.bool).scatter(-1,proxy.topk(budget,-1).indices,True)
    else:
        inclusion=-torch.expm1(budget*torch.log1p(-probability.clamp_max(1-1e-7)))
        selected=torch.rand(inclusion.shape,device=proxy.device,generator=generator)<inclusion
    # Explicit mandatory block safety budget, included in all access statistics.
    selected[...,0]=True
    selected[...,-forced_recent:]=True
    return selected
