"""Simple block scores and incrementally maintained RoPE-Key summaries."""
import math
import torch
from torch.nn import functional as F


METHODS=('mean_softmax','mean_sigmoid','max2','max4','bound','mix')


def build(k,reps=4,extrema=True):
    l=k.shape[-2];pad=(-l)%64
    packed=F.pad(k.float(),(0,0,0,pad)).reshape(*k.shape[:2],-1,64,64)
    sums=packed.reshape(*packed.shape[:3],reps,64//reps,64).sum(-2) if reps else None
    valid=(torch.arange(packed.shape[-3]*64,device=k.device)<l).reshape(-1,64)
    low=packed.masked_fill(~valid[None,None,:,:,None],torch.inf).amin(-2) if extrema else None
    high=packed.masked_fill(~valid[None,None,:,:,None],-torch.inf).amax(-2) if extrema else None
    return sums,low,high


def cache(attn,k,method):
    l=k.shape[-2];old=getattr(attn,'_metrics_cache',None)
    reps={'max2':2,'max4':4,'bound':0}.get(method,1)
    extrema=method in ('bound','mix','distill')
    if old is None or old[0]!=l-1:
        state=build(k,reps,extrema)
    else:
        sums,low,high=old[1]
        if (l-1)%64==0:
            if sums is not None:sums=torch.cat((sums,torch.zeros_like(sums[:,:,:1])),2)
            if extrema:
                low=torch.cat((low,torch.full_like(low[:,:,:1],torch.inf)),2)
                high=torch.cat((high,torch.full_like(high[:,:,:1],-torch.inf)) ,2)
        last=k[:,:,-1].float()
        if sums is not None:sums[:,:,-1,((l-1)%64)//(64//reps)]+=last
        if extrema:
            low[:,:,-1]=torch.minimum(low[:,:,-1],last)
            high[:,:,-1]=torch.maximum(high[:,:,-1],last)
        state=sums,low,high
    attn._metrics_cache=(l,state)
    return state


def sizes(length,device):
    c=math.ceil(length/64)
    return (length-torch.arange(c,device=device)*64).clamp(1,64).float()


def score(q,state,length,method,alpha=None,scales=None):
    sums,low,high=state
    groups=q.shape[1]//(sums.shape[1] if sums is not None else low.shape[1])
    n=sizes(length,q.device)
    qf=q.float().squeeze(-2)
    if method in ('mean_softmax','mean_sigmoid','mix','distill'):
        means=(sums.sum(-2)/n[:,None]).repeat_interleave(groups,1)
        mean=(qf[:,:,None]*means).sum(-1)/8
        if method.startswith('mean'):return mean
    if method in ('max2','max4'):
        r=int(method[-1]);sub=64//r
        s=sums.reshape(*sums.shape[:3],r,sums.shape[-2]//r,64).sum(-2)
        ns=(length-torch.arange(n.numel()*r,device=q.device)*sub).clamp(0,sub).reshape(-1,r)
        means=(s/ns.clamp_min(1)[...,None]).repeat_interleave(groups,1)
        values=(qf[:,:,None,None]*means).sum(-1)/8
        return values.masked_fill(ns==0,-torch.inf).amax(-1)
    if method in ('bound','mix','distill'):
        low=low.repeat_interleave(groups,1);high=high.repeat_interleave(groups,1)
        chosen=torch.where(qf[:,:,None]>=0,high,low)
        upper=(qf[:,:,None]*chosen).sum(-1)/8
        if method=='bound':return upper
        return alpha*mean/scales[...,0]+(1-alpha)*upper/scales[...,1]
    raise ValueError(method)


def mandatory(n):
    return (torch.arange(n.numel(),device=n.device)==0)|(torch.arange(n.numel(),device=n.device)>=n.numel()-2)


def probabilities(scores,n,method,temperature,parameter):
    if method=='mean_softmax':
        p=(scores+n.log()).softmax(-1)
        pi=-torch.expm1(parameter*torch.log1p(-p.clamp(max=1-1e-7)))
    else:
        relative=scores-scores.amax(-1,keepdim=True)
        pi=torch.sigmoid((relative+parameter)/temperature)
    return pi.masked_fill(mandatory(n),1.)


def macs(method):
    return {'mean_softmax':64,'mean_sigmoid':64,'max2':128,'max4':256,'bound':64,'mix':128,'distill':128}.get(method,0)
