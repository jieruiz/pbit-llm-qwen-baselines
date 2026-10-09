"""Hard-exclusion CTMC; fixed-time observations, no categorical softmax drive."""
import math
import torch
def ising_counts(scores,samples,burn=16.,spacing=4.,generator=None):
    shape=scores.shape;z=scores.reshape(-1,shape[-1]).float()
    ref=z.argmax(-1,keepdim=True);delta=z-z.gather(1,ref)
    birth=torch.sigmoid(delta).scatter(1,ref,0.);total=birth.sum(-1,keepdim=True)
    proposal=birth+torch.zeros_like(birth).scatter(1,ref,(total==0).float())
    horizon=burn+spacing*samples;cycles=max(16,math.ceil(horizon+12*math.sqrt(horizon+1)+32))
    while True:
        chosen=torch.multinomial(proposal,cycles,True,generator=generator)
        waits=-torch.rand((z.shape[0],cycles,2),device=z.device,generator=generator).clamp_min(1e-30).log()
        empty=waits[:,:,0]/total.clamp_min(1e-30)
        active=waits[:,:,1]/(1-birth.gather(1,chosen)).clamp_min(.5)
        end=(empty+active).cumsum(-1)
        if bool((end[:,-1]>=horizon).all()):break
        cycles*=2
    start=end-active
    times=(burn+spacing*torch.arange(1,samples+1,device=z.device)).expand(z.shape[0],-1).contiguous()
    at=torch.searchsorted(end.contiguous(),times).clamp_max(cycles-1)
    index=torch.where(times>=start.gather(1,at),chosen.gather(1,at),ref)
    counts=torch.zeros_like(z).scatter_add_(1,index,torch.ones_like(index,dtype=z.dtype))
    repeated=(index[:,1:]==index[:,:-1]).float().mean() if samples>1 else z.new_tensor(0.)
    return counts.reshape(shape),repeated
def iid_counts(scores,samples,generator=None):
    shape=scores.shape;p=scores.reshape(-1,shape[-1]).float().softmax(-1)
    index=torch.multinomial(p,samples,True,generator=generator)
    counts=torch.zeros_like(p).scatter_add_(1,index,torch.ones_like(index,dtype=p.dtype))
    return counts.reshape(shape)
