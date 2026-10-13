"""Fixed INT8 weights, dynamic group64 input probability, exact shared count-law."""
import torch
from torch import nn
from torch.nn import functional as F
from streaming_ffn import quantize_blocks,grouped_scale,count_bits

class EntryCache:
    def __init__(self,seed):
        self.rng=torch.Generator(device='cuda').manual_seed(seed)
        self.value=None;self.key=None;self.hits=0;self.draws=0
    def clear(self,*args):self.value=None;self.key=None
    def drive(self,x,cycles):
        key=(x.data_ptr(),tuple(x.shape))
        if self.value is not None and key==self.key:
            self.hits+=1;return self.value
        scale=grouped_scale(x.float(),64)
        p=(x.float().abs()/scale).clamp(0,1)
        n=count_bits(p,cycles,self.rng)
        self.value=x.float().sign()*scale*n/cycles;self.key=key;self.draws+=1
        return self.value

class Projection(nn.Module):
    def __init__(self,original,mode,cycles,seed,entry=None):
        super().__init__()
        assert mode in ['weights','iid'] and cycles>0
        w=original.weight.detach().float()
        decoded=quantize_blocks(w,64)
        self.register_buffer('weight_decoded',decoded)
        self.register_buffer('bias',None if original.bias is None else original.bias.detach().float())
        self.in_features=w.shape[1];self.out_features=w.shape[0]
        self.mode=mode;self.cycles=cycles;self.calls=0;self.entry=entry
        self.rng=torch.Generator(device=w.device).manual_seed(seed)
        self.register_buffer('stats',torch.zeros(4,device=w.device,dtype=torch.float64))
        self.weight_nmse=((decoded-w).square().sum()/w.square().sum().clamp_min(1e-30)).item()
    def forward(self,x):
        self.calls+=1;dtype=x.dtype;v=x.float()
        scale=grouped_scale(v,64);p=(v.abs()/scale).clamp(0,1)
        if self.mode=='weights':drive=v
        elif self.entry is not None:drive=self.entry.drive(x,self.cycles)
        else:drive=v.sign()*scale*count_bits(p,self.cycles,self.rng)/self.cycles
        self.stats+=torch.stack([torch.tensor(v.numel(),device=v.device),p.sum(dtype=torch.float64),
            v.square().sum(dtype=torch.float64),(drive-v).square().sum(dtype=torch.float64)]).double()
        return F.linear(drive,self.weight_decoded,self.bias).to(dtype)
    def diagnostics(self):
        n,power,energy,error=self.stats.tolist()
        return {'calls':self.calls,'input_values':int(n),'mean_probability':power/max(n,1),
            'sampling_input_nmse':error/max(energy,1e-30),'weight_quantization_nmse':self.weight_nmse,
            'shape':[self.out_features,self.in_features],'cycles':self.cycles,'mode':self.mode,
            'shared_input_cache':None if self.entry is None else {'hits':self.entry.hits,'draws':self.entry.draws}}

SCOPES={'out':['out_proj'],'out_z':['out_proj','in_proj_z'],
    'out_z_qkv':['out_proj','in_proj_z','in_proj_qkv']}

def install(model,scope,mode,cycles,seed):
    result={};handles=[]
    for i,l in enumerate(model.model.language_model.layers):
        if l.block_type!='linear_attention':continue
        delta=l.linear_attn
        shared=EntryCache(940001+seed*1000+i*10) if scope=='out_z_qkv' and mode=='iid' else None
        if shared:
            handles.append(delta.register_forward_pre_hook(shared.clear))
            handles.append(delta.register_forward_hook(shared.clear))
        for index,name in enumerate(SCOPES[scope]):
            entry=shared if name in ['in_proj_z','in_proj_qkv'] else None
            # Each projection type has a stable seed across cumulative scopes.
            offset={'out_proj':1,'in_proj_z':2,'in_proj_qkv':3}[name]
            module=Projection(getattr(delta,name),mode,cycles,930001+seed*1000+i*10+offset,entry)
            setattr(delta,name,module);result[f'L{i}.{name}']=module
    assert len(result)==18*len(SCOPES[scope])
    model._projection_handles=handles
    return result
