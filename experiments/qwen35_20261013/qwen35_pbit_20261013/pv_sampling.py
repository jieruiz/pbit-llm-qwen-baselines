"""Replace only the attention interface; official Q/K norms, RoPE, KV and gate stay intact."""
import copy
from dataclasses import dataclass
import torch
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv

@dataclass
class Control:
    mode:str
    samples:int
    seed:int
    layer:int
    chunk:int=32
    calls:int=0
    prefill_calls:int=0
    decode_calls:int=0
    def __post_init__(self):
        self.rng=torch.Generator(device='cuda').manual_seed(self.seed*10000+7300+self.layer)
        self.stats=torch.zeros(3,device='cuda',dtype=torch.float64)
    def summary(self):
        possible,selected,rows=self.stats.tolist()
        return {'layer':self.layer,'mode':self.mode,'samples':self.samples,'calls':self.calls,
            'prefill_calls':self.prefill_calls,'decode_calls':self.decode_calls,
            'possible_causal_V_addresses':int(possible),'selected_distinct_V_addresses':int(selected),
            'head_queries':int(rows),'selected_V_fraction':selected/max(possible,1),
            'note':'Logical selection count; reference compresses selection-add into count-weighted matmul, not GPU acceleration.'}

def probabilities(scores,valid):
    # Left-padding may make an entire query row invalid. Such rows output zero.
    has=valid.any(-1,keepdim=True)
    scores=scores.masked_fill(~valid,-torch.inf)
    scores=torch.where(has,scores,torch.zeros_like(scores))
    return scores.softmax(-1)*has

def sampled_weights(p,samples,rng):
    shape=p.shape;flat=p.reshape(-1,shape[-1]);has=flat.sum(-1,keepdim=True)>0
    safe=torch.where(has,flat,torch.ones_like(flat))
    idx=torch.multinomial(safe,samples,replacement=True,generator=rng)
    counts=torch.zeros_like(flat).scatter_add_(1,idx,torch.ones_like(idx,dtype=flat.dtype))
    return (counts/samples*has).reshape(shape)

def interface(module,query,key,value,attention_mask,dropout=0.,scaling=None,**kwargs):
    assert dropout==0. and not module.training
    c=module._pbit_control;c.calls+=1
    b,heads,n,dim=query.shape;length=key.shape[-2];offset=length-n
    assert heads==8 and dim==256 and offset>=0
    c.prefill_calls+=int(n>1);c.decode_calls+=int(n==1)
    key=repeat_kv(key,module.num_key_value_groups)
    value=repeat_kv(value,module.num_key_value_groups)
    outputs=[]
    for first in range(0,n,c.chunk):
        stop=min(first+c.chunk,n)
        scores=(query[:,:,first:stop].float()@key.float().transpose(-1,-2))*scaling
        valid=(torch.arange(length,device=query.device)[None,None,None,:]
            <=offset+torch.arange(first,stop,device=query.device)[None,None,:,None]).expand_as(scores)
        if attention_mask is not None:
            assert attention_mask.ndim==4
            am=attention_mask[:,:,first:stop,:length]
            if am.dtype==torch.bool:valid=valid & am
            else:
                valid=valid & (am>-10000)
                scores=scores+am.float()
        p=probabilities(scores,valid)
        weights=p if c.mode=='dense' else sampled_weights(p,c.samples,c.rng)
        assert not bool((weights.masked_select(~valid)!=0).any())
        c.stats+=torch.stack([valid.sum(),(weights>0).sum(),
            torch.tensor(b*heads*(stop-first),device=query.device)]).double()
        outputs.append((weights@value.float()).to(query.dtype))
    # Qwen3.5Attention.forward applies the original sigmoid gate and o_proj afterward.
    return torch.cat(outputs,2).transpose(1,2).contiguous(),None

ALL_ATTENTION_FUNCTIONS.register('qwen35_pbit_pv',interface)

def install(model,mode,seed,samples=512):
    controls=[]
    for index,layer in enumerate(model.model.language_model.layers):
        if layer.block_type!='full_attention':continue
        attention=layer.self_attn
        c=Control(mode,samples,seed,index)
        attention._pbit_control=c
        # The backbone and all DeltaNet modules keep their original shared config.
        attention.config=copy.copy(attention.config)
        attention.config._attn_implementation='qwen35_pbit_pv'
        controls.append(c)
    assert [c.layer for c in controls]==[3,7,11,15,19,23]
    return controls
