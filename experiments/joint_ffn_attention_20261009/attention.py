from types import MethodType
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb,repeat_kv
from sampling import ising_counts,iid_counts
class Controller:
    def __init__(self,mode,seed,samples=512,burn=16.,spacing=4.,chunk=32):
        self.mode=mode;self.samples=samples;self.burn=burn;self.spacing=spacing;self.chunk=chunk
        self.rng=torch.Generator(device='cuda').manual_seed(seed)
        self.calls=0;self.prefill_calls=0;self.decode_calls=0
        # head rows, possible causal positions, distinct selected, GQA union selected/possible, repeat weighted
        self.stats=torch.zeros(6,device='cuda',dtype=torch.float64)
    def summary(self):
        n,possible,selected,union,upos,repeat=self.stats.tolist()
        return {'calls':self.calls,'prefill_calls':self.prefill_calls,'decode_calls':self.decode_calls,
            'head_queries':int(n),'unique_V_fraction':selected/max(possible,1),
            'GQA_union_V_fraction':union/max(upos,1),'adjacent_repeat':repeat/max(n,1),
            'scope':'prefill AND decode; logical addresses, not physical memory traffic'}
def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
            output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None):
    c=self._joint_ctl;c.calls+=1
    b,n,_=hidden_states.shape;c.prefill_calls+=int(n>1);c.decode_calls+=int(n==1)
    if c.mode=='sdpa':
        return self._original_forward(hidden_states,attention_mask=attention_mask,position_ids=position_ids,
            past_key_value=past_key_value,output_attentions=output_attentions,use_cache=use_cache,
            cache_position=cache_position,position_embeddings=position_embeddings)
    assert not output_attentions
    q=self.q_proj(hidden_states).view(b,n,self.num_heads,self.head_dim).transpose(1,2)
    k=self.k_proj(hidden_states).view(b,n,self.num_key_value_heads,self.head_dim).transpose(1,2)
    v=self.v_proj(hidden_states).view(b,n,self.num_key_value_heads,self.head_dim).transpose(1,2)
    cos,sin=self.rotary_emb(v,position_ids) if position_embeddings is None else position_embeddings
    q,k=apply_rotary_pos_emb(q,k,cos,sin)
    if past_key_value is not None:
        k,v=past_key_value.update(k,v,self.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
    k=repeat_kv(k,self.num_key_value_groups);v=repeat_kv(v,self.num_key_value_groups)
    length=k.shape[-2];offset=length-n;outputs=[]
    for first in range(0,n,c.chunk):
        stop=min(first+c.chunk,n);qc=stop-first
        scores=(q[:,:,first:stop].float()@k.float().transpose(-1,-2))/8
        mask=torch.arange(length,device=q.device)[None,None,None,:]>(offset+torch.arange(first,stop,device=q.device))[None,None,:,None]
        if attention_mask is not None:
            assert attention_mask.ndim==4
            scores=scores+attention_mask[:,:,first:stop,:length].float()
        scores=scores.masked_fill(mask,-torch.inf)
        repeat=scores.new_tensor(0.)
        if c.mode=='dense':weights=scores.softmax(-1)
        elif c.mode=='iid':weights=iid_counts(scores,c.samples,c.rng)/c.samples
        elif c.mode=='ising':
            counts,repeat=ising_counts(scores,c.samples,c.burn,c.spacing,c.rng);weights=counts/c.samples
        else:raise ValueError(c.mode)
        assert not bool((weights.masked_select(mask.expand_as(weights))!=0).any())
        selected=weights>0;valid=(~mask).expand(b,1,qc,length)
        union=selected.reshape(b,self.num_key_value_heads,self.num_key_value_groups,qc,length).any(2)
        rows=b*self.num_heads*qc
        c.stats+=torch.stack([scores.new_tensor(rows),valid.sum()*self.num_heads,selected.sum(),
            union.sum(),valid.sum()*self.num_key_value_heads,repeat*rows]).double()
        outputs.append((weights@v.float()).to(hidden_states.dtype))
    out=torch.cat(outputs,2).transpose(1,2).contiguous().reshape(b,n,self.hidden_size)
    return self.o_proj(out),None,past_key_value
def install_attention(model,mode,seed,samples=512,burn=16.,spacing=4.,chunk=32):
    controls=[]
    for i,l in enumerate(model.model.layers):
        a=l.self_attn;c=Controller(mode,seed*10000+7300+i,samples,burn,spacing,chunk)
        a._joint_ctl=c;a._original_forward=a.forward;a.forward=MethodType(forward,a);controls.append(c)
    return controls
