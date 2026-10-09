"""Decode-only block selection, compact QK, exact Softmax, exact or sampled PV."""
from pathlib import Path
from types import MethodType
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from metrics import cache, score, sizes, mandatory

ROOT=Path(__file__).resolve().parent

def compact_attention(q,k,v,mask,samples=0,generator=None):
    # One query, GQA: compact candidate keys before the QK dot product.
    b,h,_,d=q.shape;length=k.shape[-2];groups=h//k.shape[1];rows=b*h
    tokens=mask.repeat_interleave(64,-1)[...,:length].reshape(rows,length)
    counts=tokens.sum(-1);assert bool((counts>0).all())
    width=int(counts.max().item())
    row,col=tokens.nonzero(as_tuple=True)
    slots=tokens.long().cumsum(-1)[row,col]-1
    index=torch.zeros(rows,width,device=q.device,dtype=torch.long)
    index[row,slots]=col
    valid=torch.arange(width,device=q.device)[None,:]<counts[:,None]
    r=torch.arange(rows,device=q.device);batch=r//h;head=(r%h)//groups
    keys=k[batch[:,None],head[:,None],index].float()
    logits=torch.bmm(q.reshape(rows,1,d).float(),keys.transpose(1,2)).squeeze(1)/(d**.5)
    p=logits.masked_fill(~valid,-torch.inf).softmax(-1)
    if samples:
        selected=torch.multinomial(p,samples,True,generator=generator)
        addresses=index.gather(1,selected)
        values=v[batch[:,None],head[:,None],addresses].float()
        out=values.mean(1)
        used=torch.zeros_like(tokens).scatter_(1,addresses,True)
        unique=used.sum()
        value_events=rows*samples
    else:
        values=v[batch[:,None],head[:,None],index].float()
        out=torch.bmm(p.unsqueeze(1),values).squeeze(1)
        unique=counts.sum();value_events=rows*width
    return out.reshape(b,h,1,d).to(q.dtype), {
        'selected':counts.sum(), 'possible':rows*length, 'padded_qk':rows*width,
        'value_events':value_events,'unique_v':unique,
    }

class Controller:
    def __init__(self,mode,context,seed,layer,samples=512):
        self.mode=mode;self.layer=layer;self.samples=samples
        self.select_rng=torch.Generator(device='cuda').manual_seed(seed*10000+layer+2100)
        self.pv_rng=torch.Generator(device='cuda').manual_seed(seed*10000+layer+5100)
        self.params=None
        if mode.startswith('screen'):
            self.params=torch.load(ROOT/'moment.pt',map_location='cuda',weights_only=True)[f'{context}_0.3_max4']
        self.calls=0;self.prefill_calls=0;self.decode_calls=0
        self.stats=torch.zeros(7,device='cuda',dtype=torch.float64)

    def select(self,attn,q,k):
        n=sizes(k.shape[-2],q.device)
        if self.params is None:return torch.ones((*q.shape[:2],n.numel()),device=q.device,dtype=torch.bool)
        s=score(q,cache(attn,k,'max4'),k.shape[-2],'max4')
        fixed=mandatory(n);free=(~fixed).float();den=free.sum().clamp_min(1)
        mu=(s*free).sum(-1,keepdim=True)/den
        std=(((s-mu).square()*free).sum(-1,keepdim=True)/den).sqrt().clamp_min(.01)
        p=self.params;i=self.layer
        pi=(((s-mu)/std+p['parameter'][i][None,:,None])/p['temperature'][i][None,:,None]).sigmoid()
        pi=pi.masked_fill(fixed,1.)
        return torch.rand(pi.shape,device=q.device,generator=self.select_rng)<pi

    def summary(self):
        selected,possible,padded,events,unique,scoremac,union=self.stats.tolist()
        return dict(calls=self.calls,prefill_calls=self.prefill_calls,decode_calls=self.decode_calls,
            selected_token_fraction=selected/max(possible,1),padded_qk_fraction=padded/max(possible,1),
            unique_V_fraction=unique/max(possible,1),V_vector_events_fraction=events/max(possible,1),
            selector_MAC_ratio_to_dense_QK_PV=scoremac/max(128*possible,1),
            GQA_union_fraction=union/max(possible/7,1),
            selected_token_rows=selected,possible_token_rows=possible,
            scope='decode only; sampled V events are gathers/additions, not weighted MACs; no hardware cost claim')

def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
            output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None):
    c=self._ctl;c.calls+=1;b,n,_=hidden_states.shape
    c.prefill_calls+=int(n>1);c.decode_calls+=int(n==1)
    if n>1 or c.mode=='sdpa':
        if n>1:self._metrics_cache=None
        return self._original(hidden_states,attention_mask=attention_mask,position_ids=position_ids,
            past_key_value=past_key_value,output_attentions=output_attentions,use_cache=use_cache,
            cache_position=cache_position,position_embeddings=position_embeddings)
    assert attention_mask is None and not output_attentions,'unpadded causal single-token decode only'
    q=self.q_proj(hidden_states).view(b,n,14,64).transpose(1,2)
    k=self.k_proj(hidden_states).view(b,n,2,64).transpose(1,2)
    v=self.v_proj(hidden_states).view(b,n,2,64).transpose(1,2)
    cos,sin=self.rotary_emb(v,position_ids) if position_embeddings is None else position_embeddings
    q,k=apply_rotary_pos_emb(q,k,cos,sin)
    if past_key_value is not None:k,v=past_key_value.update(k,v,self.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
    mask=c.select(self,q,k)
    out,st=compact_attention(q,k,v,mask,c.samples if c.mode.endswith('sampled') else 0,c.pv_rng)
    ns=sizes(k.shape[-2],q.device)
    union=(mask.reshape(b,2,7,-1).any(2)*ns).sum()
    scoremac=mask.numel()*256 if c.params is not None else 0
    c.stats+=torch.stack([torch.as_tensor(x,device=q.device,dtype=torch.float64) for x in
        [st['selected'],st['possible'],st['padded_qk'],st['value_events'],st['unique_v'],scoremac,union]])
    return self.o_proj(out.transpose(1,2).contiguous().reshape(b,n,896)),None,past_key_value

def install(model,mode,context,seed):
    controls=[]
    for i,layer in enumerate(model.model.layers):
        a=layer.self_attn;c=Controller(mode,context,seed,i)
        a._ctl=c;a._original=a.forward;a.forward=MethodType(forward,a);controls.append(c)
    return controls
