"""Controlled decode-only attention experiments; Qwen FFNs remain original."""
import math
from types import MethodType
import torch
from torch.nn import functional as F
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from sampling import ising_counts, iid_counts, choose_blocks


class Controller:
    def __init__(self,mode,samples,seed,device='cuda',burn=4.,spacing=1.,ratio=.25,block_size=64,rank=64):
        self.mode,self.samples,self.burn,self.spacing=mode,samples,burn,spacing
        self.ratio,self.block_size,self.rank=ratio,block_size,rank
        self.generator=torch.Generator(device=device).manual_seed(seed)
        self.stats=torch.zeros(12,dtype=torch.float64,device=device)

    def record(self,p,weights,selected,groups,extra):
        b,h,q,l=p.shape
        union=selected.reshape(b,h//groups,groups,q,l).any(2)
        rows=b*h*q
        self.stats+=torch.stack([p.new_tensor(rows),selected.float().sum(),p.new_tensor(selected.numel()),
            union.float().sum(),p.new_tensor(union.numel()),(p*selected).sum(),
            (p-weights).abs().sum()/2,(p*(p.clamp_min(1e-30).log()-weights.clamp_min(1e-30).log())).sum(),
            extra.get('adjacent_repeat',p.new_tensor(0.))*rows,
            extra.get('finite_lambda24_leak_leading_bound',p.new_tensor(0.))*rows,
            extra.get('selected_blocks',p.new_tensor(0.)),extra.get('all_blocks',p.new_tensor(0.))]).double()

    def summary(self):
        a=self.stats.tolist()
        if not a[0]: return {}
        return {'head_query_count':int(a[0]),'unique_value_fraction_per_head':a[1]/a[2],
            'unique_value_fraction_GQA_group_union':a[3]/a[4],
            'teacher_probability_mass_on_selected_positions':a[5]/a[0],
            'mean_attention_TV':a[6]/a[0],'mean_attention_KL_clamped':a[7]/a[0],
            'adjacent_sample_repeat_fraction':a[8]/a[0],
            'lambda24_invalid_mass_leading_upper_bound':a[9]/a[0],
            'selected_block_fraction':a[10]/a[11] if a[11] else None,
            'note':'Logical sparsity only. Dense reference QK/PV and diagnostic teacher softmax remain in this software implementation.'}


def summaries(attn,k,block):
    # Build once from prefix; thereafter add only the new key to current block.
    length=k.shape[-2]
    cache=getattr(attn,'_router_summary',None)
    if cache is None or cache[0]!=length-1 or cache[1].shape[:2]!=k.shape[:2]:
        pad=(-length)%block
        packed=F.pad(k.float(),(0,0,0,pad)).reshape(*k.shape[:2],-1,block,k.shape[-1])
        sums=packed.sum(-2); squares=packed.square().sum(-2)
    else:
        _,sums,squares=cache
        if (length-1)%block==0:
            sums=torch.cat([sums,torch.zeros_like(sums[:,:,:1])],-2)
            squares=torch.cat([squares,torch.zeros_like(squares[:,:,:1])],-2)
        sums[:,:,-1]+=k[:,:,-1].float(); squares[:,:,-1]+=k[:,:,-1].float().square()
    attn._router_summary=(length,sums,squares)
    sizes=torch.full((sums.shape[-2],),float(block),device=k.device)
    sizes[-1]=length-(sizes.numel()-1)*block
    mean=sums/sizes[:,None]
    var=(squares/sizes[:,None]-mean.square()).clamp_min(0)
    return mean,var,sizes


def forward(self,hidden_states,attention_mask=None,position_ids=None,past_key_value=None,
            output_attentions=False,use_cache=False,cache_position=None,position_embeddings=None):
    ctl=self._router_controller
    if hidden_states.shape[1]!=1 or ctl.mode=='sdpa':
        if hidden_states.shape[1]!=1: self._router_summary=None
        return self._original_attention_forward(hidden_states,attention_mask=attention_mask,position_ids=position_ids,
            past_key_value=past_key_value,output_attentions=output_attentions,use_cache=use_cache,
            cache_position=cache_position,position_embeddings=position_embeddings)
    if output_attentions: raise ValueError('No attention outputs in this diagnostic patch')
    b,q_len,_=hidden_states.shape
    q=self.q_proj(hidden_states).view(b,q_len,self.num_heads,self.head_dim).transpose(1,2)
    k=self.k_proj(hidden_states).view(b,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)
    v=self.v_proj(hidden_states).view(b,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)
    cos,sin=self.rotary_emb(v,position_ids) if position_embeddings is None else position_embeddings
    q,k=apply_rotary_pos_emb(q,k,cos,sin)
    if past_key_value is not None:
        k,v=past_key_value.update(k,v,self.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
    # Router constructed from compact block summaries BEFORE the dense reference QK.
    blockmask=None
    if ctl.mode.startswith('block_'):
        mean,var,sizes=summaries(self,k,ctl.block_size)
        mean=repeat_kv(mean,self.num_key_value_groups); var=repeat_kv(var,self.num_key_value_groups)
        dims=torch.arange(0,self.head_dim,self.head_dim//ctl.rank,device=q.device)[:ctl.rank]
        qs=q.float().index_select(-1,dims)
        ms=mean.index_select(-1,dims); vs=var.index_select(-1,dims)
        factor=self.head_dim/ctl.rank
        mu=(qs@ms.transpose(-1,-2))*factor/math.sqrt(self.head_dim)
        if ctl.mode=='block_moment':
            variance=(qs.square()@vs.transpose(-1,-2))*factor/self.head_dim
            proxy=mu+.5*variance+sizes.log()
        elif ctl.mode=='block_paper':
            # Paper-inspired sigmoid confidence and layer cooling; no trained
            # image mask predictor exists in Qwen. Explicitly an adaptation.
            proxy=torch.sigmoid(mu)*(self.layer_idx+1)
        else:
            proxy=mu+sizes.log()
        blockmask=choose_blocks(proxy,ctl.ratio,ctl.generator)
    k=repeat_kv(k,self.num_key_value_groups); v=repeat_kv(v,self.num_key_value_groups)
    scores=(q.float()@k.float().transpose(-1,-2))/math.sqrt(self.head_dim)
    if attention_mask is not None: scores+=attention_mask[:,:,:,:k.shape[-2]].float()
    p=scores.softmax(-1)  # reference/diagnostic; NOT used to drive Ising dynamics or proxy router
    extra={}
    if ctl.mode=='dense': weights=p
    elif ctl.mode=='iid': weights=iid_counts(scores,ctl.samples,ctl.generator)/ctl.samples
    elif ctl.mode=='ising':
        counts,extra=ising_counts(scores,ctl.samples,ctl.burn,ctl.spacing,ctl.generator)
        weights=counts/ctl.samples
    else:
        if ctl.mode.startswith('oracle'):
            mass=F.pad(p,(0,(-p.shape[-1])%ctl.block_size)).reshape(*p.shape[:-1],-1,ctl.block_size).sum(-1)
            blockmask=choose_blocks(mass.clamp_min(1e-30).log(),ctl.ratio,ctl.generator,ctl.mode=='oracle_topk')
        selected=blockmask.repeat_interleave(ctl.block_size,-1)[...,:scores.shape[-1]]
        weights=scores.masked_fill(~selected,-torch.inf).softmax(-1)
        extra={'selected_blocks':blockmask.float().sum(),'all_blocks':p.new_tensor(blockmask.numel())}
    selected=weights>0
    ctl.record(p,weights,selected,self.num_key_value_groups,extra)
    out=(weights@v.float()).to(hidden_states.dtype).transpose(1,2).contiguous().reshape(b,q_len,self.hidden_size)
    return self.o_proj(out),None,past_key_value


def install(model,controller):
    for layer in model.model.layers:
        attn=layer.self_attn
        if hasattr(attn,'_original_attention_forward'): raise ValueError('Already patched')
        attn._original_attention_forward=attn.forward
        attn._router_controller=controller
        attn.forward=MethodType(forward,attn)
