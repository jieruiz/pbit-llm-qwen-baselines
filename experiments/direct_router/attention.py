import math
import os
from pathlib import Path
from types import MethodType
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from router import Router, features, summary_cache, force_blocks
from sparse import sparse_attention


class Controller:
    def __init__(self, mode, reps, ratio, seed, backend='sparse'):
        self.mode, self.reps, self.ratio, self.backend = mode, reps, ratio, backend
        self.generator = torch.Generator(device='cuda').manual_seed(seed)
        self.stats = torch.zeros(7, dtype=torch.float64, device='cuda')
        self.router = None
        if mode == 'learned':
            checkpoint = torch.load(f'artifacts/router_r{reps}.pt', map_location='cuda', weights_only=True)
            self.router = Router(reps).cuda().eval()
            self.router.load_state_dict(checkpoint['state_dict'])
            self.offset = torch.zeros(24,14,device='cuda')
            if os.environ.get('ROUTER_CALIBRATED'):
                self.offset = torch.load(f'artifacts/offset_r{reps}.pt',map_location='cuda',weights_only=True)

    def select(self, attn, q, k):
        means, sizes = summary_cache(attn, k, self.reps)
        length = k.shape[-2]
        if self.mode == 'mean':
            score = (q.float() @ repeat_kv(means, attn.num_key_value_groups).transpose(-1,-2)).squeeze(-2) / 8 + sizes.log()
            p = score.softmax(-1)
            pi = -torch.expm1(math.ceil(p.shape[-1] * self.ratio) * torch.log1p(-p.clamp(max=1-1e-7)))
        elif self.mode == 'all':
            pi = torch.ones((*q.shape[:2], math.ceil(length / 64)), device=q.device)
        else:
            x = features(q, means, sizes, length, self.reps)
            pi = (self.router(x, self.ratio, attn.layer_idx)+self.offset[attn.layer_idx][None,:,None]).sigmoid()
        mask = force_blocks(torch.rand(pi.shape, device=q.device, generator=self.generator) < pi).contiguous()
        return mask

    def record(self, mask, length, groups):
        sizes = (length - torch.arange(mask.shape[-1], device=mask.device) * 64).clamp(1,64)
        selected = (mask * sizes).sum()
        union = (mask.reshape(mask.shape[0], -1, groups, mask.shape[-1]).any(2) * sizes).sum()
        rows = mask.shape[0] * mask.shape[1]
        router_macs = self.router.macs_per_block() if self.router is not None else 64
        self.stats += torch.stack((selected, selected.new_tensor(rows*length), union,
            selected.new_tensor(rows*length//groups), mask.sum(), selected.new_tensor(mask.numel()),
            selected.new_tensor(rows*mask.shape[-1]*router_macs))).double()

    def summary(self):
        a = self.stats.tolist()
        if not a[1]: return {}
        return {'selected_token_fraction':a[0]/a[1], 'GQA_union_fraction':a[2]/a[3], 'selected_block_fraction':a[4]/a[5],
                'QK_PV_MAC_ratio':a[0]/a[1], 'selector_MAC_ratio_to_dense_QK_PV':a[6]/(128*a[1]),
                'total_QK_PV_selector_MAC_ratio':(128*a[0]+a[6])/(128*a[1]),
                'selected_token_rows':a[0], 'possible_token_rows':a[1],
                'cost_scope':'decode QK/PV and selector linear MACs; excludes summary init/update, elementwise/reductions, projections, FFNs, memory and launch costs; NOT whole-model FLOPs'}


def forward(self, hidden_states, attention_mask=None, position_ids=None, past_key_value=None,
            output_attentions=False, use_cache=False, cache_position=None, position_embeddings=None):
    ctl = self._controller
    if hidden_states.shape[1] != 1 or ctl.mode == 'sdpa':
        if hidden_states.shape[1] != 1: self._direct_summary = None
        return self._original(hidden_states, attention_mask=attention_mask, position_ids=position_ids,
            past_key_value=past_key_value, output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings)
    if attention_mask is not None or output_attentions:
        raise ValueError('This kernel requires unpadded causal single-query decode')
    b, n, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(b,n,self.num_heads,self.head_dim).transpose(1,2)
    k = self.k_proj(hidden_states).view(b,n,self.num_key_value_heads,self.head_dim).transpose(1,2)
    v = self.v_proj(hidden_states).view(b,n,self.num_key_value_heads,self.head_dim).transpose(1,2)
    cos,sin = self.rotary_emb(v,position_ids) if position_embeddings is None else position_embeddings
    q,k = apply_rotary_pos_emb(q,k,cos,sin)
    if past_key_value is not None:
        k,v = past_key_value.update(k,v,self.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
    mask = ctl.select(self,q,k)
    snapshot = os.environ.get('ROUTER_SNAPSHOT')
    if snapshot and self.layer_idx == 12 and not getattr(ctl, '_saved', False):
        torch.save({'q':q.cpu(),'k':k.cpu(),'v':v.cpu(),'mask':mask.cpu()},snapshot)
        ctl._saved = True
    if ctl.backend == 'sparse':
        out = sparse_attention(q,k,v,mask)
        if os.environ.get('ROUTER_CHECK') and not getattr(self,'_checked',False):
            scores = q.float() @ repeat_kv(k,7).float().transpose(-1,-2) / 8
            probs = scores.masked_fill(~mask.repeat_interleave(64,-1)[...,:k.shape[-2]].unsqueeze(-2),-torch.inf).softmax(-1)
            expected = (probs @ repeat_kv(v,7).float()).to(q.dtype)
            error = (out.float()-expected.float()).abs()
            print({'check_layer':self.layer_idx,'max_error':error.max().item(),'mean_error':error.mean().item(),'output_rms':expected.float().square().mean().sqrt().item()},flush=True)
            self._checked=True
    else:  # numerical validation only; not claimed as sparse computation
        scores = q.float() @ repeat_kv(k,7).float().transpose(-1,-2) / 8
        p = scores.masked_fill(~mask.repeat_interleave(64,-1)[...,:k.shape[-2]].unsqueeze(-2),-torch.inf).softmax(-1)
        out = (p @ repeat_kv(v,7).float()).to(q.dtype)
    ctl.record(mask,k.shape[-2],7)
    out = out.transpose(1,2).contiguous().reshape(b,n,self.hidden_size)
    return self.o_proj(out),None,past_key_value


def install(model, ctl):
    for layer in model.model.layers:
        attn = layer.self_attn
        attn._original = attn.forward; attn._controller = ctl
        attn.forward = MethodType(forward,attn)
