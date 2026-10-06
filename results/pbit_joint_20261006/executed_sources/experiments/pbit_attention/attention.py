"""Decode-only Qwen2 attention patch for Transformers 4.45.2.

All original projections, RoPE, caches, FFNs, and normalization are retained.
Dense prefill is delegated to the original SDPA implementation.
"""
import math
from types import MethodType
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from sampling import aggregate


class Controller:
    def __init__(self, mode, samples, seed, device="cuda"):
        self.mode, self.samples = mode, samples
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.stats = torch.zeros(8, dtype=torch.float64, device=device)

    def record(self, probability, counts, groups):
        if counts is None:
            return
        total = counts.sum(-1)
        selected = counts > 0
        b, heads, queries, length = counts.shape
        group_union = selected.reshape(b, heads//groups, groups, queries, length).any(2)
        self.stats += torch.stack([
            total.double().sum(), total.double().square().sum(),
            total.new_tensor(total.numel(), dtype=torch.float64),
            selected.double().sum(), selected.new_tensor(selected.numel(), dtype=torch.float64),
            group_union.double().sum(), group_union.new_tensor(group_union.numel(), dtype=torch.float64),
            (total == 0).double().sum()])

    def summary(self):
        a = self.stats.tolist()
        if not a[2]:
            return {}
        mean = a[0]/a[2]
        return {"head_query_count": int(a[2]), "mean_selected_events_per_S_rounds": mean,
                "selected_events_std": max(0., a[1]/a[2]-mean**2)**.5,
                "mean_selected_events_per_round": mean/self.samples,
                "zero_total_events_fraction": a[7]/a[2],
                "unique_value_fraction_per_head": a[3]/a[4],
                "unique_value_fraction_GQA_group_union": a[5]/a[6],
                "access_note": "Logical distinct rows, not measured memory transactions; reference uses dense count @ V."}


def forward(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False,
            cache_position=None, position_embeddings=None):
    ctl = self._pbit_controller
    if hidden_states.shape[1] != 1 or ctl.mode == "sdpa":
        return self._original_attention_forward(hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings)
    if output_attentions:
        raise ValueError("diagnostic patch does not expose attention maps")
    bsz, q_len, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)
    k = self.k_proj(hidden_states).view(bsz,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)
    v = self.v_proj(hidden_states).view(bsz,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)
    cos,sin = self.rotary_emb(v,position_ids) if position_embeddings is None else position_embeddings
    q,k = apply_rotary_pos_emb(q,k,cos,sin)
    if past_key_value is not None:
        k,v = past_key_value.update(k,v,self.layer_idx,{"sin":sin,"cos":cos,"cache_position":cache_position})
    k = repeat_kv(k,self.num_key_value_groups)
    v = repeat_kv(v,self.num_key_value_groups)
    scores = (q.float() @ k.float().transpose(-1,-2))/math.sqrt(self.head_dim)
    if attention_mask is not None:
        scores += attention_mask[:,:,:,:k.shape[-2]].float()
    p = scores.softmax(-1)
    out,counts = aggregate(p,v,ctl.mode,ctl.samples,ctl.generator)
    ctl.record(p,counts,self.num_key_value_groups)
    out = out.to(hidden_states.dtype).transpose(1,2).contiguous().reshape(bsz,q_len,self.hidden_size)
    return self.o_proj(out),None,past_key_value


def install(model, controller):
    for layer in model.model.layers:
        attention = layer.self_attn
        if hasattr(attention,"_original_attention_forward"):
            raise ValueError("attention already patched")
        attention._original_attention_forward = attention.forward
        attention._pbit_controller = controller
        attention.forward = MethodType(forward,attention)
