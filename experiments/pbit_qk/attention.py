"""Decode-only stochastic QK for Qwen2 / Transformers 4.45.2.

All FFNs and PV remain original continuous operations. Projections, RoPE,
and dense prefill are retained. Diagnostics compare the same current Q/K
state, not a separately evolved exact model trajectory.
"""
import math
from types import MethodType
import torch
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from sampling import encode


class Controller:
    def __init__(self, mode, samples, seed, device='cuda'):
        self.mode, self.samples = mode, samples
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.stats = torch.zeros(11, dtype=torch.float64, device=device)

    def record(self, c, groups, exact_scores, approx_scores, exact_logp, approx_logp):
        if c is None:
            return
        b, heads, queries, d = c.shape
        selected = c > 0
        union = selected.reshape(b, heads//groups, groups, queries, d).any(2)
        p, phat = exact_logp.exp(), approx_logp.exp()
        values = [selected.double().sum(), selected.new_tensor(selected.numel(), dtype=torch.float64),
                  union.double().sum(), union.new_tensor(union.numel(), dtype=torch.float64),
                  c.double().sum(),
                  (approx_scores-exact_scores).double().square().sum(),
                  exact_scores.double().square().sum(),
                  (p*(exact_logp-approx_logp)).sum(-1).double().sum(),
                  ((p-phat).abs().sum(-1)*.5).double().sum(),
                  c.new_tensor(c.numel()/d, dtype=torch.float64),
                  (p.argmax(-1) == phat.argmax(-1)).double().sum()]
        self.stats += torch.stack(values)

    def summary(self):
        a = self.stats.tolist()
        if not a[1]:
            return {}
        return {'unique_key_feature_fraction_per_head': a[0]/a[1],
                'unique_key_feature_fraction_GQA_group_union': a[2]/a[3],
                'mean_active_bit_fraction_per_round': a[4]/(self.samples*a[1]),
                'signed_add_events_relative_to_dense_mac_terms': a[4]/a[1],
                'score_relative_l2_global': math.sqrt(a[5]/max(a[6], 1e-30)),
                'attention_KL_exact_to_sampled': a[7]/a[9],
                'attention_total_variation': a[8]/a[9],
                'attention_argmax_agreement': a[10]/a[9],
                'head_query_count': int(a[9]),
                'access_note': 'Logical union of key features across B rounds and GQA heads; not physical bandwidth.',
                'diagnostic_note': 'Exact and approximate attention compared at same current Q/K state; FP32 dense reference.'}


def forward(self, hidden_states, attention_mask=None, position_ids=None,
            past_key_value=None, output_attentions=False, use_cache=False,
            cache_position=None, position_embeddings=None):
    ctl = self._pbit_controller
    if hidden_states.shape[1] != 1 or ctl.mode == 'sdpa':
        return self._original_attention_forward(hidden_states, attention_mask=attention_mask,
            position_ids=position_ids, past_key_value=past_key_value,
            output_attentions=output_attentions, use_cache=use_cache,
            cache_position=cache_position, position_embeddings=position_embeddings)
    if output_attentions:
        raise ValueError('diagnostic patch does not expose attention maps')
    bsz, q_len, _ = hidden_states.shape
    q = self.q_proj(hidden_states).view(bsz,q_len,self.num_heads,self.head_dim).transpose(1,2)
    k = self.k_proj(hidden_states).view(bsz,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)
    v = self.v_proj(hidden_states).view(bsz,q_len,self.num_key_value_heads,self.head_dim).transpose(1,2)
    cos,sin = self.rotary_emb(v,position_ids) if position_embeddings is None else position_embeddings
    q,k = apply_rotary_pos_emb(q,k,cos,sin)
    if past_key_value is not None:
        k,v = past_key_value.update(k,v,self.layer_idx,{'sin':sin,'cos':cos,'cache_position':cache_position})
    k = repeat_kv(k,self.num_key_value_groups).float()
    v = repeat_kv(v,self.num_key_value_groups).float()
    exact_scores = (q.float() @ k.transpose(-1,-2))/math.sqrt(self.head_dim)
    c = None
    if ctl.mode == 'dense':
        scores = exact_scores
    else:
        qhat,c = encode(q, ctl.samples, ctl.mode, ctl.generator)
        scores = (qhat @ k.transpose(-1,-2))/math.sqrt(self.head_dim)
    raw_scores = scores
    exact_logits = exact_scores
    if attention_mask is not None:
        mask = attention_mask[:,:,:,:k.shape[-2]].float()
        scores = scores + mask
        exact_logits = exact_logits + mask
    p = scores.softmax(-1)
    if c is not None:
        ctl.record(c,self.num_key_value_groups,exact_scores,raw_scores,
                   exact_logits.log_softmax(-1),scores.log_softmax(-1))
    # PV is dense and continuous: there is NO value-side sampling here.
    out = p @ v
    out = out.to(hidden_states.dtype).transpose(1,2).contiguous().reshape(bsz,q_len,self.hidden_size)
    return self.o_proj(out),None,past_key_value


def install(model, controller):
    for layer in model.model.layers:
        attention = layer.self_attn
        if hasattr(attention,'_original_attention_forward'):
            raise ValueError('attention already patched')
        attention._original_attention_forward = attention.forward
        attention._pbit_controller = controller
        attention.forward = MethodType(forward,attention)
