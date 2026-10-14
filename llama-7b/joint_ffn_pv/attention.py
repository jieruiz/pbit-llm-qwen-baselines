"""Exact Q/K/softmax with either dense P@V or IID sampled P@V for LLaMA."""

from types import MethodType

import torch
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv


def iid_counts(scores, samples, generator):
    shape = scores.shape
    probabilities = scores.reshape(-1, shape[-1]).float().softmax(-1)
    index = torch.multinomial(
        probabilities, samples, replacement=True, generator=generator
    )
    counts = torch.zeros_like(probabilities).scatter_add_(
        1, index, torch.ones_like(index, dtype=probabilities.dtype)
    )
    return counts.reshape(shape)


class Controller:
    def __init__(self, mode, seed, samples=512, chunk=32):
        self.mode = mode
        self.samples = samples
        self.chunk = chunk
        self.rng = torch.Generator(device="cuda").manual_seed(seed)
        self.calls = 0
        self.prefill_calls = 0
        self.decode_calls = 0
        # query rows, possible addresses, selected addresses, MHA/GQA union, union possible
        self.stats = torch.zeros(5, device="cuda", dtype=torch.float64)

    def summary(self):
        rows, possible, selected, union, union_possible = self.stats.tolist()
        return {
            "mode": self.mode,
            "samples": self.samples,
            "calls": self.calls,
            "prefill_calls": self.prefill_calls,
            "decode_calls": self.decode_calls,
            "head_queries": int(rows),
            "unique_v_fraction": selected / max(possible, 1),
            "kv_group_union_v_fraction": union / max(union_possible, 1),
            "scope": "prefill and decode; logical V addresses, not physical traffic",
        }


def forward(
    self,
    hidden_states,
    attention_mask=None,
    position_ids=None,
    past_key_value=None,
    output_attentions=False,
    use_cache=False,
    cache_position=None,
    position_embeddings=None,
    **kwargs,
):
    del kwargs
    controller = self._pv_controller
    controller.calls += 1
    batch, query_length, _ = hidden_states.shape
    controller.prefill_calls += int(query_length > 1)
    controller.decode_calls += int(query_length == 1)
    if output_attentions:
        raise RuntimeError("This reference path does not return attention weights")

    query = self.q_proj(hidden_states).view(
        batch, query_length, self.num_heads, self.head_dim
    ).transpose(1, 2)
    key = self.k_proj(hidden_states).view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)
    value = self.v_proj(hidden_states).view(
        batch, query_length, self.num_key_value_heads, self.head_dim
    ).transpose(1, 2)

    if position_embeddings is None:
        cos, sin = self.rotary_emb(value, position_ids)
    else:
        cos, sin = position_embeddings
    query, key = apply_rotary_pos_emb(query, key, cos, sin)

    if past_key_value is not None:
        key, value = past_key_value.update(
            key,
            value,
            self.layer_idx,
            {"sin": sin, "cos": cos, "cache_position": cache_position},
        )

    key = repeat_kv(key, self.num_key_value_groups)
    value = repeat_kv(value, self.num_key_value_groups)
    key_length = key.shape[-2]
    offset = key_length - query_length
    scale = getattr(self, "scaling", self.head_dim**-0.5)
    outputs = []

    for first in range(0, query_length, controller.chunk):
        stop = min(first + controller.chunk, query_length)
        chunk_length = stop - first
        scores = (
            query[:, :, first:stop].float() @ key.float().transpose(-1, -2)
        ) * scale
        causal_mask = torch.arange(key_length, device=query.device)[None, None, None, :] > (
            offset + torch.arange(first, stop, device=query.device)
        )[None, None, :, None]
        if attention_mask is not None:
            if attention_mask.ndim != 4:
                raise RuntimeError(f"Expected 4D attention mask, got {attention_mask.shape}")
            scores = scores + attention_mask[:, :, first:stop, :key_length].float()
        scores = scores.masked_fill(causal_mask, -torch.inf)

        if controller.mode == "dense":
            weights = scores.softmax(-1)
        elif controller.mode == "iid":
            weights = iid_counts(scores, controller.samples, controller.rng)
            weights = weights / controller.samples
        else:
            raise ValueError(controller.mode)

        if bool((weights.masked_select(causal_mask.expand_as(weights)) != 0).any()):
            raise RuntimeError("Causal masking failure")
        selected = weights > 0
        valid = (~causal_mask).expand(batch, 1, chunk_length, key_length)
        union = selected.reshape(
            batch,
            self.num_key_value_heads,
            self.num_key_value_groups,
            chunk_length,
            key_length,
        ).any(2)
        rows = batch * self.num_heads * chunk_length
        controller.stats += torch.stack(
            [
                scores.new_tensor(rows),
                valid.sum() * self.num_heads,
                selected.sum(),
                union.sum(),
                valid.sum() * self.num_key_value_heads,
            ]
        ).double()
        outputs.append((weights @ value.float()).to(hidden_states.dtype))

    attention_output = (
        torch.cat(outputs, 2)
        .transpose(1, 2)
        .contiguous()
        .reshape(batch, query_length, self.hidden_size)
    )
    return self.o_proj(attention_output), None, past_key_value


def install_attention(model, mode, seed, samples=512, chunk=32):
    controls = []
    for layer_index, layer in enumerate(model.model.layers):
        attention = layer.self_attn
        controller = Controller(
            mode, seed * 10000 + 7300 + layer_index, samples=samples, chunk=chunk
        )
        attention._pv_controller = controller
        attention._original_forward = attention.forward
        attention.forward = MethodType(forward, attention)
        controls.append(controller)
    return controls
