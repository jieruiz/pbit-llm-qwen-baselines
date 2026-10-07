"""Frozen-model offline teacher; train/test source separation is explicit."""
import json
import hashlib
import math
from pathlib import Path
from types import MethodType
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
from router import features


@torch.inference_mode()
def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained('models/Qwen2.5-0.5B', torch_dtype=torch.bfloat16, attn_implementation='sdpa', local_files_only=True).cuda().eval()
    path = Path('data/wikitext-2/wiki.train.raw')
    ids = tok(path.read_text(), add_special_tokens=False, return_tensors='pt').input_ids[0]
    split = len(ids) - 65536
    train_starts = torch.linspace(0, split - 8192, 32).long().tolist()
    valid_starts = list(range(split, len(ids) - 8192 + 1, 8192))
    records = [[] for _ in range(24)]

    def wrapped(self, hidden_states, position_ids=None, position_embeddings=None, **kwargs):
        b, n, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(b, n, 14, 64).transpose(1, 2)
        k = self.k_proj(hidden_states).view(b, n, 2, 64).transpose(1, 2)
        cos, sin = self.rotary_emb(k, position_ids) if position_embeddings is None else position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        rows = []
        for length in range(1024, 8193, 1024):
            qi, ki = q[:, :, length-1:length], k[:, :, :length]
            scores = (qi.float() @ repeat_kv(ki, 7).float().transpose(-1, -2)) / 8
            mass = scores.softmax(-1).reshape(1, 14, -1, 64).sum(-1)
            xs = []
            for reps in (1, 4):
                sub = 64 // reps
                means = ki.float().reshape(1, 2, -1, sub, 64).mean(-2)
                sizes = torch.full((length // sub,), sub, device='cuda')
                x = features(qi, means, sizes, length, reps)
                xs.append(F.pad(x, (0, 0, 0, 128 - x.shape[-2])).half().cpu()[0])
            rows.append((xs[0], xs[1], F.pad(mass, (0, 128 - mass.shape[-1])).cpu()[0], length // 64))
        records[self.layer_idx].extend(rows)
        return self._collect_original(hidden_states, position_ids=position_ids, position_embeddings=position_embeddings, **kwargs)

    for layer in model.model.layers:
        layer.self_attn._collect_original = layer.self_attn.forward
        layer.self_attn.forward = MethodType(wrapped, layer.self_attn)
    for i, start in enumerate(train_starts + valid_starts):
        model.model(input_ids=ids[start:start+8192][None].cuda(), use_cache=False)
        print(json.dumps({'window': i+1, 'total': 40}), flush=True)
    out = {}
    for field, key in enumerate(('x1', 'x4', 'mass', 'counts')):
        out[key] = torch.stack([torch.stack([torch.as_tensor(row[field]) for row in layer]) for layer in records])
    # L,N,H,C,F -> L,H,N,C,F
    for key in ('x1', 'x4', 'mass'):
        out[key] = out[key].transpose(1, 2).contiguous()
    out['counts'] = out['counts'][0]
    out['train_n'] = len(train_starts) * 8
    Path('artifacts').mkdir(exist_ok=True)
    torch.save(out, 'artifacts/calibration.pt')
    Path('results').mkdir(exist_ok=True)
    Path('results/data_protocol.json').write_text(json.dumps({'source': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'source_tokens':len(ids), 'split_token':split, 'train_starts':train_starts, 'valid_starts':valid_starts, 'window':8192, 'query_lengths':list(range(1024,8193,1024)), 'test_used_in_training':False}, indent=2))


if __name__ == '__main__':
    main()
