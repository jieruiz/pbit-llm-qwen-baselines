import hashlib,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from fixed_ffn import FixedFFN
from streaming_ffn import StreamConfig
from projection import install as install_projection
from attention import install_attention
ROOT=Path(__file__).resolve().parent
def protected_hash(model):
    h=hashlib.sha256()
    for n,p in model.named_parameters():
        if '.mlp.' in n or '.self_attn.q_proj.' in n or '.self_attn.k_proj.' in n:continue
        h.update(n.encode());h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()
def load(ffn,qcycles,attn,seed,samples=512,chunk=32):
    m=AutoModelForCausalLM.from_pretrained(ROOT/'models/Qwen2.5-0.5B',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();m.requires_grad_(False)
    before=protected_hash(m)
    fc=json.loads((ROOT/'calibration_ffn.json').read_text())['policies']['group_max']
    pc=json.loads((ROOT/'calibration_projection.json').read_text())
    ffns={}
    if ffn!='original':
        for i,l in enumerate(m.model.layers):
            f=FixedFFN(l.mlp,StreamConfig(cycles=4096,interval=256,burn_fraction=.25,weight_bits=8,layer=i,seed=seed),
                fc[str(i)],12,'mean' if ffn=='mean' else 'streaming',diagnostics=True)
            l.mlp=f;ffns[i]=f
    proj=install_projection(m,pc,'qk','iid',10,qcycles,seed) if qcycles else {}
    controls=install_attention(m,attn,seed,samples,chunk=chunk)
    assert before==protected_hash(m)
    return m,ffns,proj,controls,before
