import hashlib,json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM
from fixed_ffn import FixedFFN
from streaming_ffn import StreamConfig
from attention import install
ROOT=Path(__file__).resolve().parent

def protected_hash(model):
    h=hashlib.sha256()
    for name,p in model.named_parameters():
        if '.mlp.' in name:continue
        h.update(name.encode());h.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return h.hexdigest()

def load(ffn,mode,context,seed):
    m=AutoModelForCausalLM.from_pretrained(ROOT/'models/Qwen2.5-0.5B',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval();m.requires_grad_(False)
    before=protected_hash(m);ff={}
    if ffn=='fixed':
        cal=json.loads((ROOT/'calibration_ffn.json').read_text())['policies']['group_max']
        for i,l in enumerate(m.model.layers):
            f=FixedFFN(l.mlp,StreamConfig(cycles=4096,interval=256,burn_fraction=.25,weight_bits=8,layer=i,seed=seed),cal[str(i)],12,'streaming',diagnostics=True)
            l.mlp=f;ff[i]=f
    controls=install(m,mode,context,seed)
    assert before==protected_hash(m)
    return m,ff,controls,before
