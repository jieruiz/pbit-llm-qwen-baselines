import json
import math
from pathlib import Path
from types import SimpleNamespace
import torch
import triton
from torch.nn import functional as F
from transformers.models.qwen2.modeling_qwen2 import repeat_kv
from sparse import sparse_attention
from router import summary_cache, force_blocks


def main():
    torch.manual_seed(23)
    checks=[]
    for b,l in ((1,65),(2,1024),(4,8197)):
        q=torch.randn(b,14,1,64,device='cuda',dtype=torch.bfloat16)
        k=torch.randn(b,2,l,64,device='cuda',dtype=torch.bfloat16); v=torch.randn_like(k)
        for ratio in (.1,1.):
            mask=force_blocks(torch.rand(b,14,math.ceil(l/64),device='cuda')<ratio)
            actual=sparse_attention(q,k,v,mask)
            scores=q.float()@repeat_kv(k,7).float().transpose(-1,-2)/8
            p=scores.masked_fill(~mask.repeat_interleave(64,-1)[...,:l].unsqueeze(-2),-torch.inf).softmax(-1)
            expected=(p@repeat_kv(v,7).float()).to(q.dtype)
            error=(actual.float()-expected.float()).abs().max().item()
            assert torch.allclose(actual.float(),expected.float(),atol=.002,rtol=.01),(b,l,ratio,error)
            checks.append({'batch':b,'length':l,'ratio':ratio,'max_abs_error':error})
        for reps in (1,4):
            a=SimpleNamespace(); summary_cache(a,k[:,:,:l-1],reps)
            got,_=summary_cache(a,k,reps)
            expected,_=summary_cache(SimpleNamespace(),k,reps)
            assert torch.allclose(got,expected,atol=1e-6)
    Path('results').mkdir(exist_ok=True)
    Path('results/kernel_checks.json').write_text(json.dumps({'status':'passed','checks':checks,'incremental_summary':'passed'},indent=2))
    print('kernel and incremental cache checks passed',flush=True)


if __name__=='__main__':main()
