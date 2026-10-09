import json
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from model import load,ROOT
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-0.5B',local_files_only=True)
model,ffns,proj,ctl,before=load('original',0,'sdpa',0)
x=tok('The attention implementation should preserve the causal structure and the cached decoding behavior. '*8,
      return_tensors='pt').input_ids[:,:80].cuda()
with torch.inference_mode():
    a=model(x,use_cache=False).logits.float()
    pre0=model(x[:,:64],past_key_values=DynamicCache(),use_cache=True)
    tail0=model(x[:,64:65],past_key_values=pre0.past_key_values,use_cache=True).logits.float()
    for c in ctl:c.mode='dense'
    b=model(x,use_cache=False).logits.float()
    diff=(a-b).abs();assert diff.mean()<.06 and diff.max()<.8,(diff.mean(),diff.max())
    y=x.clone();y[:,40:]=0
    changed=model(y,use_cache=False).logits.float()
    torch.testing.assert_close(b[:,:40],changed[:,:40],atol=0,rtol=0)
    pre=model(x[:,:64],past_key_values=DynamicCache(),use_cache=True)
    tail=model(x[:,64:65],past_key_values=pre.past_key_values,use_cache=True)
    err=(tail.logits[:,0].float()-b[:,64]).abs()
    original_err=(tail0[:,0]-a[:,64]).abs()
    cross_err=(tail.logits.float()-tail0).abs()
    print('cache comparison',err.mean().item(),err.max().item(),'original',original_err.mean().item(),original_err.max().item(),'cross',cross_err.mean().item(),cross_err.max().item(),flush=True)
    assert cross_err.mean()<.06 and cross_err.max()<.8
print('dense reference',diff.mean().item(),diff.max().item(),'cache',err.mean().item(),err.max().item(),flush=True)
(ROOT/'integration.json').write_text(json.dumps({'dense_mean_abs_error':diff.mean().item(),'dense_max_abs_error':diff.max().item(),
    'cache_mean_abs_error':err.mean().item(),'cache_max_abs_error':err.max().item(),
    'original_cache_mean_abs_error':original_err.mean().item(),'original_cache_max_abs_error':original_err.max().item(),
    'cached_manual_vs_sdpa_mean':cross_err.mean().item(),'cached_manual_vs_sdpa_max':cross_err.max().item(),'causal_verified':True},indent=2)+'\n')
