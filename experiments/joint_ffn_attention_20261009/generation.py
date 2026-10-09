import gc,json
import torch
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from model import load,ROOT
from run import spec,read,write,name
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-0.5B',local_files_only=True)
q=read(ROOT/'results/selection.json')['qcycles']
cases=[spec('original',0,'sdpa'),spec('fixed',0,'sdpa'),spec('fixed',q,'ising'),spec('fixed',q,'iid'),spec('fixed',0,'iid')]
records=[]
for s in cases:
    m,ff,pp,cc,_=load(s['ffn'],s['qcycles'],s['attn'],0)
    for prompt in ['The capital of France is','人工智能可以帮助人类','To calculate the area of a rectangle,']:
        x=tok(prompt,return_tensors='pt').input_ids.cuda();cache=DynamicCache();tokens=[]
        with torch.inference_mode():
            for _ in range(48):
                h=m.model(x,past_key_values=cache,use_cache=True);cache=h.past_key_values
                t=m.lm_head(h.last_hidden_state[:,-1]).argmax(-1,keepdim=True)
                tokens.append(t.item());x=t
                if t.item()==tok.eos_token_id:break
        records.append({'spec':s,'prompt':prompt,'continuation':tok.decode(tokens,skip_special_tokens=True),'tokens':len(tokens)})
    print('generation',name(s),flush=True)
    del m,ff,pp,cc,h,cache;gc.collect();torch.cuda.empty_cache()
write(ROOT/'results/generation.json',records)
