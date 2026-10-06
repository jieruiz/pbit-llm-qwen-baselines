"""Exact hard-limit transient calculation on actual, unmodified Qwen logits."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import expm_multiply
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb,repeat_kv


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    model=AutoModelForCausalLM.from_pretrained('models/Qwen2.5-0.5B',local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='sdpa').cuda().eval()
    tokenizer=AutoTokenizer.from_pretrained('models/Qwen2.5-0.5B',local_files_only=True)
    ids=tokenizer(Path('data/wikitext-2/wiki.test.raw').read_text(),add_special_tokens=False,return_tensors='pt').input_ids[0]
    out=[]
    pairs=[(0,0),(6,3),(12,7),(18,10),(23,13)]
    for ctx,examples in ((2048,32),(8192,16)):
        starts=torch.linspace(0,len(ids)-(ctx+128),examples).round().long().tolist()
        offsets=[starts[0],starts[len(starts)//2]]
        tokens=torch.stack([ids[s:s+ctx] for s in offsets]).cuda()
        with torch.inference_mode():
            pref=model.model(tokens[:,:-1],use_cache=True)
            cache=pref.past_key_values;del pref
            recorded={};hooks=[]
            for layer,head in pairs:
                def save(module,inputs,output,layer=layer):recorded[layer]=output.detach()
                hooks.append(model.model.layers[layer].self_attn.q_proj.register_forward_hook(save))
            final=model.model(tokens[:,-1:],past_key_values=cache,use_cache=True)
            cache=final.past_key_values
            for h in hooks:h.remove()
            for layer,head in pairs:
                attn=model.model.layers[layer].self_attn
                k,v=cache[layer]
                q=recorded[layer].view(2,1,14,64).transpose(1,2)
                cos,sin=attn.rotary_emb(v,torch.full((2,1),ctx-1,device='cuda',dtype=torch.long))
                q,_=apply_rotary_pos_emb(q,q,cos,sin)
                scores=(q.float()@repeat_kv(k,7).float().transpose(-1,-2))/8
                for example,z in enumerate(scores[:,head,0].double().cpu().numpy()):
                    order=np.r_[z.argmax(),np.delete(np.arange(ctx),z.argmax())]
                    delta=z[order]-z.max()
                    a=1/(1+np.exp(-np.maximum(delta[1:],-700)))
                    W=np.exp(delta[1:]).sum()
                    target=np.exp(delta);target/=target.sum()
                    rr=np.r_[np.zeros(ctx-1,dtype=int),np.arange(1,ctx),np.arange(ctx)]
                    cc=np.r_[np.arange(1,ctx),np.zeros(ctx-1,dtype=int),np.arange(ctx)]
                    val=np.r_[a,1-a,-np.r_[a.sum(),1-a]]
                    Q=coo_matrix((val,(rr,cc)),shape=(ctx,ctx)).tocsr()
                    init=np.zeros(ctx);init[0]=1
                    curve=expm_multiply(Q.T,init,start=0,stop=16,num=33,endpoint=True,traceA=Q.diagonal().sum())
                    checks={str(t):float(np.abs(curve[int(t*2)]-target).sum()/2) for t in (0,.5,1,4,8,16)}
                    record={'context':ctx,'layer':layer,'head':head,'example_start':offsets[example],
                        'p_max':float(target[0]),'birth_rate_sum':float(a.sum()),'exact_transient_TV':checks,
                        'lambda24_invalid_mass_upper_bound':float(.5*W*W/(1+W)*math.exp(-24)*math.exp(W*math.exp(-48))),
                        'lambda32_hard_vs_finite_path_difference_upper_bound_at_t4100':float(min(1.,4100*math.exp(-32)*W))}
                    out.append(record);print(json.dumps(record),flush=True)
        del cache,final,tokens,recorded
    root=Path('results/pbit_softmax_router_20261006')
    (root/'real_logit_transients.json').write_text(json.dumps({'rows':out,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'note':'20 real score rows; exact ideal hard-limit generator at beta=1, initial max-reference state; not an exhaustive hardware guarantee.'},indent=2)+'\n')


if __name__=='__main__':main()
