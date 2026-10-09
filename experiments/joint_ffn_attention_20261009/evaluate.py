import argparse,hashlib,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from model import load,protected_hash,ROOT
p=argparse.ArgumentParser()
p.add_argument('--ffn',choices=['original','fixed','mean'],default='fixed')
p.add_argument('--qcycles',type=int,default=256);p.add_argument('--attn',choices=['sdpa','dense','ising','iid'],default='ising')
p.add_argument('--samples',type=int,default=512);p.add_argument('--seed',type=int,default=0)
p.add_argument('--context',type=int,default=2048);p.add_argument('--decode',type=int,default=128)
p.add_argument('--examples',type=int,default=8);p.add_argument('--split',choices=['validation','test'],default='validation')
p.add_argument('--batch',type=int,default=4)
p.add_argument('--chunk',type=int,default=32);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);assert not out.exists()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-0.5B',local_files_only=True)
model,ffns,proj,ctl,before=load(a.ffn,a.qcycles,a.attn,a.seed,a.samples,a.chunk)
frozen={str(i)+'.'+s:getattr(m,'scale_'+s).clone() for i,m in ffns.items() for s in ('x','g','u','product')}
pfrozen={k:m.scale.clone() for k,m in proj.items()}
ids=tok((ROOT/f'data/wikitext-2/wiki.{"train" if a.split=="validation" else "test"}.raw').read_text(),
    add_special_tokens=False,return_tensors='pt').input_ids[0];source_tokens=len(ids)
if a.split=='validation':ids=ids[-65536:]
span=a.context+a.decode;starts=torch.linspace(0,len(ids)-span,a.examples).round().long().tolist()
assert len(set(starts))==a.examples
records=[];total=0.;prefilltime=decodetime=0.;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    for first in range(0,len(starts),a.batch):
        offsets=starts[first:first+a.batch]
        tokens=torch.stack([ids[offset:offset+span] for offset in offsets]).cuda()
        torch.cuda.synchronize();t=time.perf_counter()
        pref=model.model(tokens[:,:a.context-1],past_key_values=DynamicCache(),use_cache=True)
        cache=pref.past_key_values;del pref;torch.cuda.synchronize();prefilltime+=time.perf_counter()-t
        t=time.perf_counter();nll=torch.zeros(len(offsets),device='cuda')
        for step in range(a.decode):
            pos=a.context-1+step
            h=model.model(tokens[:,pos:pos+1],past_key_values=cache,use_cache=True);cache=h.past_key_values
            logits=model.lm_head(h.last_hidden_state[:,-1]).float();assert torch.isfinite(logits).all()
            nll+=F.cross_entropy(logits,tokens[:,pos+1],reduction='none')
        torch.cuda.synchronize();decodetime+=time.perf_counter()-t
        assert cache.get_seq_length()==span-1
        losses=nll.tolist();total+=sum(losses)
        for j,(offset,loss) in enumerate(zip(offsets,losses)):
            records.append({'offset':offset,'nll':loss,'tokens':a.decode,'ids_sha256':hashlib.sha256(tokens[j].cpu().numpy().tobytes()).hexdigest()})
        del cache,h,logits,tokens
        print(json.dumps({'examples':len(records),'total':a.examples,'ppl':math.exp(total/(len(records)*a.decode)),
            'prefill_seconds':prefilltime,'decode_seconds':decodetime}),flush=True)
batches=math.ceil(a.examples/a.batch);calls=batches*(a.decode+1)
assert all(c.calls==calls and c.prefill_calls==batches and c.decode_calls==batches*a.decode for c in ctl)
for i,m in ffns.items():
    assert m.forward_calls==calls
    for s in ('x','g','u','product'):assert torch.equal(getattr(m,'scale_'+s),frozen[str(i)+'.'+s])
for k,m in proj.items():assert m.calls==calls and torch.equal(m.scale,pfrozen[k])
assert protected_hash(model)==before
result={'args':vars(a),'perplexity':math.exp(total/(a.examples*a.decode)),'scored_tokens':a.examples*a.decode,
    'source_tokens':source_tokens,'records':records,'prefill_seconds':prefilltime,'decode_seconds':decodetime,
    'seconds':time.perf_counter()-start,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
    'protected_sha256_before':before,'protected_sha256_after':protected_hash(model),
    'ffn_layers':len(ffns),'projection_count':len(proj),'attention_layers':24,
    'attention_stats':[c.summary() for c in ctl],'ffn_diagnostics':{str(i):m.diagnostics() for i,m in ffns.items()},
    'projection_diagnostics':{k:m.diagnostics() for k,m in proj.items()},
    'scales_unchanged':True,'all_prefill_and_decode_verified':True,'training':False,
    'ffn_config':'fixed group_max, W8, magnitude12+sign, input4096/output3072, interval256,burn25%,ideal sigmoid',
    'attention_config':'Ising hard-exclusion CTMC burn16 spacing4; exact QK; q/k fixed projections W10 sign+9 magnitude if enabled',
    'calibration_sha256':{f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('calibration_ffn.json','calibration_projection.json')},
    'limits':['WT2 cached suffix subset, NOT full corpus PPL','all sampled prefill is included','FP32 counts and matmuls, not sparse CUDA','Ising hard penalty limit, not finite device','no energy/area/native latency claim']}
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
print('complete',out.name,result['perplexity'],flush=True)
