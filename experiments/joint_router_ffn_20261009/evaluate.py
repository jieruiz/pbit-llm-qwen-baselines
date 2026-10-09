import argparse,hashlib,json,math,time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from model import ROOT,load,protected_hash

p=argparse.ArgumentParser()
p.add_argument('--ffn',choices=['original','fixed'],default='fixed')
p.add_argument('--mode',choices=['sdpa','all_exact','all_sampled','screen_exact','screen_sampled'],required=True)
p.add_argument('--context',type=int,choices=[2048,8192],required=True)
p.add_argument('--seed',type=int,default=0)
p.add_argument('--examples',type=int,default=32);p.add_argument('--decode',type=int,default=128)
p.add_argument('--batch',type=int,default=4);p.add_argument('--output',required=True)
a=p.parse_args();out=Path(a.output);assert not out.exists()
torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen2.5-0.5B',local_files_only=True)
m,ff,ctl,before=load(a.ffn,a.mode,a.context,a.seed)
frozen={f'{i}.{s}':getattr(f,'scale_'+s).clone() for i,f in ff.items() for s in ('x','g','u','product')}
ids=tok((ROOT/'data/wikitext-2/wiki.test.raw').read_text(),add_special_tokens=False,return_tensors='pt').input_ids[0]
span=a.context+a.decode;starts=torch.linspace(0,len(ids)-span,a.examples).round().long().tolist()
assert len(set(starts))==a.examples
records=[];total=0.;pt=dt=0.;start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    for first in range(0,len(starts),a.batch):
        offsets=starts[first:first+a.batch]
        tokens=torch.stack([ids[o:o+span] for o in offsets]).cuda()
        torch.cuda.synchronize();t=time.perf_counter()
        h=m.model(tokens[:,:a.context-1],past_key_values=DynamicCache(),use_cache=True)
        cache=h.past_key_values;del h;torch.cuda.synchronize();pt+=time.perf_counter()-t
        t=time.perf_counter();loss=torch.zeros(len(offsets),device='cuda')
        for step in range(a.decode):
            pos=a.context-1+step
            h=m.model(tokens[:,pos:pos+1],past_key_values=cache,use_cache=True);cache=h.past_key_values
            logits=m.lm_head(h.last_hidden_state[:,-1]).float();assert bool(torch.isfinite(logits).all())
            loss+=F.cross_entropy(logits,tokens[:,pos+1],reduction='none')
        torch.cuda.synchronize();dt+=time.perf_counter()-t
        assert cache.get_seq_length()==span-1
        for j,(offset,nll) in enumerate(zip(offsets,loss.tolist())):
            records.append(dict(offset=offset,nll=nll,tokens=a.decode,ids_sha256=hashlib.sha256(tokens[j].cpu().numpy().tobytes()).hexdigest()))
            total+=nll
        del cache,h,tokens,logits
        print(json.dumps(dict(examples=len(records),total=a.examples,ppl=math.exp(total/(len(records)*a.decode)))),flush=True)
batches=math.ceil(a.examples/a.batch);calls=batches*(a.decode+1)
assert all(c.calls==calls and c.prefill_calls==batches and c.decode_calls==batches*a.decode for c in ctl)
assert len(ctl)==24 and len(ff)==(24 if a.ffn=='fixed' else 0)
for i,f in ff.items():
    assert f.forward_calls==calls
    for s in ('x','g','u','product'):assert torch.equal(getattr(f,'scale_'+s),frozen[f'{i}.{s}'])
assert before==protected_hash(m)
result=dict(args=vars(a),perplexity=math.exp(total/(a.examples*a.decode)),scored_tokens=a.examples*a.decode,
    source_tokens=len(ids),records=records,prefill_seconds=pt,decode_seconds=dt,seconds=time.perf_counter()-start,
    peak_allocated_bytes=torch.cuda.max_memory_allocated(),protected_sha256_before=before,protected_sha256_after=protected_hash(m),
    attention_layers=24,ffn_layers=len(ff),ffn_diagnostics={str(i):f.diagnostics() for i,f in ff.items()},
    attention_stats=[c.summary() for c in ctl],training=False,scales_unchanged=True,
    prefill_attention='original SDPA',decode_attention=a.mode,ffn_scope='all24 prefill AND decode' if ff else 'original',
    calibration_sha256={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in ('calibration_ffn.json','moment.pt')},
    limits=['WT2 suffix subset, not full corpus PPL','old train-calibrated selector; no combined retraining',
        'QK compacted before multiplication with ragged padding; padded positions charged separately',
        'sampled PV gathers512 values then averages; no exact weighted PV in that path',
        'score summaries initialized on first decode; included in timing','no hardware energy or latency measurement'])
out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
print('complete',out.name,result['perplexity'],flush=True)
