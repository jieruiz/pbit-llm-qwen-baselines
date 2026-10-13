"""Original Qwen3.5 checkpoint: full WT2, zero-shot tasks, generation and timing."""
import argparse, hashlib, importlib.metadata, json, math, os, time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

ROOT = Path(__file__).resolve().parent
DATA = Path('/home/Hongjie_Zeng/joint_ffn_attention_20261009/data')
TASKS = ['wikitext2','hellaswag','piqa','winogrande','arc_easy','arc_challenge','generation_performance']
EXPECTED = dict(hellaswag=10042,piqa=1838,winogrande=1267,arc_easy=2376,arc_challenge=1172)

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(path, value):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp'); tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str)+'\n'); tmp.replace(path)

def load():
    tok=AutoTokenizer.from_pretrained(ROOT/'models/Qwen3.5-0.8B',local_files_only=True)
    model, info=Qwen3_5ForConditionalGeneration.from_pretrained(ROOT/'models/Qwen3.5-0.8B',
        dtype=torch.bfloat16,attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
    assert not info.get('missing_keys'), info
    assert not info.get('mismatched_keys'), info
    model=model.eval().cuda()
    assert len(model.model.language_model.layers)==24
    config=model.config.text_config
    assert config.layer_types.count('full_attention')==6
    return tok,model,info

def task_eval(task,tok,model,smoke=False):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from datasets import load_dataset
    entry=json.loads((DATA/'full_benchmarks/manifest.json').read_text())[task]
    for file in entry['files']: assert sha(file['path'])==file['sha256']
    files=[f['path'] for f in entry['files']]
    rows=len(load_dataset('parquet',data_files={entry['split']:files},split=entry['split']))
    assert rows==EXPECTED[task]
    from lm_eval.tasks._yaml_loader import load_yaml
    config=load_yaml(TaskManager().task_index[task].yaml_path, resolve_func=True, recursive=True)
    config.update(dataset_path='parquet',dataset_name=None,
        dataset_kwargs={'data_files':{entry['split']:files}},training_split=None,
        validation_split=entry['split'] if entry['split']=='validation' else None,
        test_split=entry['split'] if entry['split']=='test' else None)
    lm=HFLM(pretrained=model,tokenizer=tok,backend='causal',batch_size=1,max_length=2048,
        logits_cache=False,add_bos_token=False)
    result=lm_eval.simple_evaluate(model=lm,tasks=[config],num_fewshot=0,
        limit=2 if smoke else None,log_samples=True,bootstrap_iters=1000,
        random_seed=0,numpy_random_seed=0,torch_random_seed=0,fewshot_random_seed=0)
    return {'dataset':entry,'dataset_rows':rows,'harness_result':result}

def wt2(tok,model,smoke=False):
    path=DATA/'wikitext-2/wiki.test.raw'
    ids=tok(path.read_text(),add_special_tokens=False,return_tensors='pt').input_ids
    if smoke: ids=ids[:,:128]
    total=0.;count=0;previous_end=1;records=[];start=time.perf_counter()
    for begin in range(0,ids.shape[1],1024):
        end=min(begin+2048,ids.shape[1]);x=ids[:,begin:end].cuda()
        hidden=model.model(input_ids=x,use_cache=False).last_hidden_state
        first=max(previous_end,begin+1);nll=0.
        for target in range(first,end,64):
            stop=min(target+64,end)
            logits=model.lm_head(hidden[:,target-begin-1:stop-begin-1]).float()
            assert torch.isfinite(logits).all()
            nll+=F.cross_entropy(logits.reshape(-1,logits.shape[-1]),
                x[:,target-begin:stop-begin].reshape(-1),reduction='sum').item()
        if smoke:
            direct=model(input_ids=x,use_cache=False).logits[:,:-1].float()
            reference=F.cross_entropy(direct.reshape(-1,direct.shape[-1]),x[:,1:].reshape(-1),reduction='sum').item()
            assert abs(reference-nll)<max(0.02,abs(reference)*1e-5),(reference,nll)
        count+=end-first;total+=nll;previous_end=end
        records.append({'begin':begin,'end':end,'scored':end-first,'nll_sum':nll})
        print(json.dumps({'window':len(records),'scored_tokens':count,'ppl':math.exp(total/count),'seconds':time.perf_counter()-start}),flush=True)
        if end==ids.shape[1]:break
    assert count==ids.shape[1]-1
    return {'perplexity':math.exp(total/count),'nll_sum':total,'scored_tokens':count,
        'corpus_tokens':ids.shape[1],'corpus_sha256':sha(path),'windows':records}

def generation_perf(tok,model):
    examples=[]
    for prompt in ['请简要解释什么是随机计算。','What is the difference between a probability and a random sample?','Write a Python function to add two integers.']:
        text=tok.apply_chat_template([{'role':'user','content':prompt}],tokenize=False,
            add_generation_prompt=True,enable_thinking=False)
        x=tok(text,return_tensors='pt',add_special_tokens=False).to('cuda')
        start=time.perf_counter()
        y=model.generate(**x,max_new_tokens=128,do_sample=False)
        torch.cuda.synchronize()
        examples.append({'prompt':prompt,'output':tok.decode(y[0,x.input_ids.shape[1]:],skip_special_tokens=True),
            'new_tokens':int(y.shape[1]-x.input_ids.shape[1]),'seconds':time.perf_counter()-start})
    ids=tok((DATA/'wikitext-2/wiki.test.raw').read_text(),add_special_tokens=False,return_tensors='pt').input_ids
    # Small warmup; measurements describe this backend and are not energy measurements.
    model(input_ids=ids[:,:128].cuda(),use_cache=False,logits_to_keep=1)
    timing=[]
    for length in [2048,8192]:
        torch.cuda.reset_peak_memory_stats();x=ids[:,:length].cuda()
        torch.cuda.synchronize();start=time.perf_counter()
        out=model(input_ids=x,use_cache=True,logits_to_keep=1)
        torch.cuda.synchronize();prefill=time.perf_counter()-start
        cache=out.past_key_values; nxt=out.logits[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(64):
            out=model(input_ids=nxt,past_key_values=cache,use_cache=True,logits_to_keep=1)
            cache=out.past_key_values;nxt=out.logits[:,-1].argmax(-1,keepdim=True)
        torch.cuda.synchronize();decode=time.perf_counter()-start
        timing.append({'context':length,'prefill_seconds':prefill,'decode_tokens':64,
            'decode_seconds':decode,'decode_ms_per_token':1000*decode/64,
            'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
            'peak_reserved_bytes':torch.cuda.max_memory_reserved()})
        del out,cache,x;torch.cuda.empty_cache()
    return {'examples':examples,'timing':timing,'timing_scope':'One warm run; single GPU, batch 1; not a power/energy measurement'}

def main():
    if not os.environ.get('SLURM_JOB_ID'): raise SystemExit('GPU execution requires Slurm.')
    p=argparse.ArgumentParser();p.add_argument('--task',choices=TASKS+['smoke'],required=True);a=p.parse_args()
    out=ROOT/'results'/('smoke.json' if a.task=='smoke' else f'{a.task}.json')
    if out.exists(): print('ALREADY_COMPLETE',out,flush=True);return
    torch.set_num_threads(4);torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False
    tok,model,loading=load();start=time.perf_counter()
    meta={'task':a.task,'model':json.loads((ROOT/'deployment_manifest.json').read_text()),
        'slurm_job_id':os.environ['SLURM_JOB_ID'],'gpu':torch.cuda.get_device_name(),
        'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'source_sha256':sha(__file__),
        'context':2048,'stride':1024,'batch_size':1,'dtype':'BF16 weights/activations; native FP32 internals retained',
        'attention_backend':'sdpa','zero_shot':True,'apply_chat_template_for_scoring':False,
        'scope':'All original 24 layers; no random FFN/attention; raw text likelihood; vision encoder unused',
        'loading_info':loading}
    with torch.inference_mode():
        if a.task=='smoke':
            result={'wt2_smoke':wt2(tok,model,True),'piqa_smoke':task_eval('piqa',tok,model,True)}
            x=tok('A small test of recurrent cache consistency.',return_tensors='pt').input_ids.cuda()
            one=model(input_ids=x[:,:-1],use_cache=True,logits_to_keep=1)
            cached=model(input_ids=x[:,-1:],past_key_values=one.past_key_values,use_cache=True,logits_to_keep=1).logits.float()
            full=model(input_ids=x,use_cache=False,logits_to_keep=1).logits.float()
            cos=F.cosine_similarity(cached.flatten(),full.flatten(),dim=0).item()
            assert cos>0.999,(cos,(cached-full).abs().max().item())
            result['cache_check']={'cosine':cos,'max_absolute_logit_difference':(cached-full).abs().max().item()}
        elif a.task=='wikitext2':result=wt2(tok,model)
        elif a.task=='generation_performance':result=generation_perf(tok,model)
        else:result=task_eval(a.task,tok,model)
    meta['seconds']=time.perf_counter()-start
    save(out,{'metadata':meta,'result':result});print('COMPLETE',out,flush=True)

if __name__=='__main__':main()
