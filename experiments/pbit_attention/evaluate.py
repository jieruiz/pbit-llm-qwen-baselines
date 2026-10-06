"""Matched cached, teacher-forced decode test; original FFNs throughout."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from torch.nn import functional as F
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from attention import Controller,install


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--model",default="models/Qwen2.5-0.5B")
    parser.add_argument("--text",default="data/wikitext-2/wiki.test.raw")
    parser.add_argument("--mode",choices=("sdpa","dense","tree","independent"),required=True)
    parser.add_argument("--samples",type=int,default=128)
    parser.add_argument("--seed",type=int,default=0)
    parser.add_argument("--context",type=int,default=2048)
    parser.add_argument("--decode",type=int,default=128)
    parser.add_argument("--examples",type=int,default=32)
    parser.add_argument("--batch",type=int,default=16)
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    if min(args.samples,args.context,args.decode,args.examples,args.batch)<=0 or args.context<2:
        raise ValueError("positive sizes required; context>=2")
    outpath=Path(args.output)
    if outpath.exists():
        raise FileExistsError(outpath)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32=False
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    # Verify every MLP is the untouched standard Qwen implementation.
    if any(type(layer.mlp).__name__!="Qwen2MLP" for layer in model.model.layers):
        raise RuntimeError("unexpected non-original FFN")
    def mlp_hash():
        h=hashlib.sha256()
        for layer in model.model.layers:
            for name,t in layer.mlp.state_dict().items():
                h.update(name.encode()); h.update(t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()
    original_mlp_hash=mlp_hash()
    controller=Controller(args.mode,args.samples,args.seed)
    install(model,controller)
    path=Path(args.text)
    ids=tokenizer(path.read_text(),add_special_tokens=False,return_tensors="pt").input_ids[0]
    span=args.context+args.decode
    starts=torch.linspace(0,len(ids)-span,args.examples).round().long().tolist()
    if len(set(starts))!=len(starts) or len(ids)<span:
        raise ValueError("invalid text spans")
    examples=[]
    total_nll=0.
    elapsed_decode=elapsed_prefill=0.
    torch.cuda.reset_peak_memory_stats()
    overall=time.perf_counter()
    for first in range(0,args.examples,args.batch):
        offsets=starts[first:first+args.batch]
        tokens=torch.stack([ids[s:s+span] for s in offsets]).cuda()
        torch.cuda.synchronize(); begin=time.perf_counter()
        with torch.inference_mode():
            # Hold back the last prompt token: its attention is the first sampled query.
            prefill=model.model(input_ids=tokens[:,:args.context-1],use_cache=True)
            cache=prefill.past_key_values
            del prefill
        torch.cuda.synchronize(); elapsed_prefill+=time.perf_counter()-begin
        torch.cuda.synchronize(); begin=time.perf_counter()
        batch_nll=torch.zeros(len(offsets),device="cuda")
        with torch.inference_mode():
            for step in range(args.decode):
                pos=args.context-1+step
                hidden=model.model(input_ids=tokens[:,pos:pos+1],past_key_values=cache,use_cache=True)
                cache=hidden.past_key_values
                logits=model.lm_head(hidden.last_hidden_state[:,-1]).float()
                batch_nll+=F.cross_entropy(logits,tokens[:,pos+1],reduction="none")
        torch.cuda.synchronize(); elapsed_decode+=time.perf_counter()-begin
        losses=batch_nll.tolist()
        total_nll+=sum(losses)
        for offset,nll in zip(offsets,losses):
            examples.append({"start_token":offset,"first_target_token":offset+args.context,
                "last_target_token":offset+span-1,"scored_tokens":args.decode,"nll_sum":nll})
        del cache,hidden,logits,tokens
        print(json.dumps({"completed_examples":len(examples),"total_examples":args.examples,
              "running_ppl":math.exp(total_nll/(len(examples)*args.decode)),
              "elapsed_seconds":time.perf_counter()-overall}),flush=True)
    final_hash=mlp_hash()
    if original_mlp_hash!=final_hash:
        raise RuntimeError("FFN parameters changed")
    scored=args.examples*args.decode
    result={"test":"cached_teacher_forced_decode_value_sampling","args":vars(args),
        "perplexity":math.exp(total_nll/scored),"mean_nll":total_nll/scored,"scored_tokens":scored,
        "source_tokens":len(ids),"examples":examples,"attention_layers":24,
        "ffn":"original Qwen2MLP; all weights unchanged; no stochastic FFN", "mlp_sha256":final_hash,
        "text_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')},
        "torch":torch.__version__,"transformers":transformers.__version__,"gpu":torch.cuda.get_device_name(),
        "sampling_stats":controller.summary(),"decode_seconds":elapsed_decode,"prefill_seconds":elapsed_prefill,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "precision":"BF16 model/cache, FP32 score-softmax-count aggregation; exact prefill SDPA",
        "limitations":"Sampled WT2 test suffixes, not full WT2 PPL; teacher forcing; no training; arithmetic reference not sparse CUDA acceleration"}
    outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"complete":str(outpath),"ppl":result['perplexity'],"decode_seconds":elapsed_decode}),flush=True)


if __name__=="__main__":
    main()
