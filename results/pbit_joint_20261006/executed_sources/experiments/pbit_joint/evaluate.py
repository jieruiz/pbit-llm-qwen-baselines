"""Matched cached decode: independently seeded stochastic AND FFNs and PV sampling."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import time
import sys
from types import MethodType
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"pbit_attention"))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"pdnn_ffn"))
from evaluate_multithreshold_and import prepare_student
import torch
from torch.nn import functional as F
import transformers
from transformers import AutoModelForCausalLM,AutoTokenizer
from attention import Controller,install


def chunked_ffn(self,x):
    if x.numel()//x.shape[-1]<=self._joint_chunk:
        return self._joint_original_forward(x)
    flat=x.reshape(-1,x.shape[-1])
    return torch.cat([self._joint_original_forward(t) for t in flat.split(self._joint_chunk)],0).reshape(x.shape)


def source_hashes():
    exp=Path(__file__).resolve().parents[1]
    paths=[Path(__file__),exp/"pbit_attention/attention.py",exp/"pbit_attention/sampling.py",
        exp/"pdnn_ffn/multithreshold_and_ffn.py",exp/"pdnn_ffn/evaluate_multithreshold_and.py"]
    return {str(p.relative_to(exp.parent)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


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
    parser.add_argument("--ffn",choices=("original","and"),default="original")
    parser.add_argument("--ffn-samples",type=int,default=4)
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--checkpoint-manifest",help="Previous evaluation JSON with checkpoint SHA256 identities")
    parser.add_argument("--ffn-chunk",type=int,default=256)
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
    checkpoint_records=[]
    if args.ffn=="and":
        if not args.checkpoint_dir or not args.checkpoint_manifest:
            raise ValueError("AND requires checkpoint directory and previously recorded identity manifest")
        expected=json.loads(Path(args.checkpoint_manifest).read_text())["checkpoints"]
        assert len(expected)==20
        for entry in sorted(expected,key=lambda e:e["layer"]):
            layer=entry["layer"]
            checkpoint=Path(args.checkpoint_dir)/f"best_layer{layer}.pt"
            digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if digest!=entry["sha256"]: raise ValueError(f"checkpoint hash mismatch: {layer}")
            student,payload=prepare_student(checkpoint)
            assert int(payload["training_args"]["layer"])==layer
            assert student.config.bits==4
            student.set_sample_count(args.ffn_samples)
            student._joint_original_forward=student.forward
            student._joint_chunk=args.ffn_chunk
            student.forward=MethodType(chunked_ffn,student)
            model.model.layers[layer].mlp=student
            checkpoint_records.append({"layer":layer,"sha256":digest,"filename":checkpoint.name,
                "phase":payload["phase"],"step":payload["step"],"bits":student.config.bits})
    # Verify every MLP is the untouched standard Qwen implementation.
    if args.ffn=="original" and any(type(layer.mlp).__name__!="Qwen2MLP" for layer in model.model.layers):
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
    torch.manual_seed(args.seed)  # FFN stream; attention owns a separate torch.Generator.
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
            # Prefill uses exact attention; AND FFNs remain stochastic throughout prefix and decode.
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
    result={"test":"joint_AND_FFN_and_post_softmax_PV_sampling","args":vars(args),
        "perplexity":math.exp(total_nll/scored),"mean_nll":total_nll/scored,"scored_tokens":scored,
        "source_tokens":len(ids),"examples":examples,"attention_layers":24,
        "ffn":args.ffn,"checkpoints":checkpoint_records,"replacement_layers":[r["layer"] for r in checkpoint_records], "mlp_sha256":final_hash,
        "text_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_sha256":source_hashes(),
        "torch":torch.__version__,"transformers":transformers.__version__,"gpu":torch.cuda.get_device_name(),
        "sampling_stats":controller.summary(),"decode_seconds":elapsed_decode,"prefill_seconds":elapsed_prefill,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "precision":"BF16 backbone/cache/input projections; FP32 probability banks, expanded binary FFN readout and PV; exact prefill attention",
        "limitations":"Fixed WT2 suffixes, not full WT2 PPL; teacher forcing; no retraining; original attention during prefill, chosen FFN active during both phases; dense count @ V reference, not hardware speedup",
        "rng":"FFN global CUDA generator reset after loading; attention private generator with same seed integer, independent state",
        "ffn_readout":"expanded 0/1 single-bit and AND features; chunked only over input token axis"}
    outpath.parent.mkdir(parents=True,exist_ok=True)
    outpath.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({"complete":str(outpath),"ppl":result['perplexity'],"decode_seconds":elapsed_decode}),flush=True)


if __name__=="__main__":
    main()
