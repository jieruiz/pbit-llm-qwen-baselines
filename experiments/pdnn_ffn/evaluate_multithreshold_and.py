"""Full-corpus sliding-window PPL; every sampled readout input is binary."""
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from evaluate_full_path_perplexity import parse_args
from multithreshold_and_ffn import load_checkpoint


def prepare_student(path):
    student,payload=load_checkpoint(path)
    student.gate_projection.to(dtype=torch.bfloat16)
    student.value_projection.to(dtype=torch.bfloat16)
    # Keep probability banks and the tied expanded readout in FP32. This avoids
    # excessive cancellation roundoff between signed terms. No hardware speed claim.
    student.eval()
    student.binary_readout=True
    return student,payload


def main():
    args=parse_args()
    if args.stride<=0 or args.stride>args.max_length or args.start_token<0:
        raise ValueError("invalid evaluation window")
    torch.manual_seed(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    student,payload=prepare_student(args.checkpoint)
    student.set_sample_count(args.sample_count)
    layer=int(payload["training_args"]["layer"])
    model.model.layers[layer].mlp=student
    ids=tokenizer(Path(args.text_file).read_text(),add_special_tokens=False,return_tensors="pt").input_ids
    source_tokens=ids.shape[-1]
    end=None if args.max_tokens is None else args.start_token+args.max_tokens
    ids=ids[:,args.start_token:end]
    length=ids.shape[-1]
    if length<2:
        raise ValueError("too few tokens")
    nll=0.
    scored=previous_end=windows=0
    begin_time=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for begin in range(0,length,args.stride):
        end=min(begin+args.max_length,length)
        window=ids[:,begin:end].cuda()
        with torch.inference_mode():
            logits=model(input_ids=window,use_cache=False).logits[:,:-1].float()
        labels=window[:,1:]
        mask=torch.arange(begin+1,end,device="cuda")>=max(previous_end,begin+1)
        if mask.any():
            selected=logits[:,mask,:].reshape(-1,logits.shape[-1])
            targets=labels[:,mask].reshape(-1)
            nll+=F.cross_entropy(selected,targets,reduction="sum").item()
            scored+=targets.numel()
        previous_end=end
        windows+=1
        if windows%20==0:
            print(f"windows={windows} tokens={end}/{length}",flush=True)
        if end==length:
            break
    result={"test":"multithreshold_and_full_test_perplexity", "checkpoint":str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256":hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        "student_type":payload["student_type"],"layer":layer,"bits":student.config.bits,
        "phase":payload["phase"],"step":payload["step"],"sample_count":args.sample_count,"seed":args.seed,
        "source_tokens":source_tokens,"start_token":args.start_token,"end_token":args.start_token+length,
        "corpus_tokens":length,"scored_tokens":scored,"max_length":args.max_length,"stride":args.stride,
        "mean_negative_log_likelihood":nll/scored,"perplexity":math.exp(nll/scored),"windows":windows,
        "elapsed_seconds":time.perf_counter()-begin_time,"peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "readout":"expanded 0/1 single and AND terms; tied FP32 weights; FP32 path accumulation",
        "precision":"BF16 backbone and input projections; FP32 probability banks and readout; BF16 FFN output"}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
