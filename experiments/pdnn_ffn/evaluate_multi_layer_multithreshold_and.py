"""Multi-layer AND-bank evaluation with actual expanded binary readout."""
import hashlib
import json
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from transformers import AutoModelForCausalLM,AutoTokenizer
from evaluate_multi_layer_full_path_perplexity import parse_args
from evaluate_multithreshold_and import prepare_student


def main():
    args=parse_args()
    if args.stride<=0 or args.stride>args.max_length or args.start_token<0 or args.sample_count<0:
        raise ValueError("invalid evaluation arguments")
    if args.max_tokens is not None and args.max_tokens<=0:
        raise ValueError("max_tokens must be positive")
    torch.manual_seed(args.seed)
    tokenizer=AutoTokenizer.from_pretrained(args.model,local_files_only=True)
    model=AutoModelForCausalLM.from_pretrained(args.model,local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda().eval()
    records={}
    for path in args.checkpoints:
        student,payload=prepare_student(path)
        layer=int(payload["training_args"]["layer"])
        if layer in records or layer<0 or layer>=len(model.model.layers):
            raise ValueError(f"duplicate or invalid layer: {layer}")
        student.set_sample_count(args.sample_count)
        model.model.layers[layer].mlp=student
        records[layer]={"layer":layer,"path":str(Path(path).resolve()),
            "sha256":hashlib.sha256(Path(path).read_bytes()).hexdigest(),"bits":student.config.bits,
            "phase":payload["phase"],"step":payload["step"],
            "parameters":sum(p.numel() for p in student.parameters())}
    ids=tokenizer(Path(args.text_file).read_text(),add_special_tokens=False,return_tensors="pt").input_ids
    source_tokens=ids.shape[-1]
    end=None if args.max_tokens is None else args.start_token+args.max_tokens
    ids=ids[:,args.start_token:end]
    length=ids.shape[-1]
    if length<2:
        raise ValueError("too few corpus tokens")
    nll=0.
    scored=previous_end=windows=0
    start=time.perf_counter()
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
            print(f"N={args.sample_count} windows={windows} tokens={end}/{length}",flush=True)
        if end==length:
            break
    result={"test":"multi_layer_multithreshold_and_full_test_perplexity","layers":sorted(records),
        "checkpoints":[records[k] for k in sorted(records)],"replacement_count":len(records),
        "student_parameters_total":sum(r["parameters"] for r in records.values()),
        "sample_count":args.sample_count,"seed":args.seed,"source_tokens":source_tokens,
        "start_token":args.start_token,"end_token":args.start_token+length,"corpus_tokens":length,
        "scored_tokens":scored,"max_length":args.max_length,"stride":args.stride,"windows":windows,
        "mean_negative_log_likelihood":nll/scored,"perplexity":math.exp(nll/scored),
        "evaluation_elapsed_seconds":time.perf_counter()-start,
        "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
        "readout":"expanded single-bit and AND terms; shared FP32 columns, FP32 path accumulation",
        "precision":"BF16 backbone/input projections/output; FP32 probability banks and readout"}
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result),flush=True)


if __name__=="__main__":
    main()
