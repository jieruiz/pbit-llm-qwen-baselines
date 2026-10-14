"""Full WT2 evaluation for LLaMA-7B fixed stochastic FFN and sampled P@V."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoTokenizer

from model import ROOT, load, protected_hash


VARIANTS = {
    "original_dense": ("original", "dense"),
    "attention_pv": ("original", "iid"),
    "ffn_dense": ("fixed", "dense"),
    "joint_pv": ("fixed", "iid"),
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def evaluate(model, tokenizer, data_path, context, stride, smoke):
    ids = tokenizer(
        data_path.read_text(encoding="utf-8"),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids
    if smoke:
        ids = ids[:, :128]
    total = 0.0
    count = 0
    previous_end = 1
    records = []
    started = time.perf_counter()
    for begin in range(0, ids.shape[1], stride):
        end = min(begin + context, ids.shape[1])
        inputs = ids[:, begin:end].cuda()
        hidden = model.model(
            input_ids=inputs, use_cache=False, return_dict=True
        ).last_hidden_state
        first = max(previous_end, begin + 1)
        window_nll = 0.0
        for target in range(first, end, 128):
            stop = min(target + 128, end)
            logits = model.lm_head(
                hidden[:, target - begin - 1 : stop - begin - 1]
            ).float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"Non-finite logits at window {begin}")
            window_nll += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                inputs[:, target - begin : stop - begin].reshape(-1),
                reduction="sum",
            ).item()
        newly_scored = end - first
        total += window_nll
        count += newly_scored
        previous_end = end
        records.append(
            {"begin": begin, "end": end, "scored": newly_scored, "nll_sum": window_nll}
        )
        print(
            json.dumps(
                {
                    "window": len(records),
                    "scored_tokens": count,
                    "ppl": math.exp(total / count),
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        if end == ids.shape[1]:
            break
    if count != ids.shape[1] - 1:
        raise RuntimeError(f"Scored {count}, expected {ids.shape[1] - 1}")
    return {
        "perplexity": math.exp(total / count),
        "nll_sum": total,
        "scored_tokens": count,
        "corpus_tokens": int(ids.shape[1]),
        "corpus_sha256": sha256(data_path),
        "seconds": time.perf_counter() - started,
        "windows": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--chunk", type=int, default=32)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    load_started = time.perf_counter()
    model, ffns, controls, before, original_parameter_count = load(
        args.model, *VARIANTS[args.variant], args.seed, args.samples, args.chunk
    )
    load_seconds = time.perf_counter() - load_started

    with torch.inference_mode():
        result = evaluate(
            model, tokenizer, args.data, args.context, args.stride, args.smoke
        )
    after = protected_hash(model)
    if before != after:
        raise RuntimeError("Protected model parameters changed during evaluation")
    if not all(control.calls > 0 for control in controls):
        raise RuntimeError("At least one attention controller was not exercised")
    if args.variant.startswith("ffn") or args.variant.startswith("joint"):
        if not all(ffn.forward_calls > 0 for ffn in ffns.values()):
            raise RuntimeError("At least one FFN replacement was not exercised")

    sources = [
        "evaluate.py",
        "model.py",
        "attention.py",
        "fixed_ffn.py",
        "streaming_ffn.py",
        "calibration_ffn.json",
    ]
    payload = {
        "metadata": {
            "status": "complete",
            "variant": args.variant,
            "variant_config": VARIANTS[args.variant],
            "seed": args.seed,
            "samples": args.samples,
            "attention_chunk": args.chunk,
            "smoke": args.smoke,
            "context": args.context,
            "stride": args.stride,
            "batch_size": 1,
            "precision": "BF16 interface; FP32 stochastic reference and CE",
            "ffn_config": "all32 fixed group_max, W8, magnitude12+sign, input4096/output3072, interval256, burn25%, ideal sigmoid",
            "attention_config": "exact Q/K/softmax; dense P@V or 512-sample IID P@V; all32 layers",
            "model_path": str(args.model),
            "data_path": str(args.data),
            "load_seconds": load_seconds,
            "original_parameter_count": original_parameter_count,
            "live_parameter_count": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "cuda_device": torch.cuda.get_device_name(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "versions": {
                name: importlib.metadata.version(name)
                for name in ("torch", "transformers", "tokenizers", "safetensors")
            },
            "source_sha256": {name: sha256(ROOT / name) for name in sources},
            "protected_sha256": before,
            "ffn_layers": len(ffns),
            "ffn_diagnostics": {
                str(index): module.diagnostics() for index, module in ffns.items()
            },
            "attention_diagnostics": [control.summary() for control in controls],
        },
        "result": result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(f"COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
