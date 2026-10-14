"""Reproducible original LLaMA-1 7B baseline on the existing WT2 protocol."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


EXPECTED = {
    "model_type": "llama",
    "hidden_size": 4096,
    "intermediate_size": 11008,
    "num_hidden_layers": 32,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "max_position_embeddings": 2048,
    "vocab_size": 32000,
}

VERIFIED_WEIGHT_SHA256 = {
    "model-00001-of-00002.safetensors": (
        "d43476fdd2fca0c44d55ee930039dd5dafb6331764dc0b5e5f89c60b551fcc12"
    ),
    "model-00002-of-00002.safetensors": (
        "94712c4b457815e0f8aaa4f78170aebba2b9ee3339ff2b40fceb983338fbafee"
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def validate_config(model_path: Path) -> tuple[object, dict[str, object]]:
    config = AutoConfig.from_pretrained(model_path, local_files_only=True)
    observed = {}
    for key, expected in EXPECTED.items():
        value = getattr(config, key, None)
        if key == "num_key_value_heads" and value is None:
            value = getattr(config, "num_attention_heads")
        observed[key] = value
        if value != expected:
            raise RuntimeError(
                f"Not original LLaMA-1 7B: {key}={value!r}, expected {expected!r}"
            )
    if bool(getattr(config, "tie_word_embeddings", False)):
        raise RuntimeError("Expected untied input/output embeddings for original LLaMA-1 7B")
    return config, observed


def model_file_manifest(model_path: Path) -> list[dict[str, object]]:
    names = {
        "config.json",
        "generation_config.json",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer.model",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    }
    rows = []
    for path in sorted(p for p in model_path.iterdir() if p.is_file()):
        row: dict[str, object] = {"name": path.name, "bytes": path.stat().st_size}
        if path.name in names:
            row["sha256"] = sha256(path)
        rows.append(row)
    return rows


def cached_decode_probe(model, ids: torch.Tensor) -> dict[str, float]:
    probe = ids[:, :128].cuda()
    full = model(probe, use_cache=False).logits[:, -1].float()
    prefix = model(probe[:, :-1], use_cache=True)
    cached = model(
        probe[:, -1:], past_key_values=prefix.past_key_values, use_cache=True
    ).logits[:, -1].float()
    delta = (full - cached).abs()
    return {
        "mean_abs_logit_difference": delta.mean().item(),
        "max_abs_logit_difference": delta.max().item(),
    }


def evaluate_wikitext2(model, tokenizer, data_path: Path, context: int, stride: int):
    text = data_path.read_text(encoding="utf-8")
    ids = tokenizer(
        text, add_special_tokens=False, return_tensors="pt"
    ).input_ids
    if ids.shape[1] < 2:
        raise RuntimeError("Corpus tokenization produced fewer than two tokens")

    total_nll = 0.0
    scored_tokens = 0
    previous_end = 1
    records = []
    started = time.perf_counter()

    for begin in range(0, ids.shape[1], stride):
        end = min(begin + context, ids.shape[1])
        input_ids = ids[:, begin:end].cuda(non_blocking=True)
        hidden = model.model(
            input_ids=input_ids, use_cache=False, return_dict=True
        ).last_hidden_state

        first = max(previous_end, begin + 1)
        window_nll = 0.0
        # Chunk the LM head to avoid materializing all [context, vocab] logits.
        for target in range(first, end, 128):
            stop = min(target + 128, end)
            logits = model.lm_head(
                hidden[:, target - begin - 1 : stop - begin - 1]
            ).float()
            if not bool(torch.isfinite(logits).all()):
                raise RuntimeError(f"Non-finite logits in window beginning at {begin}")
            window_nll += F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                input_ids[:, target - begin : stop - begin].reshape(-1),
                reduction="sum",
            ).item()

        newly_scored = end - first
        total_nll += window_nll
        scored_tokens += newly_scored
        previous_end = end
        record = {
            "begin": begin,
            "end": end,
            "scored": newly_scored,
            "nll_sum": window_nll,
        }
        records.append(record)
        print(
            json.dumps(
                {
                    "window": len(records),
                    "scored_tokens": scored_tokens,
                    "ppl": math.exp(total_nll / scored_tokens),
                    "seconds": time.perf_counter() - started,
                }
            ),
            flush=True,
        )
        if end == ids.shape[1]:
            break

    if scored_tokens != ids.shape[1] - 1:
        raise RuntimeError(
            f"Scored {scored_tokens} tokens, expected {ids.shape[1] - 1}"
        )
    return {
        "perplexity": math.exp(total_nll / scored_tokens),
        "nll_sum": total_nll,
        "scored_tokens": scored_tokens,
        "corpus_tokens": ids.shape[1],
        "corpus_sha256": sha256(data_path),
        "windows": records,
    }, ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--attn-implementation", default="sdpa", choices=["sdpa", "eager"])
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    args.model = args.model.resolve()
    args.data = args.data.resolve()
    args.output = args.output.resolve()
    if not args.model.is_dir():
        raise FileNotFoundError(args.model)
    if not args.data.is_file():
        raise FileNotFoundError(args.data)

    config, architecture = validate_config(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    if len(tokenizer) != EXPECTED["vocab_size"]:
        raise RuntimeError(f"Tokenizer size is {len(tokenizer)}, expected 32000")

    metadata: dict[str, object] = {
        "status": "configuration_checked" if args.check_only else "running",
        "protocol": {
            "corpus": "WikiText-2 test raw",
            "context": args.context,
            "stride": args.stride,
            "batch_size": 1,
            "add_special_tokens": False,
            "precision": "bfloat16 weights/activations; float32 logits and CE",
            "use_cache_for_ppl": False,
            "attention_implementation": args.attn_implementation,
        },
        "architecture": architecture,
        "config_class": type(config).__name__,
        "model_path": str(args.model),
        "model_files": model_file_manifest(args.model),
        "verified_weight_sha256": VERIFIED_WEIGHT_SHA256,
        "data_path": str(args.data),
        "host": platform.node(),
        "python": platform.python_version(),
        "versions": {
            name: package_version(name)
            for name in ("torch", "transformers", "tokenizers", "sentencepiece")
        },
        "source_sha256": sha256(Path(__file__).resolve()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    if args.check_only:
        print(json.dumps(metadata, indent=2, ensure_ascii=False))
        return

    torch.set_num_threads(4)
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the baseline run")

    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    ).cuda().eval()
    model.requires_grad_(False)
    metadata.update(
        {
            "load_seconds": time.perf_counter() - load_started,
            "parameter_count": sum(p.numel() for p in model.parameters()),
            "cuda_device": torch.cuda.get_device_name(),
            "cuda_capability": torch.cuda.get_device_capability(),
        }
    )

    with torch.inference_mode():
        result, corpus_ids = evaluate_wikitext2(
            model, tokenizer, args.data, args.context, args.stride
        )
        cache_probe = cached_decode_probe(model, corpus_ids)

    metadata.update(
        {
            "status": "complete",
            "peak_cuda_bytes": torch.cuda.max_memory_allocated(),
            "cache_probe": cache_probe,
        }
    )
    payload = {"metadata": metadata, "result": result}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(f"COMPLETE {args.output}", flush=True)


if __name__ == "__main__":
    main()
