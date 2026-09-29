from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from full_path_pdnn_ffn import load_full_path_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a full-path bipolar P-DNN FFN replacement")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--start-token", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride <= 0 or args.stride > args.max_length:
        raise ValueError("stride must be in (0, max_length]")
    if args.start_token < 0:
        raise ValueError("start-token must be non-negative")
    if args.max_tokens is not None and args.max_tokens <= 0:
        raise ValueError("max-tokens must be positive")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    student, payload = load_full_path_checkpoint(args.checkpoint)
    student.to(dtype=torch.bfloat16).eval()
    student.set_sample_count(args.sample_count)
    layer = int(payload["training_args"]["layer"])
    model.model.layers[layer].mlp = student

    token_ids = tokenizer(
        Path(args.text_file).read_text(encoding="utf-8"),
        return_tensors="pt",
        add_special_tokens=False,
    ).input_ids
    source_tokens = token_ids.shape[-1]
    stop_token = None if args.max_tokens is None else args.start_token + args.max_tokens
    token_ids = token_ids[:, args.start_token : stop_token]
    sequence_length = token_ids.shape[-1]
    if sequence_length < 2:
        raise ValueError("The selected corpus segment must contain at least two tokens")
    total_nll = 0.0
    total_scored_tokens = 0
    previous_end = 0
    windows = 0
    torch.cuda.reset_peak_memory_stats()
    for begin in range(0, sequence_length, args.stride):
        end = min(begin + args.max_length, sequence_length)
        window = token_ids[:, begin:end].cuda()
        with torch.inference_mode():
            logits = model(input_ids=window, use_cache=False).logits[:, :-1].float()
        labels = window[:, 1:]
        positions = torch.arange(begin + 1, end, device="cuda")
        first_scored = max(previous_end, begin + 1)
        mask = positions >= first_scored
        if mask.any():
            selected_logits = logits[:, mask, :].reshape(-1, logits.shape[-1])
            selected_labels = labels[:, mask].reshape(-1)
            total_nll += F.cross_entropy(selected_logits, selected_labels, reduction="sum").item()
            total_scored_tokens += selected_labels.numel()
        previous_end = end
        windows += 1
        print(f"sample_count={args.sample_count} window={windows} end={end}/{sequence_length}", flush=True)
        if end == sequence_length:
            break

    mean_nll = total_nll / total_scored_tokens
    result = {
        "test": "full_path_pdnn_ffn_sliding_window_perplexity",
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "student_type": payload.get("student_type"),
        "layer": layer,
        "sample_count": args.sample_count,
        "seed": args.seed,
        "source_tokens": source_tokens,
        "start_token": args.start_token,
        "end_token": args.start_token + sequence_length,
        "corpus_tokens": sequence_length,
        "scored_tokens": total_scored_tokens,
        "max_length": args.max_length,
        "stride": args.stride,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": math.exp(mean_nll),
        "windows": windows,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
