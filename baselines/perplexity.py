import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from common import device_metadata, load_local_model, save_json, set_deterministic


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--stride", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.stride <= 0 or args.stride > args.max_length:
        raise ValueError("stride must be in (0, max_length]")
    set_deterministic(args.seed)
    tokenizer, model = load_local_model(args.model, "cuda")
    text = Path(args.text_file).read_text(encoding="utf-8")
    token_ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids
    if args.max_tokens is not None:
        token_ids = token_ids[:, : args.max_tokens]
    sequence_length = token_ids.shape[-1]
    if sequence_length < 2:
        raise ValueError("The corpus must contain at least two tokens")

    total_nll = 0.0
    total_scored_tokens = 0
    previous_end = 0
    windows = 0
    for begin in range(0, sequence_length, args.stride):
        end = min(begin + args.max_length, sequence_length)
        window = token_ids[:, begin:end].to("cuda")
        with torch.inference_mode():
            logits = model(input_ids=window, use_cache=False).logits[:, :-1].float()
        labels = window[:, 1:]
        target_global_positions = torch.arange(begin + 1, end, device="cuda")
        first_scored_position = max(previous_end, begin + 1)
        score_mask = target_global_positions >= first_scored_position
        if score_mask.any():
            selected_logits = logits[:, score_mask, :].reshape(-1, logits.shape[-1])
            selected_labels = labels[:, score_mask].reshape(-1)
            nll = F.cross_entropy(selected_logits, selected_labels, reduction="sum")
            total_nll += nll.item()
            total_scored_tokens += selected_labels.numel()
        windows += 1
        previous_end = end
        print(f"window={windows} end={end}/{sequence_length}")
        if end == sequence_length:
            break

    mean_nll = total_nll / total_scored_tokens
    result = {
        "test": "sliding_window_perplexity",
        "model_path": args.model,
        "text_file": str(Path(args.text_file).resolve()),
        "corpus_tokens": sequence_length,
        "scored_tokens": total_scored_tokens,
        "max_length": args.max_length,
        "stride": args.stride,
        "mean_negative_log_likelihood": mean_nll,
        "perplexity": math.exp(mean_nll),
        "windows": windows,
        "environment": device_metadata("cuda"),
    }
    save_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()

