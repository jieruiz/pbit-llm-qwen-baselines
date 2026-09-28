import argparse
import statistics
import time

import torch

from common import device_metadata, load_local_model, save_json, set_deterministic


def timed_forward(model, input_ids: torch.Tensor) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        model(input_ids=input_ids, use_cache=True)
    torch.cuda.synchronize()
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--lengths", type=int, nargs="+", default=[128, 512, 2048])
    parser.add_argument("--new-tokens", type=int, default=64)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires CUDA")
    set_deterministic(args.seed)
    tokenizer, model = load_local_model(args.model, "cuda")

    prefill = []
    for length in args.lengths:
        input_ids = torch.randint(
            low=0,
            high=model.config.vocab_size,
            size=(1, length),
            device="cuda",
        )
        timed_forward(model, input_ids)
        torch.cuda.reset_peak_memory_stats()
        samples = [timed_forward(model, input_ids) for _ in range(args.repeats)]
        prefill.append(
            {
                "sequence_length": length,
                "median_seconds": statistics.median(samples),
                "tokens_per_second": length / statistics.median(samples),
                "peak_memory_bytes": torch.cuda.max_memory_allocated(),
                "samples_seconds": samples,
            }
        )

    prompt = "Probabilistic computing can be useful because"
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=4,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=args.new_tokens,
            pad_token_id=tokenizer.eos_token_id,
        )
    torch.cuda.synchronize()
    generation_seconds = time.perf_counter() - start
    actual_new_tokens = output_ids.shape[-1] - inputs.input_ids.shape[-1]

    result = {
        "test": "inference_performance",
        "model_path": args.model,
        "seed": args.seed,
        "repeats": args.repeats,
        "prefill": prefill,
        "generation": {
            "requested_new_tokens": args.new_tokens,
            "actual_new_tokens": actual_new_tokens,
            "seconds": generation_seconds,
            "tokens_per_second": actual_new_tokens / generation_seconds,
            "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        },
        "environment": device_metadata("cuda"),
    }
    save_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()

