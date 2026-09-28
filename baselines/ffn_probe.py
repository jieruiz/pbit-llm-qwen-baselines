import argparse

import torch

from common import device_metadata, load_local_model, save_json, set_deterministic


def tensor_stats(value: torch.Tensor) -> dict:
    value = value.detach().float()
    quantiles = torch.quantile(
        value.reshape(-1),
        torch.tensor([0.01, 0.1, 0.5, 0.9, 0.99], device=value.device),
    )
    return {
        "shape": list(value.shape),
        "mean": value.mean().item(),
        "std": value.std().item(),
        "rms": value.square().mean().sqrt().item(),
        "min": value.min().item(),
        "max": value.max().item(),
        "fraction_positive": (value > 0).float().mean().item(),
        "fraction_abs_lt_1e-3": (value.abs() < 1e-3).float().mean().item(),
        "quantiles": {
            name: number.item()
            for name, number in zip(["p01", "p10", "p50", "p90", "p99"], quantiles)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--layer", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--prompt",
        default="概率计算使用随机变量表达和处理信息。Probabilistic computing represents information with random variables.",
    )
    args = parser.parse_args()

    set_deterministic(args.seed)
    tokenizer, model = load_local_model(args.model, "cuda")
    if not 0 <= args.layer < len(model.model.layers):
        raise ValueError(f"layer must be in [0, {len(model.model.layers) - 1}]")
    mlp = model.model.layers[args.layer].mlp
    captured = {}

    def capture(name):
        def hook(_module, _inputs, output):
            captured[name] = output.detach()
        return hook

    handles = [
        mlp.gate_proj.register_forward_hook(capture("gate_projection")),
        mlp.up_proj.register_forward_hook(capture("up_projection")),
        mlp.down_proj.register_forward_hook(capture("down_projection")),
    ]
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        model(**inputs, use_cache=False)
    for handle in handles:
        handle.remove()

    gate = captured["gate_projection"]
    up = captured["up_projection"]
    silu_gate = torch.nn.functional.silu(gate)
    gated_product = silu_gate * up
    result = {
        "test": "ffn_activation_statistics",
        "model_path": args.model,
        "layer": args.layer,
        "prompt": args.prompt,
        "tokens": inputs.input_ids.shape[-1],
        "gate_projection": tensor_stats(gate),
        "silu_gate": tensor_stats(silu_gate),
        "up_projection": tensor_stats(up),
        "gated_product": tensor_stats(gated_product),
        "down_projection": tensor_stats(captured["down_projection"]),
        "environment": device_metadata("cuda"),
    }
    save_json(result, args.output)
    print(result)


if __name__ == "__main__":
    main()

