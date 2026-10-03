"""Estimate stochastic bias/variance on fixed teacher inputs, without mean-field substitution."""
import argparse
import json
from pathlib import Path

import torch

from full_path_pdnn_ffn import load_full_path_checkpoint


def path_moments(paths, target):
    """Unbiased variance and noise-corrected squared-bias estimates over iid paths."""
    count = paths.shape[0]
    if count < 2:
        raise ValueError("at least two paths are required")
    values = paths.float()
    variance, mean = torch.var_mean(values, dim=0, unbiased=True)
    variance_mse = variance.mean()
    finite_mean_mse = (mean - target.float()).square().mean()
    bias_mse = finite_mean_mse - variance_mse / count
    result = {
        "target_power": target.float().square().mean().item(),
        "single_path_variance_mse": variance_mse.item(),
        "finite_sample_mean_mse": finite_mean_mse.item(),
        "corrected_bias_mse": bias_mse.item(),
        "empirical_single_path_mse": (values - target.float()).square().mean().item(),
    }
    for sample_count in (4, 16):
        result[f"predicted_n{sample_count}_mse"] = (bias_mse + variance_mse / sample_count).item()
        if count % sample_count == 0:
            # Reproduce forward's native-dtype addition, including BF16 rounding.
            groups = paths.reshape(count // sample_count, sample_count, *paths.shape[1:])
            output_sum = groups[:, 0].clone()
            for index in range(1, sample_count):
                output_sum = output_sum + groups[:, index]
            actual = output_sum / float(sample_count)
            result[f"empirical_n{sample_count}_mse"] = (actual.float() - target.float()).square().mean().item()
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--paths", type=int, default=256)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=27183)
    args = parser.parse_args()
    if args.paths < 16 or args.paths % 16 or args.batches <= 0:
        raise ValueError("use a positive number of batches and a path count divisible by 16")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from train_full_path_distillation import FFNCapture, teacher_examples, random_batch

    student, payload = load_full_path_checkpoint(args.checkpoint)
    student.to(dtype=torch.bfloat16).eval()
    teacher = AutoModelForCausalLM.from_pretrained(
        args.model, local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    ).cuda().eval()
    teacher.requires_grad_(False)
    capture = FFNCapture()
    layer = int(payload["training_args"]["layer"])
    hook = teacher.model.layers[layer].mlp.register_forward_hook(capture)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    tokens = tokenizer(Path(args.train_text).read_text(), add_special_tokens=False, return_tensors="pt").input_ids[0]
    tokens = tokens[-int(payload["training_args"]["validation_tokens"]):]
    generator = torch.Generator().manual_seed(args.seed)
    records = []
    with torch.inference_mode():
        for batch in range(args.batches):
            ids = random_batch(tokens, 1, args.sequence_length, generator).cuda()
            inputs, target = teacher_examples(teacher, capture, ids)
            torch.manual_seed(args.seed + 1000 + batch)
            paths = torch.stack([student.sampled_path_forward(inputs) for _ in range(args.paths)])
            record = path_moments(paths, target)
            record["mean_field_mse"] = (student.mean_field_forward(inputs).float() - target.float()).square().mean().item()
            records.append(record)
            del paths
    hook.remove()
    means = {key: sum(record[key] for record in records) / len(records) for key in records[0]}
    normalized = {key.replace("_mse", "_nmse"): value / means["target_power"] for key, value in means.items() if key.endswith("_mse")}
    result = {
        "checkpoint": str(Path(args.checkpoint)), "layer": layer,
        "architecture": student.config.architecture, "args": vars(args),
        "input_distribution": "fixed held-out original-teacher inputs, common across all checkpoints",
        "means": means, "normalized": normalized, "batch_records": records,
        "notes": [
            "The exact expected output is estimated by repeated full-path sampling, not by mean-field.",
            "Corrected squared bias subtracts variance/R; it is an unbiased finite-sample estimate and may be negative.",
            "Predicted MSE uses real-valued averaging; empirical N=4/16 additionally includes native BF16 summation rounding.",
            "256 paths per input, one fixed probe seed, 1024 input tokens by default; no confidence interval claimed.",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(normalized), flush=True)


if __name__ == "__main__":
    main()
