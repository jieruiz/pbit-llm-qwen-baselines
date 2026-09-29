from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from full_path_pdnn_ffn import FullPathPBitFFN, load_full_path_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure p-bit probability saturation")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--sample-paths", type=int, default=4)
    parser.add_argument("--validation-tokens", type=int, default=65536)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


class CaptureInput:
    def __init__(self) -> None:
        self.value: torch.Tensor | None = None

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.value = args[0].detach()


class ProbabilityStats:
    def __init__(self) -> None:
        self.count = 0
        self.total = 0.0
        self.entropy_total = 0.0
        self.low_count = 0
        self.high_count = 0

    def update(self, probability: torch.Tensor) -> None:
        value = probability.detach().float()
        clipped = value.clamp(1e-7, 1.0 - 1e-7)
        self.count += value.numel()
        self.total += float(value.sum())
        self.entropy_total += float(
            (-(clipped * clipped.log() + (1.0 - clipped) * (1.0 - clipped).log())).sum()
        )
        self.low_count += int((value < 0.01).sum())
        self.high_count += int((value > 0.99).sum())

    def result(self) -> dict[str, float | int]:
        return {
            "count": self.count,
            "mean_probability": self.total / self.count,
            "mean_entropy_nats": self.entropy_total / self.count,
            "below_0p01_fraction": self.low_count / self.count,
            "above_0p99_fraction": self.high_count / self.count,
            "saturated_fraction": (self.low_count + self.high_count) / self.count,
        }


def as_probability(student: FullPathPBitFFN, mean: torch.Tensor) -> torch.Tensor:
    if student.config.coding == "binary":
        return mean
    return (mean + 1.0) * 0.5


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if min(args.batches, args.batch_size, args.sequence_length, args.sample_paths) <= 0:
        raise ValueError("batch, sequence, and sample arguments must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    student, payload = load_full_path_checkpoint(args.checkpoint)
    student.eval()
    layer = int(payload["training_args"]["layer"])

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    tokens = tokenizer(
        Path(args.text_file).read_text(encoding="utf-8"),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
    tokens = tokens[-min(args.validation_tokens, tokens.numel()) :]
    maximum = tokens.numel() - args.sequence_length
    if maximum <= 0:
        raise ValueError("text is shorter than sequence length")

    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    teacher.requires_grad_(False)
    capture = CaptureInput()
    hook = teacher.model.layers[layer].mlp.register_forward_hook(capture)
    generator = torch.Generator().manual_seed(args.seed + 1907)
    input_stats = ProbabilityStats()
    hidden_stats = [ProbabilityStats() for _ in student.config.hidden_sizes]

    with torch.no_grad():
        for _ in range(args.batches):
            starts = torch.randint(0, maximum, (args.batch_size,), generator=generator)
            ids = torch.stack([tokens[start : start + args.sequence_length] for start in starts.tolist()]).cuda()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher.model(input_ids=ids, use_cache=False, return_dict=True)
            if capture.value is None:
                raise RuntimeError("failed to capture FFN input")
            input_mean = student.input_mean(capture.value)
            input_stats.update(as_probability(student, input_mean))
            for _ in range(args.sample_paths):
                state = student._sample_state(input_mean)
                for index, projection in enumerate(student.projections[:-1]):
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        hidden_mean = student.hidden_mean(projection(state), index)
                    hidden_stats[index].update(as_probability(student, hidden_mean))
                    state = student._sample_state(hidden_mean)

    hook.remove()
    result = {
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "layer": layer,
        "coding": student.config.coding,
        "learnable_encoding": student.config.learnable_encoding,
        "input_threshold": float(student.effective_input_threshold().detach().cpu()),
        "input_temperature": float(student.effective_input_temperature().detach().cpu()),
        "hidden_thresholds": [
            float(student.effective_hidden_threshold(index).detach().cpu())
            for index in range(len(student.config.hidden_sizes))
        ],
        "hidden_temperatures": [
            float(student.effective_hidden_temperature(index).detach().cpu())
            for index in range(len(student.config.hidden_sizes))
        ],
        "sample_paths": args.sample_paths,
        "input_probability": input_stats.result(),
        "hidden_probabilities": [stats.result() for stats in hidden_stats],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
