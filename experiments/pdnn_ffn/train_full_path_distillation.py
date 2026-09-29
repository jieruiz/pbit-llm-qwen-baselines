from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from full_path_pdnn_ffn import FullPathPBitFFN, FullPathPBitFFNConfig


def parse_hidden_sizes(value: str) -> tuple[int, ...]:
    sizes = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if not sizes or any(size <= 0 for size in sizes):
        raise argparse.ArgumentTypeError("hidden sizes must be comma-separated positive integers")
    return sizes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Distill one Qwen FFN into a full-path P-DNN")
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--layer", type=int, default=12)
    parser.add_argument("--hidden-sizes", type=parse_hidden_sizes, default=(4864,))
    parser.add_argument("--coding", choices=("bipolar", "binary"), default="bipolar")
    parser.add_argument("--input-temperature", type=float, default=1.0)
    parser.add_argument("--hidden-temperature", type=float, default=1.0)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--validation-tokens", type=int, default=65536)
    parser.add_argument("--mean-steps", type=int, default=2000)
    parser.add_argument("--sample-steps", type=int, default=2000)
    parser.add_argument("--train-samples", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--sample-learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--cosine-loss-weight", type=float, default=0.05)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--validate-every", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_batch(tokens: torch.Tensor, batch_size: int, sequence_length: int, generator: torch.Generator) -> torch.Tensor:
    maximum = tokens.numel() - sequence_length
    if maximum <= 0:
        raise ValueError("corpus split is shorter than sequence_length")
    starts = torch.randint(0, maximum, (batch_size,), generator=generator)
    return torch.stack([tokens[start : start + sequence_length] for start in starts.tolist()])


class FFNCapture:
    def __init__(self) -> None:
        self.inputs: torch.Tensor | None = None
        self.outputs: torch.Tensor | None = None

    def __call__(self, module: nn.Module, args: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
        self.inputs = args[0].detach()
        self.outputs = output.detach()


@torch.no_grad()
def teacher_examples(
    teacher: AutoModelForCausalLM,
    capture: FFNCapture,
    input_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    capture.inputs = None
    capture.outputs = None
    with torch.autocast("cuda", dtype=torch.bfloat16):
        teacher.model(input_ids=input_ids, use_cache=False, return_dict=True)
    if capture.inputs is None or capture.outputs is None:
        raise RuntimeError("FFN hook did not capture tensors")
    return capture.inputs, capture.outputs


def metrics(prediction: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    pred = prediction.float()
    tgt = target.float()
    mse = F.mse_loss(pred, tgt)
    target_power = tgt.square().mean().clamp_min(1e-12)
    normalized_mse = mse / target_power
    cosine = F.cosine_similarity(
        pred.reshape(-1, pred.shape[-1]),
        tgt.reshape(-1, tgt.shape[-1]),
        dim=-1,
    ).mean()
    return normalized_mse, {
        "mse": float(mse.detach()),
        "normalized_mse": float(normalized_mse.detach()),
        "cosine_similarity": float(cosine.detach()),
        "target_rms": float(target_power.sqrt().detach()),
        "prediction_rms": float(pred.square().mean().sqrt().detach()),
    }


@torch.no_grad()
def validate(
    student: FullPathPBitFFN,
    teacher: AutoModelForCausalLM,
    capture: FFNCapture,
    validation_tokens: torch.Tensor,
    args: argparse.Namespace,
    sample_count: int,
) -> dict[str, float]:
    student.eval()
    student.set_sample_count(sample_count)
    generator = torch.Generator().manual_seed(args.seed + 100003 + sample_count)
    sums: dict[str, float] = {}
    for _ in range(args.validation_batches):
        ids = random_batch(validation_tokens, args.batch_size, args.sequence_length, generator).cuda(non_blocking=True)
        inputs, targets = teacher_examples(teacher, capture, ids)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction = student(inputs)
        _, batch_metrics = metrics(prediction, targets)
        for key, value in batch_metrics.items():
            sums[key] = sums.get(key, 0.0) + value
    student.train()
    return {key: value / args.validation_batches for key, value in sums.items()}


def save_checkpoint(
    output_dir: Path,
    name: str,
    student: FullPathPBitFFN,
    optimizer: torch.optim.Optimizer,
    args: argparse.Namespace,
    phase: str,
    step: int,
    validation: dict[str, float],
) -> None:
    training_args = vars(args).copy()
    training_args["hidden_sizes"] = list(args.hidden_sizes)
    payload = {
        "student_type": f"full_path_{args.coding}_pdnn_v1",
        "student_config": student.checkpoint_config(),
        "student_state_dict": {key: value.detach().cpu() for key, value in student.state_dict().items()},
        "optimizer_state_dict": optimizer.state_dict(),
        "training_args": training_args,
        "phase": phase,
        "step": step,
        "validation": validation,
    }
    torch.save(payload, output_dir / name)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_metrics.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    text = Path(args.train_text).read_text(encoding="utf-8")
    all_tokens = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids[0]
    if all_tokens.numel() <= args.validation_tokens + args.sequence_length:
        raise ValueError("corpus does not contain enough tokens for the requested validation split")
    training_tokens = all_tokens[: -args.validation_tokens].contiguous()
    validation_tokens = all_tokens[-args.validation_tokens :].contiguous()

    teacher = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    teacher.requires_grad_(False)
    if args.layer < 0 or args.layer >= len(teacher.model.layers):
        raise ValueError(f"layer must be in [0, {len(teacher.model.layers) - 1}]")
    teacher_mlp = teacher.model.layers[args.layer].mlp
    capture = FFNCapture()
    hook = teacher_mlp.register_forward_hook(capture)

    config = FullPathPBitFFNConfig(
        input_size=teacher.config.hidden_size,
        hidden_sizes=args.hidden_sizes,
        output_size=teacher.config.hidden_size,
        input_temperature=args.input_temperature,
        hidden_temperature=args.hidden_temperature,
        coding=args.coding,
    )
    student = FullPathPBitFFN(config).cuda().train()
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator().manual_seed(args.seed + 17)
    total_steps = args.mean_steps + args.sample_steps
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    with log_path.open("w", encoding="utf-8") as log_file:
        header = {
            "event": "start",
            "student_type": f"full_path_{args.coding}_pdnn_v1",
            "args": {**vars(args), "hidden_sizes": list(args.hidden_sizes)},
            "model_parameters": sum(parameter.numel() for parameter in teacher.parameters()),
            "student_parameters": sum(parameter.numel() for parameter in student.parameters()),
            "corpus_tokens": int(all_tokens.numel()),
            "training_tokens": int(training_tokens.numel()),
            "validation_tokens": int(validation_tokens.numel()),
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
        }
        log_file.write(json.dumps(header) + "\n")
        log_file.flush()
        print(json.dumps(header), flush=True)

        for global_step in range(1, total_steps + 1):
            if global_step <= args.mean_steps:
                phase = "mean_field"
                phase_step = global_step
                sample_count = 0
            else:
                phase = "full_path_sample_aware"
                phase_step = global_step - args.mean_steps
                sample_count = args.train_samples
                if phase_step == 1:
                    for parameter_group in optimizer.param_groups:
                        parameter_group["lr"] = args.sample_learning_rate
            student.set_sample_count(sample_count)
            ids = random_batch(training_tokens, args.batch_size, args.sequence_length, generator).cuda(non_blocking=True)
            inputs, targets = teacher_examples(teacher, capture, ids)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prediction = student(inputs)
            relative_mse, train_metrics = metrics(prediction, targets)
            cosine_loss = 1.0 - F.cosine_similarity(
                prediction.float().reshape(-1, prediction.shape[-1]),
                targets.float().reshape(-1, targets.shape[-1]),
                dim=-1,
            ).mean()
            loss = relative_mse + args.cosine_loss_weight * cosine_loss
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()

            if global_step == 1 or global_step % args.log_every == 0:
                elapsed = time.perf_counter() - start_time
                record = {
                    "event": "train",
                    "global_step": global_step,
                    "phase": phase,
                    "phase_step": phase_step,
                    "sample_count": sample_count,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "loss": float(loss.detach()),
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": global_step * args.batch_size * args.sequence_length / elapsed,
                    **train_metrics,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(json.dumps(record), flush=True)

            phase_end = global_step in {args.mean_steps, total_steps}
            if global_step % args.validate_every == 0 or phase_end:
                validation_samples = 0 if phase == "mean_field" else args.train_samples
                validation = validate(student, teacher, capture, validation_tokens, args, validation_samples)
                record = {
                    "event": "validation",
                    "global_step": global_step,
                    "phase": phase,
                    "sample_count": validation_samples,
                    **validation,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(json.dumps(record), flush=True)
                checkpoint_name = "student_mean.pt" if global_step == args.mean_steps else "student_latest.pt"
                save_checkpoint(output_dir, checkpoint_name, student, optimizer, args, phase, phase_step, validation)

        final_validations = {}
        validation_counts = sorted({0, 1, 4, 8, 16, args.train_samples})
        for count in validation_counts:
            final_validations[str(count)] = validate(student, teacher, capture, validation_tokens, args, count)
        elapsed = time.perf_counter() - start_time
        summary = {
            "event": "complete",
            "elapsed_seconds": elapsed,
            "tokens_processed": total_steps * args.batch_size * args.sequence_length,
            "effective_tokens_per_second": total_steps * args.batch_size * args.sequence_length / elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "final_validation_by_sample_count": final_validations,
        }
        log_file.write(json.dumps(summary) + "\n")
        log_file.flush()
        print(json.dumps(summary), flush=True)
        save_checkpoint(
            output_dir,
            "student_sampled.pt",
            student,
            optimizer,
            args,
            "full_path_sample_aware",
            args.sample_steps,
            final_validations[str(args.train_samples)],
        )
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    hook.remove()


if __name__ == "__main__":
    main()
