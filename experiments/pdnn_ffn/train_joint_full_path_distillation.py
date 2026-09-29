from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from full_path_pdnn_ffn import FullPathPBitFFN, load_full_path_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Jointly adapt multiple full-path bipolar P-DNN FFNs")
    parser.add_argument("--model", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--sequence-length", type=int, default=256)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--validation-tokens", type=int, default=65536)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--validation-every", type=int, default=100)
    parser.add_argument("--sample-count", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--kd-weight", type=float, default=0.8)
    parser.add_argument("--kd-temperature", type=float, default=1.0)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_fields = (
        "steps",
        "sequence_length",
        "micro_batch_size",
        "gradient_accumulation",
        "validation_tokens",
        "validation_batches",
        "validation_every",
        "sample_count",
        "log_every",
    )
    for field in positive_integer_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"{field.replace('_', '-')} must be positive")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.kd_temperature <= 0:
        raise ValueError("kd-temperature must be positive")
    if not 0.0 <= args.kd_weight <= 1.0:
        raise ValueError("kd-weight must be in [0, 1]")
    if args.warmup_steps < 0:
        raise ValueError("warmup-steps must be non-negative")


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def random_batch(
    tokens: torch.Tensor,
    batch_size: int,
    sequence_length: int,
    generator: torch.Generator,
) -> torch.Tensor:
    maximum = tokens.numel() - sequence_length
    if maximum <= 0:
        raise ValueError("training split is shorter than sequence-length")
    starts = torch.randint(0, maximum, (batch_size,), generator=generator)
    return torch.stack([tokens[start : start + sequence_length] for start in starts.tolist()])


def distillation_losses(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    input_ids: torch.Tensor,
    kd_weight: float,
    kd_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    student = student_logits[:, :-1].float()
    teacher = teacher_logits[:, :-1].float()
    labels = input_ids[:, 1:]
    token_count = labels.numel()

    cross_entropy = F.cross_entropy(student.reshape(-1, student.shape[-1]), labels.reshape(-1))
    student_log_probabilities = F.log_softmax(student / kd_temperature, dim=-1)
    with torch.no_grad():
        teacher_log_probabilities = F.log_softmax(teacher / kd_temperature, dim=-1)
    kd_loss = F.kl_div(
        student_log_probabilities,
        teacher_log_probabilities,
        reduction="sum",
        log_target=True,
    )
    kd_loss = kd_loss * (kd_temperature**2) / token_count
    total = kd_weight * kd_loss + (1.0 - kd_weight) * cross_entropy
    return total, cross_entropy, kd_loss


def install_students(
    model: AutoModelForCausalLM,
    checkpoint_paths: list[str],
    sample_count: int,
) -> tuple[dict[int, FullPathPBitFFN], dict[int, dict]]:
    students: dict[int, FullPathPBitFFN] = {}
    source_payloads: dict[int, dict] = {}
    for checkpoint_path in checkpoint_paths:
        student, payload = load_full_path_checkpoint(checkpoint_path)
        layer = int(payload["training_args"]["layer"])
        if layer in students:
            raise ValueError(f"multiple checkpoints target decoder layer {layer}")
        if layer < 0 or layer >= len(model.model.layers):
            raise ValueError(f"checkpoint targets invalid decoder layer {layer}")
        student.set_sample_count(sample_count)
        student.train()
        student.requires_grad_(True)
        model.model.layers[layer].mlp = student
        students[layer] = student
        source_payloads[layer] = payload
    return dict(sorted(students.items())), source_payloads


def learning_rate_multiplier(step: int, steps: int, warmup_steps: int) -> float:
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / warmup_steps
    decay_steps = max(1, steps - warmup_steps)
    progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(
    teacher: AutoModelForCausalLM,
    student_model: AutoModelForCausalLM,
    validation_tokens: torch.Tensor,
    args: argparse.Namespace,
    seed: int,
) -> dict[str, float]:
    maximum = validation_tokens.numel() - args.sequence_length
    if maximum <= 0:
        raise ValueError("validation split is shorter than sequence-length")
    starts = torch.linspace(0, maximum, steps=args.validation_batches).round().long().tolist()
    sums = {"loss": 0.0, "cross_entropy": 0.0, "kd_loss": 0.0}
    scored_tokens = 0
    cuda_device = torch.cuda.current_device()
    with torch.random.fork_rng(devices=[cuda_device]):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        for start in starts:
            ids = validation_tokens[start : start + args.sequence_length].unsqueeze(0).cuda(non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                teacher_logits = teacher(input_ids=ids, use_cache=False).logits
                student_logits = student_model(input_ids=ids, use_cache=False).logits
            loss, cross_entropy, kd_loss = distillation_losses(
                student_logits,
                teacher_logits,
                ids,
                args.kd_weight,
                args.kd_temperature,
            )
            token_count = ids[:, 1:].numel()
            sums["loss"] += float(loss) * token_count
            sums["cross_entropy"] += float(cross_entropy) * token_count
            sums["kd_loss"] += float(kd_loss) * token_count
            scored_tokens += token_count
    means = {key: value / scored_tokens for key, value in sums.items()}
    means["perplexity"] = math.exp(means["cross_entropy"])
    means["scored_tokens"] = scored_tokens
    return means


def save_students(
    output_dir: Path,
    prefix: str,
    students: dict[int, FullPathPBitFFN],
    source_payloads: dict[int, dict],
    args: argparse.Namespace,
    step: int,
    validation: dict[str, float],
) -> None:
    joint_args = vars(args).copy()
    joint_args["checkpoints"] = [str(path) for path in args.checkpoints]
    for layer, student in students.items():
        source = source_payloads[layer]
        training_args = dict(source["training_args"])
        training_args["joint_training"] = joint_args
        training_args["joint_step"] = step
        payload = {
            "student_type": source.get("student_type", "full_path_bipolar_pdnn_v1"),
            "student_config": student.checkpoint_config(),
            "student_state_dict": {key: value.detach().cpu() for key, value in student.state_dict().items()},
            "training_args": training_args,
            "phase": "joint_end_to_end_distillation",
            "step": step,
            "validation": validation,
        }
        torch.save(payload, output_dir / f"{prefix}_layer{layer}.pt")


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_metrics.jsonl"

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    all_tokens = tokenizer(
        Path(args.train_text).read_text(encoding="utf-8"),
        add_special_tokens=False,
        return_tensors="pt",
    ).input_ids[0]
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
    student_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=False,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).cuda().eval()
    student_model.requires_grad_(False)
    students, source_payloads = install_students(student_model, args.checkpoints, args.sample_count)
    trainable_parameters = [parameter for student in students.values() for parameter in student.parameters()]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_multiplier(step, args.steps, args.warmup_steps),
    )
    generator = torch.Generator().manual_seed(args.seed + 17)
    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()

    header = {
        "event": "start",
        "args": vars(args),
        "layers": list(students),
        "base_model_parameters": sum(parameter.numel() for parameter in student_model.parameters()),
        "trainable_parameters": sum(parameter.numel() for parameter in trainable_parameters),
        "corpus_tokens": int(all_tokens.numel()),
        "training_tokens": int(training_tokens.numel()),
        "validation_tokens": int(validation_tokens.numel()),
        "effective_tokens_per_update": args.micro_batch_size
        * args.sequence_length
        * args.gradient_accumulation,
        "gpu": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }

    with log_path.open("w", encoding="utf-8") as log_file:
        log_file.write(json.dumps(header) + "\n")
        log_file.flush()
        print(json.dumps(header), flush=True)

        initial_validation = validate(
            teacher,
            student_model,
            validation_tokens,
            args,
            args.seed + 100000,
        )
        initial_record = {"event": "validation", "step": 0, **initial_validation}
        log_file.write(json.dumps(initial_record) + "\n")
        log_file.flush()
        print(json.dumps(initial_record), flush=True)
        best_perplexity = initial_validation["perplexity"]
        best_step = 0
        save_students(output_dir, "best", students, source_payloads, args, 0, initial_validation)

        for step in range(1, args.steps + 1):
            update_start = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            accumulated = {"loss": 0.0, "cross_entropy": 0.0, "kd_loss": 0.0}
            for _ in range(args.gradient_accumulation):
                ids = random_batch(
                    training_tokens,
                    args.micro_batch_size,
                    args.sequence_length,
                    generator,
                ).cuda(non_blocking=True)
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    teacher_logits = teacher(input_ids=ids, use_cache=False).logits
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    student_logits = student_model(input_ids=ids, use_cache=False).logits
                loss, cross_entropy, kd_loss = distillation_losses(
                    student_logits,
                    teacher_logits,
                    ids,
                    args.kd_weight,
                    args.kd_temperature,
                )
                (loss / args.gradient_accumulation).backward()
                accumulated["loss"] += float(loss.detach()) / args.gradient_accumulation
                accumulated["cross_entropy"] += float(cross_entropy.detach()) / args.gradient_accumulation
                accumulated["kd_loss"] += float(kd_loss.detach()) / args.gradient_accumulation

            gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.max_gradient_norm)
            optimizer.step()
            scheduler.step()
            update_elapsed = time.perf_counter() - update_start

            if step == 1 or step % args.log_every == 0:
                elapsed = time.perf_counter() - start_time
                record = {
                    "event": "train",
                    "step": step,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient_norm),
                    "update_seconds": update_elapsed,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": step
                    * args.micro_batch_size
                    * args.sequence_length
                    * args.gradient_accumulation
                    / elapsed,
                    **accumulated,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(json.dumps(record), flush=True)

            if step % args.validation_every == 0 or step == args.steps:
                validation = validate(
                    teacher,
                    student_model,
                    validation_tokens,
                    args,
                    args.seed + 100000 + step,
                )
                record = {"event": "validation", "step": step, **validation}
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(json.dumps(record), flush=True)
                save_students(output_dir, "latest", students, source_payloads, args, step, validation)
                if validation["perplexity"] < best_perplexity:
                    best_perplexity = validation["perplexity"]
                    best_step = step
                    save_students(output_dir, "best", students, source_payloads, args, step, validation)

        elapsed = time.perf_counter() - start_time
        summary = {
            "event": "complete",
            "layers": list(students),
            "steps": args.steps,
            "best_step": best_step,
            "best_validation_perplexity": best_perplexity,
            "elapsed_seconds": elapsed,
            "effective_tokens_per_second": args.steps
            * args.micro_batch_size
            * args.sequence_length
            * args.gradient_accumulation
            / elapsed,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        }
        log_file.write(json.dumps(summary) + "\n")
        log_file.flush()
        print(json.dumps(summary), flush=True)
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
