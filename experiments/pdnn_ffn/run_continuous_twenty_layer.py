"""Extend raw floating-input and matched sigmoid students from 10 to 20 layers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import torch
from run_continuous_ten_layer import run, write, read, pooled, PAIRS, MODES, EXP

ROOT = Path("results/continuous_raw_twenty_layer_20261001")
PREVIOUS = Path("results/continuous_raw_ten_layer_20261001")
OLD_LAYERS = (7, 8, 9, 10, 11, 12, 13, 14, 15, 18)
LAYERS = (1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22)
NEW_LAYERS = tuple(layer for layer in LAYERS if layer not in OLD_LAYERS)
SOURCES = ("run_continuous_twenty_layer.py", "run_continuous_ten_layer.py",
           "continuous_input_ffn.py", "full_path_pdnn_ffn.py", "multibit_input_ffn.py",
           "train_full_path_distillation.py", "train_joint_full_path_distillation.py",
           "evaluate_full_path_perplexity.py", "evaluate_multi_layer_full_path_perplexity.py")
PROTOCOL = {
    "layers": list(LAYERS), "new_layers": list(NEW_LAYERS), "old_layers": list(OLD_LAYERS),
    "retained_original_layers": [0, 2, 3, 23], "modes": list(MODES), "training_seed": 0,
    "previous_run": str(PREVIOUS), "model": "Qwen2.5-0.5B Base", "hidden_width": 4864,
    "input_policy": "continuous_raw: raw BF16 activations, no clipping or input encoding; sigmoid: binary 0/1 input sampling",
    "hidden_policy": "binary 0/1 p-bits, learned scalar positive temperature, all matrix biases retained, no separate p-bit threshold",
    "local_mean_steps": 2000, "local_sample_steps": 6000, "local_train_samples": 4,
    "local_batch_size": 4, "sequence_length": 256, "local_learning_rates": [0.0003, 0.0001],
    "initial_input_temperature_sigmoid": 0.125, "initial_hidden_temperature": 0.5,
    "independent_stage": "All 20 locally trained final checkpoints; old 10 reused with SHA256 identity verification",
    "staged_stage": "Previous 10-layer best joint checkpoints plus 10 new local students",
    "joint_start": "staged", "joint_steps": 1000, "joint_samples": 4,
    "joint_learning_rate": 0.00001, "joint_kd_weight": 0.8,
    "joint_micro_batch": 1, "joint_gradient_accumulation": 4,
    "selection": "Minimum held-out train-tail validation PPL; no test-based checkpoint selection",
    "inherited_best_steps": {"continuous_raw": 900, "sigmoid": 1000},
    "budget_note": "Both arms follow same schedules and selection policy. Old 10 layers have prior joint adaptation; this is a staged 10-to-20 experiment, not a fresh 20-layer joint run.",
    "evaluation_stages": ["independent", "staged", "joint"],
    "evaluation_pairs": [list(pair) for pair in PAIRS],
    "test": "Full WikiText-2 raw test; 2048 context, 1024 stride",
    "historical_note": "Older 43.383146 binary twenty-layer result used 2000 local sample updates and different encoding settings; historical reference only",
}


def checkpoint(mode, layer, stage="independent"):
    if stage == "joint":
        return ROOT / "joint" / mode / f"best_layer{layer}.pt"
    if layer in OLD_LAYERS:
        if stage == "staged":
            return PREVIOUS / "joint" / mode / f"best_layer{layer}.pt"
        return PREVIOUS / "individual" / mode / f"layer{layer}" / "student_sampled.pt"
    return ROOT / "individual" / mode / f"layer{layer}" / "student_sampled.pt"


def verify(mode, layer, stage="independent"):
    path = checkpoint(mode, layer, stage)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args, config = payload["training_args"], payload["student_config"]
    if (args["layer"] != layer or config["input_encoding"] != mode
            or list(config["hidden_sizes"]) != [4864] or config["coding"] != "binary"
            or config["architecture"] != "serial" or config["learnable_thresholds"]):
        raise RuntimeError(f"Incorrect checkpoint configuration: {path}")
    inherited = layer in OLD_LAYERS and stage != "joint"
    if inherited:
        previous_stage = "joint" if stage == "staged" else "independent"
        identity = read(PREVIOUS / "checkpoints" / previous_stage / mode / f"layer{layer}.json")
        if identity["sha256"] != sha:
            raise RuntimeError(f"Inherited checkpoint changed: {path}")
    if stage == "joint" or (stage == "staged" and layer in OLD_LAYERS):
        summary_root = ROOT if stage == "joint" else PREVIOUS
        step = read(summary_root / "joint" / mode / "summary.json")["best_step"]
        if payload["phase"] != "joint_end_to_end_distillation" or payload["step"] != step:
            raise RuntimeError(f"Incorrect joint checkpoint step: {path}")
    else:
        if (payload["phase"] != "full_path_sample_aware" or payload["step"] != 6000
                or args["mean_steps"] != 2000 or args["sample_steps"] != 6000):
            raise RuntimeError(f"Incorrect local training duration: {path}")
    write(ROOT / "checkpoints" / stage / mode / f"layer{layer}.json", {
        "path": str(path), "sha256": sha, "inherited": inherited,
        "phase": payload["phase"], "step": payload["step"], "config": config,
    })


def individual(mode, layer, gpu):
    output = checkpoint(mode, layer).parent
    progress = ROOT / "progress" / f"{mode}_layer{layer}.json"
    if not (output / "summary.json").exists():
        if output.exists():
            raise RuntimeError(f"Partial local output exists; manual recovery required: {output}")
        write(progress, {"stage": "training", "gpu": gpu})
        run("train_full_path_distillation.py", [
            "--model", "models/Qwen2.5-0.5B", "--train-text", "data/wikitext-2/wiki.train.raw",
            "--output-dir", output, "--layer", layer, "--hidden-sizes", 4864,
            "--coding", "binary", "--temperature-only", "--input-temperature", 0.125,
            "--hidden-temperature", 0.5, "--input-encoding", mode, "--mean-steps", 2000,
            "--sample-steps", 6000, "--train-samples", 4, "--validation-sample-count", 4,
            "--isolate-validation-rng", "--seed", 0,
        ], gpu, ROOT / "logs" / f"train_{mode}_layer{layer}.log")
    verify(mode, layer)
    write(progress, {"stage": "single_layer_evaluation", "gpu": gpu})
    for count in (0, 4):
        destination = ROOT / "evaluation" / "single_layer" / mode / f"layer{layer}" / f"samples{count}_seed0.json"
        if not destination.exists():
            run("evaluate_full_path_perplexity.py", [
                "--model", "models/Qwen2.5-0.5B", "--checkpoint", checkpoint(mode, layer),
                "--text-file", "data/wikitext-2/wiki.test.raw", "--output", destination,
                "--sample-count", count, "--seed", 0,
            ], gpu, ROOT / "logs" / f"single_{mode}_layer{layer}_n{count}.log")
    write(progress, {"stage": "complete", "gpu": gpu})


def evaluate(mode, stage, count, seed, gpu):
    destination = ROOT / "evaluation" / stage / mode / f"samples{count}_seed{seed}.json"
    if destination.exists():
        return
    run("evaluate_multi_layer_full_path_perplexity.py", [
        "--model", "models/Qwen2.5-0.5B", "--checkpoints",
        *[checkpoint(mode, layer, stage) for layer in LAYERS],
        "--text-file", "data/wikitext-2/wiki.test.raw", "--output", destination,
        "--sample-count", count, "--seed", seed,
    ], gpu, ROOT / "logs" / f"{stage}_{mode}_n{count}_seed{seed}.log")


def joint(mode, gpu):
    output = ROOT / "joint" / mode
    if not (output / "summary.json").exists():
        if output.exists():
            raise RuntimeError(f"Partial joint output exists; manual recovery required: {output}")
        run("train_joint_full_path_distillation.py", [
            "--model", "models/Qwen2.5-0.5B", "--checkpoints",
            *[checkpoint(mode, layer, "staged") for layer in LAYERS],
            "--train-text", "data/wikitext-2/wiki.train.raw", "--output-dir", output,
            "--steps", 1000, "--sample-count", 4, "--seed", 0,
        ], gpu, ROOT / "logs" / f"joint_{mode}.log")
    for layer in LAYERS:
        verify(mode, layer, "joint")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU IDs must be nonempty and unique")
    memory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    usage = {int(line.split(",")[0]): int(line.split(",")[1]) for line in memory.splitlines()}
    if any(usage.get(gpu, 100000) > 500 for gpu in gpus):
        raise RuntimeError(f"Requested GPU is occupied: {usage}")
    hashes = {name: hashlib.sha256((EXP / name).read_bytes()).hexdigest() for name in SOURCES}
    # The core code must be the same as in the source experiment.
    for name, sha in read(PREVIOUS / "source_sha256.json").items():
        if hashes[name] != sha:
            raise RuntimeError(f"Source differs from ten-layer run: {name}")
    if args.resume:
        if read(ROOT / "protocol.json") != PROTOCOL or read(ROOT / "source_sha256.json") != hashes:
            raise RuntimeError("Resume protocol or sources changed")
    else:
        ROOT.mkdir(parents=True, exist_ok=False)
        write(ROOT / "protocol.json", PROTOCOL)
        write(ROOT / "source_sha256.json", hashes)
    try:
        write(ROOT / "status.json", {"stage": "verifying_inherited_checkpoints"})
        for mode in MODES:
            for layer in OLD_LAYERS:
                verify(mode, layer)
                verify(mode, layer, "staged")
                for count in (0, 4):
                    relative = Path("evaluation") / "single_layer" / mode / f"layer{layer}" / f"samples{count}_seed0.json"
                    target = ROOT / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        shutil.copy2(PREVIOUS / relative, target)
        write(ROOT / "status.json", {"stage": "individual_training_and_evaluation"})
        pooled([(mode, layer) for layer in NEW_LAYERS for mode in MODES], individual, gpus)
        for mode in MODES:
            for layer in NEW_LAYERS:
                verify(mode, layer, "staged")
        write(ROOT / "status.json", {"stage": "independent_and_staged_evaluation"})
        pooled([(mode, stage, count, seed) for mode in MODES
                for stage in ("independent", "staged") for count, seed in PAIRS], evaluate, gpus)
        write(ROOT / "status.json", {"stage": "joint_training"})
        pooled([(mode,) for mode in MODES], joint, gpus[:2])
        write(ROOT / "status.json", {"stage": "joint_evaluation"})
        pooled([(mode, "joint", count, seed) for mode in MODES for count, seed in PAIRS], evaluate, gpus)
        write(ROOT / "status.json", {"stage": "complete"})
    except Exception as error:
        write(ROOT / "status.json", {"stage": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
