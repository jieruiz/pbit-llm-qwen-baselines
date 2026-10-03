"""Matched ten-layer raw floating-input versus sigmoid-input experiment."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import torch

ROOT = Path("results/continuous_raw_ten_layer_20261001")
EXP = Path("experiments/pdnn_ffn")
LAYERS = (7, 8, 9, 10, 11, 12, 13, 14, 15, 18)
MODES = ("continuous_raw", "sigmoid")
PAIRS = ((0, 0), (4, 0), (4, 1), (4, 2), (16, 0))
SOURCES = ("continuous_input_ffn.py", "full_path_pdnn_ffn.py", "multibit_input_ffn.py",
           "train_full_path_distillation.py", "train_joint_full_path_distillation.py",
           "evaluate_full_path_perplexity.py", "evaluate_multi_layer_full_path_perplexity.py",
           "run_continuous_ten_layer.py")
PROTOCOL = {
    "layers": list(LAYERS), "modes": list(MODES), "training_seed": 0,
    "model": "Qwen2.5-0.5B Base", "hidden_width": 4864, "coding": "binary 0/1",
    "mean_steps": 2000, "sample_steps": 6000, "train_samples": 4,
    "local_batch_size": 4, "sequence_length": 256,
    "local_learning_rates": [0.0003, 0.0001],
    "joint_steps": 1000, "joint_learning_rate": 0.00001, "joint_kd_weight": 0.8,
    "joint_micro_batch": 1, "joint_gradient_accumulation": 4,
    "input_temperature_sigmoid_initial": 0.125, "hidden_temperature_initial": 0.5,
    "temperature_policy": "Learn scalar temperatures; no separate p-bit threshold; all matrix biases retained",
    "continuous_policy": "Identity input: no p-bit, no quantization, no calibration, no clipping",
    "precision": "BF16 activations/autocast with FP32 trainable parameters; continuous means floating-point, not FP32-only inference",
    "initialization": "Fresh all 20 local students; seed 0 matrix draws and train windows shared; sampling RNG consumption differs",
    "joint_selection": "Minimum held-out train-tail validation PPL; never test-set selection",
    "evaluation_pairs": [list(pair) for pair in PAIRS],
    "test": "Full WikiText-2 raw test, context 2048 stride 1024",
    "comparison": "Equal width, data, updates and N; sigmoid has one extra input-temperature scalar per FFN",
    "historical_reference": "Older ten-layer 18.568570 used only 2000 sample updates and different temperature/threshold policy; not a matched control",
}


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def run(script, args, gpu, log):
    command = [sys.executable, str(EXP / script), *map(str, args)]
    log.parent.mkdir(parents=True, exist_ok=True)
    write(log.with_suffix(".command.json"), {"command": command, "gpu": gpu})
    with log.open("w") as stream:
        subprocess.run(command, check=True, stdout=stream, stderr=subprocess.STDOUT,
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                            "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"})


def checkpoint(mode, layer, joint=False):
    if joint:
        return ROOT / "joint" / mode / f"best_layer{layer}.pt"
    return ROOT / "individual" / mode / f"layer{layer}" / "student_sampled.pt"


def verify_checkpoint(mode, layer, joint=False):
    path = checkpoint(mode, layer, joint)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args, config = payload["training_args"], payload["student_config"]
    if (int(args["layer"]) != layer or config["input_encoding"] != mode
            or list(config["hidden_sizes"]) != [4864] or config["coding"] != "binary"):
        raise RuntimeError(f"Wrong checkpoint identity: {path}")
    if not joint and (payload["phase"] != "full_path_sample_aware" or payload["step"] != 6000):
        raise RuntimeError(f"Wrong training duration: {path}")
    if joint:
        expected_step = read(ROOT / "joint" / mode / "summary.json")["best_step"]
        if payload["phase"] != "joint_end_to_end_distillation" or payload["step"] != expected_step:
            raise RuntimeError(f"Wrong joint checkpoint: {path}")
    write(ROOT / "checkpoints" / ("joint" if joint else "independent") / mode / f"layer{layer}.json", {
        "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "phase": payload["phase"], "step": payload["step"], "config": config,
    })


def individual(mode, layer, gpu):
    output = checkpoint(mode, layer).parent
    progress = ROOT / "progress" / f"{mode}_layer{layer}.json"
    if not (output / "summary.json").exists():
        if output.exists():
            raise RuntimeError(f"Partial local training exists, manual recovery needed: {output}")
        write(progress, {"stage": "training", "gpu": gpu})
        run("train_full_path_distillation.py", [
            "--model", "models/Qwen2.5-0.5B", "--train-text", "data/wikitext-2/wiki.train.raw",
            "--output-dir", output, "--layer", layer, "--hidden-sizes", 4864,
            "--coding", "binary", "--temperature-only", "--input-temperature", 0.125,
            "--hidden-temperature", 0.5, "--input-encoding", mode, "--mean-steps", 2000,
            "--sample-steps", 6000, "--train-samples", 4, "--validation-sample-count", 4,
            "--isolate-validation-rng", "--seed", 0,
        ], gpu, ROOT / "logs" / f"train_{mode}_layer{layer}.log")
    verify_checkpoint(mode, layer)
    write(progress, {"stage": "single_layer_evaluation", "gpu": gpu})
    for count in (0, 4):
        output = ROOT / "evaluation" / "single_layer" / mode / f"layer{layer}" / f"samples{count}_seed0.json"
        if output.exists():
            continue
        run("evaluate_full_path_perplexity.py", [
            "--model", "models/Qwen2.5-0.5B", "--checkpoint", checkpoint(mode, layer),
            "--text-file", "data/wikitext-2/wiki.test.raw", "--output", output,
            "--sample-count", count, "--seed", 0,
        ], gpu, ROOT / "logs" / f"single_{mode}_layer{layer}_n{count}.log")
    write(progress, {"stage": "complete", "gpu": gpu})


def pooled(items, worker, gpus):
    # One sequential queue per GPU, no two child processes share the same GPU.
    def consume(gpu, queue):
        for item in queue:
            worker(*item, gpu)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        jobs = [pool.submit(consume, gpu, items[index::len(gpus)]) for index, gpu in enumerate(gpus)]
        for job in jobs:
            job.result()


def evaluate(mode, stage, count, seed, gpu):
    output = ROOT / "evaluation" / stage / mode / f"samples{count}_seed{seed}.json"
    if output.exists():
        return
    run("evaluate_multi_layer_full_path_perplexity.py", [
        "--model", "models/Qwen2.5-0.5B", "--checkpoints",
        *[checkpoint(mode, layer, joint=stage == "joint") for layer in LAYERS],
        "--text-file", "data/wikitext-2/wiki.test.raw", "--output", output,
        "--sample-count", count, "--seed", seed,
    ], gpu, ROOT / "logs" / f"{stage}_{mode}_n{count}_seed{seed}.log")


def joint(mode, gpu):
    output = ROOT / "joint" / mode
    if not (output / "summary.json").exists():
        if output.exists():
            raise RuntimeError(f"Partial joint training exists, manual recovery needed: {output}")
        run("train_joint_full_path_distillation.py", [
            "--model", "models/Qwen2.5-0.5B", "--checkpoints",
            *[checkpoint(mode, layer) for layer in LAYERS],
            "--train-text", "data/wikitext-2/wiki.train.raw", "--output-dir", output,
            "--steps", 1000, "--sample-count", 4, "--seed", 0,
        ], gpu, ROOT / "logs" / f"joint_{mode}.log")
    for layer in LAYERS:
        verify_checkpoint(mode, layer, joint=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    args = parser.parse_args()
    gpus = [int(value) for value in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus):
        raise ValueError("GPU list must be nonempty and unique")
    memory = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    used = {int(line.split(",")[0]): int(line.split(",")[1]) for line in memory.splitlines()}
    if any(used.get(gpu, 100000) > 500 for gpu in gpus):
        raise RuntimeError(f"Requested GPU occupied: {used}")
    hashes = {name: hashlib.sha256((EXP / name).read_bytes()).hexdigest() for name in SOURCES}
    if args.resume:
        if read(ROOT / "protocol.json") != PROTOCOL or read(ROOT / "source_sha256.json") != hashes:
            raise RuntimeError("Resume protocol/source differs from original run")
    else:
        ROOT.mkdir(parents=True, exist_ok=False)
        write(ROOT / "protocol.json", PROTOCOL)
        write(ROOT / "source_sha256.json", hashes)
    try:
        write(ROOT / "status.json", {"stage": "individual_training_and_evaluation"})
        pooled([(mode, layer) for layer in LAYERS for mode in MODES], individual, gpus)
        write(ROOT / "status.json", {"stage": "independent_evaluation"})
        pooled([(mode, "independent", count, seed) for mode in MODES for count, seed in PAIRS], evaluate, gpus)
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
