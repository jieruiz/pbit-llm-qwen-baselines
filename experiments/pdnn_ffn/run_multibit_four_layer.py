"""Four-layer experiment for the stochastic four-bit input P-DNN FFN."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import torch


ROOT = Path("results/input_multibit_four_layer_9_12_15_18_20260930")
EXPERIMENT = Path("experiments/pdnn_ffn")
LAYERS = (9, 12, 15, 18)
GPUS = (0, 1, 2, 3, 4, 5, 6, 7)
EXISTING_LAYER12 = Path("results/input_multibit_layer12_20260930/training/stochastic_k4/student_sampled.pt")
EXISTING_CALIBRATION12 = Path("results/input_multibit_layer12_20260930/calibration.json")
EVALUATIONS = ((0, 0), (4, 0), (4, 1), (4, 2), (16, 0))


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def run(script, args, gpu, log):
    command = [sys.executable, str(EXPERIMENT / script), *map(str, args)]
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)
    return command


def checkpoint_for(layer, joint=False):
    if joint:
        return ROOT / "joint" / f"best_layer{layer}.pt"
    if layer == 12:
        return EXISTING_LAYER12
    return ROOT / "individual" / f"layer{layer}" / "student_sampled.pt"


def calibration_for(layer):
    return ROOT / "calibration" / f"layer{layer}" / "calibration.json"


def calibrate(layer, gpu):
    output = calibration_for(layer).parent
    if (output / "calibration.json").exists() and (output / "projection_audit.json").exists():
        return
    run("calibrate_multibit_input.py", ["--output-dir", output, "--layer", layer], gpu,
        ROOT / "logs" / f"calibrate_layer{layer}.log")


def verify_checkpoint(path, layer):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    args = payload["training_args"]
    config = payload["student_config"]
    if int(args["layer"]) != layer:
        raise RuntimeError(f"{path} targets layer {args['layer']}, expected {layer}")
    if payload["phase"] != "full_path_sample_aware" or int(payload["step"]) != 6000:
        raise RuntimeError(f"unexpected training checkpoint metadata: {path}")
    if config["input_encoding"] != "stochastic" or int(config["input_bits"]) != 4:
        raise RuntimeError(f"unexpected input encoder metadata: {path}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "phase": payload["phase"],
        "phase_step": int(payload["step"]),
        "student_config": config,
    }


def train_individual(layer, gpu):
    if layer == 12:
        return
    output = ROOT / "individual" / f"layer{layer}"
    if (output / "student_sampled.pt").exists() and (output / "summary.json").exists():
        return
    if output.exists():
        raise RuntimeError(f"partial single-layer output already exists: {output}")
    run("train_full_path_distillation.py", [
        "--model", "models/Qwen2.5-0.5B", "--train-text", "data/wikitext-2/wiki.train.raw",
        "--output-dir", output, "--layer", layer, "--hidden-sizes", 4864,
        "--coding", "binary", "--temperature-only", "--input-temperature", 0.125,
        "--hidden-temperature", 0.5, "--input-encoding", "stochastic", "--input-bits", 4,
        "--input-calibration", calibration_for(layer), "--mean-steps", 2000,
        "--sample-steps", 6000, "--train-samples", 4, "--validation-sample-count", 4,
        "--isolate-validation-rng", "--seed", 0,
    ], gpu, ROOT / "logs" / f"train_layer{layer}.log")


def evaluate_single_layer(layer, gpu):
    checkpoint = checkpoint_for(layer)
    output_dir = ROOT / "evaluation" / "single_layer" / f"layer{layer}"
    commands = []
    for count, seed in EVALUATIONS:
        output = output_dir / f"samples{count}_seed{seed}_ppl.json"
        if not output.exists():
            commands.append(run("evaluate_full_path_perplexity.py", [
                "--model", "models/Qwen2.5-0.5B", "--checkpoint", checkpoint,
                "--text-file", "data/wikitext-2/wiki.test.raw", "--output", output,
                "--sample-count", count, "--seed", seed,
            ], gpu, ROOT / "logs" / f"single_layer{layer}_samples{count}_seed{seed}.log"))
    moments = ROOT / "moments" / f"layer{layer}.json"
    if not moments.exists():
        commands.append(run("measure_path_moments.py", [
            "--model", "models/Qwen2.5-0.5B", "--checkpoint", checkpoint,
            "--train-text", "data/wikitext-2/wiki.train.raw", "--output", moments,
        ], gpu, ROOT / "logs" / f"moments_layer{layer}.log"))
    write_json(ROOT / "commands" / f"single_layer{layer}.json", commands)


def evaluate_multi(stage):
    output_dir = ROOT / "evaluation" / stage
    checkpoints = [checkpoint_for(layer, joint=stage == "joint") for layer in LAYERS]

    def one(index, count, seed):
        output = output_dir / f"samples{count}_seed{seed}_ppl.json"
        if output.exists():
            return
        run("evaluate_multi_layer_full_path_perplexity.py", [
            "--model", "models/Qwen2.5-0.5B", "--checkpoints", *checkpoints,
            "--text-file", "data/wikitext-2/wiki.test.raw", "--output", output,
            "--sample-count", count, "--seed", seed,
        ], GPUS[index], ROOT / "logs" / f"{stage}_samples{count}_seed{seed}.log")

    with ThreadPoolExecutor(max_workers=len(EVALUATIONS)) as pool:
        jobs = [pool.submit(one, index, count, seed) for index, (count, seed) in enumerate(EVALUATIONS)]
        for job in jobs:
            job.result()


def train_joint():
    output = ROOT / "joint"
    if (output / "summary.json").exists():
        return
    if output.exists():
        raise RuntimeError(f"partial joint output already exists: {output}")
    run("train_joint_full_path_distillation.py", [
        "--model", "models/Qwen2.5-0.5B", "--checkpoints", *[checkpoint_for(layer) for layer in LAYERS],
        "--train-text", "data/wikitext-2/wiki.train.raw", "--output-dir", output,
        "--steps", 1000, "--sample-count", 4, "--seed", 0,
    ], 0, ROOT / "logs" / "joint.log")


def main():
    # Stage outputs are immutable and checked before reuse, so a failed run can
    # resume without repeating completed calibration, training, or evaluation.
    ROOT.mkdir(parents=True, exist_ok=True)
    write_json(ROOT / "protocol.json", {
        "model": "Qwen2.5-0.5B Base", "layers": LAYERS, "architecture": "serial",
        "coding": "0/1", "hidden_width": 4864, "input_encoding": "stochastic adjacent rounding",
        "input_bits": 4, "individual_training": "2000 mean-field + 6000 N=4 updates",
        "joint_training": "1000 N=4 end-to-end updates; 0.8 KL + 0.2 next-token CE",
        "training_seed": 0, "evaluation_pairs": EVALUATIONS,
        "checkpoint_selection": "student_sampled.pt for independent; held-out train-tail best validation for joint",
        "calibration": "per-layer frozen per-channel abs 99.9 percentile; train-only, excludes validation tail",
        "baseline_comparison": "results/gated_four_layer_9_12_15_18_20260930 serial",
        "cost_note": "K=4 costs four binary input projections plus one binary hidden readout per path.",
    })
    write_json(ROOT / "status.json", {"status": "calibrating"})
    try:
        destination = calibration_for(12).parent
        destination.mkdir(parents=True, exist_ok=True)
        shutil.copy2(EXISTING_CALIBRATION12, destination / "calibration.json")
        source_audit = EXISTING_CALIBRATION12.parent / "projection_audit.json"
        if source_audit.exists():
            shutil.copy2(source_audit, destination / "projection_audit.json")
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(calibrate, layer, gpu) for layer, gpu in zip((9, 15, 18), GPUS)]
            for job in jobs:
                job.result()

        write_json(ROOT / "status.json", {"status": "training_individual"})
        with ThreadPoolExecutor(max_workers=3) as pool:
            jobs = [pool.submit(train_individual, layer, gpu) for layer, gpu in zip((9, 15, 18), GPUS)]
            for job in jobs:
                job.result()
        checkpoint_records = {f"layer{layer}": verify_checkpoint(checkpoint_for(layer), layer) for layer in LAYERS}
        write_json(ROOT / "independent_checkpoints.json", checkpoint_records)

        write_json(ROOT / "status.json", {"status": "evaluating_individual_and_independent"})
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(evaluate_single_layer, layer, gpu) for layer, gpu in zip(LAYERS, GPUS)]
            for job in jobs:
                job.result()
        evaluate_multi("independent")

        write_json(ROOT / "status.json", {"status": "joint_training"})
        train_joint()
        joint_records = {f"layer{layer}": {
            "path": str(checkpoint_for(layer, joint=True)),
            "sha256": hashlib.sha256(checkpoint_for(layer, joint=True).read_bytes()).hexdigest(),
        } for layer in LAYERS}
        write_json(ROOT / "joint_checkpoints.json", joint_records)
        write_json(ROOT / "status.json", {"status": "evaluating_joint"})
        evaluate_multi("joint")

        source_names = (
            "full_path_pdnn_ffn.py", "multibit_input_ffn.py", "train_full_path_distillation.py",
            "train_joint_full_path_distillation.py", "calibrate_multibit_input.py",
            "evaluate_full_path_perplexity.py", "evaluate_multi_layer_full_path_perplexity.py",
            "measure_path_moments.py", "run_multibit_four_layer.py",
        )
        write_json(ROOT / "source_sha256.json", {
            name: hashlib.sha256((EXPERIMENT / name).read_bytes()).hexdigest() for name in source_names
        })
        write_json(ROOT / "status.json", {"status": "complete"})
    except Exception as error:
        write_json(ROOT / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
