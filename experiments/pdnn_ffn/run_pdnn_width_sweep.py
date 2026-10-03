"""Run an equal-budget width sweep for serial and gated 0/1 p-bit FFNs."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


MODEL = "models/Qwen2.5-0.5B"
TRAIN_TEXT = "data/wikitext-2/wiki.train.raw"
TEST_TEXT = "data/wikitext-2/wiki.test.raw"
ROOT = Path("results/pdnn_width_sweep_layer12_20260930")

CONFIGS = [
    {
        "name": "serial_width4864", "architecture": "serial", "width": 4864,
        "checkpoint": "results/gated_sample_budget_20260929/serial_train4/student_step6000.pt",
        "summary": "results/gated_sample_budget_20260929/serial_train4/summary.json",
        "train": False,
    },
    {"name": "serial_width6144", "architecture": "serial", "width": 6144, "train": True},
    {"name": "serial_width7296", "architecture": "serial", "width": 7296, "train": True},
    {
        "name": "gated_dual_rail_width3242", "architecture": "gated_dual_rail", "width": 3242,
        "checkpoint": "results/gated_sample_budget_20260929/gated_dual_rail_train4/student_step6000.pt",
        "summary": "results/gated_sample_budget_20260929/gated_dual_rail_train4/summary.json",
        "train": False,
    },
    {"name": "gated_dual_rail_width4096", "architecture": "gated_dual_rail", "width": 4096, "train": True},
    {"name": "gated_dual_rail_width4864", "architecture": "gated_dual_rail", "width": 4864, "train": True},
]


def parameter_count(architecture, width, model_width=896):
    if architecture == "serial":
        return 2 * model_width * width + width + model_width + 2
    return 3 * model_width * width + 2 * width + model_width + 3


def run(command, gpu, log_path):
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    if ROOT.exists():
        raise FileExistsError(f"refusing to overwrite {ROOT}")
    ROOT.mkdir(parents=True)
    selected_gpus = list(range(6))
    usage = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in usage.strip().splitlines()}
    if any(memory[gpu] > 500 for gpu in selected_gpus):
        raise RuntimeError(f"selected GPUs are not idle: {memory}")

    protocol = {
        "purpose": "Test whether wider p-bit FFNs improve layer-12 replacement quality.",
        "model": MODEL,
        "layer": 12,
        "coding": "binary_0_1",
        "temperature_only": True,
        "input_temperature": 0.125,
        "hidden_temperature": 0.5,
        "training": {"mean_steps": 2000, "sample_steps": 6000, "train_samples": 4, "seed": 0},
        "evaluation": {"sample_counts_and_seeds": [[0, 0], [4, 0], [4, 1], [4, 2], [16, 0]]},
        "moments": {"paths": 256, "batches": 4, "sequence_length": 256, "seed": 27183},
        "model_width": 896,
        "original_qwen_ffn_intermediate_width": 4864,
        "original_qwen_three_matrix_weight_count": 3 * 896 * 4864,
        "configs": [],
    }
    for config in CONFIGS:
        entry = dict(config)
        entry["trainable_parameter_count"] = parameter_count(config["architecture"], config["width"])
        protocol["configs"].append(entry)
    (ROOT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")

    training = [config for config in CONFIGS if config["train"]]

    def train_one(index, config):
        output = ROOT / "training" / config["name"]
        command = [
            sys.executable, "experiments/pdnn_ffn/train_full_path_distillation.py",
            "--model", MODEL, "--train-text", TRAIN_TEXT, "--output-dir", str(output),
            "--layer", "12", "--hidden-sizes", str(config["width"]),
            "--architecture", config["architecture"], "--coding", "binary", "--temperature-only",
            "--input-temperature", "0.125", "--hidden-temperature", "0.5",
            "--mean-steps", "2000", "--sample-steps", "6000", "--train-samples", "4",
            "--checkpoint-every", "2000", "--validation-sample-count", "4",
            "--isolate-validation-rng", "--seed", "0",
        ]
        run(command, selected_gpus[index], ROOT / "logs" / f"train_{config['name']}.log")
        config["checkpoint"] = str(output / "student_sampled.pt")
        config["summary"] = str(output / "summary.json")

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(train_one, index, config) for index, config in enumerate(training)]
            for future in futures:
                future.result()
        (ROOT / "training_complete.json").write_text(json.dumps({"status": "complete"}) + "\n")
        (ROOT / "moments").mkdir(parents=True, exist_ok=True)

        def evaluate_one(index, config):
            output = ROOT / "evaluation" / config["name"]
            output.mkdir(parents=True)
            commands = []
            for sample_count, seed in protocol["evaluation"]["sample_counts_and_seeds"]:
                name = f"samples{sample_count}_seed{seed}"
                command = [
                    sys.executable, "experiments/pdnn_ffn/evaluate_full_path_perplexity.py",
                    "--model", MODEL, "--checkpoint", config["checkpoint"], "--text-file", TEST_TEXT,
                    "--output", str(output / f"{name}.json"), "--sample-count", str(sample_count),
                    "--seed", str(seed),
                ]
                commands.append(command)
                run(command, selected_gpus[index], ROOT / "logs" / f"eval_{config['name']}_{name}.log")
            (output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
            moment_command = [
                sys.executable, "experiments/pdnn_ffn/measure_path_moments.py",
                "--model", MODEL, "--checkpoint", config["checkpoint"], "--train-text", TRAIN_TEXT,
                "--output", str(ROOT / "moments" / f"{config['name']}.json"),
            ]
            run(moment_command, selected_gpus[index], ROOT / "logs" / f"moments_{config['name']}.log")

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(evaluate_one, index, config) for index, config in enumerate(CONFIGS)]
            for future in futures:
                future.result()

        manifest = {}
        for config in CONFIGS:
            manifest[config["name"]] = {
                "checkpoint": config["checkpoint"],
                "checkpoint_sha256": sha256(config["checkpoint"]),
                "summary": config["summary"],
                "trainable_parameter_count": parameter_count(config["architecture"], config["width"]),
            }
        (ROOT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        (ROOT / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
    except Exception as error:
        (ROOT / "status.json").write_text(json.dumps({"status": "failed", "error": repr(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
