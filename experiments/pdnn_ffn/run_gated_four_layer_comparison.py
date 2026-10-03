"""Four-layer serial/gated comparison on layers 9, 12, 15, and 18.

The runner is restart-aware at stage boundaries and refuses to reuse partial
single-layer outputs. It is intended to run detached on the experiment host.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path("results/gated_four_layer_9_12_15_18_20260930")
EXPERIMENT = Path("experiments/pdnn_ffn")
LAYERS = (9, 12, 15, 18)
ARCHITECTURES = ("serial", "gated_dual_rail")
IDLE_GPUS = (0, 1, 2, 3, 4, 5, 7)
LAYER12_SOURCES = {
    "serial": Path("results/gated_sample_budget_20260929/serial_train4/student_step6000.pt"),
    "gated_dual_rail": Path("results/gated_sample_budget_20260929/gated_dual_rail_train4/student_step6000.pt"),
}


def run(command, gpu, log_path):
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "OMP_NUM_THREADS": "4",
        "TOKENIZERS_PARALLELISM": "false",
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as stream:
        subprocess.run(command, env=environment, stdout=stream, stderr=subprocess.STDOUT, check=True)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def train_individual(architecture, layer, gpu):
    output = ROOT / "individual" / architecture / f"layer{layer}"
    if output.exists():
        if not (output / "student_sampled.pt").exists() or not (output / "summary.json").exists():
            raise RuntimeError(f"partial output already exists: {output}")
        return
    command = [
        sys.executable, str(EXPERIMENT / "train_full_path_distillation.py"),
        "--model", "models/Qwen2.5-0.5B",
        "--train-text", "data/wikitext-2/wiki.train.raw",
        "--output-dir", str(output),
        "--layer", str(layer),
        "--architecture", architecture,
        "--hidden-sizes", "4864" if architecture == "serial" else "3242",
        "--coding", "binary", "--temperature-only",
        "--input-temperature", "0.125", "--hidden-temperature", "0.5",
        "--mean-steps", "2000", "--sample-steps", "6000", "--train-samples", "4",
        "--isolate-validation-rng", "--seed", "0",
    ]
    run(command, gpu, ROOT / "logs" / f"individual_{architecture}_layer{layer}.log")


def source_checkpoints(architecture, joint=False):
    if joint:
        return [ROOT / "joint" / architecture / f"best_layer{layer}.pt" for layer in LAYERS]
    paths = []
    for layer in LAYERS:
        paths.append(
            LAYER12_SOURCES[architecture]
            if layer == 12
            else ROOT / "individual" / architecture / f"layer{layer}" / "student_sampled.pt"
        )
    return paths


def evaluate_set(architecture, stage):
    checkpoints = source_checkpoints(architecture, joint=stage == "joint")
    output_dir = ROOT / "evaluation" / stage / architecture
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs = [(0, 0), (4, 0), (4, 1), (4, 2), (16, 0)]

    def one(index, count, seed):
        output = output_dir / f"samples{count}_seed{seed}_ppl.json"
        if output.exists():
            return
        command = [
            sys.executable, str(EXPERIMENT / "evaluate_multi_layer_full_path_perplexity.py"),
            "--model", "models/Qwen2.5-0.5B", "--checkpoints", *map(str, checkpoints),
            "--text-file", "data/wikitext-2/wiki.test.raw", "--output", str(output),
            "--sample-count", str(count), "--seed", str(seed),
        ]
        run(command, IDLE_GPUS[index], ROOT / "logs" / f"eval_{stage}_{architecture}_samples{count}_seed{seed}.log")

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(one, index, count, seed) for index, (count, seed) in enumerate(jobs)]
        for future in futures:
            future.result()


def train_joint(architecture, gpu):
    output = ROOT / "joint" / architecture
    if (output / "summary.json").exists():
        return
    if output.exists():
        raise RuntimeError(f"partial joint output already exists: {output}")
    checkpoints = source_checkpoints(architecture)
    command = [
        sys.executable, str(EXPERIMENT / "train_joint_full_path_distillation.py"),
        "--model", "models/Qwen2.5-0.5B", "--checkpoints", *map(str, checkpoints),
        "--train-text", "data/wikitext-2/wiki.train.raw", "--output-dir", str(output),
        "--steps", "1000", "--sample-count", "4", "--seed", "0",
    ]
    run(command, gpu, ROOT / "logs" / f"joint_{architecture}.log")


def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    write_json(ROOT / "protocol.json", {
        "layers": LAYERS,
        "architectures": ARCHITECTURES,
        "individual_training": "2000 mean-field + 6000 N=4 updates",
        "joint_training": "1000 N=4 end-to-end updates; 0.8 KL + 0.2 next-token CE",
        "layer12_sources": {key: str(value) for key, value in LAYER12_SOURCES.items()},
        "comparison": "Near-equal parameters, equal updates and tokens; gated has more dense arithmetic.",
        "training_seed": 0,
        "inference": "Full WikiText-2; N=4 seeds 0/1/2; mean-field and N=16 seed 0.",
    })
    try:
        jobs = [(architecture, layer) for architecture in ARCHITECTURES for layer in LAYERS if layer != 12]
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(train_individual, architecture, layer, gpu) for gpu, (architecture, layer) in zip(IDLE_GPUS, jobs)]
            for future in futures:
                future.result()
        write_json(ROOT / "individual_complete.json", {"status": "complete"})

        for architecture in ARCHITECTURES:
            evaluate_set(architecture, "independent")
        write_json(ROOT / "independent_evaluation_complete.json", {"status": "complete"})

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [executor.submit(train_joint, architecture, gpu) for gpu, architecture in zip(IDLE_GPUS, ARCHITECTURES)]
            for future in futures:
                future.result()
        write_json(ROOT / "joint_training_complete.json", {"status": "complete"})

        for architecture in ARCHITECTURES:
            evaluate_set(architecture, "joint")

        hashes = {}
        for architecture in ARCHITECTURES:
            for stage, checkpoints in (("independent", source_checkpoints(architecture)), ("joint", source_checkpoints(architecture, joint=True))):
                for path in checkpoints:
                    layer = int(path.stem.split("layer")[-1]) if "layer" in path.stem else int(json.loads((Path(path).parent / "train_metrics.jsonl").read_text().splitlines()[0])["args"]["layer"])
                    hashes[f"{stage}/{architecture}/layer{layer}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        write_json(ROOT / "checkpoint_hashes.json", hashes)
        write_json(ROOT / "status.json", {"status": "complete"})
    except Exception as error:
        write_json(ROOT / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
