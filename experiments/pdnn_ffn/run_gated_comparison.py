"""Matched-parameter gate/serial pilot, with one explicitly selected GPU per job.

Run from the project root. Existing output directories are rejected to preserve
prior results. Checkpoints stay on the training host; JSON records are portable.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs="+", required=True, type=int)
    parser.add_argument("--layers", nargs="+", type=int, default=[10, 12, 19])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    jobs = [(layer, architecture) for layer in args.layers for architecture in ("serial", "gated_dual_rail")]
    if len(args.gpus) != len(jobs) or len(set(args.gpus)) != len(args.gpus):
        raise ValueError("select one distinct idle GPU for each layer/architecture pair")
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in usage.strip().splitlines()}
    if any(memory[gpu] > 500 for gpu in args.gpus):
        raise RuntimeError("a selected GPU is already in use; choose idle GPUs explicitly")
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    experiment = Path("experiments/pdnn_ffn")

    def run_one(gpu, layer, architecture):
        output = root / f"layer{layer}_{architecture}"
        output.mkdir()
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "TOKENIZERS_PARALLELISM": "false", "OMP_NUM_THREADS": "4"}
        commands = []

        def run(script, arguments, name):
            command = [sys.executable, str(experiment / script), *map(str, arguments)]
            commands.append({"command": command, "log": name})
            (output / "commands.json").write_text(json.dumps({"gpu": gpu, "commands": commands}, indent=2) + "\n")
            with (output / name).open("w") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)

        try:
            run("train_full_path_distillation.py", [
                "--model", "models/Qwen2.5-0.5B", "--train-text", "data/wikitext-2/wiki.train.raw",
                "--output-dir", output, "--layer", layer, "--architecture", architecture,
                "--hidden-sizes", 4864 if architecture == "serial" else 3242,
                "--coding", "binary", "--temperature-only", "--input-temperature", 0.125,
                "--hidden-temperature", 0.5, "--seed", 0,
            ], "training.log")
            for count, seed in ((0, 0), (4, 0), (4, 1), (4, 2), (16, 0)):
                run("evaluate_full_path_perplexity.py", [
                    "--model", "models/Qwen2.5-0.5B", "--checkpoint", output / "student_sampled.pt",
                    "--text-file", "data/wikitext-2/wiki.test.raw", "--output", output / f"samples{count}_seed{seed}_ppl.json",
                    "--sample-count", count, "--seed", seed,
                ], f"eval_samples{count}_seed{seed}.log")
            run("measure_encoding_saturation.py", [
                "--model", "models/Qwen2.5-0.5B", "--checkpoint", output / "student_sampled.pt",
                "--text-file", "data/wikitext-2/wiki.train.raw", "--output", output / "saturation.json",
            ], "saturation.log")
            (output / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
        except Exception as error:
            (output / "status.json").write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
            raise
        print(f"completed layer={layer} architecture={architecture}", flush=True)

    with ThreadPoolExecutor(max_workers=len(jobs)) as executor:
        futures = [executor.submit(run_one, gpu, *job) for gpu, job in zip(args.gpus, jobs)]
        for future in futures:
            future.result()
    (root / "status.json").write_text(json.dumps({"status": "complete", "jobs": jobs}) + "\n")


if __name__ == "__main__":
    main()
