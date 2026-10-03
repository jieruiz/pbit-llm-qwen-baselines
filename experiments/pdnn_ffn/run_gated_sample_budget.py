"""Controlled forks of existing layer-12 mean-field checkpoints; no overwrite."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", nargs=4, type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dir", default="results/gated_dual_rail_20260929")
    parser.add_argument("--steps", type=int, default=6000)
    args = parser.parse_args()
    if args.steps < 2000 or args.steps % 2000:
        raise ValueError("steps must be at least 2000 and divisible by 2000")
    jobs = [(arch, count) for arch in ("serial", "gated_dual_rail") for count in (4, 16)]
    if len(set(args.gpus)) != 4:
        raise ValueError("four distinct idle GPUs required")
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in usage.strip().splitlines()}
    if any(memory[gpu] > 500 for gpu in args.gpus):
        raise RuntimeError("a selected GPU is already in use")
    sources = {arch: Path(args.source_dir) / f"layer12_{arch}" / "student_mean.pt" for arch, _ in jobs}
    source_hashes = {arch: hashlib.sha256(path.read_bytes()).hexdigest() for arch, path in sources.items()}
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=False)
    (root / "protocol.json").write_text(json.dumps({
        "args": vars(args), "source_sha256": source_hashes,
        "restored_optimizer": True, "training_window_offset": 2000,
        "validation_rng_isolated": True, "training_rng_reseeded": True,
        "note": "Controlled forks, not bit-exact continuation: old checkpoints lack RNG state. Equal updates/tokens; N=16 uses four times the stochastic paths.",
    }, indent=2) + "\n")

    def one(gpu, architecture, train_count):
        output = root / f"{architecture}_train{train_count}"
        output.mkdir()
        environment = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4", "TOKENIZERS_PARALLELISM": "false"}
        commands = []

        def run(script, arguments, log_name):
            command = [sys.executable, f"experiments/pdnn_ffn/{script}", *map(str, arguments)]
            commands.append(command)
            (output / "commands.json").write_text(json.dumps({"gpu": gpu, "commands": commands}, indent=2) + "\n")
            with (output / log_name).open("w") as log:
                subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)

        try:
            run("train_full_path_distillation.py", [
                "--model", "models/Qwen2.5-0.5B", "--train-text", "data/wikitext-2/wiki.train.raw",
                "--output-dir", output, "--layer", 12, "--initialize-from", sources[architecture],
                "--mean-steps", 0, "--sample-steps", args.steps, "--train-samples", train_count,
                "--training-window-offset", 2000, "--checkpoint-every", 2000,
                "--validation-sample-count", 4, "--isolate-validation-rng", "--seed", 0,
            ], "training.log")
            for step in sorted({2000, args.steps}):
                checkpoint = output / f"student_step{step}.pt"
                for count, seed in [(0, 0), *[(count, seed) for count in (4, 16) for seed in range(3)]]:
                    name = f"step{step}_samples{count}_seed{seed}"
                    run("evaluate_full_path_perplexity.py", [
                        "--model", "models/Qwen2.5-0.5B", "--checkpoint", checkpoint,
                        "--text-file", "data/wikitext-2/wiki.test.raw", "--output", output / f"{name}_ppl.json",
                        "--sample-count", count, "--seed", seed,
                    ], f"{name}.log")
            for label, checkpoint in [("warm", sources[architecture]), ("step2000", output / "student_step2000.pt"), (f"step{args.steps}", output / f"student_step{args.steps}.pt")]:
                run("measure_path_moments.py", [
                    "--model", "models/Qwen2.5-0.5B", "--checkpoint", checkpoint,
                    "--train-text", "data/wikitext-2/wiki.train.raw", "--output", output / f"{label}_moments.json",
                ], f"{label}_moments.log")
            hashes = {f"student_step{step}.pt": hashlib.sha256((output / f"student_step{step}.pt").read_bytes()).hexdigest() for step in sorted({2000, args.steps})}
            (output / "checkpoint_hashes.json").write_text(json.dumps(hashes, indent=2) + "\n")
            (output / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
        except Exception as error:
            (output / "status.json").write_text(json.dumps({"status": "failed", "error": str(error)}) + "\n")
            raise
        try:
            print(f"complete {output.name}", flush=True)
        except BrokenPipeError:
            # An SSH stdout disconnect must not mark finished experiments failed.
            pass

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(one, gpu, *job) for gpu, job in zip(args.gpus, jobs)]
        for future in futures:
            future.result()
    (root / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")


if __name__ == "__main__":
    main()
