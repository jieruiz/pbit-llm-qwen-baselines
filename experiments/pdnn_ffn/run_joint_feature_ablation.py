"""Prespecified four-FFN budget/feature comparison in one GPU allocation."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

LAYERS = (10, 11, 12, 13)
CONDITIONS = (("reference_1000", 1000, 0.0), ("long_3000", 3000, 0.0),
              ("feature_3000", 3000, 0.1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-text", required=True)
    parser.add_argument("--test-text", required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_dir.resolve()
    if root.exists() and any(root.iterdir()):
        raise RuntimeError("Refusing to overwrite an experiment directory")
    root.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    plan = {
        "base_main_commit": "f41bbbb",
        "layers": LAYERS, "coding": "bipolar", "input_temperature": 0.25,
        "hidden_temperature": 1.0, "joint_training_seeds": [0],
        "inference_seeds": [0, 1, 2], "training_paths": 4,
        "conditions": [{"name": n, "steps": s, "feature_weight": w} for n, s, w in CONDITIONS],
        "feature_target": "same-token FFN continuous outputs AFTER per-FFN path averaging; each model uses its own inputs",
        "validation": "64 fixed 256-token windows across the held-out last 65536 train tokens; fixed RNG seed 12345",
        "selection": "lowest validation CE/PPL; no test-based selection or feature-weight tuning",
        "scope": "Exploratory, one joint-training seed per condition. Inference repeats are not training replications.",
        "scheduler": "fresh optimizer and 50-step warmup/cosine for each total budget; 3000 is not a continuation of 1000",
        "job_id": os.environ.get("SLURM_JOB_ID"),
    }
    for name, path in (("train", args.train_text), ("test", args.test_text)):
        plan[name + "_sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    (root / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (root / "command_logs").mkdir()

    def run(command, label):
        start = time.perf_counter()
        command = [sys.executable, *map(str, command)]
        print("START", label, flush=True)
        with (root / "command_logs" / (label + ".log")).open("w", encoding="utf-8") as log:
            result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
        event = {"label": label, "command": command, "seconds": time.perf_counter() - start,
                 "returncode": result.returncode}
        with (root / "events.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(event) + "\n")
        if result.returncode:
            raise RuntimeError(f"{label} failed; inspect command_logs/{label}.log")
        print("DONE", label, round(event["seconds"], 2), flush=True)

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    assert abs(baseline["perplexity"] - 11.652735047455899) < 1e-5
    assert baseline["scored_tokens"] == 299077
    local_common = [scripts / "train_full_path_distillation.py", "--model", args.model,
                    "--train-text", args.train_text, "--input-temperature", "0.25",
                    "--hidden-temperature", "1.0", "--seed", "0"]
    smoke = root / "smoke"
    run([*local_common, "--layer", "12", "--mean-steps", "2", "--sample-steps", "2",
         "--sequence-length", "32", "--batch-size", "1", "--validation-batches", "1",
         "--output-dir", smoke / "local"], "smoke_local")

    def train_joint(checkpoints, output, steps, weight, smoke_mode=False):
        command = [scripts / "train_joint_feature_distillation.py", "--model", args.model,
                   "--train-text", args.train_text, "--checkpoints", *checkpoints,
                   "--output-dir", output, "--steps", steps, "--feature-weight", weight,
                   "--seed", "0", "--sample-count", "4", "--validation-seed", "12345",
                   "--log-every", "50", "--validation-every", "500", "--validation-batches", "64"]
        if smoke_mode:
            command += ["--sequence-length", "32", "--gradient-accumulation", "1",
                        "--validation-batches", "2", "--validation-every", "2",
                        "--log-every", "1", "--warmup-steps", "1"]
        run(command, output.name + "_train")

    train_joint([smoke / "local" / "student_sampled.pt"], smoke / "joint", 4, 0.1, True)

    def evaluate(checkpoints, output, samples=4, seed=0, max_tokens=None):
        command = [scripts / "evaluate_multi_layer_full_path_perplexity.py", "--model", args.model,
                   "--checkpoints", *checkpoints, "--text-file", args.test_text, "--output", output,
                   "--sample-count", samples, "--seed", seed]
        if max_tokens is not None:
            command += ["--max-tokens", max_tokens]
        run(command, output.parent.name + "_" + output.stem)

    evaluate([smoke / "joint" / "best_layer12.pt"], smoke / "ppl.json", max_tokens=128)
    sources = []
    for layer in LAYERS:
        directory = root / "initial" / f"layer{layer}"
        run([*local_common, "--layer", layer, "--output-dir", directory], f"initial_layer{layer}")
        sources.append(directory / "student_sampled.pt")
    identities = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    (root / "initial-checkpoints.json").write_text(json.dumps(identities, indent=2) + "\n", encoding="utf-8")
    evaluate(sources, root / "initial" / "ppl_n4_s0.json")
    # All runs have fixed conditions before any full-test PPL is inspected.
    for name, steps, weight in CONDITIONS:
        train_joint(sources, root / name, steps, weight)
    for name, _, _ in CONDITIONS:
        directory = root / name
        checkpoints = [directory / f"best_layer{layer}.pt" for layer in LAYERS]
        for samples, seed in ((4, 0), (4, 1), (4, 2), (0, 0), (16, 0)):
            evaluate(checkpoints, directory / f"ppl_n{samples}_s{seed}.json", samples, seed)
    (root / "RUN_COMPLETE").write_text("complete\n", encoding="utf-8")
    print("RUN_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
