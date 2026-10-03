"""Correct global-step/phase-step checkpoint confusion without retraining.

Keep the original evaluation for audit, and save corrected results separately.
"""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

import torch
from run_pdnn_width_sweep import CONFIGS, ROOT, MODEL, TRAIN_TEXT, TEST_TEXT, run, sha256, parameter_count


def main():
    output = ROOT / "corrected_final"
    usage = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
    memory = {int(row.split(",")[0]): int(row.split(",")[1]) for row in usage.strip().splitlines()}
    gpus = [gpu for gpu in sorted(memory) if memory[gpu] < 500][:4]
    if len(gpus) < 4:
        raise RuntimeError(f"Need four idle GPUs: {memory}")
    output.mkdir(exist_ok=False)
    (output / "moments").mkdir()
    protocol = json.loads((ROOT / "protocol.json").read_text())
    protocol["correction"] = "New-width student_step6000.pt represented 4000 sample updates. Use student_sampled.pt, verified phase step=6000."
    protocol["limitations"] = ["One training seed only.", "Baseline sample stage reseeded on resume; new widths were trained continuously."]
    manifest = {}
    for config in CONFIGS:
        name = config["name"]
        if config["train"]:
            config["checkpoint"] = str(ROOT / "training" / name / "student_sampled.pt")
            config["summary"] = str(ROOT / "training" / name / "summary.json")
        payload = torch.load(config["checkpoint"], map_location="cpu", weights_only=False)
        assert payload["phase"] == "full_path_sample_aware", name
        assert payload["step"] == 6000, (name, payload["step"])
        assert payload["training_args"]["layer"] == 12, name
        manifest[name] = {
            "checkpoint": config["checkpoint"], "checkpoint_sha256": sha256(config["checkpoint"]),
            "summary": config["summary"], "trainable_parameter_count": parameter_count(config["architecture"], config["width"]),
            "verified_phase": payload["phase"], "verified_sample_phase_step": payload["step"],
        }
        del payload
        if not config["train"]:
            shutil.copytree(ROOT / "evaluation" / name, output / "evaluation" / name)
            shutil.copyfile(ROOT / "moments" / f"{name}.json", output / "moments" / f"{name}.json")
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "status.json").write_text(json.dumps({"status": "running"}) + "\n")

    def one(gpu, config):
        name = config["name"]
        for count, seed in protocol["evaluation"]["sample_counts_and_seeds"]:
            label = f"samples{count}_seed{seed}"
            command = [sys.executable, "experiments/pdnn_ffn/evaluate_full_path_perplexity.py", "--model", MODEL,
                       "--checkpoint", config["checkpoint"], "--text-file", TEST_TEXT, "--output",
                       str(output / "evaluation" / name / f"{label}.json"), "--sample-count", str(count), "--seed", str(seed)]
            run(command, gpu, output / "logs" / f"{name}_{label}.log")
        run([sys.executable, "experiments/pdnn_ffn/measure_path_moments.py", "--model", MODEL,
             "--checkpoint", config["checkpoint"], "--train-text", TRAIN_TEXT, "--output",
             str(output / "moments" / f"{name}.json")], gpu, output / "logs" / f"{name}_moments.log")

    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            jobs = [executor.submit(one, gpu, config) for gpu, config in zip(gpus, [c for c in CONFIGS if c["train"]])]
            for job in jobs:
                job.result()
        import summarize_pdnn_width_sweep as summary
        summary.ROOT = output
        summary.main()
        (output / "status.json").write_text(json.dumps({"status": "complete"}) + "\n")
    except Exception as error:
        (output / "status.json").write_text(json.dumps({"status": "failed", "error": repr(error)}) + "\n")
        raise


if __name__ == "__main__":
    main()
