"""Recover the moment/manifest stage after completed training and PPL evaluation."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys

from run_pdnn_width_sweep import CONFIGS, MODEL, ROOT, TRAIN_TEXT, parameter_count, run, sha256


def main():
    (ROOT / "moments").mkdir(parents=True, exist_ok=True)
    for config in CONFIGS:
        if config["train"]:
            output = ROOT / "training" / config["name"]
            config["checkpoint"] = str(output / "student_sampled.pt")
            config["summary"] = str(output / "summary.json")

    def one(index, config):
        output = ROOT / "moments" / f"{config['name']}.json"
        command = [
            sys.executable, "experiments/pdnn_ffn/measure_path_moments.py",
            "--model", MODEL, "--checkpoint", config["checkpoint"], "--train-text", TRAIN_TEXT,
            "--output", str(output),
        ]
        run(command, index, ROOT / "logs" / f"moments_{config['name']}.log")

    try:
        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(one, index, config) for index, config in enumerate(CONFIGS)]
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
        (ROOT / "status.json").write_text(json.dumps({"status": "complete", "recovered_moment_stage": True}, indent=2) + "\n")
    except Exception as error:
        (ROOT / "status.json").write_text(json.dumps({"status": "failed", "error": repr(error)}, indent=2) + "\n")
        raise


if __name__ == "__main__":
    main()
