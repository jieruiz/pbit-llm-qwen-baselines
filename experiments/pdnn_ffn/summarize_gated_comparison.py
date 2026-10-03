"""Collect finished pilot metrics and checkpoint/source hashes without weights."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    if json.loads((root / "status.json").read_text())["status"] != "complete":
        raise RuntimeError("wait until every comparison job is complete")
    runs = []
    for folder in sorted(root.glob("layer*")):
        summary = json.loads((folder / "summary.json").read_text())
        with (folder / "train_metrics.jsonl").open() as stream:
            header = json.loads(next(stream))
        evaluations = [json.loads((folder / f"samples4_seed{seed}_ppl.json").read_text()) for seed in range(3)]
        if any(value["scored_tokens"] != 299077 for value in evaluations):
            raise ValueError("unexpected WikiText-2 test scoring count")
        ppls = [value["perplexity"] for value in evaluations]
        saturation = json.loads((folder / "saturation.json").read_text())
        runs.append({
            "run": folder.name,
            "layer": header["args"]["layer"],
            "architecture": header["args"]["architecture"],
            "parameters": header["student_parameters"],
            "hidden_sizes": header["args"]["hidden_sizes"],
            "n4_ppl_mean": statistics.mean(ppls),
            "n4_ppl_population_std": statistics.pstdev(ppls),
            "n4_ppls": ppls,
            "mean_field_ppl": json.loads((folder / "samples0_seed0_ppl.json").read_text())["perplexity"],
            "n16_ppl": json.loads((folder / "samples16_seed0_ppl.json").read_text())["perplexity"],
            "n4_validation_nmse": summary["final_validation_by_sample_count"]["4"]["normalized_mse"],
            "training_seconds": summary["elapsed_seconds"],
            "peak_training_allocated_bytes": summary["peak_allocated_bytes"],
            "temperatures": {
                "input": saturation["input_temperature"],
                "hidden_or_gate": saturation["hidden_temperatures"][0],
                "value": saturation.get("value_temperature"),
            },
            "checkpoint_sha256": sha256(folder / "student_sampled.pt"),
        })
    source_files = [
        "full_path_pdnn_ffn.py", "train_full_path_distillation.py",
        "evaluate_full_path_perplexity.py", "measure_encoding_saturation.py",
        "run_gated_comparison.py", "test_gated_pdnn_ffn.py", "summarize_gated_comparison.py",
    ]
    result = {
        "runs": runs,
        "source_sha256": {name: sha256(Path(__file__).parent / name) for name in source_files},
        "notes": [
            "One training seed; N=4 uses three inference seeds, reported as population standard deviation.",
            "Both arms learn temperatures, retain matrix biases, and have no separate p-bit thresholds.",
            "Near-equal parameters, different widths; dense matrix arithmetic is 33.3% higher for gated.",
            "Single FFN replacement; no multi-layer quality or hardware efficiency claim.",
            "NMSE sample-count groups use different validation windows; compare architectures at the same count.",
        ],
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
