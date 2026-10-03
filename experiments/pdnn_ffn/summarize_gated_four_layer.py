import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    root = args.directory
    status = json.loads((root / "status.json").read_text())
    if status["status"] != "complete":
        raise RuntimeError(f"experiment is not complete: {status}")
    protocol = json.loads((root / "protocol.json").read_text())
    rows = []
    for stage in ("independent", "joint"):
        for architecture in protocol["architectures"]:
            directory = root / "evaluation" / stage / architecture
            records = {}
            for count in (0, 4, 16):
                seeds = (0, 1, 2) if count == 4 else (0,)
                values = [json.loads((directory / f"samples{count}_seed{seed}_ppl.json").read_text()) for seed in seeds]
                if any(value["layers"] != protocol["layers"] or value["scored_tokens"] != 299077 for value in values):
                    raise ValueError("evaluation protocol mismatch")
                ppls = [value["perplexity"] for value in values]
                records[f"n{count}_ppl_mean"] = statistics.mean(ppls)
                records[f"n{count}_ppl_population_std"] = statistics.pstdev(ppls)
                records[f"n{count}_ppls"] = ppls
            rows.append({"stage": stage, "architecture": architecture, **records})
    joint_training = {}
    for architecture in protocol["architectures"]:
        summary = json.loads((root / "joint" / architecture / "summary.json").read_text())
        if summary["event"] != "complete" or summary["layers"] != protocol["layers"]:
            raise ValueError("joint training summary mismatch")
        joint_training[architecture] = summary
    individual = {}
    for architecture in protocol["architectures"]:
        individual[architecture] = {}
        for layer in (9, 15, 18):
            summary = json.loads((root / "individual" / architecture / f"layer{layer}" / "summary.json").read_text())
            individual[architecture][str(layer)] = {
                "elapsed_seconds": summary["elapsed_seconds"],
                "peak_allocated_bytes": summary["peak_allocated_bytes"],
                "n4_validation_nmse": summary["final_validation_by_sample_count"]["4"]["normalized_mse"],
            }
    source_names = [
        "full_path_pdnn_ffn.py", "train_full_path_distillation.py",
        "train_joint_full_path_distillation.py", "evaluate_multi_layer_full_path_perplexity.py",
        "run_gated_four_layer_comparison.py", "summarize_gated_four_layer.py",
    ]
    source_hashes = {name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest() for name in source_names}
    checkpoint_hashes = {}
    for architecture in protocol["architectures"]:
        independent_paths = []
        for layer in protocol["layers"]:
            if layer == 12:
                independent_paths.append(Path(protocol["layer12_sources"][architecture]))
            else:
                independent_paths.append(root / "individual" / architecture / f"layer{layer}" / "student_sampled.pt")
        for layer, path in zip(protocol["layers"], independent_paths):
            checkpoint_hashes[f"independent/{architecture}/layer{layer}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for layer in protocol["layers"]:
            path = root / "joint" / architecture / f"best_layer{layer}.pt"
            checkpoint_hashes[f"joint/{architecture}/layer{layer}/{path.name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    result = {
        "protocol": protocol,
        "rows": rows,
        "joint_training": joint_training,
        "individual_training": individual,
        "checkpoint_sha256": checkpoint_hashes,
        "source_sha256": source_hashes,
        "notes": [
            "N=4 reports three inference seeds and population standard deviation; all training uses seed 0.",
            "The test set is used only after training; joint checkpoint selection uses held-out train-tail validation.",
            "Equal parameter count and updates are not equal arithmetic: gated performs more dense operations per path.",
        ],
    }
    (root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
