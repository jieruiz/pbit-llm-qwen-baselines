"""Summarize the layer-12 serial/gated width sweep."""
import csv
import json
from pathlib import Path
import statistics


ROOT = Path("results/pdnn_width_sweep_layer12_20260930")


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    protocol = read_json(ROOT / "protocol.json")
    manifest = read_json(ROOT / "manifest.json")
    rows = []
    for config in protocol["configs"]:
        name = config["name"]
        evaluation = ROOT / "evaluation" / name
        mean_field = read_json(evaluation / "samples0_seed0.json")
        n4_records = [read_json(evaluation / f"samples4_seed{seed}.json") for seed in range(3)]
        n16 = read_json(evaluation / "samples16_seed0.json")
        moments = read_json(ROOT / "moments" / f"{name}.json")
        summary_path = manifest[name]["summary"]
        training_summary = read_json(summary_path)
        n4_values = [record["perplexity"] for record in n4_records]
        normalized = moments["normalized"]
        rows.append({
            "name": name,
            "architecture": config["architecture"],
            "width": config["width"],
            "trainable_parameters": manifest[name]["trainable_parameter_count"],
            "trainable_parameters_millions": manifest[name]["trainable_parameter_count"] / 1e6,
            "mean_field_ppl": mean_field["perplexity"],
            "n4_ppl_mean": statistics.mean(n4_values),
            "n4_ppl_population_sd": statistics.pstdev(n4_values),
            "n4_ppl_by_seed": n4_values,
            "n16_ppl": n16["perplexity"],
            "scored_tokens": mean_field["scored_tokens"],
            "ppl_peak_allocated_bytes": max(
                [mean_field["peak_allocated_bytes"], n16["peak_allocated_bytes"]]
                + [record["peak_allocated_bytes"] for record in n4_records]
            ),
            "corrected_bias_nmse": normalized["corrected_bias_nmse"],
            "single_path_variance_nmse": normalized["single_path_variance_nmse"],
            "predicted_n4_nmse": normalized["predicted_n4_nmse"],
            "empirical_n4_nmse": normalized["empirical_n4_nmse"],
            "predicted_n16_nmse": normalized["predicted_n16_nmse"],
            "empirical_n16_nmse": normalized["empirical_n16_nmse"],
            "mean_field_nmse": normalized["mean_field_nmse"],
            "training_seconds": training_summary["elapsed_seconds"],
            "training_peak_allocated_bytes": training_summary["peak_allocated_bytes"],
            "effective_tokens_per_second": training_summary["effective_tokens_per_second"],
            "checkpoint_sha256": manifest[name]["checkpoint_sha256"],
        })
    rows.sort(key=lambda row: (row["architecture"], row["width"]))

    by_name = {row["name"]: row for row in rows}
    comparisons = []
    pairs = [(4864, 3242), (6144, 4096), (7296, 4864)]
    for serial_width, gated_width in pairs:
        serial = by_name[f"serial_width{serial_width}"]
        gated = by_name[f"gated_dual_rail_width{gated_width}"]
        comparisons.append({
            "serial_width": serial_width,
            "gated_width": gated_width,
            "serial_parameters": serial["trainable_parameters"],
            "gated_parameters": gated["trainable_parameters"],
            "parameter_difference_fraction": (gated["trainable_parameters"] - serial["trainable_parameters"]) / serial["trainable_parameters"],
            "gated_minus_serial_mean_field_ppl": gated["mean_field_ppl"] - serial["mean_field_ppl"],
            "gated_minus_serial_n4_ppl": gated["n4_ppl_mean"] - serial["n4_ppl_mean"],
            "gated_minus_serial_n16_ppl": gated["n16_ppl"] - serial["n16_ppl"],
        })

    result = {
        "status": "complete",
        "rows": rows,
        "matched_parameter_comparisons": comparisons,
        "notes": [
            "All models replace only Qwen layer 12 and use 0/1 coding with learnable temperatures and matrix biases.",
            "N=4 reports three complete WikiText-2 runs; the displayed standard deviation is population SD over seeds 0,1,2.",
            "The baseline pair reuses the controlled 2000 mean-field + 6000 N=4 checkpoints; wider models use the same schedule from scratch.",
            "Fixed-input moments use common original-teacher inputs and 256 sampled paths per input.",
        ],
    }
    (ROOT / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    fieldnames = [key for key in rows[0] if key != "n4_ppl_by_seed"]
    with (ROOT / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fieldnames} for row in rows)


if __name__ == "__main__":
    main()
