"""Validate saved paired conditions and summarize an exploratory joint run."""
import argparse
import json
import math
from pathlib import Path
import statistics

from run_joint_feature_ablation import CONDITIONS, LAYERS


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.results
    assert (root / "RUN_COMPLETE").is_file()
    baseline = read(args.baseline)
    assert baseline["scored_tokens"] == 299077
    report = {"baseline_ppl": baseline["perplexity"],
              "initial_n4_seed0_ppl": read(root / "initial" / "ppl_n4_s0.json")["perplexity"],
              "conditions": {},
              "scope": "One joint-training seed per condition; sample standard deviation across three inference seeds only."}
    reference_args = None
    initial_validation = None
    for name, steps, weight in CONDITIONS:
        directory = root / name
        summary = read(directory / "summary.json")
        records = [json.loads(line) for line in (directory / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines()]
        header = records[0]
        cfg = header["args"]
        assert cfg["steps"] == steps and cfg["feature_weight"] == weight
        assert cfg["validation_seed"] == 12345 and cfg["validation_batches"] == 64
        assert header["layers"] == list(LAYERS) and header["effective_tokens_per_update"] == 1024
        comparable = {k: v for k, v in cfg.items() if k not in ("steps", "feature_weight", "output_dir")}
        if reference_args is None:
            reference_args = comparable
        assert comparable == reference_args, "Conditions changed more than budget/feature weight"
        validation = [r for r in records if r["event"] == "validation"]
        assert all(r["scored_tokens"] == 16320 for r in validation)
        if initial_validation is None:
            initial_validation = validation[0]["perplexity"]
        assert abs(validation[0]["perplexity"] - initial_validation) < 1e-8
        best = min(validation, key=lambda r: r["perplexity"])
        assert summary["best_step"] == best["step"]
        assert records[-1]["event"] == "complete" and records[-1]["steps"] == steps
        values = {}
        for count, seed in ((4, 0), (4, 1), (4, 2), (0, 0), (16, 0)):
            result = read(directory / f"ppl_n{count}_s{seed}.json")
            assert result["layers"] == list(LAYERS)
            assert result["sample_count"] == count and result["seed"] == seed
            assert result["scored_tokens"] == 299077
            assert result["max_length"] == 2048 and result["stride"] == 1024
            assert math.isfinite(result["perplexity"])
            values[f"n{count}_s{seed}"] = result["perplexity"]
        seeds = [values[f"n4_s{s}"] for s in (0, 1, 2)]
        mean = statistics.mean(seeds)
        report["conditions"][name] = {
            "steps": steps, "feature_weight": weight, "best_step": summary["best_step"],
            "best_validation_ppl": summary["best_validation_perplexity"],
            "n4_ppl_inference_seeds": seeds, "n4_mean": mean,
            "n4_sample_std": statistics.stdev(seeds), "all_ppl": values,
            "increase_percent_vs_original": (mean / baseline["perplexity"] - 1) * 100,
            "elapsed_seconds": summary["elapsed_seconds"],
            "validation_curve": [{k: r[k] for k in ("step", "perplexity")} for r in validation],
        }
    c = report["conditions"]
    report["long_minus_reference_ppl"] = c["long_3000"]["n4_mean"] - c["reference_1000"]["n4_mean"]
    report["feature_minus_same_budget_control_ppl"] = c["feature_3000"]["n4_mean"] - c["long_3000"]["n4_mean"]
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
