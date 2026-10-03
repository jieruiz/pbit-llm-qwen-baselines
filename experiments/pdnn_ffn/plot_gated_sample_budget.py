"""Plot completed sample-budget comparisons; data are read from comparison.json."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    data = json.loads((args.directory / "comparison.json").read_text())
    final_step = max(row["sample_training_steps"] for row in data["rows"])
    order = [f"{arch}_train{count}" for arch in ("serial", "gated_dual_rail") for count in (4, 16)]
    labels = ["Serial\ntrain N=4", "Serial\ntrain N=16", "Gated\ntrain N=4", "Gated\ntrain N=16"]
    final = [next(row for row in data["rows"] if row["run"] == run and row["sample_training_steps"] == final_step) for run in order]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    for run, label in zip(order, labels):
        records = [json.loads(line) for line in (args.directory / run / "train_metrics.jsonl").read_text().splitlines()]
        validation = [item for item in records if item["event"] == "validation"]
        axes[0, 0].plot([item["global_step"] for item in validation], [item["normalized_mse"] for item in validation], label=label.replace("\n", " / "))
    axes[0, 0].set(title="Common N=4 held-out validation", xlabel="Sample-training update", ylabel="NMSE")
    axes[0, 0].legend(fontsize=8)
    x = np.arange(4)
    for count, offset, color in ((4, -0.18, "#3377aa"), (16, 0.18, "#dd9944")):
        values = [row[f"n{count}_ppl_mean"] for row in final]
        errors = [row[f"n{count}_ppl_population_std"] for row in final]
        bars = axes[0, 1].bar(x + offset, values, width=0.36, yerr=errors, color=color, label=f"Inference N={count}", capsize=3)
        axes[0, 1].bar_label(bars, fmt="%.4f", fontsize=8, padding=4)
    axes[0, 1].set(title=f"Full WikiText-2 PPL / update {final_step}", ylabel="PPL (lower is better)", xticks=x, xticklabels=labels)
    axes[0, 1].set_ylim(11.65, max(row["n4_ppl_mean"] for row in final) + 0.045)
    axes[0, 1].legend(fontsize=8)
    for axis, count in zip(axes[1], (4, 16)):
        bias = [row["moments"]["corrected_bias_nmse"] for row in final]
        noise = [row["moments"]["single_path_variance_nmse"] / count for row in final]
        axis.bar(x, bias, label="Estimated squared bias", color="#547eaa")
        axis.bar(x, noise, bottom=bias, label=f"Variance / {count}", color="#e3ad56")
        axis.set(title=f"Fixed-input MSE decomposition / inference N={count}", ylabel="Normalized MSE", xticks=x, xticklabels=labels)
        axis.set_ylim(0, 1.3 * max(row["moments"]["predicted_n4_nmse"] for row in final))
        axis.legend(fontsize=8)
    for axis in axes.flat:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Layer 12: common warm starts, equal training updates, different path budgets", fontsize=13)
    fig.savefig(args.directory / "sample_budget_comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
