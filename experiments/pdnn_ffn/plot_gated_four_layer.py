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
    lookup = {(row["stage"], row["architecture"]): row for row in data["rows"]}
    groups = [
        ("independent", "serial", "Serial\nindependent"),
        ("independent", "gated_dual_rail", "Gated\nindependent"),
        ("joint", "serial", "Serial\njoint"),
        ("joint", "gated_dual_rail", "Gated\njoint"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    x = np.arange(len(groups))
    for count, offset, color in ((0, -0.23, "#6c8ebf"), (4, 0, "#d97a45"), (16, 0.23, "#73a857")):
        values = [lookup[(stage, architecture)][f"n{count}_ppl_mean"] for stage, architecture, _ in groups]
        errors = [lookup[(stage, architecture)][f"n{count}_ppl_population_std"] for stage, architecture, _ in groups]
        bars = axes[0].bar(x + offset, values, width=0.22, yerr=errors, capsize=2, color=color, label="Mean-field" if count == 0 else f"N={count}")
        axes[0].bar_label(bars, fmt="%.3f", fontsize=8, rotation=90, padding=3)
    axes[0].axhline(11.652735, color="black", linestyle="--", linewidth=1, label="Original Qwen")
    axes[0].set(title="Full WikiText-2 perplexity", ylabel="PPL (lower is better)", xticks=x, xticklabels=[label for _, _, label in groups])
    axes[0].set_ylim(11.4, max(row["n4_ppl_mean"] for row in data["rows"]) + 0.8)
    axes[0].legend(fontsize=8, ncol=2)

    for architecture, color, label in (("serial", "#3976a8", "Serial"), ("gated_dual_rail", "#c85a3e", "Gated")):
        records = [json.loads(line) for line in (args.directory / "joint" / architecture / "train_metrics.jsonl").read_text().splitlines()]
        validation = [record for record in records if record["event"] == "validation"]
        axes[1].plot([record["step"] for record in validation], [record["perplexity"] for record in validation], marker="o", markersize=3, label=label, color=color)
    axes[1].set(title="Held-out joint-training curve", xlabel="Joint update", ylabel="Validation PPL")
    axes[1].legend()
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    fig.suptitle("Layers 9, 12, 15, 18: serial versus gated 0/1 FFNs")
    fig.savefig(args.directory / "four_layer_comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
