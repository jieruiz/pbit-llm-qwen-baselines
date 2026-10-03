"""Plot perplexity, error decomposition, and cost for the width sweep."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path("results/pdnn_width_sweep_layer12_20260930")


def main():
    rows = json.loads((ROOT / "comparison.json").read_text())["rows"]
    styles = {
        "serial": ("#1f77b4", "o", "Serial"),
        "gated_dual_rail": ("#d62728", "s", "Gated dual-rail"),
    }
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for architecture, (color, marker, label) in styles.items():
        selected = sorted((row for row in rows if row["architecture"] == architecture), key=lambda row: row["trainable_parameters"])
        x = [row["trainable_parameters_millions"] for row in selected]
        axes[0].plot(x, [row["mean_field_ppl"] for row in selected], marker=marker, color=color, linestyle="--", label=f"{label}, mean-field")
        axes[0].errorbar(x, [row["n4_ppl_mean"] for row in selected], yerr=[row["n4_ppl_population_sd"] for row in selected], marker=marker, color=color, capsize=3, label=f"{label}, N=4")
        axes[0].plot(x, [row["n16_ppl"] for row in selected], marker=marker, color=color, linestyle=":", label=f"{label}, N=16")
        axes[1].plot(x, [row["corrected_bias_nmse"] for row in selected], marker=marker, color=color, label=f"{label}, bias")
        axes[1].plot(x, [row["single_path_variance_nmse"] / 4 for row in selected], marker=marker, color=color, linestyle="--", label=f"{label}, variance/4")
        # Reused baseline summaries time only the 6000-step continuation;
        # new-width summaries time all 8000 steps. Do not compare those clocks.
        axes[2].plot(x[1:], [row["training_seconds"] for row in selected[1:]], marker=marker, color=color, label=f"{label}, time")
    axes[0].set_ylabel("WikiText-2 perplexity")
    axes[0].set_title("Layer-12 replacement quality")
    axes[1].set_ylabel("Normalized MSE")
    axes[1].set_title("N=4 error components")
    axes[2].set_ylabel("Training wall time (s)")
    axes[2].set_title("Fresh runs only (8000 steps)")
    for axis in axes:
        axis.set_xlabel("Trainable parameters (millions)")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    fig.suptitle("0/1 p-bit FFN width sweep, Qwen2.5-0.5B layer 12")
    fig.tight_layout()
    fig.savefig(ROOT / "width_sweep.png", dpi=180, bbox_inches="tight")
    fig.savefig(ROOT / "width_sweep.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
