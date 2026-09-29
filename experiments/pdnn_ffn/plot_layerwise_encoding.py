from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot fixed and learned p-bit encoding results")
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    root = Path(args.results_dir)
    layers = [10, 12, 19]
    labels = [str(layer) for layer in layers]

    perplexity: dict[str, list[float]] = {"fixed": [], "learnable": []}
    input_saturation: dict[str, list[float]] = {"fixed": [], "learnable": []}
    hidden_saturation: dict[str, list[float]] = {"fixed": [], "learnable": []}
    for mode in perplexity:
        for layer in layers:
            directory = root / f"layer{layer}_{mode}"
            values = [
                float(load_json(directory / f"ppl_samples4_seed{seed}.json")["perplexity"])
                for seed in range(3)
            ]
            perplexity[mode].append(statistics.mean(values))
            saturation = load_json(directory / "encoding_saturation.json")
            input_saturation[mode].append(
                100.0 * float(saturation["input_probability"]["saturated_fraction"])
            )
            hidden_saturation[mode].append(
                100.0 * float(saturation["hidden_probabilities"][0]["saturated_fraction"])
            )

    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = list(range(len(layers)))
    width = 0.36
    colors = {"fixed": "#667085", "learnable": "#2672e3"}
    names = {"fixed": "Fixed encoding", "learnable": "Learned encoding"}
    for offset, mode in zip((-width / 2, width / 2), ("fixed", "learnable")):
        axes[0].bar(
            [value + offset for value in x],
            perplexity[mode],
            width,
            color=colors[mode],
            label=names[mode],
        )
    axes[0].axhline(11.652735, color="#b42318", linestyle="--", linewidth=1, label="Original Qwen")
    axes[0].set_xticks(x, labels)
    axes[0].set_xlabel("Decoder layer")
    axes[0].set_ylabel("WikiText-2 perplexity (N=4)")
    axes[0].set_title("Single-layer replacement")
    axes[0].legend(fontsize=8)
    axes[0].grid(axis="y", alpha=0.2)

    for mode, linestyle in (("fixed", "--"), ("learnable", "-")):
        axes[1].plot(
            x,
            input_saturation[mode],
            marker="o",
            linestyle=linestyle,
            color=colors[mode],
            label=f"{names[mode]}: input",
        )
        axes[1].plot(
            x,
            hidden_saturation[mode],
            marker="s",
            linestyle=linestyle,
            color=colors[mode],
            alpha=0.72,
            label=f"{names[mode]}: hidden",
        )
    axes[1].set_xticks(x, labels)
    axes[1].set_xlabel("Decoder layer")
    axes[1].set_ylabel("Probability outside [0.01, 0.99] (%)")
    axes[1].set_title("Encoding probability saturation")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    figure.tight_layout()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=180)


if __name__ == "__main__":
    main()
