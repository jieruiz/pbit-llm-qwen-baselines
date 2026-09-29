from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot full-path P-DNN layer sensitivity and cumulative replacement PPL")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-ppl", type=float, default=11.652735)
    return parser.parse_args()


def load_ppl(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["perplexity"])


def main() -> None:
    args = parse_args()
    single = [load_ppl(args.results_dir / f"layer{layer}_samples4_seed0_ppl.json") for layer in range(24)]
    counts = list(range(4, 25))
    cumulative = [
        load_ppl(args.results_dir / f"best_ranked_count{count}_samples4_seed0_ppl.json") for count in counts
    ]
    split_paths = [
        args.results_dir / f"split_ranked_count{count}_secondhalf_samples4_seed0_ppl.json" for count in counts
    ]
    heldout = [load_ppl(path) for path in split_paths] if all(path.exists() for path in split_paths) else None

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)

    axes[0].bar(range(24), single, color="#4472C4")
    axes[0].axhline(args.baseline_ppl, color="#222222", linestyle="--", linewidth=1.2, label="original Qwen")
    axes[0].set_yscale("log")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_xlabel("Decoder layer")
    axes[0].set_ylabel("WikiText-2 perplexity (log scale)")
    axes[0].set_title("Single-layer replacement sensitivity")
    axes[0].legend()

    axes[1].plot(
        counts,
        cumulative,
        marker="o",
        markersize=4,
        color="#C44E52",
        linewidth=1.8,
        label="full test, test-ranked (exploratory)",
    )
    if heldout is not None:
        axes[1].plot(
            counts,
            heldout,
            marker="s",
            markersize=3.5,
            color="#4C9F70",
            linewidth=1.5,
            label="held-out half, first-half-ranked",
        )
        split_baseline_path = args.results_dir / "split_original_second_half_ppl.json"
        if split_baseline_path.exists():
            axes[1].axhline(
                load_ppl(split_baseline_path),
                color="#666666",
                linestyle="-.",
                linewidth=1.0,
                label="held-out-half Qwen",
            )
    axes[1].axhline(args.baseline_ppl, color="#222222", linestyle="--", linewidth=1.2, label="original Qwen")
    axes[1].axhline(20.0, color="#DD8452", linestyle=":", linewidth=1.2, label="PPL 20")
    axes[1].set_yscale("log")
    axes[1].set_xticks(range(4, 25, 2))
    axes[1].set_xlabel("Number of replaced FFNs")
    axes[1].set_ylabel("WikiText-2 perplexity (log scale)")
    axes[1].set_title("Cumulative replacement, safest layers first")
    axes[1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
