from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot joint P-DNN training and perplexity recovery")
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--before-mean-field", type=float, default=13.715393897577322)
    parser.add_argument("--before-n4", type=float, default=13.895319672815122)
    parser.add_argument("--before-n16", type=float, default=13.758045811201104)
    parser.add_argument("--baseline-ppl", type=float, default=11.652735)
    parser.add_argument("--replacement-count", type=int, default=4)
    return parser.parse_args()


def load_ppl(path: Path) -> float:
    return float(json.loads(path.read_text(encoding="utf-8"))["perplexity"])


def main() -> None:
    args = parse_args()
    validation = []
    for line in (args.results_dir / "train_metrics.jsonl").read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        if record["event"] == "validation":
            validation.append((int(record["step"]), float(record["perplexity"])))

    before = [args.before_mean_field, args.before_n4, args.before_n16]
    after = [
        load_ppl(args.results_dir / "best_samples0_seed0_ppl.json"),
        sum(load_ppl(args.results_dir / f"best_samples4_seed{seed}_ppl.json") for seed in range(3)) / 3,
        load_ppl(args.results_dir / "best_samples16_seed0_ppl.json"),
    ]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.2), constrained_layout=True)

    axes[0].plot([step for step, _ in validation], [ppl for _, ppl in validation], marker="o", color="#4472C4")
    axes[0].set_xlabel("Optimizer updates")
    axes[0].set_ylabel("Held-out training-corpus PPL")
    axes[0].set_title("Joint adaptation curve")

    labels = ["Mean-field", "N=4", "N=16"]
    positions = list(range(len(labels)))
    width = 0.36
    axes[1].bar([position - width / 2 for position in positions], before, width, label="Before", color="#A5A5A5")
    axes[1].bar([position + width / 2 for position in positions], after, width, label="After", color="#4C9F70")
    axes[1].axhline(args.baseline_ppl, color="#222222", linestyle="--", linewidth=1.2, label="Original Qwen")
    axes[1].set_xticks(positions, labels)
    axes[1].set_ylabel("Full WikiText-2 PPL")
    axes[1].set_title(f"{args.replacement_count} replaced FFNs: before vs. after")
    axes[1].legend()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
