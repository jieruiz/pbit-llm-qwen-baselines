"""Plot the four-layer sigmoid-input and stochastic-K4 comparison."""
import json
from pathlib import Path

import matplotlib.pyplot as plt


root = Path("results/input_multibit_four_layer_9_12_15_18_20260930")
data = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
base = data["base_qwen_ppl"]
sigmoid = data["sigmoid_input_baseline"]
k4 = data["stochastic_k4"]
labels = ["Sigmoid\nindependent", "K=4\nindependent", "Sigmoid\njoint", "K=4\njoint"]
means = [sigmoid["independent"]["n4_mean"], k4["independent"]["n4_mean"],
         sigmoid["joint"]["n4_mean"], k4["joint"]["n4_mean"]]
errors = [sigmoid["independent"]["n4_population_std"], k4["independent"]["n4_population_std"],
          sigmoid["joint"]["n4_population_std"], k4["joint"]["n4_population_std"]]
colors = ["#8c8c8c", "#3274a1", "#8c8c8c", "#3274a1"]
fig, ax = plt.subplots(figsize=(8.2, 4.8))
bars = ax.bar(labels, means, yerr=errors, capsize=4, color=colors, edgecolor="white")
ax.axhline(base, color="#c44e52", linestyle="--", linewidth=1.6, label=f"Original Qwen: {base:.3f}")
ax.set_ylabel("WikiText-2 perplexity (N=4)")
ax.set_title("Four FFN replacements: layers 9, 12, 15, 18")
ax.set_ylim(11.4, 14.9)
ax.grid(axis="y", alpha=0.25)
ax.legend(frameon=False)
for bar, value in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2, value + 0.06, f"{value:.3f}", ha="center", va="bottom")
fig.tight_layout()
fig.savefig(root / "four_layer_n4_comparison.png", dpi=180)
plt.close(fig)
