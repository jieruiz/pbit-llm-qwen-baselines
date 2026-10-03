"""Plot matched ten-layer input ablation and per-layer diagnostics."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

root = Path("results/continuous_raw_ten_layer_20261001")
data = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
rows = {(r["mode"], r["stage"]): r for r in data["rows"]}
colors = {"sigmoid": "#777777", "continuous_raw": "#2166ac"}
names = {"sigmoid": "1-bit sigmoid input", "continuous_raw": "Raw floating input"}
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True)
for ax, stage in zip(axes, ("independent", "joint")):
    for mode in ("sigmoid", "continuous_raw"):
        row = rows[(mode, stage)]
        values = [row[f"n{count}_mean"] for count in (0, 4, 16)]
        errors = [row[f"n{count}_population_std"] for count in (0, 4, 16)]
        ax.errorbar(range(3), values, yerr=errors, marker="o", capsize=4,
                    label=names[mode], color=colors[mode])
        ax.annotate(f"{values[1]:.3f}", (1, values[1]), xytext=(7, 8), textcoords="offset points",
                    color=colors[mode], fontsize=9)
    ax.axhline(data["original_qwen_ppl"], linestyle="--", color="#b2182b", linewidth=1.3,
               label=f"Original Qwen {data['original_qwen_ppl']:.3f}")
    ax.set_xticks(range(3))
    ax.set_xticklabels(["Mean-field", "N=4", "N=16"])
    ax.set_title("Independent composition" if stage == "independent" else "After 1,000 joint updates")
    ax.grid(axis="y", alpha=.2)
axes[0].set_ylabel("Full WikiText-2 perplexity (lower is better)")
upper = max(row[f"n{count}_mean"] for row in rows.values() for count in (0, 4, 16)) + 0.8
axes[0].set_ylim(data["original_qwen_ppl"] - 0.55, upper)
axes[1].legend(frameon=False, fontsize=8)
fig.suptitle("Ten FFN replacements: matched width and training budget")
fig.tight_layout()
fig.savefig(root / "ten_layer_comparison.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4))
layers = [7,8,9,10,11,12,13,14,15,18]
for mode in ("sigmoid", "continuous_raw"):
    values = [data["single_layers"][mode][str(layer)]["n4"] for layer in layers]
    ax.plot(range(10), values, marker="o", color=colors[mode], label=names[mode])
ax.axhline(data["original_qwen_ppl"], linestyle="--", color="#b2182b", label="Original Qwen")
ax.set_xticks(range(10))
ax.set_xticklabels(layers)
ax.set_xlabel("Decoder layer (only one FFN replaced at a time)")
ax.set_ylabel("N=4 perplexity, seed 0")
ax.set_title("Single-layer diagnostics")
ax.grid(axis="y", alpha=.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(root / "single_layer_comparison.png", dpi=180)
plt.close(fig)
