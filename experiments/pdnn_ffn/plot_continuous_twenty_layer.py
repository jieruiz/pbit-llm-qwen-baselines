"""Scientific figures for staged 20-layer raw floating-input expansion."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

root = Path("results/continuous_raw_twenty_layer_20261001")
data = json.loads((root / "comparison.json").read_text(encoding="utf-8"))
table = {(r["mode"], r["stage"]): r for r in data["rows"]}
colors = {"sigmoid": "#777777", "continuous_raw": "#2166ac"}
names = {"sigmoid": "1-bit sigmoid input", "continuous_raw": "Raw floating input"}
stages = ("independent", "staged", "joint")
labels = ("20 local students", "10 joint + 10 local", "After 20-layer joint training")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
for mode in ("sigmoid", "continuous_raw"):
    y = [table[(mode, stage)]["n4_mean"] for stage in stages]
    errors = [table[(mode, stage)]["n4_population_std"] for stage in stages]
    axes[0].errorbar(range(3), y, yerr=errors, marker="o", capsize=4, color=colors[mode], label=names[mode])
    for i, value in enumerate(y):
        axes[0].annotate(f"{value:.2f}", (i,value), xytext=(-5 if i == 2 else 5,7),
                         ha="right" if i == 2 else "left", textcoords="offset points",
                         color=colors[mode], fontsize=8)
    r = table[(mode,"joint")]
    axes[1].errorbar(range(3), [r[f"n{n}_mean"] for n in (0,4,16)],
                     yerr=[r[f"n{n}_population_std"] for n in (0,4,16)], marker="o", capsize=4,
                     color=colors[mode], label=names[mode])
for ax in axes:
    ax.axhline(data["original_qwen_ppl"], linestyle="--", color="#b2182b", linewidth=1.3, label="Original Qwen 11.653")
    ax.set_ylabel("Full WikiText-2 perplexity")
    ax.grid(axis="y", alpha=.2)
    ax.legend(frameon=False, fontsize=8)
    ax.margins(y=.16)
axes[0].set_xticks(range(3))
axes[0].set_xticklabels(["20 local", "10 joint +\n10 local", "20 joint"])
axes[0].set_title("Twenty-layer expansion stages (N=4)")
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(["Mean-field", "N=4", "N=16"])
axes[1].set_title("Final twenty-layer joint models")
fig.tight_layout()
fig.savefig(root / "twenty_layer_comparison.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(10,4.5))
layers = data["protocol"]["layers"]
for mode in ("sigmoid", "continuous_raw"):
    ax.plot(range(len(layers)), [data["single_layers"][mode][str(l)]["n4"] for l in layers],
            marker="o", color=colors[mode], label=names[mode])
ax.axhline(data["original_qwen_ppl"], linestyle="--", color="#b2182b", label="Original Qwen")
ax.set_xticks(range(len(layers)))
ax.set_xticklabels(layers)
ax.set_xlabel("Decoder layer (one FFN replaced at a time)")
ax.set_ylabel("N=4 perplexity, seed 0")
ax.set_title("Single-layer diagnostics, including the ten newly trained layers")
ax.grid(axis="y", alpha=.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(root / "single_layer_comparison.png", dpi=180)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.5,4.5))
for mode in ("sigmoid", "continuous_raw"):
    records = [json.loads(line) for line in (root / "joint" / mode / "train_metrics.jsonl").read_text().splitlines()]
    validations = [r for r in records if r.get("event") == "validation"]
    ax.plot([r["step"] for r in validations], [r["perplexity"] for r in validations],
            marker="o", color=colors[mode], label=names[mode])
    best = data["joint_training"][mode]["best_step"]
    selected = next(r for r in validations if r["step"] == best)
    ax.scatter([best], [selected["perplexity"]], marker="*", s=160, color=colors[mode], zorder=4)
ax.set_xlabel("Twenty-layer joint optimizer updates")
ax.set_ylabel("Held-out train-tail validation perplexity")
ax.set_title("Validation checkpoint selection (stars mark selected checkpoints)")
ax.grid(alpha=.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(root / "joint_validation_curve.png", dpi=180)
plt.close(fig)
