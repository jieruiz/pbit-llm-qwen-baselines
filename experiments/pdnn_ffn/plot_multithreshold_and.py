"""Measured language-model quality and conditional bias/variance figures."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

root=Path("results/multithreshold_and_layer12_20261002")
data=json.loads((root/"comparison.json").read_text())
table={(r["bits"],r["stage"]):r for r in data["rows"]}
bits=(1,2,4)
colors=("#777777","#d95f02","#2166ac")
fig,axes=plt.subplots(1,2,figsize=(11,5.5))
for k,color in zip(bits,colors):
    r=table[(k,"sampled")]
    axes[0].errorbar(range(3),[r[f"n{n}_mean"] for n in (0,4,16)],
        yerr=[r[f"n{n}_std"] for n in (0,4,16)],marker="o",capsize=4,color=color,label=f"K=L={k}")
axes[0].axhline(data["original_qwen_ppl"],linestyle="--",color="#b2182b",label="Original Qwen")
axes[0].axhline(data["historical_raw_serial_n4_seed0"],linestyle=":",color="#4daf4a",label="Old serial N=4 (unmatched)")
axes[0].set_xticks(range(3),["Mean-field","N=4","N=16"])
axes[0].set_ylabel("Full WikiText-2 perplexity")
axes[0].set_title("Final trained models: one FFN replaced")
axes[0].legend(frameon=False,fontsize=8,loc="upper center",bbox_to_anchor=(.5,-.1),ncol=2)
bias=[table[(k,"sampled")]["moments"]["bias_nmse"] for k in bits]
variance=[table[(k,"sampled")]["moments"]["single_path_variance_nmse"]/4 for k in bits]
axes[1].bar(range(3),bias,color="#2166ac",label="Squared mean bias")
axes[1].bar(range(3),variance,bottom=bias,color="#fdae61",label="Single-path variance / 4")
axes[1].set_xticks(range(3),[f"K=L={k}" for k in bits])
axes[1].set_ylabel("Expected N=4 output NMSE")
axes[1].set_title("Fixed teacher inputs: analytic decomposition")
axes[1].legend(frameon=False,fontsize=8)
for ax in axes:
    ax.grid(axis="y",alpha=.2)
    ax.set_axisbelow(True)
fig.tight_layout()
fig.savefig(root/"final_comparison.png",dpi=180)
plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.6))
for k,color in zip(bits,colors):
    records=[json.loads(line) for line in (root/f"k{k}"/"train_metrics.jsonl").read_text().splitlines()]
    records=[r for r in records if r["event"]=="validation"]
    steps=[r["step"] for r in records]
    for ax,key in zip(axes,("bias_nmse","single_path_variance_nmse")):
        ax.plot(steps,[r[key] for r in records],marker="o",color=color,label=f"K=L={k}")
for ax,label in zip(axes,("Squared mean bias NMSE","Single-path variance NMSE")):
    ax.axvline(2000,color="black",linestyle="--",linewidth=1,label="Switch to N=4 training")
    ax.set_xlabel("Local distillation updates")
    ax.set_ylabel(label)
    ax.set_yscale("log")
    ax.grid(alpha=.2)
    ax.legend(frameon=False,fontsize=8)
fig.suptitle("Mean-field fitting and sampled training trade off bias and variance")
fig.tight_layout()
fig.savefig(root/"training_bias_variance.png",dpi=180)
plt.close(fig)

fig,ax=plt.subplots(figsize=(8,5))
for k,color in zip(bits,colors):
    r=table[(k,"sampled")]
    counts=(1,2,4,16)
    x=[n*(k*k+2*k) for n in counts]
    ax.errorbar(x,[r[f"n{n}_mean"] for n in counts],
        yerr=[r[f"n{n}_std"] for n in counts],marker="o",capsize=3,color=color,label=f"K=L={k}")
    for n,cost in zip(counts,x):
        ax.annotate(f"N={n}",(cost,r[f"n{n}_mean"]),xytext=(3,6),textcoords="offset points",fontsize=8,color=color)
ax.axhline(data["original_qwen_ppl"],linestyle="--",color="#b2182b",label="Original Qwen")
ax.set_xscale("log",base=2)
ax.set_xlabel("Binary readout terms per token: N (K² + 2K)\nExcludes two floating input projections; not a hardware energy estimate")
ax.set_ylabel("Full WikiText-2 perplexity")
ax.set_title("Sample-budget sweep of fixed final checkpoints")
ax.grid(alpha=.2)
ax.margins(x=.15,y=.15)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(root/"sample_cost_quality.png",dpi=180)
plt.close(fig)
