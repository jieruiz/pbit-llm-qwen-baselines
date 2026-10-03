import json
from pathlib import Path
import matplotlib.pyplot as plt

root=Path("results/multithreshold_and_twenty_layer_20261002")
data=json.loads((root/"comparison.json").read_text())
rows={r["stage"]:r for r in data["rows"]}
fig,ax=plt.subplots(figsize=(8.5,4.8))
counts=(0,1,2,4,16)
for stage,color,label in (("independent","#777777","20 independent students"),("joint","#2166ac","After 20-layer joint training")):
    row=rows[stage]
    ax.errorbar(range(5),[row[f"n{n}_mean"] for n in counts],
        yerr=[row[f"n{n}_std"] for n in counts],marker="o",capsize=4,color=color,label=label)
ax.axhline(data["original_qwen_ppl"],color="#b2182b",linestyle="--",label="Original Qwen")
ax.set_xticks(range(5),["Mean-field","N=1","N=2","N=4","N=16"])
ax.set_ylabel("Full WikiText-2 perplexity")
ax.set_title("Twenty-layer K=4 AND-bank replacement")
ax.legend(frameon=False)
ax.grid(axis="y",alpha=.2)
ax.margins(y=.15)
fig.tight_layout()
fig.savefig(root/"twenty_layer_quality.png",dpi=180)
plt.close(fig)

records=[json.loads(line) for line in (root/"joint"/"train_metrics.jsonl").read_text().splitlines()]
records=[r for r in records if r["event"]=="validation"]
fig,ax=plt.subplots(figsize=(8,4.4))
ax.plot([r["step"] for r in records],[r["perplexity"] for r in records],marker="o",color="#2166ac")
best=data["joint_training"]["best_step"]
chosen=next(r for r in records if r["step"]==best)
ax.scatter([best],[chosen["perplexity"]],marker="*",s=170,color="#b2182b",label=f"Selected step {best}",zorder=5)
ax.set_xlabel("Joint optimizer updates")
ax.set_ylabel("Held-out train-tail validation perplexity")
ax.set_title("Validation selection; test data never used")
ax.grid(alpha=.2)
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(root/"joint_validation_curve.png",dpi=180)
plt.close(fig)
