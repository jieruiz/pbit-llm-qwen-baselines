"""Scientific comparison figure for input position-bit encoding."""
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT=Path('results/input_multibit_layer12_20260930')


def main():
    rows=json.loads((ROOT/'comparison.json').read_text())['rows']
    audit=json.loads((ROOT/'projection_audit.json').read_text())
    fig,axes=plt.subplots(1,3,figsize=(15,4.8))
    styles={'stochastic':('#1765ad','o','Stochastic rounding'),'deterministic':('#ce6b15','s','Deterministic input')}
    for mode,(color,marker,label) in styles.items():
        subset=sorted([r for r in rows if r['input_encoding']==mode],key=lambda r:r['input_bits'])
        x=[r['input_bits'] for r in subset]
        axes[0].errorbar(x,[r['n4_ppl_mean'] for r in subset],yerr=[r['n4_ppl_population_sd'] for r in subset],color=color,marker=marker,label=label,capsize=3)
        probes=sorted([r for r in audit['rows'] if r['mode']==mode],key=lambda r:r['bits'])
        axes[1].plot(x,[r['teacher_projection_n4_nmse'] for r in probes],color=color,marker=marker,label=label)
        axes[2].plot(x,[r['bias_nmse'] for r in subset],color=color,marker=marker,label=label+' bias')
        axes[2].plot(x,[r['single_path_variance_nmse']/4 for r in subset],color=color,marker=marker,linestyle='--',label=label+' variance/4')
    for name,color,label in [('sigmoid_k1','#666666','Original sigmoid input'),('continuous','#269b62','Clipped continuous input')]:
        row=next(r for r in rows if r['name']==name)
        axes[0].axhline(row['n4_ppl_mean'],color=color,linestyle=':',label=label)
    axes[0].set_title('Full-model quality, N=4')
    axes[0].set_ylabel('WikiText-2 perplexity (lower is better)')
    axes[1].set_title('Frozen teacher gate/up projections')
    axes[1].set_ylabel('N=4 linear projection NMSE')
    axes[1].set_yscale('log')
    axes[2].set_title('Trained FFN output error')
    axes[2].set_ylabel('Normalized MSE component')
    for axis in axes:
        axis.set_xlabel('Input bits per feature, K')
        axis.set_xticks([1,2,4])
        axis.grid(alpha=.22)
        axis.legend(fontsize=7.5)
    fig.suptitle('Qwen2.5-0.5B layer 12: tied-weight input encoding, fixed width 4864')
    fig.tight_layout()
    fig.savefig(ROOT/'input_multibit.png',dpi=180,bbox_inches='tight')
    fig.savefig(ROOT/'input_multibit.pdf',bbox_inches='tight')


if __name__=='__main__':
    main()
