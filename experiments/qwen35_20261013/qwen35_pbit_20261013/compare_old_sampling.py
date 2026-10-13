import json,statistics,math
from pathlib import Path
ROOT=Path(__file__).resolve().parent
old=json.loads((ROOT.parent/'pbit_dual_or_ffn_20261012/results/test/fixed_exact_T4096_s0.json').read_text())
new=json.loads((ROOT/'formal/ffn_exact/wikitext2_seed0.json').read_text())
out={}
for label,x,width,hidden,key in [('old_fixed',old,896,4864,'diagnostics'),('new_fixed',new['metadata'],1024,3584,'ffn_diagnostics')]:
    rows=x[key].values();events=[];clips=[]
    for r in rows:
        num_product=r['x_values']/width*hidden
        events.append(r['expected_product_events']/num_product)
        clips.append([r['x_fullscale_fraction'],r['g_fullscale_fraction'],r['u_fullscale_fraction']])
    out[label]={'mean_expected_events_per_hidden_value':statistics.mean(events),
        'layer_min_events':min(events),'layer_max_events':max(events),
        'average_clip_fractions':list(map(statistics.mean,zip(*clips)))}
out['ppl_comparison']={'old_baseline':11.652735,'old_fixed4096':old['perplexity'],
    'old_relative_increase':old['perplexity']/11.652735-1,
    'new_baseline':15.661097804818592,'new_fixed4096_mean':17.07553097054799,
    'new_relative_increase':17.07553097054799/15.661097804818592-1}
(ROOT/'sampling_comparison.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
