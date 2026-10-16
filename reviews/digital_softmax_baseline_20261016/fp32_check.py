"""Check the same deterministic approximation with FP32 input and output."""
from pathlib import Path
import csv
import hashlib
import json
import math
import random

ROOT=Path(__file__).resolve().parent
source=(ROOT/'evaluate.py').read_text(encoding='utf-8').split('\nrows=[]\n',1)[0]
scope={'__file__':str(ROOT/'evaluate.py'),'__name__':'numerical_functions'}
exec(compile(source,str(ROOT/'evaluate.py'),'exec'),scope)
# Both generate() input rounding and candidate() output rounding resolve this
# global at call time. Replace BF16 rounding with IEEE FP32 rounding for both.
scope['bf16']=scope['fp32']
rows=[]
for seed in [0,1,2]:
    rng=random.Random(seed)
    for kind in ['uniform','normal_std1','normal_std4','one_peak','near_ties','large_offset','masked','tail_boundary']:
        for trial in range(8):
            x=scope['generate'](kind,rng);reference=scope['reference'](x);out=scope['candidate'](x)
            assert all(math.isfinite(v) and v>=0 for v in out)
            assert all(out[i]==0 for i,v in enumerate(x) if v==-math.inf)
            assert abs(sum(out)-1)<1e-5
            rows.append(dict(seed=seed,profile=kind,trial=trial,
                max_abs_vs_fp64=max(abs(a-b) for a,b in zip(out,reference)),
                half_L1_vs_fp64=sum(abs(a-b) for a,b in zip(out,reference))/2,
                row_sum_error=abs(sum(out)-1)))
with (ROOT/'fp32_rows.csv').open('w',encoding='utf-8',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=dict(status='synthetic numerical prototype; not Qwen PPL, RTL or hardware PPA',
 input_output='FP32',rows=len(rows),length=2048,seeds=[0,1,2],
 mean_half_L1=sum(r['half_L1_vs_fp64'] for r in rows)/len(rows),
 worst_half_L1=max(r['half_L1_vs_fp64'] for r in rows),
 worst_max_abs_error=max(r['max_abs_vs_fp64'] for r in rows),
 worst_row_sum_error=max(r['row_sum_error'] for r in rows),
 dependencies=dict(evaluate_sha256=hashlib.sha256((ROOT/'evaluate.py').read_bytes()).hexdigest()),
 implementation='Q1.24 exp LUT/interpolation and FP32 accumulation/reciprocal/multiply; delta<-16 flushed to zero. Reference is FP64 stable softmax on the identical FP32 input.')
(ROOT/'fp32_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
