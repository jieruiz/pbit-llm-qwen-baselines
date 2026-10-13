"""Read formal results and logs; safe on login node, no model execution."""
import json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[]
for p in sorted((ROOT/'results/formal').glob('*/*.json')):
    x=json.loads(p.read_text());m=x['metadata'];r=x['result']
    row={'variant':m['args']['variant'],'task':m['args']['task'],'seed':m['args']['seed'],
        'seconds':m['seconds'],'path':str(p),'ffn_layers':m['ffn_layers'],'attention_layers':m['attention_layers']}
    if 'perplexity' in r:row.update({k:r[k] for k in ['perplexity','scored_tokens','corpus_sha256']})
    if 'harness_result' in r:row.update(metrics=r['harness_result']['results'],dataset_rows=r['dataset_rows'])
    rows.append(row)
progress={}
for p in sorted((ROOT/'results/logs').glob('*.log')):
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-4096));s=f.read().decode(errors='replace')
    lines=[s for s in re.split('[\r\n]+',s) if s.strip()]
    progress[p.name]=lines[-2:]
queue=subprocess.check_output(['/opt/slurm/21.08.8/bin/squeue','-u','Hongjie_Zeng','-o','%.18i %.25j %.10T %.10M %.28R'],text=True)
print(json.dumps({'timestamp_utc':datetime.now(timezone.utc).isoformat(),'completed':rows,
    'progress':progress,'queue':queue},indent=2,ensure_ascii=False))
