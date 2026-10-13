"""Read result files and Slurm status only, no GPU execution."""
import json, re, subprocess
from datetime import datetime, timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[]
for p in sorted((ROOT/'results').glob('*.json')):
    x=json.loads(p.read_text());result=x['result']
    row={'task':p.stem,'seconds':x['metadata'].get('seconds'),'path':str(p)}
    if 'perplexity' in result:row.update({k:result[k] for k in ['perplexity','scored_tokens','corpus_tokens','corpus_sha256']})
    if 'harness_result' in result:
        row.update(metrics=result['harness_result']['results'],n_samples=result['harness_result'].get('n-samples'),dataset_rows=result['dataset_rows'])
    if 'timing' in result:row.update(result)
    if p.stem=='smoke': row['cache_check']=result['cache_check']
    rows.append(row)
progress={}
for p in sorted((ROOT/'results/logs').glob('baseline-*.log')):
    with p.open('rb') as f:
        f.seek(max(0,p.stat().st_size-8192)); content=f.read().decode(errors='replace')
    lines=[v for v in re.split(r'[\r\n]+',content) if v.strip()]
    progress[p.name]=lines[-2:]
queue=subprocess.check_output(['/opt/slurm/21.08.8/bin/squeue','-u','Hongjie_Zeng','-o','%.18i %.22j %.10T %.10M %.25R'],text=True)
output={'timestamp_utc':datetime.now(timezone.utc).isoformat(),'completed':rows,'progress':progress,'queue':queue}
print(json.dumps(output,indent=2,ensure_ascii=False))
