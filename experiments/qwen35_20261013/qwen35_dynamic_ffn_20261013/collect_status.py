"""Read PPL and profile status only."""
import json,re,subprocess
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[]
for p in sorted((ROOT/'results/formal').glob('*.json')):
    x=json.loads(p.read_text());rows.append({'file':p.name,'args':x['metadata']['args'],
        'perplexity':x['result']['perplexity'],'scored_tokens':x['result']['scored_tokens'],'seconds':x['metadata']['seconds']})
profile=ROOT/'results/deltanet_gates.json'
progress={}
for p in sorted((ROOT/'results/logs').glob('*.log')):
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4096));s=f.read().decode(errors='replace')
    lines=[s for s in re.split('[\r\n]+',s) if s.strip()];progress[p.name]=lines[-2:]
queue=subprocess.check_output(['/opt/slurm/21.08.8/bin/squeue','-u','Hongjie_Zeng','-o','%.18i %.25j %.10T %.10M %.28R'],text=True)
print(json.dumps({'timestamp_utc':datetime.now(timezone.utc).isoformat(),'completed':rows,
    'gate_profile':json.loads(profile.read_text())['aggregate'] if profile.exists() else 'queued/running',
    'progress':progress,'queue':queue},indent=2))
