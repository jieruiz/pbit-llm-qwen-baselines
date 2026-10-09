import json
from pathlib import Path
r=Path(__file__).resolve().parent/'results'
for f in ('status.json','selection.json'):
    if (r/f).exists():print(f,(r/f).read_text())
for split in ('validation','test'):
    for p in sorted((r/split).glob('*.json')):
        d=json.loads(p.read_text());print(split,p.stem,round(d['perplexity'],6),round(d['seconds'],1))
for p in (r/'jobs').glob('*.json'):
    j=json.loads(p.read_text())
    if j['status']!='complete':
        l=r/'logs'/(p.stem+'.log');lines=l.read_text().splitlines() if l.exists() else []
        print(j['status'],p.stem,lines[-1] if lines else '')
