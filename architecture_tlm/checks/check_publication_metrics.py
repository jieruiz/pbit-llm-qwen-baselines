"""Ensure portable-path edits preserve the 96 original scenario metrics."""
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
reference=json.loads((ROOT/'configs/publication_reference_metrics.json').read_text(encoding='utf-8'))
current=json.loads((ROOT/'results/system_decode/summary.json').read_text(encoding='utf-8'))
original={r['name']:r for r in reference['rows']}
assert len(current)==len(original)==96
fields=reference['checked_fields']
for row in current:
    expected=original[row['name']]
    for field in fields:
        a,b=row[field],expected[field]
        if a is None or b is None:assert a is b,(row['name'],field)
        elif isinstance(a,bool):assert a==b,(row['name'],field)
        else:assert math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12),(row['name'],field,a,b)
result=dict(status='PASS',scenarios=96,checked_fields=fields,
            description='Portable publication version preserves pre-publication numerical metrics')
(ROOT/'results/system_decode/publication_check.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
