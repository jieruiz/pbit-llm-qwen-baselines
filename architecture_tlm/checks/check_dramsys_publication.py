"""Verify that published source/configuration bytes match actual run hashes."""
import hashlib,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
from run_dramsys import JOBS
source=hashlib.sha256((root/'system_decode.py').read_bytes()).hexdigest()
contract=hashlib.sha256((root/'configs/evidence_contract.json').read_bytes()).hexdigest()
for variant,context,policy,pending in JOBS:
 name=f'{variant}_{context}_{policy}'
 r=json.loads((root/'results/dramsys'/f'{name}.json').read_text())
 assert r['source_hashes']['system_decode.py']==source,name
 assert r['source_hashes']['evidence_contract.json']==contract,name
 config=root/f'dramsys/ddr4_2ch_{policy}.json'
 assert hashlib.sha256(config.read_bytes()).hexdigest()==r['external_memory_model']['configuration_sha256'],name
 assert r['external_memory_model']['peak_pending']<=pending
 assert r['config']['context_tokens']==context
 assert r['config']['decode_tokens']==2
 assert r['energy_complete_measured'] is False
 if not r.get('publication',{}).get('raw_artifacts_omitted'):
  assert (root/'results/dramsys'/f'{name}.bridge.log').exists()
print(json.dumps(dict(passed=True,scenarios=len(JOBS),checks=['published source hashes','raw memory config hashes','finite queues','scope'],raw_logs='retained locally, omitted from public aggregate export')))
