import hashlib
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent
F=ROOT/'results/system_decode'
rows=json.loads((F/'summary.json').read_text())
shape=json.loads((ROOT/'configs/model_config.json').read_text())
source=hashlib.sha256((ROOT/'system_decode.py').read_bytes()).hexdigest()
contract=hashlib.sha256((ROOT/'configs/evidence_contract.json').read_bytes()).hexdigest()
assert len(rows)==96 and len({r['name'] for r in rows})==96
for row in rows:
    d=json.loads((F/(row['name']+'.json')).read_text())
    c=d['config'];n=c['decode_tokens'];h=shape['hidden_size'];m=shape['intermediate_size'];layers=shape['num_hidden_layers']
    assert d['source_hashes']['system_decode.py']==source
    assert d['source_hashes']['evidence_contract.json']==contract
    total_seconds=d['events'][-1]['end_ns']*1e-9
    assert math.isclose(sum(d['time_ns'].values())*1e-9,total_seconds,rel_tol=1e-10)
    assert math.isclose(sum(d['energy_J'].values())*1000/n,row['scenario_mJ_per_token'],rel_tol=1e-12)
    assert d['weight_cache']['peak_bytes']<=int(c['weight_macro']['cache_MiB']*1048576)
    assert all(v>=0 for v in d['energy_J'].values())
    assert not d['energy_complete_measured']
    if row['variant'].startswith(('fixed_sc','dual_sc')):
        trials=n*3*h*m*layers*c['samples']
        assert d['actions']['sc_product_trials']==trials
        bits=n*(h+m)*layers*c['samples']*c['rng']['input_bits_per_decision']
        if row['variant']=='dual_sc_sz':bits+=trials*c['rng']['weight_bits_per_decision']
        assert math.isclose(bits,d['bounds']['raw_rng_bits'],rel_tol=1e-10)
        assert d['bounds']['raw_rng_bits']<=d['bounds']['source_nominal_capacity_bits']*(1+1e-9)
    else:
        assert d['bounds']['raw_rng_bits']==0
    expected_KV=sum(c['context_tokens']+t for t in range(n))*layers*2*shape['num_key_value_heads']*(h//shape['num_attention_heads'])*2
    assert d['traffic']['external_KV_read_logical_bytes']==expected_KV
report=dict(status='PASS',scenario_count=len(rows),unit_checks=11,
    checked=['source hashes','energy component sum','phase time sum','LRU capacity','exact shape trial counts',
             'RNG sharing/demand and finite-rate capacity','GQA dense KV bytes','nonnegative costs','measured-energy flag'],
    shape=shape,model_config_sha256=hashlib.sha256((ROOT/'configs/model_config.json').read_bytes()).hexdigest(),
    default_config_sha256=hashlib.sha256((ROOT/'configs/system_decode_default.json').read_bytes()).hexdigest(),
    system_decode_sha256=source,cost_contract_sha256=contract,
    statement='Parameter-model consistency audit; not verification against silicon, SystemC or numerical full-model inference')
(F/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(status='PASS',scenarios=len(rows),unit_checks=11)))
