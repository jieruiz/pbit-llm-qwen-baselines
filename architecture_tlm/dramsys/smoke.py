"""Integration checks against the built real SystemC library, CPU only."""
import argparse,copy,json,sys
from pathlib import Path
from collections import defaultdict
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dramsys_backend import DRAMSysExternal,AddressSpace
from system_decode import DecodeModel,load
root=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--bridge',required=True);a=p.parse_args()
c,e,s=load();folder=root/'results/dramsys';folder.mkdir(parents=True,exist_ok=True)
def backend(name,policy='NoPowerDown'):
 cfg=dict(bridge_executable=a.bridge,configuration=str(root/f'dramsys/ddr4_2ch_{policy}.json'),
    max_pending=8,output_prefix=str(folder/name))
 return DRAMSysExternal(cfg,defaultdict(float),defaultdict(int),AddressSpace(s,'digital_bf16',2048,2))
energy=[];results=[]
for horizon in (100_000,1_000_000):
 ext=backend(f'smoke_horizon_{horizon}')
 try:
  end=ext.access(0,4096,'weights',False,address=65536)
  assert end>0
  ext.access(end,512,'KV',True,address=65536+8192)
  d=ext.finish(horizon)
  assert d['burst_count']==72 and d['read_bursts']==64 and d['write_bursts']==8
  assert d['peak_pending']<=8
  assert abs(d['end_ns']-horizon)<.001
  total=sum(ext.energy.values());assert total>0
  energy.append(total);results.append(dict(horizon_ns=horizon,energy_J=total,diagnostic=d))
 finally:ext.close()
assert energy[1]>5*energy[0],'idle/refresh energy not counted through end horizon'
c['context_tokens']=128;c['decode_tokens']=2
c['weight_macro']['cache_MiB']=64
s=copy.deepcopy(s);s['num_hidden_layers']=1;s['vocab_size']=1024
base=DecodeModel(c,'digital_bf16',e,s).run()
c['external_memory'].update(backend='dramsys',bridge_executable=a.bridge,
    configuration=str(root/'dramsys/ddr4_2ch_NoPowerDown.json'),max_pending=32,
    output_prefix=str(folder/'smoke_small_decode'))
model=DecodeModel(c,'digital_bf16',e,s)
try:r=model.run()
finally:model.ext.close()
assert r['actions']==base['actions']
assert r['traffic']==base['traffic']
assert r['metrics']['subsequent_decode_ms']<r['metrics']['first_decode_ms']
assert 'external_background' not in r['energy_J']
assert abs(sum(r['time_ns'].values())-r['external_memory_model']['end_ns'])<1
report=dict(passed=True,checks=['real_4phase_TLM_read_write','burst_counts','finite_outstanding',
 'full_horizon_energy','background_scales_with_idle_duration','closed_loop_24_shape_scheduler_smallcase',
 'warm_weight_cache_reuse','unchanged_compute_and_traffic','no_external_energy_double_count'],
 horizon_runs=results,small_decode=dict(metrics=r['metrics'],external_memory_model=r['external_memory_model']))
(folder/'integration_smoke.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(dict(passed=True,horizon_energy_J=energy,small_decode=r['metrics'])))
