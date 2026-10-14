"""Full decode scenarios and one-factor sensitivity; no remote jobs."""
import copy
import csv
import json
import statistics
from pathlib import Path
from system_decode import DecodeModel,load

ROOT=Path(__file__).resolve().parent


def main():
    cfg,evidence,shape=load()
    folder=ROOT/'results/system_decode'
    folder.mkdir(exist_ok=True)
    jobs=[]
    variants=('digital_bf16','digital_int8_ffn','fixed_sc_exact','dual_sc_sz')
    for context in (2048,8192):
        for v in variants:
            c=copy.deepcopy(cfg);c['context_tokens']=context
            jobs.append((f'base_{v}_{context}',v,c,'baseline',None))
    sweeps=[
        ('weight_macro.cache_MiB',[4,64,512,1024]),
        ('external_memory.bandwidth_GBs',[8,128]),
        ('weight_macro.load_bandwidth_GBs',[16,256]),
        ('weight_macro.read_pJ_per_byte',[.2,20]),
        ('weight_macro.write_pJ_per_byte',[.3,30]),
        ('weight_macro.area_mm2_per_MiB',[.1,2]),
        ('weight_macro.leakage_W_per_MiB',[.0001,.01]),
        ('weight_macro.active_tile_hold_W',[.00001,.01]),
        ('sc_engines',[1,32,64]),
        ('rng.sources',[1,128,1024,32768]),
        ('rng.source_power_W',[.00001208,.001208]),
        ('rng.always_running',[False]),
        ('samples',[256,1024]),
        ('sc_profile',['g1_case1','g1_case2']),
        ('sc_cost.core_multiplier',[.1,2]),
        ('sc_cost.dual_core_multiplier',[.1,2]),
        ('sc_cost.wait_W_per_engine',[.0001,.05]),
        ('external_memory.background_W',[.005,.5]),
        ('double_buffer',[False])]
    for parameter,values in sweeps:
        targets=('fixed_sc_exact','dual_sc_sz')
        if parameter.startswith(('weight_macro.cache','external_memory.bandwidth','weight_macro.load','weight_macro.area')):
            targets=('digital_bf16','fixed_sc_exact','dual_sc_sz')
        if parameter=='sc_cost.core_multiplier':targets=('fixed_sc_exact',)
        if parameter=='sc_cost.dual_core_multiplier':targets=('dual_sc_sz',)
        for value in values:
            for v in targets:
                c=copy.deepcopy(cfg)
                keys=parameter.split('.')
                target=c
                for key in keys[:-1]:target=target[key]
                target[keys[-1]]=value
                jobs.append((f'{v}__{parameter}__{value}',v,c,parameter,value))
    rows=[]
    for name,v,c,parameter,value in jobs:
        result=DecodeModel(c,v,evidence,shape).run()
        result['scenario_name']=name
        (folder/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
        row=dict(name=name,variant=v,context=c['context_tokens'],parameter=parameter,value=value,
            **result['metrics'],
            largest_time_module=max(result['time_ns'],key=result['time_ns'].get),
            largest_energy_component=max(result['energy_J'],key=result['energy_J'].get),
            weights_external_MiB_per_token=result['traffic'].get('external_weights_read_burst_bytes',0)/1048576/c['decode_tokens'],
            KV_external_MiB_per_token=result['traffic'].get('external_KV_read_burst_bytes',0)/1048576/c['decode_tokens'],
            rng_used_Gbit_per_token=result['bounds']['raw_rng_bits']/1e9/c['decode_tokens'])
        rows.append(row)
        print(json.dumps(dict(name=name,ms=row['mean_TPOT_ms'],mJ=row['scenario_mJ_per_token']),ensure_ascii=False),flush=True)
    (folder/'summary.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (folder/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    bases={r['variant']:r for r in rows if r['parameter']=='baseline' and r['context']==2048}
    sensitivities=[]
    for r in rows:
        if r['parameter']=='baseline':continue
        b=bases[r['variant']]
        sensitivities.append(dict(variant=r['variant'],parameter=r['parameter'],value=r['value'],
            time_ratio=r['mean_TPOT_ms']/b['mean_TPOT_ms'],
            energy_ratio=r['scenario_mJ_per_token']/b['scenario_mJ_per_token'],
            area_feasible=r['area_budget_feasible']))
    (folder/'sensitivity.json').write_text(json.dumps(sensitivities,indent=2)+'\n')
    print(json.dumps(dict(completed=len(rows),folder=str(folder)),indent=2))


if __name__=='__main__':main()
