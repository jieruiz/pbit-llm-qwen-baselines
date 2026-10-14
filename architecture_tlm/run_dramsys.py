"""CPU-only closed-loop DRAMSys comparisons; submit run.sbatch for main sweep."""
import argparse,copy,hashlib,json,os,time
from pathlib import Path
from system_decode import DecodeModel,load

ROOT=Path(__file__).resolve().parent
VARIANTS=('digital_bf16','digital_int8_ffn','fixed_sc_exact','dual_sc_sz')
JOBS=[(v,c,'NoPowerDown',32) for c in (2048,8192) for v in VARIANTS]
JOBS += [(v,2048,'Staggered',32) for v in VARIANTS]


def check_result(result,reference):
    assert result['actions']==reference['actions'],'compute operation count changed'
    for key,value in reference['traffic'].items():
        assert result['traffic'][key]==value,('traffic changed',key)
    energy=result['energy_J'];d=result['external_memory_model']
    assert not any(key in energy for key in ('external_background','external_weights','external_KV'))
    assert energy['external_DRAMPower_core']>0
    total_ns=sum(result['time_ns'].values())
    assert abs(total_ns-d['end_ns'])<=1.,'full system horizon mismatch'
    assert abs(sum(energy.values())*1000/2-result['metrics']['scenario_mJ_per_token'])<1e-7
    byte_total=sum(value for key,value in result['traffic'].items()
                   if key.startswith('external_') and key.endswith('_burst_bytes'))
    assert d['burst_count']*64==byte_total,'burst/byte mismatch'
    assert d['read_bursts']+d['write_bursts']==d['burst_count']
    assert 0<d['peak_pending']<=result['config']['external_memory']['max_pending']
    requests=sum(value for key,value in result['traffic'].items()
                   if key.startswith('external_') and key.endswith('_requests'))
    assert d['macro_requests']==requests
    assert result['quality_reference_ppl']==reference['quality_reference_ppl']
    return dict(compute_actions_unchanged=True,logical_and_burst_traffic_unchanged=True,
        byte_burst_conservation=True,full_horizon_energy=True,no_double_external_charge=True,
        finite_queue_bound=True,energy_sum=True,phase_time_sum=True,
        numerical_inference_run=False,quality_reference_only=True)


def run(index,bridge):
    variant,context,policy,pending=JOBS[index]
    name=f'{variant}_{context}_{policy}'
    folder=ROOT/'results/dramsys';folder.mkdir(parents=True,exist_ok=True)
    c,e,s=load();c['context_tokens']=context
    provenance=json.loads((ROOT/'dramsys/provenance.json').read_text())
    old=DecodeModel(c,variant,e,s).run()
    # Same nominal bandwidth isolates the detailed controller/address effects.
    matched=copy.deepcopy(c)
    matched['external_memory']['bandwidth_GBs']=provenance['peak_bandwidth_GBs']
    ref=DecodeModel(matched,variant,e,s).run()
    cfg=copy.deepcopy(c)
    cfg['external_memory'].update(backend='dramsys',bridge_executable=str(Path(bridge).resolve()),
        configuration=str(ROOT/f'dramsys/ddr4_2ch_{policy}.json'),max_pending=pending,
        output_prefix=str(folder/name))
    started=time.monotonic()
    model=DecodeModel(cfg,variant,e,s)
    try:result=model.run()
    finally:model.ext.close()
    result['implementation']='Python macro compute scheduler + real SystemC/TLM2 DRAMSys external memory'
    result['energy_complete_measured']=False
    result['scenario_cost_coverage']='DRAMPower core at full system horizon; bundled IO voltage zero, interface energy uncalibrated; controller/PHY ASIC cost, CTS/startup/host/prefill excluded'
    result['verification']=check_result(result,ref)
    result['dramsys_provenance']=dict(**provenance,
        bridge_sha256=hashlib.sha256(Path(bridge).read_bytes()).hexdigest(),
        simulation_wall_seconds=time.monotonic()-started,slurm_job_id=os.environ.get('SLURM_JOB_ID'),
        slurm_array_task_id=os.environ.get('SLURM_ARRAY_TASK_ID'))
    result['comparison']=dict(original_simplified=old['metrics'],
        bandwidth_matched_simplified=ref['metrics'],
        original_external_mJ_per_token=sum(v for k,v in old['energy_J'].items() if k.startswith('external_'))*500,
        matched_external_mJ_per_token=sum(v for k,v in ref['energy_J'].items() if k.startswith('external_'))*500,
        dramsys_external_mJ_per_token=sum(v for k,v in result['energy_J'].items() if k.startswith('external_'))*500)
    result['scenario_name']=name
    (folder/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(scenario=name,metrics=result['metrics'],wall_seconds=result['dramsys_provenance']['simulation_wall_seconds'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bridge',required=True);p.add_argument('--index',type=int,required=True)
    a=p.parse_args();run(a.index,a.bridge)
