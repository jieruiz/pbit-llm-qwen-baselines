"""Sensitivity estimates, not synthesis, device measurements, or a CTMC simulator.

Run: python -X utf8 estimate.py
All time mappings require measured conditional transition rates. Unit-rate
heat-bath tau is NOT automatically a device dwell/autocorrelation time.
"""
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parent
P = dict(categories=2048, samples=512, burn_process_units=16,
         spacing_process_units=4, head_dim=64, query_heads=14,
         kv_heads=2, layers=24, bias_bits=16,
         score_bytes=4, value_bytes=2,
         digital_mac_pJ=2, digital_add_pJ=1,
         digital_vector_pJ=1, digital_nonlinear_pJ=10,
         scratch_read_pJ_byte=2, scratch_write_pJ_byte=3,
         scratch_GBs=64, frequency_Hz=100_000_000,
         digital_softmax_lanes=16, digital_softmax_passes=3,
         digital_fixed_cycles=16)
M=P['categories']; N=M-1; S=P['samples']; D=P['head_dim']
H=P['burn_process_units']+P['spacing_process_units']*S
idx_bits=(M-1).bit_length()

def write_csv(name, rows):
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

core=[]
for tau in [0.1,1,10,100,1000]:
    for cell_uW in [0.1,1,4.9,10,50,100]:
        t_ns=H*tau
        core.append(dict(tau_ns=tau,cell_uW=cell_uW,array_mW=N*cell_uW/1000,
                         process_us=t_ns/1000,core_nJ=N*cell_uW*t_ns*1e-6))
write_csv('core_sensitivity.csv',core)
digital_compute_nJ=M*(6*P['digital_vector_pJ']+P['digital_nonlinear_pJ'])/1000
# An explicitly unfused 3-pass example: read scores in max/exp passes,
# write exp buffer, read exp buffer, write probabilities. Fused streaming
# hardware can remove traffic; this example is not a lower bound.
digital_scratch_nJ=M*(12*P['scratch_read_pJ_byte']+8*P['scratch_write_pJ_byte'])/1000
digital_service_us=(3*math.ceil(M/P['digital_softmax_lanes'])+P['digital_fixed_cycles'])/P['frequency_Hz']*1e6
digital_scratch_us=M*20/(P['scratch_GBs']*1e3)
thresholds=[]
for tau in [1,10,100]:
    for budget in [1,10,digital_compute_nJ,100]:
        for overhead in [0,5]:
            thresholds.append(dict(tau_ns=tau,digital_budget_nJ=budget,
                                   excluded_overhead_nJ=overhead,
                                   maximum_cell_uW=max(0,budget-overhead)*1e6/(N*H*tau),
                                   feasible_before_cell_cost=budget>overhead))
write_csv('energy_break_even.csv',thresholds)
links=[]
for Gbps in [1,16,64]:
    links.append(dict(link_Gbps=Gbps,bias_setup_us=N*P['bias_bits']/Gbps/1000,
                      packed_index_output_us=S*idx_bits/Gbps/1000,
                      raw_array_output_us=S*N/Gbps/1000))
write_csv('interface_cost.csv',links)

def unique(draws): return M*(-math.expm1(draws*math.log1p(-1/M)))
g=P['query_heads']//P['kv_heads']
pv=[]
for mode in ['single_head_no_dedup','gqa_ideal_union_cache','gqa_no_cross_head_cache']:
    dense_bytes=M*D*P['value_bytes']/(g if mode.startswith('gqa') else 1)
    sampled_bytes=(unique(S*g)/g if mode=='gqa_ideal_union_cache' else S)*D*P['value_bytes']
    dense_e=(M*D*P['digital_mac_pJ']+dense_bytes*P['scratch_read_pJ_byte'])/1000
    sample_e=(S*D*P['digital_add_pJ']+sampled_bytes*P['scratch_read_pJ_byte'])/1000
    pv.append(dict(mode=mode,dense_V_bytes_per_query_head=dense_bytes,
                   sample_V_bytes_per_query_head=sampled_bytes,
                   dense_PV_nJ=dense_e,sampled_PV_nJ=sample_e,
                   available_extra_sampler_nJ=dense_e-sample_e))
write_csv('pv_scenarios.csv',pv)
summary=dict(status='analytical_sensitivity_only_no_RTL_no_silicon_no_new_PPL',
 parameters=P,physical_pbits=N,process_horizon=H,index_bits=idx_bits,
 digital_example=dict(compute_nJ=digital_compute_nJ,scratch_nJ=digital_scratch_nJ,
                      dynamic_total_nJ=digital_compute_nJ+digital_scratch_nJ,
                      service_us=digital_service_us,scratch_us=digital_scratch_us,
                      fully_overlapped_us=max(digital_service_us,digital_scratch_us),
                      serialized_us=digital_service_us+digital_scratch_us),
 pv_scenarios=pv,
 area_anchor_ITA_softmax_mm2=0.173*0.033,
 power_anchor_ITA_softmax_mW=60.5*0.014,
 hypothetical_all_in_cell_area_budget_um2=0.173*0.033*1e6/N,
 diagnostic_uniform_zero_dwell_in_tau=2/N,
 finite_lambda_approx_1pct_invalid=math.log(N/(2*0.01)),
 note='ITA is INT8, 22nm post-layout estimate with different workload; anchors are NOT a scaled 2048-element Qwen baseline. Area budget excludes all peripherals and is only a yardstick.')
assert H==2064 and N==2047 and idx_bits==11
assert math.isclose(next(x for x in core if x['tau_ns']==1 and x['cell_uW']==1)['core_nJ'],4.225008)
assert math.isclose(digital_compute_nJ,32.768)
assert all(r['dense_V_bytes_per_query_head']>0 and r['sample_V_bytes_per_query_head']>0 for r in pv)
(ROOT/'metrics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

files=['estimate.py','core_sensitivity.csv','energy_break_even.csv','interface_cost.csv','pv_scenarios.csv','metrics.json']
(ROOT/'checksums.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},indent=2)+'\n',encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
