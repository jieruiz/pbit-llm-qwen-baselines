"""Digital softmax numerical checks and clearly scoped PPA extrapolations.

No hardware timing/power is measured by this script. Numerical checks use
synthetic BF16 rows, not Qwen logits or model perplexity.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
import random
import struct

ROOT=Path(__file__).resolve().parent
M=2048

def fp32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def bf16(x):
    if not math.isfinite(x):return x
    bits=struct.unpack('<I',struct.pack('<f',x))[0]
    bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000
    return struct.unpack('<f',struct.pack('<I',bits))[0]

Q=1<<24; F=1<<10
LUT=[round(math.exp(-j/16)*Q) for j in range(257)]
def exp_lut(delta):
    if delta==-math.inf or delta < -16:return 0.0
    y=-delta*16;j=int(y)
    frac=round((y-j)*F)
    if frac==F:j+=1;frac=0
    if j>=256:return LUT[256]/Q
    return ((LUT[j]*(F-frac)+LUT[j+1]*frac+F//2)//F)/Q

def reference(row):
    maximum=max(row)
    terms=[math.exp(x-maximum) if math.isfinite(x) else 0 for x in row]
    denom=math.fsum(terms)
    return [v/denom for v in terms]

def candidate(row):
    maximum=max(row)
    terms=[exp_lut(fp32(x-maximum)) for x in row]
    lanes=[0.0]*16
    for i,v in enumerate(terms):lanes[i%16]=fp32(lanes[i%16]+v)
    while len(lanes)>1:lanes=[fp32(lanes[i]+lanes[i+1]) for i in range(0,len(lanes),2)]
    reciprocal=fp32(1/lanes[0])
    return [bf16(fp32(v*reciprocal)) for v in terms]

def generate(kind,rng):
    if kind=='uniform':row=[-3.0]*M
    elif kind=='normal_std1':row=[rng.gauss(0,1) for _ in range(M)]
    elif kind=='normal_std4':row=[rng.gauss(0,4) for _ in range(M)]
    elif kind=='one_peak':
        row=[rng.gauss(0,1) for _ in range(M)];row[rng.randrange(M)]=16.0
    elif kind=='near_ties':row=[rng.uniform(-0.02,0.02) for _ in range(M)]
    elif kind=='large_offset':row=[1000+rng.gauss(0,4) for _ in range(M)]
    elif kind=='masked':
        valid=rng.randint(1,M);row=[rng.gauss(0,3) if i<valid else -math.inf for i in range(M)]
    elif kind=='tail_boundary':row=[0.0]+[-16+rng.uniform(-0.2,0.2) for _ in range(M-1)]
    else:raise ValueError(kind)
    return [bf16(x) for x in row]

def write_csv(name,rows):
    with (ROOT/name).open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

rows=[]
for seed in [0,1,2]:
    rng=random.Random(seed)
    for kind in ['uniform','normal_std1','normal_std4','one_peak','near_ties','large_offset','masked','tail_boundary']:
        for trial in range(8):
            row=generate(kind,rng);ref=reference(row);ref_bf=[bf16(v) for v in ref];out=candidate(row)
            assert all(math.isfinite(x) and x>=0 for x in out)
            assert all(out[i]==0 for i,x in enumerate(row) if x==-math.inf)
            assert math.isclose(sum(out),1,abs_tol=0.005)
            rows.append(dict(seed=seed,profile=kind,trial=trial,
                max_abs_vs_fp64=max(abs(a-b) for a,b in zip(out,ref)),
                half_L1_vs_fp64=sum(abs(a-b) for a,b in zip(out,ref))/2,
                max_abs_vs_rounded_bf16_reference=max(abs(a-b) for a,b in zip(out,ref_bf)),
                rounded_bf16_reference_half_L1=sum(abs(a-b) for a,b in zip(ref_bf,ref))/2,
                row_sum_error=abs(sum(out)-1)))
write_csv('numerical_rows.csv',rows)
profiles=[]
for kind in sorted({r['profile'] for r in rows}):
    group=[r for r in rows if r['profile']==kind]
    profiles.append(dict(profile=kind,rows=len(group),
        max_abs_vs_fp64=max(r['max_abs_vs_fp64'] for r in group),
        mean_half_L1_vs_fp64=sum(r['half_L1_vs_fp64'] for r in group)/len(group),
        max_half_L1_vs_fp64=max(r['half_L1_vs_fp64'] for r in group),
        max_row_sum_error=max(r['row_sum_error'] for r in group),
        max_abs_vs_rounded_bf16_reference=max(r['max_abs_vs_rounded_bf16_reference'] for r in group)))
write_csv('numerical_summary.csv',profiles)

schedule=[]
for lanes in [1,4,16,64]:
    for freq_MHz in [100,500,1000]:
        cycles=3*math.ceil(M/lanes)+16
        schedule.append(dict(lanes=lanes,frequency_MHz=freq_MHz,
                             assumed_cycles=cycles,assumed_us=cycles/freq_MHz,
                             status='schedule assumption; not RTL-verified'))
write_csv('schedule.csv',schedule)

vexp=dict(source='VEXP arXiv:2504.11227v1 Table IV and footnote 6',
    technology_nm=12,voltage_V=0.8,frequency_GHz=1,
    reported_average_softmax_Goutputs_s=0.45,
    reported_whole_core_softmax_mW=7.1,
    derived_2048_row_us=M/450,
    derived_2048_row_core_nJ=7.1*M/450,
    added_EXP_only_area_um2=968,
    reported_EXP_pJ_output=6.39,
    derived_2048_EXP_only_nJ=M*6.39/1000,
    scope='Reported post-layout results, not silicon measurements. Area is added EXP unit only; power covers whole core during softmax; shared cluster SRAM/interconnect and off-chip memory are not included as a full-system budget. 2048-row figures linearly extrapolate the reported average throughput; they are not a fresh N=2048 benchmark.')

counts=dict(max_comparisons=M-1,subtracts=M,exp_evaluations=M,sum_adds=M-1,
            reciprocals=1,normalize_multiplies=M,
            bf16_input_bytes=M*2,fp32_exp_buffer_bytes=M*4,
            bf16_output_bytes=M*2,lut_unique_payload_bytes=math.ceil(257*25/8),
            lut_note='Unsigned Q1.24 requires 25 bits to represent exp(0)=1. Unique packed payload is 804B; 32-bit coefficient words use 1028B. 16 lanes require multiport logic, banking or replication; this is not total physical memory/area.')

ising=[]
for tau in [1,10,100]:
    horizon=2064*tau
    ising.append(dict(tau_ns=tau,zero_overhead_max_cell_uW=vexp['derived_2048_row_core_nJ']*1e6/(2047*horizon),
                      max_cell_uW_after_5nJ=max(0,vexp['derived_2048_row_core_nJ']-5)*1e6/(2047*horizon)))
write_csv('ising_break_even.csv',ising)
result=dict(status='numerical prototype plus published-data extrapolation; no RTL synthesis, new model PPL or device power measurement',
    vector_length=M,numerical_rows=len(rows),seeds=[0,1,2],
    numerical_definition='Synthetic BF16 inputs. Reference: FP64 stable softmax on identical BF16 input. Candidate: unsigned Q1.24 exp LUT (step 1/16), 10-bit interpolation fraction, flush delta<-16 to zero, FP32 16-lane accumulation, FP32 reciprocal/multiply, BF16 output. Half-L1 is not called TV because BF16 outputs need not sum exactly to one.',
    vexp=vexp,operation_counts=counts,profiles=profiles,
    worst_max_abs_error=max(r['max_abs_vs_fp64'] for r in rows),
    worst_half_L1=max(r['half_L1_vs_fp64'] for r in rows),
    mean_half_L1=sum(r['half_L1_vs_fp64'] for r in rows)/len(rows),
    worst_row_sum_error=max(r['row_sum_error'] for r in rows),
    worst_additional_abs_error=max(r['max_abs_vs_rounded_bf16_reference'] for r in rows))
(ROOT/'metrics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
files=['evaluate.py','metrics.json','numerical_rows.csv','numerical_summary.csv','schedule.csv','ising_break_even.csv']
(ROOT/'checksums.json').write_text(json.dumps({n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in files},indent=2)+'\n',encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
