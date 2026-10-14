import csv
import json
import math
from pathlib import Path
from import_evidence import build
from model import Config,simulate

ROOT=Path(__file__).resolve().parent


def main():
    evidence=build()
    out=ROOT/'results'
    out.mkdir(exist_ok=True)
    checks=[]
    # Reproduce all 16 original frame energies and eight new G1/G4 profiles.
    for mode,cases in evidence['legacy_calibration_profiles'].items():
        for case,p in cases.items():
            e=p['reported_total_power_W']*p['calibration_frame_cycles']/p['clock_Hz']*1e12
            assert math.isclose(e,p['total_pJ_per_frame'],rel_tol=1e-12)
            checks.append(dict(profile=f'legacy/{mode}/{case}',energy_pJ=e,status='PASS'))
    for key,p in evidence['profiles'].items():
        e=p['total_W']*p['frame_cycles']/p['frequency_hz']*1e9
        assert math.isclose(e,p['energy_nJ'],rel_tol=1e-12)
        checks.append(dict(profile=key,energy_nJ=e,status='PASS'))
    summaries=[]
    for columns in (1,64):
        for storage in ('clocked_dff','gated_dff','sram_hold'):
            for sources in (1,11):
                c=Config(columns=columns,storage=storage,sources=sources)
                result=simulate(c,evidence['profiles']['g4_case3'],evidence['rng'])
                name=f'{storage}_cols{columns}_rng{sources}'
                (out/(name+'.json')).write_text(json.dumps(result,indent=2)+'\n')
                summaries.append(dict(name=name,storage=storage,columns=columns,sources=sources,
                    elapsed_us=result['elapsed_us'],rng_stall_cycles=result['rng_stall_cycles'],
                    weight_sram_reads=result['storage']['sram_read_words'],
                    weight_sram_writes=result['storage']['sram_write_words'],
                    hold_bits=result['storage']['local_weight_hold_bits'],
                    sram_resident_bytes=result['storage']['sram_resident_weight_bytes'],
                    covered_partial_nJ=result['energy']['covered_partial_J']*1e9,
                    total_energy_nJ=None,energy_complete=False))
    (out/'calibration_checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    (out/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    with (out/'summary.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=summaries[0]);w.writeheader();w.writerows(summaries)
    p=evidence['profiles']['g4_case3']
    shape=json.loads((ROOT/'configs/model_config.json').read_text())
    h,m,l=shape['hidden_size'],shape['intermediate_size'],shape['num_hidden_layers']
    audit=dict(ffn_int8_bytes_per_layer=3*h*m,ffn_int8_bytes_all_layers=3*h*m*l,
        int8_tile64_bytes=4096,sign_magnitude9_tile64_packed_bytes=4608,
        sign_magnitude9_padded16_tile64_sram_bytes=8192,
        sample_weight_dff_power_fraction=p['weight_total_W']/p['total_W'],
        removable_weight_dynamic_fraction_if_other_logic_unchanged=p['weight_dynamic_W']/p['total_W'],
        note='fraction is accounting ceiling for this old OR4 core; excludes ICG/SRAM cost, loading and layout changes')
    (out/'capacity_and_ceiling.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(json.dumps(dict(calibration_profiles=len(checks),scenarios=len(summaries),
                         all_complete_energy=False,weight_dynamic_ceiling_fraction=audit['removable_weight_dynamic_fraction_if_other_logic_unchanged']),indent=2))


if __name__=='__main__':main()
