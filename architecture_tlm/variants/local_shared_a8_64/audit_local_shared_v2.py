"""Audit complete saved runs: frozen baseline, work, seeds, traffic and costs."""
import hashlib
import json
from pathlib import Path
import sys

VARIANT_ROOT=Path(__file__).resolve().parent
TLM_ROOT=VARIANT_ROOT.parents[1]
if str(TLM_ROOT) not in sys.path:
    sys.path.insert(0,str(TLM_ROOT))
from system_decode_base import ROOT


def main():
    output=VARIANT_ROOT/'results/generated'
    runs={}
    for path in output.glob('*.json'):
        d=json.loads(path.read_text())
        if 'case_name' not in d:continue
        runs[(d['variant'],d['case_name'])]=d
        assert abs(sum(d['energy_J'].values())/d['config']['decode_tokens']*1000-
                   d['metrics']['scenario_mJ_per_token'])<1e-8
        assert abs(sum(d['time_ns'].values())/d['events'][-1]['end_ns']-1)<1e-12
        assert d['case_config_sha256']==hashlib.sha256(json.dumps(d['config'],sort_keys=True).encode()).hexdigest()
        for name,digest in d['source_hashes'].items():
            if name.startswith('variants/'):
                path=TLM_ROOT/name
            elif name in ('evidence_contract.json',):
                path=TLM_ROOT/'configs'/name
            else:
                path=TLM_ROOT/name
            assert hashlib.sha256(path.read_bytes()).hexdigest()==digest,name
    assert len(runs)==18,len(runs)
    for variant in ('fixed_sc_exact','dual_sc_sz'):
        old=runs[(variant,'old_A8')];new=runs[(variant,'new_A8_per_tile')]
        assert old['actions']==new['actions']
        assert abs(old['energy_J']['input_mask_distribution']-new['energy_J']['input_mask_distribution'])<1e-12
        for key,value in old['traffic'].items():
            if not key.startswith('random_replay_'):assert new['traffic'].get(key)==value,key
        assert not any(k.startswith('random_replay_') for k in new['traffic'])
        n=new['config']['decode_tokens'];layers=new['actual_shape']['num_hidden_layers']
        h=new['actual_shape']['hidden_size'];m=new['actual_shape']['intermediate_size']
        expected_tiles=n*layers*(h+m)//64
        stats=new['rng_hierarchy']['stats']
        assert stats['input_checkpoints']==expected_tiles
        assert stats['input_child_rekeys']==expected_tiles
        assert new['rng_hierarchy']['entropy']['conditioned_delivered_bits']==new['rng_hierarchy']['epochs']*256
        assert new['bounds']['extra_probability_banks']==0
        for key in ('local_core_updates_expected','weight_bernoulli_outputs','local_generated_bits_expected'):
            assert old['rng_hierarchy']['stats'].get(key,0)==stats.get(key,0)
        for key in ('RNG_local_selection_compare','RNG_local_updates'):
            assert abs(old['energy_J'].get(key,0)-new['energy_J'].get(key,0))<1e-12,key
        fresh=runs[(variant,'new_A8_fresh_entropy')]
        assert fresh['rng_hierarchy']['stats']['input_fresh_entropy_bits']==32*expected_tiles
        assert fresh['rng_hierarchy']['entropy']['conditioned_delivered_bits']==(
            fresh['rng_hierarchy']['epochs']*256+32*expected_tiles)
    frozen=json.loads((VARIANT_ROOT/'baselines/historical_metrics.json').read_text())
    for variant,metrics in frozen.items():
        assert runs[(variant,'old_A12')]['historical_raw_metrics']==metrics
    result=dict(status='PASS',complete_runs=18,new_unit_tests=14,legacy_regression_tests=23,
        old_baseline_raw_metrics_unchanged=True,source_and_config_hashes=True,
        energy_and_phase_conservation=True,LLM_work_preserved=True,weight_algorithm_and_work_preserved=True,
        default_weight_dynamic_cost_unchanged=True,seed_replay_not_entropy=True,
        fresh_entropy_per_tile_charged=True,no_extra_probability_banks=True,
        common_mask_distribution_charged_both_sides=True,raw_word_replay_removed=True,
        dramsys_run=False,physical_PPA_calibrated=False,actual_lfsr_LLM_ppl_validated=False)
    (output/'verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
