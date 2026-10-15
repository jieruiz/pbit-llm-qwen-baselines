import copy
import json
from pathlib import Path
import sys
import unittest
VARIANT_ROOT=Path(__file__).resolve().parents[1]
TLM_ROOT=VARIANT_ROOT.parents[1]
sys.path.insert(0,str(VARIANT_ROOT))
sys.path.insert(0,str(TLM_ROOT))
from run_local_shared_v2 import configs
from system_decode_base import DecodeModel,ROOT
from system_decode_local_shared_v2 import LocalSharedDecode,MatchedBaselineDecode
from rng_local_shared_v2 import LocalSharedRng
from local_shared_reference_v2 import SharedLfsr32


class LocalSharedTests(unittest.TestCase):
    def setUp(self):
        cases,self.evidence,self.shape=configs()
        self.cfg=copy.deepcopy(dict(cases)['new_A8_per_tile'])

    def small(self,variant='dual_sc_sz',change=None):
        c=copy.deepcopy(self.cfg);c['samples']=64;c['context_tokens']=128
        s=copy.deepcopy(self.shape);s.update(num_hidden_layers=1,vocab_size=1024)
        if change:change(c)
        return LocalSharedDecode(c,variant,self.evidence,s).run()

    def rng(self,change=None,weights=True):
        c=copy.deepcopy(self.cfg)
        if change:change(c)
        return LocalSharedRng(c['rng_hierarchy'],c['local_shared_input'],8,1e8,weights)

    def test_same_LLM_work_and_weight_side_events_no_replay(self):
        c=copy.deepcopy(self.cfg);c['samples']=64;c['context_tokens']=128
        s=copy.deepcopy(self.shape);s.update(num_hidden_layers=1,vocab_size=1024)
        a=MatchedBaselineDecode(c,'dual_sc_sz',self.evidence,s).run();b=self.small()
        self.assertEqual(a['actions'],b['actions'])
        for key in ('local_core_updates_expected','weight_bernoulli_outputs','local_generated_bits_expected'):
            self.assertEqual(a['rng_hierarchy']['stats'][key],b['rng_hierarchy']['stats'][key])
        self.assertEqual(b['rng_hierarchy']['local_instances'],2048)
        self.assertFalse(any(k.startswith('random_replay_') for k in b['traffic']))
        self.assertEqual(b['bounds']['extra_probability_banks'],0)
        self.assertEqual(b['bounds']['retained_input_threshold_sign_logical_bits'],576)
        self.assertAlmostEqual(a['energy_J']['input_mask_distribution'],b['energy_J']['input_mask_distribution'])

    def test_energy_and_phase_conservation(self):
        d=self.small()
        self.assertAlmostEqual(sum(d['energy_J'].values())*500,d['metrics']['scenario_mJ_per_token'])
        self.assertAlmostEqual(sum(d['time_ns'].values())/d['events'][-1]['end_ns'],1)
        self.assertEqual(d['rng_hierarchy']['shared_instances'],0)
        self.assertEqual(d['rng_hierarchy']['local_input_instances'],1)

    def test_checkpoint_not_fresh_entropy_reseed_and_counts(self):
        r=self.rng();t=r.begin_tile(0,('layer0','input0'))
        for _ in range(3):
            t=r.restore_wave(t);a,_=r.input_wave(t,8,64);b,_=r.weight_wave(t,8,64);t=max(a,b)
        r.end_tile();d=r.finish(t)
        self.assertEqual(d['entropy']['conditioned_delivered_bits'],256)
        self.assertEqual(d['stats']['input_child_rekeys'],1)
        self.assertEqual(d['stats']['input_restores'],3)
        self.assertEqual(d['stats']['input_generated_bits'],3*64*8)
        self.assertEqual(d['stats']['seed_wire_bits'],2048*192+32)

    def test_root_reseed_deferred_until_safe_input_boundary(self):
        r=self.rng(lambda c:c['rng_hierarchy']['seed'].update(reseed_interval_ns=1))
        t=r.begin_tile(0,('a',));t=r.restore_wave(t)
        a,_=r.input_wave(t,8,64);b,_=r.weight_wave(t,8,64);t=max(a,b)
        self.assertEqual(r.epoch,0)
        r.end_tile();r.begin_tile(t,('b',));self.assertEqual(r.epoch,1)
        self.assertEqual(r.entropy.delivered,512)

    def test_startup_only_root_and_per_tile_child_differ(self):
        r=self.rng(lambda c:c['rng_hierarchy']['seed'].update(reseed_interval_ns=0))
        t=r.begin_tile(0,('a',));r.end_tile();r.begin_tile(1e12,('b',))
        self.assertEqual(r.epoch,0)
        self.assertEqual(r.stats['input_child_rekeys'],2)
        self.assertEqual(r.entropy.delivered,256)

    def test_fresh_entropy_on_request_charged_and_buffer_credit(self):
        r=self.rng(lambda c:c['local_shared_input'].update(child_seed_policy='fresh_entropy'))
        t=0
        for i in range(9):t=r.begin_tile(t,(i,));r.end_tile()
        self.assertEqual(r.entropy.delivered,256+9*32)
        self.assertEqual(r.entropy.blocks,3)
        self.assertEqual(r.entropy.credit,224)
        self.assertEqual(r.stats['input_fresh_entropy_bits'],9*32)

    def test_fanout_energy_does_not_charge_weight_rng_twice(self):
        a=self.small();b=self.small(change=lambda c:c['local_shared_input'].update(mask_to_columns_pJ_per_bit_sink=1))
        self.assertEqual(a['metrics']['mean_TPOT_ms'],b['metrics']['mean_TPOT_ms'])
        self.assertEqual(a['energy_J']['RNG_local_updates'],b['energy_J']['RNG_local_updates'])
        self.assertGreater(b['energy_J']['input_mask_distribution'],a['energy_J']['input_mask_distribution'])

    def test_slow_input_source_causes_real_stall(self):
        a=self.small();b=self.small(change=lambda c:c['local_shared_input'].update(clock_hz=1e6))
        self.assertGreater(b['metrics']['mean_TPOT_ms'],a['metrics']['mean_TPOT_ms'])

    def test_area_source_parameter_only_changes_area(self):
        a=self.small();b=self.small(change=lambda c:c['local_shared_input'].update(area_mm2=.05))
        self.assertAlmostEqual(b['metrics']['parameterized_area_mm2']-a['metrics']['parameterized_area_mm2'],.049)
        self.assertEqual(a['metrics']['mean_TPOT_ms'],b['metrics']['mean_TPOT_ms'])

    def test_fixed_weight_keeps_physical_arrays_without_weight_rng(self):
        d=self.small('fixed_sc_exact')
        self.assertEqual(d['rng_hierarchy']['local_instances'],0)
        self.assertEqual(d['rng_hierarchy']['stats'].get('weight_bernoulli_outputs',0),0)

    def test_actual_scratch_capacity_check_does_not_require_old_replay(self):
        c=copy.deepcopy(self.cfg);c['context_tokens']=128
        s=copy.deepcopy(self.shape);s.update(num_hidden_layers=1,vocab_size=1024)
        c['scratchpad']['capacity_MiB']=.05
        d=LocalSharedDecode(c,'fixed_sc_exact',self.evidence,s).run()
        self.assertLess(d['bounds']['scratch_peak_bytes'],.05*1024**2)
        self.assertEqual(d['config']['scratchpad']['capacity_MiB'],.05)
        with self.assertRaises(ValueError):DecodeModel(c,'fixed_sc_exact',self.evidence,s)
        c['scratchpad']['capacity_MiB']=.01
        with self.assertRaises(ValueError):LocalSharedDecode(c,'fixed_sc_exact',self.evidence,s)

    def test_periodic_child_uses_no_extra_physical_entropy(self):
        r=self.rng(lambda c:c['local_shared_input'].update(child_seed_policy='periodic',child_rekey_interval_ns=1000))
        t=r.begin_tile(0,('a',));r.end_tile()
        t=r.begin_tile(t+100,('b',));r.end_tile()
        self.assertEqual(r.stats['input_child_rekeys'],1)
        r.begin_tile(t+2000,('c',));r.end_tile()
        self.assertEqual(r.stats['input_child_rekeys'],2)
        self.assertEqual(r.entropy.delivered,256)

    def test_reference_replay_nested_bits_and_different_tile_identity(self):
        r=SharedLfsr32(bytes(range(32)));q=[0,64,64,128,255]+[128]*59
        r.begin_tile(('token0','L0','gu',0))
        a,words=r.wave(q,256);b,again=r.wave(q,256)
        self.assertEqual(a,b);self.assertEqual(words,again)
        self.assertEqual(a[0],0);self.assertEqual(a[1],a[2]);self.assertLessEqual(a[1],a[3])
        with self.assertRaises(ValueError):r.wave([1]*64,256)
        r.end_tile();r.begin_tile(('token0','L0','down',0))
        self.assertNotEqual(words,r.wave(q,256)[1])

    def test_reference_transition_matches_existing_RTL_design_vectors(self):
        r=SharedLfsr32(bytes(range(32)));r.begin_tile(('test',))
        vectors=json.loads((VARIANT_ROOT/'tests/reference_port0.json').read_text())
        r.saved=vectors['initial_state']
        _,words=r.wave([128]*64,len(vectors['port0_words']))
        self.assertEqual(words,vectors['port0_words'])


if __name__=='__main__':unittest.main(verbosity=2)
