import copy
import math
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from system_decode import DecodeModel,load,MIB


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.cfg,cls.evidence,cls.shape=load()

    def small(self,variant,mutate=None):
        c=copy.deepcopy(self.cfg)
        c['decode_tokens']=2;c['context_tokens']=128
        if mutate:mutate(c)
        s=copy.deepcopy(self.shape);s['num_hidden_layers']=1;s['vocab_size']=1024
        return DecodeModel(c,variant,self.evidence,s).run()

    def test_exact_digital_MAC_and_dense_attention_shape(self):
        r=self.small('digital_bf16')
        h,m=896,4864
        weights=h*(h+256)+h*h+3*h*m+1024*h
        att=2*14*64*(128+129)
        self.assertEqual(r['actions']['digital_MAC'],2*weights+att)
        self.assertEqual(r['actions']['attention_QK_PV_MAC'],att)

    def test_GQA_reads_KV_once_not_once_per_Q_head(self):
        r=self.small('digital_bf16')
        self.assertEqual(r['traffic']['external_KV_read_logical_bytes'],2*2*64*2*(128+129))

    def test_SC_trial_count_and_input_random_reuse(self):
        r=self.small('fixed_sc_exact')
        T=self.cfg['samples']
        self.assertEqual(r['actions']['sc_product_trials'],2*3*896*4864*T)
        expected=2*(896+4864)*T*12
        self.assertEqual(r['bounds']['raw_rng_bits'],expected)
        self.assertNotIn('weight_registers',r['energy_J'])

    def test_dual_needs_connection_weight_randomness(self):
        r=self.small('dual_sc_sz')
        expected=2*(896+4864)*4096*12+2*3*896*4864*4096*self.cfg['rng']['weight_bits_per_decision']
        self.assertAlmostEqual(r['bounds']['raw_rng_bits']/expected,1,places=12)

    def test_resident_cache_reuses_weights_on_second_token(self):
        r=self.small('digital_bf16',lambda c:c['weight_macro'].update(cache_MiB=64))
        self.assertGreater(r['weight_cache']['hits'],r['weight_cache']['misses'])
        self.assertLess(r['metrics']['subsequent_decode_ms'],r['metrics']['first_decode_ms'])
        self.assertLessEqual(r['weight_cache']['peak_bytes'],64*MIB)

    def test_faster_random_source_improves_dual_time(self):
        slow=self.small('dual_sc_sz')
        fast=self.small('dual_sc_sz',lambda c:c['rng'].update(sources=1024))
        self.assertLess(fast['metrics']['mean_TPOT_ms'],slow['metrics']['mean_TPOT_ms']/10)

    def test_energy_component_sum_and_nonoverlapping_phase_times(self):
        r=self.small('fixed_sc_exact')
        total=sum(r['energy_J'].values())/2*1000
        self.assertAlmostEqual(total,r['metrics']['scenario_mJ_per_token'])
        end=r['events'][-1]['end_ns']
        self.assertAlmostEqual(sum(r['time_ns'].values())/end,1,places=12)
        for a,b in zip(r['events'],r['events'][1:]):self.assertAlmostEqual(a['end_ns'],b['start_ns'])

    def test_area_coeff_changes_area_not_latency(self):
        a=self.small('fixed_sc_exact')
        b=self.small('fixed_sc_exact',lambda c:c['weight_macro'].update(area_mm2_per_MiB=2))
        self.assertEqual(a['metrics']['mean_TPOT_ms'],b['metrics']['mean_TPOT_ms'])
        self.assertGreater(b['metrics']['parameterized_area_mm2'],a['metrics']['parameterized_area_mm2'])

    def test_macro_power_changes_energy_without_weight_DFF_doublecharge(self):
        a=self.small('fixed_sc_exact')
        b=self.small('fixed_sc_exact',lambda c:c['weight_macro'].update(active_tile_hold_W=.01))
        self.assertEqual(a['metrics']['mean_TPOT_ms'],b['metrics']['mean_TPOT_ms'])
        self.assertGreater(b['metrics']['scenario_mJ_per_token'],a['metrics']['scenario_mJ_per_token'])

    def test_activity_profile_is_used_consistently(self):
        a=self.small('fixed_sc_exact')
        b=self.small('fixed_sc_exact',lambda c:c.update(sc_profile='g1_case1'))
        self.assertEqual(a['metrics']['mean_TPOT_ms'],b['metrics']['mean_TPOT_ms'])
        self.assertLess(b['energy_J']['SC_logic_active'],a['energy_J']['SC_logic_active'])
        self.assertEqual(b['stochastic_activity_case'],1)

    def test_insufficient_scratch_capacity_and_uncalibrated_frequency(self):
        with self.assertRaises(ValueError):self.small('fixed_sc_exact',lambda c:c['scratchpad'].update(capacity_MiB=.01))
        with self.assertRaises(ValueError):self.small('fixed_sc_exact',lambda c:c.update(frequency_hz=400000000))


if __name__=='__main__':unittest.main(verbosity=2)
