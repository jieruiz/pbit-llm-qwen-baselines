import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model import Config,simulate,BankedSram,Rng
from import_evidence import build


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.evidence=build()

    def run_case(self,**kwargs):
        return simulate(Config(**kwargs),self.evidence['profiles']['g4_case3'],self.evidence['rng'])

    def test_sample_count_and_rng_conservation(self):
        r=self.run_case(sources=1)
        self.assertEqual(r['core_active_cycles'],8194)
        x=r['rng']
        self.assertEqual(x['consumed_bits'],2*4096*64*12)
        self.assertAlmostEqual(x['generated_bits'],x['consumed_bits']+x['discarded_bits']+x['fifo_final_bits'],places=5)
        self.assertGreater(r['rng_stall_cycles'],0)

    def test_sources_limit_throughput_and_fifo_does_not_create_bits(self):
        slow=self.run_case(sources=1)
        fast=self.run_case(sources=11)
        self.assertGreater(slow['cycles'],fast['cycles']*5)
        self.assertEqual(fast['rng_stall_cycles'],0)

    def test_weight_reuse_across_frames_and_sram_not_read_per_sample(self):
        r=self.run_case(storage='sram_hold',frames=3)
        self.assertEqual(r['storage']['sram_read_words'],64)
        self.assertEqual(r['storage']['sram_write_words'],64)
        self.assertEqual(r['storage']['per_sample_sram_reads'],0)
        self.assertEqual(r['storage']['local_weight_hold_bits'],576)
        self.assertTrue(all(e['start']>=r['events'][0]['end'] for e in r['events'][1:]))

    def test_bank_conflicts_and_finite_outstanding(self):
        a=BankedSram(Config(sram_banks=8,sram_outstanding=8))
        b=BankedSram(Config(sram_banks=1,sram_outstanding=8))
        c=BankedSram(Config(sram_banks=8,sram_outstanding=1))
        self.assertLess(a.transfer(0,64,False),b.transfer(0,64,False))
        self.assertGreater(c.transfer(0,64,False),16)

    def test_missing_sram_power_never_becomes_zero_total(self):
        r=self.run_case(storage='sram_hold')
        self.assertIsNone(r['energy']['weight_active_J'])
        self.assertIsNone(r['energy']['total_J'])
        self.assertFalse(r['energy']['energy_complete'])

    def test_area_replication_is_not_free(self):
        r=self.run_case(storage='sram_hold',columns=64)
        self.assertEqual(r['storage']['sram_resident_weight_bytes'],8192)
        self.assertEqual(r['storage']['local_weight_hold_bits'],36864)
        self.assertEqual(r['storage']['retained_probability_sign_bits'],57344)
        self.assertTrue(r['power_profile_extrapolated'])

    def test_insufficient_capacity_rejected(self):
        with self.assertRaises(ValueError):
            self.run_case(storage='sram_hold',columns=64,sram_capacity_bytes=4096)

    def test_rng_broadcast_does_not_duplicate_bits_per_output(self):
        a=self.run_case(columns=1)
        b=self.run_case(columns=64)
        self.assertEqual(a['rng']['consumed_bits'],b['rng']['consumed_bits'])


if __name__=='__main__':unittest.main(verbosity=2)
