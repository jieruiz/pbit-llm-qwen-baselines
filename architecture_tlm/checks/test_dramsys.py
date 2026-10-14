import copy,json,sys,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dramsys_backend import AddressSpace
from system_decode import load,DecodeModel

class AddressTests(unittest.TestCase):
    def test_regions_are_aligned_and_disjoint(self):
        c,e,s=load()
        for variant in ('digital_bf16','digital_int8_ffn','fixed_sc_exact','dual_sc_sz'):
            space=AddressSpace(s,variant,8192,2)
            regions=list(space.regions.values())
            for r in regions:self.assertEqual(r['base']%4096,0)
            for a,b in zip(regions,regions[1:]):self.assertLessEqual(a['base']+a['allocated_bytes'],b['base'])
            self.assertLess(space.total_bytes,8*1024**3)

    def test_KV_addresses_are_unique_across_layers(self):
        c,e,s=load();space=AddressSpace(s,'fixed_sc_exact',8192,2)
        end=space.address('L0/KV',8193*512,512)
        nxt=space.address('L1/KV',0,512)
        self.assertLess(end,nxt)
        with self.assertRaises(ValueError):space.address('L0/KV',8194*512,4096)

    def test_tied_embedding_uses_same_lmhead_region(self):
        c,e,s=load();space=AddressSpace(s,'digital_bf16',2048,2)
        self.assertEqual(space.address('lm_head',7*896*2,896*2),space.regions['lm_head']['base']+7*896*2)

    def test_config_address_bits_cover_capacity_exactly(self):
        root=Path(__file__).resolve().parents[1]
        for policy in ('NoPowerDown','Staggered'):
            c=json.loads((root/f'dramsys/ddr4_2ch_{policy}.json').read_text())['simulation']
            bits=[bit for values in c['addressmapping'].values() for bit in values]
            self.assertEqual(sorted(bits),list(range(33)))
            self.assertEqual(c['memspec']['memarchitecturespec']['nbrOfChannels'],2)
            self.assertTrue(c['simconfig']['PowerAnalysis'])
            # Preserve consecutive column reuse within a row; completeness
            # alone would not catch a low row bit interspersed with columns.
            self.assertGreater(min(c['addressmapping']['ROW_BIT']),max(c['addressmapping']['COLUMN_BIT']))

if __name__=='__main__':unittest.main()
