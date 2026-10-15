"""Separate v2 scheduler; inherited ordinary LLM and weight-side algorithms."""
import copy
import csv
import hashlib
from pathlib import Path
import sys

VARIANT_ROOT=Path(__file__).resolve().parent
TLM_ROOT=VARIANT_ROOT.parents[1]
if str(TLM_ROOT) not in sys.path:
    sys.path.insert(0,str(TLM_ROOT))
from system_decode_base import DecodeModel,ROOT,MIB
from rng_local_shared_v2 import LocalSharedRng


class MatchedBaselineDecode(DecodeModel):
    """Retain raw historical metrics, then charge the same mask link as v2.

    Historical source distribution prices raw uniform-word service, not the
    final 1-bit mask delivery to each multiply cell. Charge that common work to
    BOTH versions rather than unfairly adding it only to the new implementation.
    """
    def run(self):
        d=super().run()
        d['historical_raw_metrics']=dict(d['metrics'])
        coefficient=self.cfg['local_shared_input']['mask_to_columns_pJ_per_bit_sink']
        energy=d['actions']['sc_product_trials']*coefficient*1e-12
        d['energy_J']['input_mask_distribution']=energy
        d['metrics']['scenario_mJ_per_token']+=energy/self.cfg['decode_tokens']*1000
        d['comparison_accounting_adjustment']=dict(input_mask_distribution_J=energy,
            reason='same final mask-delivery cost as v2; absent as a distinct term in historical raw results')
        bits=self.cfg['rng']['input_bits_per_decision']
        d['quality_reference_ppl']=None
        reference={('fixed_sc_exact',8):12.4387,('dual_sc_sz',8):14.1106,
                   ('fixed_sc_exact',12):12.1240,('dual_sc_sz',12):13.9609}
        d['quality_reference']=dict(ideal_IID_independent_input_T4096_ppl=
            reference.get((self.variant,bits)) if self.cfg['samples']==4096 else None,
            hardware_bivium_ppl_validated=False)
        d['source_hashes']['variants/local_shared_a8_64/system_decode_local_shared_v2.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        return d

class LocalSharedDecode(DecodeModel):
    def __init__(self,cfg,variant,evidence,shape):
        if variant not in ('fixed_sc_exact','dual_sc_sz'):raise ValueError(variant)
        actual_peak=max(shape['vocab_size']*4+shape['hidden_size']*2,
                        2*shape['intermediate_size']*4+8)
        if actual_peak>cfg['scratchpad']['capacity_MiB']*MIB:
            raise ValueError('scratchpad cannot hold actual v2 logits/partial/state buffers')
        # Parent checks its OLD replay peak; bypass only that obsolete check,
        # then restore actual capacity for all later timing/energy/area charges.
        init_cfg=copy.deepcopy(cfg)
        parent_need=64*cfg['samples']*cfg['rng']['input_bits_per_decision']/8+2*shape['intermediate_size']*4
        init_cfg['scratchpad']['capacity_MiB']=max(cfg['scratchpad']['capacity_MiB'],parent_need/MIB)
        super().__init__(init_cfg,variant,evidence,shape)
        self.cfg=copy.deepcopy(cfg)
        if cfg['rng']['input_bits_per_decision']!=8:raise ValueError('v2 requires A8')
        self.rng_hierarchy=LocalSharedRng(cfg['rng_hierarchy'],cfg['local_shared_input'],
            self.engines,self.f,local_weights=variant=='dual_sc_sz')
        self.scratch_peak=actual_peak
        with (ROOT/'evidence/grouped_38015538/breakdown.csv').open(encoding='utf-8-sig') as f:
            rows=[r for r in csv.DictReader(f) if int(r['group'])==self.profile['group'] and
                int(r['batch'])==1 and int(r['case'])==self.profile['case']]
        self.front_W=sum(float(r['total_mW'])*1e-3 for r in rows if r['block'] in
            ('probability_registers','input_sign_registers','probability_comparators_config_selection'))
        self.body_W=64*(self.profile['other_W']-self.front_W)
        self.tile_serial=0

    def stochastic_matrix(self,name,inp,out,ready):
        if inp%64 or out%64:raise ValueError('64x64 mapping required')
        T=self.cfg['samples'];E=self.engines;current=ready;scratch=self.cfg['scratchpad']
        output_groups=out//64
        multiplier=self.cfg['sc_cost']['dual_core_multiplier' if self.variant=='dual_sc_sz' else 'core_multiplier']
        for k in range(inp//64):
            encoded=self.vectors(current,64*self.cfg['sc_cost']['input_encode_vector_ops'])
            current=self.rng_hierarchy.begin_tile(encoded,(self.tile_serial,name,k))
            self.tile_serial+=1
            wave_prefetch_at=current
            for j in range(0,output_groups,E):
                count=min(E,output_groups-j);tile=k*output_groups+j
                loaded=self.weights.tile_ready(name,tile*4096,count*4096,wave_prefetch_at,tiles=count)
                start=self.rng_hierarchy.restore_wave(max(current,loaded))
                kernel=(T+1)*self.period
                _,input_time=self.rng_hierarchy.input_wave(start,count,T)
                self.raw_bits+=T*8
                weight_time=0.
                if self.variant=='dual_sc_sz':
                    _,weight_time=self.rng_hierarchy.weight_wave(start,count,T)
                    self.raw_bits+=count*4096*T*8*128/127
                duration=max(kernel,input_time,weight_time)
                current=start+duration
                self.sc_compute_ns+=kernel
                self.rng_service_ns+=max(input_time,weight_time)
                self.sc_active_engine_ns+=count*kernel
                self.sc_wait_engine_ns+=count*(duration-kernel)
                self.actions['sc_tile_frames']+=count
                self.actions['sc_product_trials']+=count*4096*T
                self.energy['SC_logic_active']+=self.body_W*multiplier*count*kernel*1e-9
                # Reuse ONE existing frontend, not a new bank beside each PRNG.
                self.energy['SC_shared_input_frontend']+=self.front_W*multiplier*kernel*1e-9
                self.energy['SC_wait']+=count*max(0,duration-kernel)*self.cfg['sc_cost']['wait_W_per_engine']*1e-9
                wave_prefetch_at=start if self.cfg['double_buffer'] else current
            self.rng_hierarchy.end_tile()
            current=self.vectors(current,out*self.cfg['sc_cost']['finalize_vector_ops_per_partial'])
            partial_bytes=out*4
            self.traffic['scratch_partial_bytes']+=2*partial_bytes
            self.energy['scratch_access']+=partial_bytes*(scratch['read_pJ_per_byte']+scratch['write_pJ_per_byte'])*1e-12
        return current

    def run(self):
        d=super().run()
        credit=(self.engines-1)*self.cfg['local_shared_input']['frontend_area_credit_mm2_per_removed_bank']
        if credit>self.engines*self.cfg['sc_cost']['logic_area_mm2_per_engine']:
            raise ValueError('frontend area credit exceeds entire SC body')
        d['metrics']['parameterized_area_mm2']-=credit
        d['metrics']['area_budget_feasible']=d['metrics']['parameterized_area_mm2']<=self.cfg['area_budget_mm2']
        d['implementation']='local_shared_v2 Python macro scheduler; inherited dense attention/digital/weight SRAM; optional DRAMSys external backend'
        d['area_note']='input module replaces 8 shared Bivium cores; checkpoint included in module area; SRAM capacity unchanged; frontend area credit defaults to zero until characterized'
        d['quality_reference_ppl']=None
        d['quality_reference']=dict(ideal_IID_input_share64_A8_T4096_mean_ppl=
            (12.4535 if self.variant=='fixed_sc_exact' else 14.1483) if self.cfg['samples']==4096 else None,
            hardware_lfsr_ppl_validated=False,weight_noise='conditional Gaussian in dual numerical reference')
        d['quality_warning']='ideal IID shared-word reference only; actual LFSR short prefixes, replay scheduling and activity NOT validated by this TLM'
        d['bounds']['scratch_peak_bytes']=self.scratch_peak
        d['bounds']['random_word_replay_bytes']=0
        d['bounds']['retained_input_threshold_sign_logical_bits']=64*9
        d['bounds']['extra_probability_banks']=0
        for file in ('system_decode_local_shared_v2.py','rng_local_shared_v2.py'):
            key='variants/local_shared_a8_64/'+file
            d['source_hashes'][key]=hashlib.sha256((VARIANT_ROOT/file).read_bytes()).hexdigest()
        return d
