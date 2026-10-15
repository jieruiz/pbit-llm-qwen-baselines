"""Local input word broadcast with state replay; weight algorithm inherited v1.

Three distinct events: root entropy reseed; child-state derivation/rekey per
input tile; restoring the SAME checkpoint for output-wave replay. Only the first
and the explicit fresh_entropy policy acquire fresh physical entropy.
"""
import copy
import math
from rng_hierarchy import HierarchicalRng, positive


class LocalSharedRng(HierarchicalRng):
    def __init__(self, hierarchy, input_cfg, engines, frequency_hz, local_weights=True):
        super().__init__(hierarchy, engines, frequency_hz, local_weights)
        self.physical_engines=engines
        self.input=copy.deepcopy(input_cfg)
        c=self.input
        if (c['input_bits'],c['share_group'],c['state_bits'])!=(8,64,32):
            raise ValueError('v2 implements A8, share64, 32-bit input state')
        if c['child_seed_policy'] not in ('continuous','periodic','per_tile','fresh_entropy'):
            raise ValueError('unknown child seed policy')
        for key,value in c.items():
            if isinstance(value,(float,int)) and not isinstance(value,bool):
                if not math.isfinite(value) or value<0:raise ValueError(key)
        positive(c['clock_hz'],'input clock')
        positive(c['checkpoint_cycles'],'checkpoint cycles',True)
        positive(c['restore_cycles'],'restore cycles',True)
        if c['child_seed_policy']=='periodic':positive(c['child_rekey_interval_ns'],'child interval')
        self.cfg['shared']['cores']=0  # replace L1 bank; NEVER instantiate its 8 Bivium cores
        self.instances=self.engines*self.per_engine+1
        self.tile_open=False
        self.next_child_rekey=0.
        self.tile_identity=None

    def ensure_seeded(self,ready):
        if self.epoch>=0 and (ready<self.next_reseed or self.tile_open):
            return max(ready,self.seed_ready)
        c=self.cfg['seed'];p=self.input
        start=max(ready,self.shared_free,self.local_free,self.seed_ready)
        entropy_done=self.entropy.request(start,c['root_bits'])
        # Keep all weight state widths/seed bus protocol unchanged; input is 32 bits.
        wire_bits=(self.engines*self.per_engine*math.ceil(177/c['bus_bits'])*c['bus_bits']+
                   math.ceil(p['state_bits']/c['bus_bits'])*c['bus_bits'])
        expand_ns=math.ceil(wire_bits/c['expand_bits_per_cycle'])/c['expand_hz']*1e9
        bus_ns=wire_bits/c['bus_bits']/c['bus_hz']*1e9
        warmup_ns=max(p['warmup_cycles']/p['clock_hz']*1e9,
                      self.cfg['local']['warmup_cycles']*self.period if self.engines else 0)
        self.seed_ready=entropy_done+expand_ns+bus_ns+warmup_ns
        self.epoch+=1
        interval=c['reseed_interval_ns']
        self.next_reseed=self.seed_ready+interval if interval else math.inf
        self.stats['seed_wire_bits']+=wire_bits
        self.stats['root_seed_bits']+=c['root_bits']
        self.stats['seed_stall_ns']+=self.seed_ready-ready
        self.energy_dynamic['RNG_seed_expansion']+=wire_bits*c['expand_pJ_per_bit']*1e-12
        self.energy_dynamic['RNG_seed_bus']+=wire_bits*c['bus_pJ_per_bit']*1e-12
        self.energy_dynamic['RNG_local_updates']+=(self.engines*self.per_engine*
            self.cfg['local']['warmup_cycles']*self.cfg['local']['update_pJ']*1e-12)
        self.energy_dynamic['RNG_input_updates']+=p['warmup_cycles']*p['update_pJ']*1e-12
        self.events.append(dict(epoch=self.epoch,start_ns=start,end_ns=self.seed_ready,
            entropy_done_ns=entropy_done,wire_bits=wire_bits,expand_ns=expand_ns,
            bus_ns=bus_ns,warmup_ns=warmup_ns,reason='root_startup_or_periodic'))
        return self.seed_ready

    def shared_words(self,*args):
        raise RuntimeError('v2 has no raw-word input replay service')

    def begin_tile(self,ready,identity):
        if self.tile_open:raise RuntimeError('previous input tile not closed')
        ready=self.ensure_seeded(ready)
        p=self.input;c=self.cfg['seed']
        policy=p['child_seed_policy']
        rekey=(policy in ('per_tile','fresh_entropy') or
               policy=='periodic' and ready>=self.next_child_rekey)
        if rekey:
            begin=ready;bits=math.ceil(p['state_bits']/c['bus_bits'])*c['bus_bits']
            if policy=='fresh_entropy':
                ready=self.entropy.request(ready,p['state_bits'])
                self.stats['input_fresh_entropy_bits']+=p['state_bits']
            else:
                ready+=math.ceil(bits/c['expand_bits_per_cycle'])/c['expand_hz']*1e9
                self.energy_dynamic['RNG_input_child_expansion']+=bits*c['expand_pJ_per_bit']*1e-12
            ready+=bits/c['bus_bits']/c['bus_hz']*1e9
            ready+=p['warmup_cycles']/p['clock_hz']*1e9
            self.energy_dynamic['RNG_input_child_bus']+=bits*c['bus_pJ_per_bit']*1e-12
            self.energy_dynamic['RNG_input_updates']+=p['warmup_cycles']*p['update_pJ']*1e-12
            self.stats['input_child_rekeys']+=1
            self.stats['input_child_seed_wire_bits']+=bits
            self.stats['input_child_seed_ns']+=ready-begin
            self.next_child_rekey=ready+p['child_rekey_interval_ns']
        # One saved state, not T random words, and no new threshold bank.
        ready+=p['checkpoint_cycles']*self.period
        self.energy_dynamic['RNG_input_checkpoint']+=p['state_bits']*p['checkpoint_pJ_per_bit']*1e-12
        self.stats['input_checkpoints']+=1
        self.tile_identity=tuple(identity);self.tile_open=True
        self.shared_free=ready
        return ready

    def restore_wave(self,ready):
        if not self.tile_open:raise RuntimeError('no input checkpoint')
        p=self.input
        ready=max(ready,self.shared_free)+p['restore_cycles']*self.period
        self.energy_dynamic['RNG_input_restore']+=p['state_bits']*p['restore_pJ_per_bit']*1e-12
        self.stats['input_restores']+=1
        self.shared_free=ready
        return ready

    def input_wave(self,ready,engines,samples):
        if not self.tile_open:raise RuntimeError('no input tile')
        p=self.input
        if int(engines)!=engines or not 1<=engines<=self.physical_engines:
            raise ValueError('active engine count')
        positive(samples,'samples',True)
        start=max(ready,self.shared_free)
        duration=samples/p['clock_hz']*1e9
        self.shared_free=start+duration
        self.stats['input_word_updates']+=samples
        self.stats['input_generated_bits']+=samples*p['input_bits']
        self.stats['input_wave_busy_ns']+=duration
        self.energy_dynamic['RNG_input_updates']+=samples*p['update_pJ']*1e-12
        # Charge both local random-word fanout and bit-mask delivery to columns.
        self.energy_dynamic['RNG_input_word_distribution']+=(samples*p['input_bits']*64*
            p['word_to_comparators_pJ_per_bit_sink']*1e-12)
        self.energy_dynamic['input_mask_distribution']+=(samples*64*64*engines*
            p['mask_to_columns_pJ_per_bit_sink']*1e-12)
        return self.shared_free,self.shared_free-ready

    def end_tile(self):
        if not self.tile_open:raise RuntimeError('no input tile')
        self.tile_open=False;self.tile_identity=None

    def finish(self,end_ns):
        if self.tile_open:raise RuntimeError('unfinished input tile')
        d=super().finish(end_ns)
        p=self.input
        d['architecture']='local_shared_v2'
        d['shared_instances']=0;d['local_input_instances']=1
        d['area_mm2']+=p['area_mm2']
        d['energy_J']['RNG_input_leakage']=p['leakage_W']*end_ns*1e-9
        d['nominal_output_capacity_bits']+=p['input_bits']*p['clock_hz']*end_ns*1e-9
        d['input_protocol']=p
        d['storage']=dict(input_live_state_bits=32,input_checkpoint_bits=32,
            random_word_replay_bytes=0,additional_probability_banks=0,
            retained_threshold_sign_logical_bits=64*9,
            frontend_power_uses_conservative_legacy13bit_proxy=True)
        d['limitations']+=['input LFSR/route/seed costs uncharacterized',
            'root reseeding deferred until current input tile is complete',
            'one input frontend broadcasts to synchronized output arrays; physical fanout not calibrated',
            'same state restored across output waves; not a fresh entropy reseed per output wave']
        return d
