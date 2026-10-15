"""Hierarchical RNG v1: expected-retry macro model, nanoseconds/pJ/mm2.

Physical entropy is a budget, NOT a simulated or certified entropy source.
All cost coefficients are explicit scenario inputs. No external dependencies.
"""
import copy
import math
from collections import defaultdict
from functools import lru_cache


def positive(value, name, integer=False):
    if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    if integer and int(value) != value:
        raise ValueError(f'{name} must be an integer')


@lru_cache(maxsize=64)
def retry_statistics(candidates, required=64, columns=64):
    """Uniform bytes <254, then //2: exactly uniform 0..126.

    Each column retries its whole batch on underflow; completed columns freeze.
    Return batch success probability and E[max column attempts]. Assumes IID
    candidate bytes; this is a statistical timing approximation, not a bound.
    """
    if candidates < required:
        raise ValueError('candidate batch cannot fill one column')
    p = 254 / 256
    failure = sum(math.exp(math.lgamma(candidates+1)-math.lgamma(k+1)
                          - math.lgamma(candidates-k+1)
                          + k*math.log(p)+(candidates-k)*math.log1p(-p))
                  for k in range(required))
    success = 1 - failure
    if success <= 0:
        raise ValueError('batch success probability underflow')
    rounds = 1.
    tail = failure
    for _ in range(10000):
        extra = -math.expm1(columns*math.log1p(-tail)) if tail else 0.
        rounds += extra
        if extra < 1e-12:
            return success, rounds
        tail *= failure
    raise ValueError('retry expectation did not converge')


class EntropyBudget:
    """On-demand block acquisition; no free prefilled FIFO or physical bit claim.

    Raw samples are collected, then conditioned, serially. Retain unused
    conditioned bits. Health failure prevents new seed material being served.
    """
    def __init__(self, cfg):
        self.cfg = cfg
        for key in ('channels', 'sample_hz', 'raw_samples_per_block',
                    'output_bits_per_block', 'min_entropy_per_sample', 'safety_factor'):
            positive(cfg[key], 'entropy.'+key,
                     key in ('channels', 'raw_samples_per_block', 'output_bits_per_block'))
        if cfg['min_entropy_per_sample'] > 1 or cfg['safety_factor'] > 1:
            raise ValueError('raw binary sample entropy and safety factor must be <=1')
        if cfg['output_bits_per_block'] > (cfg['raw_samples_per_block'] *
                cfg['min_entropy_per_sample'] * cfg['safety_factor']):
            raise ValueError('conditioner output exceeds conservative entropy budget')
        self.free = cfg['startup_ns']
        self.credit = 0
        self.blocks = self.raw_samples = self.delivered = 0
        self.active_ns = self.conditioner_ns = 0.
        self.healthy = cfg.get('healthy', True)

    def request(self, ready, bits):
        if not self.healthy:
            raise RuntimeError('entropy health failure: refuse seed acquisition')
        positive(bits, 'entropy request', True)
        c = self.cfg
        blocks = max(0, math.ceil((bits-self.credit)/c['output_bits_per_block']))
        collect = blocks*c['raw_samples_per_block']/(c['channels']*c['sample_hz'])*1e9
        condition = blocks*c['conditioner_ns_per_block']
        self.free = max(ready, self.free) + collect + condition
        self.blocks += blocks
        self.raw_samples += blocks*c['raw_samples_per_block']
        self.active_ns += collect
        self.conditioner_ns += condition
        self.credit += blocks*c['output_bits_per_block']-bits
        self.delivered += bits
        return self.free


class HierarchicalRng:
    """L2 entropy -> root seed expansion -> L1 shared / L0 per-array Bivium.

    Whole-epoch reseeding stalls all RNG consumers at transaction boundaries.
    Interval=0 means startup only. No speculative/background reseeding in v1.
    L1 supplies replayable input words; L0 supplies per-connection weight bits.
    """
    def __init__(self, cfg, engines, frequency_hz, local_weights=True):
        self.cfg = copy.deepcopy(cfg)
        if engines < 1:
            raise ValueError('hierarchical RNG requires a stochastic engine')
        self.engines = engines if local_weights else 0
        self.f = frequency_hz
        self.period = 1e9/frequency_hz
        for section in ('entropy', 'seed', 'shared', 'local'):
            for key, value in cfg[section].items():
                if isinstance(value, (float, int)) and not isinstance(value, bool):
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f'negative/nonfinite {section}.{key}')
        for section, names in [('seed', ('root_bits', 'bus_bits', 'bus_hz', 'expand_bits_per_cycle', 'expand_hz')),
                               ('shared', ('cores', 'output_bits', 'clock_hz')),
                               ('local', ('cores_per_column', 'output_bits', 'columns', 'inputs'))]:
            for key in names:
                positive(cfg[section][key], section+'.'+key, not key.endswith('hz'))
        if (cfg['local']['inputs'], cfg['local']['columns']) != (64,64):
            raise ValueError('v1 mapping is 64x64')
        for section in ('shared', 'local'):
            positive(cfg[section]['warmup_cycles'], section+'.warmup_cycles', True)
            if cfg[section]['output_bits'] % 8:
                raise ValueError('whole-byte Bivium output required')
        l, s = cfg['local'], cfg['shared']
        self.per_engine = l['cores_per_column']*l['columns']
        self.instances = self.engines*self.per_engine+s['cores']
        candidates = l['cores_per_column']*l['output_bits']//8
        self.success, self.rounds = retry_statistics(candidates, l['inputs'], l['columns'])
        self.entropy = EntropyBudget(cfg['entropy'])
        self.shared_free = self.local_free = self.seed_ready = 0.
        self.epoch = -1
        self.next_reseed = 0.
        self.events = []
        self.stats = defaultdict(float)
        self.energy_dynamic = defaultdict(float)

    def ensure_seeded(self, ready):
        if self.epoch >= 0 and ready < self.next_reseed:
            return max(ready, self.seed_ready)
        c = self.cfg['seed']
        start = max(ready, self.shared_free, self.local_free, self.seed_ready)
        entropy_done = self.entropy.request(start, c['root_bits'])
        wire_bits = self.instances*math.ceil(177/c['bus_bits'])*c['bus_bits']
        # Conservative store-and-forward root expansion then seed bus service.
        expand_ns = math.ceil(wire_bits/c['expand_bits_per_cycle'])/c['expand_hz']*1e9
        bus_ns = wire_bits/c['bus_bits']/c['bus_hz']*1e9
        warmup_ns = max(self.cfg['shared']['warmup_cycles']/self.cfg['shared']['clock_hz']*1e9,
                        self.cfg['local']['warmup_cycles']*self.period if self.engines else 0)
        self.seed_ready = entropy_done+expand_ns+bus_ns+warmup_ns
        self.epoch += 1
        interval = c['reseed_interval_ns']
        self.next_reseed = self.seed_ready+interval if interval else math.inf
        self.stats['seed_wire_bits'] += wire_bits
        self.stats['root_seed_bits'] += c['root_bits']
        self.stats['seed_stall_ns'] += self.seed_ready-ready
        self.energy_dynamic['RNG_seed_expansion'] += wire_bits*c['expand_pJ_per_bit']*1e-12
        self.energy_dynamic['RNG_seed_bus'] += wire_bits*c['bus_pJ_per_bit']*1e-12
        for section, count in [('shared',self.cfg['shared']['cores']), ('local',self.engines*self.per_engine)]:
            t = self.cfg[section]
            self.energy_dynamic['RNG_'+section+'_updates'] += count*t['warmup_cycles']*t['update_pJ']*1e-12
        self.events.append(dict(epoch=self.epoch,start_ns=start,end_ns=self.seed_ready,
                                entropy_done_ns=entropy_done,wire_bits=wire_bits,
                                expand_ns=expand_ns,bus_ns=bus_ns,warmup_ns=warmup_ns))
        return self.seed_ready

    def shared_words(self, ready, bits):
        if bits < 0 or not math.isfinite(bits):
            raise ValueError('invalid shared request size')
        start = max(self.ensure_seeded(ready), self.shared_free)
        c = self.cfg['shared']
        cycles = math.ceil(bits/(c['cores']*c['output_bits']))
        duration = cycles/c['clock_hz']*1e9
        self.shared_free = start+duration
        self.stats['shared_requested_bits'] += bits
        self.stats['shared_generated_bits'] += cycles*c['cores']*c['output_bits']
        self.stats['shared_busy_ns'] += duration
        self.energy_dynamic['RNG_shared_updates'] += cycles*c['cores']*c['update_pJ']*1e-12
        self.energy_dynamic['RNG_shared_distribution'] += bits*c['distribution_pJ_per_bit']*1e-12
        return self.shared_free, self.shared_free-ready

    def weight_wave(self, ready, count, samples):
        if not 1 <= count <= self.engines or int(count)!=count:
            raise ValueError('invalid local engine wave')
        positive(samples,'samples',True)
        start = max(self.ensure_seeded(ready), self.local_free)
        c = self.cfg['local']
        # All active engines synchronize after a sample. Recompute maximum over
        # count*64 columns; no multiplication of parallel service by engine count.
        _, wave_rounds = retry_statistics(c['cores_per_column']*c['output_bits']//8,
                                          c['inputs'],count*c['columns'])
        cycles = math.ceil(samples*wave_rounds)
        self.local_free = start+cycles*self.period
        updates = count*self.per_engine*samples/self.success
        generated = updates*c['output_bits']
        comparisons = count*64*64*samples
        self.stats['local_generated_bits_expected'] += generated
        self.stats['weight_bernoulli_outputs'] += comparisons
        self.stats['weight_uniform_bits_ideal_expected'] += comparisons*8*128/127
        self.stats['local_core_updates_expected'] += updates
        self.stats['local_wave_busy_ns'] += cycles*self.period
        self.energy_dynamic['RNG_local_updates'] += updates*c['update_pJ']*1e-12
        self.energy_dynamic['RNG_local_selection_compare'] += comparisons*c['bernoulli_pJ']*1e-12
        return self.local_free, self.local_free-ready

    def finish(self, end_ns):
        if end_ns < max(self.local_free,self.shared_free,self.seed_ready):
            raise ValueError('system ends before random service completes')
        c = self.cfg
        e = dict(self.energy_dynamic)
        seconds = end_ns*1e-9
        # Always-on analog/bias is conservative; no uncharacterized power gating.
        e['RNG_entropy_analog'] = c['entropy']['channels']*c['entropy']['power_W_per_channel']*seconds
        e['RNG_entropy_health'] = c['entropy']['health_W']*seconds
        e['RNG_conditioner'] = self.entropy.blocks*c['entropy']['conditioner_pJ_per_block']*1e-12
        e['RNG_seed_idle'] = c['seed']['idle_W']*seconds
        e['RNG_shared_leakage'] = c['shared']['cores']*c['shared']['leakage_W_per_core']*seconds
        e['RNG_local_leakage'] = self.engines*self.per_engine*c['local']['leakage_W_per_core']*seconds
        area = (c['entropy']['channels']*c['entropy']['area_mm2_per_channel']+
                c['entropy']['health_conditioner_area_mm2']+c['seed']['area_mm2']+
                c['shared']['cores']*c['shared']['area_mm2_per_core']+
                self.engines*(self.per_engine*c['local']['area_mm2_per_core']+
                              c['local']['columns']*c['local']['column_logic_area_mm2']))
        nominal_rate = (c['shared']['cores']*c['shared']['output_bits']*c['shared']['clock_hz']+
                        self.engines*self.per_engine*c['local']['output_bits']*self.f)
        return dict(architecture='hierarchical_v1',area_mm2=area,energy_J=e,
                    nominal_output_capacity_bits=nominal_rate*seconds,
                    local_instances=self.engines*self.per_engine,shared_instances=c['shared']['cores'],
                    epochs=self.epoch+1,seed_events=self.events,stats=dict(self.stats),
                    entropy=dict(raw_samples=self.entropy.raw_samples,blocks=self.entropy.blocks,
                        conditioned_delivered_bits=self.entropy.delivered,conditioned_buffer_bits=self.entropy.credit,
                        healthy=self.entropy.healthy),
                    batch_success_probability=self.success,one_array_expected_rounds=self.rounds,
                    cost_status=c['cost_status'],
                    limitations=['expected IID-byte retry timing, not cycle-accurate or a worst-case bound',
                        'on-demand whole-epoch reseeding stalls at transaction boundaries',
                        'root seed expansion is parameterized; no certified DRBG or physical entropy simulation',
                        'local selectors/comparators charged here; inherited SC kernel remains a proxy'])
