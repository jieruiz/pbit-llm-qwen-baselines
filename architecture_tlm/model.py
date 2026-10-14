"""Small transaction-level timing workbench for fixed-weight sampled dot cores.

Models ordinary SRAM preload -> explicit local weight hold, not an imaginary
multi-read-port SRAM. No value inference or full-model token/J predictions.
"""
import heapq
import math
from dataclasses import dataclass


@dataclass
class Config:
    samples: int = 4096
    columns: int = 1
    frames: int = 2
    storage: str = 'clocked_dff'
    sources: int = 11
    whole_rng_words: bool = False
    sram_banks: int = 8
    sram_word_bytes: int = 2  # 9-bit sign/magnitude padded to 16-bit SRAM word
    sram_latency_cycles: int = 2
    sram_outstanding: int = 8
    sram_capacity_bytes: int = 65536
    rng_fifo_bits: int = 65536


class BankedSram:
    """No row cache. One non-pipelined port per bank; finite outstanding.

    Bank latency and layout are assumptions, not library timing. Every word
    completion must precede consumption; trace times are in consumer clocks.
    """
    def __init__(self, cfg):
        self.cfg=cfg
        self.free=[0]*cfg.sram_banks
        self.pending=[]
        self.reads=self.writes=0
        self.trace=[]

    def transfer(self, start, words, write, base=0):
        cursor=start
        end=start
        for word in range(words):
            while self.pending and self.pending[0]<=cursor:
                heapq.heappop(self.pending)
            if len(self.pending)>=self.cfg.sram_outstanding:
                cursor=heapq.heappop(self.pending)
                while self.pending and self.pending[0]<=cursor:
                    heapq.heappop(self.pending)
            bank=(base+word)%len(self.free)
            issue=max(cursor,self.free[bank])
            done=issue+self.cfg.sram_latency_cycles
            self.free[bank]=done
            heapq.heappush(self.pending,done)
            end=max(end,done)
            self.trace.append(dict(kind='sram_write' if write else 'sram_read',
                                   address=(base+word)*self.cfg.sram_word_bytes,
                                   bytes=self.cfg.sram_word_bytes, bank=bank,
                                   issue_cycle=issue, response_cycle=done))
        if write: self.writes+=words
        else: self.reads+=words
        return end


class Rng:
    """Always-running finite FIFO. No free RNG and no uncharacterized gating."""
    def __init__(self, cfg, frequency, source):
        self.rate=cfg.sources*source['clock_Hz']*source['raw_bits_per_cycle']/frequency
        self.capacity=cfg.rng_fifo_bits
        self.level=0.
        self.clock=0
        self.generated=self.discarded=self.consumed=0.

    def advance(self, cycle):
        assert cycle>=self.clock
        produced=(cycle-self.clock)*self.rate
        self.generated+=produced
        self.discarded+=max(0.,self.level+produced-self.capacity)
        self.level=min(self.capacity,self.level+produced)
        self.clock=cycle

    def consume(self, earliest, bits):
        assert bits<=self.capacity
        self.advance(earliest)
        if self.level+1e-8<bits:
            self.advance(earliest+math.ceil((bits-self.level)/self.rate-1e-12))
        self.level-=bits
        self.consumed+=bits
        return self.clock


def simulate(cfg, profile, source):
    assert cfg.storage in ('clocked_dff','gated_dff','sram_hold')
    assert cfg.samples>0 and cfg.frames>0 and cfg.columns>0 and cfg.sources>0
    assert cfg.sram_banks>0 and cfg.sram_latency_cycles>0 and cfg.sram_outstanding>0
    frequency=profile['frequency_hz']
    lanes=profile['inputs']
    words=lanes*cfg.columns
    hold_bits=words*9
    local_sram_bytes=words*cfg.sram_word_bytes if cfg.storage=='sram_hold' else 0
    # Clocked probability/sign registers are explicitly retained per replica.
    probability_bits=lanes*14*cfg.columns
    valid_bits=lanes*cfg.columns if cfg.storage=='sram_hold' else 0
    if local_sram_bytes>cfg.sram_capacity_bytes:
        raise ValueError('resident weight tile exceeds SRAM capacity')
    mem=BankedSram(cfg)
    rng=Rng(cfg,frequency,source)
    events=[]
    cycle=0
    if cfg.storage=='sram_hold':
        # Workbench config source writes one resident tile. DRAM not modeled.
        written=mem.transfer(cycle,words,True)
        read=mem.transfer(written,words,False)
        events.append(dict(kind='weight_sram_preload_to_hold',start=cycle,end=read,
                           reads=words,writes=words,hold_bits=hold_bits))
        cycle=read
    else:
        # Replicated cores configure all columns concurrently at each lane addr.
        # A wider load bus is an explicit architectural assumption.
        events.append(dict(kind='weight_config',start=cycle,end=cycle+lanes,
                           configured_weight_words=words))
        cycle+=lanes
    active=rng_wait=0
    rng_bits_per_sample=lanes*(16 if cfg.whole_rng_words else 12)
    for frame in range(cfg.frames):
        cfg_start=cycle
        cycle+=lanes # probability/sign load at 1 input address per clock
        events.append(dict(kind='input_config',frame=frame,start=cfg_start,end=cycle))
        begin=cycle
        cycle+=1 # start pulse; weights are ready before computation starts
        for sample in range(cfg.samples):
            ready=rng.consume(cycle,rng_bits_per_sample)
            rng_wait+=ready-cycle
            cycle=ready+1
        active+=cfg.samples+1
        events.append(dict(kind='sample_frame',frame=frame,start=begin,end=cycle,
                           samples=cfg.samples,weight_reloads=0))
    rng.advance(cycle)
    assert abs(rng.generated-rng.consumed-rng.discarded-rng.level)<1e-5
    active_seconds=active/frequency
    elapsed=cycle/frequency
    # Only explicitly covered components are added. Missing costs stay null.
    # Active power uses historical same G/B core minus weight DFFs.
    other_J=profile['other_W']*active_seconds*cfg.columns
    if cfg.storage=='clocked_dff':
        weight_J=profile['weight_total_W']*active_seconds*cfg.columns
    elif cfg.storage=='gated_dff':
        weight_J=profile['weight_leakage_W']*active_seconds*cfg.columns
    else:
        weight_J=None
    rng_J=cfg.sources*source['source_power_W']*elapsed
    partial_J=other_J+rng_J+(weight_J or 0.)
    missing=['input/weight configuration energy','non-active core power and clock tree',
             'RNG startup/distribution/FIFO retention and access','external scales/encoding/output conversion']
    if cfg.storage=='gated_dff':
        missing+=['real ICG power','new top/control power','weight leakage during non-active cycles']
    if cfg.storage=='sram_hold':
        missing+=['real SRAM read/write/leakage','local hold load/retention',
                  'valid bits/masking and new control power']
    return dict(scope='resident fixed-weight sampled dot workbench, not LLM inference',
        config=vars(cfg), cycles=cycle, elapsed_us=elapsed*1e6,
        core_active_cycles=active,rng_stall_cycles=rng_wait,
        calibration_frequency_hz=frequency,
        power_profile_extrapolated=cfg.samples!=profile['sample_cycles'] or cfg.columns!=1,
        rng=dict(generated_bits=rng.generated,consumed_bits=rng.consumed,
                 discarded_bits=rng.discarded,fifo_final_bits=rng.level,
                 sources=cfg.sources,broadcast_across_columns=True,
                 evidence='user-reported RNG, not directly characterized in this package'),
        storage=dict(weight_config_words=words,
            sram_read_words=mem.reads,sram_write_words=mem.writes,
            sram_bytes_transferred=(mem.reads+mem.writes)*cfg.sram_word_bytes,
            sram_resident_weight_bytes=local_sram_bytes,
            local_weight_hold_bits=hold_bits, retained_probability_sign_bits=probability_bits,
            sram_valid_bits=valid_bits, weight_reloads_between_frames=0,
            per_sample_sram_reads=0,
            active_weight_clock_edges_per_bit=active if cfg.storage=='clocked_dff' else 0),
        energy=dict(total_J=None,energy_complete=False,covered_partial_J=partial_J,
            non_weight_core_active_J=other_J, weight_active_J=weight_J,
            always_running_rng_J=rng_J,missing_components=missing,
            warning='partial sums across storage variants are not complete energy comparisons'),
        transactions=mem.trace, events=events)
