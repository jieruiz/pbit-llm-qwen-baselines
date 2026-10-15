"""Macro-level LT decode model. Standard library, no SystemC dependency.

Deterministic full model shapes; trace-free dense attention. Stochastic FFN
is sampled GEMV, not numerical inference. Memory/area/energy are scenarios.
"""
import copy
import hashlib
import json
import math
from collections import OrderedDict, defaultdict
from pathlib import Path

VARIANT_ROOT=Path(__file__).resolve().parent
ROOT=VARIANT_ROOT.parents[1]
MIB=1024**2


class Link:
    def __init__(self, bandwidth_GBs, latency_ns=0):
        if bandwidth_GBs<=0: raise ValueError('bandwidth must be positive')
        self.rate=bandwidth_GBs # 1 GB/s = 1 byte/ns, decimal GB
        self.latency=latency_ns
        self.free=0.

    def transfer(self, ready, size):
        start=max(ready,self.free)
        self.free=start+size/self.rate
        return self.free+self.latency


class WeightMacro:
    """LRU weight SRAM plus abstract resident tile interface. Holds covered here.

    Macro exposes prepare/keep costs, not DFF topology. Cache is separate from
    compute tile residence; replacement does not invalidate an active tile.
    """
    def __init__(self,cfg,ext,energy,traffic):
        self.cfg=cfg
        self.ext=ext
        self.energy=energy
        self.traffic=traffic
        self.line=cfg['cache_line_bytes']
        self.slots=int(cfg['cache_MiB']*MIB)//self.line
        if self.slots<1: raise ValueError('weight SRAM needs at least one cache line')
        self.cache=OrderedDict()
        self.link=Link(cfg['load_bandwidth_GBs'],cfg['load_latency_ns'])
        self.hits=self.misses=self.peak=0

    def request(self,tensor,offset,size,ready):
        done=ready
        first=offset//self.line
        last=(offset+size-1)//self.line
        keys=[(tensor,block) for block in range(first,last+1)]
        misses=[key for key in keys if key not in self.cache]
        self.hits+=len(keys)-len(misses)
        self.misses+=len(misses)
        # Coalesce consecutive missing cache lines into DMA bursts.
        runs=[]
        for key in misses:
            if runs and key[1]==runs[-1][-1][1]+1:runs[-1].append(key)
            else:runs.append([key])
        for run in runs:
            n=len(run)*self.line
            address=self.ext.layout.address(tensor,run[0][1]*self.line,n)
            done=self.ext.access(done,n,'weights',False,address=address)
            done=self.link.transfer(done,n)
            self.energy['weight_sram_access']+=n*self.cfg['write_pJ_per_byte']*1e-12
            self.traffic['weight_sram_write_bytes']+=n
        for key in keys:
            self.cache[key]=True
            self.cache.move_to_end(key)
            if len(self.cache)>self.slots:self.cache.popitem(last=False)
            self.peak=max(self.peak,len(self.cache)*self.line)
        n=len(keys)*self.line
        done=self.link.transfer(done,n)
        self.energy['weight_sram_access']+=n*self.cfg['read_pJ_per_byte']*1e-12
        self.traffic['weight_sram_read_bytes']+=n
        return done

    def tile_ready(self,tensor,offset,size,ready,tiles=1):
        done=self.request(tensor,offset,size,ready)
        self.energy['weight_resident_prepare']+=tiles*self.cfg['resident_read_pJ_per_tile']*1e-12
        return done


class External:
    def __init__(self,cfg,energy,traffic):
        self.cfg=cfg
        self.energy=energy
        self.traffic=traffic
        self.link=Link(cfg['bandwidth_GBs'],cfg['latency_ns'])

    def access(self,ready,size,kind,write,address=None):
        n=math.ceil(size/self.cfg['burst_bytes'])*self.cfg['burst_bytes']
        direction='write' if write else 'read'
        self.traffic[f'external_{kind}_{direction}_logical_bytes']+=size
        self.traffic[f'external_{kind}_{direction}_burst_bytes']+=n
        self.traffic[f'external_{kind}_{direction}_requests']+=1
        self.energy[f'external_{kind}']+=n*self.cfg[f'{direction}_pJ_per_byte']*1e-12
        return self.link.transfer(ready,n)


class DecodeModel:
    def __init__(self,cfg,variant,evidence,shape):
        self.cfg=copy.deepcopy(cfg)
        self.variant=variant
        self.evidence=evidence
        self.shape=shape
        self.energy=defaultdict(float)
        self.traffic=defaultdict(int)
        self.actions=defaultdict(int)
        from dramsys_backend import AddressSpace
        layout=AddressSpace(shape,variant,cfg['context_tokens'],cfg['decode_tokens'])
        if cfg['external_memory'].get('backend')=='dramsys':
            from dramsys_backend import DRAMSysExternal
            self.ext=DRAMSysExternal(cfg['external_memory'],self.energy,self.traffic,layout)
        else:
            self.ext=External(cfg['external_memory'],self.energy,self.traffic)
            self.ext.layout=layout
        self.weights=WeightMacro(cfg['weight_macro'],self.ext,self.energy,self.traffic)
        self.digital_free=0.
        self.rng_free=0.
        self.raw_bits=0.
        self.events=[]
        self.time_categories=defaultdict(float)
        self.rng_service_ns=0.
        self.sc_compute_ns=0.
        self.replay_ns=0.
        self.sc_wait_engine_ns=0.
        self.sc_active_engine_ns=0.
        self.f=cfg['frequency_hz']
        self.period=1e9/self.f
        self.engines=cfg['sc_engines'] if variant.startswith(('fixed_sc','dual_sc')) else 0
        architecture=cfg['rng'].get('architecture','global_pool')
        if architecture not in ('global_pool','hierarchical_v1'):
            raise ValueError('unknown RNG architecture')
        self.rng_hierarchy=None
        if self.engines and architecture=='hierarchical_v1':
            from rng_hierarchy import HierarchicalRng
            self.rng_hierarchy=HierarchicalRng(cfg['rng_hierarchy'],self.engines,self.f,
                                             local_weights=variant=='dual_sc_sz')
        self.profile=evidence['profiles'][cfg['sc_profile']]
        if self.f!=self.profile['frequency_hz']:
            raise ValueError('SC frequency extrapolation disabled; use calibrated 100MHz')
        if (cfg['tile_inputs'],cfg['tile_outputs'])!=(64,64):
            raise ValueError('only explicit 64x64 replication mapping supported')
        for category in ('digital','sc_cost'):
            if any(v<0 for k,v in cfg[category].items() if isinstance(v,(int,float))):
                raise ValueError('negative hardware parameters')
        if self.engines and not self.rng_hierarchy and cfg['rng']['sources']<1:raise ValueError('no RNG source')
        need_random=64*cfg['samples']*cfg['rng']['input_bits_per_decision']/8
        need_logits=shape['vocab_size']*4+shape['hidden_size']*2
        need_partial=2*shape['intermediate_size']*4+need_random
        self.scratch_peak=max(need_logits,need_partial) if self.engines else need_logits
        if self.scratch_peak>cfg['scratchpad']['capacity_MiB']*MIB:
            raise ValueError('scratchpad cannot hold logits/random replay and partial sums')

    def stamp(self,name,start,end,layer=None,token=None,details=None):
        if end<start-1e-6:raise AssertionError('time moved backwards')
        self.time_categories[name]+=end-start
        self.events.append(dict(name=name,start_ns=start,end_ns=end,layer=layer,token=token,
                                details=details or {}))
        return end

    def vectors(self,ready,ops=0,nonlinear=0):
        c=self.cfg['digital']
        start=max(ready,self.digital_free)
        duration=(ops/c['vector_ops_per_cycle']+nonlinear/c['nonlinear_ops_per_cycle'])*self.period
        self.digital_free=start+duration
        self.actions['vector_ops']+=ops
        self.actions['nonlinear_ops']+=nonlinear
        self.energy['digital_vector']+=ops*c['vector_op_pJ']*1e-12
        self.energy['digital_nonlinear']+=nonlinear*c['nonlinear_op_pJ']*1e-12
        local_bytes=ops*c['vector_interface_bytes_per_op']+nonlinear*c['nonlinear_interface_bytes_per_op']
        self.traffic['scratch_vector_interface_bytes']+=local_bytes
        sp=self.cfg['scratchpad']
        self.energy['scratch_access']+=local_bytes/2*(sp['read_pJ_per_byte']+sp['write_pJ_per_byte'])*1e-12
        return self.digital_free

    def digital_matrix(self,name,inp,out,ready,bytes_per_weight=2):
        c=self.cfg['digital']
        total=inp*out*bytes_per_weight
        chunk=c['weight_chunk_bytes']
        offset=0
        fetch_at=ready
        for offset in range(0,total,chunk):
            size=min(chunk,total-offset)
            arrived=self.weights.request(name,offset,size,fetch_at)
            start=max(arrived,self.digital_free)
            macs=size//bytes_per_weight
            self.digital_free=start+macs/c['macs_per_cycle']*self.period
            self.actions['digital_MAC']+=macs
            price=c['mixed_fp_int8_mac_pJ'] if bytes_per_weight==1 else c['mac_pJ']
            self.energy['digital_MAC']+=macs*price*1e-12
            fetch_at=start if self.cfg['double_buffer'] else self.digital_free
        # Include activation/result SRAM traffic separately; no weight double charge.
        scratch=(inp+out)*2
        self.energy['scratch_access']+=scratch*self.cfg['scratchpad']['read_pJ_per_byte']*1e-12
        self.traffic['scratch_matrix_interface_bytes']+=scratch
        return self.digital_free

    def rng_service(self,ready,bits):
        if self.rng_hierarchy:
            done,duration=self.rng_hierarchy.shared_words(ready,bits)
            self.raw_bits+=bits
            self.rng_service_ns+=duration
            return done,duration
        r=self.cfg['rng']
        rate=r['sources']*r['source_clock_hz']*r['raw_bits_per_source_cycle']/1e9
        start=max(ready,self.rng_free)
        duration=bits/rate
        self.rng_free=start+duration
        self.raw_bits+=bits
        self.rng_service_ns+=duration
        return self.rng_free,duration

    def sc_kernel_power(self):
        # Input probability regs/sign/comparators preserved once per tile frontend.
        source=ROOT/'evidence/grouped_38015538/breakdown.csv'
        import csv
        if not hasattr(self,'kernel_W'):
            with source.open(encoding='utf-8-sig') as f:
                rows=[r for r in csv.DictReader(f) if int(r['group'])==self.profile['group'] and int(r['batch'])==1 and int(r['case'])==self.profile['case']]
            front=sum(float(r['total_mW'])/1000 for r in rows if r['block'] in
                      ('probability_registers','input_sign_registers','probability_comparators_config_selection'))
            logic=self.profile['other_W']-front
            self.kernel_W=(64*logic+front if self.cfg['share_frontend_per_tile'] else 64*self.profile['other_W'])
        multiplier=self.cfg['sc_cost']['dual_core_multiplier' if self.variant=='dual_sc_sz' else 'core_multiplier']
        return self.kernel_W*multiplier

    def stochastic_matrix(self,name,inp,out,ready):
        assert inp%64==out%64==0
        T=self.cfg['samples']
        output_groups=out//64
        E=self.engines
        current=ready
        per_tile_bytes=4096 # external INT8 weight format, macro handles internal representation
        raw_input_bits=64*T*self.cfg['rng']['input_bits_per_decision']
        replay_bytes=raw_input_bits/8
        scratch=self.cfg['scratchpad']
        for k in range(inp//64):
            # One actual random stream per input feature reused by every output tile.
            encoded=self.vectors(current,64*self.cfg['sc_cost']['input_encode_vector_ops'])
            produced,service=self.rng_service(encoded,raw_input_bits)
            current=produced+replay_bytes/scratch['bandwidth_GBs']
            self.energy['scratch_access']+=replay_bytes*scratch['write_pJ_per_byte']*1e-12
            self.traffic['random_replay_write_bytes']+=int(replay_bytes)
            wave_prefetch_at=current
            for j in range(0,output_groups,E):
                count=min(E,output_groups-j)
                tile=k*output_groups+j
                loaded=self.weights.tile_ready(name,tile*per_tile_bytes,count*per_tile_bytes,wave_prefetch_at,tiles=count)
                start=max(current,loaded)
                kernel=(T+1)*self.period
                read_time=replay_bytes/scratch['bandwidth_GBs']
                weight_rng_time=0.
                if self.variant=='dual_sc_sz':
                    if self.rng_hierarchy:
                        _,weight_rng_time=self.rng_hierarchy.weight_wave(start,count,T)
                        # Logical expected uniform-bit demand, not physical entropy.
                        self.raw_bits+=count*4096*T*8*128/127
                        self.rng_service_ns+=weight_rng_time
                    else:
                        bits=count*4096*T*self.cfg['rng']['weight_bits_per_decision']
                        _,weight_rng_time=self.rng_service(start,bits)
                duration=max(kernel,read_time,weight_rng_time)
                current=start+duration
                self.sc_compute_ns+=kernel
                self.replay_ns+=read_time
                self.sc_active_engine_ns+=count*kernel
                self.sc_wait_engine_ns+=count*(duration-kernel)
                self.actions['sc_tile_frames']+=count
                self.actions['sc_product_trials']+=count*4096*T
                self.energy['SC_logic_active']+=self.sc_kernel_power()*count*kernel*1e-9
                self.energy['SC_wait']+=count*max(0,duration-kernel)*self.cfg['sc_cost']['wait_W_per_engine']*1e-9
                self.energy['scratch_access']+=replay_bytes*scratch['read_pJ_per_byte']*1e-12
                self.traffic['random_replay_read_bytes']+=int(replay_bytes)
                # One shared read broadcast to all live tiles, not a free independent stream.
                wave_prefetch_at=start if self.cfg['double_buffer'] else current
            # Partial sums/scales alignment occur after input tile contributions.
            current=self.vectors(current,out*self.cfg['sc_cost']['finalize_vector_ops_per_partial'])
            partial_bytes=out*4
            self.traffic['scratch_partial_bytes']+=2*partial_bytes
            self.energy['scratch_access']+=partial_bytes*(scratch['read_pJ_per_byte']+scratch['write_pJ_per_byte'])*1e-12
        return current

    def attention(self,ready,L,layer):
        s=self.shape
        q=s['num_attention_heads'];kv=s['num_key_value_heads'];d=s['hidden_size']//q
        c=self.cfg['digital'];scratch=self.cfg['scratchpad']
        # Exact dense online softmax; GQA reads shared KV once per group.
        fetch_at=ready
        compute=ready
        for offset in range(0,L,128):
            n=min(128,L-offset)
            size=2*kv*d*2*n # K and V, BF16, fused streaming
            address=self.ext.layout.address(f'L{layer}/KV',offset*self.ext.layout.kv_token_bytes,size)
            loaded=self.ext.access(fetch_at,size,'KV',False,address=address)
            start=max(loaded,self.digital_free)
            macs=2*q*d*n
            ops=6*q*n
            exp=q*n
            duration=(macs/c['macs_per_cycle']+ops/c['vector_ops_per_cycle']+exp/c['nonlinear_ops_per_cycle'])*self.period
            compute=start+duration
            self.digital_free=compute
            self.actions['attention_QK_PV_MAC']+=macs
            self.actions['digital_MAC']+=macs
            self.actions['softmax_logits']+=exp
            self.energy['digital_MAC']+=macs*c['mac_pJ']*1e-12
            self.energy['digital_vector']+=ops*c['vector_op_pJ']*1e-12
            self.energy['digital_nonlinear']+=exp*c['nonlinear_op_pJ']*1e-12
            # Separate K/V streaming buffer write/read; no dense LxL score storage.
            self.energy['scratch_access']+=size*(scratch['write_pJ_per_byte']+scratch['read_pJ_per_byte'])*1e-12
            self.traffic['scratch_KV_tile_bytes']+=2*size
            fetch_at=start if self.cfg['double_buffer'] else compute
        return compute

    def run(self):
        s=self.shape;h=s['hidden_size'];m=s['intermediate_size'];v=s['vocab_size'];layers=s['num_hidden_layers']
        kv_dim=s['num_key_value_heads']*(h//s['num_attention_heads'])
        cursor=0.;token_times=[]
        for token in range(self.cfg['decode_tokens']):
            begin=cursor
            if token==0 and self.rng_hierarchy:
                cursor=self.stamp('RNG_startup',cursor,self.rng_hierarchy.ensure_seeded(cursor),token=token)
            L=self.cfg['context_tokens']+token
            # Tied embedding and lm_head share the same cache address space.
            t=self.weights.request('lm_head',7*h*2,h*2,cursor)
            cursor=self.stamp('embedding',cursor,t,token=token)
            t=self.vectors(cursor,nonlinear=h//s['num_attention_heads'])
            cursor=self.stamp('norm_rope_residual',cursor,t,token=token)
            for layer in range(layers):
                t=self.vectors(cursor,4*h,1)
                cursor=self.stamp('norm_rope_residual',cursor,t,layer,token)
                t=self.digital_matrix(f'L{layer}/qkv',h,h+2*kv_dim,cursor)
                cursor=self.stamp('QKV_projection',cursor,t,layer,token)
                t=self.vectors(cursor,3*(h+kv_dim)+(h+2*kv_dim)) # RoPE plus QKV biases
                address=self.ext.layout.address(f'L{layer}/KV',(L-1)*self.ext.layout.kv_token_bytes,2*kv_dim*2)
                t=self.ext.access(t,2*kv_dim*2,'KV',True,address=address)
                cursor=self.stamp('norm_rope_residual',cursor,t,layer,token)
                t=self.attention(cursor,L,layer)
                cursor=self.stamp('QK_softmax_PV',cursor,t,layer,token)
                t=self.digital_matrix(f'L{layer}/Wo',h,h,cursor)
                cursor=self.stamp('output_projection',cursor,t,layer,token)
                t=self.vectors(cursor,5*h,1)
                cursor=self.stamp('norm_rope_residual',cursor,t,layer,token)
                if self.engines:
                    t=self.stochastic_matrix(f'L{layer}/gu',h,2*m,cursor)
                    cursor=self.stamp('FFN_gate_up',cursor,t,layer,token)
                    # Normal SiLU(g)*u, then fixed Sz encoding in both SC exploratory paths.
                    t=self.vectors(cursor,2*m,m)
                    cursor=self.stamp('FFN_SwiGLU',cursor,t,layer,token)
                    t=self.stochastic_matrix(f'L{layer}/down',m,h,cursor)
                    cursor=self.stamp('FFN_down',cursor,t,layer,token)
                else:
                    bpw=1 if self.variant=='digital_int8_ffn' else 2
                    t=self.digital_matrix(f'L{layer}/gu',h,2*m,cursor,bpw)
                    cursor=self.stamp('FFN_gate_up',cursor,t,layer,token)
                    t=self.vectors(cursor,2*m,m)
                    cursor=self.stamp('FFN_SwiGLU',cursor,t,layer,token)
                    t=self.digital_matrix(f'L{layer}/down',m,h,cursor,bpw)
                    cursor=self.stamp('FFN_down',cursor,t,layer,token)
                t=self.vectors(cursor,h)
                cursor=self.stamp('norm_rope_residual',cursor,t,layer,token)
            t=self.vectors(cursor,4*h,1)
            cursor=self.stamp('norm_rope_residual',cursor,t,token=token)
            t=self.digital_matrix('lm_head',h,v,cursor)
            cursor=self.stamp('lm_head',cursor,t,token=token)
            t=self.vectors(cursor,v) # greedy argmax, no vocabulary softmax
            cursor=self.stamp('token_select',cursor,t,token=token)
            token_times.append(cursor-begin)
        seconds=cursor*1e-9
        external_diagnostic=self.ext.finish(cursor) if hasattr(self.ext,'finish') else {'backend':'simplified Link'}
        # All standby terms are explicit parameters. No weight DFF added here.
        w=self.cfg['weight_macro'];sp=self.cfg['scratchpad'];r=self.cfg['rng'];sc=self.cfg['sc_cost']
        self.energy['weight_sram_leakage']+=w['cache_MiB']*w['leakage_W_per_MiB']*seconds
        self.energy['weight_macro_hold']+=self.engines*w['active_tile_hold_W']*seconds
        self.energy['scratch_leakage']+=sp['capacity_MiB']*sp['leakage_W_per_MiB']*seconds
        self.energy['digital_idle']+=self.cfg['digital']['idle_W']*seconds
        if not hasattr(self.ext,'finish'):
            self.energy['external_background']+=self.cfg['external_memory']['background_W']*seconds
        self.energy['control_interconnect']+=self.cfg['control_interconnect_W']*seconds
        rng_diagnostic=None
        if self.engines:
            rate=r['sources']*r['source_clock_hz']*r['raw_bits_per_source_cycle']
            power=r['sources']*r['source_power_W']
            if self.rng_hierarchy:
                rng_diagnostic=self.rng_hierarchy.finish(cursor)
                for key,value in rng_diagnostic['energy_J'].items():self.energy[key]+=value
                rate=rng_diagnostic['nominal_output_capacity_bits']/seconds
            else:
                self.energy['RNG_sources']+=power*(seconds if r['always_running'] else self.rng_service_ns*1e-9)
                self.energy['RNG_fabric_FIFO']+=r['fifo_distribution_W']*seconds
            self.energy['SC_idle']+=max(0,self.engines*cursor-self.sc_active_engine_ns-self.sc_wait_engine_ns)*sc['idle_W_per_engine']*1e-9
            assert self.raw_bits<=rate*seconds*(1+1e-9)
        else:rate=0.
        count=len(token_times)
        area=(w['cache_MiB']*w['area_mm2_per_MiB']+sp['capacity_MiB']*sp['area_mm2_per_MiB']+
              self.cfg['digital']['area_mm2']+self.engines*(sc['logic_area_mm2_per_engine']+w['active_tile_area_mm2'])+
              (r['sources']*r['area_mm2_per_source'] if self.engines else 0))
        if rng_diagnostic:
            area+=rng_diagnostic['area_mm2']-r['sources']*r['area_mm2_per_source']
        quality={'digital_bf16':11.652735,'digital_int8_ffn':11.681130}.get(self.variant)
        if self.variant=='dual_sc_sz':quality={256:127803.483024,1024:42.762909,4096:13.948997}.get(self.cfg['samples'])
        result=dict(variant=self.variant,scope='batch1 decode, full24 layers and lm_head; no prefill',
            implementation='Python macro-level LT model, not SystemC/TLM2 or RTL',
            config=self.cfg,quality_reference_ppl=quality,
            stochastic_core_power_provenance=('fixed-weight G1 per-cycle replicated with assumed shared frontend'
                if self.variant=='fixed_sc_exact' else 'adjustable proxy from fixed-weight G1, NOT measured dual-stream core'
                if self.variant=='dual_sc_sz' else 'no stochastic core'),
            stochastic_activity_case=self.profile['case'] if self.engines else None,
            sample_power_extrapolated=self.engines>0 and self.cfg['samples']!=4096,
            quality_warning='numerical reference only; timing/storage/cost assumptions do not validate hardware precision or correlation',
            metrics=dict(mean_TPOT_ms=sum(token_times)/count/1e6,first_decode_ms=token_times[0]/1e6,
                subsequent_decode_ms=token_times[1]/1e6 if count>1 else None,
                tokens_per_second=count/seconds,scenario_mJ_per_token=sum(self.energy.values())/count*1e3,
                parameterized_area_mm2=area,area_budget_feasible=area<=self.cfg['area_budget_mm2'],TTFT_ms=None),
            energy_J=dict(self.energy),time_ns=dict(self.time_categories),actions=dict(self.actions),traffic=dict(self.traffic),
            external_memory_model=external_diagnostic,
            weight_cache=dict(hits=self.weights.hits,misses=self.weights.misses,peak_bytes=self.weights.peak),
            bounds=dict(SC_wave_compute_ms=self.sc_compute_ns/1e6,RNG_service_ms=self.rng_service_ns/1e6,
                replay_service_ms=self.replay_ns/1e6,raw_rng_bits=self.raw_bits,
                source_nominal_capacity_bits=rate*seconds,scratch_peak_bytes=self.scratch_peak,
                local_weight_resident_bits=self.engines*w['local_resident_bits_per_tile']),
            energy_complete_measured=False,scenario_cost_coverage='all configured terms charged; unmodeled PHY/CTS/startup/host/prefill excluded',
            events=self.events,
            source_hashes={'variants/local_shared_a8_64/system_decode_base.py':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'evidence_contract.json':hashlib.sha256((ROOT/'configs/evidence_contract.json').read_bytes()).hexdigest()})
        if rng_diagnostic:
            result['rng_hierarchy']=rng_diagnostic
            result['quality_warning']+='; hierarchical Bivium streams NOT validated by the inherited PPL'
            result['scenario_cost_coverage']='configured scenario costs including RNG startup; uncalibrated local RNG/analog/root costs, SC dual proxy, no PHY/CTS/host/prefill'
            result['source_hashes']['rng_hierarchy.py']=hashlib.sha256((ROOT/'rng_hierarchy.py').read_bytes()).hexdigest()
        return result


def load():
    return (json.loads((ROOT/'configs/system_decode_default.json').read_text()),
            json.loads((ROOT/'configs/evidence_contract.json').read_text()),
            json.loads((ROOT/'configs/model_config.json').read_text()))
