"""Closed-loop external memory: Python scheduler <-> persistent SystemC process.

Each macro DMA drains before returning; within it up to max_pending 64-byte
TLM bursts overlap. Compute/next DMA overlap follows the existing double buffer
schedule. No fabricated lookup latency and no fixed trace replay.
"""
import gzip
import hashlib
import json
import math
import subprocess
from pathlib import Path


class AddressSpace:
    def __init__(self,shape,variant,context,decode_tokens):
        h=shape['hidden_size'];m=shape['intermediate_size']
        kv=shape['num_key_value_heads']*(h//shape['num_attention_heads'])
        self.regions={};cursor=65536
        def allocate(name,size):
            nonlocal cursor
            allocated=math.ceil(size/4096)*4096
            self.regions[name]=dict(base=cursor,logical_bytes=size,allocated_bytes=allocated)
            cursor+=allocated
        allocate('lm_head',shape['vocab_size']*h*2)
        bpw=2 if variant=='digital_bf16' else 1
        for layer in range(shape['num_hidden_layers']):
            allocate(f'L{layer}/qkv',h*(h+2*kv)*2)
            allocate(f'L{layer}/Wo',h*h*2)
            allocate(f'L{layer}/gu',h*2*m*bpw)
            allocate(f'L{layer}/down',m*h*bpw)
        cursor=math.ceil(cursor/(2**21))*(2**21)
        self.kv_token_bytes=2*kv*2
        for layer in range(shape['num_hidden_layers']):
            allocate(f'L{layer}/KV',(context+decode_tokens)*self.kv_token_bytes)
        self.total_bytes=cursor

    def address(self,name,offset,size):
        r=self.regions[name]
        if offset<0 or size<=0 or offset+size>r['allocated_bytes']:
            raise ValueError(f'out-of-bounds tensor request: {name}, {offset}, {size}')
        return r['base']+offset


class DRAMSysExternal:
    def __init__(self,cfg,energy,traffic,address_space):
        self.cfg=cfg;self.energy=energy;self.traffic=traffic
        self.layout=address_space;self.counter=0;self.final=None
        self.service_ns=0.;self.max_service_ns=0.;self.stalls=0
        self.diagnostic={}
        executable=Path(cfg['bridge_executable']).resolve()
        config=Path(cfg['configuration']).resolve()
        self.config_hash=hashlib.sha256(config.read_bytes()).hexdigest()
        output=Path(cfg['output_prefix']);output.parent.mkdir(parents=True,exist_ok=True)
        self.log=output.with_suffix('.bridge.log').open('w')
        self.trace=gzip.open(output.with_suffix('.requests.jsonl.gz'),'wt',encoding='utf-8')
        self.process=subprocess.Popen([str(executable),str(config),str(cfg.get('max_pending',32))],
            stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=self.log,text=True,bufsize=1)
        ready=self._read()
        if ready.get('status')!='ready':raise RuntimeError(f'DRAMSys not ready: {ready}')
        if self.layout.total_bytes>ready['memory_bytes']:raise ValueError('model exceeds DRAM capacity')
        self.capacity=ready['memory_bytes']

    def _read(self):
        while True:
            line=self.process.stdout.readline()
            if not line:
                self.log.flush()
                raise RuntimeError('DRAMSys bridge terminated; inspect .bridge.log')
            # Upstream fmt::print uses the C stdout stream for the banner,
            # bypassing C++ cout redirection. Keep it in the diagnostic log.
            if not line.lstrip().startswith('{'):
                self.log.write(line)
                continue
            return json.loads(line)

    def access(self,ready,size,kind,write,address=None):
        if address is None:raise ValueError('DRAMSys requests require actual addresses')
        if address%64:raise ValueError('unaligned DRAM address')
        n=math.ceil(size/64)*64
        self.counter+=1
        request=dict(op='access',id=self.counter,ready_ns=ready,address=address,bytes=n,write=write)
        self.process.stdin.write(json.dumps(request)+'\n');self.process.stdin.flush()
        response=self._read()
        if response.get('id')!=self.counter:raise RuntimeError('DRAM response ID mismatch')
        done=response['done_ns']
        if done<ready:raise RuntimeError('DRAM completion preceded issue')
        direction='write' if write else 'read'
        self.traffic[f'external_{kind}_{direction}_logical_bytes']+=size
        self.traffic[f'external_{kind}_{direction}_burst_bytes']+=n
        self.traffic[f'external_{kind}_{direction}_requests']+=1
        self.service_ns+=done-ready;self.max_service_ns=max(self.max_service_ns,done-ready)
        self.trace.write(json.dumps({**request,'kind':kind,'logical_bytes':size,**response})+'\n')
        return done

    def finish(self,end_ns):
        self.process.stdin.write(json.dumps(dict(op='finish',end_ns=end_ns))+'\n')
        self.process.stdin.flush()
        self.final=self._read()
        code=self.process.wait(timeout=60)
        if code or self.final.get('status')!='finished':raise RuntimeError('DRAMSys failed finalization')
        self.trace.close();self.log.close();self.process.stdin.close();self.process.stdout.close()
        e=self.final['power_at_system_end']
        if not e:raise RuntimeError('DRAMPower produced no energy')
        self.energy['external_DRAMPower_core']=sum(v['core_J'] for v in e.values())
        self.energy['external_DRAMPower_interface']=sum(v['interface_J'] for v in e.values())
        self.diagnostic=dict(backend='DRAMSys 5.6.0 / SystemC3 / DRAMPower6.3.1',
            coupling='closed-loop nonblocking TLM bursts, DMA-level barrier',
            macro_requests=self.counter,capacity_bytes=self.capacity,
            configuration_sha256=self.config_hash,address_regions=self.layout.regions,
            service_ms=self.service_ns/1e6,max_DMA_latency_ns=self.max_service_ns,
            **self.final)
        return self.diagnostic

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            self.process.wait(timeout=10)
        if not self.trace.closed:self.trace.close()
        if not self.log.closed:self.log.close()
