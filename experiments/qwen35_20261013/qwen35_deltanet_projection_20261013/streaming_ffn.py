"""Block-stream reference, with exact count-law compression (not a GPU accelerator).

Fixed deterministic weights; only activations are random. Input counts encode
signed magnitudes, and drive cumulative g/u estimates. Output bits are the AND
of three conditionally independent streams. Their Binomial count is an exact
compression while drives are held constant for a block. The next output block
uses the newly available estimate, hence one block of causal pipeline delay.
No OR approximation, quantized accumulator, physical p-bit or RTL is simulated.
"""
from dataclasses import dataclass, asdict
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class StreamConfig:
    mode: str = 'streaming'
    cycles: int = 4096
    interval: int = 256
    group: int = 64
    burn_fraction: float = .25
    weight_bits: int = 0
    seed: int = 0
    layer: int = 0
    literal: bool = False


def quantize_blocks(weight, block=64):
    """Symmetric INT8 per 64x64 tile, returned dequantized for simulation."""
    rows,cols=weight.shape
    if rows%block or cols%block:
        raise ValueError('matrix must tile exactly')
    tiles=weight.reshape(rows//block,block,cols//block,block).permute(0,2,1,3)
    scale=tiles.abs().amax((-1,-2),keepdim=True).clamp_min(1e-20)/127
    result=(tiles/scale).round().clamp(-127,127)*scale
    return result.permute(0,2,1,3).reshape(rows,cols).contiguous()


def grouped_scale(x, group):
    if x.shape[-1]%group:
        raise ValueError('group must divide width')
    # Dynamic per-token/per-group max, no clipping. Its compute/storage cost is
    # part of the unimplemented hardware interface, not free p-bit logic.
    return x.reshape(*x.shape[:-1],-1,group).abs().amax(-1,keepdim=True).clamp_min(1e-20).expand(
        *x.shape[:-1],x.shape[-1]//group,group).reshape_as(x)


def count_bits(probability, count, generator, literal=False):
    if literal:
        result=torch.zeros_like(probability)
        for _ in range(count):
            result+=(torch.rand(probability.shape,device=probability.device,generator=generator)<probability)
        return result
    return torch.binomial(torch.full_like(probability,float(count)),probability,generator=generator)


def triple_and_counts(pg,pb,pu,count,generator,literal=False):
    if literal:
        result=torch.zeros_like(pg)
        for _ in range(count):
            bits=[torch.rand(p.shape,device=p.device,generator=generator)<p for p in (pg,pb,pu)]
            result+=(bits[0]&bits[1]&bits[2])
        return result
    return count_bits(pg*pb*pu,count,generator)


class StreamingFFN(nn.Module):
    def __init__(self, original, config):
        super().__init__()
        self.config=config
        if config.mode not in ('dense','input_only','product_only','staged','streaming'):
            raise ValueError(config.mode)
        if config.cycles%config.interval or config.cycles<=0 or config.interval<=0:
            raise ValueError('cycles must be a positive multiple of interval')
        if not 0<=config.burn_fraction<1:
            raise ValueError('invalid burn fraction')
        if config.weight_bits not in (0,8):
            raise ValueError('weight_bits must be 0 or 8')
        weights=[]
        for layer in (original.gate_proj,original.up_proj,original.down_proj):
            w=layer.weight.detach().float().clone()
            if config.weight_bits==8:
                w=quantize_blocks(w,config.group)
            weights.append(w)
        self.register_buffer('wgu',torch.cat(weights[:2],dim=0))
        self.register_buffer('wd',weights[2])
        self.width=weights[0].shape[0]
        assert original.gate_proj.bias is None and original.up_proj.bias is None and original.down_proj.bias is None
        self.rng_in=torch.Generator(device=self.wgu.device).manual_seed(100000+config.seed*1000+config.layer*2)
        self.rng_out=torch.Generator(device=self.wgu.device).manual_seed(100001+config.seed*1000+config.layer*2)

    def fields(self,x):
        # Algebraic compression of exact signed bit-plane sums; FP32 arithmetic.
        both=F.linear(x,self.wgu)
        return both[...,:self.width],both[...,self.width:]

    def product_sum(self,g,u,n):
        sg,su=grouped_scale(g,self.config.group),grouped_scale(u,self.config.group)
        pg,pu=(g.abs()/sg).clamp(0,1),(u.abs()/su).clamp(0,1)
        pb=torch.sigmoid(g)
        counts=triple_and_counts(pg,pb,pu,n,self.rng_out,self.config.literal)
        # This represents n signed binary products, each with a common scale
        # inside a 64-channel group. Hardware must support block-scale rescaling.
        encoded_sum=counts*g.sign()*u.sign()*sg*su
        return F.linear(encoded_sum,self.wd)

    def forward(self,x):
        c=self.config
        input_dtype=x.dtype
        x=x.float()
        blocks=c.cycles//c.interval
        burn=int(blocks*c.burn_fraction)
        output_count=(blocks-burn)*c.interval
        if c.mode=='dense':
            g,u=self.fields(x)
            return F.linear(F.silu(g)*u,self.wd).to(input_dtype)
        if c.mode=='product_only':
            g,u=self.fields(x)
            return (self.product_sum(g,u,output_count)/output_count).to(input_dtype)
        sx=grouped_scale(x,c.group)
        probability=(x.abs()/sx).clamp(0,1)
        signed_scale=x.sign()*sx
        counts=torch.zeros_like(x)
        total=None
        for block in range(blocks):
            counts+=count_bits(probability,c.interval,self.rng_in,c.literal)
            if c.mode=='streaming' or block==blocks-1:
                # Internal running estimates, not a completed inter-layer readout.
                g,u=self.fields(signed_scale*counts/((block+1)*c.interval))
                if c.mode=='streaming' and block>=burn:
                    contribution=self.product_sum(g,u,c.interval)
                    total=contribution if total is None else total+contribution
        if c.mode=='input_only':
            return F.linear(F.silu(g)*u,self.wd).to(input_dtype)
        if c.mode=='staged':
            return (self.product_sum(g,u,output_count)/output_count).to(input_dtype)
        return (total/output_count).to(input_dtype)

    def specification(self):
        return asdict(self.config)
