"""Immutable power-of-two ranges, unsigned N-bit magnitude comparator + sign.

Q=clamp(round(abs(v)*2**N/S),0,2**N-1), bit=(uniform_uint_N<Q).
Inference never computes a maximum or updates a scale. FP32 accumulators are
reference arithmetic; this is not a finite-width accumulator/device simulation.
"""
import torch
from torch.nn import functional as F
from streaming_ffn import StreamingFFN, count_bits, triple_and_counts


def encode(value,scale,bits):
    m=2**bits
    q=(value.abs()*(m/scale)).round().clamp(0,m-1)
    probability=q/m
    decoded=value.sign()*scale*probability
    return probability,decoded


def comparator_counts(q,bits,n,generator):
    result=torch.zeros_like(q)
    for _ in range(n):
        r=torch.randint(0,2**bits,q.shape,device=q.device,generator=generator)
        result+=(r<q)
    return result


class FixedFFN(StreamingFFN):
    def __init__(self,original,config,scales,bits=12,fixed_mode='streaming',diagnostics=True):
        super().__init__(original,config)
        self.bits=bits
        self.fixed_mode=fixed_mode
        self.collect=diagnostics
        self.forward_calls=0
        for signal,width in [('x',self.wgu.shape[1]),('g',self.width),('u',self.width)]:
            s=torch.tensor(scales[signal],device=self.wgu.device,dtype=torch.float32)
            if s.numel()==1:s=s.expand(width)
            elif s.numel()==width//config.group:s=s.repeat_interleave(config.group)
            assert s.shape==(width,) and (s>0).all()
            assert torch.equal(s.log2(),s.log2().round())
            self.register_buffer('scale_'+signal,s.clone())
            # count, out-of-fullscale, above-top-code, encoded-zero, squared error, squared signal
            self.register_buffer('stats_'+signal,torch.zeros(6,device=s.device,dtype=torch.float64))
        self.register_buffer('scale_product',self.scale_g*self.scale_u)

    def encoded(self,value,signal):
        scale=getattr(self,'scale_'+signal)
        if self.fixed_mode=='ideal':
            p=(value.abs()/scale).clamp(0,1)
            decoded=value.sign()*scale*p
            top=scale
        else:
            p,decoded=encode(value,scale,self.bits)
            top=scale*(1-2.**(-self.bits))
        if self.collect:
            stat=getattr(self,'stats_'+signal)
            stat[0]+=value.numel()
            stat[1]+=(value.abs()>=scale).sum()
            stat[2]+=(value.abs()>top).sum()
            stat[3]+=(p==0).sum()
            stat[4]+=(decoded-value).square().sum(dtype=torch.float64)
            stat[5]+=value.square().sum(dtype=torch.float64)
        return p,decoded

    def forward(self,x):
        self.forward_calls+=1
        c=self.config
        dtype=x.dtype
        x=x.float()
        px,xq=self.encoded(x,'x')
        if self.fixed_mode=='mean':
            g,u=self.fields(xq)
            _,gq=self.encoded(g,'g'); _,uq=self.encoded(u,'u')
            return F.linear(F.silu(gq)*uq,self.wd).to(dtype)
        blocks=c.cycles//c.interval
        burn=int(blocks*c.burn_fraction)
        count=torch.zeros_like(x)
        signed_sum=None
        for block in range(blocks):
            count+=count_bits(px,c.interval,self.rng_in)
            if block>=burn:
                g,u=self.fields(x.sign()*self.scale_x*count/((block+1)*c.interval))
                pg,gq=self.encoded(g,'g'); pu,uq=self.encoded(u,'u')
                # Deferred sigmoid circuit: evaluated on decoded fixed-point g.
                pb=torch.sigmoid(gq)
                events=triple_and_counts(pg,pb,pu,c.interval,self.rng_out)
                signed=events*gq.sign()*uq.sign()
                signed_sum=signed if signed_sum is None else signed_sum+signed
        # Constant scales allow exact algebraic regrouping over all time blocks.
        # This eliminates repeated down matmuls in the software reference only.
        return F.linear(signed_sum*self.scale_product/((blocks-burn)*c.interval),self.wd).to(dtype)

    def diagnostics(self):
        rows={}
        for signal in ('x','g','u'):
            count,out,above,zero,se,energy=getattr(self,'stats_'+signal).tolist()
            rows[signal]={'count':int(count),'out_of_fullscale':int(out),'above_top_code':int(above),
                'encoded_zero':int(zero),'squared_error':se,'signal_energy':energy,
                'out_of_fullscale_fraction':out/max(count,1),
                'quantization_nmse':se/max(energy,1e-30)}
        return {'forward_calls':self.forward_calls,'signals':rows}
