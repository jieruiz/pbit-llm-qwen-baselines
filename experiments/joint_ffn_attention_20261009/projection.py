"""Fixed-weight stochastic projection reference, not hardware timing.
Signed N-bit dyadic weights; sign+(N-1)-bit magnitude inputs; offline group64
ranges. IID streams independent per projection, shared across output channels.
Exact Binomial count compression, FP32/64 wide carriers, BF16 I/O. Bias unchanged.
"""
import hashlib
import torch
from torch import nn
from torch.nn import functional as F
PROJS={'q':'q_proj','k':'k_proj','v':'v_proj','o':'o_proj'}
def bound(x):return 2.**torch.ceil(torch.log2(x.clamp_min(2.**-24)))
def weight_codes(w,bits,group=64):
    o,i=w.shape
    tile=w.float().reshape(o//group,group,i//group,group).permute(0,2,1,3)
    scale=bound(tile.abs().amax((-1,-2),keepdim=True));levels=2**(bits-1)
    code=(tile*levels/scale).round().clamp(-levels,levels-1)
    decoded=(code*scale/levels).permute(0,2,1,3).reshape(o,i).contiguous()
    code=code.permute(0,2,1,3).reshape(o,i).contiguous().to(torch.int32)
    return decoded,code,scale.flatten()
def encode(x,scale,bits):
    levels=2**(bits-1)
    code=(x.abs()*levels/scale).round().clamp(0,levels-1);p=code/levels
    return p,x.sign()*scale*p
def protected_hash(model):
    digest=hashlib.sha256()
    for n,p in model.named_parameters():
        if any('.self_attn.'+v+'.' in n for v in PROJS.values()):continue
        digest.update(n.encode());digest.update(p.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
    return digest.hexdigest()
class Projection(nn.Module):
    def __init__(self,original,scales,mode='iid',bits=10,cycles=1024,seed=0,collect=True,wide=False):
        super().__init__()
        self.mode=mode;self.bits=bits;self.cycles=cycles;self.collect=collect;self.calls=0
        self.dtype=torch.float64 if wide else torch.float32
        w=original.weight.detach().float();quant,codes,ws=weight_codes(w,bits)
        self.register_buffer('w',(w if mode=='reference' else quant).to(self.dtype))
        self.register_buffer('bias',None if original.bias is None else original.bias.detach().to(self.dtype))
        self.register_buffer('scale',torch.tensor(scales,device=w.device,dtype=self.dtype).repeat_interleave(64))
        assert len(self.scale)==w.shape[1]
        self.rng=torch.Generator(device=w.device).manual_seed(seed)
        self.register_buffer('stat',torch.zeros(9,device=w.device,dtype=torch.float64))
        self.register_buffer('fanout',(codes!=0).sum(0).to(self.dtype))
        pop=torch.zeros_like(codes)
        for b in range(bits):pop+=((codes.abs()>>b)&1)
        self.register_buffer('popout',pop.sum(0).to(self.dtype))
        self.in_features=w.shape[1];self.out_features=w.shape[0]
        self.weight_metadata={'codes':w.numel(),'code_bits':bits,'scale_count':ws.numel(),
            'scale_min':ws.min().item(),'scale_max':ws.max().item(),
            'zero_fraction':(codes==0).float().mean().item(),
            'bias_elements':0 if self.bias is None else self.bias.numel()}
    def forward(self,x):
        self.calls+=1;dtype=x.dtype;v=x.to(self.dtype);p,xq=encode(v,self.scale,self.bits)
        if self.mode in ('reference','weights'):drive=v
        elif self.mode=='mean':drive=xq
        elif self.mode=='iid':
            count=torch.binomial(torch.full_like(p,float(self.cycles)),p,generator=self.rng)
            drive=v.sign()*self.scale*(count/self.cycles)
        else:raise ValueError(self.mode)
        if self.collect:
            st=self.stat;levels=2**(self.bits-1)
            st[0]+=v.numel();st[1]+=(v.abs()>self.scale*(1-1/levels)).sum();st[2]+=(p==0).sum()
            st[3]+=(xq-v).square().sum(dtype=torch.float64);st[4]+=v.square().sum(dtype=torch.float64)
            st[5]+=p.sum(dtype=torch.float64);st[6]+=(p*self.fanout).sum(dtype=torch.float64)
            st[7]+=(p*self.popout).sum(dtype=torch.float64);st[8]+=(drive-xq).square().sum(dtype=torch.float64)
        return F.linear(drive,self.w,self.bias).to(dtype)
    def diagnostics(self):
        n,out,zeros,se,energy,psum,events,bit_events,sample_se=self.stat.tolist()
        tokens=int(n)//self.in_features;mac=tokens*self.in_features*self.out_features;stochastic=self.mode=='iid'
        return {'calls':self.calls,'input_values':int(n),'output_values':tokens*self.out_features,
            'tokens_including_overlapping_windows':tokens,'outside_code_range_fraction':out/max(n,1),
            'encoded_zero_fraction':zeros/max(n,1),'mean_probability':psum/max(n,1),
            'input_quantization_nmse':se/max(energy,1e-30),'sampling_vs_quantized_input_nmse':sample_se/max(energy,1e-30),
            'original_macs':mac,'expected_selected_weight_word_adds':events*self.cycles if stochastic else 0,
            'expected_active_magnitude_bitplane_events':bit_events*self.cycles if stochastic else 0,
            'bernoulli_decisions':int(n)*self.cycles if stochastic else 0,'weight':self.weight_metadata,
            'samples_per_projection':self.cycles if stochastic else 0,
            'count_accumulator_worst_same_scale_bits':(self.cycles*self.in_features*(2**(self.bits-1))).bit_length()+1 if stochastic else None}
def install(model,cal,targets,mode,bits,cycles,seed,collect=True,wide=False):
    replaced={}
    if mode=='baseline':return replaced
    for i,layer in enumerate(model.model.layers):
        for target in targets:
            attr=PROJS[target];key=f'L{i}.{target}';old=getattr(layer.self_attn,attr)
            m=Projection(old,cal['scales'][key],mode,bits,cycles,seed*10000+i*4+'qkvo'.index(target)+123,collect,wide)
            setattr(layer.self_attn,attr,m);replaced[key]=m
    return replaced
