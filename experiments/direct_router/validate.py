"""Expected coverage/cost on held-out training-tail queries; no test tokens."""
import json
import math
from pathlib import Path
import torch
from router import Router


@torch.inference_mode()
def main():
    data=torch.load('artifacts/calibration.pt',weights_only=True)
    n=data['train_n'];counts=data['counts'][n:].cuda();mass=data['mass'][:,:,n:].cuda()
    valid=torch.arange(128,device='cuda')[None]<counts[:,None]
    forced=valid & ((torch.arange(128,device='cuda')[None]==0)|(torch.arange(128,device='cuda')[None]>=counts[:,None]-2))
    output=[]
    for mode,reps in (('mean',1),('learned',1),('learned',4),('oracle',1)):
        x=data['x'+str(reps)][:,:,n:].float().cuda()
        router=None
        if mode=='learned':
            router=Router(reps).cuda().eval();router.load_state_dict(torch.load(f'artifacts/router_r{reps}.pt',map_location='cuda',weights_only=True)['state_dict'])
        for ratio in (.125,.25,.5):
            if mode=='learned':pi=router(x,ratio).sigmoid()
            else:
                if mode=='oracle':p=mass
                else:p=(x[...,0]+x[...,1]+x[...,-1]).masked_fill(~valid[None,None],-torch.inf).softmax(-1)
                pi=-torch.expm1(torch.ceil(counts*ratio)[None,None,:,None]*torch.log1p(-p.clamp(max=1-1e-7)))
            pi=pi.masked_fill(~valid[None,None],0);pi=torch.where(forced[None,None],1.,pi)
            for length in (2048,8192,0):
                ix=counts==length//64 if length else torch.ones_like(counts,dtype=torch.bool)
                output.append({'mode':mode,'reps':reps,'ratio':ratio,'context':length or 'all_1k_to_8k',
                    'expected_token_fraction':(pi[:,:,ix].sum(-1)/counts[ix]).mean().item(),
                    'expected_teacher_mass':(pi[:,:,ix]*mass[:,:,ix]).sum(-1).mean().item()})
    Path('results/heldout_coverage.json').write_text(json.dumps(output,indent=2))


if __name__=='__main__':main()
