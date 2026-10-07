"""Second-stage conservative headwise recall calibration on held-out train text."""
import json
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
    for reps in (1,4):
        router=Router(reps).cuda().eval();router.load_state_dict(torch.load(f'artifacts/router_r{reps}.pt',map_location='cuda',weights_only=True)['state_dict'])
        logits=router(data['x'+str(reps)][:,:,n:].float().cuda(),.5)
        def measure(offset):
            pi=(logits+offset[:,:,None,None]).sigmoid().masked_fill(~valid[None,None],0)
            pi=torch.where(forced[None,None],1.,pi)
            return (pi*mass).sum(-1).mean(-1), (pi.sum(-1)/counts).mean(-1)
        lo=torch.zeros(24,14,device='cuda');hi=torch.full_like(lo,16.)
        for _ in range(30):
            mid=(lo+hi)/2;coverage,_=measure(mid)
            lo=torch.where(coverage<.95,mid,lo);hi=torch.where(coverage>=.95,mid,hi)
        cov,cost=measure(hi)
        torch.save(hi,f'artifacts/offset_r{reps}.pt')
        output.append({'reps':reps,'ratio':.5,'target_per_head_expected_coverage':.95,'expected_coverage':cov.mean().item(),'expected_token_fraction':cost.mean().item(),'offset_min':hi.min().item(),'offset_max':hi.max().item(),'offset_mean':hi.mean().item(),'head_offsets':hi.cpu().tolist()})
    Path('results/calibration.json').write_text(json.dumps({'protocol':'Added after initial screen; bias chosen ONLY on training-tail validation, target >=95% expected teacher mass in each head, averaged over 1k..8k validation queries; not a per-query guarantee','results':output},indent=2))
    print(json.dumps(output,default=str),flush=True)


if __name__=='__main__':main()
