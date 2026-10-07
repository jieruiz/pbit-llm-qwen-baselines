import argparse
import copy
import json
import math
import time
from pathlib import Path
import torch
from torch.nn import functional as F
from router import Router


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--reps', type=int, required=True); ap.add_argument('--steps', type=int, default=1800)
    args = ap.parse_args()
    torch.manual_seed(71)
    data = torch.load('artifacts/calibration.pt', weights_only=True)
    x = data['x'+str(args.reps)].float().cuda(); mass = data['mass'].cuda()
    counts = data['counts'].cuda(); n = data['train_n']; c = x.shape[-2]
    valid = torch.arange(c, device='cuda')[None] < counts[:, None]
    free = valid & (torch.arange(c, device='cuda')[None] > 0) & (torch.arange(c, device='cuda')[None] < counts[:, None]-2)
    router = Router(args.reps).cuda()
    mask = valid[:n][None, None, ..., None]
    mean = (x[:, :, :n] * mask).sum((2,3)) / mask.sum((2,3))
    std = ((x[:, :, :n] - mean[:, :, None, None]).square() * mask).sum((2,3)) / mask.sum((2,3))
    router.mean[..., :-1] = mean; router.scale[..., :-1] = std.sqrt().clamp_min(.1)
    router.mean[..., -1] = math.log(.25); router.scale[..., -1] = .7
    opt = torch.optim.AdamW(router.parameters(), lr=.007, weight_decay=.001)
    ratios = (.125, .25, .5)

    def target(ratio, index):
        m = torch.ceil(counts[index].float() * ratio)[None, None, :, None]
        return -torch.expm1(m * torch.log1p(-mass[:, :, index].clamp(max=1-1e-7)))

    @torch.no_grad()
    def validate():
        metrics = []; loss = 0
        for ratio in ratios:
            logits = router(x[:, :, n:], ratio)
            t = target(ratio, slice(n,None))
            fm = free[n:][None, None]
            bce = (F.binary_cross_entropy_with_logits(logits,t,reduction='none')*fm).sum()/fm.sum()/336
            pi = logits.sigmoid().masked_fill(~valid[n:][None,None],0)
            pi = torch.where((valid[n:] & ~free[n:])[None,None], 1., pi)
            metrics.append({'ratio':ratio,'BCE_nonmandatory':bce.item(), 'expected_mass':(pi*mass[:,:,n:]).sum(-1).mean().item(), 'expected_token_fraction':(pi.sum(-1)/counts[n:]).mean().item()})
            loss += bce.item()
        return loss/3, metrics

    best = float('inf'); history = []; begin = time.perf_counter()
    for step in range(args.steps+1):
        if step % 100 == 0:
            score, metrics = validate(); history.append({'step':step, 'validation_BCE':score, 'metrics':metrics})
            if score < best:
                best = score; best_state = copy.deepcopy(router.state_dict()); best_step = step
            print(json.dumps(history[-1]), flush=True)
        if step == args.steps: break
        ix = torch.randint(n, (24,), device='cuda'); ratio = ratios[step % 3]
        logits = router(x[:, :, ix], ratio); t = target(ratio, ix); fm = free[ix][None,None]
        loss = (F.binary_cross_entropy_with_logits(logits,t,reduction='none')*fm).sum()/fm.sum()/336
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(router.parameters(),1.); opt.step()
    dest = Path(f'artifacts/router_r{args.reps}.pt')
    torch.save({'reps':args.reps,'state_dict':best_state,'best_step':best_step},dest)
    Path(f'results/training_r{args.reps}.json').write_text(json.dumps({'args':vars(args),'best_step':best_step,'best_validation_BCE':best,'seconds':time.perf_counter()-begin,'parameters':sum(p.numel() for p in router.parameters()),'history':history,'checkpoint':str(dest),'selection':'minimum mean validation BCE, test unseen'},indent=2))


if __name__ == '__main__': main()
