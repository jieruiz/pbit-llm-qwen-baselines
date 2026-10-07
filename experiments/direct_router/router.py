"""Small per-head Bernoulli routers; no block softmax at inference."""
import math
import torch
from torch import nn
from torch.nn import functional as F


def features(q, means, sizes, length, reps):
    # q [B,H,1,D], means [B,KV,C*R,D]. No full-token score matrix.
    means = means.repeat_interleave(q.shape[1] // means.shape[1], 1)
    raw = (q.float() @ means.float().transpose(-1, -2)).squeeze(-2) / math.sqrt(q.shape[-1])
    c = math.ceil(length / 64)
    raw = F.pad(raw, (0, c * reps - raw.shape[-1])).reshape(*raw.shape[:2], c, reps)
    valid = torch.arange(c * reps, device=q.device).reshape(c, reps) < sizes.numel()
    maximum = raw.masked_fill(~valid, -torch.inf).amax((-1, -2), keepdim=True)
    relative = raw - maximum
    relative = torch.where(valid, relative, torch.zeros_like(relative))
    shape = (*raw.shape[:3], 1)
    qrms = q.float().square().mean(-1).sqrt().unsqueeze(-1).expand(shape)
    age = (1 - torch.arange(c, device=q.device).float() / max(c - 1, 1)).view(1, 1, c, 1).expand(shape)
    block_sizes = (length - torch.arange(c, device=q.device) * 64).clamp(1, 64).float()
    count = torch.full(shape, math.log(c), device=q.device)
    fraction = (block_sizes / 64).log().view(1, 1, c, 1).expand(shape)
    return torch.cat((relative, maximum.expand(shape), qrms, age, count, fraction), -1)


class Router(nn.Module):
    def __init__(self, reps, hidden=16):
        super().__init__()
        self.reps, self.hidden = reps, hidden if reps == 4 else 0
        f = reps + 6  # features plus requested log-budget
        self.register_buffer('mean', torch.zeros(24, 14, f))
        self.register_buffer('scale', torch.ones(24, 14, f))
        self.w1 = nn.Parameter(torch.randn(24, 14, f, self.hidden or 1) / math.sqrt(f))
        self.b1 = nn.Parameter(torch.zeros(24, 14, self.hidden or 1))
        if self.hidden:
            self.w2 = nn.Parameter(torch.randn(24, 14, self.hidden) / math.sqrt(self.hidden))
            self.b2 = nn.Parameter(torch.full((24, 14), -2.))

    def forward(self, x, ratio, layer=None):
        x = torch.cat((x, torch.full_like(x[..., :1], math.log(ratio))), -1)
        if layer is None:  # [L,H,N,C,F]
            x = (x - self.mean[:, :, None, None]) / self.scale[:, :, None, None]
            y = torch.einsum('lhncf,lhfo->lhnco', x, self.w1) + self.b1[:, :, None, None]
            return (y.relu() * self.w2[:, :, None, None]).sum(-1) + self.b2[:, :, None, None] if self.hidden else y.squeeze(-1)
        x = (x - self.mean[layer][None, :, None]) / self.scale[layer][None, :, None]
        y = torch.einsum('bhcf,hfo->bhco', x, self.w1[layer]) + self.b1[layer][None, :, None]
        return (y.relu() * self.w2[layer][None, :, None]).sum(-1) + self.b2[layer][None, :, None] if self.hidden else y.squeeze(-1)

    def macs_per_block(self):
        return self.reps * 64 + ((self.reps + 6) * self.hidden + self.hidden if self.hidden else self.reps + 6)


def summary_cache(attn, k, reps):
    size = 64 // reps
    length = k.shape[-2]
    cache = getattr(attn, '_direct_summary', None)
    if cache is None or cache[0] != length - 1 or cache[1].shape[:2] != k.shape[:2]:
        sums = F.pad(k.float(), (0, 0, 0, (-length) % size)).reshape(*k.shape[:2], -1, size, k.shape[-1]).sum(-2)
    else:
        _, sums = cache
        if (length - 1) % size == 0:
            sums = torch.cat((sums, torch.zeros_like(sums[:, :, :1])), -2)
        sums[:, :, -1] += k[:, :, -1].float()
    attn._direct_summary = (length, sums)
    sizes = (length - torch.arange(sums.shape[-2], device=k.device) * size).clamp(1, size).float()
    return sums / sizes[:, None], sizes


def force_blocks(mask):
    mask[..., 0] = True
    mask[..., -2:] = True
    return mask
