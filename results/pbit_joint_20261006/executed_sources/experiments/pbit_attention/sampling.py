"""Two exact probability-law references for p-bit value aggregation.

Tree nodes are Bernoulli p-bits; their probabilities are subtree mass ratios.
Independent counts are Binomial(S, p), exactly the sum of S independent p-bits.
Counts @ V is an arithmetic reference, NOT an optimized sparse GPU kernel.
"""
import torch
from torch.nn import functional as F


def tree_counts(probability, samples, generator=None):
    """Sample categorical addresses via a balanced binary Bernoulli tree."""
    shape = probability.shape
    p = probability.reshape(-1, shape[-1])
    length = p.shape[-1]
    padded = 1 << (length - 1).bit_length()
    levels = [F.pad(p, (0, padded-length))]
    while levels[-1].shape[-1] > 1:
        levels.append(levels[-1].reshape(p.shape[0], -1, 2).sum(-1))
    node = torch.zeros((p.shape[0], samples), dtype=torch.long, device=p.device)
    for children in reversed(levels[:-1]):
        left = children.gather(1, node*2)
        right = children.gather(1, node*2+1)
        total = left+right
        right_probability = torch.where(total > 0, right/total.clamp_min(torch.finfo(p.dtype).tiny), 0.)
        bit = torch.rand(right_probability.shape, device=p.device, generator=generator) < right_probability
        node = node*2+bit.long()
    counts = torch.zeros_like(p)
    counts.scatter_add_(1, node, torch.ones_like(node, dtype=p.dtype))
    return counts.reshape(shape)


def independent_counts(probability, samples, generator=None):
    """Compress the S time samples, without changing their distribution."""
    return torch.binomial(torch.full_like(probability, float(samples)), probability, generator=generator)


def explicit_independent_counts(probability, samples, generator=None):
    """Literal 0/1 simulator used in probability-law validation."""
    counts = torch.zeros_like(probability)
    for _ in range(samples):
        counts += (torch.rand(probability.shape, device=probability.device, generator=generator)
                   < probability).to(probability.dtype)
    return counts


def aggregate(probability, value, mode, samples, generator=None):
    if mode == "dense":
        return probability @ value.float(), None
    function = {"tree": tree_counts, "independent": independent_counts}[mode]
    counts = function(probability, samples, generator)
    # The divisor is S even when independent p-bits select zero/multiple rows.
    return (counts/float(samples)) @ value.float(), counts
