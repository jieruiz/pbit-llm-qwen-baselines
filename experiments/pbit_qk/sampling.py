"""One-sided signed Bernoulli query encoding; K remains continuous.

Counts compress temporal bits without changing the distribution of their sum.
This is a dense arithmetic reference, not a sparse hardware implementation.
"""
import torch


def counts(probability, samples, mode, generator=None):
    if samples < 1:
        raise ValueError('samples must be positive')
    if mode == 'iid':
        return torch.binomial(torch.full_like(probability, float(samples)),
                              probability, generator=generator)
    if mode == 'stratified':
        # For intervals [n/B,(n+1)/B), all but at most one stratum's
        # indicator U_n < p are deterministic. Compress the exact count law.
        expected = probability * samples
        floor = expected.floor()
        u = torch.rand(probability.shape, device=probability.device, generator=generator)
        return floor + (u < expected-floor).to(probability.dtype)
    raise ValueError(mode)


def encode(query, samples, mode, generator=None):
    query = query.float()
    maximum = query.abs().amax(-1, keepdim=True)
    scale = torch.where(maximum > 0, maximum, torch.ones_like(maximum))
    probability = (query.abs()/scale).clamp(0, 1)
    c = counts(probability, samples, mode, generator)
    reconstructed = query.sign() * scale * (c / samples)
    return reconstructed, c
