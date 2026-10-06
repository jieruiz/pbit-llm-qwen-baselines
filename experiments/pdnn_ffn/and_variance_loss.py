"""Conditional output variance of N independent AND-bank paths.

Reference powers are fixed teacher statistics, never current student outputs.
Variance is conditional on the actual student input; cross-layer errors are
not modeled as independent, and this objective is not a model-PPL identity.
"""
import math
import torch


def normalized_output_variance(module, inputs, reference_power, sample_count=4):
    if sample_count != 4 or module.config.bits != 4:
        raise ValueError("This experiment fixes K=4 and N=4")
    if not math.isfinite(reference_power) or reference_power <= 0:
        raise ValueError("reference power must be a positive fixed scalar")
    _, variance = module.conditional_moments(inputs)
    return variance.mean() / (sample_count * (reference_power + 1e-12))


class VarianceCollector:
    """One differentiable scalar per FFN forward, without changing its RNG."""
    def __init__(self, students, powers):
        if set(students) != {int(k) for k in powers}:
            raise ValueError("reference scales must match the replaced layer set")
        self.enabled = False
        self.values = {}
        self.handles = []
        for layer, module in students.items():
            power = float(powers[str(layer)])
            if power <= 0 or not math.isfinite(power):
                raise ValueError("invalid reference scale")
            def hook(m, args, output, layer=layer, power=power):
                if self.enabled:
                    if layer in self.values:
                        raise RuntimeError("unexpected repeated FFN forward")
                    self.values[layer] = normalized_output_variance(m, args[0], power)
            self.handles.append(module.register_forward_hook(hook))
        self.layer_count = len(students)

    def begin(self):
        self.values.clear()
        self.enabled = True

    def loss(self):
        self.enabled = False
        if len(self.values) != self.layer_count:
            raise RuntimeError("not all replaced FFNs contributed variance")
        return torch.stack(list(self.values.values())).mean()

    def close(self):
        self.enabled = False
        self.values.clear()
        for handle in self.handles:
            handle.remove()
