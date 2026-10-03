"""Position-bit input quantization with tied bitplane matrix weights.

Adjacent-level stochastic rounding is an algorithmic encoder reference. Its bits
are correlated; independent physical sigmoid p-bits do not automatically realize it.
No hidden-layer sample averaging is introduced.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from full_path_pdnn_ffn import FullPathPBitFFN


def quantize_codes(inputs, bounds, bits, stochastic):
    """Quantize into 2**bits uniform levels between -bounds and +bounds."""
    if not 1 <= bits <= 8:
        raise ValueError("input bits must be in [1,8]")
    levels = 2**bits - 1
    normalized = ((inputs.float() / bounds.float()).clamp(-1, 1) + 1) * (levels / 2)
    with torch.no_grad():
        if stochastic:
            lower = normalized.floor()
            code = lower + (torch.rand_like(normalized) < normalized - lower)
        else:
            code = (normalized + 0.5).floor()
        return code.clamp(0, levels).to(torch.int64)


def bitplanes(codes, bits, dtype):
    return [((codes >> bit) & 1).to(dtype) for bit in range(bits)]


def decode_codes(codes, bounds, bits):
    return bounds.float() * (codes.float() * (2.0 / (2**bits - 1)) - 1)


class MultiBitInputFFN(FullPathPBitFFN):
    def __init__(self, config):
        if config.input_encoding not in {"stochastic", "deterministic", "continuous"}:
            raise ValueError("unknown input encoding")
        if config.coding != "binary" or config.architecture != "serial" or config.learnable_thresholds:
            raise ValueError("use serial, binary coding, and temperature-only")
        if not 1 <= config.input_bits <= 8:
            raise ValueError("input bits must be in [1,8]")
        super().__init__(config)
        # Input temperature is replaced by a frozen, train-only calibrated range.
        if hasattr(self, "input_temperature_raw"):
            del self.input_temperature_raw
        self.register_buffer("input_bound", torch.ones(config.input_size))

    def clipped_input(self, x):
        return torch.maximum(torch.minimum(x.float(), self.input_bound.float()), -self.input_bound.float())

    def input_mean(self, x):
        return self.clipped_input(x)

    def quantized_first_field(self, x):
        codes = quantize_codes(x, self.input_bound, self.config.input_bits,
                               self.config.input_encoding == "stochastic")
        projection = self.projections[0]
        weight = projection.weight
        bound = self.input_bound.to(weight.dtype)
        # Fold the per-channel lower endpoint into the affine bias. This is a
        # deterministic parameter operation, not a continuous activation matmul.
        result = -(weight * bound).sum(-1).float()
        if projection.bias is not None:
            result = result + projection.bias.float()
        for bit, state in enumerate(bitplanes(codes, self.config.input_bits, x.dtype)):
            coefficient = (2.0 * (2**bit) / (2**self.config.input_bits - 1)) * bound
            result = result + F.linear(state, weight * coefficient, None).float()
        # STE only for upstream inputs; weight gradients already use hard bits.
        # In local distillation x is frozen, so this branch is not executed.
        if torch.is_grad_enabled() and x.requires_grad:
            clipped = self.clipped_input(x)
            result = result + F.linear(clipped - clipped.detach(), weight.detach(), None).float()
        return result.to(x.dtype)

    def mean_field_forward(self, x):
        state = self.clipped_input(x)
        if self.config.input_encoding == "deterministic":
            q = decode_codes(quantize_codes(x, self.input_bound, self.config.input_bits, False),
                             self.input_bound, self.config.input_bits)
            state = state + (q - state).detach()
        state = state.to(x.dtype)
        for index, projection in enumerate(self.projections[:-1]):
            state = self.hidden_mean(projection(state), index)
        return self.projections[-1](state)

    def sampled_path_forward(self, x):
        if self.config.input_encoding == "continuous":
            field = self.projections[0](self.clipped_input(x).to(x.dtype))
        else:
            field = self.quantized_first_field(x)
        state = self._sample_state(self.hidden_mean(field, 0))
        for index, projection in enumerate(self.projections[1:-1], start=1):
            state = self._sample_state(self.hidden_mean(projection(state), index))
        return self.projections[-1](state)
