"""Staged full-FFN stochastic-matrix references for Qwen2.5-0.5B.

Modes:
  fixed_exact: stochastic activations, fixed INT8 weights, exact spatial sum.
  dual_exact:  stochastic activations and stochastic weight magnitudes, exact
               spatial count; conditional Gaussian compression of the weight
               Bernoulli counts (exact conditional mean/variance).
  or_plane:    fixed INT8 weights, per-64-input-block split-sign OR occupancy
               for every magnitude bit plane, without correction.

All modes use fixed calibration scales, T samples for gate/up, and another T
samples for the stochastic SwiGLU product/down projection. Attention is not
modified. This is a numerical reference, not a GPU acceleration kernel.
"""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


GROUP = 64
WEIGHT_LEVEL = 127.0


def quantize_int8(weight, group=GROUP):
    """Per-64x64 symmetric INT8; returns code, decoded weight and tile scale."""
    rows, cols = weight.shape
    if rows % group or cols % group:
        raise ValueError(f"weight shape {tuple(weight.shape)} is not divisible by {group}")
    tiles = weight.float().reshape(rows // group, group, cols // group, group).permute(0, 2, 1, 3)
    scales = tiles.abs().amax((-1, -2)).clamp_min(1e-20) / WEIGHT_LEVEL
    codes = (tiles / scales[:, :, None, None]).round().clamp(-127, 127).to(torch.int16)
    codes = codes.permute(0, 2, 1, 3).reshape(rows, cols).contiguous()
    expanded = scales.repeat_interleave(group, 0).repeat_interleave(group, 1)
    decoded = codes.float() * expanded
    return codes, decoded.contiguous(), scales.contiguous()


def encode_fixed(value, scale, bits):
    levels = 2**bits
    q = (value.abs() * (levels / scale)).round().clamp(0, levels - 1)
    p = q / levels
    decoded = value.sign() * scale * p
    return p, decoded


def sample_counts(probability, cycles, generator):
    return torch.binomial(torch.full_like(probability, float(cycles)), probability, generator=generator)


@dataclass
class MatrixSpec:
    codes: torch.Tensor
    decoded: torch.Tensor
    tile_scales: torch.Tensor


class StochasticMatrixFFN(nn.Module):
    def __init__(self, original, scales, mode, bits=12, cycles=4096, seed=0, layer=0,
                 or_chunk=64, collect=True):
        super().__init__()
        if mode not in ("fixed_exact", "dual_exact", "or_plane"):
            raise ValueError(mode)
        if cycles <= 0 or bits <= 0 or or_chunk <= 0 or or_chunk % GROUP:
            raise ValueError("invalid cycles/bits/or_chunk")
        self.mode = mode
        self.bits = bits
        self.cycles = cycles
        self.or_chunk = or_chunk
        self.collect = collect
        self.calls = 0
        self.width = original.gate_proj.weight.shape[0]
        if any(m.bias is not None for m in (original.gate_proj, original.up_proj, original.down_proj)):
            raise ValueError("this reference expects bias-free Qwen FFN projections")

        gu_weight = torch.cat((original.gate_proj.weight.detach(), original.up_proj.weight.detach()), 0)
        for name, weight in (("gu", gu_weight), ("down", original.down_proj.weight.detach())):
            code, decoded, tile_scales = quantize_int8(weight)
            self.register_buffer(name + "_codes", code)
            self.register_buffer(name + "_decoded", decoded)
            self.register_buffer(name + "_tile_scales", tile_scales)
            p_weight = code.abs().float() / WEIGHT_LEVEL
            expanded_scale = tile_scales.repeat_interleave(GROUP, 0).repeat_interleave(GROUP, 1)
            # tile_scales is one INT8 LSB. A Bernoulli weight stream uses
            # probability |code|/127 and event amplitude 127*LSB, so its mean
            # is code*LSB and its variance uses the full-scale amplitude.
            fullscale = expanded_scale * WEIGHT_LEVEL
            self.register_buffer(name + "_weight_variance",
                                 fullscale.square() * p_weight * (1.0 - p_weight))

        for signal, width in (("x", gu_weight.shape[1]), ("g", self.width), ("u", self.width)):
            scale = torch.tensor(scales[signal], device=gu_weight.device, dtype=torch.float32)
            if scale.numel() == 1:
                scale = scale.expand(width)
            elif scale.numel() == width // GROUP:
                scale = scale.repeat_interleave(GROUP)
            if scale.shape != (width,) or not bool((scale > 0).all()):
                raise ValueError(f"bad {signal} scale shape {tuple(scale.shape)}")
            self.register_buffer("scale_" + signal, scale.clone())
        self.register_buffer("scale_product", self.scale_g * self.scale_u)
        self.rng_input = torch.Generator(device=gu_weight.device).manual_seed(700001 + seed * 1000 + layer * 7)
        self.rng_product = torch.Generator(device=gu_weight.device).manual_seed(700002 + seed * 1000 + layer * 7)
        self.rng_weight_gu = torch.Generator(device=gu_weight.device).manual_seed(700003 + seed * 1000 + layer * 7)
        self.rng_weight_down = torch.Generator(device=gu_weight.device).manual_seed(700004 + seed * 1000 + layer * 7)
        self.rng_or_gu = torch.Generator(device=gu_weight.device).manual_seed(700005 + seed * 1000 + layer * 7)
        self.rng_or_down = torch.Generator(device=gu_weight.device).manual_seed(700006 + seed * 1000 + layer * 7)
        self.register_buffer("stats", torch.zeros(8, dtype=torch.float64, device=gu_weight.device))

    def _spec(self, name):
        return MatrixSpec(getattr(self, name + "_codes"), getattr(self, name + "_decoded"),
                          getattr(self, name + "_tile_scales"))

    def _count_projection(self, counts, sign, input_scale, name, generator, stochastic_weight):
        """Project shared input counts. Dual mode adds conditional weight-stream noise."""
        decoded = getattr(self, name + "_decoded")
        raw_mean = F.linear(sign * input_scale * counts, decoded)
        if not stochastic_weight:
            return raw_mean / self.cycles
        coefficient = getattr(self, name + "_weight_variance")
        # Conditional on each shared input stream having C ones, a connection
        # contributes Binomial(C, |code|/127). Summed conditional variance is a GEMM.
        raw_variance = F.linear(counts * input_scale.square(), coefficient).clamp_min_(0)
        noise = torch.randn(raw_mean.shape, device=raw_mean.device, dtype=raw_mean.dtype,
                            generator=generator) * raw_variance.sqrt()
        return (raw_mean + noise) / self.cycles

    def _or_projection(self, probability, sign, input_scale, name, generator):
        """Split-sign, per-bit-plane OR counts for independent input streams.

        It computes exact marginal OR probabilities in each 64-input tile and
        samples their T-cycle binomial counts. Counts for different outputs,
        planes and signs are conditionally sampled independently; literal shared
        stream correlations are not expanded in memory.
        """
        spec = self._spec(name)
        tokens = probability.numel() // probability.shape[-1]
        p = probability.reshape(tokens, probability.shape[-1]).float()
        s = sign.reshape(tokens, sign.shape[-1])
        in_features = p.shape[1]
        out_features = spec.codes.shape[0]
        groups = in_features // GROUP
        if input_scale.ndim != 1 or input_scale.numel() != in_features:
            raise ValueError("OR path requires fixed per-channel input scales")
        output = torch.zeros((tokens, out_features), device=p.device, dtype=torch.float32)
        bit_weights = (2.0 ** torch.arange(8, device=p.device, dtype=torch.float32))
        # All encoded probabilities are <1, so log1p remains finite.
        log_not = torch.log1p(-p)
        log_pos = torch.where(s > 0, log_not, torch.zeros_like(log_not))
        log_neg = torch.where(s < 0, log_not, torch.zeros_like(log_not))
        magnitude = spec.codes.abs().to(torch.int32)
        weight_positive = spec.codes > 0
        weight_negative = spec.codes < 0
        lp = log_pos.reshape(tokens, groups, GROUP).permute(1, 0, 2).contiguous()
        ln = log_neg.reshape(tokens, groups, GROUP).permute(1, 0, 2).contiguous()
        group_input_scale = input_scale.reshape(groups, GROUP)[:, 0]
        plane_index = torch.arange(8, device=p.device)
        for out_start in range(0, out_features, self.or_chunk):
            out_end = min(out_start + self.or_chunk, out_features)
            width = out_end - out_start
            mag = magnitude[out_start:out_end].reshape(width, groups, GROUP)
            planes = ((mag[:, :, :, None] >> plane_index) & 1).bool()
            wp = weight_positive[out_start:out_end].reshape(width, groups, GROUP, 1)
            wn = weight_negative[out_start:out_end].reshape(width, groups, GROUP, 1)
            mask_p = (planes & wp).permute(1, 2, 0, 3).reshape(groups, GROUP, width * 8).float()
            mask_n = (planes & wn).permute(1, 2, 0, 3).reshape(groups, GROUP, width * 8).float()
            lpos = torch.bmm(lp, mask_p) + torch.bmm(ln, mask_n)
            cpos = sample_counts((-torch.expm1(lpos)).clamp_(0, 1), self.cycles, generator)
            del lpos
            lneg = torch.bmm(lp, mask_n) + torch.bmm(ln, mask_p)
            cneg = sample_counts((-torch.expm1(lneg)).clamp_(0, 1), self.cycles, generator)
            del lneg
            signed_plane_count = (cpos - cneg).reshape(groups, tokens, width, 8)
            weighted_count = (signed_plane_count * bit_weights).sum(-1)
            out_group = torch.arange(out_start, out_end, device=p.device) // GROUP
            weight_group_scale = spec.tile_scales[out_group].transpose(0, 1)
            # One asserted bit plane represents 2**b integer LSBs. The plane
            # weighting is already applied above, so restore input_scale*LSB.
            scale = group_input_scale[:, None] * weight_group_scale
            output[:, out_start:out_end] = (weighted_count * scale[:, None, :]).sum(0) / self.cycles
            del cpos, cneg, signed_plane_count, weighted_count, mask_p, mask_n
        return output.reshape(*probability.shape[:-1], out_features)

    def forward(self, x):
        self.calls += 1
        output_dtype = x.dtype
        value = x.float()
        px, xq = encode_fixed(value, self.scale_x, self.bits)
        if self.mode == "or_plane":
            gu = self._or_projection(px, value.sign(), self.scale_x, "gu", self.rng_or_gu)
        else:
            input_counts = sample_counts(px, self.cycles, self.rng_input)
            gu = self._count_projection(input_counts, value.sign(), self.scale_x, "gu",
                                        self.rng_weight_gu, self.mode == "dual_exact")
        g, u = gu[..., :self.width], gu[..., self.width:]
        pg, gq = encode_fixed(g, self.scale_g, self.bits)
        pu, uq = encode_fixed(u, self.scale_u, self.bits)
        product_probability = pg * torch.sigmoid(gq) * pu
        product_sign = gq.sign() * uq.sign()
        if self.mode == "or_plane":
            result = self._or_projection(product_probability, product_sign, self.scale_product,
                                         "down", self.rng_or_down)
        else:
            product_counts = sample_counts(product_probability, self.cycles, self.rng_product)
            result = self._count_projection(product_counts, product_sign, self.scale_product, "down",
                                            self.rng_weight_down, self.mode == "dual_exact")
        if self.collect:
            st = self.stats
            st[0] += value.numel()
            st[1] += (value.abs() >= self.scale_x).sum()
            st[2] += (px == 0).sum()
            st[3] += g.numel() + u.numel()
            st[4] += (g.abs() >= self.scale_g).sum()
            st[5] += (u.abs() >= self.scale_u).sum()
            st[6] += product_probability.sum(dtype=torch.float64)
            st[7] += result.square().sum(dtype=torch.float64)
        return result.to(output_dtype)

    def diagnostics(self):
        x_count, x_clip, x_zero, gu_count, g_clip, u_clip, events, out_energy = self.stats.tolist()
        return {
            "calls": self.calls,
            "x_values": int(x_count),
            "x_fullscale_fraction": x_clip / max(x_count, 1),
            "x_encoded_zero_fraction": x_zero / max(x_count, 1),
            "g_fullscale_fraction": g_clip / max(gu_count / 2, 1),
            "u_fullscale_fraction": u_clip / max(gu_count / 2, 1),
            "expected_product_events": events * self.cycles,
            "output_squared_sum": out_energy,
            "mode": self.mode,
            "cycles_per_projection": self.cycles,
            "matrix_stages": 2,
            "or_correction": False,
            "dual_weight_count_compression": "conditional Gaussian; exact conditional mean and variance",
        }
