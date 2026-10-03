"""Teacher-shaped sigmoid banks with tied binary AND readout.

Inference uses binary features and statically scaled readout columns. The
factorized path is an algebraically equivalent training implementation, not
a claim of binary hardware execution or measured hardware acceleration.
"""
from dataclasses import asdict, dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class AndConfig:
    input_size: int = 896
    hidden_size: int = 4864
    output_size: int = 896
    bits: int = 2
    minimum_temperature: float = 0.02


class SigmoidBank(nn.Module):
    def __init__(self, width, bits, minimum_temperature):
        super().__init__()
        self.minimum_temperature = minimum_temperature
        self.register_buffer("scale", torch.ones(width))
        self.threshold = nn.Parameter(torch.zeros(width, bits))
        self.temperature_raw = nn.Parameter(torch.zeros(width, bits))
        self.coefficient = nn.Parameter(torch.zeros(width, bits))
        self.offset = nn.Parameter(torch.zeros(width))

    def probabilities(self, field):
        # FP32 probabilities and accumulation, including in BF16 model inference.
        normalized = field.float() / self.scale.float()
        temperature = F.softplus(self.temperature_raw.float()) + self.minimum_temperature
        return torch.sigmoid((normalized.unsqueeze(-1) - self.threshold.float()) / temperature)

    def decode(self, states):
        return self.scale.float() * (self.offset.float() + (self.coefficient.float() * states).sum(-1))

    def moments(self, probability):
        mean = self.decode(probability)
        variance = (self.scale.float().unsqueeze(-1) * self.coefficient.float()).square()
        return mean, (variance * probability * (1-probability)).sum(-1)

    @torch.no_grad()
    def initialize(self, field, silu):
        scale = torch.quantile(field.float().abs(), .995, dim=0).clamp_min(.05)
        self.scale.copy_(scale)
        bits = self.coefficient.shape[-1]
        if silu:
            thresholds = (torch.arange(bits, device=field.device).float()+.5) / bits
            self.threshold.copy_(thresholds.expand_as(self.threshold))
            self.coefficient.fill_(1/bits)
            self.offset.zero_()
            temperature = .35 / bits
        else:
            thresholds = -1 + 2*(torch.arange(bits, device=field.device).float()+.5)/bits
            self.threshold.copy_(thresholds.expand_as(self.threshold))
            self.coefficient.fill_(2/bits)
            self.offset.fill_(-1)
            temperature = .7 / bits
        self.temperature_raw.fill_(torch.log(torch.expm1(torch.tensor(temperature-self.minimum_temperature))).item())


class MultiThresholdAndFFN(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.bits < 1:
            raise ValueError("bits must be positive")
        self.config = config
        self.gate_projection = nn.Linear(config.input_size, config.hidden_size, bias=True)
        self.value_projection = nn.Linear(config.input_size, config.hidden_size, bias=True)
        self.readout = nn.Linear(config.hidden_size, config.output_size, bias=True)
        self.gate_bank = SigmoidBank(config.hidden_size, config.bits, config.minimum_temperature)
        self.value_bank = SigmoidBank(config.hidden_size, config.bits, config.minimum_temperature)
        self.sample_count = 0
        self.binary_readout = True

    @torch.no_grad()
    def initialize_teacher(self, mlp):
        for own, original in ((self.gate_projection, mlp.gate_proj),
                              (self.value_projection, mlp.up_proj), (self.readout, mlp.down_proj)):
            own.weight.copy_(original.weight)
            own.bias.zero_()

    def probabilities(self, x):
        return (self.gate_bank.probabilities(self.gate_projection(x)),
                self.value_bank.probabilities(self.value_projection(x)))

    @staticmethod
    def sample(probability):
        hard = (torch.rand_like(probability) < probability).to(probability.dtype)
        return hard + (probability-probability.detach())

    def factorized_readout(self, gate, value):
        features = self.gate_bank.decode(gate) * self.value_bank.decode(value)
        with torch.autocast(device_type=features.device.type, enabled=False):
            return F.linear(features.float(), self.readout.weight.float(), self.readout.bias.float())

    def expanded_readout(self, gate, value):
        """Each F.linear input is 0/1 for sampled states; signed coefficients are weights.

        Shared bits are reused across all ANDs; resampling per product would
        change the covariance and invalidate the factorized training identity.
        """
        g, v = self.gate_bank, self.value_bank
        c = g.scale.float().unsqueeze(-1) * g.coefficient.float()
        d = v.scale.float().unsqueeze(-1) * v.coefficient.float()
        c0, d0 = g.scale.float()*g.offset.float(), v.scale.float()*v.offset.float()
        with torch.autocast(device_type=gate.device.type, enabled=False):
            weight = self.readout.weight.float()
            output = F.linear(c0*d0, weight, self.readout.bias.float())
            for k in range(self.config.bits):
                output = output + F.linear(gate[..., k].float(), weight*(c[:, k]*d0)[None, :])
            for l in range(self.config.bits):
                output = output + F.linear(value[..., l].float(), weight*(c0*d[:, l])[None, :])
            for k in range(self.config.bits):
                for l in range(self.config.bits):
                    state = gate[..., k] * value[..., l]
                    output = output + F.linear(state.float(), weight*(c[:, k]*d[:, l])[None, :])
        return output

    def conditional_moments(self, x):
        p, q = self.probabilities(x)
        m, v = self.gate_bank.moments(p)
        n, w = self.value_bank.moments(q)
        # Channels and the two banks are conditionally independent. AND terms
        # within a channel are correlated; this expression retains that covariance.
        product_variance = v*w + v*n.square() + w*m.square()
        with torch.autocast(device_type=x.device.type, enabled=False):
            mean = F.linear(m*n, self.readout.weight.float(), self.readout.bias.float())
            variance = F.linear(product_variance, self.readout.weight.float().square())
        return mean, variance

    def set_sample_count(self, count):
        if count < 0:
            raise ValueError("sample_count must be nonnegative")
        self.sample_count = count

    def forward(self, x):
        p, q = self.probabilities(x)
        if self.sample_count == 0:
            return self.factorized_readout(p, q).to(x.dtype)
        output = None
        for _ in range(self.sample_count):
            gate, value = self.sample(p), self.sample(q)
            path = (self.expanded_readout(gate, value) if self.binary_readout else
                    self.factorized_readout(gate, value))
            output = path if output is None else output + path
        return (output/self.sample_count).to(x.dtype)

    def checkpoint_config(self):
        return asdict(self.config)


def load_checkpoint(path, device="cuda"):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload["student_type"] != "multithreshold_and_v1":
        raise ValueError("wrong checkpoint type")
    student = MultiThresholdAndFFN(AndConfig(**payload["student_config"]))
    student.load_state_dict(payload["student_state_dict"])
    return student.to(device), payload
