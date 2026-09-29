from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class PBitFFNConfig:
    input_size: int = 896
    hidden_size: int = 4864
    output_size: int = 896
    coding: str = "bipolar"
    temperature: float = 1.0


class PBitFFN(nn.Module):
    """One stochastic binary hidden layer followed by a continuous readout."""

    def __init__(self, config: PBitFFNConfig):
        super().__init__()
        if config.coding not in {"bipolar", "binary"}:
            raise ValueError("coding must be 'bipolar' or 'binary'")
        if config.temperature <= 0:
            raise ValueError("temperature must be positive")
        self.config = config
        self.input_proj = nn.Linear(config.input_size, config.hidden_size, bias=True)
        self.output_proj = nn.Linear(config.hidden_size, config.output_size, bias=True)
        self.sample_count = 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        # Qwen FFN residual updates are much smaller than unit scale. Starting
        # with a conservative readout avoids spending the first updates merely
        # shrinking an over-large random prediction.
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.1 / math.sqrt(self.config.hidden_size))
        nn.init.zeros_(self.output_proj.bias)

    def set_sample_count(self, sample_count: int) -> None:
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative; zero selects the conditional mean")
        self.sample_count = sample_count

    def conditional_mean(self, field: torch.Tensor) -> torch.Tensor:
        scaled = field / self.config.temperature
        if self.config.coding == "bipolar":
            return torch.tanh(scaled)
        return torch.sigmoid(scaled)

    def sample_average(self, mean: torch.Tensor, sample_count: int) -> torch.Tensor:
        if sample_count <= 0:
            return mean
        probability = (mean + 1.0) * 0.5 if self.config.coding == "bipolar" else mean
        with torch.no_grad():
            sample_sum = torch.zeros_like(mean)
            for _ in range(sample_count):
                bit = (torch.rand_like(probability) < probability).to(mean.dtype)
                if self.config.coding == "bipolar":
                    bit = bit.mul(2.0).sub(1.0)
                sample_sum.add_(bit)
            hard_average = sample_sum.div(float(sample_count))
        if self.training and torch.is_grad_enabled():
            # The forward path is sampled. The backward path uses the derivative
            # of the corresponding continuous sigmoid/tanh conditional mean.
            return mean + (hard_average - mean).detach()
        return hard_average

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        field = self.input_proj(hidden_states)
        mean = self.conditional_mean(field)
        hidden = self.sample_average(mean, self.sample_count)
        return self.output_proj(hidden)

    def checkpoint_config(self) -> dict:
        return asdict(self.config)


def load_student_checkpoint(path: str, device: str = "cuda") -> tuple[PBitFFN, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    model = PBitFFN(PBitFFNConfig(**payload["student_config"]))
    model.load_state_dict(payload["student_state_dict"])
    model.to(device)
    return model, payload
