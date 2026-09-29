from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn


class _HardForwardMeanBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, hard: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
        return hard

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[None, torch.Tensor]:
        return None, gradient


@dataclass
class FullPathPBitFFNConfig:
    input_size: int = 896
    hidden_sizes: tuple[int, ...] = (4864,)
    output_size: int = 896
    input_temperature: float = 1.0
    hidden_temperature: float = 1.0


class FullPathPBitFFN(nn.Module):
    """P-DNN whose every matrix input is a sampled bipolar p-bit state.

    A complete stochastic path is

        continuous input -> input p-bits -> hidden projection -> hidden p-bits
        -> ... -> continuous readout.

    Multiple paths are averaged only after the continuous readout. A sample
    count of zero selects a deterministic mean-field surrogate for warm-up and
    diagnostics; it is not the exact infinite-sample expectation of the
    stochastic multilayer network.
    """

    def __init__(self, config: FullPathPBitFFNConfig):
        super().__init__()
        if not config.hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one p-bit layer")
        if any(size <= 0 for size in config.hidden_sizes):
            raise ValueError("all hidden sizes must be positive")
        if config.input_temperature <= 0 or config.hidden_temperature <= 0:
            raise ValueError("temperatures must be positive")
        self.config = config
        sizes = [config.input_size, *config.hidden_sizes, config.output_size]
        self.projections = nn.ModuleList(
            nn.Linear(input_size, output_size, bias=True)
            for input_size, output_size in zip(sizes[:-1], sizes[1:])
        )
        self.sample_count = 0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for projection in self.projections[:-1]:
            nn.init.xavier_uniform_(projection.weight)
            nn.init.zeros_(projection.bias)
        readout = self.projections[-1]
        nn.init.normal_(
            readout.weight,
            mean=0.0,
            std=0.1 / math.sqrt(readout.in_features),
        )
        nn.init.zeros_(readout.bias)

    def set_sample_count(self, sample_count: int) -> None:
        if sample_count < 0:
            raise ValueError("sample_count must be non-negative; zero selects mean-field mode")
        self.sample_count = sample_count

    @staticmethod
    def _sample_bipolar(mean: torch.Tensor) -> torch.Tensor:
        probability = (mean + 1.0).mul(0.5).clamp_(0.0, 1.0)
        with torch.no_grad():
            hard = (torch.rand_like(probability) < probability).to(mean.dtype)
            hard = hard.mul_(2.0).sub_(1.0)
        if mean.requires_grad and torch.is_grad_enabled():
            return _HardForwardMeanBackward.apply(hard, mean)
        return hard

    def input_mean(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return torch.tanh(hidden_states / self.config.input_temperature)

    def hidden_mean(self, field: torch.Tensor) -> torch.Tensor:
        return torch.tanh(field / self.config.hidden_temperature)

    def mean_field_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        state = self.input_mean(hidden_states)
        for projection in self.projections[:-1]:
            state = self.hidden_mean(projection(state))
        return self.projections[-1](state)

    def sampled_path_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        state = self._sample_bipolar(self.input_mean(hidden_states))
        for projection in self.projections[:-1]:
            state = self._sample_bipolar(self.hidden_mean(projection(state)))
        return self.projections[-1](state)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.sample_count == 0:
            return self.mean_field_forward(hidden_states)
        output_sum = None
        for _ in range(self.sample_count):
            path_output = self.sampled_path_forward(hidden_states)
            output_sum = path_output if output_sum is None else output_sum + path_output
        return output_sum / float(self.sample_count)

    def checkpoint_config(self) -> dict:
        return asdict(self.config)


def load_full_path_checkpoint(path: str, device: str = "cuda") -> tuple[FullPathPBitFFN, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_payload = dict(payload["student_config"])
    config_payload["hidden_sizes"] = tuple(config_payload["hidden_sizes"])
    model = FullPathPBitFFN(FullPathPBitFFNConfig(**config_payload))
    model.load_state_dict(payload["student_state_dict"])
    model.to(device)
    return model, payload
