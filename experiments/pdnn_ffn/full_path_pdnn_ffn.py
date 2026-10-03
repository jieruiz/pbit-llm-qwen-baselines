from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
from torch import nn
from torch.nn import functional as F


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
    coding: str = "bipolar"
    learnable_encoding: bool = False
    minimum_temperature: float = 1e-3
    learnable_thresholds: bool = True
    architecture: str = "serial"
    input_encoding: str = "sigmoid"
    input_bits: int = 1


class FullPathPBitFFN(nn.Module):
    """P-DNN whose every matrix input is a sampled binary or bipolar state.

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
        if config.coding not in {"bipolar", "binary"}:
            raise ValueError("coding must be bipolar (-1/+1) or binary (0/1)")
        if not config.hidden_sizes:
            raise ValueError("hidden_sizes must contain at least one p-bit layer")
        if any(size <= 0 for size in config.hidden_sizes):
            raise ValueError("all hidden sizes must be positive")
        if config.input_temperature <= 0 or config.hidden_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if config.minimum_temperature <= 0:
            raise ValueError("minimum_temperature must be positive")
        if config.learnable_encoding and (
            config.input_temperature <= config.minimum_temperature
            or config.hidden_temperature <= config.minimum_temperature
        ):
            raise ValueError("learnable temperatures must exceed minimum_temperature")
        self.config = config
        sizes = [config.input_size, *config.hidden_sizes, config.output_size]
        self.projections = nn.ModuleList(
            nn.Linear(input_size, output_size, bias=True)
            for input_size, output_size in zip(sizes[:-1], sizes[1:])
        )
        self.sample_count = 0
        if config.learnable_encoding:
            if config.learnable_thresholds:
                self.input_threshold = nn.Parameter(torch.zeros(()))
                self.hidden_thresholds = nn.Parameter(torch.zeros(len(config.hidden_sizes)))
            self.input_temperature_raw = nn.Parameter(
                self._inverse_softplus(config.input_temperature - config.minimum_temperature)
            )
            self.hidden_temperature_raw = nn.Parameter(
                self._inverse_softplus(
                    torch.full((len(config.hidden_sizes),), config.hidden_temperature - config.minimum_temperature)
                )
            )
        self.reset_parameters()

    @staticmethod
    def _inverse_softplus(value: float | torch.Tensor) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32)
        return torch.log(torch.expm1(tensor))

    def effective_input_temperature(self) -> torch.Tensor:
        if self.config.learnable_encoding:
            return F.softplus(self.input_temperature_raw) + self.config.minimum_temperature
        return self.projections[0].weight.new_tensor(self.config.input_temperature)

    def effective_hidden_temperature(self, index: int) -> torch.Tensor:
        if self.config.learnable_encoding:
            return F.softplus(self.hidden_temperature_raw[index]) + self.config.minimum_temperature
        return self.projections[0].weight.new_tensor(self.config.hidden_temperature)

    def effective_input_threshold(self) -> torch.Tensor:
        if self.config.learnable_encoding and self.config.learnable_thresholds:
            return self.input_threshold
        return self.projections[0].weight.new_zeros(())

    def effective_hidden_threshold(self, index: int) -> torch.Tensor:
        if self.config.learnable_encoding and self.config.learnable_thresholds:
            return self.hidden_thresholds[index]
        return self.projections[0].weight.new_zeros(())

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

    def _sample_state(self, mean: torch.Tensor) -> torch.Tensor:
        probability = (mean + 1.0).mul(0.5).clamp_(0.0, 1.0) if self.config.coding == "bipolar" else mean
        with torch.no_grad():
            hard = (torch.rand_like(probability) < probability).to(mean.dtype)
            if self.config.coding == "bipolar":
                hard = hard.mul_(2.0).sub_(1.0)
        if mean.requires_grad and torch.is_grad_enabled():
            return _HardForwardMeanBackward.apply(hard, mean)
        return hard

    def input_mean(self, hidden_states: torch.Tensor) -> torch.Tensor:
        field = (hidden_states - self.effective_input_threshold()) / self.effective_input_temperature()
        return torch.tanh(field) if self.config.coding == "bipolar" else torch.sigmoid(field)

    def hidden_mean(self, field: torch.Tensor, index: int = 0) -> torch.Tensor:
        field = (field - self.effective_hidden_threshold(index)) / self.effective_hidden_temperature(index)
        return torch.tanh(field) if self.config.coding == "bipolar" else torch.sigmoid(field)

    def mean_field_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        state = self.input_mean(hidden_states)
        for index, projection in enumerate(self.projections[:-1]):
            state = self.hidden_mean(projection(state), index)
        return self.projections[-1](state)

    def sampled_path_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        state = self._sample_state(self.input_mean(hidden_states))
        for index, projection in enumerate(self.projections[:-1]):
            state = self._sample_state(self.hidden_mean(projection(state), index))
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


class GatedPBitFFN(FullPathPBitFFN):
    """Binary gate/value branches and a shared-weight, dual-rail readout.

    Hard paths use only 0/1 matrix inputs. The signed readout is implemented
    with two binary-input projections, not a ternary-input matrix multiply.
    Bias belongs to the affine driver; p-bits have no separate threshold.
    """

    def __init__(self, config: FullPathPBitFFNConfig):
        if config.coding != "binary" or len(config.hidden_sizes) != 1:
            raise ValueError("gated architecture requires binary coding and one hidden width")
        if config.learnable_thresholds or config.architecture != "gated_dual_rail":
            raise ValueError("gated architecture requires disabled p-bit thresholds")
        super().__init__(config)
        self.value_projection = nn.Linear(config.input_size, config.hidden_sizes[0], bias=True)
        nn.init.xavier_uniform_(self.value_projection.weight)
        nn.init.zeros_(self.value_projection.bias)
        if config.learnable_encoding:
            self.value_temperature_raw = nn.Parameter(
                self._inverse_softplus(config.hidden_temperature - config.minimum_temperature)
            )

    def effective_value_temperature(self) -> torch.Tensor:
        if self.config.learnable_encoding:
            return F.softplus(self.value_temperature_raw) + self.config.minimum_temperature
        return self.value_projection.weight.new_tensor(self.config.hidden_temperature)

    def branch_probabilities(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gate = self.hidden_mean(self.projections[0](state))
        value = torch.sigmoid(self.value_projection(state) / self.effective_value_temperature())
        return gate, value

    def dual_rail_readout(self, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        readout = self.projections[-1]
        # The same weight is used twice; the output bias is added exactly once.
        return readout(positive) - F.linear(negative, readout.weight, None)

    def mean_field_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, value = self.branch_probabilities(self.input_mean(hidden_states))
        return self.dual_rail_readout(gate * value, gate * (1.0 - value))

    def sampled_path_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        state = self._sample_state(self.input_mean(hidden_states))
        gate_probability, value_probability = self.branch_probabilities(state)
        gate = self._sample_state(gate_probability)
        value = self._sample_state(value_probability)
        return self.dual_rail_readout(gate * value, gate * (1.0 - value))


def build_full_path_student(config: FullPathPBitFFNConfig) -> FullPathPBitFFN:
    if config.input_encoding == "continuous_raw":
        from continuous_input_ffn import ContinuousInputPBitFFN
        return ContinuousInputPBitFFN(config)
    if config.input_encoding != "sigmoid":
        if config.architecture != "serial":
            raise ValueError("multi-bit input experiment currently supports serial only")
        from multibit_input_ffn import MultiBitInputFFN
        return MultiBitInputFFN(config)
    if config.architecture == "gated_dual_rail":
        return GatedPBitFFN(config)
    if config.architecture != "serial":
        raise ValueError(f"unknown architecture: {config.architecture}")
    return FullPathPBitFFN(config)


def student_type_name(student: FullPathPBitFFN) -> str:
    if student.config.input_encoding == "continuous_raw":
        return "continuous_raw_input_binary_hidden_pdnn_v1"
    if student.config.input_encoding != "sigmoid":
        return f"full_path_binary_input_{student.config.input_encoding}_k{student.config.input_bits}_v1"
    if student.config.architecture == "gated_dual_rail":
        return "full_path_binary_gated_dual_rail_v1"
    return "full_path_binary_pdnn_v2" if student.config.coding == "binary" else "full_path_bipolar_pdnn_v1"


def load_full_path_checkpoint(path: str, device: str = "cuda") -> tuple[FullPathPBitFFN, dict]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config_payload = dict(payload["student_config"])
    config_payload["hidden_sizes"] = tuple(config_payload["hidden_sizes"])
    model = build_full_path_student(FullPathPBitFFNConfig(**config_payload))
    model.load_state_dict(payload["student_state_dict"])
    model.to(device)
    return model, payload
