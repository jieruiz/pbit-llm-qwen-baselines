"""Channelwise affine p-bit fields; sampled matrix inputs remain bipolar."""
import math

import torch
from torch import nn

from full_path_pdnn_ffn import FullPathPBitFFN, FullPathPBitFFNConfig

TYPE = "full_path_bipolar_channel_calibrated_v1"


class CalibratedPBitFFN(FullPathPBitFFN):
    def __init__(self, config):
        if len(config.hidden_sizes) != 1:
            raise ValueError("This ablation supports exactly two projections")
        super().__init__(config)
        self.input_log_scale = nn.Parameter(torch.zeros(config.input_size))
        self.input_shift = nn.Parameter(torch.zeros(config.input_size))
        self.hidden_log_scale = nn.Parameter(torch.zeros(config.hidden_sizes[0]))
        self.hidden_shift = nn.Parameter(torch.zeros(config.hidden_sizes[0]))

    @staticmethod
    def calibrated_mean(field, log_scale, shift):
        # Preserve the original field rounding at identity, including BF16.
        adjusted = field.float() * log_scale.float().clamp(-math.log(4), math.log(4)).exp()
        adjusted = adjusted + shift.float().clamp(-4, 4)
        return torch.tanh(adjusted.to(field.dtype))

    def input_mean(self, hidden_states):
        return self.calibrated_mean(hidden_states / self.config.input_temperature,
                                    self.input_log_scale, self.input_shift)

    def hidden_mean(self, field):
        return self.calibrated_mean(field / self.config.hidden_temperature,
                                    self.hidden_log_scale, self.hidden_shift)


def load_checkpoint(path, device="cuda", enable_calibration=False):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = dict(payload["student_config"])
    config["hidden_sizes"] = tuple(config["hidden_sizes"])
    calibrated = payload.get("student_type") == TYPE
    cls = CalibratedPBitFFN if calibrated or enable_calibration else FullPathPBitFFN
    # Model construction must not perturb the paired training/evaluation RNG.
    with torch.random.fork_rng(devices=[]):
        model = cls(FullPathPBitFFNConfig(**config))
    if enable_calibration and not calibrated:
        model.projections.load_state_dict({k.removeprefix("projections."): v
            for k, v in payload["student_state_dict"].items()}, strict=True)
        payload["student_type"] = TYPE
    else:
        model.load_state_dict(payload["student_state_dict"], strict=True)
    return model.to(device), payload


def parameter_groups(students, learning_rate, calibration_lr, weight_decay):
    weights, calibration = [], []
    for student in students.values():
        for name, parameter in student.named_parameters():
            (weights if name.startswith("projections.") else calibration).append(parameter)
    groups = [{"params": weights, "lr": learning_rate, "weight_decay": weight_decay}]
    if calibration:
        groups.append({"params": calibration, "lr": calibration_lr, "weight_decay": 0.0})
    return groups
