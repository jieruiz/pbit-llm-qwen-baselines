"""Training-only matching of teacher/student FFN residual updates."""
from __future__ import annotations

import torch


def normalized_feature_loss(student: torch.Tensor, teacher: torch.Tensor) -> torch.Tensor:
    """Per-layer NMSE; teacher statistics never receive gradients."""
    if student.shape != teacher.shape:
        raise ValueError("Teacher and student feature shapes must agree")
    target = teacher.detach().float()
    return (student.float() - target).square().mean() / target.square().mean().clamp_min(1e-8)


class FFNFeatureDistillation:
    """Capture same-token FFN outputs from each model's own input distribution.

    Captures the continuous output AFTER each student's path average. This
    adds an auxiliary training loss, not a new forward path or inference cost.
    It matches FFN updates, not decoder residual-stream/block outputs.
    """

    def __init__(self, teacher_modules: dict, student_modules: dict):
        if not teacher_modules or teacher_modules.keys() != student_modules.keys():
            raise ValueError("Teacher and student layers must match and be nonempty")
        self.layers = sorted(teacher_modules)
        self.teacher = {}
        self.student = {}
        self.handles = []
        for layer in self.layers:
            self.handles.append(teacher_modules[layer].register_forward_hook(self._hook(layer, True)))
            self.handles.append(student_modules[layer].register_forward_hook(self._hook(layer, False)))

    def _hook(self, layer, teacher):
        def capture(module, inputs, output):
            if teacher:
                self.teacher[layer] = output.detach()
            else:
                self.student[layer] = output
        return capture

    def loss(self):
        if set(self.teacher) != set(self.layers) or set(self.student) != set(self.layers):
            raise RuntimeError("Incomplete teacher/student forward captures")
        return torch.stack([
            normalized_feature_loss(self.student[layer], self.teacher[layer]) for layer in self.layers
        ]).mean()

    def clear(self):
        self.teacher.clear()
        self.student.clear()

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.clear()
